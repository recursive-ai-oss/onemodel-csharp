using System;
using System.Collections.Generic;
using System.Threading;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Passes;

/// <summary>
/// Evaluates every node whose inputs are all constants, once, when the session is created, so later passes (packing, fusion) see
/// the folded weights and no plan repeats the work. The node runs through its own kernel in the session's mode, so the result is
/// what the node would have produced at run time: the pass preserves bits in both modes.
/// </summary>
/// <remarks>
/// Nodes whose output shapes depend on values at run time are left alone, as are nodes with a graph output among their outputs
/// and nodes whose result would be larger than both their inputs and 64 KiB (Tile, ConstantOfShape: the same blow-up rule plans
/// use). A view operator (Reshape, Squeeze, ...) of a constant yields a copy, so every constant owns its storage.
/// Shape-dependent folding (Shape, Size of a runtime input) stays in plans.
/// </remarks>
internal sealed class ConstantFoldPass : IGraphPass
{
    private const long FoldAlwaysBytes = 1 << 16;

    public string Name => "constant-fold";

    public OptimizationLevel MinLevel => OptimizationLevel.Basic;

    public bool PreservesBits => true;

    public void Run(PassGraph g, CancellationToken ct)
    {
        var isOutput = new HashSet<int>(g.OutputIds);
        var ictx = new InferContext();
        using var scratch = new Arena(BufferPool.Shared);
        var kept = new List<CompiledNode>(g.Nodes.Count);
        foreach (CompiledNode node in g.Nodes)
        {
            ct.ThrowIfCancellationRequested();
            if (!TryFold(g, node, isOutput, ictx, scratch))
            {
                kept.Add(node);
            }
        }

        g.Nodes.Clear();
        g.Nodes.AddRange(kept);
    }

    private static bool TryFold(PassGraph g, CompiledNode node, HashSet<int> isOutput, InferContext ictx, Arena scratch)
    {
        OpKernel k = node.Kernel;
        if (k.OutputsDataDependent)
        {
            return false;
        }

        long inBytes = 0;
        int present = 0;
        foreach (int id in node.Inputs)
        {
            if (id < 0)
            {
                continue;
            }

            Tensor c = g.Constant(id);
            if (c.IsEmpty)
            {
                return false;
            }

            present++;
            inBytes += (long)c.Shape.Length * DTypes.SizeOf(c.DType);
        }

        foreach (int id in node.Outputs)
        {
            if (id >= 0 && isOutput.Contains(id))
            {
                return false;
            }
        }

        var ins = new Tensor[node.Inputs.Length];
        ictx.Begin(node.Info, node.Inputs.Length, node.Outputs.Length);
        for (int i = 0; i < ins.Length; i++)
        {
            if (node.Inputs[i] >= 0)
            {
                ins[i] = g.Constant(node.Inputs[i]);
                ictx.SetInput(i, ins[i].DType, ins[i].Shape, ins[i]);
            }
        }

        try
        {
            k.Infer(ictx);
        }
        catch (Exception e) when (e is ArgumentException or InvalidOperationException)
        {
            // A model problem; the plan reports it with the shapes it has. Leave the node as it is.
            _ = e;
            return false;
        }

        if (ictx.NeedsRunTimeValues)
        {
            return false;
        }

        int nOut = node.Outputs.Length;
        bool anyConst = false;
        for (int o = 0; o < nOut; o++)
        {
            if (node.Outputs[o] >= 0 && !ictx.OutputIsSet(o))
            {
                return false;
            }

            anyConst |= ictx.OutputIsSet(o) && !ictx.OutputConstant(o).IsEmpty;
        }

        if (anyConst)
        {
            return TakeConstantOutputs(g, node, ictx);
        }

        if (present == 0)
        {
            return false;
        }

        long outBytes = 0;
        for (int o = 0; o < nOut; o++)
        {
            if (ictx.OutputIsSet(o))
            {
                outBytes += (long)ictx.OutputShape(o).Length * DTypes.SizeOf(ictx.OutputType(o));
            }
        }

        if (outBytes > Math.Max(inBytes, FoldAlwaysBytes))
        {
            return false;
        }

        if (k.ViewOfInput >= 0)
        {
            int src = node.Inputs[k.ViewOfInput];
            if (src < 0)
            {
                return false;
            }

            if (!g.Constant(src).TryReshape(ictx.OutputShape(0), out Tensor view))
            {
                return false;
            }

            Tensor copy = view.Clone();
            g.SetConstant(node.Outputs[0], copy);
            return true;
        }

        var outs = new Tensor[nOut];
        for (int o = 0; o < nOut; o++)
        {
            if (ictx.OutputIsSet(o))
            {
                outs[o] = Tensor.Create(ictx.OutputType(o), ictx.OutputShape(o), BufferPool.Shared);
            }
        }

        try
        {
            k.Run(new OpRun(ins, outs, g.Mode, g.Path, 1, scratch));
        }
        catch
        {
            foreach (Tensor t in outs)
            {
                if (!t.IsEmpty)
                {
                    t.Dispose();
                }
            }

            throw;
        }
        finally
        {
            scratch.Reset();
        }

        for (int o = 0; o < nOut; o++)
        {
            if (outs[o].IsEmpty)
            {
                continue;
            }

            if (node.Outputs[o] >= 0)
            {
                g.SetConstant(node.Outputs[o], outs[o]);
            }
            else
            {
                outs[o].Dispose();
            }
        }

        return true;
    }

    // Constant, and kernels that know their outputs from the input shapes alone: copy so the graph owns the storage.
    private static bool TakeConstantOutputs(PassGraph g, CompiledNode node, InferContext ictx)
    {
        for (int o = 0; o < node.Outputs.Length; o++)
        {
            Tensor c = ictx.OutputConstant(o);
            if (c.IsEmpty)
            {
                return false;
            }
        }

        for (int o = 0; o < node.Outputs.Length; o++)
        {
            Tensor c = ictx.OutputConstant(o);
            int id = node.Outputs[o];
            if (id >= 0)
            {
                g.SetConstant(id, c.Clone());
            }

            if (ictx.OutputConstantOwned(o))
            {
                c.Dispose();
            }
        }

        return true;
    }
}
