using System;
using System.Collections.Generic;
using System.Threading;
using RecursiveAI.OneModel.Graph.Passes;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Quant;

/// <summary>
/// Fuses the float operators a QDQ model leaves between quantized tensors, where the pair around them is single-valued:
/// <list type="bullet">
/// <item><c>DequantizeLinear -> f -> QuantizeLinear</c> for an element-wise function f (HardSwish, HardSigmoid, Relu, ...) becomes a
/// 256-entry table lookup on the bytes (<see cref="QLutOp"/>); the table is made by running the chain's own kernels over the 256
/// possible inputs, so every entry is what the chain computes.</item>
/// <item><c>DequantizeLinear, DequantizeLinear -> Mul -> QuantizeLinear</c> becomes <see cref="QLinearMulOp"/>, which does the chain's
/// float steps in the chain's order without writing the float tensors (the gate of a squeeze-and-excitation block).</item>
/// <item><c>DequantizeLinear -> GlobalAveragePool -> QuantizeLinear</c> becomes <see cref="QLinearGlobalAveragePoolOp"/>.</item>
/// </list>
/// </summary>
/// <remarks>
/// Not bit-preserving, like <see cref="QdqPass"/> (and skipped in exact mode with it): the table is bit-for-bit the chain's result for
/// the inputs it was run on, but a vector kernel's tail lanes may round a function slightly differently than its body, and the integer
/// sum of the average pool differs from the chain's float sum in the last place. Only chains whose intermediate values have no
/// other reader are fused.
/// </remarks>
internal sealed class QdqFusionPass : IGraphPass
{
    private static readonly HashSet<string> s_unary =
    [
        "HardSwish", "HardSigmoid", "Relu", "Sigmoid", "Tanh", "LeakyRelu", "Gelu", "Elu", "Selu", "Softplus", "Softsign", "Abs", "Neg",
        "Erf", "Exp", "Sqrt", "Reciprocal",
    ];

    public string Name => "qdq-fusion";

    public OptimizationLevel MinLevel => OptimizationLevel.All;

    public bool PreservesBits => false;

    public void Run(PassGraph g, CancellationToken ct)
    {
        int[] producer = g.Producers();
        int[] uses = g.UseCounts();
        var isOutput = new HashSet<int>(g.OutputIds);
        var sole = new int[g.ValueCount];
        Array.Fill(sole, -1);
        for (int i = 0; i < g.Nodes.Count; i++)
        {
            foreach (int id in g.Nodes[i].Inputs)
            {
                if (id >= 0 && uses[id] == 1)
                {
                    sole[id] = i;
                }
            }
        }

        var replaced = new Dictionary<int, CompiledNode>();
        var absorbed = new HashSet<int>();
        var ictx = new InferContext();
        using var scratch = new Arena(BufferPool.Shared);
        for (int i = 0; i < g.Nodes.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            CompiledNode n = g.Nodes[i];
            if (n.Outputs.Length != 1 || n.Outputs[0] < 0 || isOutput.Contains(n.Outputs[0]) || sole[n.Outputs[0]] < 0)
            {
                continue;
            }

            int qIndex = sole[n.Outputs[0]];
            CompiledNode q = g.Nodes[qIndex];
            if (q.Kernel is not QuantizeLinearOp || q.Inputs.Length < 3 || q.Inputs[0] != n.Outputs[0] || q.Outputs.Length != 1 ||
                !QdqPass.ScalarScale(g, q.Inputs[1]) || !QdqPass.ScalarZero(g, q.Inputs[2], out _))
            {
                continue;
            }

            CompiledNode? repl = null;
            if (n.Inputs.Length == 1 && s_unary.Contains(n.Info.OpType))
            {
                repl = TryLut(g, n, q, producer, ictx, scratch);
            }
            else if (n.Kernel is Operators.MulOp && n.Inputs.Length == 2)
            {
                repl = TryMul(g, n, q, producer);
            }
            else if (n.Kernel is Operators.GlobalAveragePoolOp && n.Inputs.Length == 1)
            {
                repl = TryAverage(g, n, q, producer);
            }

            if (repl is not null)
            {
                replaced[i] = repl;
                absorbed.Add(qIndex);
            }
        }

        if (replaced.Count == 0)
        {
            return;
        }

        var result = new List<CompiledNode>(g.Nodes.Count);
        for (int i = 0; i < g.Nodes.Count; i++)
        {
            if (!absorbed.Contains(i))
            {
                result.Add(replaced.TryGetValue(i, out CompiledNode? r) ? r : g.Nodes[i]);
            }
        }

        g.Nodes.Clear();
        g.Nodes.AddRange(result);
    }

    // A DequantizeLinear of a single-valued scale and zero point whose quantized input is 8-bit.
    private static bool TrySingleDq(PassGraph g, int[] producer, int valueId, out QdqPass.Dq dq, out DType type)
    {
        type = DType.Undefined;
        return QdqPass.TryDq(g, producer, valueId, out dq) && dq.Source >= 0 && QdqPass.ScalarScale(g, dq.Scale) &&
               QdqPass.ScalarZero(g, dq.Zero, out type);
    }

    private static CompiledNode? TryLut(PassGraph g, CompiledNode f, CompiledNode q, int[] producer, InferContext ictx, Arena scratch)
    {
        if (!TrySingleDq(g, producer, f.Inputs[0], out QdqPass.Dq x, out DType xType))
        {
            return null;
        }

        CompiledNode dq = g.Nodes[producer[f.Inputs[0]]];
        using Tensor all = Tensor.Create(xType, Shape.Create(256), BufferPool.Shared);
        Span<byte> bytes = QuantParams.MutableBytes(in all);
        for (int i = 0; i < 256; i++)
        {
            bytes[i] = (byte)i;
        }

        Tensor scale = g.Constant(x.Scale), zero = g.Constant(x.Zero);
        Tensor? floats = Eval(g, dq, [all, scale, zero], ictx, scratch);
        if (floats is null)
        {
            return null;
        }

        Tensor? mapped = null;
        Tensor? table = null;
        try
        {
            mapped = Eval(g, f, [floats.Value], ictx, scratch);
            if (mapped is null || mapped.Value.DType != DType.F32 || mapped.Value.Length != 256)
            {
                return null;
            }

            table = Eval(g, q, [mapped.Value, g.Constant(q.Inputs[1]), g.Constant(q.Inputs[2])], ictx, scratch);
            if (table is null || table.Value.Length != 256 || table.Value.DType is not (DType.U8 or DType.I8))
            {
                table?.Dispose();
                return null;
            }
        }
        finally
        {
            floats.Value.Dispose();
            mapped?.Dispose();
        }

        int tableId = g.AddValue(g.NameOf(q.Outputs[0]) + "_table", table.Value);
        OpKernel kernel = g.Own(QLutOp.Def.Create(f.Info));
        return new CompiledNode(f.Info, kernel, [x.Source, tableId], [q.Outputs[0]]);
    }

    private static CompiledNode? TryMul(PassGraph g, CompiledNode mul, CompiledNode q, int[] producer)
    {
        if (!TrySingleDq(g, producer, mul.Inputs[0], out QdqPass.Dq a, out _) || !TrySingleDq(g, producer, mul.Inputs[1], out QdqPass.Dq b, out _))
        {
            return null;
        }

        int[] inputs = [a.Source, a.Scale, a.Zero, b.Source, b.Scale, b.Zero, q.Inputs[1], q.Inputs[2]];
        return new CompiledNode(mul.Info, g.Own(QLinearMulOp.Def.Create(mul.Info)), inputs, [q.Outputs[0]]);
    }

    private static CompiledNode? TryAverage(PassGraph g, CompiledNode pool, CompiledNode q, int[] producer)
    {
        if (!TrySingleDq(g, producer, pool.Inputs[0], out QdqPass.Dq x, out _))
        {
            return null;
        }

        int[] inputs = [x.Source, x.Scale, x.Zero, q.Inputs[1], q.Inputs[2]];
        return new CompiledNode(pool.Info, g.Own(QLinearGlobalAveragePoolOp.Def.Create(pool.Info)), inputs, [q.Outputs[0]]);
    }

    // Runs one node of the graph on constant inputs, as plan-time constant folding does; null when the node cannot be run that way.
    private static Tensor? Eval(PassGraph g, CompiledNode node, Tensor[] ins, InferContext ictx, Arena scratch)
    {
        if (node.Outputs.Length != 1)
        {
            return null;
        }

        ictx.Begin(node.Info, ins.Length, 1);
        for (int i = 0; i < ins.Length; i++)
        {
            if (ins[i].IsEmpty)
            {
                return null;
            }

            ictx.SetInput(i, ins[i].DType, ins[i].Shape, ins[i]);
        }

        try
        {
            node.Kernel.Infer(ictx);
        }
        catch (Exception e) when (e is ArgumentException or InvalidOperationException)
        {
            _ = e;
            return null;
        }

        if (ictx.NeedsRunTimeValues || !ictx.OutputIsSet(0))
        {
            return null;
        }

        Tensor[] outs = [Tensor.Create(ictx.OutputType(0), ictx.OutputShape(0), BufferPool.Shared)];
        try
        {
            node.Kernel.Run(new OpRun(ins, outs, g.Mode, g.Path, 1, scratch));
        }
        catch
        {
            outs[0].Dispose();
            throw;
        }
        finally
        {
            scratch.Reset();
        }

        return outs[0];
    }
}
