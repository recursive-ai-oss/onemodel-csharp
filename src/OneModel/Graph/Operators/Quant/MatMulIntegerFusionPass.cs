using System;
using System.Collections.Generic;
using System.Threading;
using RecursiveAI.OneModel.Graph.Operators.Elementwise;
using RecursiveAI.OneModel.Graph.Passes;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Quant;

/// <summary>
/// Fuses the dequantization that follows a dynamically quantized matrix product:
/// <c>MatMulInteger -> Cast (float32) -> Mul (by the scale) [-> Add (a constant bias vector)]</c> becomes one operator that scales
/// each 32-bit sum as the tile is finished and writes float32, so the 32-bit intermediate is never stored or read back.
/// </summary>
/// <remarks>
/// Bit-preserving: the fused epilogue does the same correctly rounded steps in the same order (convert the sum to float32,
/// multiply by the scale, add the bias), with no fused multiply-add. Only a chain whose intermediate results have no other reader is
/// fused. The Mul operand must be a scalar or have one value per column; a model that has anything else there is reported by shape
/// inference when it is planned.
/// </remarks>
internal sealed class MatMulIntegerFusionPass : IGraphPass
{
    public string Name => "matmulinteger-dequantize";

    public OptimizationLevel MinLevel => OptimizationLevel.Basic;

    public bool PreservesBits => true;

    public void Run(PassGraph g, CancellationToken ct)
    {
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
        for (int i = 0; i < g.Nodes.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            CompiledNode m = g.Nodes[i];
            if (m.Kernel is not IntegerMatMulOp || m.Info.OpType != "MatMulInteger" || m.Inputs.Length < 2 || m.Outputs.Length != 1)
            {
                continue;
            }

            if (!Next(g, sole, isOutput, m, out int ci) || g.Nodes[ci].Kernel is not CastOp || g.Nodes[ci].Info.GetInt("to", 0) != 1)
            {
                continue;
            }

            CompiledNode cast = g.Nodes[ci];
            if (!Next(g, sole, isOutput, cast, out int mi) || g.Nodes[mi].Kernel is not MulOp || g.Nodes[mi].Inputs.Length != 2)
            {
                continue;
            }

            CompiledNode mul = g.Nodes[mi];
            int scale = mul.Inputs[0] == cast.Outputs[0] ? mul.Inputs[1] : mul.Inputs[0];
            if (scale < 0 || scale == cast.Outputs[0])
            {
                continue;
            }

            CompiledNode last = mul;
            int bias = -1;
            var chain = new List<int> { ci, mi };
            if (Next(g, sole, isOutput, mul, out int ai) && g.Nodes[ai].Kernel is AddOp && g.Nodes[ai].Inputs.Length == 2)
            {
                CompiledNode add = g.Nodes[ai];
                int other = add.Inputs[0] == mul.Outputs[0] ? add.Inputs[1] : add.Inputs[0];
                Tensor c = other >= 0 ? g.Constant(other) : default;
                if (!c.IsEmpty && c.DType == DType.F32 && c.Rank == 1 && c.Length > 1)
                {
                    bias = other;
                    last = add;
                    chain.Add(ai);
                }
            }

            int[] inputs = [m.Inputs[0], m.Inputs[1], m.Inputs.Length > 2 ? m.Inputs[2] : -1, m.Inputs.Length > 3 ? m.Inputs[3] : -1, scale, bias];
            OpKernel kernel = g.Own(IntegerMatMulOp.MatMulIntegerToFloat.Create(m.Info));
            // The fused node goes where the last absorbed node was: the scale may be computed between the product and the Mul.
            int lastIndex = chain[^1];
            replaced[lastIndex] = new CompiledNode(m.Info, kernel, inputs, [last.Outputs[0]]);
            absorbed.Add(i);
            foreach (int k in chain)
            {
                if (k != lastIndex)
                {
                    absorbed.Add(k);
                }
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

    // The node that is the only reader of this node's only output, when that output is not a graph output.
    private static bool Next(PassGraph g, int[] sole, HashSet<int> isOutput, CompiledNode n, out int index)
    {
        index = -1;
        if (n.Outputs.Length != 1 || n.Outputs[0] < 0 || isOutput.Contains(n.Outputs[0]))
        {
            return false;
        }

        index = sole[n.Outputs[0]];
        return index >= 0;
    }
}
