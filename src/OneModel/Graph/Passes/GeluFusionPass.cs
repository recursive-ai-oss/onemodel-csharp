using System;
using System.Threading;
using RecursiveAI.OneModel.Graph.Operators;
using RecursiveAI.OneModel.Graph.Operators.Elementwise;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Passes;

/// <summary>
/// The five-operator form of Gelu that exporters write before opset 20, Mul(Mul(x, 1 + Erf(Div(x, sqrt 2))), 0.5), becomes one
/// Gelu node (the erf form). One pass over memory instead of five, and no intermediate tensors. The fused kernel's erf is not the
/// same arithmetic as the chain of separate operators, so results differ in the last bits and the pass is skipped in exact mode.
/// </summary>
internal sealed class GeluFusionPass : IGraphPass
{
    public string Name => "gelu";

    public OptimizationLevel MinLevel => OptimizationLevel.All;

    public bool PreservesBits => false;

    public void Run(PassGraph g, CancellationToken ct)
    {
        int[] sole = PassPatterns.SoleConsumers(g);
        for (int i = 0; i < g.Nodes.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            CompiledNode div = g.Nodes[i];
            if (div.Kernel is not DivOp || div.Inputs.Length != 2 || div.Inputs[0] < 0 || !IsScalar(g, div.Inputs[1], MathF.Sqrt(2f), 1e-6f))
            {
                continue;
            }

            int x = div.Inputs[0];
            if (Next(g, sole, div, out CompiledNode erf, out int ie) is false || erf.Kernel is not FloatFunctionOp<ErfUnary>)
            {
                continue;
            }

            if (Next(g, sole, erf, out CompiledNode add, out int ia) is false || add.Kernel is not AddOp ||
                OtherScalarIs(g, add, erf.Outputs[0], 1f) is false)
            {
                continue;
            }

            if (Next(g, sole, add, out CompiledNode mul, out int im) is false || mul.Kernel is not MulOp ||
                mul.Inputs.Length != 2 || !(mul.Inputs[0] == x && mul.Inputs[1] == add.Outputs[0] || mul.Inputs[1] == x && mul.Inputs[0] == add.Outputs[0]))
            {
                continue;
            }

            if (Next(g, sole, mul, out CompiledNode half, out int ih) is false || half.Kernel is not MulOp ||
                OtherScalarIs(g, half, mul.Outputs[0], 0.5f) is false)
            {
                continue;
            }

            _ = (ie, ia, im);
            var gelu = g.Own(new FloatFunctionOp<GeluUnary>(new GeluUnary(false)));
            g.Nodes[ih] = new CompiledNode(half.Info, gelu, [x], half.Outputs);
        }
    }

    // The node that is the sole reader of node.Outputs[0], with its index.
    private static bool Next(PassGraph g, int[] sole, CompiledNode node, out CompiledNode next, out int index)
    {
        next = node;
        index = node.Outputs.Length == 1 && node.Outputs[0] >= 0 ? sole[node.Outputs[0]] : -1;
        if (index < 0)
        {
            return false;
        }

        next = g.Nodes[index];
        return next.Inputs.Length >= 1;
    }

    private static bool IsScalar(PassGraph g, int id, float value, float tolerance)
    {
        Tensor c = g.Constant(id);
        return !c.IsEmpty && c.DType == DType.F32 && c.Length == 1 && Math.Abs(c.Data<float>()[0] - value) <= tolerance;
    }

    // A two-input node reading `known` and a constant scalar equal to `value`.
    private static bool OtherScalarIs(PassGraph g, CompiledNode n, int known, float value)
    {
        if (n.Inputs.Length != 2)
        {
            return false;
        }

        int other = n.Inputs[0] == known ? n.Inputs[1] : n.Inputs[1] == known ? n.Inputs[0] : -1;
        return other >= 0 && n.Inputs[0] != n.Inputs[1] && IsScalar(g, other, value, 0f);
    }
}
