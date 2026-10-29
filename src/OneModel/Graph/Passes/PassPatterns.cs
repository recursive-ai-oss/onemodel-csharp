using System;
using System.Collections.Generic;
using RecursiveAI.OneModel.Graph.Operators;
using RecursiveAI.OneModel.Graph.Operators.Elementwise;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Passes;

/// <summary>Helpers the fusion passes share for finding patterns in a <see cref="PassGraph"/>.</summary>
internal static class PassPatterns
{
    /// <summary>
    /// For each value read by exactly one node input (and not a graph output), the index of that node; -1 for every other value.
    /// A fusion may absorb an intermediate value only when this is not -1: nobody else sees it.
    /// </summary>
    public static int[] SoleConsumers(PassGraph g)
    {
        int[] uses = g.UseCounts();
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

        return sole;
    }

    /// <summary>
    /// The activation a node applies to its single input, when it is one the epilogue knows (Relu, Clip with constant bounds,
    /// HardSwish, HardSigmoid, Sigmoid, Gelu, Tanh, LeakyRelu); otherwise null. The node's data input must be input 0.
    /// </summary>
    public static FusedActivation? ActivationOf(PassGraph g, CompiledNode n)
    {
        switch (n.Kernel)
        {
            case ReluOp:
                return new ReluActivation();
            case FloatFunctionOp<HardSwishUnary> k:
                return new FloatActivation<HardSwishUnary>(k.Function);
            case FloatFunctionOp<HardSigmoidUnary> k:
                return new FloatActivation<HardSigmoidUnary>(k.Function);
            case FloatFunctionOp<SigmoidUnary> k:
                return new FloatActivation<SigmoidUnary>(k.Function);
            case FloatFunctionOp<GeluUnary> k:
                return new FloatActivation<GeluUnary>(k.Function);
            case FloatFunctionOp<TanhUnary> k:
                return new FloatActivation<TanhUnary>(k.Function);
            case FloatFunctionOp<LeakyReluUnary> k:
                return new FloatActivation<LeakyReluUnary>(k.Function);
            case ClipOp clip:
                if (clip.BoundsAreAttributes)
                {
                    (float lo, float hi) = clip.AttributeBounds;
                    return new ClipActivation(lo, hi);
                }

                float lo2 = float.MinValue, hi2 = float.MaxValue;
                for (int i = 1; i <= 2; i++)
                {
                    int id = i < n.Inputs.Length ? n.Inputs[i] : -1;
                    if (id < 0)
                    {
                        continue;
                    }

                    Tensor c = g.Constant(id);
                    if (c.IsEmpty || c.DType != DType.F32 || c.Length != 1)
                    {
                        return null;
                    }

                    if (i == 1)
                    {
                        lo2 = c.Data<float>()[0];
                    }
                    else
                    {
                        hi2 = c.Data<float>()[0];
                    }
                }

                return new ClipActivation(lo2, hi2);
            default:
                return null;
        }
    }

    /// <summary>
    /// A constant float32 operand that adds one value per output feature: shape [N] or [1, ..., 1, N]. Returns its values and rank.
    /// </summary>
    public static bool TryBiasVector(Tensor c, int n, out float[] values, out int rank)
    {
        values = [];
        rank = 0;
        if (c.IsEmpty || c.DType != DType.F32 || c.Rank == 0 || c.Length != n || c.Shape[c.Rank - 1] != n)
        {
            return false;
        }

        for (int i = 0; i < c.Rank - 1; i++)
        {
            if (c.Shape[i] != 1)
            {
                return false;
            }
        }

        values = c.Data<float>().ToArray();
        rank = c.Rank;
        return true;
    }
}
