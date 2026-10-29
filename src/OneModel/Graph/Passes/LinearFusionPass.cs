using System;
using System.Collections.Generic;
using System.Threading;
using RecursiveAI.OneModel.Graph.Operators;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Passes;

/// <summary>
/// MatMul or Gemm with a constant float32 weight matrix becomes a <see cref="LinearOp"/>: the weights are packed once (Goto and
/// van de Geijn 2008, section 4: the panel layout the microkernel streams) instead of on every call, and a following bias Add
/// (constant vector over the last axis) and a following activation are applied to each finished tile while it is in cache.
/// </summary>
/// <remarks>
/// Bit-preserving in both modes: packing is a copy; the bias is one rounded addition per element exactly as Add does it (a Gemm
/// <c>beta</c> is applied to C once, at pass time, with the same multiply the kernel does per call); the activation runs the
/// operator's own kernel on each tile row. In fast mode the one possible difference is where a vector loop hands over to its scalar
/// tail in an activation, because rows are cut at tile edges. Gemm with alpha other than 1, transposed A, or a C that is not a
/// bias vector is left alone.
/// </remarks>
internal sealed class LinearFusionPass : IGraphPass
{
    public string Name => "linear-pack-fuse";

    public OptimizationLevel MinLevel => OptimizationLevel.All;

    public bool PreservesBits => true;

    public void Run(PassGraph g, CancellationToken ct)
    {
        int[] sole = PassPatterns.SoleConsumers(g);
        for (int i = 0; i < g.Nodes.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            CompiledNode node = g.Nodes[i];
            if (!TryMatch(g, node, out int a, out Tensor b, out bool transB, out float[]? bias, out int biasRank))
            {
                continue;
            }

            int k = transB ? b.Shape[1] : b.Shape[0];
            int n = transB ? b.Shape[0] : b.Shape[1];
            if (k == 0 || n == 0)
            {
                continue;
            }

            int last = i;
            int outId = node.Outputs[0];

            // A bias Add: the product's output is read only by it, the other operand is a bias vector.
            if (bias is null && sole[outId] is int j and >= 0 && g.Nodes[j].Kernel is AddOp)
            {
                CompiledNode add = g.Nodes[j];
                int other = add.Inputs[0] == outId ? add.Inputs[1] : add.Inputs[0];
                if (other >= 0 && add.Inputs[0] != add.Inputs[1] && PassPatterns.TryBiasVector(g.Constant(other), n, out float[] v, out int rank))
                {
                    bias = v;
                    biasRank = rank;
                    last = j;
                    outId = add.Outputs[0];
                }
            }

            FusedActivation? act = null;
            if (sole[outId] is int j2 and >= 0 && g.Nodes[j2].Inputs[0] == outId && g.Nodes[j2].Inputs.Length >= 1)
            {
                act = PassPatterns.ActivationOf(g, g.Nodes[j2]);
                if (act is not null)
                {
                    last = j2;
                }
            }

            Tensor packed = Pack(g, b, transB, k, n);
            var kernel = g.Own(new LinearOp(packed, k, n, bias, biasRank, act, g.Path));
            CompiledNode tail = g.Nodes[last];
            g.Nodes[last] = new CompiledNode(node.Info, kernel, [a], tail.Outputs);
        }
    }

    // Recognises a product with constant weights; fills the data input id, the weights, and the Gemm bias if there is one.
    private static bool TryMatch(PassGraph g, CompiledNode node, out int a, out Tensor b, out bool transB, out float[]? bias, out int biasRank)
    {
        a = -1;
        b = default;
        transB = false;
        bias = null;
        biasRank = 0;
        if (node.Inputs.Length < 2 || node.Inputs[0] < 0 || node.Inputs[1] < 0 || g.IsConstant(node.Inputs[0]))
        {
            return false;
        }

        a = node.Inputs[0];
        b = g.Constant(node.Inputs[1]);
        if (b.IsEmpty || b.DType != DType.F32 || b.Rank != 2 || !b.IsContiguous)
        {
            return false;
        }

        if (node.Kernel is MatMulOp)
        {
            return true;
        }

        if (node.Kernel is GemmOp gemm && gemm.Alpha == 1f && !gemm.TransA)
        {
            transB = gemm.TransB;
            int n = transB ? b.Shape[0] : b.Shape[1];
            if (node.Inputs.Length > 2 && node.Inputs[2] >= 0)
            {
                if (!PassPatterns.TryBiasVector(g.Constant(node.Inputs[2]), n, out float[] c, out biasRank))
                {
                    return false;
                }

                if (gemm.Beta != 1f)
                {
                    for (int i = 0; i < c.Length; i++)
                    {
                        c[i] = gemm.Beta * c[i];
                    }
                }

                bias = c;
            }

            return true;
        }

        return false;
    }

    // B as k x n packed for the session's path; a transposed B (Gemm transB) is transposed into a temporary first.
    private static Tensor Pack(PassGraph g, Tensor b, bool transB, int k, int n)
    {
        ReadOnlySpan<float> src = b.Data<float>();
        float[]? tmp = null;
        if (transB)
        {
            tmp = new float[k * n];
            for (int r = 0; r < n; r++)
            {
                for (int c = 0; c < k; c++)
                {
                    tmp[(c * n) + r] = src[(r * k) + c];
                }
            }

            src = tmp;
        }

        long len = Gemm.PackedBLength(g.Path, n, k);
        Tensor packed = Tensor.Create(DType.F32, Shape.Create(checked((int)len)));
        Gemm.PackB(g.Path, src, n, n, k, packed.MutableData<float>());
        return packed;
    }
}
