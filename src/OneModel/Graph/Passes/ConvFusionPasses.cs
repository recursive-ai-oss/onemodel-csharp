using System;
using System.Threading;
using RecursiveAI.OneModel.Graph.Operators;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Passes;

/// <summary>
/// Conv followed by inference-mode BatchNormalization with constant statistics: the batch norm is folded into the convolution's
/// weights and bias, W' = W * s per output channel and b' = (b - mean) * s + beta with s = scale / sqrt(var + epsilon), computed
/// in double precision and rounded once. The result differs from running the two operators in the last bits (the scale moves
/// before the sum), so the pass is skipped in exact mode.
/// </summary>
internal sealed class ConvBatchNormFoldPass : IGraphPass
{
    public string Name => "conv-batchnorm-fold";

    public OptimizationLevel MinLevel => OptimizationLevel.All;

    public bool PreservesBits => false;

    public void Run(PassGraph g, CancellationToken ct)
    {
        int[] sole = PassPatterns.SoleConsumers(g);
        for (int i = 0; i < g.Nodes.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            CompiledNode conv = g.Nodes[i];
            if (conv.Kernel is not ConvOp convOp || convOp.HasActivation || conv.Inputs.Length < 2 || conv.Inputs[0] < 0 || conv.Inputs[1] < 0)
            {
                continue;
            }

            int convOut = conv.Outputs[0];
            int j = sole[convOut];
            if (j < 0 || g.Nodes[j].Kernel is not BatchNormalizationOp bn || g.Nodes[j].Inputs[0] != convOut)
            {
                continue;
            }

            CompiledNode bnNode = g.Nodes[j];
            Tensor w = g.Constant(conv.Inputs[1]);
            if (w.IsEmpty || w.DType != DType.F32 || w.Rank < 3 || !w.IsContiguous)
            {
                continue;
            }

            int cout = w.Shape[0];
            if (!TryVector(g, bnNode, 1, cout, out ReadOnlySpan<float> scale) || !TryVector(g, bnNode, 2, cout, out ReadOnlySpan<float> beta) ||
                !TryVector(g, bnNode, 3, cout, out ReadOnlySpan<float> mean) || !TryVector(g, bnNode, 4, cout, out ReadOnlySpan<float> variance))
            {
                continue;
            }

            ReadOnlySpan<float> oldBias = default;
            if (conv.Inputs.Length > 2 && conv.Inputs[2] >= 0)
            {
                Tensor bt = g.Constant(conv.Inputs[2]);
                if (bt.IsEmpty || bt.DType != DType.F32 || bt.Length != cout)
                {
                    continue;
                }

                oldBias = bt.Data<float>();
            }

            Tensor w2 = Tensor.Create(DType.F32, w.Shape);
            Tensor b2 = Tensor.Create(DType.F32, Shape.Create(cout));
            ReadOnlySpan<float> wd = w.Data<float>();
            Span<float> w2d = w2.MutableData<float>(), b2d = b2.MutableData<float>();
            int per = wd.Length / cout;
            for (int c = 0; c < cout; c++)
            {
                double s = scale[c] / Math.Sqrt((double)variance[c] + bn.Epsilon);
                for (int e = 0; e < per; e++)
                {
                    w2d[(c * per) + e] = (float)(wd[(c * per) + e] * s);
                }

                double b0 = oldBias.IsEmpty ? 0.0 : oldBias[c];
                b2d[c] = (float)(((b0 - mean[c]) * s) + beta[c]);
            }

            int wId = g.AddValue(g.NameOf(conv.Inputs[1]) + "#bnfold", w2);
            int bId = g.AddValue(g.NameOf(conv.Outputs[0]) + "#bnfold_bias", b2);
            g.Nodes[j] = new CompiledNode(conv.Info, conv.Kernel, [conv.Inputs[0], wId, bId], bnNode.Outputs);
        }
    }

    private static bool TryVector(PassGraph g, CompiledNode n, int slot, int count, out ReadOnlySpan<float> v)
    {
        v = default;
        if (slot >= n.Inputs.Length || n.Inputs[slot] < 0)
        {
            return false;
        }

        Tensor t = g.Constant(n.Inputs[slot]);
        if (t.IsEmpty || t.DType != DType.F32 || t.Rank != 1 || t.Length != count)
        {
            return false;
        }

        v = t.Data<float>();
        return true;
    }
}

/// <summary>
/// Conv followed by an activation (Relu, Clip with constant bounds, HardSwish, HardSigmoid, Sigmoid, Gelu, Tanh, LeakyRelu): the
/// activation runs on each finished tile of the convolution's matrix product (on each plane after the depthwise kernel) instead of
/// as a separate pass over the output. The convolution's bias is applied there too. The activation is the operator's own kernel
/// applied to the same values, so exact mode is bit-identical; fast mode can differ only where a vector loop meets its scalar
/// tail, because rows are cut at tile edges.
/// </summary>
internal sealed class ConvActivationFusionPass : IGraphPass
{
    public string Name => "conv-activation";

    public OptimizationLevel MinLevel => OptimizationLevel.All;

    public bool PreservesBits => true;

    public void Run(PassGraph g, CancellationToken ct)
    {
        int[] sole = PassPatterns.SoleConsumers(g);
        for (int i = 0; i < g.Nodes.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            CompiledNode conv = g.Nodes[i];
            if (conv.Kernel is not ConvOp convOp || convOp.HasActivation)
            {
                continue;
            }

            int convOut = conv.Outputs[0];
            int j = sole[convOut];
            if (j < 0 || g.Nodes[j].Inputs[0] != convOut)
            {
                continue;
            }

            CompiledNode actNode = g.Nodes[j];
            FusedActivation? act = PassPatterns.ActivationOf(g, actNode);
            if (act is null)
            {
                continue;
            }

            ConvOp fused = g.Own(convOp.WithActivation(act));
            g.Nodes[j] = new CompiledNode(conv.Info, fused, conv.Inputs, actNode.Outputs);
        }
    }
}
