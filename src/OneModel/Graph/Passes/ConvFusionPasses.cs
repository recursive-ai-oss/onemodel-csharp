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
/// Conv followed by an Add of another tensor (the skip connection of a residual block): the sum is formed on each finished tile of
/// the convolution's matrix product, while the tile is in cache, and a following activation fuses on top (the activation pass runs
/// after this one). One rounded addition per element after the bias, as the stand-alone Add does, so the bits are the same; only
/// fast mode runs it, because exact mode's convolutions go through the exact matrix product. If the other operand turns out to
/// broadcast the output to a larger shape, the convolution runs into scratch and the stand-alone Add kernel finishes the job.
/// </summary>
internal sealed class ConvResidualFusionPass : IGraphPass
{
    public string Name => "conv-residual";

    public OptimizationLevel MinLevel => OptimizationLevel.All;

    public bool PreservesBits => true;

    public void Run(PassGraph g, CancellationToken ct)
    {
        if (g.Mode == ExecutionMode.Exact)
        {
            return;
        }

        int[] sole = PassPatterns.SoleConsumers(g);
        for (int i = 0; i < g.Nodes.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            CompiledNode conv = g.Nodes[i];
            if (conv.Kernel is not ConvOp convOp || convOp.HasActivation || convOp.HasResidual || conv.Inputs.Length < 2 ||
                conv.Inputs[0] < 0 || conv.Inputs[1] < 0 || g.IsConstant(conv.Inputs[0]))
            {
                continue;
            }

            Tensor w = g.Constant(conv.Inputs[1]);
            if (w.IsEmpty || w.DType != DType.F32 || (w.Rank != 3 && w.Rank != 4) || !w.IsContiguous || w.Length == 0)
            {
                continue;
            }

            // Depthwise convolutions keep their own kernel (no tile epilogue there).
            int kw = w.Shape[w.Rank - 1];
            int kh = w.Rank == 4 ? w.Shape[2] : 1;
            if (new ConvShape(1, w.Shape[1] * convOp.Group, 1, 1, w.Shape[0], kh, kw, 1, 1, 0, 0, 0, 0, 1, 1, convOp.Group).IsDepthwise)
            {
                continue;
            }

            int convOut = conv.Outputs[0];
            int j = sole[convOut];
            if (j < 0 || g.Nodes[j].Kernel is not AddOp || g.Nodes[j].Inputs.Length != 2)
            {
                continue;
            }

            CompiledNode add = g.Nodes[j];
            int other = add.Inputs[0] == convOut ? add.Inputs[1] : add.Inputs[0];
            if (other < 0 || other == convOut || (add.Inputs[0] != convOut && add.Inputs[1] != convOut))
            {
                continue;
            }

            int bias = conv.Inputs.Length > 2 ? conv.Inputs[2] : -1;
            ConvOp fused = g.Own(convOp.WithResidual());
            g.Nodes[j] = new CompiledNode(conv.Info, fused, [conv.Inputs[0], conv.Inputs[1], bias, other], add.Outputs);
        }
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

/// <summary>
/// Packs the constant weights of every non-depthwise Conv once, into the panel layout the fast convolution reads (A of the blocked
/// matrix product, <see cref="ConvPacked.PackWeights"/>), so a run never repacks them. It runs last, after the batch-norm fold and
/// the activation fusion have settled the weights. Copies only: the packed convolution gives the same bits as the unpacked one, so
/// the pass preserves bits; it does nothing in exact mode, where convolutions run through the exact matrix product on the original
/// weights. The original weights are released once nothing else reads them, so the model holds one copy.
/// </summary>
internal sealed class ConvPackPass : IGraphPass
{
    public string Name => "conv-pack";

    public OptimizationLevel MinLevel => OptimizationLevel.All;

    public bool PreservesBits => true;

    public void Run(PassGraph g, CancellationToken ct)
    {
        if (g.Mode == ExecutionMode.Exact)
        {
            return;
        }

        int[] uses = g.UseCounts();
        for (int i = 0; i < g.Nodes.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            CompiledNode conv = g.Nodes[i];
            if (conv.Kernel is not ConvOp convOp || conv.Inputs.Length < 2 || conv.Inputs[0] < 0 || conv.Inputs[1] < 0 || g.IsConstant(conv.Inputs[0]))
            {
                continue;
            }

            int wId = conv.Inputs[1];
            Tensor w = g.Constant(wId);
            if (w.IsEmpty || w.DType != DType.F32 || (w.Rank != 3 && w.Rank != 4) || !w.IsContiguous || w.Length == 0)
            {
                continue;
            }

            int group = convOp.Group;
            int kw = w.Shape[w.Rank - 1];
            int kh = w.Rank == 4 ? w.Shape[2] : 1;
            var s = new ConvShape(1, w.Shape[1] * group, 1, 1, w.Shape[0], kh, kw, 1, 1, 0, 0, 0, 0, 1, 1, group);
            if (s.IsDepthwise)
            {
                continue;
            }

            long len = ConvPacked.PackedWeightsLength(g.Path, s);
            if (len > int.MaxValue / 4)
            {
                continue;
            }

            Tensor packed = Tensor.Create(DType.F32, Shape.Create((int)len));
            ConvPacked.PackWeights(g.Path, s, w.Data<float>(), packed.MutableData<float>());
            ConvOp packedOp = g.Own(convOp.WithPackedWeights(packed, w.Shape, g.Path));
            var inputs = (int[])conv.Inputs.Clone();
            inputs[1] = -1;
            g.Nodes[i] = new CompiledNode(conv.Info, packedOp, inputs, conv.Outputs);
            if (--uses[wId] == 0)
            {
                g.ReleaseConstant(wId);
            }
        }
    }
}
