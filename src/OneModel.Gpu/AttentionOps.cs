using System;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Graph.Passes;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>
/// The fused scaled dot-product attention node that the attention rewrite makes from <c>MatMul, Mul or Div, Add, Softmax, MatMul</c>:
/// <c>softmax(scale * Q K^T + mask) V</c> in one kernel that never writes the scores (Dao et al. 2022 for the tiling, Milakov and Gimelshein
/// 2018 for the running normaliser). The same kernel serves fast and exact mode: every sum has a fixed order and there are no atomics.
/// </summary>
internal static class AttentionOps
{
    /// <summary>The table key of the node (its ONNX operator type is that of the last product, which says nothing about the fusion).</summary>
    public const string Key = "~Attention";

    public static readonly GpuOpEntry Entry = new(Key, true, _ => null, Bind);

    private static string? Bind(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        var op = (AttentionOp)ctx.Step.Node.Kernel;
        if (!ctx.HasInput(0) || !ctx.HasInput(1) || !ctx.HasInput(2))
        {
            return "attention needs Q, K and V";
        }

        for (int i = 0; i < 3; i++)
        {
            if (ctx.InputType(i) != DType.F32)
            {
                return $"input {i} is {ctx.InputType(i)}, and the fused attention kernel is used for float32";
            }
        }

        Dims(ctx.InputShape(0), op.QueryLayout, out int b, out int h, out int sq, out int d);
        Dims(ctx.InputShape(1), op.KeyLayout, out _, out _, out int sk, out _);
        if (b == 0 || h == 0 || sq == 0 || sk == 0 || d == 0)
        {
            return "an attention input is empty";
        }

        if (d > AttentionKernels.HeadDimLimit)
        {
            return $"the head dimension {d} is above the kernel's limit of {AttentionKernels.HeadDimLimit}";
        }

        for (int i = 0; i < 3; i++)
        {
            if (ctx.FitsBinding(ctx.InputShape(i).Length, DeviceElement.F32) is { } big)
            {
                return big;
            }
        }

        int[]? maskDims = null;
        bool widen = false;
        DeviceConstant? zeros = null;
        if (ctx.HasInput(3))
        {
            if (ctx.InputType(3) != DType.F32)
            {
                return "the mask is not float32";
            }

            Shape m = ctx.InputShape(3);
            maskDims = [1, 1, 1, 1];
            for (int i = 0; i < m.Rank; i++)
            {
                maskDims[4 - m.Rank + i] = m[i];
            }

            if (m.Length == 0)
            {
                return "the mask is empty";
            }

            if (maskDims[3] == 1 && sk != 1)
            {
                // One value per row: the kernel reads a mask row of one value per key, so the value is spread across the keys once.
                widen = true;
                long wide = (long)maskDims[0] * maskDims[1] * maskDims[2] * sk;
                if (ctx.FitsBinding(wide, DeviceElement.F32) is { } bigMask)
                {
                    return bigMask;
                }

                using Tensor z = Tensor.Create(DType.F32, Shape.Create(sk));
                z.MutableData<float>().Clear();
                zeros = ctx.Backend.Constants.Upload(z);
                ctx.OwnForPlan(zeros);
                maskDims[3] = sk;
            }
        }

        // Probe the layout rules now so that a shape the kernel rejects stays on the CPU instead of failing a run.
        var problem = new AttentionProblem(b, h, sq, sk, d) { Scale = op.Scale };
        try
        {
            if (maskDims is not null)
            {
                _ = GpuAttentionLayout.ForMask(maskDims, b, h, sq, sk);
            }
        }
        catch (GpuKernelShapeException e)
        {
            return e.Message;
        }

        _ = problem;
        binding = new AttentionBinding(ctx, op, b, h, sq, sk, d, maskDims, widen, zeros);
        return null;
    }

    private static void Dims(Shape s, AttentionLayout l, out int b, out int h, out int pos, out int d)
    {
        b = s[0];
        if (l == AttentionLayout.HeadsFirst)
        {
            h = s[1];
            pos = s[2];
        }
        else
        {
            pos = s[1];
            h = s[2];
        }

        d = s[3];
    }

    private static GpuAttentionLayout LayoutOf(AttentionLayout l, int heads, int rows, int d) =>
        l == AttentionLayout.HeadsFirst ? GpuAttentionLayout.HeadsFirst(heads, rows, d) : GpuAttentionLayout.PositionsFirst(heads, rows, d);

    private sealed class AttentionBinding : GpuBinding
    {
        private readonly AttentionKernels _kernels;
        private readonly AttentionProblem _problem;
        private readonly Operand _q, _k, _v, _mask;
        private readonly int[]? _maskDims;
        private readonly bool _widen;
        private readonly DeviceConstant? _zeros;
        private readonly int _outId;
        private readonly GpuKernelMode _mode;
        private readonly ulong _budget;
        private readonly int[] _maskShapeIn;

        public AttentionBinding(GpuClaimContext ctx, AttentionOp op, int b, int h, int sq, int sk, int d, int[]? maskDims, bool widen, DeviceConstant? zeros)
        {
            _kernels = ctx.Backend.Attention;
            _mode = ctx.Backend.KernelMode;
            _budget = ctx.Backend.Device.Capabilities.Compute.SubmissionWorkBudget;
            _problem = new AttentionProblem(b, h, sq, sk, d)
            {
                Scale = op.Scale,
                Q = LayoutOf(op.QueryLayout, h, sq, d),
                K = LayoutOf(op.KeyLayout, h, sk, d),
                V = LayoutOf(op.ValueLayout, h, sk, d),
                O = LayoutOf(op.OutputLayout, h, sq, d),
                MaskedRows = GpuAttentionMaskedRows.Nan,
            };
            _q = Operand.Of(ctx, 0);
            _k = Operand.Of(ctx, 1);
            _v = Operand.Of(ctx, 2);
            _outId = ctx.OutputId(0);
            _widen = widen;
            _zeros = zeros;
            _maskDims = maskDims;
            if (maskDims is not null)
            {
                _mask = Operand.Of(ctx, 3);
                Shape m = ctx.InputShape(3);
                _maskShapeIn = GpuClaimContext.Dims(m);
                ScratchF32 = widen ? (long)maskDims[0] * maskDims[1] * maskDims[2] * sk : 0;
            }
            else
            {
                _maskShapeIn = [];
            }

            // Two products over the keys, 2 flops per multiply-add, and the softmax is small beside them.
            Flops = 4.0 * b * h * sq * sk * d;
        }

        private long ScratchF32 { get; }

        public override long ScratchF32Elements => ScratchF32;

        public override DeviceElement OutputElement(int output) => DeviceElement.F32;

        public override void Encode(GpuRunContext ctx)
        {
            GpuBuffer<float> q = _q.Get<float>(ctx, out int offQ);
            GpuBuffer<float> k = _k.Get<float>(ctx, out int offK);
            GpuBuffer<float> v = _v.Get<float>(ctx, out int offV);
            GpuBuffer<float> o = ctx.Buffer<float>(_outId, out int offO);
            GpuBuffer<float>? mask = null;
            int offM = 0;
            if (_maskDims is not null)
            {
                mask = _mask.Get<float>(ctx, out offM);
                if (_widen)
                {
                    GpuBuffer<float> wide = ctx.Scratch(this, out int offW);
                    GpuBuffer<float> zeros = _zeros!.F32!;
                    int[] zeroShape = [1, 1, 1, _problem.KLen];
                    ctx.Backend.Elementwise.EncodeBinary(ctx.Encoder, GpuBinaryOp.Add, mask, offM, _maskShapeIn, zeros, 0, zeroShape, wide, offW, null, _mode);
                    mask = wide;
                    offM = offW;
                }
            }

            AttentionProblem p = _problem;
            p.Q = p.Q with { Offset = offQ };
            p.K = p.K with { Offset = offK };
            p.V = p.V with { Offset = offV };
            p.O = p.O with { Offset = offO };
            if (mask is not null)
            {
                p.Mask = GpuAttentionLayout.ForMask(_maskDims!, p.Batch, p.Heads, p.QLen, p.KLen, offM);
            }

            double ops = Flops;
            if (_budget != 0 && p.Batch > 1 && ops > _budget / 4.0 && Fits(p))
            {
                // A single dispatch that keeps the GPU busy for long can be stopped by the system and leave the output half written, so
                // a heavy attention goes in one submission piece per batch entry.
                for (int i = 0; i < p.Batch; i++)
                {
                    AttentionProblem piece = p;
                    piece.Batch = 1;
                    piece.Q = p.Q with { Offset = p.Q.Offset + i * p.Q.BatchStride };
                    piece.K = p.K with { Offset = p.K.Offset + i * p.K.BatchStride };
                    piece.V = p.V with { Offset = p.V.Offset + i * p.V.BatchStride };
                    piece.O = p.O with { Offset = p.O.Offset + i * p.O.BatchStride };
                    piece.Mask = p.Mask with { Offset = p.Mask.Offset + i * p.Mask.BatchStride };
                    ctx.Encoder.ReportWork(ops / p.Batch);
                    _kernels.Encode(ctx.Encoder, piece, q, k, v, o, mask, null, _mode);
                }

                return;
            }

            ctx.Encoder.ReportWork(ops);
            _kernels.Encode(ctx.Encoder, p, q, k, v, o, mask, null, _mode);
        }

        private static bool Fits(in AttentionProblem p)
        {
            long max = (long)(p.Batch - 1) * Math.Max(Math.Max(p.Q.BatchStride, p.K.BatchStride), Math.Max(p.V.BatchStride, p.O.BatchStride));
            return max < int.MaxValue / 2;
        }
    }
}
