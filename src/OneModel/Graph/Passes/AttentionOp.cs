using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Passes;

/// <summary>Where the sequence axis sits in a rank-4 attention tensor.</summary>
internal enum AttentionLayout
{
    /// <summary>[batch, heads, positions, headDim].</summary>
    HeadsFirst,

    /// <summary>[batch, positions, heads, headDim], as a projection reshaped to heads leaves it.</summary>
    PositionsFirst,
}

/// <summary>
/// Scaled dot-product attention as one node: softmax(scale * Q K^T + mask) V with the fused kernel of docs/kernels-attention.md,
/// made by <see cref="AttentionFusionPass"/> from MatMul, Mul or Div, Add, Softmax and MatMul (and the transposes around them).
/// Inputs: Q, K, V, and an optional additive mask broadcastable to [batch, heads, qLen, kLen]. Q, K, V and the output may be in either
/// layout, so the head split and merge transposes need no copy.
/// </summary>
internal sealed class AttentionOp : OpKernel
{
    private readonly AttentionLayout _q, _k, _v, _out;
    private readonly float _scale;

    public AttentionOp(AttentionLayout q, AttentionLayout k, AttentionLayout v, AttentionLayout output, float scale)
    {
        _q = q;
        _k = k;
        _v = v;
        _out = output;
        _scale = scale;
    }

    // The blocked kernel has an exact flavour, but the odd-head-dimension fallback does not, and the pass that makes this node
    // does not run in exact mode.
    public override bool SupportsExact => false;

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

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0) || !ctx.HasInput(1) || !ctx.HasInput(2))
        {
            throw ctx.Fail("expects Q, K and V");
        }

        for (int i = 0; i < 3; i++)
        {
            if (ctx.InputType(i) != DType.F32 || ctx.InputShape(i).Rank != 4)
            {
                throw ctx.Unsupported($"input {i} must be a rank-4 float32 tensor, got {ctx.InputType(i)} {ctx.InputShape(i)}");
            }
        }

        Dims(ctx.InputShape(0), _q, out int b, out int h, out int sq, out int d);
        Dims(ctx.InputShape(1), _k, out int bk, out int hk, out int sk, out int dk);
        Dims(ctx.InputShape(2), _v, out int bv, out int hv, out int sv, out int dv);
        if (bk != b || hk != h || bv != b || hv != h || sv != sk || dk != d || dv != d)
        {
            throw ctx.Fail($"Q {ctx.InputShape(0)}, K {ctx.InputShape(1)} and V {ctx.InputShape(2)} do not describe one attention");
        }

        if (ctx.HasInput(3))
        {
            if (ctx.InputType(3) != DType.F32)
            {
                throw ctx.Unsupported("the mask must be float32");
            }

            Shape m = ctx.InputShape(3);
            if (m.Rank > 4)
            {
                throw ctx.Fail($"the mask {m} has more than 4 dimensions");
            }

            ReadOnlySpan<int> full = [b, h, sq, sk];
            for (int i = 0; i < m.Rank; i++)
            {
                int want = full[4 - m.Rank + i];
                if (m[i] != 1 && m[i] != want)
                {
                    throw ctx.Fail($"the mask {m} cannot be broadcast to [{b}, {h}, {sq}, {sk}]");
                }
            }
        }

        ctx.SetOutput(0, DType.F32, _out == AttentionLayout.HeadsFirst ? Shape.Create(b, h, sq, d) : Shape.Create(b, sq, h, d));
    }

    public override void Run(in OpRun run)
    {
        Tensor q = run.Inputs[0], k = run.Inputs[1], v = run.Inputs[2], o = run.Outputs[0];
        Dims(q.Shape, _q, out int b, out int h, out int sq, out int d);
        Dims(k.Shape, _k, out _, out _, out int sk, out _);
        if (b * h * sq * d == 0)
        {
            return;
        }

        var s = new AttentionShape { Batch = b, Heads = h, QLen = sq, KLen = sk, HeadDim = d, Scale = _scale };
        Strides(_q, h, sq, d, out s.QBatchStride, out s.QHeadStride, out s.QRowStride);
        Strides(_k, h, sk, d, out s.KBatchStride, out s.KHeadStride, out s.KRowStride);
        Strides(_v, h, sk, d, out s.VBatchStride, out s.VHeadStride, out s.VRowStride);
        Strides(_out, h, sq, d, out s.OBatchStride, out s.OHeadStride, out s.ORowStride);

        ReadOnlySpan<float> mask = default;
        if (run.Inputs.Length > 3 && !run.Inputs[3].IsEmpty)
        {
            Tensor m = run.Inputs[3];
            // Right-align the mask's dimensions with [batch, heads, qLen, kLen]; an extent of 1 broadcasts (stride 0).
            Span<int> md = stackalloc int[4];
            for (int i = 0; i < 4; i++)
            {
                int j = i - (4 - m.Rank);
                md[i] = j >= 0 ? m.Shape[j] : 1;
            }

            ReadOnlySpan<float> data = m.Data<float>();
            if (md[3] == 1 && sk != 1)
            {
                // One value per row: the kernel wants contiguous columns, so spread it out once.
                Tensor wide = run.Scratch(DType.F32, Shape.Create(md[0], md[1], md[2], sk));
                Span<float> w = wide.MutableData<float>();
                for (int r = 0; r < md[0] * md[1] * md[2]; r++)
                {
                    w.Slice(r * sk, sk).Fill(data[r]);
                }

                data = w;
                md[3] = sk;
            }

            s.MaskRowStride = md[2] == 1 ? 0 : md[3];
            s.MaskHeadStride = md[1] == 1 ? 0 : (long)md[2] * md[3];
            s.MaskBatchStride = md[0] == 1 ? 0 : (long)md[1] * md[2] * md[3];
            mask = data;
        }

        if ((d & 1) != 0)
        {
            Reference(in s, q.Data<float>(), k.Data<float>(), v.Data<float>(), mask, o.MutableData<float>());
            return;
        }

        AttentionDriver.Run(run.Exact, run.Path, in s, q.Data<float>(), k.Data<float>(), v.Data<float>(), mask, o.MutableData<float>(), run.Threads);
    }

    private static void Strides(AttentionLayout l, int heads, int pos, int d, out long batch, out long head, out long row)
    {
        if (l == AttentionLayout.HeadsFirst)
        {
            row = d;
            head = (long)pos * d;
            batch = heads * head;
        }
        else
        {
            row = (long)heads * d;
            head = d;
            batch = pos * row;
        }
    }

    // An odd head dimension is outside the blocked kernel's contract; this plain version keeps such models running.
    private static void Reference(in AttentionShape s, ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
        ReadOnlySpan<float> mask, Span<float> o)
    {
        var p = new float[s.KLen];
        for (int b = 0; b < s.Batch; b++)
        {
            for (int h = 0; h < s.Heads; h++)
            {
                for (int i = 0; i < s.QLen; i++)
                {
                    long qo = (b * s.QBatchStride) + (h * s.QHeadStride) + (i * s.QRowStride);
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < s.KLen; j++)
                    {
                        long ko = (b * s.KBatchStride) + (h * s.KHeadStride) + (j * s.KRowStride);
                        float dot = 0;
                        for (int x = 0; x < s.HeadDim; x++)
                        {
                            dot += q[(int)(qo + x)] * k[(int)(ko + x)];
                        }

                        float sc = dot * s.Scale;
                        if (!mask.IsEmpty)
                        {
                            sc += mask[(int)((b * s.MaskBatchStride) + (h * s.MaskHeadStride) + (i * s.MaskRowStride) + j)];
                        }

                        p[j] = sc;
                        max = Math.Max(max, sc);
                    }

                    float sum = 0;
                    for (int j = 0; j < s.KLen; j++)
                    {
                        p[j] = MathF.Exp(p[j] - max);
                        sum += p[j];
                    }

                    long oo = (b * s.OBatchStride) + (h * s.OHeadStride) + (i * s.ORowStride);
                    for (int x = 0; x < s.HeadDim; x++)
                    {
                        float acc = 0;
                        for (int j = 0; j < s.KLen; j++)
                        {
                            acc += p[j] * v[(int)((b * s.VBatchStride) + (h * s.VHeadStride) + (j * s.VRowStride) + x)];
                        }

                        o[(int)(oo + x)] = acc / sum;
                    }
                }
            }
        }
    }
}
