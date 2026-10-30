using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Transformer;

/// <summary>
/// The Attention operator of the default domain (opset 23 and 24): scaled dot-product attention with optional grouped key/value heads,
/// an additive or boolean mask, causal masking, key/value caches, soft-capping and the intermediate scores as an optional output.
/// </summary>
/// <remarks>
/// Per batch entry and query head: <c>S = scale * Q K^T</c> (scale defaults to 1/sqrt(head_size)); with <c>softcap</c> not zero
/// <c>S = softcap * tanh(S / softcap)</c>; the mask is added (boolean masks keep true and send false to -infinity; causal masking keeps
/// key j for query i when j &lt;= i, counting positions after the past); keys at or beyond <c>nonpad_kv_seqlen[b]</c> (opset 24) are
/// dropped; then a row-wise softmax and <c>Y = P V</c>. A query head h reads key/value head h / (q_heads / kv_heads). Scores are
/// accumulated in binary32 in increasing index order. This is a plain loop kernel for coverage and correctness; the fused
/// attention pass in <c>Graph/Passes</c> remains the fast path for transformer encoders. Not exact: it uses exp and tanh from the
/// platform's math library.
/// </remarks>
internal sealed class OnnxAttentionOp : OpKernel
{
    public static readonly OpDef Def = new("Attention", 23, int.MaxValue, static n => new OnnxAttentionOp(n));

    private readonly bool _causal;
    private readonly int _qHeads, _kvHeads, _qkMode;
    private readonly float _scale, _softcap;
    private readonly bool _hasScale;

    private OnnxAttentionOp(NodeInfo n)
    {
        _causal = n.GetInt("is_causal", 0) != 0;
        _qHeads = (int)n.GetInt("q_num_heads", 0);
        _kvHeads = (int)n.GetInt("kv_num_heads", 0);
        _qkMode = (int)n.GetInt("qk_matmul_output_mode", 0);
        _hasScale = n.Has("scale");
        _scale = n.GetFloat("scale", 0);
        _softcap = n.GetFloat("softcap", 0);
        if (_qkMode is < 0 or > 3)
        {
            throw n.Error("attribute 'qk_matmul_output_mode' must be 0 to 3");
        }
    }

    private struct Geometry
    {
        public int Batch, QHeads, KvHeads, QSeq, KvSeq, PastSeq, HeadSize, VHeadSize;
        public bool ThreeD;

        public readonly int Total => PastSeq + KvSeq;
    }

    private Geometry Resolve(Shape q, Shape k, Shape v, Shape pastK, bool hasPast, InferContext ctx)
    {
        var g = new Geometry { ThreeD = q.Rank == 3 };
        if (q.Rank is not (3 or 4) || k.Rank != q.Rank || v.Rank != q.Rank)
        {
            throw ctx.Fail("Q, K and V must all be 3-D or all 4-D");
        }

        g.Batch = q[0];
        if (g.ThreeD)
        {
            if (_qHeads <= 0 || _kvHeads <= 0)
            {
                throw ctx.Fail("3-D inputs need the q_num_heads and kv_num_heads attributes");
            }

            g.QHeads = _qHeads;
            g.KvHeads = _kvHeads;
            g.QSeq = q[1];
            g.KvSeq = k[1];
            if (q[2] % g.QHeads != 0 || k[2] % g.KvHeads != 0 || v[2] % g.KvHeads != 0)
            {
                throw ctx.Fail("the hidden sizes are not divisible by the head counts");
            }

            g.HeadSize = q[2] / g.QHeads;
            g.VHeadSize = v[2] / g.KvHeads;
            if (k[2] / g.KvHeads != g.HeadSize)
            {
                throw ctx.Fail("K and Q head sizes differ");
            }
        }
        else
        {
            g.QHeads = q[1];
            g.KvHeads = k[1];
            g.QSeq = q[2];
            g.KvSeq = k[2];
            g.HeadSize = q[3];
            g.VHeadSize = v[3];
            if (k[3] != g.HeadSize || v[1] != g.KvHeads || v[2] != g.KvSeq)
            {
                throw ctx.Fail("K and V do not fit Q");
            }
        }

        if (g.KvHeads == 0 || g.QHeads % g.KvHeads != 0 || k[0] != g.Batch || v[0] != g.Batch)
        {
            throw ctx.Fail("the query head count must be a multiple of the key/value head count, with equal batch sizes");
        }

        if (hasPast)
        {
            if (pastK.Rank != 4 || pastK[0] != g.Batch || pastK[1] != g.KvHeads || pastK[3] != g.HeadSize)
            {
                throw ctx.Fail($"past_key {pastK} does not fit the key heads");
            }

            g.PastSeq = pastK[2];
        }

        return g;
    }

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 3 || !ctx.HasInput(0) || !ctx.HasInput(1) || !ctx.HasInput(2))
        {
            throw ctx.Fail("expects Q, K and V");
        }

        if (ctx.InputType(0) != DType.F32 || ctx.InputType(1) != DType.F32 || ctx.InputType(2) != DType.F32)
        {
            throw ctx.Unsupported($"element type {ctx.InputType(0)} is not supported yet (float32 only)");
        }

        bool hasPast = ctx.HasInput(4) && ctx.HasInput(5);
        Geometry g = Resolve(ctx.InputShape(0), ctx.InputShape(1), ctx.InputShape(2), hasPast ? ctx.InputShape(4) : default, hasPast, ctx);
        ctx.SetOutput(0, DType.F32, g.ThreeD
            ? Shape.Create(g.Batch, g.QSeq, g.QHeads * g.VHeadSize)
            : Shape.Create(g.Batch, g.QHeads, g.QSeq, g.VHeadSize));
        if (ctx.OutputCount > 1)
        {
            ctx.SetOutput(1, DType.F32, Shape.Create(g.Batch, g.KvHeads, g.Total, g.HeadSize));
        }

        if (ctx.OutputCount > 2)
        {
            ctx.SetOutput(2, DType.F32, Shape.Create(g.Batch, g.KvHeads, g.Total, g.VHeadSize));
        }

        if (ctx.OutputCount > 3)
        {
            ctx.SetOutput(3, DType.F32, Shape.Create(g.Batch, g.QHeads, g.QSeq, g.Total));
        }

        if (ctx.HasInput(3) && ctx.InputType(3) != DType.Bool && ctx.InputType(3) != DType.F32)
        {
            throw ctx.Unsupported($"attn_mask of element type {ctx.InputType(3)} is not supported yet");
        }
    }

    public override void Run(in OpRun run)
    {
        Tensor qt = run.Inputs[0], kt = run.Inputs[1], vt = run.Inputs[2];
        bool hasPast = run.Inputs.Length > 5 && !run.Inputs[4].IsEmpty && !run.Inputs[5].IsEmpty;
        Tensor pastK = hasPast ? run.Inputs[4] : default, pastV = hasPast ? run.Inputs[5] : default;
        Geometry g = ResolveRun(qt.Shape, kt.Shape, vt.Shape, hasPast ? pastK.Shape : default, hasPast);
        int total = g.Total, hs = g.HeadSize, vhs = g.VHeadSize;

        // Present key and value: the cache followed by the new keys, per (batch, head), in the 4-D layout.
        Tensor presentK = run.Outputs.Length > 1 && !run.Outputs[1].IsEmpty ? run.Outputs[1] : run.Scratch(DType.F32, Shape.Create(g.Batch, g.KvHeads, total, hs));
        Tensor presentV = run.Outputs.Length > 2 && !run.Outputs[2].IsEmpty ? run.Outputs[2] : run.Scratch(DType.F32, Shape.Create(g.Batch, g.KvHeads, total, vhs));
        Build(presentK.MutableData<float>(), hasPast ? pastK.Data<float>() : default, kt.Data<float>(), g, hs, g.PastSeq, g.KvSeq, g.ThreeD);
        Build(presentV.MutableData<float>(), hasPast ? pastV.Data<float>() : default, vt.Data<float>(), g, vhs, g.PastSeq, g.KvSeq, g.ThreeD);

        ReadOnlySpan<float> q = qt.Data<float>(), pk = presentK.Data<float>(), pv = presentV.Data<float>();
        Span<float> y = run.Outputs[0].MutableData<float>();
        Span<float> qkOut = run.Outputs.Length > 3 && !run.Outputs[3].IsEmpty ? run.Outputs[3].MutableData<float>() : default;
        Tensor mask = run.Inputs.Length > 3 ? run.Inputs[3] : default;
        ReadOnlySpan<long> nonpad = run.Inputs.Length > 6 && !run.Inputs[6].IsEmpty ? run.Inputs[6].Data<long>() : default;
        float scale = _hasScale ? _scale : 1f / MathF.Sqrt(hs);
        int group = g.QHeads / g.KvHeads;
        Span<float> row = run.Scratch(DType.F32, Shape.Create(total)).MutableData<float>();

        for (int b = 0; b < g.Batch; b++)
        {
            for (int h = 0; h < g.QHeads; h++)
            {
                int kvh = h / group;
                ReadOnlySpan<float> keys = pk.Slice(((b * g.KvHeads) + kvh) * total * hs, total * hs);
                ReadOnlySpan<float> vals = pv.Slice(((b * g.KvHeads) + kvh) * total * vhs, total * vhs);
                for (int i = 0; i < g.QSeq; i++)
                {
                    ReadOnlySpan<float> qi = g.ThreeD ? q.Slice((((b * g.QSeq) + i) * g.QHeads * hs) + (h * hs), hs) : q.Slice((((((b * g.QHeads) + h) * g.QSeq) + i)) * hs, hs);
                    long qkBase = ((((long)b * g.QHeads) + h) * g.QSeq + i) * total;
                    for (int j = 0; j < total; j++)
                    {
                        float dot = 0;
                        ReadOnlySpan<float> kj = keys.Slice(j * hs, hs);
                        for (int d = 0; d < hs; d++)
                        {
                            dot += qi[d] * kj[d];
                        }

                        float s = scale * dot;
                        Save(qkOut, 0, qkBase + j, s);
                        if (_softcap != 0)
                        {
                            s = _softcap * MathF.Tanh(s / _softcap);
                        }

                        Save(qkOut, 1, qkBase + j, s);
                        if (!mask.IsEmpty)
                        {
                            s = AddMask(mask, s, b, h, i, j);
                        }

                        // Causal: key j is visible to query i when j <= i, counting the cache before the new keys.
                        if (_causal && j > i + g.PastSeq)
                        {
                            s = float.NegativeInfinity;
                        }

                        if (!nonpad.IsEmpty && j >= nonpad[b])
                        {
                            s = float.NegativeInfinity;
                        }

                        Save(qkOut, 2, qkBase + j, s);
                        row[j] = s;
                    }

                    float max = float.NegativeInfinity;
                    for (int j = 0; j < total; j++)
                    {
                        max = MathF.Max(max, row[j]);
                    }

                    float sum = 0;
                    for (int j = 0; j < total; j++)
                    {
                        // A fully masked row has no finite maximum; it keeps weight 0 everywhere instead of NaN.
                        row[j] = float.IsNegativeInfinity(max) ? 0f : MathF.Exp(row[j] - max);
                        sum += row[j];
                    }

                    for (int j = 0; j < total; j++)
                    {
                        row[j] = sum == 0 ? 0f : row[j] / sum;
                        Save(qkOut, 3, qkBase + j, row[j]);
                    }

                    Span<float> yi = g.ThreeD ? y.Slice((((b * g.QSeq) + i) * g.QHeads * vhs) + (h * vhs), vhs) : y.Slice(((((b * g.QHeads) + h) * g.QSeq) + i) * vhs, vhs);
                    yi.Clear();
                    for (int j = 0; j < total; j++)
                    {
                        float p = row[j];
                        ReadOnlySpan<float> vj = vals.Slice(j * vhs, vhs);
                        for (int d = 0; d < vhs; d++)
                        {
                            yi[d] += p * vj[d];
                        }
                    }
                }
            }
        }
    }

    // The quantity written to qk_matmul_output is the one the mode names: 0 scaled scores, 1 after soft-capping, 2 after the mask, 3 after softmax.
    private void Save(Span<float> qkOut, int stage, long at, float value)
    {
        if (!qkOut.IsEmpty && _qkMode == stage)
        {
            qkOut[(int)at] = value;
        }
    }

    private static float AddMask(Tensor mask, float s, int b, int h, int i, int j)
    {
        // The mask broadcasts against (batch, heads, query, key) from the right.
        Shape ms = mask.Shape;
        Span<int> coord = [b, h, i, j];
        int at = 0, stride = 1;
        for (int d = ms.Rank - 1; d >= 0; d--)
        {
            int c = ms[d] == 1 ? 0 : coord[4 - ms.Rank + d];
            if (c >= ms[d])
            {
                // A mask shorter than the keys (padded key/value caches) leaves the remaining keys out.
                return float.NegativeInfinity;
            }

            at += c * stride;
            stride *= ms[d];
        }

        if (mask.DType == DType.Bool)
        {
            return mask.Data<bool>()[at] ? s : float.NegativeInfinity;
        }

        return s + mask.Data<float>()[at];
    }

    // Copies past then new elements into the (batch, head, total, size) layout.
    private static void Build(Span<float> dst, ReadOnlySpan<float> past, ReadOnlySpan<float> now, in Geometry g, int size, int pastSeq, int newSeq, bool threeD)
    {
        int total = pastSeq + newSeq;
        for (int b = 0; b < g.Batch; b++)
        {
            for (int h = 0; h < g.KvHeads; h++)
            {
                Span<float> d = dst.Slice(((b * g.KvHeads) + h) * total * size, total * size);
                if (pastSeq > 0)
                {
                    past.Slice(((b * g.KvHeads) + h) * pastSeq * size, pastSeq * size).CopyTo(d);
                }

                for (int t = 0; t < newSeq; t++)
                {
                    ReadOnlySpan<float> src = threeD
                        ? now.Slice((((b * newSeq) + t) * g.KvHeads * size) + (h * size), size)
                        : now.Slice(((((b * g.KvHeads) + h) * newSeq) + t) * size, size);
                    src.CopyTo(d.Slice((pastSeq + t) * size, size));
                }
            }
        }
    }

    private Geometry ResolveRun(Shape q, Shape k, Shape v, Shape pastK, bool hasPast)
    {
        var g = new Geometry { ThreeD = q.Rank == 3, Batch = q[0] };
        if (g.ThreeD)
        {
            g.QHeads = _qHeads;
            g.KvHeads = _kvHeads;
            g.QSeq = q[1];
            g.KvSeq = k[1];
            g.HeadSize = q[2] / g.QHeads;
            g.VHeadSize = v[2] / g.KvHeads;
        }
        else
        {
            g.QHeads = q[1];
            g.KvHeads = k[1];
            g.QSeq = q[2];
            g.KvSeq = k[2];
            g.HeadSize = q[3];
            g.VHeadSize = v[3];
        }

        g.PastSeq = hasPast ? pastK[2] : 0;
        return g;
    }
}
