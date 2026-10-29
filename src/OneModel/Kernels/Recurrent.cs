using System;
using RecursiveAI.OneModel.Kernels.Exact;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>Which recurrent cell a kernel call runs.</summary>
internal enum RecurrentCell
{
    /// <summary>Plain RNN: one gate.</summary>
    Rnn = 1,

    /// <summary>GRU (Cho et al. 2014): gates z, r, h in that order.</summary>
    Gru = 3,

    /// <summary>LSTM (Hochreiter and Schmidhuber 1997, with peepholes after Gers and Schmidhuber 2000): gates i, o, f, c in that order.</summary>
    Lstm = 4,
}

/// <summary>The activation functions ONNX lets a recurrent operator name.</summary>
internal enum RecurrentActivationKind
{
    Sigmoid,
    Tanh,
    Relu,
    Affine,
    LeakyRelu,
    ThresholdedRelu,
    ScaledTanh,
    HardSigmoid,
    Elu,
    Softsign,
    Softplus,
}

/// <summary>One activation with its (already defaulted) alpha and beta.</summary>
internal readonly struct RecurrentActivation
{
    public RecurrentActivation(RecurrentActivationKind kind, float alpha, float beta)
    {
        Kind = kind;
        Alpha = alpha;
        Beta = beta;
    }

    public RecurrentActivationKind Kind { get; }

    public float Alpha { get; }

    public float Beta { get; }

    /// <summary>
    /// Applies the activation in place to <paramref name="x"/>. <paramref name="tmp"/> must be at least as long. Sigmoid, tanh, exp and
    /// log come from the kernels' own polynomials, so exact mode is bit-identical on every machine; everything else is IEEE
    /// add/multiply/divide, compares and selects.
    /// </summary>
    public void Apply(Span<float> x, Span<float> tmp, bool exact, VecIsa isa)
    {
        switch (Kind)
        {
            case RecurrentActivationKind.Sigmoid:
                ElementwiseMath.Apply(ElementwiseFn.Sigmoid, exact, isa, x, x);
                break;
            case RecurrentActivationKind.Tanh:
                ElementwiseMath.Apply(ElementwiseFn.Tanh, exact, isa, x, x);
                break;
            case RecurrentActivationKind.Relu:
                for (int i = 0; i < x.Length; i++)
                {
                    float v = x[i];
                    x[i] = v > 0f || float.IsNaN(v) ? v : 0f;
                }

                break;
            case RecurrentActivationKind.Affine:
                for (int i = 0; i < x.Length; i++)
                {
                    float t = Alpha * x[i];
                    x[i] = t + Beta;
                }

                break;
            case RecurrentActivationKind.LeakyRelu:
                for (int i = 0; i < x.Length; i++)
                {
                    float v = x[i];
                    x[i] = v < 0f ? Alpha * v : v;
                }

                break;
            case RecurrentActivationKind.ThresholdedRelu:
                for (int i = 0; i < x.Length; i++)
                {
                    float v = x[i];
                    x[i] = v > Alpha || float.IsNaN(v) ? v : 0f;
                }

                break;
            case RecurrentActivationKind.ScaledTanh:
                for (int i = 0; i < x.Length; i++)
                {
                    x[i] = Beta * x[i];
                }

                ElementwiseMath.Apply(ElementwiseFn.Tanh, exact, isa, x, x);
                for (int i = 0; i < x.Length; i++)
                {
                    x[i] = Alpha * x[i];
                }

                break;
            case RecurrentActivationKind.HardSigmoid:
                for (int i = 0; i < x.Length; i++)
                {
                    float t = Alpha * x[i];
                    float v = t + Beta;
                    x[i] = v > 1f ? 1f : (v < 0f ? 0f : v);
                }

                break;
            case RecurrentActivationKind.Elu:
                ElementwiseMath.Apply(ElementwiseFn.Exp, exact, isa, x, tmp);
                for (int i = 0; i < x.Length; i++)
                {
                    float v = x[i];
                    x[i] = v >= 0f || float.IsNaN(v) ? v : Alpha * (tmp[i] - 1f);
                }

                break;
            case RecurrentActivationKind.Softsign:
                for (int i = 0; i < x.Length; i++)
                {
                    float v = x[i];
                    x[i] = v / (1f + MathF.Abs(v));
                }

                break;
            case RecurrentActivationKind.Softplus:
                ElementwiseMath.Apply(ElementwiseFn.Exp, exact, isa, x, tmp);
                for (int i = 0; i < x.Length; i++)
                {
                    tmp[i] += 1f;
                }

                ElementwiseMath.Apply(ElementwiseFn.Log, exact, isa, tmp, tmp);

                // Above 20, log(1 + e^x) equals x to fp32 precision, and e^x would overflow soon after.
                for (int i = 0; i < x.Length; i++)
                {
                    float v = x[i];
                    x[i] = v > 20f || float.IsNaN(v) ? v : tmp[i];
                }

                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(x), Kind, "Unknown activation.");
        }
    }
}

/// <summary>Everything one direction of a recurrent layer needs. Spans are views; nothing is copied.</summary>
internal ref struct RecurrentDirection
{
    public RecurrentCell Cell;
    public bool Exact;
    public KernelPath Path;
    public int Threads;

    public int Seq, Batch, Input, Hidden;
    public bool Reverse;

    // X as [rows, Input] with row(t, b) = t * XTimeStep + b * XBatchStep.
    public ReadOnlySpan<float> X;
    public int XTimeStep, XBatchStep;

    // W^T as [Input, G*Hidden] and R^T as [Hidden, G*Hidden], this direction only.
    public ReadOnlySpan<float> WT, RT;

    // Biases, G*Hidden each, or empty. Peepholes: 3*Hidden in order i, o, f, or empty (LSTM only).
    public ReadOnlySpan<float> Wb, Rb, Peep;

    // Valid length per batch row, or empty for all Seq steps.
    public ReadOnlySpan<int> Lens;

    // Initial state as Batch rows of Hidden starting at the span's origin, row b at b * StateBatchStride; empty means zeros.
    public ReadOnlySpan<float> H0, C0;

    // Outputs; any may be empty. Y step (t, b) lives at YOffset + t * YTimeStep + b * YBatchStep; state row b at StateOffset + b * StateBatchStride.
    public Span<float> Y, YH, YC;
    public int YOffset, YTimeStep, YBatchStep, StateOffset, StateBatchStride;

    public RecurrentActivation F, G, Hact;
    public float Clip; // 0 = off
    public bool InputForget, LinearBeforeReset;

    // Scratch, sized by RecurrentKernel.ScratchFloats and sliced by the caller.
    public Span<float> Xw, Rec, Hbuf, Cbuf, Tmp, Tmp2, Tmp3;
}

/// <summary>
/// Recurrent layers for ONNX's RNN, GRU and LSTM, after the operator definitions in <c>Operators.md</c>. The input projection of all
/// time steps is one matrix product on the shared GEMM; each step then does the small recurrent product and the gate arithmetic.
/// The order of operations is fixed (docs/ops-recurrent.md), uses no fused multiply-add outside the GEMM and the activation
/// polynomials, and so is bit-identical across machines in exact mode.
/// </summary>
internal static class RecurrentKernel
{
    /// <summary>Gates per cell (the width multiplier of W, R and B).</summary>
    public static int Gates(RecurrentCell cell) => (int)cell;

    /// <summary>The scratch floats one direction needs: Xw, Rec, Hbuf, Cbuf, Tmp, Tmp2, Tmp3, in that order.</summary>
    public static long ScratchFloats(RecurrentCell cell, int seq, int batch, int hidden)
    {
        long gh = (long)Gates(cell) * hidden;
        return ((long)seq * batch * gh) + ((long)batch * gh) + (2L * batch * hidden) + hidden + (2L * batch * hidden);
    }

    /// <summary>Maps the session's kernel path to the lane width of the elementwise functions.</summary>
    public static VecIsa Isa(KernelPath path) => path switch
    {
        KernelPath.Avx512 => VecIsa.V512,
        KernelPath.Avx2 => VecIsa.V256,
        KernelPath.Neon or KernelPath.Portable128 => VecIsa.V128,
        _ => VecIsa.Scalar,
    };

    /// <summary>Slices one flat scratch buffer into the fields the direction needs.</summary>
    public static void Bind(ref RecurrentDirection d, Span<float> scratch)
    {
        long gh = (long)Gates(d.Cell) * d.Hidden;
        int xw = checked((int)((long)d.Seq * d.Batch * gh));
        int rec = checked((int)((long)d.Batch * gh));
        int bh = d.Batch * d.Hidden;
        int o = 0;
        d.Xw = scratch.Slice(o, xw); o += xw;
        d.Rec = scratch.Slice(o, rec); o += rec;
        d.Hbuf = scratch.Slice(o, bh); o += bh;
        d.Cbuf = scratch.Slice(o, bh); o += bh;
        d.Tmp = scratch.Slice(o, d.Hidden); o += d.Hidden;
        d.Tmp2 = scratch.Slice(o, bh); o += bh;
        d.Tmp3 = scratch.Slice(o, bh);
    }

    /// <summary>Runs one direction over all time steps.</summary>
    public static void Run(ref RecurrentDirection d)
    {
        int seq = d.Seq, batch = d.Batch, hid = d.Hidden;
        int g = Gates(d.Cell);
        int gh = g * hid;
        KernelMode mode = d.Exact ? KernelMode.Exact : KernelMode.Fast;
        VecIsa isa = Isa(d.Path);
        bool exact = d.Exact;

        // State starts from the given initial values, or zeros.
        for (int b = 0; b < batch; b++)
        {
            Span<float> hrow = d.Hbuf.Slice(b * hid, hid);
            Span<float> crow = d.Cbuf.Slice(b * hid, hid);
            if (d.H0.IsEmpty)
            {
                hrow.Clear();
            }
            else
            {
                d.H0.Slice(b * d.StateBatchStride, hid).CopyTo(hrow);
            }

            if (d.C0.IsEmpty)
            {
                crow.Clear();
            }
            else
            {
                d.C0.Slice(b * d.StateBatchStride, hid).CopyTo(crow);
            }
        }

        int maxLen = seq;
        if (!d.Lens.IsEmpty)
        {
            maxLen = 0;
            foreach (int l in d.Lens)
            {
                maxLen = Math.Max(maxLen, l);
            }
        }

        if (maxLen > 0 && batch > 0)
        {
            // Input projection for every time step at once: Xw[row] = X[row] * W^T  (Goto & van de Geijn: one large, well-shaped product).
            Gemm.Run(mode, d.Path, seq * batch, gh, d.Input, d.X, d.Input, d.WT, gh, d.Xw, gh, d.Threads);
        }

        // The per-step products have only `batch` rows; handing them to many threads costs more than it saves for small batches.
        // Results do not depend on the thread count, so this changes speed only.
        int stepThreads = batch >= 16 ? d.Threads : 1;
        bool gruSplit = d.Cell == RecurrentCell.Gru && !d.LinearBeforeReset;
        for (int s = 0; s < maxLen; s++)
        {
            // The recurrent product for the whole batch: Rec = H * R^T. For GRU with linear_before_reset = 0 only the z and r
            // columns can be formed now; the h column needs r first.
            StepProduct(ref d, batch, gruSplit ? 2 * hid : gh, hid, d.Hbuf, d.RT, gh, d.Rec, gh, stepThreads);

            if (gruSplit)
            {
                for (int b = 0; b < batch; b++)
                {
                    if (!Active(d, b, s))
                    {
                        continue;
                    }

                    int t = Time(d, b, s);
                    GruGatesZr(ref d, b, t, gh, isa, exact);
                }

                // (r (.) H) * Rh^T, where Rh^T is the last hid columns of R^T.
                StepProduct(ref d, batch, hid, hid, d.Tmp2, d.RT.Slice(2 * hid), gh, d.Tmp3, hid, stepThreads);
            }

            for (int b = 0; b < batch; b++)
            {
                if (!Active(d, b, s))
                {
                    continue;
                }

                int t = Time(d, b, s);
                switch (d.Cell)
                {
                    case RecurrentCell.Lstm: LstmStep(ref d, b, t, gh, isa, exact); break;
                    case RecurrentCell.Gru: GruStep(ref d, b, t, gh, isa, exact); break;
                    default: RnnStep(ref d, b, t, gh, isa, exact); break;
                }
            }
        }

        if (!d.YH.IsEmpty)
        {
            for (int b = 0; b < batch; b++)
            {
                Span<float> dst = d.YH.Slice(d.StateOffset + (b * d.StateBatchStride), hid);
                if (!Active(d, b, 0))
                {
                    // A row of length 0 never ran; its final state is defined as zeros (docs/ops-recurrent.md).
                    dst.Clear();
                }
                else
                {
                    Store(dst, d.Hbuf.Slice(b * hid, hid), exact);
                }
            }
        }

        if (!d.YC.IsEmpty && d.Cell == RecurrentCell.Lstm)
        {
            for (int b = 0; b < batch; b++)
            {
                Span<float> dst = d.YC.Slice(d.StateOffset + (b * d.StateBatchStride), hid);
                if (!Active(d, b, 0))
                {
                    dst.Clear();
                }
                else
                {
                    Store(dst, d.Cbuf.Slice(b * hid, hid), exact);
                }
            }
        }
    }

    // C[m, n] = A[m, k] * B[k, n] for the few rows of a recurrent step: a streaming product for small batches, the blocked GEMM above.
    private static void StepProduct(ref RecurrentDirection d, int m, int n, int k, ReadOnlySpan<float> a, ReadOnlySpan<float> b, int ldb,
        Span<float> c, int ldc, int threads)
    {
        if (m <= RecurrentGemv.MaxRows)
        {
            RecurrentGemv.Run(d.Exact, Isa(d.Path), m, n, k, a, k, b, ldb, c, ldc);
        }
        else
        {
            Gemm.Run(d.Exact ? KernelMode.Exact : KernelMode.Fast, d.Path, m, n, k, a, k, b, ldb, c, ldc, threads);
        }
    }

    private static bool Active(in RecurrentDirection d, int b, int s) => d.Lens.IsEmpty || s < d.Lens[b];

    // Step s of a batch row maps to time s going forward, or len-1-s going backward over the row's own valid range.
    private static int Time(in RecurrentDirection d, int b, int s)
    {
        if (!d.Reverse)
        {
            return s;
        }

        int len = d.Lens.IsEmpty ? d.Seq : d.Lens[b];
        return len - 1 - s;
    }

    // pre = (((xw + rec) + wb) + rb): the canonical order for a pre-activation.
    private static void PreActivation(ReadOnlySpan<float> xw, Span<float> row, ReadOnlySpan<float> wb, ReadOnlySpan<float> rb, int offset, int count)
    {
        for (int j = offset; j < offset + count; j++)
        {
            float v = xw[j] + row[j];
            if (!wb.IsEmpty)
            {
                v += wb[j];
            }

            if (!rb.IsEmpty)
            {
                v += rb[j];
            }

            row[j] = v;
        }
    }

    private static void ClipSpan(Span<float> x, float clip)
    {
        if (clip <= 0f)
        {
            return;
        }

        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            x[i] = v > clip ? clip : (v < -clip ? -clip : v);
        }
    }

    private static void Store(Span<float> dst, ReadOnlySpan<float> src, bool exact)
    {
        if (!exact)
        {
            src.CopyTo(dst);
            return;
        }

        for (int i = 0; i < src.Length; i++)
        {
            float v = src[i];
            dst[i] = float.IsNaN(v) ? ExactElementwise.CanonicalNaN : v;
        }
    }

    private static void WriteY(ref RecurrentDirection d, int b, int t, bool exact)
    {
        if (!d.Y.IsEmpty)
        {
            int hid = d.Hidden;
            Store(d.Y.Slice(d.YOffset + (t * d.YTimeStep) + (b * d.YBatchStep), hid), d.Hbuf.Slice(b * hid, hid), exact);
        }
    }

    private static ReadOnlySpan<float> XwRow(in RecurrentDirection d, int b, int t, int gh) =>
        d.Xw.Slice(((t * d.XTimeStep) + (b * d.XBatchStep)) * gh, gh);

    // LSTM, Operators.md: it = f(Xt Wi + Ht-1 Ri + Pi (.) Ct-1 + Wbi + Rbi), ft likewise with Pf, ct = g(...), Ct = ft (.) Ct-1 + it (.) ct,
    // ot = f(Xt Wo + Ht-1 Ro + Po (.) Ct + Wbo + Rbo), Ht = ot (.) h(Ct). Gate order in the weights is i, o, f, c.
    private static void LstmStep(ref RecurrentDirection d, int b, int t, int gh, VecIsa isa, bool exact)
    {
        int hid = d.Hidden;
        Span<float> row = d.Rec.Slice(b * gh, gh);
        ReadOnlySpan<float> xw = XwRow(d, b, t, gh);
        Span<float> h = d.Hbuf.Slice(b * hid, hid);
        Span<float> c = d.Cbuf.Slice(b * hid, hid);
        PreActivation(xw, row, d.Wb, d.Rb, 0, gh);

        Span<float> gi = row.Slice(0, hid);
        Span<float> go = row.Slice(hid, hid);
        Span<float> gf = row.Slice(2 * hid, hid);
        Span<float> gc = row.Slice(3 * hid, hid);
        if (!d.Peep.IsEmpty)
        {
            for (int j = 0; j < hid; j++)
            {
                float pi = d.Peep[j] * c[j];
                gi[j] += pi;
                float pf = d.Peep[(2 * hid) + j] * c[j];
                gf[j] += pf;
            }
        }

        ClipSpan(gi, d.Clip);
        ClipSpan(gf, d.Clip);
        ClipSpan(gc, d.Clip);
        d.F.Apply(gi, d.Tmp, exact, isa);
        if (!d.InputForget)
        {
            d.F.Apply(gf, d.Tmp, exact, isa);
        }

        d.G.Apply(gc, d.Tmp, exact, isa);

        for (int j = 0; j < hid; j++)
        {
            // With input_forget the forget gate is coupled to the input gate: ft = 1 - it.
            float fg = d.InputForget ? 1f - gi[j] : gf[j];
            float keep = fg * c[j];
            float add = gi[j] * gc[j];
            c[j] = keep + add;
        }

        if (!d.Peep.IsEmpty)
        {
            for (int j = 0; j < hid; j++)
            {
                float po = d.Peep[hid + j] * c[j];
                go[j] += po;
            }
        }

        ClipSpan(go, d.Clip);
        d.F.Apply(go, d.Tmp, exact, isa);
        c.CopyTo(d.Tmp3.Slice(0, hid));
        Span<float> tc = d.Tmp3.Slice(0, hid);
        d.Hact.Apply(tc, d.Tmp, exact, isa);
        for (int j = 0; j < hid; j++)
        {
            h[j] = go[j] * tc[j];
        }

        WriteY(ref d, b, t, exact);
    }

    // GRU, first part: zt and rt, and r (.) Ht-1 for the product with Rh when linear_before_reset = 0.
    private static void GruGatesZr(ref RecurrentDirection d, int b, int t, int gh, VecIsa isa, bool exact)
    {
        int hid = d.Hidden;
        Span<float> row = d.Rec.Slice(b * gh, gh);
        ReadOnlySpan<float> xw = XwRow(d, b, t, gh);
        PreActivation(xw, row, d.Wb, d.Rb, 0, 2 * hid);
        Span<float> gz = row.Slice(0, hid);
        Span<float> gr = row.Slice(hid, hid);
        ClipSpan(gz, d.Clip);
        ClipSpan(gr, d.Clip);
        d.F.Apply(gz, d.Tmp, exact, isa);
        d.F.Apply(gr, d.Tmp, exact, isa);
        Span<float> rh = d.Tmp2.Slice(b * hid, hid);
        Span<float> h = d.Hbuf.Slice(b * hid, hid);
        for (int j = 0; j < hid; j++)
        {
            rh[j] = gr[j] * h[j];
        }
    }

    // GRU, Operators.md: zt = f(Xt Wz + Ht-1 Rz + Wbz + Rbz), rt = f(... r ...),
    //   linear_before_reset = 0: ht = g(Xt Wh + (rt (.) Ht-1) Rh + Rbh + Wbh)
    //   linear_before_reset = 1: ht = g(Xt Wh + rt (.) (Ht-1 Rh + Rbh) + Wbh)
    //   Ht = (1 - zt) (.) ht + zt (.) Ht-1.   Gate order in the weights is z, r, h.
    private static void GruStep(ref RecurrentDirection d, int b, int t, int gh, VecIsa isa, bool exact)
    {
        int hid = d.Hidden;
        Span<float> row = d.Rec.Slice(b * gh, gh);
        ReadOnlySpan<float> xw = XwRow(d, b, t, gh);
        Span<float> h = d.Hbuf.Slice(b * hid, hid);
        Span<float> cand;
        if (d.LinearBeforeReset)
        {
            PreActivation(xw, row, d.Wb, default, 0, 2 * hid);
            if (!d.Rb.IsEmpty)
            {
                // Rb for z and r joins the sum in the canonical place, after Wb.
                for (int j = 0; j < 2 * hid; j++)
                {
                    row[j] += d.Rb[j];
                }
            }

            Span<float> gz = row.Slice(0, hid);
            Span<float> gr = row.Slice(hid, hid);
            ClipSpan(gz, d.Clip);
            ClipSpan(gr, d.Clip);
            d.F.Apply(gz, d.Tmp, exact, isa);
            d.F.Apply(gr, d.Tmp, exact, isa);
            cand = row.Slice(2 * hid, hid);
            for (int j = 0; j < hid; j++)
            {
                float rec = cand[j];
                if (!d.Rb.IsEmpty)
                {
                    rec += d.Rb[(2 * hid) + j];
                }

                float scaled = gr[j] * rec;
                float v = xw[(2 * hid) + j] + scaled;
                if (!d.Wb.IsEmpty)
                {
                    v += d.Wb[(2 * hid) + j];
                }

                cand[j] = v;
            }
        }
        else
        {
            cand = d.Tmp3.Slice(b * hid, hid);
            ReadOnlySpan<float> xh = xw.Slice(2 * hid, hid);
            for (int j = 0; j < hid; j++)
            {
                float v = xh[j] + cand[j];
                if (!d.Wb.IsEmpty)
                {
                    v += d.Wb[(2 * hid) + j];
                }

                if (!d.Rb.IsEmpty)
                {
                    v += d.Rb[(2 * hid) + j];
                }

                cand[j] = v;
            }
        }

        ClipSpan(cand, d.Clip);
        d.G.Apply(cand, d.Tmp, exact, isa);
        for (int j = 0; j < hid; j++)
        {
            float z = row[j];
            float fresh = (1f - z) * cand[j];
            float old = z * h[j];
            h[j] = fresh + old;
        }

        WriteY(ref d, b, t, exact);
    }

    // RNN, Operators.md: Ht = f(Xt Wi + Ht-1 Ri + Wbi + Rbi).
    private static void RnnStep(ref RecurrentDirection d, int b, int t, int gh, VecIsa isa, bool exact)
    {
        int hid = d.Hidden;
        Span<float> row = d.Rec.Slice(b * gh, gh);
        PreActivation(XwRow(d, b, t, gh), row, d.Wb, d.Rb, 0, gh);
        ClipSpan(row, d.Clip);
        d.F.Apply(row, d.Tmp.Length >= hid ? d.Tmp : default, exact, isa);
        row.CopyTo(d.Hbuf.Slice(b * hid, hid));
        WriteY(ref d, b, t, exact);
    }
}
