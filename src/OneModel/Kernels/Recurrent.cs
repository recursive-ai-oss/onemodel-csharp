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

    // Scratch, sized by RecurrentKernel.ScratchFloats and sliced by the caller. A step reads the hidden state from Hbuf and writes the
    // new one to Hnext (the two swap every step), so the hidden-unit chunks of a step never read what another chunk is writing.
    public Span<float> Xw, Rec, Hbuf, Cbuf, Tmp, Tmp2, Tmp3, Hnext;
}

/// <summary>
/// Recurrent layers for ONNX's RNN, GRU and LSTM, after the operator definitions in <c>Operators.md</c>. The input projection of all
/// time steps is one matrix product on the shared GEMM; each step then does the small recurrent product and the gate arithmetic.
/// The order of operations is fixed (docs/ops-recurrent.md), uses no fused multiply-add outside the GEMM and the activation
/// polynomials, and so is bit-identical across machines in exact mode.
/// </summary>
/// <remarks>
/// A step is split into chunks of <see cref="ChunkWidth"/> hidden units (all gates of those units, both directions of a bidirectional
/// layer side by side). A chunk forms its columns of the recurrent product and runs the gate arithmetic for them, so it needs
/// nothing from the other chunks until the next step; the chunk boundaries come from the hidden size alone, never from the thread
/// count, and every output element is computed by one chunk in one fixed order.
/// </remarks>
internal static class RecurrentKernel
{
    /// <summary>Hidden units per chunk of a step: two AVX-512 vectors per gate, four AVX2 ones. A constant of the kernel.</summary>
    public const int ChunkWidth = 32;

    /// <summary>The most rows of a step product given to the streaming kernel; larger batches use the blocked GEMM on the chunk's columns.</summary>
    private const int MaxGemvRows = 16;

    /// <summary>Gates per cell (the width multiplier of W, R and B).</summary>
    public static int Gates(RecurrentCell cell) => (int)cell;

    /// <summary>The scratch floats one direction needs: Xw, Rec, Hbuf, Cbuf, Tmp, Tmp2, Tmp3, Hnext, in that order.</summary>
    public static long ScratchFloats(RecurrentCell cell, int seq, int batch, int hidden)
    {
        long gh = (long)Gates(cell) * hidden;
        return ((long)seq * batch * gh) + ((long)batch * gh) + (2L * batch * hidden) + hidden + (3L * batch * hidden);
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
        d.Tmp3 = scratch.Slice(o, bh); o += bh;
        d.Hnext = scratch.Slice(o, bh);
    }

    /// <summary>Runs one direction over all time steps.</summary>
    public static void Run(ref RecurrentDirection d) => Run(ref d, ref d, 1);

    /// <summary>
    /// Runs one direction (<paramref name="dirs"/> = 1) or both directions of a layer (<paramref name="dirs"/> = 2, each with its own
    /// scratch) over all time steps. Both directions step together so one step's chunks cover both.
    /// </summary>
    public static void Run(ref RecurrentDirection d0, ref RecurrentDirection d1, int dirs)
    {
        int maxLen = d0.Seq;
        if (!d0.Lens.IsEmpty)
        {
            maxLen = 0;
            foreach (int l in d0.Lens)
            {
                maxLen = Math.Max(maxLen, l);
            }
        }

        for (int q = 0; q < dirs; q++)
        {
            ref RecurrentDirection d = ref (q == 0 ? ref d0 : ref d1);
            InitState(ref d);
            if (maxLen > 0 && d.Batch > 0)
            {
                // Input projection for every time step at once: Xw[row] = X[row] * W^T  (Goto & van de Geijn: one large, well-shaped product).
                int gh = Gates(d.Cell) * d.Hidden;
                Gemm.Run(d.Exact ? KernelMode.Exact : KernelMode.Fast, d.Path, d.Seq * d.Batch, gh, d.Input, d.X, d.Input, d.WT, gh, d.Xw, gh, d.Threads);
            }
        }

        bool finalInNext = RecurrentSteps.Run(ref d0, ref d1, dirs, maxLen);

        for (int q = 0; q < dirs; q++)
        {
            ref RecurrentDirection d = ref (q == 0 ? ref d0 : ref d1);
            WriteFinal(ref d, finalInNext);
        }
    }

    private static void InitState(ref RecurrentDirection d)
    {
        int hid = d.Hidden;

        // State starts from the given initial values, or zeros.
        for (int b = 0; b < d.Batch; b++)
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
    }

    private static void WriteFinal(ref RecurrentDirection d, bool inNext)
    {
        int batch = d.Batch, hid = d.Hidden;
        bool exact = d.Exact;
        Span<float> h = inNext ? d.Hnext : d.Hbuf;
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
                    Store(dst, h.Slice(b * hid, hid), exact);
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

    /// <summary>
    /// One phase of one step for the hidden units <c>[j0, j0 + n)</c> of one direction. Phase 0 forms the recurrent product for those
    /// units and, except for a GRU with <c>linear_before_reset = 0</c> (<paramref name="gruSplit"/>), finishes the step. In the split
    /// case phase 0 stops after the update and reset gates, and phase 1 (which runs after every chunk has finished phase 0, because
    /// it needs <c>r ⊙ H</c> for all hidden units) forms the candidate product and finishes the step.
    /// </summary>
    internal static void StepChunk(ref RecurrentDirection d, int phase, int s, int j0, int n, bool gruSplit)
    {
        int batch = d.Batch, hid = d.Hidden;
        int gh = Gates(d.Cell) * hid;
        VecIsa isa = Isa(d.Path);
        bool exact = d.Exact;

        // The recurrent product for the whole batch, these columns only: Rec = H * R^T. For a split GRU only the z and r columns can
        // be formed now; the h column needs r first.
        bool finishes;
        if (phase == 0)
        {
            int gates = gruSplit ? 2 : Gates(d.Cell);
            RangeProduct(ref d, batch, n, hid, d.Hbuf, d.RT.Slice(j0), gh, d.Rec.Slice(j0), gh, gates, hid);

            finishes = !gruSplit;
        }
        else
        {
            // (r (.) H) * Rh^T, where Rh^T is the last hid columns of R^T.
            RangeProduct(ref d, batch, n, hid, d.Tmp2, d.RT.Slice((2 * hid) + j0), gh, d.Tmp3.Slice(j0), hid, 1, 0);
            finishes = true;
        }

        for (int b = 0; b < batch; b++)
        {
            if (!Active(d, b, s))
            {
                if (finishes)
                {
                    // A row past its length keeps its state.
                    d.Hbuf.Slice((b * hid) + j0, n).CopyTo(d.Hnext.Slice((b * hid) + j0, n));
                }

                continue;
            }

            int t = Time(d, b, s);
            if (phase == 1)
            {
                GruStep(ref d, b, t, gh, j0, n, isa, exact);
            }
            else if (gruSplit)
            {
                GruGatesZr(ref d, b, t, gh, j0, n, isa, exact);
            }
            else
            {
                switch (d.Cell)
                {
                    case RecurrentCell.Lstm: LstmStep(ref d, b, t, gh, j0, n, isa, exact); break;
                    case RecurrentCell.Gru: GruStep(ref d, b, t, gh, j0, n, isa, exact); break;
                    default: RnnStep(ref d, b, t, gh, j0, n, isa, exact); break;
                }
            }
        }
    }

    // C[m, n] = A[m, k] * B[k, n] for the few rows of a recurrent step, for `groups` column blocks of width n starting `groupStride`
    // columns apart (the gates): a streaming product for small batches, the blocked GEMM above.
    private static void RangeProduct(ref RecurrentDirection d, int m, int n, int k, ReadOnlySpan<float> a, ReadOnlySpan<float> b, int ldb,
        Span<float> c, int ldc, int groups, int groupStride)
    {
        if (m <= MaxGemvRows)
        {
            RecurrentGemv.Run(d.Exact, Isa(d.Path), m, n, k, a, k, b, ldb, c, ldc, groups, groupStride);
            return;
        }

        // One thread: the chunks of the step are already spread over the threads.
        for (int g = 0; g < groups; g++)
        {
            Gemm.Run(d.Exact ? KernelMode.Exact : KernelMode.Fast, d.Path, m, n, k, a, k, b.Slice(g * groupStride), ldb, c.Slice(g * groupStride), ldc, 1);
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

    // pre = (((xw + rec) + wb) + rb): the canonical order for a pre-activation, for the columns [offset, offset + count) of the row.
    private static void PreActivation(VecIsa isa, ReadOnlySpan<float> xw, Span<float> row, ReadOnlySpan<float> wb, ReadOnlySpan<float> rb, int offset, int count)
    {
        RecurrentVec.PreActivation(isa, xw.Slice(offset, count), row.Slice(offset, count), wb.IsEmpty ? default : wb.Slice(offset, count),
            rb.IsEmpty ? default : rb.Slice(offset, count));
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

    // Y for the units [j0, j0 + n) of batch row b at time t, from the new hidden state.
    private static void WriteY(ref RecurrentDirection d, int b, int t, int j0, int n, bool exact)
    {
        if (!d.Y.IsEmpty)
        {
            int hid = d.Hidden;
            Store(d.Y.Slice(d.YOffset + (t * d.YTimeStep) + (b * d.YBatchStep) + j0, n), d.Hnext.Slice((b * hid) + j0, n), exact);
        }
    }

    private static ReadOnlySpan<float> XwRow(in RecurrentDirection d, int b, int t, int gh) =>
        d.Xw.Slice(((t * d.XTimeStep) + (b * d.XBatchStep)) * gh, gh);

    // LSTM, Operators.md: it = f(Xt Wi + Ht-1 Ri + Pi (.) Ct-1 + Wbi + Rbi), ft likewise with Pf, ct = g(...), Ct = ft (.) Ct-1 + it (.) ct,
    // ot = f(Xt Wo + Ht-1 Ro + Po (.) Ct + Wbo + Rbo), Ht = ot (.) h(Ct). Gate order in the weights is i, o, f, c.
    private static void LstmStep(ref RecurrentDirection d, int b, int t, int gh, int j0, int n, VecIsa isa, bool exact)
    {
        int hid = d.Hidden;
        Span<float> row = d.Rec.Slice(b * gh, gh);
        ReadOnlySpan<float> xw = XwRow(d, b, t, gh);
        Span<float> h = d.Hnext.Slice((b * hid) + j0, n);
        Span<float> c = d.Cbuf.Slice((b * hid) + j0, n);
        Span<float> tmp = d.Tmp.Slice(j0, n);
        for (int g = 0; g < 4; g++)
        {
            PreActivation(isa, xw, row, d.Wb, d.Rb, (g * hid) + j0, n);
        }

        Span<float> gi = row.Slice(j0, n);
        Span<float> go = row.Slice(hid + j0, n);
        Span<float> gf = row.Slice((2 * hid) + j0, n);
        Span<float> gc = row.Slice((3 * hid) + j0, n);
        if (!d.Peep.IsEmpty)
        {
            for (int j = 0; j < n; j++)
            {
                float pi = d.Peep[j0 + j] * c[j];
                gi[j] += pi;
                float pf = d.Peep[(2 * hid) + j0 + j] * c[j];
                gf[j] += pf;
            }
        }

        ClipSpan(gi, d.Clip);
        ClipSpan(gf, d.Clip);
        ClipSpan(gc, d.Clip);
        d.F.Apply(gi, tmp, exact, isa);
        if (!d.InputForget)
        {
            d.F.Apply(gf, tmp, exact, isa);
        }

        d.G.Apply(gc, tmp, exact, isa);

        if (d.InputForget)
        {
            for (int j = 0; j < n; j++)
            {
                // With input_forget the forget gate is coupled to the input gate: ft = 1 - it.
                float fg = 1f - gi[j];
                float keep = fg * c[j];
                float add = gi[j] * gc[j];
                c[j] = keep + add;
            }
        }
        else
        {
            RecurrentVec.CellUpdate(isa, c, gf, gi, gc);
        }

        if (!d.Peep.IsEmpty)
        {
            for (int j = 0; j < n; j++)
            {
                float po = d.Peep[hid + j0 + j] * c[j];
                go[j] += po;
            }
        }

        ClipSpan(go, d.Clip);
        d.F.Apply(go, tmp, exact, isa);
        Span<float> tc = d.Tmp3.Slice((b * hid) + j0, n);
        c.CopyTo(tc);
        d.Hact.Apply(tc, tmp, exact, isa);
        RecurrentVec.Multiply(isa, h, go, tc);

        WriteY(ref d, b, t, j0, n, exact);
    }

    // GRU, first part: zt and rt, and r (.) Ht-1 for the product with Rh when linear_before_reset = 0.
    private static void GruGatesZr(ref RecurrentDirection d, int b, int t, int gh, int j0, int n, VecIsa isa, bool exact)
    {
        int hid = d.Hidden;
        Span<float> row = d.Rec.Slice(b * gh, gh);
        ReadOnlySpan<float> xw = XwRow(d, b, t, gh);
        PreActivation(isa, xw, row, d.Wb, d.Rb, j0, n);
        PreActivation(isa, xw, row, d.Wb, d.Rb, hid + j0, n);
        Span<float> gz = row.Slice(j0, n);
        Span<float> gr = row.Slice(hid + j0, n);
        Span<float> tmp = d.Tmp.Slice(j0, n);
        ClipSpan(gz, d.Clip);
        ClipSpan(gr, d.Clip);
        d.F.Apply(gz, tmp, exact, isa);
        d.F.Apply(gr, tmp, exact, isa);
        Span<float> rh = d.Tmp2.Slice((b * hid) + j0, n);
        ReadOnlySpan<float> h = d.Hbuf.Slice((b * hid) + j0, n);
        for (int j = 0; j < n; j++)
        {
            rh[j] = gr[j] * h[j];
        }
    }

    // GRU, Operators.md: zt = f(Xt Wz + Ht-1 Rz + Wbz + Rbz), rt = f(... r ...),
    //   linear_before_reset = 0: ht = g(Xt Wh + (rt (.) Ht-1) Rh + Rbh + Wbh)
    //   linear_before_reset = 1: ht = g(Xt Wh + rt (.) (Ht-1 Rh + Rbh) + Wbh)
    //   Ht = (1 - zt) (.) ht + zt (.) Ht-1.   Gate order in the weights is z, r, h.
    private static void GruStep(ref RecurrentDirection d, int b, int t, int gh, int j0, int n, VecIsa isa, bool exact)
    {
        int hid = d.Hidden;
        Span<float> row = d.Rec.Slice(b * gh, gh);
        ReadOnlySpan<float> xw = XwRow(d, b, t, gh);
        ReadOnlySpan<float> hp = d.Hbuf.Slice((b * hid) + j0, n);
        Span<float> h = d.Hnext.Slice((b * hid) + j0, n);
        Span<float> tmp = d.Tmp.Slice(j0, n);
        Span<float> gz = row.Slice(j0, n);
        Span<float> cand;
        if (d.LinearBeforeReset)
        {
            PreActivation(isa, xw, row, d.Wb, default, j0, n);
            PreActivation(isa, xw, row, d.Wb, default, hid + j0, n);
            if (!d.Rb.IsEmpty)
            {
                // Rb for z and r joins the sum in the canonical place, after Wb.
                for (int j = 0; j < n; j++)
                {
                    row[j0 + j] += d.Rb[j0 + j];
                    row[hid + j0 + j] += d.Rb[hid + j0 + j];
                }
            }

            Span<float> gr = row.Slice(hid + j0, n);
            ClipSpan(gz, d.Clip);
            ClipSpan(gr, d.Clip);
            d.F.Apply(gz, tmp, exact, isa);
            d.F.Apply(gr, tmp, exact, isa);
            cand = row.Slice((2 * hid) + j0, n);
            for (int j = 0; j < n; j++)
            {
                float rec = cand[j];
                if (!d.Rb.IsEmpty)
                {
                    rec += d.Rb[(2 * hid) + j0 + j];
                }

                float scaled = gr[j] * rec;
                float v = xw[(2 * hid) + j0 + j] + scaled;
                if (!d.Wb.IsEmpty)
                {
                    v += d.Wb[(2 * hid) + j0 + j];
                }

                cand[j] = v;
            }
        }
        else
        {
            cand = d.Tmp3.Slice((b * hid) + j0, n);
            ReadOnlySpan<float> xh = xw.Slice((2 * hid) + j0, n);
            for (int j = 0; j < n; j++)
            {
                float v = xh[j] + cand[j];
                if (!d.Wb.IsEmpty)
                {
                    v += d.Wb[(2 * hid) + j0 + j];
                }

                if (!d.Rb.IsEmpty)
                {
                    v += d.Rb[(2 * hid) + j0 + j];
                }

                cand[j] = v;
            }
        }

        ClipSpan(cand, d.Clip);
        d.G.Apply(cand, tmp, exact, isa);
        for (int j = 0; j < n; j++)
        {
            float z = gz[j];
            float fresh = (1f - z) * cand[j];
            float old = z * hp[j];
            h[j] = fresh + old;
        }

        WriteY(ref d, b, t, j0, n, exact);
    }

    // RNN, Operators.md: Ht = f(Xt Wi + Ht-1 Ri + Wbi + Rbi).
    private static void RnnStep(ref RecurrentDirection d, int b, int t, int gh, int j0, int n, VecIsa isa, bool exact)
    {
        int hid = d.Hidden;
        Span<float> row = d.Rec.Slice((b * gh) + j0, n);
        PreActivation(isa, XwRow(d, b, t, gh), d.Rec.Slice(b * gh, gh), d.Wb, d.Rb, j0, n);
        ClipSpan(row, d.Clip);
        d.F.Apply(row, d.Tmp.Length >= hid ? d.Tmp.Slice(j0, n) : default, exact, isa);
        row.CopyTo(d.Hnext.Slice((b * hid) + j0, n));
        WriteY(ref d, b, t, j0, n, exact);
    }
}
