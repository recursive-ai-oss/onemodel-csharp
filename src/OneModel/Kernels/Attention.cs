using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using RecursiveAI.OneModel.Kernels.Exact;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Geometry of a scaled dot-product attention call: O = softmax(scale * Q K^T + mask) V for every (batch, head). Strides are in
/// elements. Within a row the head dimension is contiguous. The mask, when given, is added to the scaled scores; its columns are
/// contiguous.
/// </summary>
internal struct AttentionShape
{
    /// <summary>Batch size.</summary>
    public int Batch;

    /// <summary>Number of heads (queries, keys and values all have this many).</summary>
    public int Heads;

    /// <summary>Query positions.</summary>
    public int QLen;

    /// <summary>Key and value positions.</summary>
    public int KLen;

    /// <summary>Head dimension; must be even.</summary>
    public int HeadDim;

    /// <summary>Factor applied to Q K^T, usually 1/sqrt(HeadDim).</summary>
    public float Scale;

    /// <summary>Element strides of Q: per batch, per head, per position.</summary>
    public long QBatchStride, QHeadStride, QRowStride;

    /// <summary>Element strides of K.</summary>
    public long KBatchStride, KHeadStride, KRowStride;

    /// <summary>Element strides of V.</summary>
    public long VBatchStride, VHeadStride, VRowStride;

    /// <summary>Element strides of the output.</summary>
    public long OBatchStride, OHeadStride, ORowStride;

    /// <summary>Element strides of the additive mask (use 0 to broadcast over batch, head or rows).</summary>
    public long MaskBatchStride, MaskHeadStride, MaskRowStride;

    /// <summary>Q, K, V and O laid out as contiguous [batch, head, position, headDim] arrays; the mask (if used) as [batch, qLen, kLen] shared by all heads.</summary>
    public static AttentionShape Contiguous(int batch, int heads, int qLen, int kLen, int headDim, float scale) => new()
    {
        Batch = batch,
        Heads = heads,
        QLen = qLen,
        KLen = kLen,
        HeadDim = headDim,
        Scale = scale,
        QRowStride = headDim,
        QHeadStride = (long)qLen * headDim,
        QBatchStride = (long)heads * qLen * headDim,
        KRowStride = headDim,
        KHeadStride = (long)kLen * headDim,
        KBatchStride = (long)heads * kLen * headDim,
        VRowStride = headDim,
        VHeadStride = (long)kLen * headDim,
        VBatchStride = (long)heads * kLen * headDim,
        ORowStride = headDim,
        OHeadStride = (long)qLen * headDim,
        OBatchStride = (long)heads * qLen * headDim,
        MaskRowStride = kLen,
        MaskHeadStride = 0,
        MaskBatchStride = (long)qLen * kLen,
    };
}

/// <summary>
/// Fused scaled dot-product attention with a FlashAttention-style tiling and online softmax (Dao et al. 2022, algorithm 1):
/// queries in blocks of <see cref="QueryBlock"/> rows, keys and values in blocks of <see cref="KeyBlock"/>; per query row a running
/// maximum m and sum l, with the output accumulator rescaled by exp(m_old - m_new) when the maximum grows.
/// The exact flavour fixes every rounding step (see docs/kernels-attention.md) and is bit-identical at every lane width, and for
/// any partition of the work items over threads. The fast flavour runs the same code with fused multiply-adds and the cheaper
/// exponential.
/// </summary>
internal static class Attention
{
    /// <summary>Query rows per work item (and per packed key panel).</summary>
    public const int QueryBlock = 32;

    /// <summary>Key positions per block of the online softmax. Part of the exact definition.</summary>
    public const int KeyBlock = 64;

    /// <summary>Floats of scratch one work item needs; allocate once per worker.</summary>
    public static int ScratchLength(int headDim) =>
        (headDim * KeyBlock) + (QueryBlock * KeyBlock) + (QueryBlock * headDim) + (2 * QueryBlock) + KeyBlock + 64;

    /// <summary>Number of independent work items: (batch, head, query block) triples. Item order has no effect on any result.</summary>
    public static int ItemCount(in AttentionShape s) =>
        checked(s.Batch * s.Heads * ((s.QLen + QueryBlock - 1) / QueryBlock));

    /// <summary>Computes work items [<paramref name="itemStart"/>, <paramref name="itemEnd"/>).</summary>
    /// <param name="exact">Exact (bit-reproducible) or fast flavour.</param>
    /// <param name="isa">Lane width. Narrowed automatically until the head dimension is a multiple of twice the lane count.</param>
    /// <param name="s">Geometry.</param>
    /// <param name="q">Queries.</param>
    /// <param name="k">Keys.</param>
    /// <param name="v">Values.</param>
    /// <param name="mask">Additive mask, or empty for none.</param>
    /// <param name="o">Output; written for the rows of the processed items only.</param>
    /// <param name="itemStart">First work item.</param>
    /// <param name="itemEnd">One past the last work item.</param>
    /// <param name="scratch">At least <see cref="ScratchLength"/> floats owned by the caller's worker.</param>
    /// <exception cref="ArgumentException">A span is too short for the geometry, the head dimension is odd or non-positive, or scratch is too small.</exception>
    public static void Compute(
        bool exact,
        VecIsa isa,
        in AttentionShape s,
        ReadOnlySpan<float> q,
        ReadOnlySpan<float> k,
        ReadOnlySpan<float> v,
        ReadOnlySpan<float> mask,
        Span<float> o,
        int itemStart,
        int itemEnd,
        Span<float> scratch)
    {
        Validate(s, q.Length, k.Length, v.Length, mask.Length, o.Length, scratch.Length);
        int items = ItemCount(s);
        if (itemStart < 0 || itemEnd > items || itemStart > itemEnd)
        {
            throw new ArgumentOutOfRangeException(nameof(itemEnd), "The item range is outside [0, ItemCount].");
        }

        isa = VecIsaInfo.Resolve(isa);
        while (isa != VecIsa.Scalar && s.HeadDim % (2 * LanesOf(isa)) != 0)
        {
            isa = isa == VecIsa.V512 ? VecIsa.V256 : isa == VecIsa.V256 ? VecIsa.V128 : VecIsa.Scalar;
        }

        if (exact)
        {
            switch (isa)
            {
                case VecIsa.V512: Run<Vector512<float>, ExactVec512>(s, q, k, v, mask, o, itemStart, itemEnd, scratch); break;
                case VecIsa.V256: Run<Vector256<float>, ExactVec256>(s, q, k, v, mask, o, itemStart, itemEnd, scratch); break;
                case VecIsa.V128: Run<Vector128<float>, ExactVec128>(s, q, k, v, mask, o, itemStart, itemEnd, scratch); break;
                default: Run<float, ExactVecScalar>(s, q, k, v, mask, o, itemStart, itemEnd, scratch); break;
            }
        }
        else
        {
            switch (isa)
            {
                case VecIsa.V512: Run<Vector512<float>, FastVec512>(s, q, k, v, mask, o, itemStart, itemEnd, scratch); break;
                case VecIsa.V256: Run<Vector256<float>, FastVec256>(s, q, k, v, mask, o, itemStart, itemEnd, scratch); break;
                case VecIsa.V128: Run<Vector128<float>, FastVec128>(s, q, k, v, mask, o, itemStart, itemEnd, scratch); break;
                default: Run<float, FastVecScalar>(s, q, k, v, mask, o, itemStart, itemEnd, scratch); break;
            }
        }
    }

    private static int LanesOf(VecIsa isa) => isa switch
    {
        VecIsa.V512 => 16,
        VecIsa.V256 => 8,
        VecIsa.V128 => 4,
        _ => 1,
    };

    private static long Extent(long batchStride, long headStride, long rowStride, int batch, int heads, int rows, int cols) =>
        ((batch - 1) * batchStride) + ((heads - 1) * headStride) + ((rows - 1) * rowStride) + cols;

    private static void Validate(in AttentionShape s, int qLen, int kLen, int vLen, int maskLen, int oLen, int scratchLen)
    {
        if (s.Batch < 0 || s.Heads < 0 || s.QLen < 0 || s.KLen < 0 || s.HeadDim <= 0 || (s.HeadDim & 1) != 0 || s.HeadDim > 512)
        {
            throw new ArgumentException("Sizes must be non-negative and the head dimension even, positive and at most 512.");
        }

        if (s.Batch == 0 || s.Heads == 0 || s.QLen == 0)
        {
            return;
        }

        if (s.QRowStride < s.HeadDim || s.KRowStride < s.HeadDim || s.VRowStride < s.HeadDim || s.ORowStride < s.HeadDim
            || s.QBatchStride < 0 || s.QHeadStride < 0 || s.KBatchStride < 0 || s.KHeadStride < 0 || s.VBatchStride < 0 || s.VHeadStride < 0
            || s.OBatchStride < 0 || s.OHeadStride < 0 || s.MaskBatchStride < 0 || s.MaskHeadStride < 0 || s.MaskRowStride < 0)
        {
            throw new ArgumentException("Strides must be non-negative and row strides at least the head dimension.");
        }

        if (Extent(s.QBatchStride, s.QHeadStride, s.QRowStride, s.Batch, s.Heads, s.QLen, s.HeadDim) > qLen
            || Extent(s.OBatchStride, s.OHeadStride, s.ORowStride, s.Batch, s.Heads, s.QLen, s.HeadDim) > oLen
            || (s.KLen > 0 && (Extent(s.KBatchStride, s.KHeadStride, s.KRowStride, s.Batch, s.Heads, s.KLen, s.HeadDim) > kLen
                               || Extent(s.VBatchStride, s.VHeadStride, s.VRowStride, s.Batch, s.Heads, s.KLen, s.HeadDim) > vLen)))
        {
            throw new ArgumentException("A tensor span is shorter than the shape and strides require.");
        }

        if (maskLen > 0 && Extent(s.MaskBatchStride, s.MaskHeadStride, s.MaskRowStride, s.Batch, s.Heads, s.QLen, s.KLen) > maskLen)
        {
            throw new ArgumentException("The mask span is shorter than the shape and strides require.");
        }

        if (scratchLen < ScratchLength(s.HeadDim))
        {
            throw new ArgumentException("The scratch span is shorter than ScratchLength(headDim).");
        }
    }

    // Bounds: Validate() established that q, k, v, o and the mask cover every (batch, head, row, column) index the loops below form
    // (Extent above is the largest index + 1), and that scratch holds ScratchLength(HeadDim) floats, which is carved up as
    // kt | tile | acc | m | l | maskRow | misc below. All loop counters are clamped to the shape. Refs are formed with Unsafe.Add.
    private static void Run<TV, TO>(
        in AttentionShape s,
        ReadOnlySpan<float> q,
        ReadOnlySpan<float> k,
        ReadOnlySpan<float> v,
        ReadOnlySpan<float> mask,
        Span<float> o,
        int itemStart,
        int itemEnd,
        Span<float> scratch)
        where TV : struct
        where TO : IVecOps<TV>
    {
        const int Bc = KeyBlock;
        const int Qb = QueryBlock;
        int lanes = TO.Lanes;
        int nr = 2 * lanes;
        int dim = s.HeadDim;
        bool hasMask = !mask.IsEmpty;
        int qBlocks = (s.QLen + Qb - 1) / Qb;

        Span<float> ktS = scratch.Slice(0, dim * Bc);
        Span<float> tileS = scratch.Slice(dim * Bc, Qb * Bc);
        Span<float> accS = scratch.Slice((dim * Bc) + (Qb * Bc), Qb * dim);
        int rest = (dim * Bc) + (Qb * Bc) + (Qb * dim);
        Span<float> mS = scratch.Slice(rest, Qb);
        Span<float> lS = scratch.Slice(rest + Qb, Qb);
        Span<float> maskRowS = scratch.Slice(rest + (2 * Qb), Bc);
        Span<float> miscS = scratch.Slice(rest + (2 * Qb) + Bc, 64);

        ref float kt = ref MemoryMarshal.GetReference(ktS);
        ref float tile = ref MemoryMarshal.GetReference(tileS);
        ref float acc = ref MemoryMarshal.GetReference(accS);
        ref float maskRow = ref MemoryMarshal.GetReference(maskRowS);
        ref float misc = ref MemoryMarshal.GetReference(miscS);
        ref float qBaseRef = ref MemoryMarshal.GetReference(q);
        ref float kBaseRef = ref MemoryMarshal.GetReference(k);
        ref float vBaseRef = ref MemoryMarshal.GetReference(v);
        ref float oBaseRef = ref MemoryMarshal.GetReference(o);
        ref float mask0 = ref MemoryMarshal.GetReference(mask);

        TV scaleV = TO.Splat(s.Scale);
        TV nanV = TO.Splat(BitConverter.UInt32BitsToSingle(Softmax.QuietNaNBits));

        // Zero the tile once: rows beyond the item's row count are computed from a duplicate of the last row and never read back,
        // but they must not hold uninitialised memory.
        tileS.Clear();

        for (int item = itemStart; item < itemEnd; item++)
        {
            int bh = item / qBlocks;
            int qb = item % qBlocks;
            int b = bh / s.Heads;
            int h = bh % s.Heads;
            int rowStart = qb * Qb;
            int rows = Math.Min(Qb, s.QLen - rowStart);
            int rowsPadded = (rows + 3) & ~3;

            long qBase = (b * s.QBatchStride) + (h * s.QHeadStride);
            long kBase = (b * s.KBatchStride) + (h * s.KHeadStride);
            long vBase = (b * s.VBatchStride) + (h * s.VHeadStride);
            long oBase = (b * s.OBatchStride) + (h * s.OHeadStride);
            long maskBase = (b * s.MaskBatchStride) + (h * s.MaskHeadStride);

            accS.Slice(0, rowsPadded * dim).Clear();
            for (int r = 0; r < rows; r++)
            {
                mS[r] = float.NegativeInfinity;
                lS[r] = 0f;
            }

            for (int kb = 0; kb < s.KLen; kb += Bc)
            {
                int n = Math.Min(Bc, s.KLen - kb);
                int nPad = (n + nr - 1) / nr * nr;

                // Goto & van de Geijn 2008 §4 (packing): copy the key block transposed into a dim x Bc panel so the score
                // microkernel streams contiguous key lanes for each head-dimension index. Padding columns are zero.
                for (int j = 0; j < n; j++)
                {
                    ref float krow = ref Unsafe.Add(ref kBaseRef, (nint)(kBase + ((kb + j) * s.KRowStride)));
                    for (int d = 0; d < dim; d++)
                    {
                        Unsafe.Add(ref kt, (d * Bc) + j) = Unsafe.Add(ref krow, d);
                    }
                }

                for (int j = n; j < nPad; j++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        Unsafe.Add(ref kt, (d * Bc) + j) = 0f;
                    }
                }

                // Raw scores S = Q K^T, four query rows by two vectors of keys at a time. Per output the sum runs over the head
                // dimension in increasing order from +0, one accumulator, so the result does not depend on the lane width.
                for (int rg = 0; rg < rowsPadded; rg += 4)
                {
                    ref float qr0 = ref Unsafe.Add(ref qBaseRef, (nint)(qBase + ((rowStart + Math.Min(rg, rows - 1)) * s.QRowStride)));
                    ref float qr1 = ref Unsafe.Add(ref qBaseRef, (nint)(qBase + ((rowStart + Math.Min(rg + 1, rows - 1)) * s.QRowStride)));
                    ref float qr2 = ref Unsafe.Add(ref qBaseRef, (nint)(qBase + ((rowStart + Math.Min(rg + 2, rows - 1)) * s.QRowStride)));
                    ref float qr3 = ref Unsafe.Add(ref qBaseRef, (nint)(qBase + ((rowStart + Math.Min(rg + 3, rows - 1)) * s.QRowStride)));
                    for (int cg = 0; cg < nPad; cg += nr)
                    {
                        TV a00 = TO.Zero, a01 = TO.Zero, a10 = TO.Zero, a11 = TO.Zero;
                        TV a20 = TO.Zero, a21 = TO.Zero, a30 = TO.Zero, a31 = TO.Zero;
                        for (int d = 0; d < dim; d++)
                        {
                            TV b0 = TO.Load(ref kt, (nuint)((d * Bc) + cg));
                            TV b1 = TO.Load(ref kt, (nuint)((d * Bc) + cg + lanes));
                            TV x0 = TO.Splat(Unsafe.Add(ref qr0, d));
                            TV x1 = TO.Splat(Unsafe.Add(ref qr1, d));
                            TV x2 = TO.Splat(Unsafe.Add(ref qr2, d));
                            TV x3 = TO.Splat(Unsafe.Add(ref qr3, d));
                            a00 = TO.Accumulate(a00, x0, b0);
                            a01 = TO.Accumulate(a01, x0, b1);
                            a10 = TO.Accumulate(a10, x1, b0);
                            a11 = TO.Accumulate(a11, x1, b1);
                            a20 = TO.Accumulate(a20, x2, b0);
                            a21 = TO.Accumulate(a21, x2, b1);
                            a30 = TO.Accumulate(a30, x3, b0);
                            a31 = TO.Accumulate(a31, x3, b1);
                        }

                        int t0 = (rg * Bc) + cg;
                        TO.Store(ref tile, (nuint)t0, a00);
                        TO.Store(ref tile, (nuint)(t0 + lanes), a01);
                        TO.Store(ref tile, (nuint)(t0 + Bc), a10);
                        TO.Store(ref tile, (nuint)(t0 + Bc + lanes), a11);
                        TO.Store(ref tile, (nuint)(t0 + (2 * Bc)), a20);
                        TO.Store(ref tile, (nuint)(t0 + (2 * Bc) + lanes), a21);
                        TO.Store(ref tile, (nuint)(t0 + (3 * Bc)), a30);
                        TO.Store(ref tile, (nuint)(t0 + (3 * Bc) + lanes), a31);
                    }
                }

                if (!hasMask)
                {
                    for (int j = 0; j < n; j++)
                    {
                        Unsafe.Add(ref maskRow, j) = 0f;
                    }

                    for (int j = n; j < nPad; j++)
                    {
                        Unsafe.Add(ref maskRow, j) = float.NegativeInfinity;
                    }
                }

                // Online softmax (Dao et al. 2022, alg. 1) per query row.
                for (int r = 0; r < rows; r++)
                {
                    ref float srow = ref Unsafe.Add(ref tile, r * Bc);

                    // S = mask + scale * dot, one pass that also collects the row maximum (max ignores NaN and does not depend on
                    // order or lane width). The mask row is zero where nothing is masked and -inf in the padding columns, so
                    // padding carries no weight and cannot win the maximum.
                    if (hasMask)
                    {
                        long mo = maskBase + ((rowStart + r) * s.MaskRowStride) + kb;
                        for (int j = 0; j < n; j++)
                        {
                            Unsafe.Add(ref maskRow, j) = Unsafe.Add(ref mask0, (nint)(mo + j));
                        }

                        for (int j = n; j < nPad; j++)
                        {
                            Unsafe.Add(ref maskRow, j) = float.NegativeInfinity;
                        }
                    }

                    TV runMax = TO.Splat(float.NegativeInfinity);
                    for (int c = 0; c < nPad; c += lanes)
                    {
                        TV sv = TO.Accumulate(TO.Load(ref maskRow, (nuint)c), TO.Load(ref srow, (nuint)c), scaleV);
                        TO.Store(ref srow, (nuint)c, sv);
                        runMax = TO.Select(TO.GreaterThan(sv, runMax), sv, runMax);
                    }

                    TO.Store(ref misc, 0, runMax);
                    float blockMax = float.NegativeInfinity;
                    for (int t = 0; t < lanes; t++)
                    {
                        float lane = Unsafe.Add(ref misc, t);
                        blockMax = lane > blockMax ? lane : blockMax;
                    }
                    float mOld = mS[r];
                    float mNew = blockMax > mOld ? blockMax : mOld;

                    // A row with nothing but -inf so far contributes nothing; shifting by 0 avoids (-inf) - (-inf).
                    float shift = float.IsNegativeInfinity(mNew) ? 0f : mNew;
                    mS[r] = mNew;

                    // exp(-inf) = 0 and exp(0) = 1 exactly, so the two common cases skip the exponential without changing a bit.
                    float alpha;
                    if (float.IsNegativeInfinity(mOld))
                    {
                        alpha = 0f;
                    }
                    else if (mOld == shift)
                    {
                        alpha = 1f;
                    }
                    else
                    {
                        TO.Store(ref misc, 0, TO.SoftmaxExp(TO.Splat(mOld - shift)));
                        alpha = misc;
                    }

                    TV vshift = TO.Splat(shift);
                    for (int c = 0; c < nPad; c += lanes)
                    {
                        TO.Store(ref srow, (nuint)c, TO.SoftmaxExp(TO.Sub(TO.Load(ref srow, (nuint)c), vshift)));
                    }

                    float blockSum = RowReduce.Sum16<TV, TO>(ref srow, nPad);
                    lS[r] = (lS[r] * alpha) + blockSum;

                    if (alpha != 1f)
                    {
                        ref float arow = ref Unsafe.Add(ref acc, r * dim);
                        TV va = TO.Splat(alpha);
                        for (int d = 0; d < dim; d += lanes)
                        {
                            TO.Store(ref arow, (nuint)d, TO.Mul(TO.Load(ref arow, (nuint)d), va));
                        }
                    }
                }

                // O += P V, four rows by two vectors of the head dimension; sequential over the block's keys in increasing order.
                for (int rg = 0; rg < rowsPadded; rg += 4)
                {
                    for (int dg = 0; dg < dim; dg += nr)
                    {
                        ref float ar0 = ref Unsafe.Add(ref acc, (rg * dim) + dg);
                        TV a00 = TO.Load(ref ar0, 0), a01 = TO.Load(ref ar0, (nuint)lanes);
                        TV a10 = TO.Load(ref ar0, (nuint)dim), a11 = TO.Load(ref ar0, (nuint)(dim + lanes));
                        TV a20 = TO.Load(ref ar0, (nuint)(2 * dim)), a21 = TO.Load(ref ar0, (nuint)((2 * dim) + lanes));
                        TV a30 = TO.Load(ref ar0, (nuint)(3 * dim)), a31 = TO.Load(ref ar0, (nuint)((3 * dim) + lanes));
                        ref float p0 = ref Unsafe.Add(ref tile, rg * Bc);
                        ref float vcol = ref Unsafe.Add(ref vBaseRef, (nint)(vBase + (kb * s.VRowStride) + dg));
                        for (int j = 0; j < n; j++)
                        {
                            nint vo = (nint)(j * s.VRowStride);
                            TV b0 = TO.Load(ref vcol, (nuint)vo);
                            TV b1 = TO.Load(ref vcol, (nuint)(vo + lanes));
                            TV x0 = TO.Splat(Unsafe.Add(ref p0, j));
                            TV x1 = TO.Splat(Unsafe.Add(ref p0, Bc + j));
                            TV x2 = TO.Splat(Unsafe.Add(ref p0, (2 * Bc) + j));
                            TV x3 = TO.Splat(Unsafe.Add(ref p0, (3 * Bc) + j));
                            a00 = TO.Accumulate(a00, x0, b0);
                            a01 = TO.Accumulate(a01, x0, b1);
                            a10 = TO.Accumulate(a10, x1, b0);
                            a11 = TO.Accumulate(a11, x1, b1);
                            a20 = TO.Accumulate(a20, x2, b0);
                            a21 = TO.Accumulate(a21, x2, b1);
                            a30 = TO.Accumulate(a30, x3, b0);
                            a31 = TO.Accumulate(a31, x3, b1);
                        }

                        TO.Store(ref ar0, 0, a00);
                        TO.Store(ref ar0, (nuint)lanes, a01);
                        TO.Store(ref ar0, (nuint)dim, a10);
                        TO.Store(ref ar0, (nuint)(dim + lanes), a11);
                        TO.Store(ref ar0, (nuint)(2 * dim), a20);
                        TO.Store(ref ar0, (nuint)((2 * dim) + lanes), a21);
                        TO.Store(ref ar0, (nuint)(3 * dim), a30);
                        TO.Store(ref ar0, (nuint)((3 * dim) + lanes), a31);
                    }
                }
            }

            // O = acc / l. Any NaN (fully masked row, inf or NaN input) becomes the canonical quiet NaN.
            for (int r = 0; r < rows; r++)
            {
                ref float arow = ref Unsafe.Add(ref acc, r * dim);
                ref float orow = ref Unsafe.Add(ref oBaseRef, (nint)(oBase + ((rowStart + r) * s.ORowStride)));
                float l = lS[r];
                if (TO.Fused)
                {
                    TV inv = TO.Splat(1f / l);
                    for (int d = 0; d < dim; d += lanes)
                    {
                        TV res = TO.Mul(TO.Load(ref arow, (nuint)d), inv);
                        TO.Store(ref orow, (nuint)d, TO.Select(TO.Equal(res, res), res, nanV));
                    }
                }
                else
                {
                    TV vl = TO.Splat(l);
                    for (int d = 0; d < dim; d += lanes)
                    {
                        TV res = TO.Div(TO.Load(ref arow, (nuint)d), vl);
                        TO.Store(ref orow, (nuint)d, TO.Select(TO.Equal(res, res), res, nanV));
                    }
                }
            }
        }

    }
}
