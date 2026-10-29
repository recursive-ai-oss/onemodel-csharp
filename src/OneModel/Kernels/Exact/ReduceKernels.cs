using System;
using System.Numerics;
using System.Runtime.Intrinsics;

namespace RecursiveAI.OneModel.Kernels.Exact;

// Reductions over any set of axes (ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd, ReduceL1, ReduceL2, ReduceSumSquare,
// ReduceLogSum, ReduceLogSumExp). docs/ops-norms.md is the normative text; the summary:
//
// The input is first reduced to a canonical geometry: dimensions of size one are dropped and neighbouring dimensions that are both
// reduced or both kept are merged, so the shape becomes alternating runs of kept and reduced extents. Which of two layouts follows
// depends only on the shape and the axes:
//   * column layout: the last run is kept (the reduced axes are not innermost). Each output is accumulated sequentially over the
//     reduced index tuples in row-major order, starting from the identity, and neighbouring outputs share one vector, so a reduction
//     over a leading axis is linear and vectorised, never a strided gather per output.
//   * row layout: the last run is reduced. Each output reduces a sequence made of the reduced index tuples in row-major order. Sums
//     (and the L1, L2, sum-of-squares and log-sum forms) add element q of that sequence to partial sum q mod 16 and combine the 16
//     partials by halving, exactly as RowReduce.Sum16 does. Products are sequential. Max and min are exact in any order.
// Every operation is a single IEEE add, multiply, divide or square root, so the result is the same at every lane width and in both
// execution modes; NaN results are rewritten to the quiet NaN 0x7FC00000. Work is split into chunks whose size depends only on the
// geometry. Each output is computed by one chunk, so no partial result is ever combined across chunks.

/// <summary>The reduction functions.</summary>
internal enum ReduceOp
{
    Sum,
    Mean,
    Max,
    Min,
    Prod,
    L1,
    L2,
    SumSquare,
    LogSum,
    LogSumExp,
}

/// <summary>Eight ints stored inline, so a geometry is a plain value that needs no allocation.</summary>
[System.Runtime.CompilerServices.InlineArray(ReduceGeometryLimits.MaxRuns)]
internal struct RunArray
{
    private int _element0;
}

/// <summary>The canonical geometry of a reduction: alternating runs of kept and reduced extents of a row-major input.</summary>
internal readonly struct ReduceGeometry
{
    /// <summary>Elements of work one chunk should hold; chunk sizes are derived from it and the geometry alone.</summary>
    internal const int ChunkWork = 32768;

    /// <summary>Width of the column-layout tile, in outputs.</summary>
    internal const int Tile = 256;

    private ReduceGeometry(in RunArray keptSize, in RunArray keptStride, int keptRuns, in RunArray redSize, in RunArray redStride, int redRuns,
        bool column, long inputLength)
    {
        KeptSize = keptSize;
        KeptStride = keptStride;
        KeptRuns = keptRuns;
        RedSize = redSize;
        RedStride = redStride;
        RedRuns = redRuns;
        ColumnLayout = column;
        long outputs = 1;
        for (int i = 0; i < keptRuns; i++)
        {
            outputs *= keptSize[i];
        }

        long reduced = 1;
        for (int i = 0; i < redRuns; i++)
        {
            reduced *= redSize[i];
        }

        Outputs = outputs;
        ReducedCount = reduced;
        InputLength = inputLength;
        Inner = 0;
        TilesPerRow = 0;
        RowLen = 0;
        if (column)
        {
            Inner = keptSize[keptRuns - 1];
            HeadKeptRuns = keptRuns - 1;
            HeadRedRuns = redRuns;
            long head = Inner == 0 ? 0 : outputs / Inner;
            TilesPerRow = (Inner + Tile - 1) / Tile;
            Units = head * TilesPerRow;
            long perUnit = Math.Max(1L, Math.Min(Inner, Tile) * Math.Max(1L, reduced));
            UnitsPerChunk = (int)Math.Max(1L, ChunkWork / perUnit);
        }
        else
        {
            RowLen = redSize[redRuns - 1];
            HeadKeptRuns = keptRuns;
            HeadRedRuns = redRuns - 1;
            Units = outputs;
            UnitsPerChunk = (int)Math.Max(1L, ChunkWork / Math.Max(1L, reduced));
        }
    }

    /// <summary>Extents of the kept runs, in input order, and their element strides in the input.</summary>
    public RunArray KeptSize { get; }

    public RunArray KeptStride { get; }

    /// <summary>Number of kept runs.</summary>
    public int KeptRuns { get; }

    /// <summary>Extents of the reduced runs, in input order, and their element strides in the input.</summary>
    public RunArray RedSize { get; }

    public RunArray RedStride { get; }

    /// <summary>Number of reduced runs.</summary>
    public int RedRuns { get; }

    /// <summary>True when the last run is kept.</summary>
    public bool ColumnLayout { get; }

    /// <summary>Column layout: extent of the last (contiguous, kept) run.</summary>
    public int Inner { get; }

    /// <summary>Column layout: tiles of <see cref="Tile"/> outputs per row of <see cref="Inner"/>.</summary>
    public int TilesPerRow { get; }

    /// <summary>Row layout: extent of the last (contiguous, reduced) run.</summary>
    public int RowLen { get; }

    /// <summary>Number of leading kept runs that are decomposed into an input offset (all of them in row layout, all but the last in column layout).</summary>
    public int HeadKeptRuns { get; }

    /// <summary>Number of reduced runs the odometer walks (all of them in column layout, all but the last in row layout).</summary>
    public int HeadRedRuns { get; }

    /// <summary>Output elements.</summary>
    public long Outputs { get; }

    /// <summary>Input elements reduced into each output.</summary>
    public long ReducedCount { get; }

    /// <summary>Input elements.</summary>
    public long InputLength { get; }

    /// <summary>Units of work of the vector kernel: tiles in column layout, outputs in row layout.</summary>
    public long Units { get; }

    /// <summary>Units per chunk of the vector kernel.</summary>
    public int UnitsPerChunk { get; }

    /// <summary>Chunks of the vector kernel.</summary>
    public int Chunks => (int)((Units + UnitsPerChunk - 1) / UnitsPerChunk);

    /// <summary>Outputs per chunk of the generic kernel, which makes one output per unit.</summary>
    public int OutputsPerChunk => (int)Math.Max(1L, ChunkWork / Math.Max(1L, ReducedCount));

    /// <summary>Chunks of the generic kernel.</summary>
    public int GenericChunks => (int)((Outputs + OutputsPerChunk - 1) / OutputsPerChunk);

    /// <summary>
    /// Builds the geometry for an input of the given dimensions (row-major, contiguous; at most 8) where <paramref name="reduced"/>
    /// marks the axes to reduce.
    /// </summary>
    public static ReduceGeometry Create(ReadOnlySpan<int> dims, ReadOnlySpan<bool> reduced)
    {
        long total = 1;
        foreach (int d in dims)
        {
            total *= d;
        }

        // Runs from the outermost axis in: size-1 axes vanish, neighbours with the same status merge.
        Span<int> size = stackalloc int[ReduceGeometryLimits.MaxRuns];
        Span<bool> isRed = stackalloc bool[ReduceGeometryLimits.MaxRuns];
        int runs = 0;
        for (int a = 0; a < dims.Length; a++)
        {
            if (dims[a] == 1)
            {
                continue;
            }

            if (runs > 0 && isRed[runs - 1] == reduced[a])
            {
                size[runs - 1] = checked(size[runs - 1] * dims[a]);
            }
            else
            {
                size[runs] = dims[a];
                isRed[runs] = reduced[a];
                runs++;
            }
        }

        int kept = 0, red = 0;
        for (int i = 0; i < runs; i++)
        {
            if (isRed[i])
            {
                red++;
            }
            else
            {
                kept++;
            }
        }

        RunArray ks = default, kst = default, rs = default, rst = default;
        long stride = 1;
        int ki = kept, ri = red;
        for (int i = runs - 1; i >= 0; i--)
        {
            if (isRed[i])
            {
                rs[--ri] = size[i];
                rst[ri] = (int)Math.Min(stride, int.MaxValue);
            }
            else
            {
                ks[--ki] = size[i];
                kst[ki] = (int)Math.Min(stride, int.MaxValue);
            }

            stride *= size[i];
        }

        bool column = runs == 0 || !isRed[runs - 1];
        if (column && kept == 0)
        {
            // Nothing is kept (the input is a scalar or all its axes have extent one): one output, one trivial inner run.
            ks[0] = 1;
            kst[0] = 1;
            kept = 1;
        }

        return new ReduceGeometry(ks, kst, kept, rs, rst, red, column, total);
    }
}


/// <summary>The reduction kernels.</summary>
internal static class ReduceKernels
{
    private const uint QuietNaNBits = 0x7FC00000u;

    private static float Nan => BitConverter.UInt32BitsToSingle(QuietNaNBits);

    /// <summary>True when <see cref="FloatRange"/> handles the operation (everything except log-sum-exp).</summary>
    public static bool IsVectorOp(ReduceOp op) => op != ReduceOp.LogSumExp;

    /// <summary>
    /// Computes units <c>[unit0, unit1)</c> of <paramref name="g"/> (see <see cref="ReduceGeometry.Units"/>) for float input into
    /// <paramref name="y"/>. Different unit ranges write disjoint outputs.
    /// </summary>
    public static void FloatRange(ReduceOp op, VecIsa isa, in ReduceGeometry g, ReadOnlySpan<float> x, Span<float> y, long unit0, long unit1)
    {
        if (unit0 < 0 || unit1 > g.Units || x.Length < g.InputLength || y.Length < g.Outputs || !IsVectorOp(op))
        {
            throw new ArgumentException("The unit range, an operand length or the operation does not fit the geometry.");
        }

        isa = VecIsaInfo.Resolve(isa);
        switch (isa)
        {
            case VecIsa.V512: Dispatch<Vector512<float>, ExactVec512>(op, g, x, y, unit0, unit1); break;
            case VecIsa.V256: Dispatch<Vector256<float>, ExactVec256>(op, g, x, y, unit0, unit1); break;
            case VecIsa.V128: Dispatch<Vector128<float>, ExactVec128>(op, g, x, y, unit0, unit1); break;
            default: Dispatch<float, ExactVecScalar>(op, g, x, y, unit0, unit1); break;
        }
    }

    private static void Dispatch<TV, TO>(ReduceOp op, in ReduceGeometry g, ReadOnlySpan<float> x, Span<float> y, long u0, long u1)
        where TV : struct
        where TO : IVecOps<TV>
    {
        switch (op)
        {
            case ReduceOp.Sum:
            case ReduceOp.Mean:
            case ReduceOp.LogSum:
                SumLike<TV, TO, TermId>(op, g, x, y, u0, u1);
                break;
            case ReduceOp.L1:
                SumLike<TV, TO, TermAbs>(op, g, x, y, u0, u1);
                break;
            case ReduceOp.L2:
            case ReduceOp.SumSquare:
                SumLike<TV, TO, TermSquare>(op, g, x, y, u0, u1);
                break;
            case ReduceOp.Prod:
                Product<TV, TO>(g, x, y, u0, u1);
                break;
            default:
                Extreme<TV, TO>(op == ReduceOp.Max, g, x, y, u0, u1);
                break;
        }
    }

    // ---- terms -------------------------------------------------------------------------------------------------------------------

    private interface IReduceTerm
    {
        static abstract TV Apply<TV, TO>(TV x)
            where TV : struct
            where TO : IVecOps<TV>;

        static abstract float Apply(float x);
    }

    private readonly struct TermId : IReduceTerm
    {
        public static TV Apply<TV, TO>(TV x)
            where TV : struct
            where TO : IVecOps<TV> => x;

        public static float Apply(float x) => x;
    }

    private readonly struct TermAbs : IReduceTerm
    {
        public static TV Apply<TV, TO>(TV x)
            where TV : struct
            where TO : IVecOps<TV> => TO.And(x, TO.Splat(BitConverter.UInt32BitsToSingle(0x7FFFFFFFu)));

        public static float Apply(float x) => float.Abs(x);
    }

    private readonly struct TermSquare : IReduceTerm
    {
        public static TV Apply<TV, TO>(TV x)
            where TV : struct
            where TO : IVecOps<TV> => TO.Mul(x, x);

        public static float Apply(float x) => x * x;
    }

    // ---- shared helpers ----------------------------------------------------------------------------------------------------------

    // Input offset of kept-tuple number h over the first `runs` kept runs (row-major decomposition).
    private static long KeptOffset(in ReduceGeometry g, long h, int runs)
    {
        long off = 0;
        for (int i = runs - 1; i >= 0; i--)
        {
            long s = g.KeptSize[i];
            off += (h % s) * g.KeptStride[i];
            h /= s;
        }

        return off;
    }

    // Advances the odometer over the first `runs` reduced runs and updates the offset; returns false after the last tuple.
    private static bool Next(in ReduceGeometry g, int runs, Span<int> idx, ref long off)
    {
        for (int i = runs - 1; i >= 0; i--)
        {
            off += g.RedStride[i];
            if (++idx[i] < g.RedSize[i])
            {
                return true;
            }

            off -= (long)g.RedSize[i] * g.RedStride[i];
            idx[i] = 0;
        }

        return false;
    }

    private static float Post(ReduceOp op, float acc, long count)
    {
        float v = acc;
        switch (op)
        {
            case ReduceOp.Mean:
                v = count > 0 ? acc / count : acc;
                break;
            case ReduceOp.L2:
                v = float.Sqrt(acc);
                break;
            case ReduceOp.LogSum:
                v = Transcendental.Log<float, ExactVecScalar>(acc);
                break;
        }

        return float.IsNaN(v) ? Nan : v;
    }

    private static float PostElement(ReduceOp op, float acc, long count) => op is ReduceOp.Sum or ReduceOp.SumSquare or ReduceOp.L1 or ReduceOp.Max or ReduceOp.Min or ReduceOp.Prod
        ? (float.IsNaN(acc) ? Nan : acc)
        : Post(op, acc, count);

    // ---- sums ----------------------------------------------------------------------------------------------------------------------

    private static void SumLike<TV, TO, TT>(ReduceOp op, in ReduceGeometry g, ReadOnlySpan<float> x, Span<float> y, long u0, long u1)
        where TV : struct
        where TO : IVecOps<TV>
        where TT : IReduceTerm
    {
        Span<int> idx = stackalloc int[ReduceGeometryLimits.MaxRuns];
        int lanes = TO.Lanes;
        long count = g.ReducedCount;
        if (g.ColumnLayout)
        {
            int inner = g.Inner;
            for (long u = u0; u < u1; u++)
            {
                long h = u / g.TilesPerRow;
                int i0 = (int)(u % g.TilesPerRow) * ReduceGeometry.Tile;
                int i1 = Math.Min(inner, i0 + ReduceGeometry.Tile);
                long outBase = (h * inner) + i0;
                long inBase = KeptOffset(g, h, g.HeadKeptRuns) + i0;
                int width = i1 - i0;
                int body = width - (width % lanes);
                Span<float> acc = y.Slice((int)outBase, width);
                acc.Clear();

                idx.Clear();
                long off = 0;
                bool more = count > 0;
                while (more)
                {
                    ReadOnlySpan<float> row = x.Slice((int)(inBase + off), width);
                    for (int j = 0; j < body; j += lanes)
                    {
                        TO.Store(acc, j, TO.Add(TO.Load(acc, j), TT.Apply<TV, TO>(TO.Load(row, j))));
                    }

                    for (int j = body; j < width; j++)
                    {
                        acc[j] += TT.Apply(row[j]);
                    }

                    more = Next(g, g.HeadRedRuns, idx, ref off);
                }

                for (int j = 0; j < width; j++)
                {
                    acc[j] = PostElement(op, acc[j], count);
                }
            }

            return;
        }

        Span<float> part = stackalloc float[16];
        int rowLen = g.RowLen;
        for (long o = u0; o < u1; o++)
        {
            long inBase = KeptOffset(g, o, g.HeadKeptRuns);
            float sum;
            if (g.HeadRedRuns == 0)
            {
                sum = count > 0 ? SumTerms16<TV, TO, TT>(x.Slice((int)inBase, rowLen), part) : 0f;
            }
            else
            {
                // Several reduced runs: the same partial-sum rule, with element q of the row-major sequence going to partial q mod 16.
                part.Clear();
                idx.Clear();
                long off = 0, q = 0;
                bool more = count > 0;
                while (more)
                {
                    ReadOnlySpan<float> seg = x.Slice((int)(inBase + off), rowLen);
                    for (int j = 0; j < seg.Length; j++, q++)
                    {
                        part[(int)(q & 15)] += TT.Apply(seg[j]);
                    }

                    more = Next(g, g.HeadRedRuns, idx, ref off);
                }

                sum = NormKernels.Combine16(part);
            }

            y[(int)o] = Post(op, sum, count);
        }
    }

    // Sum of term(x[j]) in the canonical order of RowReduce.Sum16.
    private static float SumTerms16<TV, TO, TT>(ReadOnlySpan<float> x, Span<float> part)
        where TV : struct
        where TO : IVecOps<TV>
        where TT : IReduceTerm
    {
        part.Clear();
        int lanes = TO.Lanes;
        int full = x.Length & ~15;
        for (int c = 0; c < full; c += 16)
        {
            for (int t = 0; t < 16; t += lanes)
            {
                TO.Store(part, t, TO.Add(TO.Load(part, t), TT.Apply<TV, TO>(TO.Load(x, c + t))));
            }
        }

        for (int j = full; j < x.Length; j++)
        {
            part[j & 15] += TT.Apply(x[j]);
        }

        return NormKernels.Combine16(part);
    }

    // ---- products ------------------------------------------------------------------------------------------------------------------

    private static void Product<TV, TO>(in ReduceGeometry g, ReadOnlySpan<float> x, Span<float> y, long u0, long u1)
        where TV : struct
        where TO : IVecOps<TV>
    {
        Span<int> idx = stackalloc int[ReduceGeometryLimits.MaxRuns];
        if (g.ColumnLayout)
        {
            int lanes = TO.Lanes, inner = g.Inner;
            for (long u = u0; u < u1; u++)
            {
                long h = u / g.TilesPerRow;
                int i0 = (int)(u % g.TilesPerRow) * ReduceGeometry.Tile;
                int width = Math.Min(inner, i0 + ReduceGeometry.Tile) - i0;
                int body = width - (width % lanes);
                long inBase = KeptOffset(g, h, g.HeadKeptRuns) + i0;
                Span<float> acc = y.Slice((int)((h * inner) + i0), width);
                acc.Fill(1f);
                idx.Clear();
                long off = 0;
                bool more = g.ReducedCount > 0;
                while (more)
                {
                    ReadOnlySpan<float> row = x.Slice((int)(inBase + off), width);
                    for (int j = 0; j < body; j += lanes)
                    {
                        TO.Store(acc, j, TO.Mul(TO.Load(acc, j), TO.Load(row, j)));
                    }

                    for (int j = body; j < width; j++)
                    {
                        acc[j] *= row[j];
                    }

                    more = Next(g, g.HeadRedRuns, idx, ref off);
                }

                for (int j = 0; j < width; j++)
                {
                    acc[j] = float.IsNaN(acc[j]) ? Nan : acc[j];
                }
            }

            return;
        }

        for (long o = u0; o < u1; o++)
        {
            long inBase = KeptOffset(g, o, g.HeadKeptRuns);
            float acc = 1f;
            idx.Clear();
            long off = 0;
            bool more = g.ReducedCount > 0;
            while (more)
            {
                ReadOnlySpan<float> seg = x.Slice((int)(inBase + off), g.RowLen);
                for (int j = 0; j < seg.Length; j++)
                {
                    acc *= seg[j];
                }

                more = Next(g, g.HeadRedRuns, idx, ref off);
            }

            y[(int)o] = float.IsNaN(acc) ? Nan : acc;
        }
    }

    // ---- maximum and minimum ---------------------------------------------------------------------------------------------------------

    // The rule for NaN in a maximum or minimum is that of a sequential scan: the running value starts as the first element and is
    // replaced only by a strictly better one. A NaN is never better, so it is ignored, except that a NaN first element is never
    // replaced. (The comparisons are all false for NaN.) ONNX Runtime's result for NaN inputs depends on its internal code path; this
    // rule matches it where the reduced axes are not innermost and on the probed small inputs. The spec leaves NaN open.
    private static float Fold(bool max, float m, float v) => (max ? v > m : v < m) ? v : m;

    private static TV Fold<TV, TO>(bool max, TV m, TV v)
        where TV : struct
        where TO : IVecOps<TV> => max ? TO.Select(TO.GreaterThan(v, m), v, m) : TO.Select(TO.LessThan(v, m), v, m);

    // The same value as a sequential scan from the first element, computed with vectors that start at the identity: the NaN-ignoring
    // extreme of everything, or NaN when the first element is NaN.
    private static void Extreme<TV, TO>(bool max, in ReduceGeometry g, ReadOnlySpan<float> x, Span<float> y, long u0, long u1)
        where TV : struct
        where TO : IVecOps<TV>
    {
        Span<int> idx = stackalloc int[ReduceGeometryLimits.MaxRuns];
        Span<float> lane = stackalloc float[16];
        int lanes = TO.Lanes;
        float identity = max ? float.NegativeInfinity : float.PositiveInfinity;
        if (g.ColumnLayout)
        {
            int inner = g.Inner;
            for (long u = u0; u < u1; u++)
            {
                long h = u / g.TilesPerRow;
                int i0 = (int)(u % g.TilesPerRow) * ReduceGeometry.Tile;
                int width = Math.Min(inner, i0 + ReduceGeometry.Tile) - i0;
                int body = width - (width % lanes);
                long inBase = KeptOffset(g, h, g.HeadKeptRuns) + i0;
                Span<float> acc = y.Slice((int)((h * inner) + i0), width);
                acc.Fill(identity);
                idx.Clear();
                long off = 0;
                bool more = g.ReducedCount > 0, first = true;
                while (more)
                {
                    ReadOnlySpan<float> row = x.Slice((int)(inBase + off), width);
                    if (first)
                    {
                        row.CopyTo(acc);
                        first = false;
                        more = Next(g, g.HeadRedRuns, idx, ref off);
                        continue;
                    }

                    for (int j = 0; j < body; j += lanes)
                    {
                        TO.Store(acc, j, Fold<TV, TO>(max, TO.Load(acc, j), TO.Load(row, j)));
                    }

                    for (int j = body; j < width; j++)
                    {
                        acc[j] = Fold(max, acc[j], row[j]);
                    }

                    more = Next(g, g.HeadRedRuns, idx, ref off);
                }

                for (int j = 0; j < width; j++)
                {
                    acc[j] = float.IsNaN(acc[j]) ? Nan : acc[j];
                }
            }

            return;
        }

        for (long o = u0; o < u1; o++)
        {
            long inBase = KeptOffset(g, o, g.HeadKeptRuns);
            float m = identity;
            idx.Clear();
            long off = 0;
            bool more = g.ReducedCount > 0;
            while (more)
            {
                ReadOnlySpan<float> seg = x.Slice((int)(inBase + off), g.RowLen);
                int body = seg.Length - (seg.Length % lanes);
                if (body > 0)
                {
                    TV acc = TO.Splat(identity);
                    for (int j = 0; j < body; j += lanes)
                    {
                        acc = Fold<TV, TO>(max, acc, TO.Load(seg, j));
                    }

                    TO.Store(lane, 0, acc);
                    for (int t = 0; t < lanes; t++)
                    {
                        m = Fold(max, m, lane[t]);
                    }
                }

                for (int j = body; j < seg.Length; j++)
                {
                    m = Fold(max, m, seg[j]);
                }

                more = Next(g, g.HeadRedRuns, idx, ref off);
            }

            y[(int)o] = g.ReducedCount > 0 && float.IsNaN(x[(int)inBase]) ? Nan : m;
        }
    }

    // ---- log-sum-exp (scalar) --------------------------------------------------------------------------------------------------------

    /// <summary>
    /// Log-sum-exp for outputs <c>[o0, o1)</c>: <c>m = max</c>, <c>s = sum exp(x - m)</c> in sequence order, <c>m + log(s)</c>; -inf
    /// when every element is -inf and when nothing is reduced. Uses the library's own exponential and logarithm.
    /// </summary>
    public static void LogSumExpRange(in ReduceGeometry g, ReadOnlySpan<float> x, Span<float> y, long o0, long o1)
    {
        if (o0 < 0 || o1 > g.Outputs || x.Length < g.InputLength || y.Length < g.Outputs)
        {
            throw new ArgumentException("The output range or an operand length does not fit the geometry.");
        }

        Span<int> idx = stackalloc int[ReduceGeometryLimits.MaxRuns];
        for (long o = o0; o < o1; o++)
        {
            long inBase = KeptOffset(g, o, g.KeptRuns);
            float m = float.NegativeInfinity;
            idx.Clear();
            long off = 0;
            bool more = g.ReducedCount > 0;
            while (more)
            {
                float v = x[(int)(inBase + off)];
                m = float.IsNaN(m) || float.IsNaN(v) ? float.NaN : (v > m ? v : m);
                more = AllNext(g, idx, ref off);
            }

            float r;
            if (float.IsNaN(m))
            {
                r = Nan;
            }
            else if (float.IsNegativeInfinity(m) || float.IsPositiveInfinity(m))
            {
                r = m;
            }
            else
            {
                float s = 0f;
                idx.Clear();
                off = 0;
                more = g.ReducedCount > 0;
                while (more)
                {
                    s += Transcendental.Exp<float, ExactVecScalar>(x[(int)(inBase + off)] - m);
                    more = AllNext(g, idx, ref off);
                }

                r = m + Transcendental.Log<float, ExactVecScalar>(s);
            }

            y[(int)o] = float.IsNaN(r) ? Nan : r;
        }
    }

    private static bool AllNext(in ReduceGeometry g, Span<int> idx, ref long off)
    {
        // Row-major over every reduced run.
        for (int i = g.RedRuns - 1; i >= 0; i--)
        {
            off += g.RedStride[i];
            if (++idx[i] < g.RedSize[i])
            {
                return true;
            }

            off -= (long)g.RedSize[i] * g.RedStride[i];
            idx[i] = 0;
        }

        return false;
    }

    // ---- every other element type --------------------------------------------------------------------------------------------------

    /// <summary>
    /// Outputs <c>[o0, o1)</c> for element types other than float32, one output at a time, accumulating sequentially in row-major
    /// order of the reduced index tuples. Integer arithmetic wraps; the mean of integers truncates toward zero.
    /// </summary>
    public static void GenericRange<T>(ReduceOp op, in ReduceGeometry g, ReadOnlySpan<T> x, Span<T> y, long o0, long o1)
        where T : unmanaged, INumber<T>, IMinMaxValue<T>
    {
        if (o0 < 0 || o1 > g.Outputs || x.Length < g.InputLength || y.Length < g.Outputs)
        {
            throw new ArgumentException("The output range or an operand length does not fit the geometry.");
        }

        Span<int> idx = stackalloc int[ReduceGeometryLimits.MaxRuns];
        bool isFloat = typeof(T) == typeof(double) || typeof(T) == typeof(float);
        T identity = op switch
        {
            ReduceOp.Max => isFloat ? T.CreateChecked(double.NegativeInfinity) : T.MinValue,
            ReduceOp.Min => isFloat ? T.CreateChecked(double.PositiveInfinity) : T.MaxValue,
            ReduceOp.Prod => T.One,
            _ => T.Zero,
        };

        for (long o = o0; o < o1; o++)
        {
            long inBase = KeptOffset(g, o, g.KeptRuns);
            T acc = identity;
            idx.Clear();
            long off = 0;
            bool more = g.ReducedCount > 0, first = true;
            while (more)
            {
                T v = x[(int)(inBase + off)];
                unchecked
                {
                    switch (op)
                    {
                        case ReduceOp.Sum:
                        case ReduceOp.Mean:
                            acc += v;
                            break;
                        case ReduceOp.L1:
                            acc += T.Abs(v);
                            break;
                        case ReduceOp.L2:
                        case ReduceOp.SumSquare:
                            acc += v * v;
                            break;
                        case ReduceOp.Prod:
                            acc *= v;
                            break;
                        case ReduceOp.Max:
                            acc = first || v > acc ? v : acc;
                            break;
                        default:
                            acc = first || v < acc ? v : acc;
                            break;
                    }
                }

                first = false;
                more = AllNext(g, idx, ref off);
            }

            if (op == ReduceOp.Mean && g.ReducedCount > 0)
            {
                acc /= T.CreateTruncating(g.ReducedCount);
            }
            else if (op == ReduceOp.L2)
            {
                acc = T.CreateTruncating(double.Sqrt(double.CreateTruncating(acc)));
            }

            y[(int)o] = acc;
        }
    }

    // ---- index reductions ------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// ArgMax / ArgMin along one axis of an input viewed as <c>[outer, length, inner]</c>. The first (or, with
    /// <paramref name="selectLast"/>, the last) extreme element wins: the scan starts at element 0 and moves on only for a strictly
    /// better (or, selecting last, an equal or better) value, so NaN never wins unless it is first. Writes <c>outer * inner</c> int64
    /// indices. <paramref name="best"/> is scratch of at least <c>inner</c> elements.
    /// </summary>
    public static void ArgExtreme<T>(bool max, bool selectLast, ReadOnlySpan<T> x, Span<long> y, Span<T> best, int outer, int length, int inner)
        where T : unmanaged, INumber<T>
    {
        if (outer < 0 || length < 0 || inner < 0 || (long)outer * length * inner > x.Length || (long)outer * inner > y.Length || best.Length < inner)
        {
            throw new ArgumentException("A span is shorter than the sizes require.");
        }

        if (length == 0 || inner == 0)
        {
            y.Clear();
            return;
        }

        for (int o = 0; o < outer; o++)
        {
            ReadOnlySpan<T> block = x.Slice(o * length * inner, length * inner);
            Span<long> idx = y.Slice(o * inner, inner);
            block.Slice(0, inner).CopyTo(best);
            idx.Clear();
            for (int r = 1; r < length; r++)
            {
                ReadOnlySpan<T> row = block.Slice(r * inner, inner);
                for (int i = 0; i < inner; i++)
                {
                    T v = row[i], b = best[i];
                    bool better = max ? (selectLast ? v >= b : v > b) : (selectLast ? v <= b : v < b);
                    if (better)
                    {
                        best[i] = v;
                        idx[i] = r;
                    }
                }
            }
        }
    }
}

internal static class ReduceGeometryLimits
{
    /// <summary>Most runs a geometry can hold: alternating kept and reduced runs of a rank-8 shape.</summary>
    public const int MaxRuns = 8;
}
