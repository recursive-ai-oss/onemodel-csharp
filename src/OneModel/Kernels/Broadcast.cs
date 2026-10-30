using System;
using System.Numerics;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Binary element-wise operations with ONNX (NumPy) broadcasting over contiguous row-major operands. The output shape is the
/// broadcast of the two input shapes; each input is read with stride 0 along the dimensions where it has extent 1.
/// </summary>
internal static class Broadcast
{
    /// <summary>
    /// c = op(a, b) with broadcasting. Dimensions that both operands treat alike are merged so the inner loop runs over as many elements
    /// as possible and has stride 0 or 1 for each operand.
    /// </summary>
    /// <param name="path">The session's instruction set.</param>
    /// <param name="canonicalNaN">Rewrite NaN results (exact mode); only floating-point types are affected.</param>
    /// <param name="outShape">The broadcast of <paramref name="aShape"/> and <paramref name="bShape"/>.</param>
    /// <param name="aShape">The shape of <paramref name="a"/>.</param>
    /// <param name="bShape">The shape of <paramref name="b"/>.</param>
    /// <param name="a">Contiguous, aShape.Length elements.</param>
    /// <param name="b">Contiguous, bShape.Length elements.</param>
    /// <param name="c">Contiguous, outShape.Length elements. May share memory with a or b when that operand has the output's shape.</param>
    /// <param name="threads">Upper bound on threads for a large output; the result does not depend on it.</param>
    public static void Binary<T, TRow>(KernelPath path, bool canonicalNaN, Shape outShape, Shape aShape, Shape bShape,
        ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> c, int threads = 1)
        where T : unmanaged, INumber<T>
        where TRow : IRowKernel<T> =>
        Binary<T, T, T, SameTypeRow<T, TRow>>(path, canonicalNaN, outShape, aShape, bShape, a, b, c, threads);

    /// <summary>
    /// The same traversal for operands and result of different element types (comparisons produce bool, Pow mixes base and exponent
    /// types). <typeparamref name="TRow"/> computes one row.
    /// </summary>
    public static void Binary<TA, TB, TC, TRow>(KernelPath path, bool canonicalNaN, Shape outShape, Shape aShape, Shape bShape,
        ReadOnlySpan<TA> a, ReadOnlySpan<TB> b, Span<TC> c, int threads = 1)
        where TA : unmanaged
        where TB : unmanaged
        where TC : unmanaged
        where TRow : IRowKernel<TA, TB, TC>
    {
        int total = outShape.Length;
        if (total == 0)
        {
            return;
        }

        if (a.Length < aShape.Length || b.Length < bShape.Length || c.Length < total)
        {
            throw new ArgumentException("An operand is shorter than its shape.");
        }

        int rank = outShape.Rank;
        Span<int> size = stackalloc int[Shape.MaxRank];
        Span<int> sa = stackalloc int[Shape.MaxRank];
        Span<int> sb = stackalloc int[Shape.MaxRank];
        int accA = 1, accB = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            size[d] = outShape[d];
            int da = d - (rank - aShape.Rank), db = d - (rank - bShape.Rank);
            int ea = da >= 0 ? aShape[da] : 1, eb = db >= 0 ? bShape[db] : 1;
            sa[d] = ea == 1 ? 0 : accA;
            sb[d] = eb == 1 ? 0 : accB;
            accA *= ea;
            accB *= eb;
        }

        // Drop extent-1 dimensions, then merge neighbours that step both operands consistently.
        int m = 0;
        for (int d = 0; d < rank; d++)
        {
            if (size[d] == 1)
            {
                continue;
            }

            if (m > 0 && sa[m - 1] == sa[d] * size[d] && sb[m - 1] == sb[d] * size[d])
            {
                size[m - 1] *= size[d];
                sa[m - 1] = sa[d];
                sb[m - 1] = sb[d];
            }
            else
            {
                size[m] = size[d];
                sa[m] = sa[d];
                sb[m] = sb[d];
                m++;
            }
        }

        if (m == 0)
        {
            TRow.Row(path, canonicalNaN, a, 0, b, 0, c, 1);
            return;
        }

        int inner = size[m - 1];
        int ia = sa[m - 1], ib = sb[m - 1];
        long outer = total / inner;
        if (threads > 1 && total >= 2 * ChunkElements && m <= 2)
        {
            RunParallel<TA, TB, TC, TRow>(path, canonicalNaN, a, b, c, size[0], sa[0], sb[0], m, inner, ia, ib, threads);
            return;
        }

        Span<int> idx = stackalloc int[Shape.MaxRank];
        int pa = 0, pb = 0;
        for (long row = 0; row < outer; row++)
        {
            int po = (int)(row * inner);
            TRow.Row(path, canonicalNaN, a.Slice(pa), ia, b.Slice(pb), ib, c.Slice(po, inner), inner);

            // Advance the outer index like an odometer.
            for (int d = m - 2; d >= 0; d--)
            {
                idx[d]++;
                pa += sa[d];
                pb += sb[d];
                if (idx[d] < size[d])
                {
                    break;
                }

                pa -= sa[d] * size[d];
                pb -= sb[d] * size[d];
                idx[d] = 0;
            }
        }
    }

    // Output elements per chunk of the parallel form: a constant of the kernel, so the split never depends on the thread count.
    private const int ChunkElements = 32768;

    // Large outputs of at most two merged dimensions (same shape, a scalar, a row or a column broadcast): chunks of whole rows when rows
    // are short, of pieces of the one row when it is long. Each output element is computed from its own inputs, so the split changes
    // nothing but who computes what.
    private static unsafe void RunParallel<TA, TB, TC, TRow>(KernelPath path, bool canonicalNaN, ReadOnlySpan<TA> a, ReadOnlySpan<TB> b, Span<TC> c,
        int outerSize, int outerA, int outerB, int m, int inner, int ia, int ib, int threads)
        where TA : unmanaged
        where TB : unmanaged
        where TC : unmanaged
        where TRow : IRowKernel<TA, TB, TC>
    {
        fixed (TA* pa = a)
        fixed (TB* pb = b)
        fixed (TC* pc = c)
        {
            BinaryJob<TA, TB, TC, TRow> job = BinaryJob<TA, TB, TC, TRow>.Instance ??= new BinaryJob<TA, TB, TC, TRow>();
            job.Path = path;
            job.Canonical = canonicalNaN;
            job.A = pa;
            job.B = pb;
            job.C = pc;
            job.ALength = a.Length;
            job.BLength = b.Length;
            job.Rows = m == 2 ? outerSize : 1;
            job.OuterA = m == 2 ? outerA : 0;
            job.OuterB = m == 2 ? outerB : 0;
            job.Inner = inner;
            job.IA = ia;
            job.IB = ib;
            // Short rows go several to a chunk; a long row is cut into pieces of ChunkElements.
            job.PiecesPerRow = inner > ChunkElements ? (inner + ChunkElements - 1) / ChunkElements : 1;
            job.RowsPerChunk = inner >= ChunkElements ? 1 : Math.Max(1, ChunkElements / inner);
            int chunks = job.PiecesPerRow > 1 ? job.Rows * job.PiecesPerRow : (job.Rows + job.RowsPerChunk - 1) / job.RowsPerChunk;
            ParallelFor.Run(job, chunks, threads, (long)job.Rows * inner);
        }
    }

    private sealed unsafe class BinaryJob<TA, TB, TC, TRow> : ChunkJob
        where TA : unmanaged
        where TB : unmanaged
        where TC : unmanaged
        where TRow : IRowKernel<TA, TB, TC>
    {
        [ThreadStatic]
        public static BinaryJob<TA, TB, TC, TRow>? Instance;

        public KernelPath Path;
        public bool Canonical;
        public TA* A;
        public TB* B;
        public TC* C;
        public int ALength;
        public int BLength;
        public int Rows;
        public int OuterA;
        public int OuterB;
        public int Inner;
        public int IA;
        public int IB;
        public int PiecesPerRow;
        public int RowsPerChunk;

        public override void RunChunk(int chunk)
        {
            if (PiecesPerRow > 1)
            {
                int row = chunk / PiecesPerRow;
                int start = (chunk % PiecesPerRow) * ChunkElements;
                int len = Math.Min(ChunkElements, Inner - start);
                Row(row, start, len);
                return;
            }

            int r0 = chunk * RowsPerChunk;
            int r1 = Math.Min(Rows, r0 + RowsPerChunk);
            for (int r = r0; r < r1; r++)
            {
                Row(r, 0, Inner);
            }
        }

        // Elements start..start+len-1 of row r; an operand with step 0 along the row reads its one value.
        private void Row(int r, int start, int len)
        {
            long pa = ((long)r * OuterA) + (IA == 0 ? 0 : start);
            long pb = ((long)r * OuterB) + (IB == 0 ? 0 : start);
            long po = ((long)r * Inner) + start;
            TRow.Row(Path, Canonical,
                new ReadOnlySpan<TA>(A + pa, (int)Math.Min(ALength - pa, len)), IA,
                new ReadOnlySpan<TB>(B + pb, (int)Math.Min(BLength - pb, len)), IB,
                new Span<TC>(C + po, len), len);
        }
    }
}

/// <summary>Computes one row of a broadcast operation on operands and result of any element types.</summary>
internal interface IRowKernel<TA, TB, TC> where TA : unmanaged where TB : unmanaged where TC : unmanaged
{
    static abstract void Row(KernelPath path, bool canonicalNaN, ReadOnlySpan<TA> a, int sa, ReadOnlySpan<TB> b, int sb, Span<TC> c, int n);
}

/// <summary>Adapts a one-type row kernel to the three-type traversal.</summary>
internal readonly struct SameTypeRow<T, TRow> : IRowKernel<T, T, T> where T : unmanaged where TRow : IRowKernel<T>
{
    public static void Row(KernelPath path, bool canonicalNaN, ReadOnlySpan<T> a, int sa, ReadOnlySpan<T> b, int sb, Span<T> c, int n) =>
        TRow.Row(path, canonicalNaN, a, sa, b, sb, c, n);
}

/// <summary>Computes one row of a broadcast operation: n elements, each operand stepping by 0 or 1.</summary>
internal interface IRowKernel<T> where T : unmanaged
{
    static abstract void Row(KernelPath path, bool canonicalNaN, ReadOnlySpan<T> a, int sa, ReadOnlySpan<T> b, int sb, Span<T> c, int n);
}

/// <summary>The vectorized single-precision row.</summary>
internal readonly struct FloatRows<TOp> : IRowKernel<float> where TOp : IBinaryOp
{
    public static void Row(KernelPath path, bool canonicalNaN, ReadOnlySpan<float> a, int sa, ReadOnlySpan<float> b, int sb, Span<float> c, int n) =>
        ExactElementwise.Row<TOp>(path, canonicalNaN, a, sa, b, sb, c, n);
}

/// <summary>A scalar row for every other numeric type. Integers wrap around; doubles get the canonical NaN in exact mode.</summary>
internal readonly struct NumberRows<T, TOp> : IRowKernel<T> where T : unmanaged, INumber<T> where TOp : IBinaryOp
{
    private const long QuietNaN64 = 0x7FF8000000000000L;

    public static void Row(KernelPath path, bool canonicalNaN, ReadOnlySpan<T> a, int sa, ReadOnlySpan<T> b, int sb, Span<T> c, int n)
    {
        for (int i = 0; i < n; i++)
        {
            T r = TOp.Apply(a[sa == 0 ? 0 : i], b[sb == 0 ? 0 : i]);
            if (canonicalNaN && T.IsNaN(r))
            {
                r = T.CreateTruncating(BitConverter.Int64BitsToDouble(QuietNaN64));
            }

            c[i] = r;
        }
    }
}
