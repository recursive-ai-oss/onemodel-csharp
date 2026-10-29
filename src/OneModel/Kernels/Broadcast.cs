using System;
using System.Numerics;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Tensors;

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
    public static void Binary<T, TRow>(KernelPath path, bool canonicalNaN, Shape outShape, Shape aShape, Shape bShape,
        ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> c)
        where T : unmanaged, INumber<T>
        where TRow : IRowKernel<T>
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
