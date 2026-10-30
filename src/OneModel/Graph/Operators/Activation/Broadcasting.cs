using System;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Activation;

/// <summary>One element of a binary operation on possibly different types.</summary>
internal interface IElementFn<TA, TB, TC>
{
    static abstract TC Apply(TA a, TB b);
}

/// <summary>
/// A plain odometer loop for binary operations with NumPy broadcasting. These operators are not hot paths, so the loop favours
/// being obviously right over vector width; the fast element-wise kernels live in Kernels/Broadcast.cs.
/// </summary>
internal static class Broadcasting
{
    public static void Binary<TA, TB, TC, TF>(Shape outShape, Shape aShape, Shape bShape, ReadOnlySpan<TA> a, ReadOnlySpan<TB> b, Span<TC> c)
        where TF : IElementFn<TA, TB, TC>
    {
        int rank = outShape.Rank;
        Span<int> sa = stackalloc int[Shape.MaxRank];
        Span<int> sb = stackalloc int[Shape.MaxRank];
        Span<int> idx = stackalloc int[Shape.MaxRank];
        Strides(aShape, rank, sa);
        Strides(bShape, rank, sb);
        int total = outShape.Length;
        int oa = 0, ob = 0;
        for (int i = 0; i < total; i++)
        {
            c[i] = TF.Apply(a[oa], b[ob]);
            for (int d = rank - 1; d >= 0; d--)
            {
                idx[d]++;
                oa += sa[d];
                ob += sb[d];
                if (idx[d] < outShape[d])
                {
                    break;
                }

                oa -= sa[d] * outShape[d];
                ob -= sb[d] * outShape[d];
                idx[d] = 0;
            }
        }
    }

    /// <summary>Element strides of a contiguous input right-aligned to <paramref name="rank"/> dimensions; 0 where it is broadcast.</summary>
    private static void Strides(Shape s, int rank, Span<int> strides)
    {
        int run = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            int k = d - (rank - s.Rank);
            if (k < 0)
            {
                strides[d] = 0;
                continue;
            }

            strides[d] = s[k] == 1 ? 0 : run;
            run *= s[k];
        }
    }
}
