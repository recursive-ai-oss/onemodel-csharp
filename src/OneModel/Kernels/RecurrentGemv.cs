using System;
using System.Runtime.Intrinsics;
using RecursiveAI.OneModel.Kernels.Exact;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// The small matrix products of a recurrent step: <c>C[m, n] = A[m, k] · B[k, n]</c> with only a few rows (the batch). The blocked
/// GEMM packs B on every call, which dominates when m is 1; here B is streamed row by row instead, and each output is one accumulator
/// over increasing k from +0 (<c>acc + a·b</c>, unfused in exact mode), the same order as <c>kernels-gemm.md</c>, so exact results
/// equal the exact GEMM's bit for bit.
/// </summary>
internal static class RecurrentGemv
{
    /// <summary>The most rows for which this path is faster than the blocked GEMM.</summary>
    public const int MaxRows = 8;

    /// <summary>Computes C = A · B. Spans must hold <c>(m-1)*ld + extent</c> elements; checked here before any vector access.</summary>
    public static void Run(bool exact, VecIsa isa, int m, int n, int k, ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc)
    {
        if (m < 0 || n < 0 || k < 0 || (m > 0 && (lda < k || ldc < n)) || (k > 0 && ldb < n))
        {
            throw new ArgumentException("Negative size or a stride smaller than the row length.");
        }

        if (m == 0 || n == 0)
        {
            return;
        }

        long aNeed = ((long)(m - 1) * lda) + k;
        long bNeed = k == 0 ? 0 : ((long)(k - 1) * ldb) + n;
        long cNeed = ((long)(m - 1) * ldc) + n;
        if (a.Length < aNeed || b.Length < bNeed || c.Length < cNeed)
        {
            throw new ArgumentException("A span is shorter than the sizes and strides require.");
        }

        isa = VecIsaInfo.Resolve(isa);
        if (exact)
        {
            switch (isa)
            {
                case VecIsa.V512: Core<Vector512<float>, ExactVec512>(m, n, k, a, lda, b, ldb, c, ldc); break;
                case VecIsa.V256: Core<Vector256<float>, ExactVec256>(m, n, k, a, lda, b, ldb, c, ldc); break;
                case VecIsa.V128: Core<Vector128<float>, ExactVec128>(m, n, k, a, lda, b, ldb, c, ldc); break;
                default: Core<float, ExactVecScalar>(m, n, k, a, lda, b, ldb, c, ldc); break;
            }
        }
        else
        {
            switch (isa)
            {
                case VecIsa.V512: Core<Vector512<float>, FastVec512>(m, n, k, a, lda, b, ldb, c, ldc); break;
                case VecIsa.V256: Core<Vector256<float>, FastVec256>(m, n, k, a, lda, b, ldb, c, ldc); break;
                case VecIsa.V128: Core<Vector128<float>, FastVec128>(m, n, k, a, lda, b, ldb, c, ldc); break;
                default: Core<float, FastVecScalar>(m, n, k, a, lda, b, ldb, c, ldc); break;
            }
        }
    }

    private static void Core<TV, TO>(int m, int n, int k, ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc)
        where TV : struct
        where TO : IVecOps<TV>
    {
        int lanes = TO.Lanes;

        // Bounds: Run checked that every a[i*lda + kk], b[kk*ldb + j .. + lanes) and c[i*ldc + j .. + lanes) used below lies inside the
        // spans, because j + lanes <= n in every vector access.
        for (int i = 0; i < m; i++)
        {
            ReadOnlySpan<float> arow = a.Slice(i * lda, k);
            Span<float> crow = c.Slice(i * ldc, n);
            int j = 0;

            // Four column vectors at a time: four independent accumulator chains hide the add latency.
            for (; j + (4 * lanes) <= n; j += 4 * lanes)
            {
                TV c0 = TO.Zero, c1 = TO.Zero, c2 = TO.Zero, c3 = TO.Zero;
                for (int kk = 0; kk < k; kk++)
                {
                    TV x = TO.Splat(arow[kk]);
                    int at = (kk * ldb) + j;
                    c0 = TO.Accumulate(c0, x, TO.Load(b, at));
                    c1 = TO.Accumulate(c1, x, TO.Load(b, at + lanes));
                    c2 = TO.Accumulate(c2, x, TO.Load(b, at + (2 * lanes)));
                    c3 = TO.Accumulate(c3, x, TO.Load(b, at + (3 * lanes)));
                }

                TO.Store(crow, j, c0);
                TO.Store(crow, j + lanes, c1);
                TO.Store(crow, j + (2 * lanes), c2);
                TO.Store(crow, j + (3 * lanes), c3);
            }

            for (; j + lanes <= n; j += lanes)
            {
                TV c0 = TO.Zero;
                for (int kk = 0; kk < k; kk++)
                {
                    c0 = TO.Accumulate(c0, TO.Splat(arow[kk]), TO.Load(b, (kk * ldb) + j));
                }

                TO.Store(crow, j, c0);
            }

            // Columns left over: the same accumulation, one at a time, product rounded before the sum.
            for (; j < n; j++)
            {
                float acc = 0f;
                for (int kk = 0; kk < k; kk++)
                {
                    float p = arow[kk] * b[(kk * ldb) + j];
                    acc += p;
                }

                crow[j] = acc;
            }
        }
    }
}
