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

    /// <summary>
    /// Computes C = A · B. Spans must hold <c>(m-1)*ld + extent</c> elements; checked here before any vector access. With
    /// <paramref name="groups"/> above 1 the product is formed for that many column blocks of width <paramref name="n"/> at once, block g
    /// starting at column <c>g * groupStride</c> of both B and C (the gates of one slice of hidden units): the blocks are independent, and
    /// handing them over together lets a single row keep more accumulator chains in flight.
    /// </summary>
    public static void Run(bool exact, VecIsa isa, int m, int n, int k, ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc,
        int groups = 1, int groupStride = 0)
    {
        if (m < 0 || n < 0 || k < 0 || groups < 1 || groups > 4 || (groups > 1 && groupStride < n) || (m > 0 && (lda < k || ldc < n)) || (k > 0 && ldb < n))
        {
            throw new ArgumentException("Negative size or a stride smaller than the row length.");
        }

        if (m == 0 || n == 0)
        {
            return;
        }

        long aNeed = ((long)(m - 1) * lda) + k;
        long span = (long)(groups - 1) * groupStride;
        long bNeed = k == 0 ? 0 : ((long)(k - 1) * ldb) + span + n;
        long cNeed = ((long)(m - 1) * ldc) + span + n;
        if (a.Length < aNeed || b.Length < bNeed || c.Length < cNeed)
        {
            throw new ArgumentException("A span is shorter than the sizes and strides require.");
        }

        isa = VecIsaInfo.Resolve(isa);
        if (exact)
        {
            switch (isa)
            {
                case VecIsa.V512: Core<Vector512<float>, ExactVec512>(m, n, k, a, lda, b, ldb, c, ldc, groups, groupStride); break;
                case VecIsa.V256: Core<Vector256<float>, ExactVec256>(m, n, k, a, lda, b, ldb, c, ldc, groups, groupStride); break;
                case VecIsa.V128: Core<Vector128<float>, ExactVec128>(m, n, k, a, lda, b, ldb, c, ldc, groups, groupStride); break;
                default: Core<float, ExactVecScalar>(m, n, k, a, lda, b, ldb, c, ldc, groups, groupStride); break;
            }
        }
        else
        {
            switch (isa)
            {
                case VecIsa.V512: Core<Vector512<float>, FastVec512>(m, n, k, a, lda, b, ldb, c, ldc, groups, groupStride); break;
                case VecIsa.V256: Core<Vector256<float>, FastVec256>(m, n, k, a, lda, b, ldb, c, ldc, groups, groupStride); break;
                case VecIsa.V128: Core<Vector128<float>, FastVec128>(m, n, k, a, lda, b, ldb, c, ldc, groups, groupStride); break;
                default: Core<float, FastVecScalar>(m, n, k, a, lda, b, ldb, c, ldc, groups, groupStride); break;
            }
        }
    }

    private static void Core<TV, TO>(int m, int n, int k, ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc, int groups, int gs)
        where TV : struct
        where TO : IVecOps<TV>
    {
        int lanes = TO.Lanes;

        // Bounds: Run checked that every a[i*lda + kk], b[kk*ldb + j .. + lanes) and c[i*ldc + j .. + lanes) used below lies inside the
        // spans, because j + lanes <= n in every vector access.
        int i = 0;

        // Four rows at a time: each loaded vector of B serves four rows, so B is streamed once per four rows instead of once per row.
        // Every output is still one accumulator over increasing kk from +0, so the bits equal the one-row loop's.
        for (; i + 4 <= m; i += 4)
        {
            for (int g = 0; g < groups; g++)
            {
                Rows4<TV, TO>(i, n, k, a, lda, b.Slice(g * gs), ldb, c.Slice(g * gs), ldc);
            }
        }

        for (; i < m; i++)
        {
            if (groups > 1 && n % (2 * lanes) == 0)
            {
                Row1Blocks<TV, TO>(i, n, k, a, lda, b, ldb, c, ldc, groups, gs);
                continue;
            }

            for (int g = 0; g < groups; g++)
            {
                Row1<TV, TO>(i, n, k, a, lda, b.Slice(g * gs), ldb, c.Slice(g * gs), ldc);
            }
        }
    }

    // One row of the product for one column block: four vectors of columns at a time, then one, then single columns.
    private static void Row1<TV, TO>(int i, int n, int k, ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc)
        where TV : struct
        where TO : IVecOps<TV>
    {
        int lanes = TO.Lanes;
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

    // One row over up to four column blocks at once: 4 blocks x 2 vectors = 8 independent accumulator chains, where one block alone has
    // two or four. Blocks past `groups` read block 0 and are not stored (a wasted quarter or half at most). n is a multiple of 2 * lanes.
    private static void Row1Blocks<TV, TO>(int i, int n, int k, ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc,
        int groups, int gs)
        where TV : struct
        where TO : IVecOps<TV>
    {
        int lanes = TO.Lanes;
        ReadOnlySpan<float> arow = a.Slice(i * lda, k);
        Span<float> crow = c.Slice(i * ldc, n + ((groups - 1) * gs));
        int o1 = groups > 1 ? gs : 0, o2 = groups > 2 ? 2 * gs : 0, o3 = groups > 3 ? 3 * gs : 0;
        for (int j = 0; j < n; j += 2 * lanes)
        {
            TV c00 = TO.Zero, c01 = TO.Zero, c10 = TO.Zero, c11 = TO.Zero, c20 = TO.Zero, c21 = TO.Zero, c30 = TO.Zero, c31 = TO.Zero;
            for (int kk = 0; kk < k; kk++)
            {
                TV x = TO.Splat(arow[kk]);
                int at = (kk * ldb) + j;
                c00 = TO.Accumulate(c00, x, TO.Load(b, at));
                c01 = TO.Accumulate(c01, x, TO.Load(b, at + lanes));
                c10 = TO.Accumulate(c10, x, TO.Load(b, at + o1));
                c11 = TO.Accumulate(c11, x, TO.Load(b, at + o1 + lanes));
                c20 = TO.Accumulate(c20, x, TO.Load(b, at + o2));
                c21 = TO.Accumulate(c21, x, TO.Load(b, at + o2 + lanes));
                c30 = TO.Accumulate(c30, x, TO.Load(b, at + o3));
                c31 = TO.Accumulate(c31, x, TO.Load(b, at + o3 + lanes));
            }

            TO.Store(crow, j, c00);
            TO.Store(crow, j + lanes, c01);
            if (groups > 1)
            {
                TO.Store(crow, j + gs, c10);
                TO.Store(crow, j + gs + lanes, c11);
            }

            if (groups > 2)
            {
                TO.Store(crow, j + (2 * gs), c20);
                TO.Store(crow, j + (2 * gs) + lanes, c21);
            }

            if (groups > 3)
            {
                TO.Store(crow, j + (3 * gs), c30);
                TO.Store(crow, j + (3 * gs) + lanes, c31);
            }
        }
    }

    // Rows i0..i0+3 of the product. Register tile: 4 rows x 2 vectors of columns (8 accumulators, 2 vectors of B, one broadcast).
    private static void Rows4<TV, TO>(int i0, int n, int k, ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc)
        where TV : struct
        where TO : IVecOps<TV>
    {
        int lanes = TO.Lanes;
        ReadOnlySpan<float> a0 = a.Slice(i0 * lda, k);
        ReadOnlySpan<float> a1 = a.Slice((i0 + 1) * lda, k);
        ReadOnlySpan<float> a2 = a.Slice((i0 + 2) * lda, k);
        ReadOnlySpan<float> a3 = a.Slice((i0 + 3) * lda, k);
        Span<float> c0r = c.Slice(i0 * ldc, n);
        Span<float> c1r = c.Slice((i0 + 1) * ldc, n);
        Span<float> c2r = c.Slice((i0 + 2) * ldc, n);
        Span<float> c3r = c.Slice((i0 + 3) * ldc, n);
        int j = 0;

        // Four-lane vectors (NEON, SSE): sixteen accumulators hide the fused multiply-add latency of cores with four pipes.
        for (; lanes == 4 && j + 16 <= n; j += 16)
        {
            TV c00 = TO.Zero, c01 = TO.Zero, c02 = TO.Zero, c03 = TO.Zero, c10 = TO.Zero, c11 = TO.Zero, c12 = TO.Zero, c13 = TO.Zero;
            TV c20 = TO.Zero, c21 = TO.Zero, c22 = TO.Zero, c23 = TO.Zero, c30 = TO.Zero, c31 = TO.Zero, c32 = TO.Zero, c33 = TO.Zero;
            for (int kk = 0; kk < k; kk++)
            {
                int at = (kk * ldb) + j;
                TV b0 = TO.Load(b, at);
                TV b1 = TO.Load(b, at + 4);
                TV b2 = TO.Load(b, at + 8);
                TV b3 = TO.Load(b, at + 12);
                TV x = TO.Splat(a0[kk]);
                c00 = TO.Accumulate(c00, x, b0);
                c01 = TO.Accumulate(c01, x, b1);
                c02 = TO.Accumulate(c02, x, b2);
                c03 = TO.Accumulate(c03, x, b3);
                x = TO.Splat(a1[kk]);
                c10 = TO.Accumulate(c10, x, b0);
                c11 = TO.Accumulate(c11, x, b1);
                c12 = TO.Accumulate(c12, x, b2);
                c13 = TO.Accumulate(c13, x, b3);
                x = TO.Splat(a2[kk]);
                c20 = TO.Accumulate(c20, x, b0);
                c21 = TO.Accumulate(c21, x, b1);
                c22 = TO.Accumulate(c22, x, b2);
                c23 = TO.Accumulate(c23, x, b3);
                x = TO.Splat(a3[kk]);
                c30 = TO.Accumulate(c30, x, b0);
                c31 = TO.Accumulate(c31, x, b1);
                c32 = TO.Accumulate(c32, x, b2);
                c33 = TO.Accumulate(c33, x, b3);
            }

            TO.Store(c0r, j, c00);
            TO.Store(c0r, j + 4, c01);
            TO.Store(c0r, j + 8, c02);
            TO.Store(c0r, j + 12, c03);
            TO.Store(c1r, j, c10);
            TO.Store(c1r, j + 4, c11);
            TO.Store(c1r, j + 8, c12);
            TO.Store(c1r, j + 12, c13);
            TO.Store(c2r, j, c20);
            TO.Store(c2r, j + 4, c21);
            TO.Store(c2r, j + 8, c22);
            TO.Store(c2r, j + 12, c23);
            TO.Store(c3r, j, c30);
            TO.Store(c3r, j + 4, c31);
            TO.Store(c3r, j + 8, c32);
            TO.Store(c3r, j + 12, c33);
        }

        for (; j + (2 * lanes) <= n; j += 2 * lanes)
        {
            TV c00 = TO.Zero, c01 = TO.Zero, c10 = TO.Zero, c11 = TO.Zero, c20 = TO.Zero, c21 = TO.Zero, c30 = TO.Zero, c31 = TO.Zero;
            for (int kk = 0; kk < k; kk++)
            {
                int at = (kk * ldb) + j;
                TV b0 = TO.Load(b, at);
                TV b1 = TO.Load(b, at + lanes);
                TV x = TO.Splat(a0[kk]);
                c00 = TO.Accumulate(c00, x, b0);
                c01 = TO.Accumulate(c01, x, b1);
                x = TO.Splat(a1[kk]);
                c10 = TO.Accumulate(c10, x, b0);
                c11 = TO.Accumulate(c11, x, b1);
                x = TO.Splat(a2[kk]);
                c20 = TO.Accumulate(c20, x, b0);
                c21 = TO.Accumulate(c21, x, b1);
                x = TO.Splat(a3[kk]);
                c30 = TO.Accumulate(c30, x, b0);
                c31 = TO.Accumulate(c31, x, b1);
            }

            TO.Store(c0r, j, c00);
            TO.Store(c0r, j + lanes, c01);
            TO.Store(c1r, j, c10);
            TO.Store(c1r, j + lanes, c11);
            TO.Store(c2r, j, c20);
            TO.Store(c2r, j + lanes, c21);
            TO.Store(c3r, j, c30);
            TO.Store(c3r, j + lanes, c31);
        }

        for (; j + lanes <= n; j += lanes)
        {
            TV c0 = TO.Zero, c1 = TO.Zero, c2 = TO.Zero, c3 = TO.Zero;
            for (int kk = 0; kk < k; kk++)
            {
                TV b0 = TO.Load(b, (kk * ldb) + j);
                c0 = TO.Accumulate(c0, TO.Splat(a0[kk]), b0);
                c1 = TO.Accumulate(c1, TO.Splat(a1[kk]), b0);
                c2 = TO.Accumulate(c2, TO.Splat(a2[kk]), b0);
                c3 = TO.Accumulate(c3, TO.Splat(a3[kk]), b0);
            }

            TO.Store(c0r, j, c0);
            TO.Store(c1r, j, c1);
            TO.Store(c2r, j, c2);
            TO.Store(c3r, j, c3);
        }

        // Columns left over, as in the one-row loop: product rounded before the sum.
        for (; j < n; j++)
        {
            float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
            for (int kk = 0; kk < k; kk++)
            {
                float bv = b[(kk * ldb) + j];
                float p0 = a0[kk] * bv;
                s0 += p0;
                float p1 = a1[kk] * bv;
                s1 += p1;
                float p2 = a2[kk] * bv;
                s2 += p2;
                float p3 = a3[kk] * bv;
                s3 += p3;
            }

            c0r[j] = s0;
            c1r[j] = s1;
            c2r[j] = s2;
            c3r[j] = s3;
        }
    }
}
