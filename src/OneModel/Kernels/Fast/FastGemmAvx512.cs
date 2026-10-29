using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace RecursiveAI.OneModel.Kernels.Fast;

/// <summary>
/// AVX-512 GEMM microkernel: a 12x32 tile of C is computed over kc steps of k. Every output accumulates its products in increasing k, one
/// separately rounded multiply and one separately rounded add per step, from +0 (or from the C value when accumulating a later
/// k block), so the result does not depend on the tile shape (fused multiply-add, so the rounding is one per step).
/// Bounds the caller guarantees: a points at kc*12 floats (a packed strip, element (p, r) at p*12+r), b at kc*32 floats (element (p, j)
/// at p*32+j), and c at 12 rows of 32 floats, row r at c + r*ldc. Gemm validates the extents of the caller-visible matrices before any tile runs.
/// </summary>
internal static unsafe class FastGemmAvx512
{
    /// <summary>Tile rows.</summary>
    public const int Mr = 12;

    /// <summary>Tile columns.</summary>
    public const int Nr = 32;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Kernel(int kc, float* a, float* b, float* c, nint ldc, int accumulate)
    {
        Vector512<float> c0_0;
        Vector512<float> c0_1;
        Vector512<float> c1_0;
        Vector512<float> c1_1;
        Vector512<float> c2_0;
        Vector512<float> c2_1;
        Vector512<float> c3_0;
        Vector512<float> c3_1;
        Vector512<float> c4_0;
        Vector512<float> c4_1;
        Vector512<float> c5_0;
        Vector512<float> c5_1;
        Vector512<float> c6_0;
        Vector512<float> c6_1;
        Vector512<float> c7_0;
        Vector512<float> c7_1;
        Vector512<float> c8_0;
        Vector512<float> c8_1;
        Vector512<float> c9_0;
        Vector512<float> c9_1;
        Vector512<float> c10_0;
        Vector512<float> c10_1;
        Vector512<float> c11_0;
        Vector512<float> c11_1;
        if (accumulate != 0)
        {
            c0_0 = Avx512F.LoadVector512(c);
            c0_1 = Avx512F.LoadVector512(c + 16);
            c1_0 = Avx512F.LoadVector512(c + 1 * ldc);
            c1_1 = Avx512F.LoadVector512(c + 1 * ldc + 16);
            c2_0 = Avx512F.LoadVector512(c + 2 * ldc);
            c2_1 = Avx512F.LoadVector512(c + 2 * ldc + 16);
            c3_0 = Avx512F.LoadVector512(c + 3 * ldc);
            c3_1 = Avx512F.LoadVector512(c + 3 * ldc + 16);
            c4_0 = Avx512F.LoadVector512(c + 4 * ldc);
            c4_1 = Avx512F.LoadVector512(c + 4 * ldc + 16);
            c5_0 = Avx512F.LoadVector512(c + 5 * ldc);
            c5_1 = Avx512F.LoadVector512(c + 5 * ldc + 16);
            c6_0 = Avx512F.LoadVector512(c + 6 * ldc);
            c6_1 = Avx512F.LoadVector512(c + 6 * ldc + 16);
            c7_0 = Avx512F.LoadVector512(c + 7 * ldc);
            c7_1 = Avx512F.LoadVector512(c + 7 * ldc + 16);
            c8_0 = Avx512F.LoadVector512(c + 8 * ldc);
            c8_1 = Avx512F.LoadVector512(c + 8 * ldc + 16);
            c9_0 = Avx512F.LoadVector512(c + 9 * ldc);
            c9_1 = Avx512F.LoadVector512(c + 9 * ldc + 16);
            c10_0 = Avx512F.LoadVector512(c + 10 * ldc);
            c10_1 = Avx512F.LoadVector512(c + 10 * ldc + 16);
            c11_0 = Avx512F.LoadVector512(c + 11 * ldc);
            c11_1 = Avx512F.LoadVector512(c + 11 * ldc + 16);
        }
        else
        {
            c0_0 = Vector512<float>.Zero;
            c0_1 = Vector512<float>.Zero;
            c1_0 = Vector512<float>.Zero;
            c1_1 = Vector512<float>.Zero;
            c2_0 = Vector512<float>.Zero;
            c2_1 = Vector512<float>.Zero;
            c3_0 = Vector512<float>.Zero;
            c3_1 = Vector512<float>.Zero;
            c4_0 = Vector512<float>.Zero;
            c4_1 = Vector512<float>.Zero;
            c5_0 = Vector512<float>.Zero;
            c5_1 = Vector512<float>.Zero;
            c6_0 = Vector512<float>.Zero;
            c6_1 = Vector512<float>.Zero;
            c7_0 = Vector512<float>.Zero;
            c7_1 = Vector512<float>.Zero;
            c8_0 = Vector512<float>.Zero;
            c8_1 = Vector512<float>.Zero;
            c9_0 = Vector512<float>.Zero;
            c9_1 = Vector512<float>.Zero;
            c10_0 = Vector512<float>.Zero;
            c10_1 = Vector512<float>.Zero;
            c11_0 = Vector512<float>.Zero;
            c11_1 = Vector512<float>.Zero;
        }

        for (int p = 0; p < kc; p++)
        {
            Vector512<float> b0 = Avx512F.LoadVector512(b);
            Vector512<float> b1 = Avx512F.LoadVector512(b + 16);
            Vector512<float> a0 = Vector512.Create(*(a));
            c0_0 = Avx512F.FusedMultiplyAdd(a0, b0, c0_0);
            c0_1 = Avx512F.FusedMultiplyAdd(a0, b1, c0_1);
            Vector512<float> a1 = Vector512.Create(*(a + 1));
            c1_0 = Avx512F.FusedMultiplyAdd(a1, b0, c1_0);
            c1_1 = Avx512F.FusedMultiplyAdd(a1, b1, c1_1);
            Vector512<float> a2 = Vector512.Create(*(a + 2));
            c2_0 = Avx512F.FusedMultiplyAdd(a2, b0, c2_0);
            c2_1 = Avx512F.FusedMultiplyAdd(a2, b1, c2_1);
            Vector512<float> a3 = Vector512.Create(*(a + 3));
            c3_0 = Avx512F.FusedMultiplyAdd(a3, b0, c3_0);
            c3_1 = Avx512F.FusedMultiplyAdd(a3, b1, c3_1);
            Vector512<float> a4 = Vector512.Create(*(a + 4));
            c4_0 = Avx512F.FusedMultiplyAdd(a4, b0, c4_0);
            c4_1 = Avx512F.FusedMultiplyAdd(a4, b1, c4_1);
            Vector512<float> a5 = Vector512.Create(*(a + 5));
            c5_0 = Avx512F.FusedMultiplyAdd(a5, b0, c5_0);
            c5_1 = Avx512F.FusedMultiplyAdd(a5, b1, c5_1);
            Vector512<float> a6 = Vector512.Create(*(a + 6));
            c6_0 = Avx512F.FusedMultiplyAdd(a6, b0, c6_0);
            c6_1 = Avx512F.FusedMultiplyAdd(a6, b1, c6_1);
            Vector512<float> a7 = Vector512.Create(*(a + 7));
            c7_0 = Avx512F.FusedMultiplyAdd(a7, b0, c7_0);
            c7_1 = Avx512F.FusedMultiplyAdd(a7, b1, c7_1);
            Vector512<float> a8 = Vector512.Create(*(a + 8));
            c8_0 = Avx512F.FusedMultiplyAdd(a8, b0, c8_0);
            c8_1 = Avx512F.FusedMultiplyAdd(a8, b1, c8_1);
            Vector512<float> a9 = Vector512.Create(*(a + 9));
            c9_0 = Avx512F.FusedMultiplyAdd(a9, b0, c9_0);
            c9_1 = Avx512F.FusedMultiplyAdd(a9, b1, c9_1);
            Vector512<float> a10 = Vector512.Create(*(a + 10));
            c10_0 = Avx512F.FusedMultiplyAdd(a10, b0, c10_0);
            c10_1 = Avx512F.FusedMultiplyAdd(a10, b1, c10_1);
            Vector512<float> a11 = Vector512.Create(*(a + 11));
            c11_0 = Avx512F.FusedMultiplyAdd(a11, b0, c11_0);
            c11_1 = Avx512F.FusedMultiplyAdd(a11, b1, c11_1);
            a += 12;
            b += 32;
        }

        Avx512F.Store(c, c0_0);
        Avx512F.Store(c + 16, c0_1);
        Avx512F.Store(c + 1 * ldc, c1_0);
        Avx512F.Store(c + 1 * ldc + 16, c1_1);
        Avx512F.Store(c + 2 * ldc, c2_0);
        Avx512F.Store(c + 2 * ldc + 16, c2_1);
        Avx512F.Store(c + 3 * ldc, c3_0);
        Avx512F.Store(c + 3 * ldc + 16, c3_1);
        Avx512F.Store(c + 4 * ldc, c4_0);
        Avx512F.Store(c + 4 * ldc + 16, c4_1);
        Avx512F.Store(c + 5 * ldc, c5_0);
        Avx512F.Store(c + 5 * ldc + 16, c5_1);
        Avx512F.Store(c + 6 * ldc, c6_0);
        Avx512F.Store(c + 6 * ldc + 16, c6_1);
        Avx512F.Store(c + 7 * ldc, c7_0);
        Avx512F.Store(c + 7 * ldc + 16, c7_1);
        Avx512F.Store(c + 8 * ldc, c8_0);
        Avx512F.Store(c + 8 * ldc + 16, c8_1);
        Avx512F.Store(c + 9 * ldc, c9_0);
        Avx512F.Store(c + 9 * ldc + 16, c9_1);
        Avx512F.Store(c + 10 * ldc, c10_0);
        Avx512F.Store(c + 10 * ldc + 16, c10_1);
        Avx512F.Store(c + 11 * ldc, c11_0);
        Avx512F.Store(c + 11 * ldc + 16, c11_1);
    }
}
