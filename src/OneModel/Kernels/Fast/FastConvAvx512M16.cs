using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace RecursiveAI.OneModel.Kernels.Fast;

/// <summary>
/// AVX-512 convolution microkernel: a 16x16 tile of C over kc steps of k. Sixteen rows suit output-channel counts that are
/// multiples of 8 or 16, and sixteen columns waste little on small feature maps.
/// Every output accumulates its products in increasing k with one fused multiply-add per step, from +0 (or from the C value when
/// accumulating a later k block), the same order as every other tile shape, so the result does not depend on the tile shape.
/// Bounds the caller guarantees: a points at kc*16 floats (a packed strip, element (p, r) at p*16+r), b at kc*16 floats (element (p, j)
/// at p*16+j), and c at 16 rows of 16 floats, row r at c + r*ldc. The convolution driver validates the extents before any tile runs.
/// </summary>
internal static unsafe class FastConvAvx512M16
{
    /// <summary>Tile rows.</summary>
    public const int Mr = 16;

    /// <summary>Tile columns.</summary>
    public const int Nr = 16;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Kernel(int kc, float* a, float* b, float* c, nint ldc, int accumulate)
    {
        Vector512<float> c0_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 0 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c1_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 1 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c2_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 2 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c3_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 3 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c4_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 4 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c5_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 5 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c6_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 6 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c7_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 7 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c8_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 8 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c9_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 9 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c10_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 10 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c11_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 11 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c12_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 12 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c13_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 13 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c14_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 14 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c15_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 15 * ldc + 0) : Vector512<float>.Zero;

        for (int p = 0; p < kc; p++)
        {
            Vector512<float> b0 = Avx512F.LoadVector512(b + 0);
            Vector512<float> a0 = Vector512.Create(*(a + 0));
            c0_0 = Avx512F.FusedMultiplyAdd(a0, b0, c0_0);
            Vector512<float> a1 = Vector512.Create(*(a + 1));
            c1_0 = Avx512F.FusedMultiplyAdd(a1, b0, c1_0);
            Vector512<float> a2 = Vector512.Create(*(a + 2));
            c2_0 = Avx512F.FusedMultiplyAdd(a2, b0, c2_0);
            Vector512<float> a3 = Vector512.Create(*(a + 3));
            c3_0 = Avx512F.FusedMultiplyAdd(a3, b0, c3_0);
            Vector512<float> a4 = Vector512.Create(*(a + 4));
            c4_0 = Avx512F.FusedMultiplyAdd(a4, b0, c4_0);
            Vector512<float> a5 = Vector512.Create(*(a + 5));
            c5_0 = Avx512F.FusedMultiplyAdd(a5, b0, c5_0);
            Vector512<float> a6 = Vector512.Create(*(a + 6));
            c6_0 = Avx512F.FusedMultiplyAdd(a6, b0, c6_0);
            Vector512<float> a7 = Vector512.Create(*(a + 7));
            c7_0 = Avx512F.FusedMultiplyAdd(a7, b0, c7_0);
            Vector512<float> a8 = Vector512.Create(*(a + 8));
            c8_0 = Avx512F.FusedMultiplyAdd(a8, b0, c8_0);
            Vector512<float> a9 = Vector512.Create(*(a + 9));
            c9_0 = Avx512F.FusedMultiplyAdd(a9, b0, c9_0);
            Vector512<float> a10 = Vector512.Create(*(a + 10));
            c10_0 = Avx512F.FusedMultiplyAdd(a10, b0, c10_0);
            Vector512<float> a11 = Vector512.Create(*(a + 11));
            c11_0 = Avx512F.FusedMultiplyAdd(a11, b0, c11_0);
            Vector512<float> a12 = Vector512.Create(*(a + 12));
            c12_0 = Avx512F.FusedMultiplyAdd(a12, b0, c12_0);
            Vector512<float> a13 = Vector512.Create(*(a + 13));
            c13_0 = Avx512F.FusedMultiplyAdd(a13, b0, c13_0);
            Vector512<float> a14 = Vector512.Create(*(a + 14));
            c14_0 = Avx512F.FusedMultiplyAdd(a14, b0, c14_0);
            Vector512<float> a15 = Vector512.Create(*(a + 15));
            c15_0 = Avx512F.FusedMultiplyAdd(a15, b0, c15_0);
            a += 16;
            b += 16;
        }

        Avx512F.Store(c + 0 * ldc + 0, c0_0);
        Avx512F.Store(c + 1 * ldc + 0, c1_0);
        Avx512F.Store(c + 2 * ldc + 0, c2_0);
        Avx512F.Store(c + 3 * ldc + 0, c3_0);
        Avx512F.Store(c + 4 * ldc + 0, c4_0);
        Avx512F.Store(c + 5 * ldc + 0, c5_0);
        Avx512F.Store(c + 6 * ldc + 0, c6_0);
        Avx512F.Store(c + 7 * ldc + 0, c7_0);
        Avx512F.Store(c + 8 * ldc + 0, c8_0);
        Avx512F.Store(c + 9 * ldc + 0, c9_0);
        Avx512F.Store(c + 10 * ldc + 0, c10_0);
        Avx512F.Store(c + 11 * ldc + 0, c11_0);
        Avx512F.Store(c + 12 * ldc + 0, c12_0);
        Avx512F.Store(c + 13 * ldc + 0, c13_0);
        Avx512F.Store(c + 14 * ldc + 0, c14_0);
        Avx512F.Store(c + 15 * ldc + 0, c15_0);
    }
}
