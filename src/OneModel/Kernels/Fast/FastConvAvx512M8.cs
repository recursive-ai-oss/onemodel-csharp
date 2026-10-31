using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace RecursiveAI.OneModel.Kernels.Fast;

/// <summary>
/// AVX-512 convolution microkernel: an 8x32 tile of C over kc steps of k. Eight rows suit output-channel counts that are
/// multiples of 8 but not of 16 (24, 40, 72, 88, 120, ...).
/// Every output accumulates its products in increasing k with one fused multiply-add per step, from +0 (or from the C value when
/// accumulating a later k block), the same order as every other tile shape, so the result does not depend on the tile shape.
/// Bounds the caller guarantees: a points at kc*8 floats (a packed strip, element (p, r) at p*8+r), b at kc*32 floats (element (p, j)
/// at p*32+j), and c at 8 rows of 32 floats, row r at c + r*ldc. The convolution driver validates the extents before any tile runs.
/// </summary>
internal static unsafe class FastConvAvx512M8
{
    /// <summary>Tile rows.</summary>
    public const int Mr = 8;

    /// <summary>Tile columns.</summary>
    public const int Nr = 32;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Kernel(int kc, float* a, float* b, float* c, nint ldc, int accumulate)
    {
        Vector512<float> c0_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 0 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c0_1 = accumulate != 0 ? Avx512F.LoadVector512(c + 0 * ldc + 16) : Vector512<float>.Zero;
        Vector512<float> c1_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 1 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c1_1 = accumulate != 0 ? Avx512F.LoadVector512(c + 1 * ldc + 16) : Vector512<float>.Zero;
        Vector512<float> c2_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 2 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c2_1 = accumulate != 0 ? Avx512F.LoadVector512(c + 2 * ldc + 16) : Vector512<float>.Zero;
        Vector512<float> c3_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 3 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c3_1 = accumulate != 0 ? Avx512F.LoadVector512(c + 3 * ldc + 16) : Vector512<float>.Zero;
        Vector512<float> c4_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 4 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c4_1 = accumulate != 0 ? Avx512F.LoadVector512(c + 4 * ldc + 16) : Vector512<float>.Zero;
        Vector512<float> c5_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 5 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c5_1 = accumulate != 0 ? Avx512F.LoadVector512(c + 5 * ldc + 16) : Vector512<float>.Zero;
        Vector512<float> c6_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 6 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c6_1 = accumulate != 0 ? Avx512F.LoadVector512(c + 6 * ldc + 16) : Vector512<float>.Zero;
        Vector512<float> c7_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 7 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c7_1 = accumulate != 0 ? Avx512F.LoadVector512(c + 7 * ldc + 16) : Vector512<float>.Zero;

        for (int p = 0; p < kc; p++)
        {
            Vector512<float> b0 = Avx512F.LoadVector512(b + 0);
            Vector512<float> b1 = Avx512F.LoadVector512(b + 16);
            Vector512<float> a0 = Vector512.Create(*(a + 0));
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
            a += 8;
            b += 32;
        }

        Avx512F.Store(c + 0 * ldc + 0, c0_0);
        Avx512F.Store(c + 0 * ldc + 16, c0_1);
        Avx512F.Store(c + 1 * ldc + 0, c1_0);
        Avx512F.Store(c + 1 * ldc + 16, c1_1);
        Avx512F.Store(c + 2 * ldc + 0, c2_0);
        Avx512F.Store(c + 2 * ldc + 16, c2_1);
        Avx512F.Store(c + 3 * ldc + 0, c3_0);
        Avx512F.Store(c + 3 * ldc + 16, c3_1);
        Avx512F.Store(c + 4 * ldc + 0, c4_0);
        Avx512F.Store(c + 4 * ldc + 16, c4_1);
        Avx512F.Store(c + 5 * ldc + 0, c5_0);
        Avx512F.Store(c + 5 * ldc + 16, c5_1);
        Avx512F.Store(c + 6 * ldc + 0, c6_0);
        Avx512F.Store(c + 6 * ldc + 16, c6_1);
        Avx512F.Store(c + 7 * ldc + 0, c7_0);
        Avx512F.Store(c + 7 * ldc + 16, c7_1);
    }

    /// <summary>
    /// The same tile with B read in place instead of from a packed strip: element (p, j) at b + p*ldb + j, so a 1x1 convolution reads its
    /// input planes directly. Reads 32 floats at every row p of kc, which the caller keeps inside the plane (full strips only).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void KernelStrided(int kc, float* a, float* b, nint ldb, float* c, nint ldc, int accumulate)
    {
        Vector512<float> c0_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 0 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c0_1 = accumulate != 0 ? Avx512F.LoadVector512(c + 0 * ldc + 16) : Vector512<float>.Zero;
        Vector512<float> c1_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 1 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c1_1 = accumulate != 0 ? Avx512F.LoadVector512(c + 1 * ldc + 16) : Vector512<float>.Zero;
        Vector512<float> c2_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 2 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c2_1 = accumulate != 0 ? Avx512F.LoadVector512(c + 2 * ldc + 16) : Vector512<float>.Zero;
        Vector512<float> c3_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 3 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c3_1 = accumulate != 0 ? Avx512F.LoadVector512(c + 3 * ldc + 16) : Vector512<float>.Zero;
        Vector512<float> c4_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 4 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c4_1 = accumulate != 0 ? Avx512F.LoadVector512(c + 4 * ldc + 16) : Vector512<float>.Zero;
        Vector512<float> c5_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 5 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c5_1 = accumulate != 0 ? Avx512F.LoadVector512(c + 5 * ldc + 16) : Vector512<float>.Zero;
        Vector512<float> c6_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 6 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c6_1 = accumulate != 0 ? Avx512F.LoadVector512(c + 6 * ldc + 16) : Vector512<float>.Zero;
        Vector512<float> c7_0 = accumulate != 0 ? Avx512F.LoadVector512(c + 7 * ldc + 0) : Vector512<float>.Zero;
        Vector512<float> c7_1 = accumulate != 0 ? Avx512F.LoadVector512(c + 7 * ldc + 16) : Vector512<float>.Zero;

        for (int p = 0; p < kc; p++)
        {
            Vector512<float> b0 = Avx512F.LoadVector512(b + 0);
            Vector512<float> b1 = Avx512F.LoadVector512(b + 16);
            Vector512<float> a0 = Vector512.Create(*(a + 0));
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
            a += 8;
            b += ldb;
        }

        Avx512F.Store(c + 0 * ldc + 0, c0_0);
        Avx512F.Store(c + 0 * ldc + 16, c0_1);
        Avx512F.Store(c + 1 * ldc + 0, c1_0);
        Avx512F.Store(c + 1 * ldc + 16, c1_1);
        Avx512F.Store(c + 2 * ldc + 0, c2_0);
        Avx512F.Store(c + 2 * ldc + 16, c2_1);
        Avx512F.Store(c + 3 * ldc + 0, c3_0);
        Avx512F.Store(c + 3 * ldc + 16, c3_1);
        Avx512F.Store(c + 4 * ldc + 0, c4_0);
        Avx512F.Store(c + 4 * ldc + 16, c4_1);
        Avx512F.Store(c + 5 * ldc + 0, c5_0);
        Avx512F.Store(c + 5 * ldc + 16, c5_1);
        Avx512F.Store(c + 6 * ldc + 0, c6_0);
        Avx512F.Store(c + 6 * ldc + 16, c6_1);
        Avx512F.Store(c + 7 * ldc + 0, c7_0);
        Avx512F.Store(c + 7 * ldc + 16, c7_1);
    }
}
