using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace RecursiveAI.OneModel.Kernels.Exact;

/// <summary>
/// AVX2 GEMM microkernel: a 6x16 tile of C is computed over kc steps of k. Every output accumulates its products in increasing k, one
/// separately rounded multiply and one separately rounded add per step, from +0 (or from the C value when accumulating a later
/// k block), so the result does not depend on the tile shape (no fused multiply-add).
/// Bounds the caller guarantees: a points at kc*6 floats (a packed strip, element (p, r) at p*6+r), b at kc*16 floats (element (p, j)
/// at p*16+j), and c at 6 rows of 16 floats, row r at c + r*ldc. Gemm validates the extents of the caller-visible matrices before any tile runs.
/// </summary>
internal static unsafe class ExactGemmAvx2
{
    /// <summary>Tile rows.</summary>
    public const int Mr = 6;

    /// <summary>Tile columns.</summary>
    public const int Nr = 16;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Kernel(int kc, float* a, float* b, float* c, nint ldc, int accumulate)
    {
        Vector256<float> c0_0;
        Vector256<float> c0_1;
        Vector256<float> c1_0;
        Vector256<float> c1_1;
        Vector256<float> c2_0;
        Vector256<float> c2_1;
        Vector256<float> c3_0;
        Vector256<float> c3_1;
        Vector256<float> c4_0;
        Vector256<float> c4_1;
        Vector256<float> c5_0;
        Vector256<float> c5_1;
        if (accumulate != 0)
        {
            c0_0 = Avx.LoadVector256(c);
            c0_1 = Avx.LoadVector256(c + 8);
            c1_0 = Avx.LoadVector256(c + 1 * ldc);
            c1_1 = Avx.LoadVector256(c + 1 * ldc + 8);
            c2_0 = Avx.LoadVector256(c + 2 * ldc);
            c2_1 = Avx.LoadVector256(c + 2 * ldc + 8);
            c3_0 = Avx.LoadVector256(c + 3 * ldc);
            c3_1 = Avx.LoadVector256(c + 3 * ldc + 8);
            c4_0 = Avx.LoadVector256(c + 4 * ldc);
            c4_1 = Avx.LoadVector256(c + 4 * ldc + 8);
            c5_0 = Avx.LoadVector256(c + 5 * ldc);
            c5_1 = Avx.LoadVector256(c + 5 * ldc + 8);
        }
        else
        {
            c0_0 = Vector256<float>.Zero;
            c0_1 = Vector256<float>.Zero;
            c1_0 = Vector256<float>.Zero;
            c1_1 = Vector256<float>.Zero;
            c2_0 = Vector256<float>.Zero;
            c2_1 = Vector256<float>.Zero;
            c3_0 = Vector256<float>.Zero;
            c3_1 = Vector256<float>.Zero;
            c4_0 = Vector256<float>.Zero;
            c4_1 = Vector256<float>.Zero;
            c5_0 = Vector256<float>.Zero;
            c5_1 = Vector256<float>.Zero;
        }

        for (int p = 0; p < kc; p++)
        {
            Vector256<float> b0 = Avx.LoadVector256(b);
            Vector256<float> b1 = Avx.LoadVector256(b + 8);
            Vector256<float> a0 = Avx.BroadcastScalarToVector256(a);
            c0_0 = Avx.Add(c0_0, Avx.Multiply(a0, b0));
            c0_1 = Avx.Add(c0_1, Avx.Multiply(a0, b1));
            Vector256<float> a1 = Avx.BroadcastScalarToVector256(a + 1);
            c1_0 = Avx.Add(c1_0, Avx.Multiply(a1, b0));
            c1_1 = Avx.Add(c1_1, Avx.Multiply(a1, b1));
            Vector256<float> a2 = Avx.BroadcastScalarToVector256(a + 2);
            c2_0 = Avx.Add(c2_0, Avx.Multiply(a2, b0));
            c2_1 = Avx.Add(c2_1, Avx.Multiply(a2, b1));
            Vector256<float> a3 = Avx.BroadcastScalarToVector256(a + 3);
            c3_0 = Avx.Add(c3_0, Avx.Multiply(a3, b0));
            c3_1 = Avx.Add(c3_1, Avx.Multiply(a3, b1));
            Vector256<float> a4 = Avx.BroadcastScalarToVector256(a + 4);
            c4_0 = Avx.Add(c4_0, Avx.Multiply(a4, b0));
            c4_1 = Avx.Add(c4_1, Avx.Multiply(a4, b1));
            Vector256<float> a5 = Avx.BroadcastScalarToVector256(a + 5);
            c5_0 = Avx.Add(c5_0, Avx.Multiply(a5, b0));
            c5_1 = Avx.Add(c5_1, Avx.Multiply(a5, b1));
            a += 6;
            b += 16;
        }

        Avx.Store(c, c0_0);
        Avx.Store(c + 8, c0_1);
        Avx.Store(c + 1 * ldc, c1_0);
        Avx.Store(c + 1 * ldc + 8, c1_1);
        Avx.Store(c + 2 * ldc, c2_0);
        Avx.Store(c + 2 * ldc + 8, c2_1);
        Avx.Store(c + 3 * ldc, c3_0);
        Avx.Store(c + 3 * ldc + 8, c3_1);
        Avx.Store(c + 4 * ldc, c4_0);
        Avx.Store(c + 4 * ldc + 8, c4_1);
        Avx.Store(c + 5 * ldc, c5_0);
        Avx.Store(c + 5 * ldc + 8, c5_1);
    }
}
