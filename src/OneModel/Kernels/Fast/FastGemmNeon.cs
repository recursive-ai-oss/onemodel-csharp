using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace RecursiveAI.OneModel.Kernels.Fast;

/// <summary>
/// Arm64 NEON GEMM microkernel: a 8x12 tile of C over kc steps of k. Every output accumulates its products in increasing k (one fused multiply-add per step), from +0
/// (or from the C value when accumulating a later k block), so the result does not depend on the tile shape.
/// Bounds the caller guarantees: a points at kc*8 floats (a packed strip, element (p, r) at p*8+r), b at kc*12 floats (element (p, j)
/// at p*12+j), and c at 8 rows of 12 floats, row r at c + r*ldc. Gemm validates the extents of the caller-visible matrices before any tile runs.
/// </summary>
internal static unsafe class FastGemmNeon
{
    /// <summary>Tile rows.</summary>
    public const int Mr = 8;

    /// <summary>Tile columns.</summary>
    public const int Nr = 12;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Kernel(int kc, float* a, float* b, float* c, nint ldc, int accumulate)
    {
        Vector128<float> c0_0;
        Vector128<float> c0_1;
        Vector128<float> c0_2;
        Vector128<float> c1_0;
        Vector128<float> c1_1;
        Vector128<float> c1_2;
        Vector128<float> c2_0;
        Vector128<float> c2_1;
        Vector128<float> c2_2;
        Vector128<float> c3_0;
        Vector128<float> c3_1;
        Vector128<float> c3_2;
        Vector128<float> c4_0;
        Vector128<float> c4_1;
        Vector128<float> c4_2;
        Vector128<float> c5_0;
        Vector128<float> c5_1;
        Vector128<float> c5_2;
        Vector128<float> c6_0;
        Vector128<float> c6_1;
        Vector128<float> c6_2;
        Vector128<float> c7_0;
        Vector128<float> c7_1;
        Vector128<float> c7_2;
        if (accumulate != 0)
        {
            c0_0 = AdvSimd.LoadVector128(c);
            c0_1 = AdvSimd.LoadVector128(c + 4);
            c0_2 = AdvSimd.LoadVector128(c + 8);
            c1_0 = AdvSimd.LoadVector128(c + 1 * ldc);
            c1_1 = AdvSimd.LoadVector128(c + 1 * ldc + 4);
            c1_2 = AdvSimd.LoadVector128(c + 1 * ldc + 8);
            c2_0 = AdvSimd.LoadVector128(c + 2 * ldc);
            c2_1 = AdvSimd.LoadVector128(c + 2 * ldc + 4);
            c2_2 = AdvSimd.LoadVector128(c + 2 * ldc + 8);
            c3_0 = AdvSimd.LoadVector128(c + 3 * ldc);
            c3_1 = AdvSimd.LoadVector128(c + 3 * ldc + 4);
            c3_2 = AdvSimd.LoadVector128(c + 3 * ldc + 8);
            c4_0 = AdvSimd.LoadVector128(c + 4 * ldc);
            c4_1 = AdvSimd.LoadVector128(c + 4 * ldc + 4);
            c4_2 = AdvSimd.LoadVector128(c + 4 * ldc + 8);
            c5_0 = AdvSimd.LoadVector128(c + 5 * ldc);
            c5_1 = AdvSimd.LoadVector128(c + 5 * ldc + 4);
            c5_2 = AdvSimd.LoadVector128(c + 5 * ldc + 8);
            c6_0 = AdvSimd.LoadVector128(c + 6 * ldc);
            c6_1 = AdvSimd.LoadVector128(c + 6 * ldc + 4);
            c6_2 = AdvSimd.LoadVector128(c + 6 * ldc + 8);
            c7_0 = AdvSimd.LoadVector128(c + 7 * ldc);
            c7_1 = AdvSimd.LoadVector128(c + 7 * ldc + 4);
            c7_2 = AdvSimd.LoadVector128(c + 7 * ldc + 8);
        }
        else
        {
            c0_0 = Vector128<float>.Zero;
            c0_1 = Vector128<float>.Zero;
            c0_2 = Vector128<float>.Zero;
            c1_0 = Vector128<float>.Zero;
            c1_1 = Vector128<float>.Zero;
            c1_2 = Vector128<float>.Zero;
            c2_0 = Vector128<float>.Zero;
            c2_1 = Vector128<float>.Zero;
            c2_2 = Vector128<float>.Zero;
            c3_0 = Vector128<float>.Zero;
            c3_1 = Vector128<float>.Zero;
            c3_2 = Vector128<float>.Zero;
            c4_0 = Vector128<float>.Zero;
            c4_1 = Vector128<float>.Zero;
            c4_2 = Vector128<float>.Zero;
            c5_0 = Vector128<float>.Zero;
            c5_1 = Vector128<float>.Zero;
            c5_2 = Vector128<float>.Zero;
            c6_0 = Vector128<float>.Zero;
            c6_1 = Vector128<float>.Zero;
            c6_2 = Vector128<float>.Zero;
            c7_0 = Vector128<float>.Zero;
            c7_1 = Vector128<float>.Zero;
            c7_2 = Vector128<float>.Zero;
        }

        for (int p = 0; p < kc; p++)
        {
            Vector128<float> b0 = AdvSimd.LoadVector128(b);
            Vector128<float> b1 = AdvSimd.LoadVector128(b + 4);
            Vector128<float> b2 = AdvSimd.LoadVector128(b + 8);
            Vector128<float> a_lo = AdvSimd.LoadVector128(a);
            Vector128<float> a_hi = AdvSimd.LoadVector128(a + 4);
            c0_0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c0_0, b0, a_lo, 0);
            c0_1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c0_1, b1, a_lo, 0);
            c0_2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c0_2, b2, a_lo, 0);
            c1_0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c1_0, b0, a_lo, 1);
            c1_1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c1_1, b1, a_lo, 1);
            c1_2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c1_2, b2, a_lo, 1);
            c2_0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c2_0, b0, a_lo, 2);
            c2_1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c2_1, b1, a_lo, 2);
            c2_2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c2_2, b2, a_lo, 2);
            c3_0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c3_0, b0, a_lo, 3);
            c3_1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c3_1, b1, a_lo, 3);
            c3_2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c3_2, b2, a_lo, 3);
            c4_0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c4_0, b0, a_hi, 0);
            c4_1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c4_1, b1, a_hi, 0);
            c4_2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c4_2, b2, a_hi, 0);
            c5_0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c5_0, b0, a_hi, 1);
            c5_1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c5_1, b1, a_hi, 1);
            c5_2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c5_2, b2, a_hi, 1);
            c6_0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c6_0, b0, a_hi, 2);
            c6_1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c6_1, b1, a_hi, 2);
            c6_2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c6_2, b2, a_hi, 2);
            c7_0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c7_0, b0, a_hi, 3);
            c7_1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c7_1, b1, a_hi, 3);
            c7_2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c7_2, b2, a_hi, 3);
            a += 8;
            b += 12;
        }

        AdvSimd.Store(c, c0_0);
        AdvSimd.Store(c + 4, c0_1);
        AdvSimd.Store(c + 8, c0_2);
        AdvSimd.Store(c + 1 * ldc, c1_0);
        AdvSimd.Store(c + 1 * ldc + 4, c1_1);
        AdvSimd.Store(c + 1 * ldc + 8, c1_2);
        AdvSimd.Store(c + 2 * ldc, c2_0);
        AdvSimd.Store(c + 2 * ldc + 4, c2_1);
        AdvSimd.Store(c + 2 * ldc + 8, c2_2);
        AdvSimd.Store(c + 3 * ldc, c3_0);
        AdvSimd.Store(c + 3 * ldc + 4, c3_1);
        AdvSimd.Store(c + 3 * ldc + 8, c3_2);
        AdvSimd.Store(c + 4 * ldc, c4_0);
        AdvSimd.Store(c + 4 * ldc + 4, c4_1);
        AdvSimd.Store(c + 4 * ldc + 8, c4_2);
        AdvSimd.Store(c + 5 * ldc, c5_0);
        AdvSimd.Store(c + 5 * ldc + 4, c5_1);
        AdvSimd.Store(c + 5 * ldc + 8, c5_2);
        AdvSimd.Store(c + 6 * ldc, c6_0);
        AdvSimd.Store(c + 6 * ldc + 4, c6_1);
        AdvSimd.Store(c + 6 * ldc + 8, c6_2);
        AdvSimd.Store(c + 7 * ldc, c7_0);
        AdvSimd.Store(c + 7 * ldc + 4, c7_1);
        AdvSimd.Store(c + 7 * ldc + 8, c7_2);
    }
}
