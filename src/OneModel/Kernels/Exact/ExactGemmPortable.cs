using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace RecursiveAI.OneModel.Kernels.Exact;

/// <summary>
/// Portable four-lane (Vector128) GEMM microkernel: a 4x8 tile of C over kc steps of k. Every output accumulates its products in increasing k (separately rounded multiply, then add), from +0
/// (or from the C value when accumulating a later k block), so the result does not depend on the tile shape.
/// Bounds the caller guarantees: a points at kc*4 floats (a packed strip, element (p, r) at p*4+r), b at kc*8 floats (element (p, j)
/// at p*8+j), and c at 4 rows of 8 floats, row r at c + r*ldc. Gemm validates the extents of the caller-visible matrices before any tile runs.
/// </summary>
internal static unsafe class ExactGemmPortable
{
    /// <summary>Tile rows.</summary>
    public const int Mr = 4;

    /// <summary>Tile columns.</summary>
    public const int Nr = 8;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Kernel(int kc, float* a, float* b, float* c, nint ldc, int accumulate)
    {
        Vector128<float> c0_0;
        Vector128<float> c0_1;
        Vector128<float> c1_0;
        Vector128<float> c1_1;
        Vector128<float> c2_0;
        Vector128<float> c2_1;
        Vector128<float> c3_0;
        Vector128<float> c3_1;
        if (accumulate != 0)
        {
            c0_0 = Vector128.Load(c);
            c0_1 = Vector128.Load(c + 4);
            c1_0 = Vector128.Load(c + 1 * ldc);
            c1_1 = Vector128.Load(c + 1 * ldc + 4);
            c2_0 = Vector128.Load(c + 2 * ldc);
            c2_1 = Vector128.Load(c + 2 * ldc + 4);
            c3_0 = Vector128.Load(c + 3 * ldc);
            c3_1 = Vector128.Load(c + 3 * ldc + 4);
        }
        else
        {
            c0_0 = Vector128<float>.Zero;
            c0_1 = Vector128<float>.Zero;
            c1_0 = Vector128<float>.Zero;
            c1_1 = Vector128<float>.Zero;
            c2_0 = Vector128<float>.Zero;
            c2_1 = Vector128<float>.Zero;
            c3_0 = Vector128<float>.Zero;
            c3_1 = Vector128<float>.Zero;
        }

        for (int p = 0; p < kc; p++)
        {
            Vector128<float> b0 = Vector128.Load(b);
            Vector128<float> b1 = Vector128.Load(b + 4);
            Vector128<float> a0 = Vector128.Create(a[0]);
            Vector128<float> a1 = Vector128.Create(a[1]);
            Vector128<float> a2 = Vector128.Create(a[2]);
            Vector128<float> a3 = Vector128.Create(a[3]);
            c0_0 = Vector128.Add(c0_0, Vector128.Multiply(a0, b0));
            c0_1 = Vector128.Add(c0_1, Vector128.Multiply(a0, b1));
            c1_0 = Vector128.Add(c1_0, Vector128.Multiply(a1, b0));
            c1_1 = Vector128.Add(c1_1, Vector128.Multiply(a1, b1));
            c2_0 = Vector128.Add(c2_0, Vector128.Multiply(a2, b0));
            c2_1 = Vector128.Add(c2_1, Vector128.Multiply(a2, b1));
            c3_0 = Vector128.Add(c3_0, Vector128.Multiply(a3, b0));
            c3_1 = Vector128.Add(c3_1, Vector128.Multiply(a3, b1));
            a += 4;
            b += 8;
        }

        Vector128.Store(c0_0, c);
        Vector128.Store(c0_1, c + 4);
        Vector128.Store(c1_0, c + 1 * ldc);
        Vector128.Store(c1_1, c + 1 * ldc + 4);
        Vector128.Store(c2_0, c + 2 * ldc);
        Vector128.Store(c2_1, c + 2 * ldc + 4);
        Vector128.Store(c3_0, c + 3 * ldc);
        Vector128.Store(c3_1, c + 3 * ldc + 4);
    }
}
