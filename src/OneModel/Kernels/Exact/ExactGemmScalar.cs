using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace RecursiveAI.OneModel.Kernels.Exact;

/// <summary>
/// Scalar GEMM microkernel: a 4x4 tile of C over kc steps of k. Every output accumulates its products in increasing k (separately rounded multiply, then add), from +0
/// (or from the C value when accumulating a later k block), so the result does not depend on the tile shape.
/// Bounds the caller guarantees: a points at kc*4 floats (a packed strip, element (p, r) at p*4+r), b at kc*4 floats (element (p, j)
/// at p*4+j), and c at 4 rows of 4 floats, row r at c + r*ldc. Gemm validates the extents of the caller-visible matrices before any tile runs.
/// </summary>
internal static unsafe class ExactGemmScalar
{
    /// <summary>Tile rows.</summary>
    public const int Mr = 4;

    /// <summary>Tile columns.</summary>
    public const int Nr = 4;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Kernel(int kc, float* a, float* b, float* c, nint ldc, int accumulate)
    {
        float* t = stackalloc float[16];
        if (accumulate != 0)
        {
            for (int r = 0; r < 4; r++)
            {
                for (int j = 0; j < 4; j++)
                {
                    t[(r * 4) + j] = c[(r * ldc) + j];
                }
            }
        }
        else
        {
            for (int i = 0; i < 16; i++)
            {
                t[i] = 0f;
            }
        }

        for (int p = 0; p < kc; p++)
        {
            for (int r = 0; r < 4; r++)
            {
                float av = a[r];
                for (int j = 0; j < 4; j++)
                {
                    t[(r * 4) + j] = t[(r * 4) + j] + (av * b[j]);
                }
            }

            a += 4;
            b += 4;
        }

        for (int r = 0; r < 4; r++)
        {
            for (int j = 0; j < 4; j++)
            {
                c[(r * ldc) + j] = t[(r * 4) + j];
            }
        }
    }
}
