using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace RecursiveAI.OneModel.Kernels.Fast;

/// <summary>
/// Matrix-vector microkernels for the fast mode: one row of A against four packed B strips at a time. With one row the register tile
/// of the matrix product would waste all but one of its rows, and the product is bound by streaming B once; four strips give eight
/// independent accumulator chains, enough to hide the latency of the fused multiply-add. Each output accumulates its products in
/// increasing k with one fused multiply-add per step, from +0 or from the stored partial sum of the previous k block, which is exactly
/// what the tile microkernels do, so the results equal the matrix product's bit for bit.
/// Bounds the caller guarantees: <c>a</c> points at kc floats; <c>b</c> at four strips of kc*nr floats, strip i at b + i*stripStride
/// (element (p, j) of a strip at p*nr + j); <c>c</c> at 4*nr floats.
/// </summary>
internal static unsafe class FastGemv
{
    /// <summary>AVX-512: strips of 32 columns.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Kernel4Avx512(int kc, float* a, float* b, nint stripStride, float* c, int accumulate)
    {
        Vector512<float> c00, c01, c10, c11, c20, c21, c30, c31;
        if (accumulate != 0)
        {
            c00 = Avx512F.LoadVector512(c);
            c01 = Avx512F.LoadVector512(c + 16);
            c10 = Avx512F.LoadVector512(c + 32);
            c11 = Avx512F.LoadVector512(c + 48);
            c20 = Avx512F.LoadVector512(c + 64);
            c21 = Avx512F.LoadVector512(c + 80);
            c30 = Avx512F.LoadVector512(c + 96);
            c31 = Avx512F.LoadVector512(c + 112);
        }
        else
        {
            c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = Vector512<float>.Zero;
        }

        float* b1 = b + stripStride;
        float* b2 = b1 + stripStride;
        float* b3 = b2 + stripStride;
        for (int p = 0; p < kc; p++)
        {
            Vector512<float> av = Vector512.Create(a[p]);
            c00 = Avx512F.FusedMultiplyAdd(av, Avx512F.LoadVector512(b), c00);
            c01 = Avx512F.FusedMultiplyAdd(av, Avx512F.LoadVector512(b + 16), c01);
            c10 = Avx512F.FusedMultiplyAdd(av, Avx512F.LoadVector512(b1), c10);
            c11 = Avx512F.FusedMultiplyAdd(av, Avx512F.LoadVector512(b1 + 16), c11);
            c20 = Avx512F.FusedMultiplyAdd(av, Avx512F.LoadVector512(b2), c20);
            c21 = Avx512F.FusedMultiplyAdd(av, Avx512F.LoadVector512(b2 + 16), c21);
            c30 = Avx512F.FusedMultiplyAdd(av, Avx512F.LoadVector512(b3), c30);
            c31 = Avx512F.FusedMultiplyAdd(av, Avx512F.LoadVector512(b3 + 16), c31);
            b += 32;
            b1 += 32;
            b2 += 32;
            b3 += 32;
        }

        Avx512F.Store(c, c00);
        Avx512F.Store(c + 16, c01);
        Avx512F.Store(c + 32, c10);
        Avx512F.Store(c + 48, c11);
        Avx512F.Store(c + 64, c20);
        Avx512F.Store(c + 80, c21);
        Avx512F.Store(c + 96, c30);
        Avx512F.Store(c + 112, c31);
    }

    /// <summary>AVX-512, one strip of 32 columns (the tail of a row of strips).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Kernel1Avx512(int kc, float* a, float* b, float* c, int accumulate)
    {
        Vector512<float> c0 = accumulate != 0 ? Avx512F.LoadVector512(c) : Vector512<float>.Zero;
        Vector512<float> c1 = accumulate != 0 ? Avx512F.LoadVector512(c + 16) : Vector512<float>.Zero;
        for (int p = 0; p < kc; p++)
        {
            Vector512<float> av = Vector512.Create(a[p]);
            c0 = Avx512F.FusedMultiplyAdd(av, Avx512F.LoadVector512(b), c0);
            c1 = Avx512F.FusedMultiplyAdd(av, Avx512F.LoadVector512(b + 16), c1);
            b += 32;
        }

        Avx512F.Store(c, c0);
        Avx512F.Store(c + 16, c1);
    }

    /// <summary>AVX2 + FMA: strips of 16 columns.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Kernel4Avx2(int kc, float* a, float* b, nint stripStride, float* c, int accumulate)
    {
        Vector256<float> c00, c01, c10, c11, c20, c21, c30, c31;
        if (accumulate != 0)
        {
            c00 = Avx.LoadVector256(c);
            c01 = Avx.LoadVector256(c + 8);
            c10 = Avx.LoadVector256(c + 16);
            c11 = Avx.LoadVector256(c + 24);
            c20 = Avx.LoadVector256(c + 32);
            c21 = Avx.LoadVector256(c + 40);
            c30 = Avx.LoadVector256(c + 48);
            c31 = Avx.LoadVector256(c + 56);
        }
        else
        {
            c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = Vector256<float>.Zero;
        }

        float* b1 = b + stripStride;
        float* b2 = b1 + stripStride;
        float* b3 = b2 + stripStride;
        for (int p = 0; p < kc; p++)
        {
            Vector256<float> av = Vector256.Create(a[p]);
            c00 = Fma.MultiplyAdd(av, Avx.LoadVector256(b), c00);
            c01 = Fma.MultiplyAdd(av, Avx.LoadVector256(b + 8), c01);
            c10 = Fma.MultiplyAdd(av, Avx.LoadVector256(b1), c10);
            c11 = Fma.MultiplyAdd(av, Avx.LoadVector256(b1 + 8), c11);
            c20 = Fma.MultiplyAdd(av, Avx.LoadVector256(b2), c20);
            c21 = Fma.MultiplyAdd(av, Avx.LoadVector256(b2 + 8), c21);
            c30 = Fma.MultiplyAdd(av, Avx.LoadVector256(b3), c30);
            c31 = Fma.MultiplyAdd(av, Avx.LoadVector256(b3 + 8), c31);
            b += 16;
            b1 += 16;
            b2 += 16;
            b3 += 16;
        }

        Avx.Store(c, c00);
        Avx.Store(c + 8, c01);
        Avx.Store(c + 16, c10);
        Avx.Store(c + 24, c11);
        Avx.Store(c + 32, c20);
        Avx.Store(c + 40, c21);
        Avx.Store(c + 48, c30);
        Avx.Store(c + 56, c31);
    }

    /// <summary>AVX2 + FMA, one strip of 16 columns.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Kernel1Avx2(int kc, float* a, float* b, float* c, int accumulate)
    {
        Vector256<float> c0 = accumulate != 0 ? Avx.LoadVector256(c) : Vector256<float>.Zero;
        Vector256<float> c1 = accumulate != 0 ? Avx.LoadVector256(c + 8) : Vector256<float>.Zero;
        for (int p = 0; p < kc; p++)
        {
            Vector256<float> av = Vector256.Create(a[p]);
            c0 = Fma.MultiplyAdd(av, Avx.LoadVector256(b), c0);
            c1 = Fma.MultiplyAdd(av, Avx.LoadVector256(b + 8), c1);
            b += 16;
        }

        Avx.Store(c, c0);
        Avx.Store(c + 8, c1);
    }
}
