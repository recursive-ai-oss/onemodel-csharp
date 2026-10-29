using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// AVX2 int8 microkernel: a 6x16 tile of int32 sums over k pairs. The 8-bit B values are sign-extended to 16 bits and the A pair
/// (two 16-bit values in one 32-bit lane) is broadcast; <c>vpmaddwd</c> multiplies adjacent 16-bit lanes and adds each pair into
/// 32 bits. Unlike the byte-pair multiply-add (<c>vpmaddubsw</c>), which saturates its 16-bit pair sums, this never saturates, so
/// the sums are the exact integer products and equal the scalar kernel's on every input.
/// Bounds the caller guarantees: <c>a</c> points at kg*24 bytes (per k pair, six rows of two int16: element (kg, r, j) at
/// kg*24 + r*4 + j*2), <c>b</c> at kg*32 bytes (per k pair, sixteen columns of two int8: element (kg, c, j) at kg*32 + c*2 + j) and
/// <c>c</c> at 6 rows of 16 int32, row r at c + r*ldc.
/// </summary>
internal static unsafe class Int8GemmAvx2
{
    public const int Mr = 6;
    public const int Nr = 16;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Kernel(int kg, byte* a, byte* b, int* c, nint ldc, int accumulate)
    {
        Vector256<int> c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
        if (accumulate != 0)
        {
            c00 = Avx.LoadVector256(c);
            c01 = Avx.LoadVector256(c + 8);
            c10 = Avx.LoadVector256(c + ldc);
            c11 = Avx.LoadVector256(c + ldc + 8);
            c20 = Avx.LoadVector256(c + (2 * ldc));
            c21 = Avx.LoadVector256(c + (2 * ldc) + 8);
            c30 = Avx.LoadVector256(c + (3 * ldc));
            c31 = Avx.LoadVector256(c + (3 * ldc) + 8);
            c40 = Avx.LoadVector256(c + (4 * ldc));
            c41 = Avx.LoadVector256(c + (4 * ldc) + 8);
            c50 = Avx.LoadVector256(c + (5 * ldc));
            c51 = Avx.LoadVector256(c + (5 * ldc) + 8);
        }
        else
        {
            c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = Vector256<int>.Zero;
        }

        for (int p = 0; p < kg; p++)
        {
            Vector256<short> b0 = Avx2.ConvertToVector256Int16((sbyte*)b);
            Vector256<short> b1 = Avx2.ConvertToVector256Int16((sbyte*)(b + 16));
            Vector256<short> x = Vector256.Create(*(int*)a).AsInt16();
            c00 += Avx2.MultiplyAddAdjacent(x, b0);
            c01 += Avx2.MultiplyAddAdjacent(x, b1);
            x = Vector256.Create(*(int*)(a + 4)).AsInt16();
            c10 += Avx2.MultiplyAddAdjacent(x, b0);
            c11 += Avx2.MultiplyAddAdjacent(x, b1);
            x = Vector256.Create(*(int*)(a + 8)).AsInt16();
            c20 += Avx2.MultiplyAddAdjacent(x, b0);
            c21 += Avx2.MultiplyAddAdjacent(x, b1);
            x = Vector256.Create(*(int*)(a + 12)).AsInt16();
            c30 += Avx2.MultiplyAddAdjacent(x, b0);
            c31 += Avx2.MultiplyAddAdjacent(x, b1);
            x = Vector256.Create(*(int*)(a + 16)).AsInt16();
            c40 += Avx2.MultiplyAddAdjacent(x, b0);
            c41 += Avx2.MultiplyAddAdjacent(x, b1);
            x = Vector256.Create(*(int*)(a + 20)).AsInt16();
            c50 += Avx2.MultiplyAddAdjacent(x, b0);
            c51 += Avx2.MultiplyAddAdjacent(x, b1);
            a += 24;
            b += 32;
        }

        Avx.Store(c, c00);
        Avx.Store(c + 8, c01);
        Avx.Store(c + ldc, c10);
        Avx.Store(c + ldc + 8, c11);
        Avx.Store(c + (2 * ldc), c20);
        Avx.Store(c + (2 * ldc) + 8, c21);
        Avx.Store(c + (3 * ldc), c30);
        Avx.Store(c + (3 * ldc) + 8, c31);
        Avx.Store(c + (4 * ldc), c40);
        Avx.Store(c + (4 * ldc) + 8, c41);
        Avx.Store(c + (5 * ldc), c50);
        Avx.Store(c + (5 * ldc) + 8, c51);
    }
}
