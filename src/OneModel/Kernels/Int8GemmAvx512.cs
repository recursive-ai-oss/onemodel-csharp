using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// AVX-512BW int8 microkernel: a 12x32 tile of int32 sums over k pairs, the 512-bit form of <see cref="Int8GemmAvx2"/> (same packed
/// layout, never-saturating 16-bit pair multiply-add, exact integer sums).
/// Bounds the caller guarantees: <c>a</c> points at kg*48 bytes (per k pair, twelve rows of two int16), <c>b</c> at kg*64 bytes
/// (per k pair, thirty-two columns of two int8) and <c>c</c> at 12 rows of 32 int32, row r at c + r*ldc.
/// </summary>
internal static unsafe class Int8GemmAvx512
{
    public const int Mr = 12;
    public const int Nr = 32;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Kernel(int kg, byte* a, byte* b, int* c, nint ldc, int accumulate)
    {
        Vector512<int> c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
        Vector512<int> c60, c61, c70, c71, c80, c81, c90, c91, ca0, ca1, cb0, cb1;
        if (accumulate != 0)
        {
            c00 = Avx512F.LoadVector512(c);
            c01 = Avx512F.LoadVector512(c + 16);
            c10 = Avx512F.LoadVector512(c + ldc);
            c11 = Avx512F.LoadVector512(c + ldc + 16);
            c20 = Avx512F.LoadVector512(c + (2 * ldc));
            c21 = Avx512F.LoadVector512(c + (2 * ldc) + 16);
            c30 = Avx512F.LoadVector512(c + (3 * ldc));
            c31 = Avx512F.LoadVector512(c + (3 * ldc) + 16);
            c40 = Avx512F.LoadVector512(c + (4 * ldc));
            c41 = Avx512F.LoadVector512(c + (4 * ldc) + 16);
            c50 = Avx512F.LoadVector512(c + (5 * ldc));
            c51 = Avx512F.LoadVector512(c + (5 * ldc) + 16);
            c60 = Avx512F.LoadVector512(c + (6 * ldc));
            c61 = Avx512F.LoadVector512(c + (6 * ldc) + 16);
            c70 = Avx512F.LoadVector512(c + (7 * ldc));
            c71 = Avx512F.LoadVector512(c + (7 * ldc) + 16);
            c80 = Avx512F.LoadVector512(c + (8 * ldc));
            c81 = Avx512F.LoadVector512(c + (8 * ldc) + 16);
            c90 = Avx512F.LoadVector512(c + (9 * ldc));
            c91 = Avx512F.LoadVector512(c + (9 * ldc) + 16);
            ca0 = Avx512F.LoadVector512(c + (10 * ldc));
            ca1 = Avx512F.LoadVector512(c + (10 * ldc) + 16);
            cb0 = Avx512F.LoadVector512(c + (11 * ldc));
            cb1 = Avx512F.LoadVector512(c + (11 * ldc) + 16);
        }
        else
        {
            c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = Vector512<int>.Zero;
            c60 = c61 = c70 = c71 = c80 = c81 = c90 = c91 = ca0 = ca1 = cb0 = cb1 = Vector512<int>.Zero;
        }

        for (int p = 0; p < kg; p++)
        {
            Vector512<short> b0 = Avx512BW.ConvertToVector512Int16(Avx.LoadVector256((sbyte*)b));
            Vector512<short> b1 = Avx512BW.ConvertToVector512Int16(Avx.LoadVector256((sbyte*)(b + 32)));
            Vector512<short> x = Vector512.Create(*(int*)a).AsInt16();
            c00 += Avx512BW.MultiplyAddAdjacent(x, b0);
            c01 += Avx512BW.MultiplyAddAdjacent(x, b1);
            x = Vector512.Create(*(int*)(a + 4)).AsInt16();
            c10 += Avx512BW.MultiplyAddAdjacent(x, b0);
            c11 += Avx512BW.MultiplyAddAdjacent(x, b1);
            x = Vector512.Create(*(int*)(a + 8)).AsInt16();
            c20 += Avx512BW.MultiplyAddAdjacent(x, b0);
            c21 += Avx512BW.MultiplyAddAdjacent(x, b1);
            x = Vector512.Create(*(int*)(a + 12)).AsInt16();
            c30 += Avx512BW.MultiplyAddAdjacent(x, b0);
            c31 += Avx512BW.MultiplyAddAdjacent(x, b1);
            x = Vector512.Create(*(int*)(a + 16)).AsInt16();
            c40 += Avx512BW.MultiplyAddAdjacent(x, b0);
            c41 += Avx512BW.MultiplyAddAdjacent(x, b1);
            x = Vector512.Create(*(int*)(a + 20)).AsInt16();
            c50 += Avx512BW.MultiplyAddAdjacent(x, b0);
            c51 += Avx512BW.MultiplyAddAdjacent(x, b1);
            x = Vector512.Create(*(int*)(a + 24)).AsInt16();
            c60 += Avx512BW.MultiplyAddAdjacent(x, b0);
            c61 += Avx512BW.MultiplyAddAdjacent(x, b1);
            x = Vector512.Create(*(int*)(a + 28)).AsInt16();
            c70 += Avx512BW.MultiplyAddAdjacent(x, b0);
            c71 += Avx512BW.MultiplyAddAdjacent(x, b1);
            x = Vector512.Create(*(int*)(a + 32)).AsInt16();
            c80 += Avx512BW.MultiplyAddAdjacent(x, b0);
            c81 += Avx512BW.MultiplyAddAdjacent(x, b1);
            x = Vector512.Create(*(int*)(a + 36)).AsInt16();
            c90 += Avx512BW.MultiplyAddAdjacent(x, b0);
            c91 += Avx512BW.MultiplyAddAdjacent(x, b1);
            x = Vector512.Create(*(int*)(a + 40)).AsInt16();
            ca0 += Avx512BW.MultiplyAddAdjacent(x, b0);
            ca1 += Avx512BW.MultiplyAddAdjacent(x, b1);
            x = Vector512.Create(*(int*)(a + 44)).AsInt16();
            cb0 += Avx512BW.MultiplyAddAdjacent(x, b0);
            cb1 += Avx512BW.MultiplyAddAdjacent(x, b1);
            a += 48;
            b += 64;
        }

        Avx512F.Store(c, c00);
        Avx512F.Store(c + 16, c01);
        Avx512F.Store(c + ldc, c10);
        Avx512F.Store(c + ldc + 16, c11);
        Avx512F.Store(c + (2 * ldc), c20);
        Avx512F.Store(c + (2 * ldc) + 16, c21);
        Avx512F.Store(c + (3 * ldc), c30);
        Avx512F.Store(c + (3 * ldc) + 16, c31);
        Avx512F.Store(c + (4 * ldc), c40);
        Avx512F.Store(c + (4 * ldc) + 16, c41);
        Avx512F.Store(c + (5 * ldc), c50);
        Avx512F.Store(c + (5 * ldc) + 16, c51);
        Avx512F.Store(c + (6 * ldc), c60);
        Avx512F.Store(c + (6 * ldc) + 16, c61);
        Avx512F.Store(c + (7 * ldc), c70);
        Avx512F.Store(c + (7 * ldc) + 16, c71);
        Avx512F.Store(c + (8 * ldc), c80);
        Avx512F.Store(c + (8 * ldc) + 16, c81);
        Avx512F.Store(c + (9 * ldc), c90);
        Avx512F.Store(c + (9 * ldc) + 16, c91);
        Avx512F.Store(c + (10 * ldc), ca0);
        Avx512F.Store(c + (10 * ldc) + 16, ca1);
        Avx512F.Store(c + (11 * ldc), cb0);
        Avx512F.Store(c + (11 * ldc) + 16, cb1);
    }
}
