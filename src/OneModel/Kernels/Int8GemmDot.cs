using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Arm dot-product (SDOT) int8 microkernel: a 4x16 tile of int32 sums over k quads. SDOT multiplies four signed bytes of each lane
/// of B by one selected quad of A and adds the four products into the 32-bit lane, with no saturation anywhere, so the sums equal
/// the scalar kernel's exactly.
/// Bounds the caller guarantees: <c>a</c> points at kq*16 bytes (per k quad, four rows of four int8: element (kq, r, j) at
/// kq*16 + r*4 + j), <c>b</c> at kq*64 bytes (per k quad, sixteen columns of four int8) and <c>c</c> at 4 rows of 16 int32.
/// </summary>
internal static unsafe class Int8GemmDot
{
    public const int Mr = 4;
    public const int Nr = 16;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Kernel(int kq, byte* a, byte* b, int* c, nint ldc, int accumulate)
    {
        Vector128<int> c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33;
        if (accumulate != 0)
        {
            c00 = AdvSimd.LoadVector128(c);
            c01 = AdvSimd.LoadVector128(c + 4);
            c02 = AdvSimd.LoadVector128(c + 8);
            c03 = AdvSimd.LoadVector128(c + 12);
            c10 = AdvSimd.LoadVector128(c + ldc);
            c11 = AdvSimd.LoadVector128(c + ldc + 4);
            c12 = AdvSimd.LoadVector128(c + ldc + 8);
            c13 = AdvSimd.LoadVector128(c + ldc + 12);
            c20 = AdvSimd.LoadVector128(c + (2 * ldc));
            c21 = AdvSimd.LoadVector128(c + (2 * ldc) + 4);
            c22 = AdvSimd.LoadVector128(c + (2 * ldc) + 8);
            c23 = AdvSimd.LoadVector128(c + (2 * ldc) + 12);
            c30 = AdvSimd.LoadVector128(c + (3 * ldc));
            c31 = AdvSimd.LoadVector128(c + (3 * ldc) + 4);
            c32 = AdvSimd.LoadVector128(c + (3 * ldc) + 8);
            c33 = AdvSimd.LoadVector128(c + (3 * ldc) + 12);
        }
        else
        {
            c00 = c01 = c02 = c03 = c10 = c11 = c12 = c13 = Vector128<int>.Zero;
            c20 = c21 = c22 = c23 = c30 = c31 = c32 = c33 = Vector128<int>.Zero;
        }

        for (int p = 0; p < kq; p++)
        {
            Vector128<sbyte> x = AdvSimd.LoadVector128((sbyte*)a);
            Vector128<sbyte> b0 = AdvSimd.LoadVector128((sbyte*)b);
            Vector128<sbyte> b1 = AdvSimd.LoadVector128((sbyte*)(b + 16));
            Vector128<sbyte> b2 = AdvSimd.LoadVector128((sbyte*)(b + 32));
            Vector128<sbyte> b3 = AdvSimd.LoadVector128((sbyte*)(b + 48));
            c00 = Dp.DotProductBySelectedQuadruplet(c00, b0, x, 0);
            c01 = Dp.DotProductBySelectedQuadruplet(c01, b1, x, 0);
            c02 = Dp.DotProductBySelectedQuadruplet(c02, b2, x, 0);
            c03 = Dp.DotProductBySelectedQuadruplet(c03, b3, x, 0);
            c10 = Dp.DotProductBySelectedQuadruplet(c10, b0, x, 1);
            c11 = Dp.DotProductBySelectedQuadruplet(c11, b1, x, 1);
            c12 = Dp.DotProductBySelectedQuadruplet(c12, b2, x, 1);
            c13 = Dp.DotProductBySelectedQuadruplet(c13, b3, x, 1);
            c20 = Dp.DotProductBySelectedQuadruplet(c20, b0, x, 2);
            c21 = Dp.DotProductBySelectedQuadruplet(c21, b1, x, 2);
            c22 = Dp.DotProductBySelectedQuadruplet(c22, b2, x, 2);
            c23 = Dp.DotProductBySelectedQuadruplet(c23, b3, x, 2);
            c30 = Dp.DotProductBySelectedQuadruplet(c30, b0, x, 3);
            c31 = Dp.DotProductBySelectedQuadruplet(c31, b1, x, 3);
            c32 = Dp.DotProductBySelectedQuadruplet(c32, b2, x, 3);
            c33 = Dp.DotProductBySelectedQuadruplet(c33, b3, x, 3);
            a += 16;
            b += 64;
        }

        AdvSimd.Store(c, c00);
        AdvSimd.Store(c + 4, c01);
        AdvSimd.Store(c + 8, c02);
        AdvSimd.Store(c + 12, c03);
        AdvSimd.Store(c + ldc, c10);
        AdvSimd.Store(c + ldc + 4, c11);
        AdvSimd.Store(c + ldc + 8, c12);
        AdvSimd.Store(c + ldc + 12, c13);
        AdvSimd.Store(c + (2 * ldc), c20);
        AdvSimd.Store(c + (2 * ldc) + 4, c21);
        AdvSimd.Store(c + (2 * ldc) + 8, c22);
        AdvSimd.Store(c + (2 * ldc) + 12, c23);
        AdvSimd.Store(c + (3 * ldc), c30);
        AdvSimd.Store(c + (3 * ldc) + 4, c31);
        AdvSimd.Store(c + (3 * ldc) + 8, c32);
        AdvSimd.Store(c + (3 * ldc) + 12, c33);
    }
}
