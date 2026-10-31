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

    /// <summary>
    /// One column of C (a matrix-vector product) for <paramref name="strips"/> strips of four rows: <c>c[r * ldc] (+)= sum over k of A[r, k] * b[k]</c>.
    /// The generic kernel computes sixteen columns at once and a vector has only one real column, so this reads the first column of the
    /// packed B strip (per k quad, the first four of its 64 bytes) and puts four rows in the four lanes of one SDOT. Several strips run
    /// side by side so the dependent adds of one strip overlap.
    /// Bounds the caller guarantees: each strip of <c>a</c> holds kq*16 bytes at <c>stripBytes</c> intervals, <c>b</c> holds kq*64 bytes
    /// (only the first four of each 64 are read, but a 16-byte load starts there), and <c>c</c> has 4*strips rows of <c>ldc</c> ints.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Gemv(int kq, byte* a, long stripBytes, int strips, byte* b, int* c, nint ldc, int accumulate)
    {
        int s = 0;
        for (; s + 4 <= strips; s += 4)
        {
            byte* a0 = a + (s * stripBytes), a1 = a0 + stripBytes, a2 = a1 + stripBytes, a3 = a2 + stripBytes;
            int* c0 = c + (4L * s * ldc);
            Vector128<int> r0 = accumulate != 0 ? Load4(c0, ldc) : Vector128<int>.Zero;
            Vector128<int> r1 = accumulate != 0 ? Load4(c0 + (4 * ldc), ldc) : Vector128<int>.Zero;
            Vector128<int> r2 = accumulate != 0 ? Load4(c0 + (8 * ldc), ldc) : Vector128<int>.Zero;
            Vector128<int> r3 = accumulate != 0 ? Load4(c0 + (12 * ldc), ldc) : Vector128<int>.Zero;
            for (int p = 0; p < kq; p++)
            {
                Vector128<sbyte> bq = AdvSimd.LoadVector128((sbyte*)(b + (p * 64L)));
                r0 = Dp.DotProductBySelectedQuadruplet(r0, AdvSimd.LoadVector128((sbyte*)(a0 + (p * 16L))), bq, 0);
                r1 = Dp.DotProductBySelectedQuadruplet(r1, AdvSimd.LoadVector128((sbyte*)(a1 + (p * 16L))), bq, 0);
                r2 = Dp.DotProductBySelectedQuadruplet(r2, AdvSimd.LoadVector128((sbyte*)(a2 + (p * 16L))), bq, 0);
                r3 = Dp.DotProductBySelectedQuadruplet(r3, AdvSimd.LoadVector128((sbyte*)(a3 + (p * 16L))), bq, 0);
            }

            Store4(c0, ldc, r0);
            Store4(c0 + (4 * ldc), ldc, r1);
            Store4(c0 + (8 * ldc), ldc, r2);
            Store4(c0 + (12 * ldc), ldc, r3);
        }

        for (; s < strips; s++)
        {
            byte* a0 = a + (s * stripBytes);
            int* c0 = c + (4L * s * ldc);
            Vector128<int> r0 = accumulate != 0 ? Load4(c0, ldc) : Vector128<int>.Zero;
            Vector128<int> r1 = Vector128<int>.Zero;
            int p = 0;
            for (; p + 2 <= kq; p += 2)
            {
                r0 = Dp.DotProductBySelectedQuadruplet(r0, AdvSimd.LoadVector128((sbyte*)(a0 + (p * 16L))), AdvSimd.LoadVector128((sbyte*)(b + (p * 64L))), 0);
                r1 = Dp.DotProductBySelectedQuadruplet(r1, AdvSimd.LoadVector128((sbyte*)(a0 + ((p + 1) * 16L))), AdvSimd.LoadVector128((sbyte*)(b + ((p + 1) * 64L))), 0);
            }

            if (p < kq)
            {
                r0 = Dp.DotProductBySelectedQuadruplet(r0, AdvSimd.LoadVector128((sbyte*)(a0 + (p * 16L))), AdvSimd.LoadVector128((sbyte*)(b + (p * 64L))), 0);
            }

            Store4(c0, ldc, r0 + r1);
        }
    }

    private static Vector128<int> Load4(int* c, nint ldc) => Vector128.Create(c[0], c[ldc], c[2 * ldc], c[3 * ldc]);

    private static void Store4(int* c, nint ldc, Vector128<int> v)
    {
        c[0] = v.GetElement(0);
        c[ldc] = v.GetElement(1);
        c[2 * ldc] = v.GetElement(2);
        c[3 * ldc] = v.GetElement(3);
    }
}
