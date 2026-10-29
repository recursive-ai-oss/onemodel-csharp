using System;
using System.Numerics;
using System.Runtime.Intrinsics;

namespace RecursiveAI.OneModel.Kernels.Exact;

// Element-wise arithmetic that is IEEE-exact by construction: one correctly rounded + or * (or a comparison and a select) per element,
// no fused operations, so a vector lane computes exactly what the scalar twin computes. Both execution modes use these loops. The one
// difference is the NaN rule: hardware disagrees about the sign and payload of a NaN it creates (x86 makes 0xFFC00000 for inf - inf,
// Arm 0x7FC00000), so exact mode rewrites every NaN result to 0x7FC00000. Fast mode leaves them as the hardware made them.

/// <summary>A binary element-wise operation with scalar and vector forms that round identically.</summary>
internal interface IBinaryOp
{
    static abstract float Apply(float a, float b);

    static abstract Vector128<float> Apply(Vector128<float> a, Vector128<float> b);

    static abstract Vector256<float> Apply(Vector256<float> a, Vector256<float> b);

    /// <summary>The operation on any other numeric type; integers wrap around.</summary>
    static abstract T Apply<T>(T a, T b) where T : unmanaged, INumber<T>;
}

internal readonly struct AddArith : IBinaryOp
{
    public static float Apply(float a, float b) => a + b;

    public static Vector128<float> Apply(Vector128<float> a, Vector128<float> b) => a + b;

    public static Vector256<float> Apply(Vector256<float> a, Vector256<float> b) => a + b;

    public static T Apply<T>(T a, T b) where T : unmanaged, INumber<T> => unchecked(a + b);
}

internal readonly struct MulArith : IBinaryOp
{
    public static float Apply(float a, float b) => a * b;

    public static Vector128<float> Apply(Vector128<float> a, Vector128<float> b) => a * b;

    public static Vector256<float> Apply(Vector256<float> a, Vector256<float> b) => a * b;

    public static T Apply<T>(T a, T b) where T : unmanaged, INumber<T> => unchecked(a * b);
}

internal static class ExactElementwise
{
    private const uint QuietNaN = 0x7FC00000u;

    /// <summary>The canonical NaN of exact mode.</summary>
    public static float CanonicalNaN => BitConverter.UInt32BitsToSingle(QuietNaN);

    /// <summary>
    /// c[i] = op(a[i * sa], b[i * sb]) for i in [0, n). Strides are 0 (the operand is one value repeated) or 1. The loops use the
    /// widest vector the path allows up to 256 bits: element-wise work is memory-bound, so 512 bits would gain nothing.
    /// </summary>
    /// <param name="path">The session's instruction set.</param>
    /// <param name="canonicalNaN">Rewrite NaN results to 0x7FC00000 (exact mode).</param>
    /// <param name="a">At least 1 element, or n when sa is 1.</param>
    /// <param name="sa">Step of a: 0 or 1.</param>
    /// <param name="sb">Step of b: 0 or 1.</param>
    /// <param name="b">At least 1 element, or n when sb is 1.</param>
    /// <param name="c">At least n elements. May be the same memory as a or b when that operand's stride is 1.</param>
    /// <param name="n">Count, at least 0.</param>
    public static void Row<TOp>(KernelPath path, bool canonicalNaN, ReadOnlySpan<float> a, int sa, ReadOnlySpan<float> b, int sb, Span<float> c, int n)
        where TOp : IBinaryOp
    {
        int i = 0;
        if (path is KernelPath.Avx2 or KernelPath.Avx512 && Vector256.IsHardwareAccelerated)
        {
            Vector256<float> nan = Vector256.Create(CanonicalNaN);
            Vector256<float> sA = sa == 0 ? Vector256.Create(a[0]) : default;
            Vector256<float> sB = sb == 0 ? Vector256.Create(b[0]) : default;
            for (; i + 8 <= n; i += 8)
            {
                Vector256<float> x = sa == 0 ? sA : Vector256.Create(a.Slice(i, 8));
                Vector256<float> y = sb == 0 ? sB : Vector256.Create(b.Slice(i, 8));
                Vector256<float> r = TOp.Apply(x, y);
                if (canonicalNaN)
                {
                    r = Vector256.ConditionalSelect(Vector256.Equals(r, r), r, nan);
                }

                r.CopyTo(c.Slice(i, 8));
            }
        }

        if (path != KernelPath.Scalar && Vector128.IsHardwareAccelerated)
        {
            Vector128<float> nan = Vector128.Create(CanonicalNaN);
            Vector128<float> sA = sa == 0 ? Vector128.Create(a[0]) : default;
            Vector128<float> sB = sb == 0 ? Vector128.Create(b[0]) : default;
            for (; i + 4 <= n; i += 4)
            {
                Vector128<float> x = sa == 0 ? sA : Vector128.Create(a.Slice(i, 4));
                Vector128<float> y = sb == 0 ? sB : Vector128.Create(b.Slice(i, 4));
                Vector128<float> r = TOp.Apply(x, y);
                if (canonicalNaN)
                {
                    r = Vector128.ConditionalSelect(Vector128.Equals(r, r), r, nan);
                }

                r.CopyTo(c.Slice(i, 4));
            }
        }

        for (; i < n; i++)
        {
            float r = TOp.Apply(a[sa == 0 ? 0 : i], b[sb == 0 ? 0 : i]);
            c[i] = canonicalNaN && float.IsNaN(r) ? CanonicalNaN : r;
        }
    }

    /// <summary>
    /// y[i] = x[i] when x[i] is greater than 0 or is a NaN, else +0 (so -0 becomes +0). A NaN passes through; in exact mode it is
    /// rewritten to 0x7FC00000. <paramref name="y"/> may be the same memory as <paramref name="x"/>.
    /// </summary>
    public static void Relu(KernelPath path, bool canonicalNaN, ReadOnlySpan<float> x, Span<float> y)
    {
        int n = x.Length;
        int i = 0;
        if (path is KernelPath.Avx2 or KernelPath.Avx512 && Vector256.IsHardwareAccelerated)
        {
            Vector256<float> zero = Vector256<float>.Zero;
            Vector256<float> nan = Vector256.Create(CanonicalNaN);
            for (; i + 8 <= n; i += 8)
            {
                Vector256<float> v = Vector256.Create(x.Slice(i, 8));
                Vector256<float> isNumber = Vector256.Equals(v, v);
                Vector256<float> keep = Vector256.GreaterThan(v, zero) | ~isNumber;
                Vector256<float> r = v & keep;
                if (canonicalNaN)
                {
                    r = Vector256.ConditionalSelect(isNumber, r, nan);
                }

                r.CopyTo(y.Slice(i, 8));
            }
        }

        if (path != KernelPath.Scalar && Vector128.IsHardwareAccelerated)
        {
            Vector128<float> zero = Vector128<float>.Zero;
            Vector128<float> nan = Vector128.Create(CanonicalNaN);
            for (; i + 4 <= n; i += 4)
            {
                Vector128<float> v = Vector128.Create(x.Slice(i, 4));
                Vector128<float> isNumber = Vector128.Equals(v, v);
                Vector128<float> keep = Vector128.GreaterThan(v, zero) | ~isNumber;
                Vector128<float> r = v & keep;
                if (canonicalNaN)
                {
                    r = Vector128.ConditionalSelect(isNumber, r, nan);
                }

                r.CopyTo(y.Slice(i, 4));
            }
        }

        for (; i < n; i++)
        {
            float v = x[i];
            y[i] = float.IsNaN(v) ? (canonicalNaN ? CanonicalNaN : v) : (v > 0f ? v : 0f);
        }
    }

    /// <summary>Relu for the other numeric types (the same rule: positive values and NaN pass, everything else becomes 0).</summary>
    public static void Relu<T>(ReadOnlySpan<T> x, Span<T> y) where T : unmanaged, INumber<T>
    {
        for (int i = 0; i < x.Length; i++)
        {
            T v = x[i];
            y[i] = T.IsNaN(v) || v > T.Zero ? v : T.Zero;
        }
    }
}
