using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace RecursiveAI.OneModel.Kernels.Exact;

// Single-precision element-wise functions written once over IVecOps, like the transcendentals they are built from. Exact mode runs
// them on the exact lane operations (no fused multiply-add, own polynomials, canonical NaN), so scalar, 128-, 256- and 512-bit
// instantiations return the same bits. Fast mode runs the same code on the fast lane operations, which fuse multiply-adds where the
// CPU can. docs/ops-elementwise.md lists each function's definition and special values.

/// <summary>A function of one float, expressed over lanes. Parameters (alpha, beta) are fields of the implementing struct.</summary>
internal interface IFloatUnary
{
    TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV>;
}

internal readonly struct ExpUnary : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV> => Transcendental.Exp<TV, TO>(x);
}

internal readonly struct LogUnary : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV> => Transcendental.Log<TV, TO>(x);
}

internal readonly struct TanhUnary : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV> => Transcendental.Tanh<TV, TO>(x);
}

internal readonly struct SigmoidUnary : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV> => Transcendental.Sigmoid<TV, TO>(x);
}

internal readonly struct ErfUnary : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV> => Transcendental.Erf<TV, TO>(x);
}

/// <summary>Gelu: 0.5 x (1 + erf(x / sqrt 2)), or the tanh approximation.</summary>
internal readonly struct GeluUnary(bool tanhForm) : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV> =>
        tanhForm ? Transcendental.GeluTanh<TV, TO>(x) : Transcendental.GeluErf<TV, TO>(x);
}

/// <summary>LeakyRelu: x for x &gt;= 0 (and NaN), alpha * x for x &lt; 0.</summary>
internal readonly struct LeakyReluUnary(float alpha) : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV> =>
        TO.Select(TO.LessThan(x, TO.Zero), TO.Mul(x, TO.Splat(alpha)), x);
}

/// <summary>HardSigmoid: clamp(alpha * x + beta, 0, 1), computed as a rounded product and then a rounded sum.</summary>
internal readonly struct HardSigmoidUnary(float alpha, float beta) : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV> => Clamp01<TV, TO>(TO.Add(TO.Mul(x, TO.Splat(alpha)), TO.Splat(beta)));

    internal static TV Clamp01<TV, TO>(TV t) where TV : struct where TO : IVecOps<TV>
    {
        TV r = TO.Select(TO.GreaterThan(t, TO.Splat(1f)), TO.Splat(1f), t);
        return TO.Select(TO.LessThan(r, TO.Zero), TO.Zero, r);
    }
}

/// <summary>HardSwish: x * clamp(x / 6 + 0.5, 0, 1), with the constants alpha = 1/6 as a float and beta = 0.5.</summary>
internal readonly struct HardSwishUnary : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV> =>
        TO.Mul(x, HardSigmoidUnary.Clamp01<TV, TO>(TO.Add(TO.Mul(x, TO.Splat(0.16666667f)), TO.Splat(0.5f))));
}

internal static class FloatUnary
{
    /// <summary>
    /// y[i] = op(x[i]) for all i. The widest lane group the path allows runs first and a scalar instantiation of the same code
    /// finishes the tail, so every element sees the same operations. y may be the same memory as x.
    /// </summary>
    /// <param name="path">The session's instruction set.</param>
    /// <param name="exact">Use the exact lane operations and rewrite NaN results to the canonical NaN.</param>
    /// <param name="op">The function.</param>
    /// <param name="x">Input.</param>
    /// <param name="y">Output, at least as long as x.</param>
    public static void Run<TOp>(KernelPath path, bool exact, TOp op, ReadOnlySpan<float> x, Span<float> y) where TOp : struct, IFloatUnary
    {
        if (y.Length < x.Length)
        {
            throw new ArgumentException("The output is shorter than the input.");
        }

        int i = 0;
        if (exact)
        {
            if (path == KernelPath.Avx512 && Vector512.IsHardwareAccelerated)
            {
                i = Loop<TOp, Vector512<float>, ExactVec512>(op, true, x, y, i);
            }
            else if (path is KernelPath.Avx2 or KernelPath.Avx512 && Vector256.IsHardwareAccelerated)
            {
                i = Loop<TOp, Vector256<float>, ExactVec256>(op, true, x, y, i);
            }
            else if (path != KernelPath.Scalar && Vector128.IsHardwareAccelerated)
            {
                i = Loop<TOp, Vector128<float>, ExactVec128>(op, true, x, y, i);
            }

            Loop<TOp, float, ExactVecScalar>(op, true, x, y, i);
        }
        else
        {
            if (path == KernelPath.Avx512 && Vector512.IsHardwareAccelerated)
            {
                i = Loop<TOp, Vector512<float>, FastVec512>(op, false, x, y, i);
            }
            else if (path is KernelPath.Avx2 or KernelPath.Avx512 && Vector256.IsHardwareAccelerated)
            {
                i = Loop<TOp, Vector256<float>, FastVec256>(op, false, x, y, i);
            }
            else if (path != KernelPath.Scalar && Vector128.IsHardwareAccelerated)
            {
                i = Loop<TOp, Vector128<float>, FastVec128>(op, false, x, y, i);
            }

            Loop<TOp, float, FastVecScalar>(op, false, x, y, i);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Loop<TOp, TV, TO>(TOp op, bool canonicalNaN, ReadOnlySpan<float> x, Span<float> y, int i)
        where TOp : struct, IFloatUnary
        where TV : struct
        where TO : IVecOps<TV>
    {
        int lanes = TO.Lanes;
        int n = x.Length;
        for (; i + lanes <= n; i += lanes)
        {
            TV r = op.Apply<TV, TO>(TO.Load(x, i));
            if (canonicalNaN)
            {
                r = Transcendental.Canonical<TV, TO>(r);
            }

            TO.Store(y, i, r);
        }

        return i;
    }
}
