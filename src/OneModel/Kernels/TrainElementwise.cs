using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using RecursiveAI.OneModel.Kernels.Exact;

namespace RecursiveAI.OneModel.Kernels;

// Element-wise loops for the eager (training) operators and their gradients. Each function is written once over IVecOps lanes and
// instantiated at the widest lane group the path allows, with a scalar instantiation of the same code finishing the tail, so every
// element sees the same operations whatever its position. Every operation here is a single IEEE +, -, *, / or a compare-and-select
// (no fused multiply-add is formed, even in fast mode), so the vector lanes give the bits of the scalar loop. Gradients of the
// transcendental activations call the shared polynomials, which have the same property in exact mode. Nothing here depends on the
// thread count: the loops run on the calling thread.

/// <summary>A function of two floats over lanes: the forward binary operators and the gradient rules that read two tensors.</summary>
internal interface IFloatBinary
{
    TV Apply<TV, TO>(TV a, TV b) where TV : struct where TO : IVecOps<TV>;
}

internal readonly struct AddBin : IFloatBinary
{
    public TV Apply<TV, TO>(TV a, TV b) where TV : struct where TO : IVecOps<TV> => TO.Add(a, b);
}

internal readonly struct SubBin : IFloatBinary
{
    public TV Apply<TV, TO>(TV a, TV b) where TV : struct where TO : IVecOps<TV> => TO.Sub(a, b);
}

internal readonly struct MulBin : IFloatBinary
{
    public TV Apply<TV, TO>(TV a, TV b) where TV : struct where TO : IVecOps<TV> => TO.Mul(a, b);
}

internal readonly struct DivBin : IFloatBinary
{
    public TV Apply<TV, TO>(TV a, TV b) where TV : struct where TO : IVecOps<TV> => TO.Div(a, b);
}

/// <summary>Relu backward: g where y &gt; 0, else 0 (a is the incoming gradient, b the saved output).</summary>
internal readonly struct ReluGradBin : IFloatBinary
{
    public TV Apply<TV, TO>(TV g, TV y) where TV : struct where TO : IVecOps<TV> => TO.Select(TO.GreaterThan(y, TO.Zero), g, TO.Zero);
}

/// <summary>Sigmoid backward: g * y * (1 - y), multiplied left to right.</summary>
internal readonly struct SigmoidGradBin : IFloatBinary
{
    public TV Apply<TV, TO>(TV g, TV y) where TV : struct where TO : IVecOps<TV> => TO.Mul(TO.Mul(g, y), TO.Sub(TO.Splat(1f), y));
}

/// <summary>Tanh backward: g * (1 - y * y).</summary>
internal readonly struct TanhGradBin : IFloatBinary
{
    public TV Apply<TV, TO>(TV g, TV y) where TV : struct where TO : IVecOps<TV> => TO.Mul(g, TO.Sub(TO.Splat(1f), TO.Mul(y, y)));
}

/// <summary>GELU backward (a is the incoming gradient, b the saved input): g times the derivative of the erf or tanh form.</summary>
internal readonly struct GeluGradBin(bool tanhForm) : IFloatBinary
{
    public TV Apply<TV, TO>(TV g, TV v) where TV : struct where TO : IVecOps<TV>
    {
        TV slope;
        if (tanhForm)
        {
            // gelu(x) = 0.5 x (1 + tanh(u)), u = k (x + 0.044715 x^3), k = sqrt(2/pi)
            TV u = TO.Mul(TO.Splat(0.79788456f), TO.Add(v, TO.Mul(TO.Mul(TO.Mul(TO.Splat(0.044715f), v), v), v)));
            TV t = Transcendental.Tanh<TV, TO>(u);
            TV left = TO.Mul(TO.Splat(0.5f), TO.Add(TO.Splat(1f), t));
            TV right = TO.Mul(TO.Mul(TO.Mul(TO.Mul(TO.Splat(0.5f), v), TO.Sub(TO.Splat(1f), TO.Mul(t, t))), TO.Splat(0.79788456f)),
                TO.Add(TO.Splat(1f), TO.Mul(TO.Mul(TO.Splat(3f * 0.044715f), v), v)));
            slope = TO.Add(left, right);
        }
        else
        {
            // gelu(x) = x Phi(x); gelu'(x) = Phi(x) + x phi(x)
            TV cdf = TO.Mul(TO.Splat(0.5f), TO.Add(TO.Splat(1f), Transcendental.Erf<TV, TO>(TO.Mul(v, TO.Splat(0.70710677f)))));
            TV pdf = TO.Mul(Transcendental.Exp<TV, TO>(TO.Mul(TO.Mul(TO.Splat(-0.5f), v), v)), TO.Splat(0.3989423f));
            slope = TO.Add(cdf, TO.Mul(v, pdf));
        }

        return TO.Mul(g, slope);
    }
}

/// <summary>Negation as a multiplication by -1, which flips the sign of every value (zeros, infinities and subnormals included) exactly.</summary>
internal readonly struct NegUn : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV> => TO.Mul(x, TO.Splat(-1f));
}

/// <summary>Relu forward: x where x &gt; 0 or NaN, else 0.</summary>
internal readonly struct ReluUn : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV> =>
        TO.Select(TO.GreaterThan(x, TO.Zero), x, TO.Select(TO.Equal(x, x), TO.Zero, x));
}

internal static class TrainElementwise
{
    /// <summary>y[i] = op(a[i], b[i]) for i in [0, n). Spans must hold at least n values; y may alias a or b exactly (same offset).</summary>
    public static void Binary<TOp>(KernelPath path, bool exact, TOp op, ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> y, int n)
        where TOp : struct, IFloatBinary
    {
        if (a.Length < n || b.Length < n || y.Length < n)
        {
            throw new ArgumentException("A buffer is shorter than the element count.");
        }

        int i = 0;
        if (exact)
        {
            if (path == KernelPath.Avx512 && Vector512.IsHardwareAccelerated)
            {
                i = Loop<TOp, Vector512<float>, ExactVec512>(op, true, a, b, y, n, i);
            }
            else if (path is KernelPath.Avx2 or KernelPath.Avx512 && Vector256.IsHardwareAccelerated)
            {
                i = Loop<TOp, Vector256<float>, ExactVec256>(op, true, a, b, y, n, i);
            }
            else if (path != KernelPath.Scalar && Vector128.IsHardwareAccelerated)
            {
                i = Loop<TOp, Vector128<float>, ExactVec128>(op, true, a, b, y, n, i);
            }

            Loop<TOp, float, ExactVecScalar>(op, true, a, b, y, n, i);
        }
        else
        {
            if (path == KernelPath.Avx512 && Vector512.IsHardwareAccelerated)
            {
                i = Loop<TOp, Vector512<float>, FastVec512>(op, false, a, b, y, n, i);
            }
            else if (path is KernelPath.Avx2 or KernelPath.Avx512 && Vector256.IsHardwareAccelerated)
            {
                i = Loop<TOp, Vector256<float>, FastVec256>(op, false, a, b, y, n, i);
            }
            else if (path != KernelPath.Scalar && Vector128.IsHardwareAccelerated)
            {
                i = Loop<TOp, Vector128<float>, FastVec128>(op, false, a, b, y, n, i);
            }

            Loop<TOp, float, FastVecScalar>(op, false, a, b, y, n, i);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Loop<TOp, TV, TO>(TOp op, bool canonicalNaN, ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> y, int n, int i)
        where TOp : struct, IFloatBinary
        where TV : struct
        where TO : IVecOps<TV>
    {
        int lanes = TO.Lanes;
        for (; i + lanes <= n; i += lanes)
        {
            TV r = op.Apply<TV, TO>(TO.Load(a, i), TO.Load(b, i));
            if (canonicalNaN)
            {
                r = Transcendental.Canonical<TV, TO>(r);
            }

            TO.Store(y, i, r);
        }

        return i;
    }
}
