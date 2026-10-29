using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using RecursiveAI.OneModel.Kernels.Exact;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>The elementwise functions of <see cref="ElementwiseMath"/>.</summary>
internal enum ElementwiseFn
{
    /// <summary>e^x.</summary>
    Exp,

    /// <summary>Natural logarithm.</summary>
    Log,

    /// <summary>Hyperbolic tangent.</summary>
    Tanh,

    /// <summary>Error function.</summary>
    Erf,

    /// <summary>Logistic sigmoid.</summary>
    Sigmoid,

    /// <summary>GELU, erf form.</summary>
    GeluErf,

    /// <summary>GELU, tanh form.</summary>
    GeluTanh,
}

internal interface IUnaryFn
{
    static abstract TV Apply<TV, TO>(TV x)
        where TV : struct
        where TO : IVecOps<TV>;
}

internal readonly struct ExpFn : IUnaryFn
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static TV Apply<TV, TO>(TV x)
        where TV : struct
        where TO : IVecOps<TV> => TO.Exp(x);
}

internal readonly struct LogFn : IUnaryFn
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static TV Apply<TV, TO>(TV x)
        where TV : struct
        where TO : IVecOps<TV> => Transcendental.Log<TV, TO>(x);
}

internal readonly struct TanhFn : IUnaryFn
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static TV Apply<TV, TO>(TV x)
        where TV : struct
        where TO : IVecOps<TV> => Transcendental.Tanh<TV, TO>(x);
}

internal readonly struct ErfFn : IUnaryFn
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static TV Apply<TV, TO>(TV x)
        where TV : struct
        where TO : IVecOps<TV> => Transcendental.Erf<TV, TO>(x);
}

internal readonly struct SigmoidFn : IUnaryFn
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static TV Apply<TV, TO>(TV x)
        where TV : struct
        where TO : IVecOps<TV> => Transcendental.Sigmoid<TV, TO>(x);
}

internal readonly struct GeluErfFn : IUnaryFn
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static TV Apply<TV, TO>(TV x)
        where TV : struct
        where TO : IVecOps<TV> => Transcendental.GeluErf<TV, TO>(x);
}

internal readonly struct GeluTanhFn : IUnaryFn
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static TV Apply<TV, TO>(TV x)
        where TV : struct
        where TO : IVecOps<TV> => Transcendental.GeluTanh<TV, TO>(x);
}

/// <summary>
/// Span-level entry points for the elementwise transcendental functions, in exact and fast flavours at a chosen lane width.
/// Exact results are bit-identical for every <see cref="VecIsa"/>; fast results may differ between widths and CPUs.
/// </summary>
internal static class ElementwiseMath
{
    /// <summary>Applies <paramref name="fn"/> to every element of <paramref name="x"/>, writing <paramref name="y"/>. The spans may be the same span but must not otherwise overlap.</summary>
    /// <exception cref="ArgumentException"><paramref name="y"/> is shorter than <paramref name="x"/>.</exception>
    public static void Apply(ElementwiseFn fn, bool exact, VecIsa isa, ReadOnlySpan<float> x, Span<float> y)
    {
        if (y.Length < x.Length)
        {
            throw new ArgumentException("The output is shorter than the input.", nameof(y));
        }

        switch (fn)
        {
            case ElementwiseFn.Exp: Run<ExpFn>(exact, isa, x, y); break;
            case ElementwiseFn.Log: Run<LogFn>(exact, isa, x, y); break;
            case ElementwiseFn.Tanh: Run<TanhFn>(exact, isa, x, y); break;
            case ElementwiseFn.Erf: Run<ErfFn>(exact, isa, x, y); break;
            case ElementwiseFn.Sigmoid: Run<SigmoidFn>(exact, isa, x, y); break;
            case ElementwiseFn.GeluErf: Run<GeluErfFn>(exact, isa, x, y); break;
            case ElementwiseFn.GeluTanh: Run<GeluTanhFn>(exact, isa, x, y); break;
            default: throw new ArgumentOutOfRangeException(nameof(fn));
        }
    }

    private static void Run<TF>(bool exact, VecIsa isa, ReadOnlySpan<float> x, Span<float> y)
        where TF : IUnaryFn
    {
        isa = VecIsaInfo.Resolve(isa);
        if (exact)
        {
            switch (isa)
            {
                case VecIsa.V512: Map<Vector512<float>, ExactVec512, TF>(x, y); break;
                case VecIsa.V256: Map<Vector256<float>, ExactVec256, TF>(x, y); break;
                case VecIsa.V128: Map<Vector128<float>, ExactVec128, TF>(x, y); break;
                default: Map<float, ExactVecScalar, TF>(x, y); break;
            }
        }
        else
        {
            switch (isa)
            {
                case VecIsa.V512: Map<Vector512<float>, FastVec512, TF>(x, y); break;
                case VecIsa.V256: Map<Vector256<float>, FastVec256, TF>(x, y); break;
                case VecIsa.V128: Map<Vector128<float>, FastVec128, TF>(x, y); break;
                default: Map<float, FastVecScalar, TF>(x, y); break;
            }
        }
    }

    private static void Map<TV, TO, TF>(ReadOnlySpan<float> x, Span<float> y)
        where TV : struct
        where TO : IVecOps<TV>
        where TF : IUnaryFn
    {
        int lanes = TO.Lanes;
        int n = x.Length;

        // Bounds: y.Length >= n was checked by the caller, and every vector access below lies inside [0, n).
        int i = 0;
        for (; i + lanes <= n; i += lanes)
        {
            TO.Store(y, i, TF.Apply<TV, TO>(TO.Load(x, i)));
        }

        if (i < n)
        {
            // The tail goes through a zero-padded chunk so it runs the same vector code as the body.
            Span<float> buf = stackalloc float[16];
            buf.Clear();
            x.Slice(i).CopyTo(buf);
            TO.Store(buf, 0, TF.Apply<TV, TO>(TO.Load(buf, 0)));
            buf.Slice(0, n - i).CopyTo(y.Slice(i));
        }
    }
}
