using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Graph.Operators.Elementwise;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Activation;

// The remaining activation functions of the default domain. Each is a lane function over IVecOps, so one definition gives the scalar,
// vector, fast and exact forms; exp and log are the library's own polynomials (docs/exact-transcendentals.md), so these operators are
// exact. The constants follow the operator definitions in Operators.md.

/// <summary>Elu: x for x &gt;= 0 (and NaN), alpha * (e^x - 1) for x &lt; 0.</summary>
internal readonly struct EluUnary(float alpha) : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV> =>
        TO.Select(TO.LessThan(x, TO.Zero), TO.Mul(TO.Splat(alpha), TO.Sub(Transcendental.Exp<TV, TO>(x), TO.Splat(1f))), x);
}

/// <summary>Selu: gamma * x for x &gt; 0, gamma * (alpha * e^x - alpha) for x &lt;= 0.</summary>
internal readonly struct SeluUnary(float alpha, float gamma) : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV>
    {
        TV neg = TO.Sub(TO.Mul(TO.Splat(alpha), Transcendental.Exp<TV, TO>(x)), TO.Splat(alpha));
        return TO.Mul(TO.Splat(gamma), TO.Select(TO.GreaterThan(x, TO.Zero), x, TO.Select(TO.Equal(x, x), neg, x)));
    }
}

/// <summary>Celu: max(0, x) + min(0, alpha * (e^(x / alpha) - 1)).</summary>
internal readonly struct CeluUnary(float alpha) : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV>
    {
        TV neg = TO.Mul(TO.Splat(alpha), TO.Sub(Transcendental.Exp<TV, TO>(TO.Div(x, TO.Splat(alpha))), TO.Splat(1f)));
        return TO.Select(TO.LessThan(x, TO.Zero), neg, x);
    }
}

/// <summary>Softplus: ln(e^x + 1), computed as max(x, 0) + ln(1 + e^-|x|) so a large x does not overflow.</summary>
internal readonly struct SoftplusUnary : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV> => Compute<TV, TO>(x);

    internal static TV Compute<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV>
    {
        TV negAbs = TO.Select(TO.GreaterThan(x, TO.Zero), TO.Sub(TO.Zero, x), x);
        TV tail = Transcendental.Log<TV, TO>(TO.Add(TO.Splat(1f), Transcendental.Exp<TV, TO>(negAbs)));
        return TO.Add(TO.Select(TO.GreaterThan(x, TO.Zero), x, TO.Zero), tail);
    }
}

/// <summary>Softsign: x / (1 + |x|).</summary>
internal readonly struct SoftsignUnary : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV> =>
        TO.Div(x, TO.Add(TO.Splat(1f), TO.Select(TO.LessThan(x, TO.Zero), TO.Sub(TO.Zero, x), x)));
}

/// <summary>Mish: x * tanh(softplus(x)).</summary>
internal readonly struct MishUnary : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV> =>
        TO.Mul(x, Transcendental.Tanh<TV, TO>(SoftplusUnary.Compute<TV, TO>(x)));
}

/// <summary>Swish: x * sigmoid(alpha * x).</summary>
internal readonly struct SwishUnary(float alpha) : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV> =>
        TO.Mul(x, Transcendental.Sigmoid<TV, TO>(TO.Mul(TO.Splat(alpha), x)));
}

/// <summary>ThresholdedRelu: x where x &gt; alpha, else 0.</summary>
internal readonly struct ThresholdedReluUnary(float alpha) : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV> =>
        TO.Select(TO.GreaterThan(x, TO.Splat(alpha)), x, TO.Zero);
}

/// <summary>Shrink: x + bias for x &lt; -lambd, x - bias for x &gt; lambd, 0 between.</summary>
internal readonly struct ShrinkUnary(float lambd, float bias) : IFloatUnary
{
    public TV Apply<TV, TO>(TV x) where TV : struct where TO : IVecOps<TV>
    {
        TV low = TO.Select(TO.LessThan(x, TO.Splat(-lambd)), TO.Add(x, TO.Splat(bias)), TO.Zero);
        return TO.Select(TO.GreaterThan(x, TO.Splat(lambd)), TO.Sub(x, TO.Splat(bias)), TO.Select(TO.Equal(x, x), low, x));
    }
}

/// <summary>The OpDefs of the activation functions.</summary>
internal static class Activations
{
    private const float SeluAlpha = 1.67326319217681884765625f;
    private const float SeluGamma = 1.05070102214813232421875f;

    public static readonly OpDef Elu = new("Elu", 6, int.MaxValue, static n => new FloatFunctionOp<EluUnary>(new EluUnary(n.GetFloat("alpha", 1f))));
    public static readonly OpDef Selu = new("Selu", 6, int.MaxValue,
        static n => new FloatFunctionOp<SeluUnary>(new SeluUnary(n.GetFloat("alpha", SeluAlpha), n.GetFloat("gamma", SeluGamma))));
    public static readonly OpDef Celu = new("Celu", 12, int.MaxValue, static n => new FloatFunctionOp<CeluUnary>(new CeluUnary(n.GetFloat("alpha", 1f))));
    public static readonly OpDef Softplus = new("Softplus", 1, int.MaxValue, static _ => new FloatFunctionOp<SoftplusUnary>(default));
    public static readonly OpDef Softsign = new("Softsign", 1, int.MaxValue, static _ => new FloatFunctionOp<SoftsignUnary>(default));
    public static readonly OpDef Mish = new("Mish", 18, int.MaxValue, static _ => new FloatFunctionOp<MishUnary>(default));
    public static readonly OpDef Swish = new("Swish", 24, int.MaxValue, static n => new FloatFunctionOp<SwishUnary>(new SwishUnary(n.GetFloat("alpha", 1f))));
    public static readonly OpDef ThresholdedRelu = new("ThresholdedRelu", 10, int.MaxValue,
        static n => new FloatFunctionOp<ThresholdedReluUnary>(new ThresholdedReluUnary(n.GetFloat("alpha", 1f))));
    public static readonly OpDef Shrink = new("Shrink", 9, int.MaxValue,
        static n => new FloatFunctionOp<ShrinkUnary>(new ShrinkUnary(n.GetFloat("lambd", 0.5f), n.GetFloat("bias", 0f))));
}

/// <summary>PRelu: Y = X for X &gt;= 0 (and NaN), slope * X below; the slope is broadcast to X's shape in one direction only.</summary>
internal sealed class PReluOp : OpKernel
{
    public static readonly OpDef Def = new("PRelu", 7, int.MaxValue, static _ => new PReluOp());

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 2 || !ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects two inputs");
        }

        DType t = ctx.InputType(0);
        if (t != ctx.InputType(1))
        {
            throw ctx.Fail($"the inputs have different element types ({t} and {ctx.InputType(1)})");
        }

        if (t is not (DType.F32 or DType.F64 or DType.I32 or DType.I64))
        {
            throw ctx.Unsupported($"element type {t} is not supported yet");
        }

        if (!Shape.TryBroadcast(ctx.InputShape(0), ctx.InputShape(1), out Shape s) || s != ctx.InputShape(0))
        {
            throw ctx.Fail($"slope shape {ctx.InputShape(1)} cannot be broadcast to the input shape {ctx.InputShape(0)}");
        }

        ctx.SetOutput(0, t, s);
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], s = run.Inputs[1], y = run.Outputs[0];
        switch (y.DType)
        {
            case DType.F32: Broadcasting.Binary<float, float, float, Fn<float>>(y.Shape, x.Shape, s.Shape, x.Data<float>(), s.Data<float>(), y.MutableData<float>()); break;
            case DType.F64: Broadcasting.Binary<double, double, double, Fn<double>>(y.Shape, x.Shape, s.Shape, x.Data<double>(), s.Data<double>(), y.MutableData<double>()); break;
            case DType.I32: Broadcasting.Binary<int, int, int, Fn<int>>(y.Shape, x.Shape, s.Shape, x.Data<int>(), s.Data<int>(), y.MutableData<int>()); break;
            case DType.I64: Broadcasting.Binary<long, long, long, Fn<long>>(y.Shape, x.Shape, s.Shape, x.Data<long>(), s.Data<long>(), y.MutableData<long>()); break;
            default: throw new InvalidOperationException($"Element type {y.DType} reached the kernel although inference rejects it.");
        }
    }

    private readonly struct Fn<T> : IElementFn<T, T, T> where T : unmanaged, System.Numerics.INumber<T>
    {
        public static T Apply(T x, T slope) => x < T.Zero ? unchecked(slope * x) : x;
    }
}

/// <summary>The sine, cosine, tangent and their inverses and hyperbolic forms, on binary32 and binary64.</summary>
/// <remarks>
/// Not exact: the values come from the platform's math library, which the exact mode may not use because its results can differ by
/// a unit in the last place between machines. Binary32 inputs are computed in binary64 and rounded once.
/// </remarks>
internal sealed class TrigOp<TFn> : OpKernel where TFn : IDoubleFn
{
    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        DType t = ctx.InputType(0);
        if (t is not (DType.F32 or DType.F64))
        {
            throw ctx.Unsupported($"element type {t} is not supported yet");
        }

        ctx.SetOutput(0, t, ctx.InputShape(0));
    }

    public override int InPlaceInput => 0;

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        if (y.DType == DType.F32)
        {
            ReadOnlySpan<float> a = x.Data<float>();
            Span<float> b = y.MutableData<float>();
            for (int i = 0; i < a.Length; i++)
            {
                b[i] = (float)TFn.F(a[i]);
            }
        }
        else
        {
            ReadOnlySpan<double> a = x.Data<double>();
            Span<double> b = y.MutableData<double>();
            for (int i = 0; i < a.Length; i++)
            {
                b[i] = TFn.F(a[i]);
            }
        }
    }
}

internal interface IDoubleFn
{
    static abstract double F(double x);
}

internal readonly struct SinFn : IDoubleFn { public static double F(double x) => Math.Sin(x); }

internal readonly struct CosFn : IDoubleFn { public static double F(double x) => Math.Cos(x); }

internal readonly struct TanFn : IDoubleFn { public static double F(double x) => Math.Tan(x); }

internal readonly struct AsinFn : IDoubleFn { public static double F(double x) => Math.Asin(x); }

internal readonly struct AcosFn : IDoubleFn { public static double F(double x) => Math.Acos(x); }

internal readonly struct AtanFn : IDoubleFn { public static double F(double x) => Math.Atan(x); }

internal readonly struct SinhFn : IDoubleFn { public static double F(double x) => Math.Sinh(x); }

internal readonly struct CoshFn : IDoubleFn { public static double F(double x) => Math.Cosh(x); }

internal readonly struct AsinhFn : IDoubleFn { public static double F(double x) => Math.Asinh(x); }

internal readonly struct AcoshFn : IDoubleFn { public static double F(double x) => Math.Acosh(x); }

internal readonly struct AtanhFn : IDoubleFn { public static double F(double x) => Math.Atanh(x); }

/// <summary>The OpDefs of the trigonometric family.</summary>
internal static class Trigonometry
{
    public static readonly OpDef Sin = new("Sin", 7, int.MaxValue, static _ => new TrigOp<SinFn>());
    public static readonly OpDef Cos = new("Cos", 7, int.MaxValue, static _ => new TrigOp<CosFn>());
    public static readonly OpDef Tan = new("Tan", 7, int.MaxValue, static _ => new TrigOp<TanFn>());
    public static readonly OpDef Asin = new("Asin", 7, int.MaxValue, static _ => new TrigOp<AsinFn>());
    public static readonly OpDef Acos = new("Acos", 7, int.MaxValue, static _ => new TrigOp<AcosFn>());
    public static readonly OpDef Atan = new("Atan", 7, int.MaxValue, static _ => new TrigOp<AtanFn>());
    public static readonly OpDef Sinh = new("Sinh", 9, int.MaxValue, static _ => new TrigOp<SinhFn>());
    public static readonly OpDef Cosh = new("Cosh", 9, int.MaxValue, static _ => new TrigOp<CoshFn>());
    public static readonly OpDef Asinh = new("Asinh", 9, int.MaxValue, static _ => new TrigOp<AsinhFn>());
    public static readonly OpDef Acosh = new("Acosh", 9, int.MaxValue, static _ => new TrigOp<AcoshFn>());
    public static readonly OpDef Atanh = new("Atanh", 9, int.MaxValue, static _ => new TrigOp<AtanhFn>());
}
