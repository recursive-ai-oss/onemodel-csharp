using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Elementwise;

/// <summary>Shape and type rules shared by the unary element-wise operators: one input, an output of the same type and shape.</summary>
internal abstract class UnaryKernelBase : OpKernel
{
    public override bool SupportsExact => true;

    public override int InPlaceInput => 0;

    protected abstract bool IsSupported(DType t);

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        DType t = ctx.InputType(0);
        if (!IsSupported(t))
        {
            throw ctx.Unsupported($"element type {t} is not supported yet");
        }

        ctx.SetOutput(0, t, ctx.InputShape(0));
    }
}

/// <summary>A single-precision function from <see cref="IFloatUnary"/>, vectorized, exact and fast.</summary>
internal sealed class FloatFunctionOp<TOp>(TOp op) : UnaryKernelBase where TOp : struct, IFloatUnary
{
    protected override bool IsSupported(DType t) => t == DType.F32;

    public override void Run(in OpRun run) =>
        FloatUnary.Run(run.Path, run.Exact, op, run.Inputs[0].Data<float>(), run.Outputs[0].MutableData<float>());
}

/// <summary>Namespace for the OpDefs of the float functions.</summary>
internal static class FloatFunctions
{
    public static readonly OpDef Exp = new("Exp", 6, int.MaxValue, static _ => new FloatFunctionOp<ExpUnary>(default));
    public static readonly OpDef Log = new("Log", 6, int.MaxValue, static _ => new FloatFunctionOp<LogUnary>(default));
    public static readonly OpDef Tanh = new("Tanh", 6, int.MaxValue, static _ => new FloatFunctionOp<TanhUnary>(default));
    public static readonly OpDef Sigmoid = new("Sigmoid", 6, int.MaxValue, static _ => new FloatFunctionOp<SigmoidUnary>(default));
    public static readonly OpDef Erf = new("Erf", 9, int.MaxValue, static _ => new FloatFunctionOp<ErfUnary>(default));
    public static readonly OpDef HardSwish = new("HardSwish", 14, int.MaxValue, static _ => new FloatFunctionOp<HardSwishUnary>(default));

    public static readonly OpDef LeakyRelu = new("LeakyRelu", 6, int.MaxValue,
        static n => new FloatFunctionOp<LeakyReluUnary>(new LeakyReluUnary(n.GetFloat("alpha", 0.01f))));

    public static readonly OpDef HardSigmoid = new("HardSigmoid", 6, int.MaxValue,
        static n => new FloatFunctionOp<HardSigmoidUnary>(new HardSigmoidUnary(n.GetFloat("alpha", 0.2f), n.GetFloat("beta", 0.5f))));

    public static readonly OpDef Gelu = new("Gelu", 20, int.MaxValue, static n => n.GetString("approximate", "none") switch
    {
        "none" => new FloatFunctionOp<GeluUnary>(new GeluUnary(false)),
        "tanh" => new FloatFunctionOp<GeluUnary>(new GeluUnary(true)),
        var other => throw n.Error($"attribute 'approximate' is '{other}', expected 'none' or 'tanh'"),
    });
}

/// <summary>Neg, Abs, Sign on any numeric type (Neg: signed types).</summary>
internal sealed class NumericUnaryOp<TOp>(bool signedOnly) : UnaryKernelBase where TOp : INumberUnary
{
    protected override bool IsSupported(DType t) => TypeDispatch.IsNumber(t) && (!signedOnly || TypeDispatch.IsSigned(t));

    public override void Run(in OpRun run)
    {
        var v = new Visitor { X = run.Inputs[0], Y = run.Outputs[0], Canonical = run.Exact };
        TypeDispatch.Number(v.Y.DType, ref v);
    }

    private struct Visitor : INumberVisitor
    {
        public Tensor X, Y;
        public bool Canonical;

        public void Visit<T>() where T : unmanaged, System.Numerics.INumber<T> =>
            UnaryLoops.Number<T, TOp>(Canonical, X.Data<T>(), Y.MutableData<T>());
    }
}

/// <summary>Reciprocal, Sqrt, Floor, Ceil, Round on float16, float and double: one IEEE operation per element.</summary>
internal sealed class FloatScalarOp<TOp> : UnaryKernelBase where TOp : IFloatScalarUnary
{
    protected override bool IsSupported(DType t) => TypeDispatch.IsFloat(t);

    public override void Run(in OpRun run)
    {
        var v = new Visitor { X = run.Inputs[0], Y = run.Outputs[0], Canonical = run.Exact };
        TypeDispatch.Float(v.Y.DType, ref v);
    }

    private struct Visitor : IFloatVisitor
    {
        public Tensor X, Y;
        public bool Canonical;

        public void Visit<T>() where T : unmanaged, System.Numerics.IFloatingPointIeee754<T> =>
            UnaryLoops.Float<T, TOp>(Canonical, X.Data<T>(), Y.MutableData<T>());
    }
}

internal static class NumericFunctions
{
    public static readonly OpDef Neg = new("Neg", 6, int.MaxValue, static _ => new NumericUnaryOp<NegUnary>(true));
    public static readonly OpDef Abs = new("Abs", 6, int.MaxValue, static _ => new NumericUnaryOp<AbsUnary>(false));
    public static readonly OpDef Sign = new("Sign", 9, int.MaxValue, static _ => new NumericUnaryOp<SignUnary>(false));
    public static readonly OpDef Reciprocal = new("Reciprocal", 6, int.MaxValue, static _ => new FloatScalarOp<ReciprocalUnary>());
    public static readonly OpDef Sqrt = new("Sqrt", 6, int.MaxValue, static _ => new FloatScalarOp<SqrtUnary>());
    public static readonly OpDef Floor = new("Floor", 6, int.MaxValue, static _ => new FloatScalarOp<FloorUnary>());
    public static readonly OpDef Ceil = new("Ceil", 6, int.MaxValue, static _ => new FloatScalarOp<CeilUnary>());
    public static readonly OpDef Round = new("Round", 11, int.MaxValue, static _ => new FloatScalarOp<RoundUnary>());
}

/// <summary>
/// Clip: Y = min(max(X, lo), hi). Before opset 11 the bounds are the attributes min and max (floats); from 11 they are the optional
/// scalar inputs 1 and 2, which may be runtime values. An omitted bound is unbounded. A NaN input stays NaN.
/// </summary>
internal sealed class ClipOp : UnaryKernelBase
{
    public static readonly OpDef Def = new("Clip", 6, int.MaxValue, static n => new ClipOp(n));

    private readonly bool _attributes;
    private readonly bool _hasLoAttr, _hasHiAttr;
    private readonly float _loAttr, _hiAttr;

    private ClipOp(NodeInfo n)
    {
        _attributes = n.Opset < 11;
        if (_attributes)
        {
            _hasLoAttr = n.Has("min");
            _hasHiAttr = n.Has("max");
            _loAttr = n.GetFloat("min", float.MinValue);
            _hiAttr = n.GetFloat("max", float.MaxValue);
        }
    }

    protected override bool IsSupported(DType t) => TypeDispatch.IsNumber(t);

    public override void Infer(InferContext ctx)
    {
        base.Infer(ctx);
        DType t = ctx.InputType(0);
        for (int i = 1; i <= 2; i++)
        {
            if (ctx.HasInput(i))
            {
                if (ctx.InputType(i) != t)
                {
                    throw ctx.Fail($"bound input {i} has element type {ctx.InputType(i)}, expected {t}");
                }

                if (ctx.InputShape(i).Length != 1)
                {
                    throw ctx.Fail($"bound input {i} must be a scalar, got shape {ctx.InputShape(i)}");
                }
            }
        }
    }

    public override void Run(in OpRun run)
    {
        var v = new Visitor
        {
            X = run.Inputs[0],
            Y = run.Outputs[0],
            Lo = run.Inputs.Length > 1 ? run.Inputs[1] : default,
            Hi = run.Inputs.Length > 2 ? run.Inputs[2] : default,
            Owner = this,
        };
        TypeDispatch.Number(v.Y.DType, ref v);
    }

    private struct Visitor : INumberVisitor
    {
        public Tensor X, Y, Lo, Hi;
        public ClipOp Owner;

        public void Visit<T>() where T : unmanaged, System.Numerics.INumber<T>
        {
            bool hasLo, hasHi;
            T lo = T.Zero, hi = T.Zero;
            if (Owner._attributes)
            {
                hasLo = Owner._hasLoAttr;
                hasHi = Owner._hasHiAttr;
                lo = T.CreateSaturating(Owner._loAttr);
                hi = T.CreateSaturating(Owner._hiAttr);
            }
            else
            {
                hasLo = !Lo.IsEmpty;
                hasHi = !Hi.IsEmpty;
                if (hasLo)
                {
                    lo = Lo.Data<T>()[0];
                }

                if (hasHi)
                {
                    hi = Hi.Data<T>()[0];
                }
            }

            // An omitted bound is the type's finite extreme, not infinity: ONNX Runtime clips +-inf to +-MaxValue even with no bounds.
            if (NumKind.IsFloat<T>())
            {
                if (!hasLo)
                {
                    hasLo = true;
                    lo = NumKind.LowestFinite<T>();
                }

                if (!hasHi)
                {
                    hasHi = true;
                    hi = NumKind.HighestFinite<T>();
                }
            }

            UnaryLoops.Clip(X.Data<T>(), Y.MutableData<T>(), hasLo, lo, hasHi, hi);
        }
    }
}
