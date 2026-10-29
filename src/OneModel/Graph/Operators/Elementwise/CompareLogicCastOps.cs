using System;
using System.Numerics;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Elementwise;

/// <summary>Comparison with broadcasting: two inputs of one numeric type, a bool output. Integer and bool results are exact by construction.</summary>
internal sealed class CompareOp<TCmp> : OpKernel where TCmp : ICompare
{
    private readonly bool _allowBool;
    private readonly string _what;

    public CompareOp(bool allowBool, string what)
    {
        _allowBool = allowBool;
        _what = what;
    }

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

        if (!TypeDispatch.IsNumber(t) && !(_allowBool && t == DType.Bool))
        {
            throw ctx.Unsupported($"{_what} on element type {t} is not supported yet");
        }

        if (!Shape.TryBroadcast(ctx.InputShape(0), ctx.InputShape(1), out Shape s))
        {
            throw ctx.Fail($"shapes {ctx.InputShape(0)} and {ctx.InputShape(1)} cannot be broadcast together");
        }

        ctx.SetOutput(0, DType.Bool, s);
    }

    public override void Run(in OpRun run)
    {
        Tensor a = run.Inputs[0], b = run.Inputs[1], c = run.Outputs[0];
        if (a.DType == DType.Bool)
        {
            Broadcast.Binary<bool, bool, bool, BoolRows<EqualBool>>(run.Path, false, c.Shape, a.Shape, b.Shape, a.Data<bool>(), b.Data<bool>(), c.MutableData<bool>());
            return;
        }

        var v = new Visitor { A = a, B = b, C = c, Path = run.Path };
        TypeDispatch.Number(a.DType, ref v);
    }

    private struct Visitor : INumberVisitor
    {
        public Tensor A, B, C;
        public KernelPath Path;

        public void Visit<T>() where T : unmanaged, INumber<T> =>
            Broadcast.Binary<T, T, bool, CompareRows<T, TCmp>>(Path, false, C.Shape, A.Shape, B.Shape, A.Data<T>(), B.Data<T>(), C.MutableData<bool>());
    }
}

/// <summary>Namespace for the OpDefs of the comparisons.</summary>
internal static class Comparisons
{
    public static readonly OpDef Equal = new("Equal", 7, int.MaxValue, static _ => new CompareOp<EqualCmp>(true, "Equal"));
    public static readonly OpDef Less = new("Less", 7, int.MaxValue, static _ => new CompareOp<LessCmp>(false, "Less"));
    public static readonly OpDef Greater = new("Greater", 7, int.MaxValue, static _ => new CompareOp<GreaterCmp>(false, "Greater"));
    public static readonly OpDef LessOrEqual = new("LessOrEqual", 12, int.MaxValue, static _ => new CompareOp<LessOrEqualCmp>(false, "LessOrEqual"));
    public static readonly OpDef GreaterOrEqual = new("GreaterOrEqual", 12, int.MaxValue, static _ => new CompareOp<GreaterOrEqualCmp>(false, "GreaterOrEqual"));
}

/// <summary>And, Or, Xor on bool with broadcasting.</summary>
internal sealed class LogicOp<TOp> : OpKernel where TOp : IBoolOp
{
    public override bool SupportsExact => true;

    public override int InPlaceInput => 0;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 2 || !ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects two inputs");
        }

        if (ctx.InputType(0) != DType.Bool || ctx.InputType(1) != DType.Bool)
        {
            throw ctx.Fail($"expects bool inputs, got {ctx.InputType(0)} and {ctx.InputType(1)}");
        }

        if (!Shape.TryBroadcast(ctx.InputShape(0), ctx.InputShape(1), out Shape s))
        {
            throw ctx.Fail($"shapes {ctx.InputShape(0)} and {ctx.InputShape(1)} cannot be broadcast together");
        }

        ctx.SetOutput(0, DType.Bool, s);
    }

    public override void Run(in OpRun run)
    {
        Tensor a = run.Inputs[0], b = run.Inputs[1], c = run.Outputs[0];
        Broadcast.Binary<bool, bool, bool, BoolRows<TOp>>(run.Path, false, c.Shape, a.Shape, b.Shape, a.Data<bool>(), b.Data<bool>(), c.MutableData<bool>());
    }
}

/// <summary>Not: logical negation of a bool tensor.</summary>
internal sealed class NotOp : OpKernel
{
    public static readonly OpDef Def = new("Not", 1, int.MaxValue, static _ => new NotOp());

    public override bool SupportsExact => true;

    public override int InPlaceInput => 0;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        if (ctx.InputType(0) != DType.Bool)
        {
            throw ctx.Fail($"expects a bool input, got {ctx.InputType(0)}");
        }

        ctx.SetOutput(0, DType.Bool, ctx.InputShape(0));
    }

    public override void Run(in OpRun run)
    {
        ReadOnlySpan<bool> x = run.Inputs[0].Data<bool>();
        Span<bool> y = run.Outputs[0].MutableData<bool>();
        for (int i = 0; i < x.Length; i++)
        {
            y[i] = !x[i];
        }
    }
}

internal static class Logic
{
    public static readonly OpDef And = new("And", 7, int.MaxValue, static _ => new LogicOp<AndBool>());
    public static readonly OpDef Or = new("Or", 7, int.MaxValue, static _ => new LogicOp<OrBool>());
    public static readonly OpDef Xor = new("Xor", 7, int.MaxValue, static _ => new LogicOp<XorBool>());
}

/// <summary>Where: Y = condition ? X : Y over three broadcast inputs, for any stored element type.</summary>
internal sealed class WhereOp : OpKernel
{
    public static readonly OpDef Def = new("Where", 9, int.MaxValue, static _ => new WhereOp());

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 3 || !ctx.HasInput(0) || !ctx.HasInput(1) || !ctx.HasInput(2))
        {
            throw ctx.Fail("expects three inputs");
        }

        if (ctx.InputType(0) != DType.Bool)
        {
            throw ctx.Fail($"the condition must be bool, got {ctx.InputType(0)}");
        }

        DType t = ctx.InputType(1);
        if (t != ctx.InputType(2))
        {
            throw ctx.Fail($"the value inputs have different element types ({t} and {ctx.InputType(2)})");
        }

        if (!Shape.TryBroadcast(ctx.InputShape(0), ctx.InputShape(1), out Shape s) || !Shape.TryBroadcast(s, ctx.InputShape(2), out s))
        {
            throw ctx.Fail($"shapes {ctx.InputShape(0)}, {ctx.InputShape(1)} and {ctx.InputShape(2)} cannot be broadcast together");
        }

        ctx.SetOutput(0, t, s);
    }

    public override void Run(in OpRun run)
    {
        var v = new Visitor { C = run.Inputs[0], X = run.Inputs[1], Y = run.Inputs[2], O = run.Outputs[0] };
        TypeDispatch.Any(v.O.DType, ref v);
    }

    private struct Visitor : ITypeVisitor
    {
        public Tensor C, X, Y, O;

        public void Visit<T>() where T : unmanaged =>
            SelectKernel.Where(O.Shape, C.Shape, X.Shape, Y.Shape, C.Data<bool>(), X.Data<T>(), Y.Data<T>(), O.MutableData<T>());
    }
}

/// <summary>Cast: converts every element to the type named by the attribute <c>to</c>. The rules are in <see cref="CastKernel"/>.</summary>
internal sealed class CastOp : OpKernel
{
    public static readonly OpDef Def = new("Cast", 6, int.MaxValue, static n => new CastOp(ReadTarget(n)));

    private readonly DType _to;

    private CastOp(DType to) => _to = to;

    public override bool SupportsExact => true;

    internal static DType ReadTarget(NodeInfo n)
    {
        if (!n.Has("to"))
        {
            throw n.Error("missing attribute 'to'");
        }

        long code = n.GetInt("to", 0);
        if (code < 0 || code > int.MaxValue || !DTypes.TryFromOnnx((int)code, out DType t))
        {
            throw n.Unsupported($"cast to ONNX element type {DTypes.OnnxName((int)Math.Clamp(code, 0, int.MaxValue))} is not supported");
        }

        return t;
    }

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        ctx.SetOutput(0, _to, ctx.InputShape(0));
    }

    public override void Run(in OpRun run) => CastKernel.Run(run.Inputs[0], run.Outputs[0]);
}

/// <summary>CastLike: Cast to the element type of the second input; the values of the second input are not read.</summary>
internal sealed class CastLikeOp : OpKernel
{
    public static readonly OpDef Def = new("CastLike", 15, int.MaxValue, static _ => new CastLikeOp());

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 2 || !ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects two inputs");
        }

        ctx.SetOutput(0, ctx.InputType(1), ctx.InputShape(0));
    }

    public override void Run(in OpRun run) => CastKernel.Run(run.Inputs[0], run.Outputs[0]);
}
