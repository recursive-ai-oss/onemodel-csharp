using System;
using System.Numerics;
using RecursiveAI.OneModel.Graph.Operators.Activation;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Bitwise;

/// <summary>Called with the concrete element type of an integer tensor.</summary>
internal interface IIntegerVisitor
{
    void Visit<T>() where T : unmanaged, IBinaryInteger<T>;
}

internal static class IntegerDispatch
{
    public static bool IsInteger(DType t) => t is DType.I8 or DType.I16 or DType.I32 or DType.I64 or DType.U8 or DType.U16 or DType.U32 or DType.U64;

    public static bool IsSigned(DType t) => t is DType.I8 or DType.I16 or DType.I32 or DType.I64;

    public static void Run<TV>(DType t, ref TV v) where TV : struct, IIntegerVisitor
    {
        switch (t)
        {
            case DType.I8: v.Visit<sbyte>(); break;
            case DType.I16: v.Visit<short>(); break;
            case DType.I32: v.Visit<int>(); break;
            case DType.I64: v.Visit<long>(); break;
            case DType.U8: v.Visit<byte>(); break;
            case DType.U16: v.Visit<ushort>(); break;
            case DType.U32: v.Visit<uint>(); break;
            case DType.U64: v.Visit<ulong>(); break;
            default: throw new InvalidOperationException($"Element type {t} is not an integer type.");
        }
    }
}

internal interface IBitFn
{
    static abstract T Apply<T>(T a, T b) where T : unmanaged, IBinaryInteger<T>;
}

internal readonly struct AndFn : IBitFn { public static T Apply<T>(T a, T b) where T : unmanaged, IBinaryInteger<T> => a & b; }

internal readonly struct OrFn : IBitFn { public static T Apply<T>(T a, T b) where T : unmanaged, IBinaryInteger<T> => a | b; }

internal readonly struct XorFn : IBitFn { public static T Apply<T>(T a, T b) where T : unmanaged, IBinaryInteger<T> => a ^ b; }

// BitShift: the shifted-out bits are lost and a shift by the type's width or more gives 0.
internal readonly struct ShiftLeftFn : IBitFn
{
    public static T Apply<T>(T a, T b) where T : unmanaged, IBinaryInteger<T> =>
        ulong.CreateSaturating(b) >= (ulong)(T.AllBitsSet.GetByteCount() * 8) ? T.Zero : a << int.CreateTruncating(b);
}

internal readonly struct ShiftRightFn : IBitFn
{
    public static T Apply<T>(T a, T b) where T : unmanaged, IBinaryInteger<T> =>
        ulong.CreateSaturating(b) >= (ulong)(T.AllBitsSet.GetByteCount() * 8) ? T.Zero : a >> int.CreateTruncating(b);
}

/// <summary>An integer operation with broadcasting: BitwiseAnd, BitwiseOr, BitwiseXor (opset 18) and BitShift (opset 11, unsigned types).</summary>
internal sealed class BitBinaryOp<TFn> : OpKernel where TFn : IBitFn
{
    private readonly bool _unsignedOnly;

    public BitBinaryOp(bool unsignedOnly)
    {
        _unsignedOnly = unsignedOnly;
    }

    public override bool SupportsExact => true;

    public override int InPlaceInput => 0;

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

        if (!IntegerDispatch.IsInteger(t) || (_unsignedOnly && IntegerDispatch.IsSigned(t)))
        {
            throw ctx.Unsupported($"element type {t} is not supported");
        }

        if (!Shape.TryBroadcast(ctx.InputShape(0), ctx.InputShape(1), out Shape s))
        {
            throw ctx.Fail($"shapes {ctx.InputShape(0)} and {ctx.InputShape(1)} cannot be broadcast together");
        }

        ctx.SetOutput(0, t, s);
    }

    public override void Run(in OpRun run)
    {
        var v = new Visitor { A = run.Inputs[0], B = run.Inputs[1], C = run.Outputs[0] };
        IntegerDispatch.Run(v.C.DType, ref v);
    }

    private struct Visitor : IIntegerVisitor
    {
        public Tensor A, B, C;

        public void Visit<T>() where T : unmanaged, IBinaryInteger<T> =>
            Broadcasting.Binary<T, T, T, Fn<T>>(C.Shape, A.Shape, B.Shape, A.Data<T>(), B.Data<T>(), C.MutableData<T>());
    }

    private readonly struct Fn<T> : IElementFn<T, T, T> where T : unmanaged, IBinaryInteger<T>
    {
        public static T Apply(T a, T b) => TFn.Apply(a, b);
    }
}

/// <summary>BitwiseNot: the complement of every bit.</summary>
internal sealed class BitwiseNotOp : OpKernel
{
    public static readonly OpDef Def = new("BitwiseNot", 18, int.MaxValue, static _ => new BitwiseNotOp());

    public override bool SupportsExact => true;

    public override int InPlaceInput => 0;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        if (!IntegerDispatch.IsInteger(ctx.InputType(0)))
        {
            throw ctx.Unsupported($"element type {ctx.InputType(0)} is not supported");
        }

        ctx.SetOutput(0, ctx.InputType(0), ctx.InputShape(0));
    }

    public override void Run(in OpRun run)
    {
        var v = new Visitor { X = run.Inputs[0], Y = run.Outputs[0] };
        IntegerDispatch.Run(v.Y.DType, ref v);
    }

    private struct Visitor : IIntegerVisitor
    {
        public Tensor X, Y;

        public void Visit<T>() where T : unmanaged, IBinaryInteger<T>
        {
            ReadOnlySpan<T> x = X.Data<T>();
            Span<T> y = Y.MutableData<T>();
            for (int i = 0; i < x.Length; i++)
            {
                y[i] = ~x[i];
            }
        }
    }
}

/// <summary>IsInf: true where the input is infinite; detect_negative and detect_positive (both default 1) restrict which sign counts.</summary>
internal sealed class IsInfOp : OpKernel
{
    public static readonly OpDef Def = new("IsInf", 10, int.MaxValue, static n => new IsInfOp(n.GetInt("detect_negative", 1) != 0, n.GetInt("detect_positive", 1) != 0));

    private readonly bool _negative, _positive;

    private IsInfOp(bool negative, bool positive)
    {
        _negative = negative;
        _positive = positive;
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        if (ctx.InputType(0) is not (DType.F32 or DType.F64))
        {
            throw ctx.Unsupported($"element type {ctx.InputType(0)} is not supported yet");
        }

        ctx.SetOutput(0, DType.Bool, ctx.InputShape(0));
    }

    public override void Run(in OpRun run)
    {
        Span<bool> y = run.Outputs[0].MutableData<bool>();
        if (run.Inputs[0].DType == DType.F32)
        {
            ReadOnlySpan<float> x = run.Inputs[0].Data<float>();
            for (int i = 0; i < x.Length; i++)
            {
                y[i] = (_positive && float.IsPositiveInfinity(x[i])) || (_negative && float.IsNegativeInfinity(x[i]));
            }
        }
        else
        {
            ReadOnlySpan<double> x = run.Inputs[0].Data<double>();
            for (int i = 0; i < x.Length; i++)
            {
                y[i] = (_positive && double.IsPositiveInfinity(x[i])) || (_negative && double.IsNegativeInfinity(x[i]));
            }
        }
    }
}

/// <summary>The OpDefs of the bitwise operators.</summary>
internal static class BitwiseOps
{
    public static readonly OpDef And = new("BitwiseAnd", 18, int.MaxValue, static _ => new BitBinaryOp<AndFn>(false));
    public static readonly OpDef Or = new("BitwiseOr", 18, int.MaxValue, static _ => new BitBinaryOp<OrFn>(false));
    public static readonly OpDef Xor = new("BitwiseXor", 18, int.MaxValue, static _ => new BitBinaryOp<XorFn>(false));

    public static readonly OpDef BitShift = new("BitShift", 11, int.MaxValue, static n => n.GetString("direction", "") switch
    {
        "LEFT" => new BitBinaryOp<ShiftLeftFn>(true),
        "RIGHT" => new BitBinaryOp<ShiftRightFn>(true),
        _ => throw n.Error("attribute 'direction' must be LEFT or RIGHT"),
    });
}
