using System;
using System.Numerics;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Elementwise;

/// <summary>Runs one broadcasting binary operation of a <see cref="IBinaryOp"/> on tensors of any numeric element type.</summary>
internal struct BinaryNumberRun<TOp> : INumberVisitor where TOp : IBinaryOp
{
    public Tensor A, B, C;
    public KernelPath Path;
    public bool Exact;

    public void Visit<T>() where T : unmanaged, INumber<T>
    {
        if (typeof(T) == typeof(float))
        {
            Broadcast.Binary<float, FloatRows<TOp>>(Path, Exact, C.Shape, A.Shape, B.Shape, A.Data<float>(), B.Data<float>(), C.MutableData<float>());
        }
        else
        {
            Broadcast.Binary<T, NumberRows<T, TOp>>(Path, Exact, C.Shape, A.Shape, B.Shape, A.Data<T>(), B.Data<T>(), C.MutableData<T>());
        }
    }
}

/// <summary>
/// Max, Min, Sum, Mean over one or more inputs with broadcasting (ONNX opset 8 and later; before that all shapes are equal, which
/// is a special case). Inputs are folded left to right, so the result of a sum depends on the input order and not on anything else.
/// Intermediate results of the fold live in run scratch.
/// </summary>
internal sealed class VariadicOp<TOp> : OpKernel where TOp : IBinaryOp
{
    private readonly bool _mean;
    private readonly bool _floatOnly;

    public VariadicOp(bool floatOnly, bool mean)
    {
        _floatOnly = floatOnly;
        _mean = mean;
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 1 || !ctx.HasInput(0))
        {
            throw ctx.Fail("expects at least one input");
        }

        DType t = ctx.InputType(0);
        bool ok = _floatOnly ? TypeDispatch.IsFloat(t) : TypeDispatch.IsNumber(t);
        if (!ok)
        {
            throw ctx.Unsupported($"element type {t} is not supported yet");
        }

        Shape s = ctx.InputShape(0);
        for (int i = 1; i < ctx.InputCount; i++)
        {
            if (!ctx.HasInput(i))
            {
                throw ctx.Fail($"input {i} is missing");
            }

            if (ctx.InputType(i) != t)
            {
                throw ctx.Fail($"the inputs have different element types ({t} and {ctx.InputType(i)})");
            }

            if (!Shape.TryBroadcast(s, ctx.InputShape(i), out s))
            {
                throw ctx.Fail($"shape {ctx.InputShape(i)} of input {i} cannot be broadcast with {s}");
            }
        }

        ctx.SetOutput(0, t, s);
    }

    public override void Run(in OpRun run)
    {
        ReadOnlySpan<Tensor> inputs = run.Inputs;
        Tensor output = run.Outputs[0];
        if (inputs.Length == 1)
        {
            output.CopyFrom(inputs[0]);
        }
        else
        {
            Tensor acc = inputs[0];
            for (int k = 1; k < inputs.Length; k++)
            {
                bool last = k == inputs.Length - 1;
                Tensor dst = last ? output : run.Scratch(output.DType, Shape.Broadcast(acc.Shape, inputs[k].Shape));
                var v = new BinaryNumberRun<TOp> { A = acc, B = inputs[k], C = dst, Path = run.Path, Exact = run.Exact };
                TypeDispatch.Number(output.DType, ref v);
                acc = dst;
            }
        }

        if (_mean)
        {
            var d = new Divide { Y = output, Count = inputs.Length };
            TypeDispatch.Float(output.DType, ref d);
        }
    }

    private struct Divide : IFloatVisitor
    {
        public Tensor Y;
        public int Count;

        public void Visit<T>() where T : unmanaged, IFloatingPointIeee754<T>
        {
            Span<T> y = Y.MutableData<T>();
            T n = T.CreateTruncating(Count);
            for (int i = 0; i < y.Length; i++)
            {
                y[i] /= n;
            }
        }
    }
}

internal static class Variadics
{
    public static readonly OpDef Max = new("Max", 6, int.MaxValue, static _ => new VariadicOp<MaxArith>(false, false));
    public static readonly OpDef Min = new("Min", 6, int.MaxValue, static _ => new VariadicOp<MinArith>(false, false));
    public static readonly OpDef Sum = new("Sum", 6, int.MaxValue, static _ => new VariadicOp<AddArith>(true, false));
    public static readonly OpDef Mean = new("Mean", 6, int.MaxValue, static _ => new VariadicOp<AddArith>(true, true));
}

/// <summary>
/// Pow: Y = X ** E with broadcasting; the base type X is float, double, int32 or int64 and the exponent type is any of those four
/// (opset 15 allows the mix). The output has the type of X. Not an exact operator (the library pow is not bit-reproducible).
/// </summary>
internal sealed class PowOp : OpKernel
{
    public static readonly OpDef Def = new("Pow", 7, int.MaxValue, static _ => new PowOp());

    public override int InPlaceInput => 0;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 2 || !ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects two inputs");
        }

        DType x = ctx.InputType(0), e = ctx.InputType(1);
        if (!Supported(x) || !Supported(e))
        {
            throw ctx.Unsupported($"Pow on element types {x} and {e} is not supported yet");
        }

        if (!Shape.TryBroadcast(ctx.InputShape(0), ctx.InputShape(1), out Shape s))
        {
            throw ctx.Fail($"shapes {ctx.InputShape(0)} and {ctx.InputShape(1)} cannot be broadcast together");
        }

        ctx.SetOutput(0, x, s);
    }

    private static bool Supported(DType t) => t is DType.F32 or DType.F64 or DType.I32 or DType.I64 or DType.U32 or DType.U64;

    public override void Run(in OpRun run)
    {
        Tensor a = run.Inputs[0], b = run.Inputs[1], c = run.Outputs[0];
        switch (a.DType)
        {
            case DType.F32: Exponent<float>(run, a, b, c); break;
            case DType.F64: Exponent<double>(run, a, b, c); break;
            case DType.I32: Exponent<int>(run, a, b, c); break;
            case DType.I64: Exponent<long>(run, a, b, c); break;
            default: throw new InvalidOperationException($"Element type {a.DType} reached the kernel although inference rejects it.");
        }
    }

    private static void Exponent<TX>(in OpRun run, Tensor a, Tensor b, Tensor c) where TX : unmanaged, INumber<TX>
    {
        switch (b.DType)
        {
            case DType.F32: Go<TX, float>(run, a, b, c); break;
            case DType.F64: Go<TX, double>(run, a, b, c); break;
            case DType.I32: Go<TX, int>(run, a, b, c); break;
            case DType.I64: Go<TX, long>(run, a, b, c); break;
            case DType.U32: Go<TX, uint>(run, a, b, c); break;
            case DType.U64: Go<TX, ulong>(run, a, b, c); break;
            default: throw new InvalidOperationException($"Element type {b.DType} reached the kernel although inference rejects it.");
        }
    }

    private static void Go<TX, TY>(in OpRun run, Tensor a, Tensor b, Tensor c) where TX : unmanaged, INumber<TX> where TY : unmanaged, INumber<TY> =>
        Broadcast.Binary<TX, TY, TX, PowRows<TX, TY>>(run.Path, false, c.Shape, a.Shape, b.Shape, a.Data<TX>(), b.Data<TY>(), c.MutableData<TX>());
}
