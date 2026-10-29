using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

// The pattern for binary element-wise operators with broadcasting (Sub, Div, Pow, Max, Min, And, Or, Equal, ...): derive from this class
// with the arithmetic struct for TOp, say which element types the operator accepts, and add one OpDef. Nothing else is needed.

/// <summary>Shape and type rules and the run method shared by binary arithmetic operators.</summary>
internal abstract class BinaryArithmeticKernel<TOp> : OpKernel where TOp : IBinaryOp
{
    /// <summary>One correctly rounded operation per element: identical in both modes apart from the NaN rule.</summary>
    public override bool SupportsExact => true;

    // The output may overwrite input 0 when it has the same size; every output element reads input 0 at its own position only.
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

        if (!IsSupported(t))
        {
            throw ctx.Unsupported($"element type {t} is not supported yet");
        }

        if (!Shape.TryBroadcast(ctx.InputShape(0), ctx.InputShape(1), out Shape s))
        {
            throw ctx.Fail($"shapes {ctx.InputShape(0)} and {ctx.InputShape(1)} cannot be broadcast together");
        }

        ctx.SetOutput(0, t, s);
    }

    public override void Run(in OpRun run)
    {
        Tensor a = run.Inputs[0], b = run.Inputs[1], c = run.Outputs[0];
        Shape sa = a.Shape, sb = b.Shape, so = c.Shape;
        KernelPath path = run.Path;
        bool exact = run.Exact;
        switch (c.DType)
        {
            case DType.F32:
                Broadcast.Binary<float, FloatRows<TOp>>(path, exact, so, sa, sb, a.Data<float>(), b.Data<float>(), c.MutableData<float>());
                break;
            case DType.F64:
                Broadcast.Binary<double, NumberRows<double, TOp>>(path, exact, so, sa, sb, a.Data<double>(), b.Data<double>(), c.MutableData<double>());
                break;
            case DType.I8:
                Broadcast.Binary<sbyte, NumberRows<sbyte, TOp>>(path, exact, so, sa, sb, a.Data<sbyte>(), b.Data<sbyte>(), c.MutableData<sbyte>());
                break;
            case DType.I16:
                Broadcast.Binary<short, NumberRows<short, TOp>>(path, exact, so, sa, sb, a.Data<short>(), b.Data<short>(), c.MutableData<short>());
                break;
            case DType.I32:
                Broadcast.Binary<int, NumberRows<int, TOp>>(path, exact, so, sa, sb, a.Data<int>(), b.Data<int>(), c.MutableData<int>());
                break;
            case DType.I64:
                Broadcast.Binary<long, NumberRows<long, TOp>>(path, exact, so, sa, sb, a.Data<long>(), b.Data<long>(), c.MutableData<long>());
                break;
            case DType.U8:
                Broadcast.Binary<byte, NumberRows<byte, TOp>>(path, exact, so, sa, sb, a.Data<byte>(), b.Data<byte>(), c.MutableData<byte>());
                break;
            case DType.U16:
                Broadcast.Binary<ushort, NumberRows<ushort, TOp>>(path, exact, so, sa, sb, a.Data<ushort>(), b.Data<ushort>(), c.MutableData<ushort>());
                break;
            case DType.U32:
                Broadcast.Binary<uint, NumberRows<uint, TOp>>(path, exact, so, sa, sb, a.Data<uint>(), b.Data<uint>(), c.MutableData<uint>());
                break;
            case DType.U64:
                Broadcast.Binary<ulong, NumberRows<ulong, TOp>>(path, exact, so, sa, sb, a.Data<ulong>(), b.Data<ulong>(), c.MutableData<ulong>());
                break;
            default:
                throw new InvalidOperationException($"Element type {c.DType} reached the kernel although inference rejects it.");
        }
    }

    private static bool IsSupported(DType t) =>
        t is DType.F32 or DType.F64 or DType.I8 or DType.I16 or DType.I32 or DType.I64 or DType.U8 or DType.U16 or DType.U32 or DType.U64;
}

/// <summary>Add: C = A + B with broadcasting.</summary>
internal sealed class AddOp : BinaryArithmeticKernel<AddArith>
{
    // Opset 7 introduced NumPy-style broadcasting (earlier versions broadcast through attributes); 13 and 14 widened the types.
    public static readonly OpDef Def = new("Add", 7, int.MaxValue, static _ => new AddOp());
}

/// <summary>Mul: C = A * B with broadcasting.</summary>
internal sealed class MulOp : BinaryArithmeticKernel<MulArith>
{
    public static readonly OpDef Def = new("Mul", 7, int.MaxValue, static _ => new MulOp());
}
