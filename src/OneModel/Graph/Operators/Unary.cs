using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

// The pattern for unary element-wise operators (Neg, Abs, Sigmoid, Tanh, Gelu, ...): output has the input's type and shape, the kernel
// maps each element, and the output may reuse the input's storage.

/// <summary>Relu: Y = X where X is positive, 0 elsewhere; NaN stays NaN.</summary>
internal sealed class ReluOp : OpKernel
{
    public static readonly OpDef Def = new("Relu", 6, int.MaxValue, static _ => new ReluOp());

    public override bool SupportsExact => true;

    public override int InPlaceInput => 0;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        DType t = ctx.InputType(0);
        if (t is not (DType.F32 or DType.F64 or DType.I8 or DType.I16 or DType.I32 or DType.I64))
        {
            throw ctx.Unsupported($"element type {t} is not supported yet");
        }

        ctx.SetOutput(0, t, ctx.InputShape(0));
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        switch (y.DType)
        {
            case DType.F32:
                ExactElementwise.Relu(run.Path, run.Exact, x.Data<float>(), y.MutableData<float>());
                break;
            case DType.F64:
                ExactElementwise.Relu(x.Data<double>(), y.MutableData<double>());
                break;
            case DType.I8:
                ExactElementwise.Relu(x.Data<sbyte>(), y.MutableData<sbyte>());
                break;
            case DType.I16:
                ExactElementwise.Relu(x.Data<short>(), y.MutableData<short>());
                break;
            case DType.I32:
                ExactElementwise.Relu(x.Data<int>(), y.MutableData<int>());
                break;
            case DType.I64:
                ExactElementwise.Relu(x.Data<long>(), y.MutableData<long>());
                break;
            default:
                throw new InvalidOperationException($"Element type {y.DType} reached the kernel although inference rejects it.");
        }
    }
}
