using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Elementwise;

// Binary arithmetic beyond Add and Mul: the same base class, the same broadcasting loops.

/// <summary>Sub: C = A - B with broadcasting.</summary>
internal sealed class SubOp : BinaryArithmeticKernel<SubArith>
{
    public static readonly OpDef Def = new("Sub", 7, int.MaxValue, static _ => new SubOp());
}

/// <summary>Div: C = A / B with broadcasting. Integer division truncates toward zero; x / 0 is 0 for integers and follows IEEE for floats.</summary>
internal sealed class DivOp : BinaryArithmeticKernel<DivArith>
{
    public static readonly OpDef Def = new("Div", 7, int.MaxValue, static _ => new DivOp());
}

/// <summary>Mod with fmod = 1: the sign of the result follows the dividend. Floats and integers.</summary>
internal sealed class FmodOp : BinaryArithmeticKernel<FmodArith>
{
    public static readonly OpDef Def = new("Mod", 10, int.MaxValue, static n => n.GetInt("fmod", 0) != 0 ? new FmodOp() : new FloorModOp());
}

/// <summary>Mod with fmod = 0 (the default): the sign of the result follows the divisor. Integers only, as the ONNX operator requires.</summary>
internal sealed class FloorModOp : BinaryArithmeticKernel<FloorModArith>
{
    protected override bool IsSupported(DType t) => TypeDispatch.IsNumber(t) && !TypeDispatch.IsFloat(t);
}
