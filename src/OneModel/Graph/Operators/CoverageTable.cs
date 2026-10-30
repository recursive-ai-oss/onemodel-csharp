using RecursiveAI.OneModel.Graph.Operators.Activation;
using RecursiveAI.OneModel.Graph.Operators.Bitwise;
using RecursiveAI.OneModel.Graph.Operators.Loss;
using RecursiveAI.OneModel.Graph.Operators.Scatter;

namespace RecursiveAI.OneModel.Graph.Operators;

/// <summary>The operators added to bring the default domain to full backend-test coverage, one line each; OpTable.cs includes this list.</summary>
internal static class CoverageTable
{
    public static readonly Runtime.OpDef[] All =
    [
        // Activations and trigonometry.
        Activations.Elu,
        Activations.Selu,
        Activations.Celu,
        Activations.Softplus,
        Activations.Softsign,
        Activations.Mish,
        Activations.Swish,
        Activations.ThresholdedRelu,
        Activations.Shrink,
        PReluOp.Def,
        Trigonometry.Sin,
        Trigonometry.Cos,
        Trigonometry.Tan,
        Trigonometry.Asin,
        Trigonometry.Acos,
        Trigonometry.Atan,
        Trigonometry.Sinh,
        Trigonometry.Cosh,
        Trigonometry.Asinh,
        Trigonometry.Acosh,
        Trigonometry.Atanh,

        // Bitwise operators, bit shifts and IsInf.
        BitwiseOps.And,
        BitwiseOps.Or,
        BitwiseOps.Xor,
        BitwiseOps.BitShift,
        BitwiseNotOp.Def,
        IsInfOp.Def,

        // Scatter and compress.
        ScatterElementsOp.Def,
        ScatterElementsOp.Legacy,
        CompressOp.Def,

        // Losses.
        NllLossOp.Nll,
        NllLossOp.SoftmaxCrossEntropy,
    ];
}
