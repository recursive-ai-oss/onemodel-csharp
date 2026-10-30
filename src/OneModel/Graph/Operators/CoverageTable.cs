using RecursiveAI.OneModel.Graph.Operators.Activation;
using RecursiveAI.OneModel.Graph.Operators.Bitwise;
using RecursiveAI.OneModel.Graph.Operators.Loss;
using RecursiveAI.OneModel.Graph.Operators.Scatter;
using RecursiveAI.OneModel.Graph.Operators.Signal;
using RecursiveAI.OneModel.Graph.Operators.Transformer;
using RecursiveAI.OneModel.Graph.Operators.Vision;

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

        // Data movement.
        BlockShuffleOp.DepthToSpace,
        BlockShuffleOp.SpaceToDepth,
        EyeLikeOp.Def,
        ReverseSequenceOp.Def,
        CenterCropPadOp.Def,
        BitCastOp.Def,
        TensorScatterOp.Def,

        // Numeric operators.
        CumProdOp.Def,
        LpNormalizationOp.Def,
        MeanVarianceNormalizationOp.Def,
        LrnOp.Def,
        DetOp.Def,
        EinsumOp.Def,

        // Spatial operators.
        LpPoolOp.Def,
        MaxUnpoolOp.Def,
        Col2ImOp.Def,

        // Attention.
        OnnxAttentionOp.Def,
        RotaryEmbeddingOp.Def,

        // Spectral operators and windows.
        DftOp.Def,
        StftOp.Def,
        WindowOp.Hann,
        WindowOp.Hamming,
        WindowOp.Blackman,

        // Sampling, detection and sets.
        AffineGridOp.Def,
        GridSampleOp.Def,
        NonMaxSuppressionOp.Def,
        RoiAlignOp.Def,
        UniqueOp.Def,

        // Losses.
        NllLossOp.Nll,
        NllLossOp.SoftmaxCrossEntropy,
    ];
}
