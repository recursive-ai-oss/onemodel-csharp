using RecursiveAI.OneModel.Graph.Operators.Elementwise;

namespace RecursiveAI.OneModel.Graph.Operators;

/// <summary>
/// The static table of shipped operators: one line per operator, each pointing at the <c>OpDef</c> its file defines. To add an
/// operator, write its file under Graph/Operators/ and add its line here.
/// </summary>
internal static class OpTable
{
    public static readonly Runtime.OpDef[] All =
    [
        AddOp.Def,
        MulOp.Def,
        ReluOp.Def,
        MatMulOp.Def,
        GemmOp.Def,
        ReshapeOp.Def,
        IdentityOp.Def,
        ConstantOp.Def,

        // Shape and indexing family: views, copies, gathers, scatters, generators.
        ShapeOp.Def,
        SizeOp.Def,
        SqueezeOp.Def,
        UnsqueezeOp.Def,
        FlattenOp.Def,
        DropoutOp.Def,
        TransposeOp.Def,
        ConcatOp.Def,
        SplitOp.Def,
        SliceOp.Def,
        ExpandOp.Def,
        TileOp.Def,
        GatherOp.Def,
        GatherElementsOp.Def,
        GatherNDOp.Def,
        ScatterNDOp.Def,
        PadOp.Def,
        ConstantOfShapeOp.Def,
        RangeOp.Def,
        OneHotOp.Def,
        TriluOp.Def,
        NonZeroOp.Def,
        TopKOp.Def,

        // Recurrent family: RNN, GRU, LSTM.
        LstmOp.Def,
        GruOp.Def,
        RnnOp.Def,

        // Element-wise, activations, comparisons, logic, Where and Cast.
        SubOp.Def,
        DivOp.Def,
        FmodOp.Def,
        PowOp.Def,
        Variadics.Max,
        Variadics.Min,
        Variadics.Sum,
        Variadics.Mean,
        FloatFunctions.Exp,
        FloatFunctions.Log,
        FloatFunctions.Tanh,
        FloatFunctions.Sigmoid,
        FloatFunctions.Erf,
        FloatFunctions.HardSigmoid,
        FloatFunctions.HardSwish,
        FloatFunctions.Gelu,
        FloatFunctions.LeakyRelu,
        NumericFunctions.Neg,
        NumericFunctions.Abs,
        NumericFunctions.Sign,
        NumericFunctions.Reciprocal,
        NumericFunctions.Sqrt,
        NumericFunctions.Floor,
        NumericFunctions.Ceil,
        NumericFunctions.Round,
        ClipOp.Def,
        Comparisons.Equal,
        Comparisons.Less,
        Comparisons.Greater,
        Comparisons.LessOrEqual,
        Comparisons.GreaterOrEqual,
        Logic.And,
        Logic.Or,
        Logic.Xor,
        NotOp.Def,
        WhereOp.Def,
        CastOp.Def,
        CastLikeOp.Def,
    ];
}
