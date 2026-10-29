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

        // Normalisation, softmax, reductions and cumulative sums.
        LayerNormalizationOp.Def,
        RmsNormalizationOp.Def,
        InstanceNormalizationOp.Def,
        GroupNormalizationOp.DefPerGroup,
        GroupNormalizationOp.DefPerChannel,
        SoftmaxOp.DefCoerce,
        SoftmaxOp.DefAxis,
        LogSoftmaxOp.DefCoerce,
        LogSoftmaxOp.DefAxis,
        HardmaxOp.DefCoerce,
        HardmaxOp.DefAxis,
        .. ReduceKernel.Defs,
        ArgExtremeKernel.ArgMaxDef,
        ArgExtremeKernel.ArgMinDef,
        CumSumOp.Def,
    ];
}
