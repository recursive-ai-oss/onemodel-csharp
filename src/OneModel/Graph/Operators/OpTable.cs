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

        // Recurrent family: RNN, GRU, LSTM.
        LstmOp.Def,
        GruOp.Def,
        RnnOp.Def,
    ];
}
