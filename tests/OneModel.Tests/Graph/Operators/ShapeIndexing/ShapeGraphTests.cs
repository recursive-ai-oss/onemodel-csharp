using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.ShapeIndexing;

/// <summary>Shape-math chains fold at plan time, and the data-movement operators run without allocating.</summary>
public class ShapeGraphTests
{
    private static string[] StepsOf(byte[] model, DType type, params int[] shape)
    {
        using IrModel ir = OnnxImporter.Load(model);
        using CompiledGraph g = CompiledGraph.Create(ir.Graph, OpRegistry.Default, ExecutionMode.Fast, default);
        var env = new SessionEnv(ExecutionMode.Fast, KernelPaths.Best(), 1, RecursiveAI.OneModel.Tensors.BufferPool.Shared);
        ExecutionPlan plan = ExecutionPlan.Build(g, env, [type], [Shape.Create(shape)], default);
        try
        {
            return plan.Steps.Select(s => s.Node.Info.OpType).ToArray();
        }
        finally
        {
            plan.Release();
        }
    }

    // BERT's head split: reshape [N, S, 768] to [N, S, 12, 64] with the first two extents read from the input's shape.
    private static ModelBuilder BertSplit() => new ModelBuilder(17)
        .Input("x", 1, "N", "S", 768).Output("y", 1)
        .Initializer("i0", [0L]).Initializer("i1", [1L]).Initializer("heads", [12L, 64L], 2).Initializer("ax", [0L], 1)
        .Node("Shape", ["x"], ["sh"])
        .Node("Gather", ["sh", "i0"], ["n"], ("axis", 0L)).Node("Gather", ["sh", "i1"], ["s"], ("axis", 0L))
        .Node("Unsqueeze", ["n", "ax"], ["n1"]).Node("Unsqueeze", ["s", "ax"], ["s1"])
        .Node("Concat", ["n1", "s1", "heads"], ["target"], ("axis", 0L))
        .Node("Reshape", ["x", "target"], ["r"])
        .Node("Transpose", ["r"], ["y"], ("perm", new long[] { 0, 2, 1, 3 }));

    [Fact]
    public void BertHeadSplitChainFoldsAwayLeavingOnlyTheDataMovement()
    {
        string[] steps = StepsOf(BertSplit().Build(), DType.F32, 4, 7, 768);
        Assert.Equal(["Reshape", "Transpose"], steps);

        using var s = Session.Load(new MemoryStream(BertSplit().Build()));
        float[] x = Enumerable.Range(0, 4 * 7 * 768).Select(i => (float)i).ToArray();
        float[] y = new float[x.Length];
        Shape r = s.Run("x", x.AsSpan(), [4, 7, 768], "y", y.AsSpan());
        Assert.Equal(Shape.Create(4, 12, 7, 64), r);
        // y[n, h, t, d] = x[n, t, h * 64 + d]
        Assert.Equal(x[(((2 * 7) + 5) * 768) + (9 * 64) + 33], y[(((2 * 12) + 9) * 7 * 64) + (5 * 64) + 33]);
    }

    [Fact]
    public void WhisperStylePositionAndMaskChainsFold()
    {
        // positions = Range(0, T, 1); pos = Gather(table, positions); out = Gather(embed, tokens) + pos
        // mask = Trilu(ConstantOfShape([T, T]) filled with 1, k=1)
        float[] table = Enumerable.Range(0, 32 * 4).Select(i => (float)i).ToArray();
        float[] embed = Enumerable.Range(0, 10 * 4).Select(i => i * 0.5f).ToArray();
        byte[] model = new ModelBuilder(17)
            .Input("tokens", 7, "B", "T").Output("y", 1).Output("mask", 1)
            .Initializer("table", table, 32, 4).Initializer("embed", embed, 10, 4)
            .Initializer("one", [1L]).Initializer("zero", [0L]).Initializer("delta", [1L]).Initializer("k", [1L], 1).Initializer("ax", [0L], 1)
            .Node("Shape", ["tokens"], ["sh"])
            .Node("Gather", ["sh", "one"], ["t"], ("axis", 0L))
            .Node("Range", ["zero", "t", "delta"], ["positions"])
            .Node("Gather", ["table", "positions"], ["pos"], ("axis", 0L))
            .Node("Gather", ["embed", "tokens"], ["emb"], ("axis", 0L))
            .Node("Add", ["emb", "pos"], ["y"])
            .Node("Unsqueeze", ["t", "ax"], ["t1"])
            .Node("Concat", ["t1", "t1"], ["sq"], ("axis", 0L))
            .Node("ConstantOfShape", ["sq"], ["ones"], ("value", ModelBuilder.FloatTensor("v", [1f], 1)))
            .Node("Trilu", ["ones", "k"], ["mask"], ("upper", 1L)).Build();

        string[] steps = StepsOf(model, DType.I64, 2, 5);
        Assert.Equal(["Gather", "Add"], steps);

        using var s = Session.Load(new MemoryStream(model));
        using Tensor tokens = Tensor.FromSpan<long>([1, 2, 3, 4, 5, 9, 8, 7, 6, 0], Shape.Create(2, 5));
        var outs = new Tensor[2];
        s.Run(["tokens"], [tokens], ["y", "mask"], outs);
        Assert.Equal(Shape.Create(2, 5, 4), outs[0].Shape);
        Assert.Equal(19.5f, outs[0].ToArray<float>()[(1 * 5 * 4) + 1]); // embed[9,1] + table[0,1]
        Assert.Equal(Shape.Create(5, 5), outs[1].Shape);
        float[] mask = outs[1].ToArray<float>();
        for (int r = 0; r < 5; r++)
        {
            for (int c = 0; c < 5; c++)
            {
                Assert.Equal(c >= r + 1 ? 1f : 0f, mask[(r * 5) + c]);
            }
        }

        outs[0].Dispose();
        outs[1].Dispose();
    }

    private static long AllocatedBytes(Action loop)
    {
        long best = long.MaxValue;
        for (int attempt = 0; attempt < 3 && best != 0; attempt++)
        {
            long before = GC.GetAllocatedBytesForCurrentThread();
            loop();
            best = Math.Min(best, GC.GetAllocatedBytesForCurrentThread() - before);
        }

        return best;
    }

    [Fact]
    public void MovementOperatorsAllocateNothingInSteadyState()
    {
        // x [4, 6, 8] -> Transpose -> Slice (runtime-shaped starts are constants here) -> Gather -> Concat -> Pad(reflect) -> TopK
        byte[] model = new ModelBuilder(17)
            .Input("x", 1, 4, 6, 8).Output("vals", 1).Output("idx", 7)
            .Initializer("starts", [1L], 1).Initializer("ends", [5L], 1).Initializer("axes", [1L], 1)
            .Initializer("gi", [0L, 2L, 1L], 3).Initializer("pads", [0L, 1L, 0L, 0L, 2L, 0L], 6).Initializer("k", [3L], 1)
            .Node("Transpose", ["x"], ["t"], ("perm", new long[] { 2, 1, 0 }))
            .Node("Slice", ["t", "starts", "ends", "axes"], ["sl"])
            .Node("Gather", ["sl", "gi"], ["g"], ("axis", 0L))
            .Node("Concat", ["g", "g"], ["c"], ("axis", 2L))
            .Node("Pad", ["c", "pads"], ["p"], ("mode", "reflect"))
            .Node("TopK", ["p", "k"], ["vals", "idx"], ("axis", -1L)).Build();
        using var s = Session.Load(new MemoryStream(model), new SessionOptions { IntraOpThreads = 1 });
        using Tensor x = Tensor.FromSpan<float>(Enumerable.Range(0, 4 * 6 * 8).Select(i => (float)((i * 7) % 31)).ToArray(), Shape.Create(4, 6, 8));
        Tensor[] ins = [x];
        string[] inNames = ["x"], outNames = ["vals", "idx"];
        var outs = new Tensor[2];
        s.Run(inNames, ins, outNames, outs); // allocates the caller's outputs once
        for (int i = 0; i < 20; i++)
        {
            s.Run(inNames, ins, outNames, outs);
        }

        Assert.Equal(0, AllocatedBytes(() =>
        {
            for (int i = 0; i < 300; i++)
            {
                s.Run(inNames, ins, outNames, outs);
            }
        }));
        outs[0].Dispose();
        outs[1].Dispose();
    }
}
