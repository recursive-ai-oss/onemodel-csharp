using RecursiveAI.OneModel.Graph;

namespace RecursiveAI.OneModel.Tests.Graph.Passes;

public class LinearFusionTests
{
    private static byte[] LinearModel(int k, int n, string? activation, bool bias, bool gemm)
    {
        var b = new ModelBuilder().Input("x", 1, "N", k).Output("y", 1, "N", n);
        string last = "t";
        if (gemm)
        {
            b.Initializer("w", PassTestKit.RandomFloats(k * n, 1), n, k);
            if (bias)
            {
                b.Initializer("c", PassTestKit.RandomFloats(n, 2), n);
                b.Node("Gemm", ["x", "w", "c"], ["t"], ("transB", 1L));
            }
            else
            {
                b.Node("Gemm", ["x", "w"], ["t"], ("transB", 1L));
            }
        }
        else
        {
            b.Initializer("w", PassTestKit.RandomFloats(k * n, 1), k, n);
            b.Node("MatMul", ["x", "w"], ["t"]);
            if (bias)
            {
                b.Initializer("c", PassTestKit.RandomFloats(n, 2), n);
                b.Node("Add", ["c", "t"], ["t2"]);
                last = "t2";
            }
        }

        if (activation is null)
        {
            b.Node("Identity", [last], ["y"]);
        }
        else if (activation == "Clip")
        {
            b.Initializer("lo", [0f], 1).Initializer("hi", [0.5f], 1);
            b.Node("Clip", [last, "lo", "hi"], ["y"]);
        }
        else
        {
            b.Node(activation, [last], ["y"]);
        }

        return b.Build();
    }

    public static IEnumerable<object[]> Cases()
    {
        foreach (bool gemm in new[] { false, true })
        {
            foreach (string? act in new string?[] { null, "Relu", "Clip", "HardSwish", "Sigmoid" })
            {
                foreach (bool bias in new[] { false, true })
                {
                    yield return new object[] { gemm, act!, bias };
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void FusedMatchesUnfusedBitForBitInExactMode(bool gemm, string? act, bool bias)
    {
        // k = 300 crosses the 256-deep k block, n = 37 leaves ragged strips, and the row counts leave ragged tiles.
        foreach ((int k, int n) in new[] { (300, 37), (17, 130), (1, 5) })
        {
            byte[] model = LinearModel(k, n, act, bias, gemm);
            foreach (int rows in new[] { 1, 3, 7, 97, 130 })
            {
                float[] x = PassTestKit.RandomFloats(rows * k, 3 + rows);
                float[] none = PassTestKit.Run(model, x, [rows, k], OptimizationLevel.None, ExecutionMode.Exact);
                float[] all = PassTestKit.Run(model, x, [rows, k], OptimizationLevel.All, ExecutionMode.Exact);
                Assert.True(PassTestKit.BitEqual(none, all), $"k={k} n={n} rows={rows}");
            }
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void FusedMatchesUnfusedInFastModeAndNeverDependsOnThreads(bool gemm, string? act, bool bias)
    {
        byte[] model = LinearModel(300, 37, act, bias, gemm);
        float[] x = PassTestKit.RandomFloats(130 * 300, 9);
        float[] none = PassTestKit.Run(model, x, [130, 300], OptimizationLevel.None, ExecutionMode.Fast);
        float[] one = PassTestKit.Run(model, x, [130, 300], OptimizationLevel.All, ExecutionMode.Fast, 1);
        float[] many = PassTestKit.Run(model, x, [130, 300], OptimizationLevel.All, ExecutionMode.Fast, 8);
        Assert.True(PassTestKit.BitEqual(one, many));
        for (int i = 0; i < none.Length; i++)
        {
            Assert.True(Math.Abs(none[i] - one[i]) <= 1e-5f * (1f + Math.Abs(none[i])), $"element {i}: {none[i]} vs {one[i]}");
        }
    }

    [Fact]
    public void MatMulBiasReluBecomesOneNode()
    {
        byte[] model = LinearModel(16, 8, "Relu", true, false);
        Assert.Equal(["LinearOp"], PassTestKit.Kernels(model, OptimizationLevel.All, ExecutionMode.Fast));
        Assert.Equal(["MatMulOp", "AddOp", "ReluOp"], PassTestKit.Kernels(model, OptimizationLevel.None, ExecutionMode.Fast));
        Assert.Equal(["MatMulOp", "AddOp", "ReluOp"], PassTestKit.Kernels(model, OptimizationLevel.Basic, ExecutionMode.Fast));
    }
}
