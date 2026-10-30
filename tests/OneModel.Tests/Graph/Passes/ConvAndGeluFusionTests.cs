using RecursiveAI.OneModel.Graph;

namespace RecursiveAI.OneModel.Tests.Graph.Passes;

public class ConvAndGeluFusionTests
{
    // x [1, cin, 9, 11] -> Conv(kernel, group) [+ bias] [-> BatchNormalization] [-> activation] -> y
    private static byte[] ConvModel(int cin, int cout, int kernel, int group, bool bias, bool bn, string? act)
    {
        var b = new ModelBuilder().Input("x", 1, 1, cin, 9, 11).Output("y", 1, 1, cout, 9, 11);
        b.Initializer("w", PassTestKit.RandomFloats(cout * (cin / group) * kernel * kernel, 4, 0.5f), cout, cin / group, kernel, kernel);
        string[] convIn = ["x", "w"];
        if (bias)
        {
            b.Initializer("b", PassTestKit.RandomFloats(cout, 5), cout);
            convIn = ["x", "w", "b"];
        }

        b.Node("Conv", convIn, ["c"], ("kernel_shape", new long[] { kernel, kernel }), ("pads", new long[] { kernel / 2, kernel / 2, kernel / 2, kernel / 2 }), ("group", (long)group));
        string last = "c";
        if (bn)
        {
            b.Initializer("s", PassTestKit.RandomFloats(cout, 6).Select(v => 1f + (0.5f * v)).ToArray(), cout);
            b.Initializer("o", PassTestKit.RandomFloats(cout, 7), cout);
            b.Initializer("m", PassTestKit.RandomFloats(cout, 8), cout);
            b.Initializer("v", PassTestKit.RandomFloats(cout, 9).Select(v => 1f + (0.5f * v)).ToArray(), cout);
            b.Node("BatchNormalization", ["c", "s", "o", "m", "v"], ["n"], ("epsilon", 1e-3f));
            last = "n";
        }

        if (act is null)
        {
            b.Node("Identity", [last], ["y"]);
        }
        else
        {
            b.Node(act, [last], ["y"]);
        }

        return b.Build();
    }

    public static IEnumerable<object[]> ConvCases()
    {
        // (cin, cout, kernel, group): regular 3x3, pointwise, grouped, depthwise
        foreach ((int cin, int cout, int k, int g) in new[] { (5, 7, 3, 1), (6, 9, 1, 1), (6, 8, 3, 2), (6, 6, 3, 6) })
        {
            foreach (string? act in new string?[] { null, "Relu", "HardSwish" })
            {
                yield return new object[] { cin, cout, k, g, act! };
            }
        }
    }

    [Theory]
    [MemberData(nameof(ConvCases))]
    public void ConvActivationIsBitIdenticalInExactMode(int cin, int cout, int k, int g, string? act)
    {
        byte[] model = ConvModel(cin, cout, k, g, true, false, act);
        float[] x = PassTestKit.RandomFloats(cin * 9 * 11, 11);
        float[] none = PassTestKit.Run(model, x, [1, cin, 9, 11], OptimizationLevel.None, ExecutionMode.Exact);
        float[] all = PassTestKit.Run(model, x, [1, cin, 9, 11], OptimizationLevel.All, ExecutionMode.Exact);
        Assert.True(PassTestKit.BitEqual(none, all));
    }

    [Theory]
    [MemberData(nameof(ConvCases))]
    public void ConvBatchNormActivationMatchesUnfusedInFastMode(int cin, int cout, int k, int g, string? act)
    {
        byte[] model = ConvModel(cin, cout, k, g, true, true, act);
        float[] x = PassTestKit.RandomFloats(cin * 9 * 11, 12);
        float[] none = PassTestKit.Run(model, x, [1, cin, 9, 11], OptimizationLevel.None, ExecutionMode.Fast);
        float[] one = PassTestKit.Run(model, x, [1, cin, 9, 11], OptimizationLevel.All, ExecutionMode.Fast, 1);
        float[] many = PassTestKit.Run(model, x, [1, cin, 9, 11], OptimizationLevel.All, ExecutionMode.Fast, 4);
        Assert.True(PassTestKit.BitEqual(one, many));
        for (int i = 0; i < none.Length; i++)
        {
            Assert.True(Math.Abs(none[i] - one[i]) <= 2e-5f * (1f + Math.Abs(none[i])), $"element {i}: {none[i]} vs {one[i]}");
        }

        string[] fast = PassTestKit.Kernels(model, OptimizationLevel.All, ExecutionMode.Fast);
        Assert.Equal(act is null ? ["ConvOp", "IdentityOp"] : ["ConvOp"], fast);
        string[] exact = PassTestKit.Kernels(model, OptimizationLevel.All, ExecutionMode.Exact);
        Assert.Contains("BatchNormalizationOp", exact);
    }

    private static byte[] GeluModel()
    {
        var b = new ModelBuilder().Input("x", 1, "N", 40).Output("y", 1, "N", 40);
        b.Initializer("r2", [1.4142135381698608f], 1).Initializer("one", [1f], 1).Initializer("half", [0.5f], 1);
        b.Node("Div", ["x", "r2"], ["d"]);
        b.Node("Erf", ["d"], ["e"]);
        b.Node("Add", ["e", "one"], ["a"]);
        b.Node("Mul", ["x", "a"], ["m"]);
        b.Node("Mul", ["m", "half"], ["y"]);
        return b.Build();
    }

    [Fact]
    public void ErfGeluChainBecomesOneNodeAndStaysClose()
    {
        byte[] model = GeluModel();
        Assert.Equal(["FloatFunctionOp`1"], PassTestKit.Kernels(model, OptimizationLevel.All, ExecutionMode.Fast));
        Assert.Equal(5, PassTestKit.Kernels(model, OptimizationLevel.All, ExecutionMode.Exact).Length);
        float[] x = PassTestKit.RandomFloats(7 * 40, 13, 4f);
        float[] none = PassTestKit.Run(model, x, [7, 40], OptimizationLevel.None, ExecutionMode.Fast);
        float[] all = PassTestKit.Run(model, x, [7, 40], OptimizationLevel.All, ExecutionMode.Fast);
        for (int i = 0; i < none.Length; i++)
        {
            Assert.True(Math.Abs(none[i] - all[i]) <= 2e-6f * (1f + Math.Abs(none[i])), $"element {i}: {none[i]} vs {all[i]}");
        }
    }

    [Fact]
    public void GeluInsideAMatMulBiasChainFusesIntoTheLinearNode()
    {
        var b = new ModelBuilder().Input("x", 1, "N", 16).Output("y", 1, "N", 24);
        b.Initializer("w", PassTestKit.RandomFloats(16 * 24, 1), 16, 24).Initializer("c", PassTestKit.RandomFloats(24, 2), 24);
        b.Initializer("r2", [1.4142135381698608f], 1).Initializer("one", [1f], 1).Initializer("half", [0.5f], 1);
        b.Node("MatMul", ["x", "w"], ["t"]);
        b.Node("Add", ["c", "t"], ["u"]);
        b.Node("Div", ["u", "r2"], ["d"]);
        b.Node("Erf", ["d"], ["e"]);
        b.Node("Add", ["e", "one"], ["a"]);
        b.Node("Mul", ["u", "a"], ["m"]);
        b.Node("Mul", ["m", "half"], ["y"]);
        Assert.Equal(["LinearOp"], PassTestKit.Kernels(b.Build(), OptimizationLevel.All, ExecutionMode.Fast));
    }
}
