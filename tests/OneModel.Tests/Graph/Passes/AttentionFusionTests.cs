using RecursiveAI.OneModel.Graph;

namespace RecursiveAI.OneModel.Tests.Graph.Passes;

public class AttentionFusionTests
{
    private const int B = 2, S = 13;

    // x [B, S, H*D] -> q, k, v projections (scale, shift: cheap stand-ins) -> head split -> attention chain -> head merge.
    private static byte[] Model(int heads, int d, string mask, bool div, bool merge)
    {
        int e = heads * d;
        var b = new ModelBuilder().Input("x", 1, B, S, e);
        b.Initializer("two", [2f], 1).Initializer("one", [1f], 1);
        b.Initializer("split", [0L, 0L, heads, d], 4);
        b.Initializer("join", [0L, 0L, e], 3);
        b.Initializer("scale", [div ? 4f : 0.25f], 1);
        b.Node("Mul", ["x", "two"], ["kp"]);
        b.Node("Add", ["x", "one"], ["vp"]);
        b.Node("Reshape", ["x", "split"], ["qr"]);
        b.Node("Reshape", ["kp", "split"], ["kr"]);
        b.Node("Reshape", ["vp", "split"], ["vr"]);
        b.Node("Transpose", ["qr"], ["q"], ("perm", new long[] { 0, 2, 1, 3 }));
        b.Node("Transpose", ["kr"], ["kt"], ("perm", new long[] { 0, 2, 3, 1 }));
        b.Node("Transpose", ["vr"], ["v"], ("perm", new long[] { 0, 2, 1, 3 }));
        b.Node("MatMul", ["q", "kt"], ["s"]);
        b.Node(div ? "Div" : "Mul", ["s", "scale"], ["ss"]);
        string cur = "ss";
        if (mask == "batch")
        {
            var m = new float[B * S];
            for (int i = 0; i < m.Length; i++)
            {
                m[i] = i % 5 == 4 ? -1e9f : 0f;
            }

            b.Initializer("mask", m, B, 1, 1, S);
            b.Node("Add", ["ss", "mask"], ["sm"]);
            cur = "sm";
        }
        else if (mask == "causal")
        {
            var m = new float[S * S];
            for (int i = 0; i < S; i++)
            {
                for (int j = 0; j < S; j++)
                {
                    m[(i * S) + j] = j > i ? -1e9f : 0f;
                }
            }

            b.Initializer("mask", m, 1, 1, S, S);
            b.Node("Add", ["mask", "ss"], ["sm"]);
            cur = "sm";
        }

        b.Node("Softmax", [cur], ["p"], ("axis", -1L));
        b.Node("MatMul", ["p", "v"], ["o"]);
        if (merge)
        {
            b.Node("Transpose", ["o"], ["ot"], ("perm", new long[] { 0, 2, 1, 3 }));
            b.Node("Reshape", ["ot", "join"], ["y"]);
            b.Output("y", 1, B, S, e);
        }
        else
        {
            b.Node("Identity", ["o"], ["y"]);
            b.Output("y", 1, B, heads, S, d);
        }

        return b.Build();
    }

    public static IEnumerable<object[]> Cases()
    {
        foreach ((int h, int d) in new[] { (3, 16), (2, 8), (4, 6), (2, 7) })
        {
            foreach (string mask in new[] { "none", "batch", "causal" })
            {
                foreach (bool div in new[] { false, true })
                {
                    foreach (bool merge in new[] { false, true })
                    {
                        yield return new object[] { h, d, mask, div, merge };
                    }
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void FusedAttentionMatchesTheSeparateOperators(int h, int d, string mask, bool div, bool merge)
    {
        byte[] model = Model(h, d, mask, div, merge);
        float[] x = PassTestKit.RandomFloats(B * S * h * d, 21, 1.5f);
        int[] shape = [B, S, h * d];
        float[] none = PassTestKit.Run(model, x, shape, OptimizationLevel.None, ArithmeticMode.Fast);
        float[] one = PassTestKit.Run(model, x, shape, OptimizationLevel.All, ArithmeticMode.Fast, 1);
        float[] many = PassTestKit.Run(model, x, shape, OptimizationLevel.All, ArithmeticMode.Fast, 8);
        Assert.True(PassTestKit.BitEqual(one, many));
        for (int i = 0; i < none.Length; i++)
        {
            Assert.True(Math.Abs(none[i] - one[i]) <= 1e-5f * (1f + Math.Abs(none[i])), $"element {i}: {none[i]} vs {one[i]}");
        }

        Assert.Contains("AttentionOp", PassTestKit.Kernels(model, OptimizationLevel.All, ArithmeticMode.Fast));
        Assert.DoesNotContain("AttentionOp", PassTestKit.Kernels(model, OptimizationLevel.All, ArithmeticMode.Exact));
        Assert.DoesNotContain("SoftmaxOp", PassTestKit.Kernels(model, OptimizationLevel.All, ArithmeticMode.Fast));
    }
}
