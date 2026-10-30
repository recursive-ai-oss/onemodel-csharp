using RecursiveAI.OneModel.Graph;

namespace RecursiveAI.OneModel.Tests.Graph.Passes;

/// <summary>
/// Dequantize, a float operator, quantize between two quantized tensors become one byte operator (a table, a product with a gate, an
/// average) that gives the chain's bytes.
/// </summary>
public class QuantizedFusionTests
{
    private static byte Zero(bool s, int v) => s ? (byte)(sbyte)v : (byte)v;

    private static void Params(ModelBuilder m, string name, bool signed, float scale, int zero)
    {
        m.Initializer(name + "s", [scale], 1);
        m.RawInitializer(name + "z", signed ? 3 : 2, [1], [Zero(signed, zero)]);
    }

    [Theory]
    [InlineData("HardSwish", false, false, 5000)]
    [InlineData("HardSwish", true, true, 40001)]
    [InlineData("HardSigmoid", false, true, 777)]
    [InlineData("Relu", true, false, 3000)]
    [InlineData("Sigmoid", false, false, 2001)]
    public void TableEqualsTheChain(string op, bool inSigned, bool outSigned, int n)
    {
        var m = new ModelBuilder().Input("x", 1, 1, 1, n).Output("y", 1, 1, 1, n);
        Params(m, "a", inSigned, 0.043f, inSigned ? -9 : 101);
        Params(m, "c", outSigned, 0.0291f, outSigned ? 4 : 17);
        m.Node("QuantizeLinear", ["x", "as", "az"], ["qa"]);
        m.Node("DequantizeLinear", ["qa", "as", "az"], ["da"]);
        m.Node(op, ["da"], ["f"]);
        m.Node("QuantizeLinear", ["f", "cs", "cz"], ["qc"]);
        m.Node("DequantizeLinear", ["qc", "cs", "cz"], ["y"]);
        byte[] model = m.Build();
        float[] x = PassTestKit.RandomFloats(n, 5, 6f);
        x[0] = float.NaN;
        string[] kernels = PassTestKit.Kernels(model, OptimizationLevel.All, ExecutionMode.Fast);
        Assert.Equal(["QuantizeLinearOp", "QLutOp", "DequantizeLinearOp"], kernels);
        float[] want = PassTestKit.Run(model, x, [1, 1, n], OptimizationLevel.None, ExecutionMode.Fast);
        foreach (int threads in new[] { 1, 3 })
        {
            float[] got = PassTestKit.Run(model, x, [1, 1, n], OptimizationLevel.All, ExecutionMode.Fast, threads);
            Assert.True(PassTestKit.BitEqual(want, got), $"op={op} threads={threads}");
        }
    }

    [Theory]
    [InlineData(false, true, 2, 8, 7, 7)]
    [InlineData(true, true, 1, 37, 15, 17)]
    [InlineData(false, false, 3, 5, 1, 1)]
    public void GateProductEqualsTheChain(bool aSigned, bool gateSigned, int n, int c, int h, int w)
    {
        var m = new ModelBuilder().Input("x", 1, n, c, h, w).Output("y", 1, n, c, h, w);
        Params(m, "a", aSigned, 0.031f, aSigned ? -5 : 117);
        Params(m, "b", gateSigned, 0.0071f, gateSigned ? 9 : 100);
        Params(m, "c", aSigned, 0.02f, aSigned ? 3 : 128);
        m.Node("QuantizeLinear", ["x", "as", "az"], ["qa"]);
        m.Node("GlobalAveragePool", ["x"], ["g"]);
        m.Node("QuantizeLinear", ["g", "bs", "bz"], ["qb"]);
        m.Node("DequantizeLinear", ["qa", "as", "az"], ["da"]);
        m.Node("DequantizeLinear", ["qb", "bs", "bz"], ["db"]);
        m.Node("Mul", ["da", "db"], ["p"]);
        m.Node("QuantizeLinear", ["p", "cs", "cz"], ["qc"]);
        m.Node("DequantizeLinear", ["qc", "cs", "cz"], ["y"]);
        byte[] model = m.Build();
        float[] x = PassTestKit.RandomFloats(n * c * h * w, 8, 4f);
        Assert.Contains("QLinearMulOp", PassTestKit.Kernels(model, OptimizationLevel.All, ExecutionMode.Fast));
        float[] want = PassTestKit.Run(model, x, [n, c, h, w], OptimizationLevel.None, ExecutionMode.Fast);
        foreach (int threads in new[] { 1, 3 })
        {
            float[] got = PassTestKit.Run(model, x, [n, c, h, w], OptimizationLevel.All, ExecutionMode.Fast, threads);
            Assert.True(PassTestKit.BitEqual(want, got), $"threads={threads}");
        }
    }

    [Theory]
    [InlineData(false, false, 2, 12, 7, 7)]
    [InlineData(true, true, 1, 20, 14, 14)]
    [InlineData(false, true, 1, 3, 1, 1)]
    public void AverageIsWithinOneStep(bool inSigned, bool outSigned, int n, int c, int h, int w)
    {
        var m = new ModelBuilder().Input("x", 1, n, c, h, w).Output("y", 1, n, c, 1, 1);
        Params(m, "a", inSigned, 0.05f, inSigned ? -3 : 120);
        Params(m, "c", outSigned, 0.011f, outSigned ? 2 : 90);
        m.Node("QuantizeLinear", ["x", "as", "az"], ["qa"]);
        m.Node("DequantizeLinear", ["qa", "as", "az"], ["da"]);
        m.Node("GlobalAveragePool", ["da"], ["p"]);
        m.Node("QuantizeLinear", ["p", "cs", "cz"], ["qc"]);
        m.Node("DequantizeLinear", ["qc", "cs", "cz"], ["y"]);
        byte[] model = m.Build();
        float[] x = PassTestKit.RandomFloats(n * c * h * w, 12, 3f);
        Assert.Contains("QLinearGlobalAveragePoolOp", PassTestKit.Kernels(model, OptimizationLevel.All, ExecutionMode.Fast));
        float[] want = PassTestKit.Run(model, x, [n, c, h, w], OptimizationLevel.None, ExecutionMode.Fast);
        float[] got = PassTestKit.Run(model, x, [n, c, h, w], OptimizationLevel.All, ExecutionMode.Fast);
        for (int i = 0; i < want.Length; i++)
        {
            Assert.True(Math.Abs(want[i] - got[i]) <= 0.011f * 1.001f, $"element {i}: {want[i]} vs {got[i]}");
        }
    }
}
