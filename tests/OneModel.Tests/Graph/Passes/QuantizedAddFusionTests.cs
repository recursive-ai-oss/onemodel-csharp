using RecursiveAI.OneModel.Graph;

namespace RecursiveAI.OneModel.Tests.Graph.Passes;

/// <summary>DequantizeLinear, DequantizeLinear, Add, QuantizeLinear becomes one quantized addition that gives the same bytes.</summary>
public class QuantizedAddFusionTests
{
    // y = Dequantize(Quantize(Dequantize(Quantize(x)) + Dequantize(Quantize(0.7 x))))
    private static byte[] Model(int n, bool aSigned, bool bSigned, bool ySigned, float aScale, float bScale, float yScale)
    {
        var m = new ModelBuilder().Input("x", 1, 1, n).Output("y", 1, 1, n);
        m.Initializer("k", [0.7f], 1);
        byte Zero(bool s, int v) => s ? (byte)(sbyte)v : (byte)v;
        void Params(string name, bool signed, float scale, int zero)
        {
            m.Initializer(name + "s", [scale], 1);
            m.RawInitializer(name + "z", signed ? 3 : 2, [1], [Zero(signed, zero)]);
        }

        Params("a", aSigned, aScale, aSigned ? -5 : 117);
        Params("b", bSigned, bScale, bSigned ? 9 : 100);
        Params("c", ySigned, yScale, ySigned ? 3 : 128);
        m.Node("Mul", ["x", "k"], ["x2"]);
        m.Node("QuantizeLinear", ["x", "as", "az"], ["qa"]);
        m.Node("QuantizeLinear", ["x2", "bs", "bz"], ["qb"]);
        m.Node("DequantizeLinear", ["qa", "as", "az"], ["da"]);
        m.Node("DequantizeLinear", ["qb", "bs", "bz"], ["db"]);
        m.Node("Add", ["da", "db"], ["sum"]);
        m.Node("QuantizeLinear", ["sum", "cs", "cz"], ["qc"]);
        m.Node("DequantizeLinear", ["qc", "cs", "cz"], ["y"]);
        return m.Build();
    }

    [Theory]
    [InlineData(100, false, false, false)]
    [InlineData(40000, true, false, true)]
    [InlineData(40013, false, true, false)]
    [InlineData(70001, true, true, true)]
    public void FusedAdditionEqualsTheChain(int n, bool aSigned, bool bSigned, bool ySigned)
    {
        byte[] model = Model(n, aSigned, bSigned, ySigned, 0.031f, 0.0173f, 0.047f);
        float[] x = PassTestKit.RandomFloats(n, 21, 3f);
        x[0] = float.NaN;
        x[1] = 1e9f;
        x[2] = -1e9f;
        Assert.Contains("QLinearAddOp", PassTestKit.Kernels(model, OptimizationLevel.All, ArithmeticMode.Fast));
        float[] want = PassTestKit.Run(model, x, [1, n], OptimizationLevel.None, ArithmeticMode.Fast);
        foreach (int threads in new[] { 1, 3 })
        {
            float[] got = PassTestKit.Run(model, x, [1, n], OptimizationLevel.All, ArithmeticMode.Fast, threads);
            Assert.True(PassTestKit.BitEqual(want, got), $"n={n} threads={threads}");
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MaxPoolBetweenMatchingQuantizersRunsOnBytes(bool isSigned)
    {
        var m = new ModelBuilder().Input("x", 1, 1, 3, 17, 19).Output("y", 1, 1, 3, 9, 10);
        m.Initializer("s", [0.05f], 1);
        m.RawInitializer("z", isSigned ? 3 : 2, [1], [isSigned ? unchecked((byte)-7) : (byte)90]);
        m.Node("QuantizeLinear", ["x", "s", "z"], ["q"]);
        m.Node("DequantizeLinear", ["q", "s", "z"], ["d"]);
        m.Node("MaxPool", ["d"], ["p"], ("kernel_shape", new long[] { 3, 3 }), ("strides", new long[] { 2, 2 }), ("pads", new long[] { 1, 1, 1, 1 }));
        m.Node("QuantizeLinear", ["p", "s", "z"], ["r"]);
        m.Node("DequantizeLinear", ["r", "s", "z"], ["y"]);
        byte[] model = m.Build();
        float[] x = PassTestKit.RandomFloats(3 * 17 * 19, 33, 9f);
        string[] kernels = PassTestKit.Kernels(model, OptimizationLevel.All, ArithmeticMode.Fast);
        Assert.Equal(["QuantizeLinearOp", "MaxPoolOp", "DequantizeLinearOp"], kernels);
        float[] want = PassTestKit.Run(model, x, [1, 3, 17, 19], OptimizationLevel.None, ArithmeticMode.Fast);
        float[] got = PassTestKit.Run(model, x, [1, 3, 17, 19], OptimizationLevel.All, ArithmeticMode.Fast);
        Assert.True(PassTestKit.BitEqual(want, got));
    }
}
