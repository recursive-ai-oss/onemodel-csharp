using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tests.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu.Tests;

/// <summary>
/// A matrix product followed by a bias addition and an activation is one dispatch where the intermediate results are read by nobody else. The
/// float32 result must have the bits it has with the three operators run separately, in both modes.
/// </summary>
public class EpilogueTests
{
    private static byte[] Layer(string? activation, bool half, bool shared, int k = 20, int n = 24)
    {
        int type = half ? 10 : 1;
        var b = new ModelBuilder(20).Input("x", type, 2, "S", k).Output("y", type);
        if (half)
        {
            b.RawInitializer("w", 10, [k, n], Halves(GpuTestHost.Random(k * n, 3, 0.5f))).RawInitializer("bias", 10, [n], Halves(GpuTestHost.Random(n, 4)));
        }
        else
        {
            b.Initializer("w", GpuTestHost.Random(k * n, 3, 0.5f), k, n).Initializer("bias", GpuTestHost.Random(n, 4), n);
        }

        // Without a second reader the last operator's result is the graph's output; with one, an addition reads it and the product.
        string Out(string name, bool last = true) => shared || !last ? name : "y";
        b.Node("MatMul", ["x", "w"], ["m"]).Node("Add", ["m", "bias"], [Out("a", activation is null)]);
        string last = "a";
        switch (activation)
        {
            case "relu":
                b.Node("Relu", ["a"], [Out("act")]);
                last = "act";
                break;
            case "gelu":
                b.Node("Gelu", ["a"], [Out("act")]);
                last = "act";
                break;
            case "tanh":
                b.Node("Gelu", ["a"], [Out("act")], ("approximate", "tanh"));
                last = "act";
                break;
        }

        if (shared)
        {
            // The product is read by a second operator, so it must be kept as it is.
            b.Node("Add", [last, "m"], ["y"]);
        }

        return b.Build();
    }

    private static byte[] Halves(float[] v)
    {
        byte[] raw = new byte[v.Length * 2];
        for (int i = 0; i < v.Length; i++)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteHalfLittleEndian(raw.AsSpan(i * 2), (Half)v[i]);
        }

        return raw;
    }

    private static float[] Floats(in Tensor t) => t.DType == DType.F16 ? [.. t.Data<Half>().ToArray().Select(h => (float)h)] : t.Data<float>().ToArray();

    private static (float[] Output, long Dispatches) RunOnDevice(byte[] model, Dictionary<string, Tensor> feed, ArithmeticMode mode, GpuOptions? options)
    {
        using IDisposable? scope = options is null ? null : GpuTestHost.With(options);
        using Session s = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true));
        float[] first;
        using (SessionOutputs o = s.Run(feed))
        {
            first = Floats(o["y"]);
        }

        OneModelGpu.ResetTimings(s);
        using (SessionOutputs o = s.Run(feed))
        {
            Assert.Equal(GpuTestHost.Hash(first), GpuTestHost.Hash(Floats(o["y"])));
        }

        GpuTestHost.AssertValidationClean(s);
        return (first, OneModelGpu.GetInfo(s).Timings.Dispatches);
    }

    [Theory]
    [InlineData(ArithmeticMode.Fast, null)]
    [InlineData(ArithmeticMode.Exact, null)]
    [InlineData(ArithmeticMode.Fast, "relu")]
    [InlineData(ArithmeticMode.Exact, "gelu")]
    [InlineData(ArithmeticMode.Fast, "gelu")]
    [InlineData(ArithmeticMode.Fast, "tanh")]
    [InlineData(ArithmeticMode.Exact, "tanh")]
    public void AFoldedBiasAndActivationHaveTheBitsOfTheSeparateOperators(ArithmeticMode mode, string? activation)
    {
        GpuTestHost.Require();
        byte[] model = Layer(activation, half: false, shared: false);
        using Tensor x = Tensor.FromSpan<float>(GpuTestHost.Random(2 * 9 * 20, 5), Shape.Create(2, 9, 20));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        (float[] folded, long dispatchesFolded) = RunOnDevice(model, feed, mode, null);
        (float[] separate, long dispatchesSeparate) = RunOnDevice(model, feed, mode, new GpuOptions { Validation = true, CollectTimings = true, FuseEpilogues = false });
        Assert.Equal(GpuTestHost.Hash(separate), GpuTestHost.Hash(folded));
        if (dispatchesSeparate > 0)
        {
            int saved = 1 + (activation is null ? 0 : 1);
            Assert.Equal(dispatchesSeparate - saved, dispatchesFolded);
        }

        using Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu(mode));
        using SessionOutputs o = cpu.Run(feed);
        double error = GpuTestHost.ScaledError(folded, Floats(o["y"]));
        // The device's tanh form of Gelu is within 1e-4 of a double reference rather than the 1e-5 of the other functions.
        double bound = activation == "tanh" ? 3e-4 : 1e-5;
        Assert.True(error <= bound, $"{mode} {activation}: scaled error against the CPU {error:E2}");
    }

    [Fact]
    public void AProductThatAnotherOperatorAlsoReadsIsNotFolded()
    {
        GpuTestHost.Require();
        byte[] model = Layer("gelu", half: false, shared: true);
        using Tensor x = Tensor.FromSpan<float>(GpuTestHost.Random(2 * 9 * 20, 5), Shape.Create(2, 9, 20));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        (float[] folded, long dispatchesFolded) = RunOnDevice(model, feed, ArithmeticMode.Fast, null);
        (float[] separate, long dispatchesSeparate) = RunOnDevice(model, feed, ArithmeticMode.Fast, new GpuOptions { Validation = true, CollectTimings = true, FuseEpilogues = false });
        Assert.Equal(GpuTestHost.Hash(separate), GpuTestHost.Hash(folded));
        Assert.Equal(dispatchesSeparate, dispatchesFolded);
        using Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu());
        using SessionOutputs o = cpu.Run(feed);
        Assert.True(GpuTestHost.ScaledError(folded, Floats(o["y"])) <= 1e-5);
    }

    [Theory]
    [InlineData("gelu")]
    [InlineData(null)]
    public void AHalfPrecisionProductFoldsInFastModeWithinHalfTolerance(string? activation)
    {
        GpuTestHost.Require();
        byte[] model = Layer(activation, half: true, shared: false);
        using Tensor x = Tensor.FromSpan<Half>([.. GpuTestHost.Random(2 * 9 * 20, 5).Select(v => (Half)v)], Shape.Create(2, 9, 20));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        (float[] folded, long dispatchesFolded) = RunOnDevice(model, feed, ArithmeticMode.Fast, null);
        (float[] separate, long dispatchesSeparate) = RunOnDevice(model, feed, ArithmeticMode.Fast, new GpuOptions { Validation = true, CollectTimings = true, FuseEpilogues = false });
        Assert.True(GpuTestHost.ScaledError(folded, separate) <= 2e-3, $"{GpuTestHost.ScaledError(folded, separate):E2}");
        if (dispatchesSeparate > 0)
        {
            Assert.True(dispatchesFolded < dispatchesSeparate);
        }

        using Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu(ArithmeticMode.Exact));
        using SessionOutputs o = cpu.Run(feed);
        Assert.True(GpuTestHost.ScaledError(folded, Floats(o["y"])) <= 2e-3);

        // Exact mode keeps the separate operators for half precision, and gives the same bits as without folding.
        (float[] exact, _) = RunOnDevice(model, feed, ArithmeticMode.Exact, null);
        (float[] exactSeparate, _) = RunOnDevice(model, feed, ArithmeticMode.Exact, new GpuOptions { Validation = true, CollectTimings = true, FuseEpilogues = false });
        Assert.Equal(GpuTestHost.Hash(exactSeparate), GpuTestHost.Hash(exact));
    }
}
