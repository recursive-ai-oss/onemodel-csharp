using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tests.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu.Tests;

/// <summary>
/// The convolution family on float16 models: convolution (dense, depthwise, pointwise, strided), transposed convolution, pooling, Resize and
/// Concat on half-precision tensors, against the CPU session in the same mode. A model must run on the device completely, in one region.
/// </summary>
public class HalfVisionTests
{
    private const double HalfTolerance = 2e-3;

    private static byte[] HalfOf(float[] v)
    {
        byte[] raw = new byte[v.Length * 2];
        for (int i = 0; i < v.Length; i++)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteHalfLittleEndian(raw.AsSpan(i * 2), (Half)v[i]);
        }

        return raw;
    }

    private static ModelBuilder Weights(ModelBuilder b, string name, int seed, params long[] dims)
    {
        int n = 1;
        foreach (long d in dims)
        {
            n *= (int)d;
        }

        return b.RawInitializer(name, 10, dims, HalfOf(GpuTestHost.Random(n, seed, 0.4f)));
    }

    private static double Compare(byte[] model, int[] shape, ArithmeticMode mode, out SessionDiagnostics diag)
    {
        GpuTestHost.Require();
        int count = shape.Aggregate(1, (a, c) => a * c);
        using Tensor x = Tensor.FromSpan<Half>([.. GpuTestHost.Random(count, 5, 2f).Select(v => (Half)v)], Shape.Create(shape));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        float[] expected;
        using (Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu(mode)))
        using (SessionOutputs o = cpu.Run(feed))
        {
            expected = [.. o["y"].Data<Half>().ToArray().Select(h => (float)h)];
        }

        using Session gpu = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true));
        float[] first;
        using (SessionOutputs o = gpu.Run(feed))
        {
            first = [.. o["y"].Data<Half>().ToArray().Select(h => (float)h)];
        }

        using (SessionOutputs o = gpu.Run(feed))
        {
            Assert.Equal(GpuTestHost.Hash(first), GpuTestHost.Hash([.. o["y"].Data<Half>().ToArray().Select(h => (float)h)]));
        }

        diag = gpu.GetDiagnostics(feed);
        Assert.True(diag.Nodes.All(n => n.FallbackReason is null), string.Join("; ", diag.Nodes.Where(n => n.FallbackReason is not null).Select(n => n.OpType + ": " + n.FallbackReason)));
        Assert.Equal(1, diag.DeviceRegions);
        GpuTestHost.AssertValidationClean(gpu);
        return GpuTestHost.ScaledError(first, expected);
    }

    [Theory]
    [InlineData(ArithmeticMode.Fast)]
    [InlineData(ArithmeticMode.Exact)]
    public void ABlockOfEveryWindowOperatorRunsInOneRegion(ArithmeticMode mode)
    {
        ModelBuilder b = new ModelBuilder().Input("x", 10, 2, 4, 16, 16).Output("y", 10);
        b = Weights(b, "w1", 1, 8, 4, 3, 3);
        b = Weights(b, "b1", 2, 8);
        b = Weights(b, "wd", 3, 8, 1, 3, 3);
        b = Weights(b, "wp", 4, 8, 8, 1, 1);
        b = Weights(b, "wt", 5, 16, 4, 2, 2);
        b = b.Initializer("roi", Array.Empty<float>(), 0).Initializer("scales", [1f, 1f, 2f, 2f], 4);
        byte[] model = b
            .Node("Conv", ["x", "w1", "b1"], ["c1"], ("pads", new long[] { 1, 1, 1, 1 }))
            .Node("Relu", ["c1"], ["r"])
            .Node("Conv", ["r", "wd"], ["d"], ("pads", new long[] { 1, 1, 1, 1 }), ("group", 8L))
            .Node("Conv", ["d", "wp"], ["p"])
            .Node("Add", ["r", "p"], ["s"])
            .Node("MaxPool", ["s"], ["mp"], ("kernel_shape", new long[] { 2, 2 }), ("strides", new long[] { 2, 2 }))
            .Node("AveragePool", ["s"], ["ap"], ("kernel_shape", new long[] { 2, 2 }), ("strides", new long[] { 2, 2 }))
            .Node("Concat", ["mp", "ap"], ["cat"], ("axis", 1L))
            .Node("Resize", ["cat", "roi", "scales"], ["up"], ("mode", "nearest"))
            .Node("ConvTranspose", ["up", "wt"], ["ct"], ("strides", new long[] { 2, 2 }))
            .Node("GlobalAveragePool", ["ct"], ["y"])
            .Build();
        double err = Compare(model, [2, 4, 16, 16], mode, out SessionDiagnostics d);
        Assert.True(err <= HalfTolerance, $"{mode} scaled error {err}");
        Assert.Equal(d.Nodes.Length, d.NodesOnDevice);
    }

    [Theory]
    [InlineData(ArithmeticMode.Fast)]
    [InlineData(ArithmeticMode.Exact)]
    public void ALargeConvolutionUsesArenaScratchOverABatch(ArithmeticMode mode)
    {
        // Enough channels and positions that the unfolding variants need column and accumulator scratch, with a batch of three images.
        ModelBuilder b = new ModelBuilder().Input("x", 10, 3, 32, 20, 20).Output("y", 10);
        b = Weights(b, "w", 6, 64, 32, 3, 3);
        b = Weights(b, "bias", 7, 64);
        byte[] model = b.Node("Conv", ["x", "w", "bias"], ["c"], ("pads", new long[] { 1, 1, 1, 1 }), ("strides", new long[] { 2, 2 }))
            .Node("Relu", ["c"], ["y"]).Build();
        double err = Compare(model, [3, 32, 20, 20], mode, out _);
        Assert.True(err <= HalfTolerance, $"{mode} scaled error {err}");
    }

    [Fact]
    public void ConcatOfHalfTensorsIsExactAlongEveryAxis()
    {
        foreach (int axis in new[] { 0, 1, 3 })
        {
            int[] shape = [2, 3, 5, 4];
            ModelBuilder b = new ModelBuilder().Input("x", 10, Dims(shape)).Output("y", 10);
            byte[] model = b.Node("Neg", ["x"], ["n"]).Node("Concat", ["x", "n", "x"], ["y"], ("axis", (long)axis)).Build();
            double err = Compare(model, shape, ArithmeticMode.Fast, out _);
            Assert.Equal(0.0, err);
        }
    }

    private static object[] Dims(int[] s) => [.. s.Select(d => (object)d)];
}
