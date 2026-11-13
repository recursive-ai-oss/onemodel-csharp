using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tests.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu.Tests;

/// <summary>
/// One-axis <c>Conv</c> (<c>[N, C, L]</c>) on the device against the CPU session in the same mode, on shapes that hit window edges, strides,
/// dilation, groups, the automatic paddings and a bias or none; plus the speech-encoder stem (two convolutions with Gelu between and after).
/// </summary>
public class Conv1dTests
{
    private static byte[] ConvModel(int n, int c, int len, int oc, int k, long stride, long dilation, long[] pads, long group, string autoPad, bool bias, int elem)
    {
        int cpg = c / (int)group;
        var b = new ModelBuilder().Input("x", elem, n, c, len).Output("y", elem)
            .Initializer("w", GpuTestHost.Random(oc * cpg * k, 7, 0.5f), oc, cpg, k);
        var attrs = new List<(string, object)> { ("strides", new long[] { stride }), ("dilations", new long[] { dilation }), ("group", group) };
        if (autoPad != "NOTSET")
        {
            attrs.Add(("auto_pad", autoPad));
        }
        else
        {
            attrs.Add(("pads", pads));
        }

        string[] inputs = ["x", "w"];
        if (bias)
        {
            b.Initializer("bias", GpuTestHost.Random(oc, 8), oc);
            inputs = ["x", "w", "bias"];
        }

        return b.Node("Conv", inputs, ["y"], [.. attrs]).Build();
    }

    // Fast mode sums in the kernel's order; exact in the deterministic kernel's. Both are within the family's tolerance of the CPU.
    public static IEnumerable<object[]> Cases() => new object[][]
    {
        // n, c, len, oc, k, stride, dilation, padBegin, padEnd, group, autoPad, bias
        new object[] { 1, 80, 300, 96, 3, 1, 1, 1, 1, 1, "NOTSET", true },     // the stem's first convolution, shortened
        new object[] { 1, 96, 300, 96, 3, 2, 1, 1, 1, 1, "NOTSET", true },     // its second: stride 2
        new object[] { 3, 5, 17, 7, 3, 1, 1, 1, 1, 1, "NOTSET", false },       // odd sizes, batch, no bias
        new object[] { 2, 4, 1, 6, 1, 1, 1, 0, 0, 1, "NOTSET", true },         // length 1
        new object[] { 1, 8, 33, 8, 5, 3, 2, 4, 4, 1, "NOTSET", true },        // dilation and stride
        new object[] { 2, 8, 40, 16, 3, 1, 1, 0, 0, 4, "NOTSET", false },      // groups
        new object[] { 1, 6, 31, 6, 4, 2, 1, 0, 0, 6, "NOTSET", true },        // depthwise
        new object[] { 1, 3, 29, 5, 4, 3, 1, 0, 0, 1, "SAME_UPPER", true },    // automatic padding
        new object[] { 1, 3, 29, 5, 4, 3, 1, 0, 0, 1, "SAME_LOWER", true },
        new object[] { 1, 3, 29, 5, 4, 2, 1, 0, 0, 1, "VALID", true },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void ConvMatchesTheCpu(int n, int c, int len, int oc, int k, int stride, int dilation, int padBegin, int padEnd, int group, string autoPad, bool bias)
    {
        GpuTestHost.Require();
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            byte[] model = ConvModel(n, c, len, oc, k, stride, dilation, [padBegin, padEnd], group, autoPad, bias, 1);
            using Tensor x = Tensor.FromSpan<float>(GpuTestHost.Random(n * c * len, 5), Shape.Create(n, c, len));
            var feed = new Dictionary<string, Tensor> { ["x"] = x };
            float[] expected;
            using (Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu(mode)))
            using (SessionOutputs o = cpu.Run(feed))
            {
                expected = o["y"].Data<float>().ToArray();
            }

            using Session gpu = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true));
            float[] first;
            using (SessionOutputs o = gpu.Run(feed))
            {
                first = o["y"].Data<float>().ToArray();
            }

            using (SessionOutputs o = gpu.Run(feed))
            {
                Assert.Equal(GpuTestHost.Hash(first), GpuTestHost.Hash(o["y"].Data<float>()));
            }

            SessionDiagnostics d = gpu.GetDiagnostics(feed);
            Assert.Equal(1, d.NodesOnDevice);
            Assert.True(GpuTestHost.ScaledError(first, expected) <= 1e-5, $"{mode}: {GpuTestHost.ScaledError(first, expected)}");
            GpuTestHost.AssertValidationClean(gpu);
        }
    }

    [Fact]
    public void HalfConvMatchesTheCpuWithinTheHalfTolerance()
    {
        GpuTestHost.Require();
        // float16 values and weights: the graph carries the data as float16 tensors.
        var b = new ModelBuilder().Input("x", 10, 1, 8, 50).Output("y", 10);
        float[] w = GpuTestHost.Random(12 * 8 * 3, 7, 0.5f), bias = GpuTestHost.Random(12, 8);
        byte[] model = b.RawInitializer("w", 10, [12, 8, 3], HalfOf(w)).RawInitializer("bias", 10, [12], HalfOf(bias))
            .Node("Conv", ["x", "w", "bias"], ["y"], ("pads", new long[] { 1, 1 })).Build();
        using Tensor x = Tensor.FromSpan<Half>([.. GpuTestHost.Random(8 * 50, 5).Select(v => (Half)v)], Shape.Create(1, 8, 50));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        float[] expected;
        using (Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu()))
        using (SessionOutputs o = cpu.Run(feed))
        {
            expected = [.. o["y"].Data<Half>().ToArray().Select(h => (float)h)];
        }

        using Session gpu = GpuTestHost.Load(model, GpuTestHost.Gpu(require: true));
        using SessionOutputs g = gpu.Run(feed);
        float[] got = [.. g["y"].Data<Half>().ToArray().Select(h => (float)h)];
        Assert.True(GpuTestHost.ScaledError(got, expected) <= 2e-3, $"{GpuTestHost.ScaledError(got, expected)}");
        GpuTestHost.AssertValidationClean(gpu);
    }

    private static byte[] HalfOf(float[] v)
    {
        byte[] raw = new byte[v.Length * 2];
        for (int i = 0; i < v.Length; i++)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteHalfLittleEndian(raw.AsSpan(i * 2), (Half)v[i]);
        }

        return raw;
    }

    [Fact]
    public void ThreeDimensionalConvStaysOnTheCpuWithAReason()
    {
        GpuTestHost.Require();
        byte[] model = new ModelBuilder().Input("x", 1, 1, 2, 3, 5, 5).Output("y", 1)
            .Initializer("w", GpuTestHost.Random(2 * 2 * 2 * 3 * 3, 7), 2, 2, 2, 3, 3).Node("Conv", ["x", "w"], ["y"]).Build();
        using Tensor x = Tensor.FromSpan<float>(GpuTestHost.Random(150, 5), Shape.Create(1, 2, 3, 5, 5));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        using Session gpu = GpuTestHost.Load(model, GpuTestHost.Gpu());
        SessionDiagnostics d = gpu.GetDiagnostics(feed);
        Assert.Equal(0, d.NodesOnDevice);
        Assert.Contains("GPU", d.Nodes[0].FallbackReason);

        // The reason depends on the shape, which is known when the plan is made, so RequireGpu fails the first run (as for any shape-dependent refusal).
        using Session strict = GpuTestHost.Load(model, GpuTestHost.Gpu(require: true));
        SessionException e = Assert.Throws<SessionException>(() => strict.Run(feed));
        Assert.Equal(["Conv"], e.Operators);
    }

    [Fact]
    public void TheSpeechEncoderStemRunsInOneRegionWithGelu()
    {
        GpuTestHost.Require();
        const int Mels = 40, Width = 96, Frames = 200;
        byte[] model = new ModelBuilder(20).Input("mel", 1, 1, Mels, Frames).Output("y", 1)
            .Initializer("w1", GpuTestHost.Random(Width * Mels * 3, 1, 0.4f), Width, Mels, 3).Initializer("b1", GpuTestHost.Random(Width, 2), Width)
            .Initializer("w2", GpuTestHost.Random(Width * Width * 3, 3, 0.2f), Width, Width, 3).Initializer("b2", GpuTestHost.Random(Width, 4), Width)
            .Node("Conv", ["mel", "w1", "b1"], ["c1"], ("kernel_shape", new long[] { 3 }), ("pads", new long[] { 1, 1 }))
            .Node("Gelu", ["c1"], ["g1"])
            .Node("Conv", ["g1", "w2", "b2"], ["c2"], ("kernel_shape", new long[] { 3 }), ("strides", new long[] { 2 }), ("pads", new long[] { 1, 1 }))
            .Node("Gelu", ["c2"], ["g2"])
            .Node("Transpose", ["g2"], ["y"], ("perm", new long[] { 0, 2, 1 })).Build();
        using Tensor x = Tensor.FromSpan<float>(GpuTestHost.Random(Mels * Frames, 5), Shape.Create(1, Mels, Frames));
        var feed = new Dictionary<string, Tensor> { ["mel"] = x };
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            float[] expected;
            using (Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu(mode)))
            using (SessionOutputs o = cpu.Run(feed))
            {
                expected = o["y"].Data<float>().ToArray();
            }

            using Session gpu = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true));
            using SessionOutputs g = gpu.Run(feed);
            SessionDiagnostics d = gpu.GetDiagnostics(feed);
            Assert.Equal(1, d.DeviceRegions);
            Assert.True(d.Nodes.All(n => n.Region >= 0));
            Assert.True(GpuTestHost.ScaledError(g["y"].Data<float>(), expected) <= 1e-4, $"{mode}");
            GpuTestHost.AssertValidationClean(gpu);
        }
    }
}
