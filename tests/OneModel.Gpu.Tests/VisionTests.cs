using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tests.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu.Tests;

/// <summary>
/// The convolution family on the device: Conv with its fused bias, activation and residual, pooling, Resize, ConvTranspose and Concat,
/// each on short models against the CPU session in the same mode. A model must run on the device completely, in one region.
/// </summary>
public class VisionTests
{
    private static int Count(int[] s) => s.Aggregate(1, (a, b) => a * b);

    private static object[] Dims(int[] s) => [.. s.Select(d => (object)d)];

    private static Tensor Make(float[] v, int[] shape) => Tensor.FromSpan<float>(v, Shape.Create(shape));

    /// <summary>Runs on the CPU and the device; returns the scaled error, and checks the model ran on the device in one region and (exact) twice alike.</summary>
    private static double Compare(byte[] model, Dictionary<string, Tensor> feed, ArithmeticMode mode, out SessionDiagnostics diag, string output = "y")
    {
        GpuTestHost.Require();
        float[] expected;
        using (Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu(mode)))
        using (SessionOutputs o = cpu.Run(feed))
        {
            expected = o[output].Data<float>().ToArray();
        }

        using Session gpu = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true));
        float[] first;
        using (SessionOutputs o = gpu.Run(feed))
        {
            first = o[output].Data<float>().ToArray();
        }

        using (SessionOutputs o = gpu.Run(feed))
        {
            Assert.Equal(GpuTestHost.Hash(first), GpuTestHost.Hash(o[output].Data<float>()));
        }

        diag = gpu.GetDiagnostics(feed);
        Assert.True(diag.Nodes.All(n => n.FallbackReason is null), string.Join("; ", diag.Nodes.Where(n => n.FallbackReason is not null).Select(n => n.OpType + ": " + n.FallbackReason)));
        Assert.Equal(1, diag.DeviceRegions);
        GpuTestHost.AssertValidationClean(gpu);
        return GpuTestHost.ScaledError(first, expected);
    }

    private static ModelBuilder ConvModel(int[] x, int m, int[] k, int groups, bool bias, params (string, object)[] attrs)
    {
        var b = new ModelBuilder().Input("x", 1, Dims(x)).Output("y", 1)
            .Initializer("w", GpuTestHost.Random(m * (x[1] / groups) * Count(k), 11, 0.5f), [m, x[1] / groups, .. k.Select(v => (long)v)]);
        if (bias)
        {
            b = b.Initializer("b", GpuTestHost.Random(m, 12), m);
        }

        return b.Node("Conv", bias ? ["x", "w", "b"] : ["x", "w"], ["y"], attrs);
    }

    // n, c, h, w, m, kernel, strides, dilations, pads, groups, bias, auto_pad
    public static TheoryData<int[], int, int[], long[], long[], long[], int, bool, string> ConvCases => new()
    {
        { [1, 3, 17, 19], 8, [3, 3], [1, 1], [1, 1], [1, 1, 1, 1], 1, true, "NOTSET" },
        { [2, 16, 32, 32], 24, [3, 3], [2, 2], [1, 1], [1, 1, 1, 1], 1, false, "NOTSET" },
        { [1, 8, 20, 20], 8, [3, 3], [1, 1], [1, 1], [1, 1, 1, 1], 8, true, "NOTSET" },
        { [2, 8, 15, 15], 16, [3, 3], [2, 2], [1, 1], [1, 1, 1, 1], 8, true, "NOTSET" },
        { [1, 4, 9, 9], 6, [1, 1], [1, 1], [1, 1], [0, 0, 0, 0], 1, true, "NOTSET" },
        { [1, 4, 12, 12], 6, [3, 3], [1, 1], [2, 2], [2, 2, 2, 2], 2, true, "NOTSET" },
        { [1, 3, 14, 14], 5, [7, 7], [2, 2], [1, 1], [3, 3, 3, 3], 1, true, "NOTSET" },
        { [2, 5, 13, 11], 7, [3, 3], [2, 2], [1, 1], [0, 0, 0, 0], 1, false, "SAME_UPPER" },
        { [1, 5, 13, 11], 7, [3, 3], [2, 2], [1, 1], [0, 0, 0, 0], 1, false, "SAME_LOWER" },
        { [1, 2, 33, 1], 3, [3, 1], [1, 1], [1, 1], [1, 0, 1, 0], 1, true, "NOTSET" },
        { [1, 1, 1, 1], 1, [1, 1], [1, 1], [1, 1], [0, 0, 0, 0], 1, true, "NOTSET" },
    };

    [Theory]
    [MemberData(nameof(ConvCases))]
    public void ConvolutionMatchesTheCpu(int[] x, int m, int[] k, long[] strides, long[] dilations, long[] pads, int groups, bool bias, string autoPad)
    {
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            var attrs = new List<(string, object)> { ("strides", strides), ("dilations", dilations), ("group", (long)groups) };
            if (autoPad == "NOTSET")
            {
                attrs.Add(("pads", pads));
            }
            else
            {
                attrs.Add(("auto_pad", autoPad));
            }

            byte[] model = ConvModel(x, m, k, groups, bias, [.. attrs]).Build();
            using Tensor tx = Make(GpuTestHost.Random(Count(x), 3, 2f), x);
            double err = Compare(model, new() { ["x"] = tx }, mode, out SessionDiagnostics d);
            Assert.Equal(1, d.NodesOnDevice);
            Assert.True(err <= 1e-5, $"{mode} scaled error {err}");
        }
    }

    [Fact]
    public void OneDimensionalConvolution()
    {
        byte[] model = new ModelBuilder().Input("x", 1, 2, 6, 50).Output("y", 1)
            .Initializer("w", GpuTestHost.Random(10 * 3 * 5, 4, 0.5f), 10, 3, 5)
            .Node("Conv", ["x", "w"], ["y"], ("pads", new long[] { 2, 2 }), ("strides", new long[] { 2 }), ("group", 2L)).Build();
        using Tensor tx = Make(GpuTestHost.Random(2 * 6 * 50, 5, 2f), [2, 6, 50]);
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            Assert.True(Compare(model, new() { ["x"] = tx }, mode, out _) <= 1e-5);
        }
    }

    // Conv + BatchNormalization + Add (skip) + Relu, the body of a residual block, and a Clip and HardSwish after a depthwise one.
    [Theory]
    [InlineData(ArithmeticMode.Fast)]
    [InlineData(ArithmeticMode.Exact)]
    public void ResidualBlockFusesAndRunsInOneRegion(ArithmeticMode mode)
    {
        int c = 8;
        float[] vec(int seed, float shift) => GpuTestHost.Random(c, seed, 0.5f).Select(v => v + shift).ToArray();
        byte[] model = new ModelBuilder().Input("x", 1, 2, c, 12, 12).Output("y", 1)
            .Initializer("w1", GpuTestHost.Random(c * c * 9, 21, 0.3f), c, c, 3, 3).Initializer("s", vec(22, 1f), c).Initializer("be", vec(23, 0), c)
            .Initializer("mu", vec(24, 0), c).Initializer("va", vec(25, 1f), c)
            .Initializer("w2", GpuTestHost.Random(c * 9, 26, 0.3f), c, 1, 3, 3).Initializer("b2", vec(27, 0), c)
            .Node("Conv", ["x", "w1"], ["c1"], ("pads", new long[] { 1, 1, 1, 1 }))
            .Node("BatchNormalization", ["c1", "s", "be", "mu", "va"], ["n1"], ("epsilon", 1e-5f))
            .Node("Relu", ["n1"], ["r1"])
            .Node("Conv", ["r1", "w2", "b2"], ["c2"], ("pads", new long[] { 1, 1, 1, 1 }), ("group", (long)c))
            .Node("HardSwish", ["c2"], ["h2"])
            .Node("Add", ["h2", "x"], ["a"])
            .Node("Clip", ["a"], ["y"], ("min", 0f), ("max", 6f)).Build();
        using Tensor tx = Make(GpuTestHost.Random(2 * c * 144, 6, 3f), [2, c, 12, 12]);
        double err = Compare(model, new() { ["x"] = tx }, mode, out SessionDiagnostics d);
        Assert.True(err <= 1e-5, $"{mode} scaled error {err}");
        Assert.True(d.NodesOnDevice <= 4, $"the rewrites should have fused the chain, {d.NodesOnDevice} nodes remain");
    }

    [Theory]
    [InlineData(ArithmeticMode.Fast)]
    [InlineData(ArithmeticMode.Exact)]
    public void ConvolutionResidualAndActivation(ArithmeticMode mode)
    {
        int c = 6;
        byte[] model = new ModelBuilder().Input("x", 1, 1, c, 9, 9).Input("skip", 1, 1, c, 9, 9).Output("y", 1)
            .Initializer("w", GpuTestHost.Random(c * c * 9, 31, 0.3f), c, c, 3, 3).Initializer("b", GpuTestHost.Random(c, 32), c)
            .Node("Conv", ["x", "w", "b"], ["c"], ("pads", new long[] { 1, 1, 1, 1 }))
            .Node("Add", ["c", "skip"], ["a"]).Node("LeakyRelu", ["a"], ["y"], ("alpha", 0.1f)).Build();
        using Tensor tx = Make(GpuTestHost.Random(c * 81, 7, 3f), [1, c, 9, 9]), ts = Make(GpuTestHost.Random(c * 81, 8, 3f), [1, c, 9, 9]);
        double err = Compare(model, new() { ["x"] = tx, ["skip"] = ts }, mode, out SessionDiagnostics d);
        Assert.True(err <= 1e-5, $"{mode} scaled error {err}");
        Assert.Equal(1, d.NodesOnDevice);
    }

    public static TheoryData<string, int[], long[], long[], long[], long, long> PoolCases => new()
    {
        { "MaxPool", [1, 3, 15, 15], [3, 3], [2, 2], [1, 1, 1, 1], 0, 0 },
        { "MaxPool", [2, 4, 14, 13], [2, 2], [2, 2], [0, 0, 0, 0], 1, 0 },
        { "MaxPool", [1, 2, 9, 9], [3, 3], [1, 1], [1, 1, 1, 1], 0, 0 },
        { "AveragePool", [1, 3, 16, 16], [2, 2], [2, 2], [0, 0, 0, 0], 0, 0 },
        { "AveragePool", [2, 3, 15, 14], [3, 3], [2, 2], [1, 1, 1, 1], 0, 1 },
        { "AveragePool", [1, 3, 15, 14], [3, 3], [2, 2], [1, 1, 1, 1], 0, 0 },
        { "AveragePool", [1, 2, 9, 9], [3, 3], [2, 2], [1, 1, 1, 1], 1, 1 },
    };

    [Theory]
    [MemberData(nameof(PoolCases))]
    public void PoolingMatchesTheCpu(string op, int[] x, long[] kernel, long[] strides, long[] pads, long ceil, long includePad)
    {
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            var attrs = new List<(string, object)> { ("kernel_shape", kernel), ("strides", strides), ("pads", pads), ("ceil_mode", ceil) };
            if (op == "AveragePool")
            {
                attrs.Add(("count_include_pad", includePad));
            }

            byte[] model = new ModelBuilder().Input("x", 1, Dims(x)).Output("y", 1).Node(op, ["x"], ["y"], [.. attrs]).Build();
            using Tensor tx = Make(GpuTestHost.Random(Count(x), 9, 4f), x);
            double err = Compare(model, new() { ["x"] = tx }, mode, out _);
            Assert.True(err <= 1e-5, $"{op} {mode} scaled error {err}");
        }
    }

    [Theory]
    [InlineData("GlobalAveragePool", 7, 7)]
    [InlineData("GlobalAveragePool", 1, 1)]
    [InlineData("GlobalAveragePool", 13, 29)]
    [InlineData("GlobalMaxPool", 9, 5)]
    public void GlobalPoolingMatchesTheCpu(string op, int h, int w)
    {
        byte[] model = new ModelBuilder().Input("x", 1, 3, 10, h, w).Output("y", 1).Node(op, ["x"], ["y"]).Build();
        using Tensor tx = Make(GpuTestHost.Random(3 * 10 * h * w, 10, 4f), [3, 10, h, w]);
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            Assert.True(Compare(model, new() { ["x"] = tx }, mode, out _) <= 1e-5);
        }
    }

    [Fact]
    public void MaxPoolIndicesStayOnTheCpuAndRequireGpuNamesThem()
    {
        GpuTestHost.Require();
        byte[] model = new ModelBuilder().Input("x", 1, 1, 2, 8, 8).Output("y", 1).Output("i", 7)
            .Node("MaxPool", ["x"], ["y", "i"], ("kernel_shape", new long[] { 2, 2 }), ("strides", new long[] { 2, 2 })).Build();
        using Tensor tx = Make(GpuTestHost.Random(128, 1), [1, 2, 8, 8]);
        var feed = new Dictionary<string, Tensor> { ["x"] = tx };
        using Session s = GpuTestHost.Load(model, GpuTestHost.Gpu());
        SessionDiagnostics d = s.GetDiagnostics(feed);
        Assert.Equal(0, d.NodesOnDevice);
        Assert.Contains("indices", d.Nodes[0].FallbackReason);
        // The indices output is known only once the plan is made, so RequireGpu fails there (the first run or its diagnostics).
        using Session strict = GpuTestHost.Load(model, GpuTestHost.Gpu(require: true));
        var e = Assert.Throws<SessionException>(() => strict.GetDiagnostics(feed));
        Assert.Equal(SessionErrorKind.DeviceUnsupported, e.Kind);
        Assert.Contains("MaxPool", e.Message);
    }

    // mode, coordinate transformation, nearest mode, input h, w, output h, w (scales) or sizes
    public static TheoryData<string, string, string, int, int, float, float, bool> ResizeCases => new()
    {
        { "nearest", "asymmetric", "floor", 8, 8, 2f, 2f, false },
        { "nearest", "half_pixel", "round_prefer_floor", 7, 9, 2f, 3f, false },
        { "nearest", "half_pixel", "round_prefer_ceil", 5, 5, 0.6f, 0.6f, false },
        { "linear", "half_pixel", "round_prefer_floor", 6, 7, 2f, 2f, false },
        { "linear", "align_corners", "floor", 6, 7, 3f, 2f, false },
        { "linear", "pytorch_half_pixel", "floor", 9, 9, 0.5f, 0.5f, false },
        { "linear", "asymmetric", "floor", 10, 12, 13f, 17f, true },
        { "nearest", "half_pixel_symmetric", "ceil", 10, 10, 15f, 15f, true },
    };

    [Theory]
    [MemberData(nameof(ResizeCases))]
    public void ResizeMatchesTheCpu(string mode, string coord, string nearest, int h, int w, float a, float b, bool bySizes)
    {
        var builder = new ModelBuilder().Input("x", 1, 2, 3, h, w).Output("y", 1).Initializer("roi", Array.Empty<float>(), 0);
        string[] inputs;
        if (bySizes)
        {
            builder = builder.Initializer("sizes", new long[] { 2, 3, (long)a, (long)b }, 4);
            inputs = ["x", "roi", "", "sizes"];
        }
        else
        {
            builder = builder.Initializer("scales", [1f, 1f, a, b], 4);
            inputs = ["x", "roi", "scales"];
        }

        byte[] model = builder.Node("Resize", inputs, ["y"], ("mode", mode), ("coordinate_transformation_mode", coord), ("nearest_mode", nearest)).Build();
        using Tensor tx = Make(GpuTestHost.Random(2 * 3 * h * w, 14, 5f), [2, 3, h, w]);
        foreach (ArithmeticMode am in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            double err = Compare(model, new() { ["x"] = tx }, am, out _);
            Assert.True(err <= (mode == "nearest" ? 0 : 1e-5), $"{mode} {coord} {am} scaled error {err}");
        }
    }

    [Fact]
    public void TransposedConvolutionMatchesTheCpu()
    {
        byte[] model = new ModelBuilder().Input("x", 1, 2, 6, 7, 5).Output("y", 1)
            .Initializer("w", GpuTestHost.Random(6 * 4 * 3 * 3, 15, 0.5f), 6, 4, 3, 3).Initializer("b", GpuTestHost.Random(8, 16), 8)
            .Node("ConvTranspose", ["x", "w", "b"], ["y"], ("strides", new long[] { 2, 2 }), ("pads", new long[] { 1, 1, 1, 1 }), ("output_padding", new long[] { 1, 1 }), ("group", 2L)).Build();
        using Tensor tx = Make(GpuTestHost.Random(2 * 6 * 35, 17, 2f), [2, 6, 7, 5]);
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            double err = Compare(model, new() { ["x"] = tx }, mode, out _);
            Assert.True(err <= 1e-5, $"{mode} scaled error {err}");
        }
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(3, 1)]
    [InlineData(2, 2)]
    [InlineData(1, 0)]
    public void ConcatMatchesTheCpu(int batch, int axis)
    {
        byte[] model = new ModelBuilder().Input("a", 1, batch, 3, 5, 4).Input("b", 1, batch, 2, 5, 4).Input("c", 1, batch, 1, 5, 4).Output("y", 1).Node("Concat", ["a", "b", "c"], ["y"], ("axis", (long)axis)).Build();
        int[] sa = [batch, 3, 5, 4], sb = [batch, 2, 5, 4], sc = [batch, 1, 5, 4];
        if (axis != 1)
        {
            // Along another axis the other extents must agree.
            sa = [batch, 3, 5, 4];
            sb = [batch, 3, 5, 4];
            sc = [batch, 3, 5, 4];
            model = new ModelBuilder().Input("a", 1, Dims(sa)).Input("b", 1, Dims(sb)).Input("c", 1, Dims(sc)).Output("y", 1).Node("Concat", ["a", "b", "c"], ["y"], ("axis", (long)axis)).Build();
        }

        using Tensor ta = Make(GpuTestHost.Random(Count(sa), 1), sa), tb = Make(GpuTestHost.Random(Count(sb), 2), sb), tc = Make(GpuTestHost.Random(Count(sc), 3), sc);
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            Assert.Equal(0.0, Compare(model, new() { ["a"] = ta, ["b"] = tb, ["c"] = tc }, mode, out _));
        }
    }

    [Theory]
    [InlineData("HardSwish")]
    [InlineData("HardSigmoid")]
    public void HardActivationsMatchTheCpu(string op)
    {
        byte[] model = new ModelBuilder().Input("x", 1, 5, 37).Output("y", 1).Node(op, ["x"], ["y"]).Build();
        using Tensor tx = Make(GpuTestHost.Random(5 * 37, 18, 14f), [5, 37]);
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            Assert.True(Compare(model, new() { ["x"] = tx }, mode, out _) <= 1e-6);
        }
    }

    private static int BindGroups(GpuBackend backend) =>
        backend.Elementwise.BindGroupsCreated + backend.GetGemm(DeviceElement.F32).BindGroupsCreated + backend.Conv.BindGroupsCreated
        + backend.Pool.BindGroupsCreated + backend.Resize.BindGroupsCreated + backend.Layout.BindGroupsCreated;

    // Weights are uploaded once and the arena and bind groups are stable: the second run of a shape allocates nothing on the device.
    [Fact]
    public void SecondRunAllocatesAndBindsNothing()
    {
        GpuTestHost.Require();
        byte[] model = ConvModel([2, 8, 16, 16], 8, [3, 3], 1, true, ("pads", new long[] { 1, 1, 1, 1 })).Build();
        using Tensor tx = Make(GpuTestHost.Random(2 * 8 * 256, 3, 2f), [2, 8, 16, 16]);
        var feed = new Dictionary<string, Tensor> { ["x"] = tx };
        using Session s = GpuTestHost.Load(model, GpuTestHost.Gpu());
        using (s.Run(feed))
        {
        }

        var backend = (GpuBackend)s.Backend!;
        GpuSessionInfo before = backend.Snapshot(null);
        int groups = BindGroups(backend);
        using (s.Run(feed))
        {
        }

        GpuSessionInfo after = backend.Snapshot(null);
        Assert.Equal(before.DeviceAllocations, after.DeviceAllocations);
        Assert.Equal(groups, BindGroups(backend));
        GpuTestHost.AssertValidationClean(s);
    }
}
