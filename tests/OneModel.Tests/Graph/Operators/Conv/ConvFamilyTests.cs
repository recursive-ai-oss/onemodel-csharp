using System.Text;
using System.Text.RegularExpressions;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.ConvFamily;

/// <summary>
/// The convolution / pooling / normalisation / resize family: generated cases checked against ONNX Runtime's recorded outputs,
/// the frozen backend node cases, a naive convolution reference, thread-count independence and allocation-free steady state.
/// The case directories come from tools/ops_conv_oracle.py (ONEMODEL_OPS_CONV) and tools/fetch_backend_test.py
/// (ONEMODEL_BACKEND_TEST); a test whose directory is not set does nothing. ONEMODEL_BACKEND_REPORT receives the per-case list.
/// </summary>
public partial class ConvFamilyTests
{
    [GeneratedRegex("^test_(conv_|convtranspose|basic_conv|maxpool|averagepool|globalaveragepool|globalmaxpool|batchnorm|resize|upsample)")]
    private static partial Regex WantedNode();

    [Theory]
    [InlineData(ExecutionMode.Fast)]
    [InlineData(ExecutionMode.Exact)]
    public void GeneratedCasesMatchOnnxRuntime(ExecutionMode mode)
    {
        string? dir = Environment.GetEnvironmentVariable("ONEMODEL_OPS_CONV");
        if (string.IsNullOrEmpty(dir))
        {
            return;
        }

        var failures = new List<string>();
        int pass = 0;
        var report = new StringBuilder();
        foreach (string c in Directory.GetDirectories(dir).Order(StringComparer.Ordinal))
        {
            string outcome = CaseRunner.Run(c, mode, scaled: true, out _);
            report.Append(Path.GetFileName(c)).Append(": ").AppendLine(outcome);
            if (outcome == "PASS")
            {
                pass++;
            }
            else
            {
                failures.Add($"{Path.GetFileName(c)}: {outcome}");
            }
        }

        Write(report, $"generated cases, mode {mode}: pass {pass}, fail {failures.Count}");
        Assert.True(failures.Count == 0, $"{failures.Count} failed of {pass + failures.Count}:{Environment.NewLine}{string.Join(Environment.NewLine, failures.Take(40))}");
    }

    [Theory]
    [InlineData(ExecutionMode.Fast)]
    [InlineData(ExecutionMode.Exact)]
    public void BackendNodeCases(ExecutionMode mode)
    {
        string? root = Environment.GetEnvironmentVariable("ONEMODEL_BACKEND_TEST");
        if (string.IsNullOrEmpty(root))
        {
            return;
        }

        string node = Directory.Exists(Path.Combine(root, "node")) ? Path.Combine(root, "node") : root;
        var failures = new List<string>();
        var report = new StringBuilder();
        var perOp = new SortedDictionary<string, (int Pass, int Fail, int Gap)>();
        foreach (string c in Directory.GetDirectories(node).Order(StringComparer.Ordinal))
        {
            string name = Path.GetFileName(c);
            if (!WantedNode().IsMatch(name) || name.Contains("training_mode", StringComparison.Ordinal))
            {
                continue;
            }

            string op = Regex.Match(name, "^test_([a-z]+)").Groups[1].Value;
            string outcome = CaseRunner.Run(c, mode, scaled: false, out long opset);
            report.Append(name).Append(" (opset ").Append(opset).Append("): ").AppendLine(outcome);
            (int p, int f, int g) = perOp.GetValueOrDefault(op);
            if (outcome == "PASS")
            {
                p++;
            }
            else if (outcome.StartsWith("UNSUPPORTED", StringComparison.Ordinal))
            {
                g++;
            }
            else
            {
                f++;
                failures.Add($"{name}: {outcome}");
            }

            perOp[op] = (p, f, g);
        }

        report.AppendLine($"backend cases, mode {mode}");
        foreach ((string op, (int p, int f, int g)) in perOp)
        {
            report.AppendLine($"{op}: pass {p}, fail {f}, gap {g}");
        }

        Write(report, "");
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    private static void Write(StringBuilder report, string summary)
    {
        if (summary.Length > 0)
        {
            report.AppendLine(summary);
        }

        string? file = Environment.GetEnvironmentVariable("ONEMODEL_BACKEND_REPORT");
        if (!string.IsNullOrEmpty(file))
        {
            File.AppendAllText(file, report.ToString());
        }
    }

    // Exact mode is bit-identical at 1 and 4 threads; the hash of every output is also written to the report, to compare across
    // instruction-set settings (DOTNET_EnableAVX2=0 and so on) and machines.
    [Fact]
    public void ExactOutputsAreBitIdenticalAcrossThreadCounts()
    {
        string? dir = Environment.GetEnvironmentVariable("ONEMODEL_OPS_CONV");
        if (string.IsNullOrEmpty(dir))
        {
            return;
        }

        string[] hashes = new string[2];
        int[] threads = [1, 4];
        for (int k = 0; k < 2; k++)
        {
            using var h = System.Security.Cryptography.IncrementalHash.CreateHash(System.Security.Cryptography.HashAlgorithmName.SHA256);
            foreach (string c in Directory.GetDirectories(dir).Order(StringComparer.Ordinal))
            {
                string outcome = CaseRunner.Run(c, ExecutionMode.Exact, scaled: true, out _, h, threads[k]);
                Assert.True(outcome == "PASS", $"{Path.GetFileName(c)}: {outcome}");
            }

            hashes[k] = Convert.ToHexString(h.GetHashAndReset());
        }

        Assert.Equal(hashes[0], hashes[1]);
        Write(new StringBuilder(), "");
        string? file = Environment.GetEnvironmentVariable("ONEMODEL_HASH_REPORT");
        if (!string.IsNullOrEmpty(file))
        {
            File.AppendAllText(file, $"exact-hash {hashes[0]}{Environment.NewLine}");
        }
    }

    // A naive double-accumulating convolution to compare the kernel against, over shapes that exercise padding, dilation, groups.
    [Theory]
    [InlineData(1, 3, 9, 8, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1)]
    [InlineData(2, 4, 7, 7, 6, 3, 3, 2, 2, 2, 1, 0, 2, 2, 1, 2)]
    [InlineData(1, 6, 10, 5, 6, 3, 2, 1, 2, 1, 0, 1, 1, 1, 2, 6)]
    [InlineData(1, 2, 6, 6, 4, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1)]
    public void ConvolutionMatchesNaiveReferenceAndIgnoresThreadCount(int n, int cin, int h, int w, int cout, int kh, int kw, int sh, int sw,
        int pt, int pl, int pb, int pr, int dh, int dw, int groups)
    {
        var s = new ConvShape(n, cin, h, w, cout, kh, kw, sh, sw, pt, pl, pb, pr, dh, dw, groups);
        var rng = new Random(5);
        float[] x = Fill(rng, n * cin * h * w), wt = Fill(rng, cout * (cin / groups) * kh * kw), bias = Fill(rng, cout);
        int outLen = n * cout * s.OutH * s.OutW;
        double[] expected = new double[outLen];
        int cinG = cin / groups, coutG = cout / groups;
        for (int img = 0; img < n; img++)
        {
            for (int oc = 0; oc < cout; oc++)
            {
                for (int oy = 0; oy < s.OutH; oy++)
                {
                    for (int ox = 0; ox < s.OutW; ox++)
                    {
                        double acc = bias[oc];
                        for (int ci = 0; ci < cinG; ci++)
                        {
                            for (int ky = 0; ky < kh; ky++)
                            {
                                for (int kx = 0; kx < kw; kx++)
                                {
                                    int iy = (oy * sh) - pt + (ky * dh), ix = (ox * sw) - pl + (kx * dw);
                                    if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                                    {
                                        int c = ((oc / coutG) * cinG) + ci;
                                        acc += (double)x[((((img * cin) + c) * h) + iy) * w + ix] * wt[((((oc * cinG) + ci) * kh) + ky) * kw + kx];
                                    }
                                }
                            }
                        }

                        expected[((((img * cout) + oc) * s.OutH) + oy) * s.OutW + ox] = acc;
                    }
                }
            }
        }

        foreach (KernelMode mode in new[] { KernelMode.Fast, KernelMode.Exact })
        {
            float[] one = new float[outLen], many = new float[outLen];
            Conv.Run(mode, KernelPaths.Best(), in s, x, wt, one, 1, bias);
            Conv.Run(mode, KernelPaths.Best(), in s, x, wt, many, 4, bias);
            Assert.Equal(one, many);
            double max = expected.Max(Math.Abs), err = 0;
            for (int i = 0; i < outLen; i++)
            {
                err = Math.Max(err, Math.Abs(one[i] - expected[i]));
            }

            Assert.True(err <= 1e-5 * Math.Max(1, max), $"{mode}: error {err} vs max {max}");
        }
    }

    // NaN in max pooling follows IEEE maxNum: ignored unless the whole window is NaN. Ties go to the first element, and the index is
    // flat over the whole tensor (column-major within each plane for storage_order 1).
    [Fact]
    public void MaxPoolNaNRuleAndIndices()
    {
        float nan = float.NaN;
        // One plane 2x4, kernel 1x2, stride 1x2: windows [nan, 1], [2, 2], then with a second plane [nan, nan], [5, nan].
        float[] x = [nan, 1, 2, 2, nan, nan, 5, nan];
        var g = new PoolGeom(new Dim3(1, 1, 4), new Dim3(1, 1, 2), new Dim3(1, 1, 2), new Dim3(1, 1, 1), default, default, new Dim3(1, 1, 2));
        float[] y = new float[4];
        long[] idx = new long[4];
        Pooling.Max<float>(g, 2, x, y, idx, columnMajor: false);
        Assert.Equal(1f, y[0]);
        Assert.Equal(2f, y[1]);
        Assert.True(float.IsNaN(y[2]));
        Assert.Equal(5f, y[3]);
        Assert.Equal(new long[] { 1, 2, 4, 6 }, idx);
        Pooling.GlobalMax([nan, -1f, nan, nan], y.AsSpan(0, 2), 2);
        Assert.Equal(-1f, y[0]);
        Assert.Equal(float.MinValue, y[1]);
    }

    // Indicative single-thread timings of the heavy shapes (ResNet-50 and MobileNetV3 layers), printed to the report file when
    // ONEMODEL_OPS_TIMING names one. Not a benchmark: the bar runner owns those.
    [Fact]
    public void IndicativeConvTimings()
    {
        string? file = Environment.GetEnvironmentVariable("ONEMODEL_OPS_TIMING");
        if (string.IsNullOrEmpty(file))
        {
            return;
        }

        (string Name, ConvShape S)[] shapes =
        [
            ("resnet stem 7x7/2 3->64 224", new ConvShape(1, 3, 224, 224, 64, 7, 7, 2, 2, 3, 3, 3, 3, 1, 1, 1)),
            ("resnet 3x3 64->64 56", new ConvShape(1, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
            ("resnet 1x1 256->64 56", new ConvShape(1, 256, 56, 56, 64, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1)),
            ("mobilenet dw 3x3 /2 240 28", new ConvShape(1, 240, 28, 28, 240, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 240)),
            ("mobilenet dw 5x5 672 14", new ConvShape(1, 672, 14, 14, 672, 5, 5, 1, 1, 2, 2, 2, 2, 1, 1, 672)),
            ("whisper conv1d 3 /1 80->384 as 1x3000", new ConvShape(1, 80, 1, 3000, 384, 1, 3, 1, 1, 0, 1, 0, 1, 1, 1, 1)),
        ];
        var rng = new Random(3);
        var sb = new StringBuilder();
        foreach ((string name, ConvShape s) in shapes)
        {
            float[] x = Fill(rng, s.N * s.Cin * s.H * s.W), w = Fill(rng, s.Cout * (s.Cin / s.Groups) * s.KH * s.KW), bias = Fill(rng, s.Cout);
            float[] y = new float[s.N * s.Cout * s.OutH * s.OutW];
            for (int i = 0; i < 3; i++)
            {
                Conv.Run(KernelMode.Fast, KernelPaths.Best(), in s, x, w, y, 1, bias);
            }

            double best = double.MaxValue;
            for (int r = 0; r < 5; r++)
            {
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                Conv.Run(KernelMode.Fast, KernelPaths.Best(), in s, x, w, y, 1, bias);
                best = Math.Min(best, System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
            }

            sb.AppendLine($"{name}: {best:F2} ms, {s.Flops / (best * 1e6):F1} GFLOP/s (1 thread, path {KernelPaths.Best()})");
        }

        File.AppendAllText(file, sb.ToString());
    }

    private static float[] Fill(Random rng, int n)
    {
        float[] a = new float[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = (float)((rng.NextDouble() * 2) - 1);
        }

        return a;
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

    // One model through every operator of the family, cut after the first `nodes` of them: a Conv with bias, BatchNormalization,
    // MaxPool with indices, a depthwise Conv, Resize, ConvTranspose, AveragePool, GlobalAveragePool. A steady-state run on one thread
    // allocates nothing.
    [Theory]
    [InlineData(ExecutionMode.Fast, 1)]
    [InlineData(ExecutionMode.Fast, 2)]
    [InlineData(ExecutionMode.Fast, 3)]
    [InlineData(ExecutionMode.Fast, 4)]
    [InlineData(ExecutionMode.Fast, 5)]
    [InlineData(ExecutionMode.Fast, 6)]
    [InlineData(ExecutionMode.Fast, 7)]
    [InlineData(ExecutionMode.Fast, 8)]
    [InlineData(ExecutionMode.Exact, 8)]
    public void SteadyStateRunAllocatesNothing(ExecutionMode mode, int nodes)
    {
        var rng = new Random(9);
        var model = new ModelBuilder(17)
            .Input("x", 1, 1, 4, 16, 16)
            .Initializer("w1", Fill(rng, 8 * 4 * 3 * 3), 8, 4, 3, 3)
            .Initializer("b1", Fill(rng, 8), 8)
            .Initializer("scale", Fill(rng, 8), 8)
            .Initializer("bnb", Fill(rng, 8), 8)
            .Initializer("mean", Fill(rng, 8), 8)
            .Initializer("var", [.. Fill(rng, 8).Select(Math.Abs).Select(v => v + 0.5f)], 8)
            .Initializer("wd", Fill(rng, 8 * 1 * 3 * 3), 8, 1, 3, 3)
            .Initializer("wt", Fill(rng, 8 * 8 * 2 * 2), 8, 8, 2, 2)
            .Initializer("scales", [1f, 1f, 2f, 2f], 4)
            .Initializer("roi", Array.Empty<float>(), 0);
        string[] last = ["c1", "bn", "mp", "dw", "up", "ct", "ap", "y"];
        for (int i = 0; i < nodes; i++)
        {
            _ = i switch
            {
                0 => model.Node("Conv", ["x", "w1", "b1"], ["c1"], ("kernel_shape", new long[] { 3, 3 }), ("pads", new long[] { 1, 1, 1, 1 })),
                1 => model.Node("BatchNormalization", ["c1", "scale", "bnb", "mean", "var"], ["bn"]),
                2 => model.Node("MaxPool", ["bn"], ["mp", "idx"], ("kernel_shape", new long[] { 2, 2 }), ("strides", new long[] { 2, 2 })),
                3 => model.Node("Conv", ["mp", "wd"], ["dw"], ("kernel_shape", new long[] { 3, 3 }), ("pads", new long[] { 1, 1, 1, 1 }), ("group", 8L)),
                4 => model.Node("Resize", ["dw", "roi", "scales"], ["up"]),
                5 => model.Node("ConvTranspose", ["up", "wt"], ["ct"], ("kernel_shape", new long[] { 2, 2 }), ("strides", new long[] { 2, 2 })),
                6 => model.Node("AveragePool", ["ct"], ["ap"], ("kernel_shape", new long[] { 2, 2 }), ("strides", new long[] { 2, 2 })),
                _ => model.Node("GlobalAveragePool", ["ap"], ["y"]),
            };
        }

        var cut = model.Output(last[nodes - 1], 1, "a", "b", "c", "d");
        using IrModel ir = OnnxImporter.Load(cut.Build());
        using var s = Session.Create(ir, new SessionOptions { Mode = mode, IntraOpThreads = 1 });
        float[] input = Fill(rng, 4 * 16 * 16), y = new float[8 * 16 * 16 * 4];
        int[] shape = [1, 4, 16, 16];
        for (int i = 0; i < 20; i++)
        {
            s.Run("x", input.AsSpan(), shape, last[nodes - 1], y.AsSpan());
        }

        float[] first = (float[])y.Clone();
        Assert.Equal(0, AllocatedBytes(() =>
        {
            for (int i = 0; i < 200; i++)
            {
                s.Run("x", input.AsSpan(), shape, last[nodes - 1], y.AsSpan());
            }
        }));
        Assert.Equal(first, y);
    }
}
