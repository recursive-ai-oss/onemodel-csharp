using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tests.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu.Tests;

/// <summary>The pieces of an encoder body the fused forms cover: attention as exporters write it, the five-operator Gelu, and the embedding lookup.</summary>
public class TransformerTests
{
    private const int Heads = 3;

    // x [B, S, H*D] -> split heads -> softmax(Q Kt / sqrt(D) + mask) V -> merge heads, written as MatMul, Div, Add, Softmax, MatMul.
    private static byte[] AttentionChain(int d, bool maskPerRow)
    {
        int hd = Heads * d;
        var b = new ModelBuilder()
            .Input("x", 1, "B", "S", hd).Input("mask", 1, "B", 1, 1, "S").Output("y", 1, "B", "S", hd)
            .Initializer("split", [0L, 0L, Heads, d], 4).Initializer("merge", [0L, 0L, hd], 3)
            .Initializer("root", [MathF.Sqrt(d)], 1).Initializer("bias", [0.25f], 1);
        if (maskPerRow)
        {
            b = new ModelBuilder().Input("x", 1, "B", "S", hd).Input("mask", 1, "B", 1, "S", 1).Output("y", 1, "B", "S", hd)
                .Initializer("split", [0L, 0L, Heads, d], 4).Initializer("merge", [0L, 0L, hd], 3)
                .Initializer("root", [MathF.Sqrt(d)], 1).Initializer("bias", [0.25f], 1);
        }

        return b
            .Node("Reshape", ["x", "split"], ["x4"])
            .Node("Transpose", ["x4"], ["q"], ("perm", new long[] { 0, 2, 1, 3 }))
            .Node("Transpose", ["x4"], ["kt"], ("perm", new long[] { 0, 2, 3, 1 }))
            .Node("Transpose", ["x4"], ["v"], ("perm", new long[] { 0, 2, 1, 3 }))
            .Node("MatMul", ["q", "kt"], ["s0"]).Node("Div", ["s0", "root"], ["s1"]).Node("Add", ["s1", "mask"], ["s2"])
            .Node("Softmax", ["s2"], ["p"], ("axis", -1L)).Node("MatMul", ["p", "v"], ["o"])
            .Node("Transpose", ["o"], ["o2"], ("perm", new long[] { 0, 2, 1, 3 })).Node("Reshape", ["o2", "merge"], ["m"]).Node("Add", ["m", "bias"], ["y"]).Build();
    }

    private static float[] Run(byte[] model, ModelOptions options, int batch, int seq, int hd, float[] x, float[] mask, int[] maskShape, out SessionDiagnostics? d, bool diagnose)
    {
        using Session s = GpuTestHost.Load(model, options);
        using Tensor tx = Tensor.FromSpan<float>(x, Shape.Create(batch, seq, hd)), tm = Tensor.FromSpan<float>(mask, Shape.Create(maskShape));
        var feed = new Dictionary<string, Tensor> { ["x"] = tx, ["mask"] = tm };
        d = diagnose ? s.GetDiagnostics(feed) : null;
        using SessionOutputs o = s.Run(feed);
        float[] y = o["y"].Data<float>().ToArray();
        if (options.Device == ComputeDevice.Gpu)
        {
            GpuTestHost.AssertValidationClean(s);
        }

        return y;
    }

    [Theory]
    [InlineData(1, 1, 8, false)]
    [InlineData(2, 17, 8, false)]
    [InlineData(2, 70, 16, false)]
    [InlineData(3, 33, 32, false)]
    [InlineData(2, 9, 8, true)]
    public void FusedAttentionMatchesTheCpuInBothModes(int batch, int seq, int d, bool maskPerRow)
    {
        GpuTestHost.Require();
        // This test is about the kernel, so it asks for it.
        using IDisposable fused = GpuTestHost.With(new GpuOptions { Validation = true, CollectTimings = true, FusedAttention = true });
        byte[] model = AttentionChain(d, maskPerRow);
        int hd = Heads * d;
        float[] x = GpuTestHost.Random(batch * seq * hd, 5, 4f);
        int[] maskShape = maskPerRow ? [batch, 1, seq, 1] : [batch, 1, 1, seq];
        float[] mask = GpuTestHost.Random(maskShape.Aggregate(1, (a, c) => a * c), 6, 3f);
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            float[] cpu = Run(model, GpuTestHost.Cpu(mode), batch, seq, hd, x, mask, maskShape, out _, false);
            float[] gpu = Run(model, GpuTestHost.Gpu(mode), batch, seq, hd, x, mask, maskShape, out SessionDiagnostics? diag, true);
            Assert.Equal(1, diag!.DeviceRegions);
            Assert.DoesNotContain(diag.Nodes, n => n.OpType == "Softmax");
            Assert.All(diag.Nodes, n => Assert.Null(n.FallbackReason));
            double err = GpuTestHost.ScaledError(gpu, cpu);
            Assert.True(err <= 1e-5, $"{mode} scaled error {err}");
            if (mode == ArithmeticMode.Exact)
            {
                float[] again = Run(model, GpuTestHost.Gpu(mode), batch, seq, hd, x, mask, maskShape, out _, false);
                Assert.Equal(GpuTestHost.Hash(gpu), GpuTestHost.Hash(again));
            }
        }
    }

    [Theory]
    [InlineData(ArithmeticMode.Fast)]
    [InlineData(ArithmeticMode.Exact)]
    public void TheChainOptionKeepsTheSoftmaxOnTheDeviceAndMatchesTheCpu(ArithmeticMode mode)
    {
        GpuTestHost.Require();
        using IDisposable chain = GpuTestHost.With(new GpuOptions { Validation = true, CollectTimings = true, FusedAttention = false });
        const int batch = 2, seq = 33, d = 16;
        byte[] model = AttentionChain(d, false);
        int hd = Heads * d;
        float[] x = GpuTestHost.Random(batch * seq * hd, 5, 4f);
        int[] maskShape = [batch, 1, 1, seq];
        float[] mask = GpuTestHost.Random(batch * seq, 6, 3f);
        float[] cpu = Run(model, GpuTestHost.Cpu(mode), batch, seq, hd, x, mask, maskShape, out _, false);
        float[] gpu = Run(model, GpuTestHost.Gpu(mode), batch, seq, hd, x, mask, maskShape, out SessionDiagnostics? diag, true);
        Assert.Equal(1, diag!.DeviceRegions);
        Assert.Contains(diag.Nodes, n => n.OpType == "Softmax" && n.FallbackReason is null);
        double err = GpuTestHost.ScaledError(gpu, cpu);
        Assert.True(err <= 1e-5, $"{mode} scaled error {err}");
    }

    // Gelu written as x * 0.5 * (1 + erf(x / sqrt 2)), the form exporters used before the operator existed.
    [Theory]
    [InlineData(ArithmeticMode.Fast)]
    [InlineData(ArithmeticMode.Exact)]
    public void TheGeluChainRunsAsOneKernel(ArithmeticMode mode)
    {
        GpuTestHost.Require();
        byte[] model = new ModelBuilder()
            .Input("x", 1, "N", 50).Output("y", 1, "N", 50)
            .Initializer("r2", [MathF.Sqrt(2f)], 1).Initializer("one", [1f], 1).Initializer("half", [0.5f], 1)
            .Node("Div", ["x", "r2"], ["a"]).Node("Erf", ["a"], ["b"]).Node("Add", ["b", "one"], ["c"])
            .Node("Mul", ["x", "c"], ["d"]).Node("Mul", ["d", "half"], ["y"]).Build();
        using Tensor x = Tensor.FromSpan<float>(GpuTestHost.Random(7 * 50, 3, 8f), Shape.Create(7, 50));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        float[] cpu, gpu;
        using (Session c = GpuTestHost.Load(model, GpuTestHost.Cpu(mode)))
        using (SessionOutputs o = c.Run(feed))
        {
            cpu = o["y"].Data<float>().ToArray();
        }

        using Session g = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true));
        SessionDiagnostics d = g.GetDiagnostics(feed);
        Assert.Equal(1, d.NodesOnDevice);
        using (SessionOutputs o = g.Run(feed))
        {
            gpu = o["y"].Data<float>().ToArray();
        }

        Assert.True(GpuTestHost.ScaledError(gpu, cpu) <= 1e-5);
        GpuTestHost.AssertValidationClean(g);
    }

    private static byte[] Embedding(int rows, int width) => new ModelBuilder()
        .Input("ids", 7, "B", "S").Output("y", 1, "B", "S", width)
        .Initializer("table", GpuTestHost.Random(rows * width, 8), rows, width)
        .Node("Gather", ["table", "ids"], ["y"]).Build();

    [Theory]
    [InlineData(ArithmeticMode.Fast)]
    [InlineData(ArithmeticMode.Exact)]
    public void GatherLooksUpEmbeddingsAndRejectsABadId(ArithmeticMode mode)
    {
        GpuTestHost.Require();
        const int rows = 40, width = 24;
        byte[] model = Embedding(rows, width);
        long[] ids = [0, 39, 7, -1, 12, 12];
        using Tensor t = Tensor.FromSpan<long>(ids, Shape.Create(2, 3));
        var feed = new Dictionary<string, Tensor> { ["ids"] = t };
        float[] cpu;
        using (Session c = GpuTestHost.Load(model, GpuTestHost.Cpu(mode)))
        using (SessionOutputs o = c.Run(feed))
        {
            cpu = o["y"].Data<float>().ToArray();
        }

        using Session g = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true));
        SessionDiagnostics d = g.GetDiagnostics(feed);
        Assert.Equal(1, d.NodesOnDevice);
        using (SessionOutputs o = g.Run(feed))
        {
            Assert.Equal(cpu, o["y"].Data<float>().ToArray());
        }

        using Tensor bad = Tensor.FromSpan<long>([1, 2, 40, 3, 4, 5], Shape.Create(2, 3));
        SessionException e = Assert.Throws<SessionException>(() => g.Run(new Dictionary<string, Tensor> { ["ids"] = bad }));
        Assert.Equal(SessionErrorKind.InvalidInput, e.Kind);
        using Session c2 = GpuTestHost.Load(model, GpuTestHost.Cpu(mode));
        SessionException ce = Assert.Throws<SessionException>(() => c2.Run(new Dictionary<string, Tensor> { ["ids"] = bad }));
        Assert.Equal(ce.Message, e.Message);
        // The session still works afterwards.
        using (SessionOutputs o = g.Run(feed))
        {
            Assert.Equal(cpu, o["y"].Data<float>().ToArray());
        }

        GpuTestHost.AssertValidationClean(g);
    }

    // x -> DynamicQuantizeLinear -> MatMulInteger -> Cast -> Mul(scale) [-> Add(bias)], the pattern dynamically quantized encoders use. Every
    // step is exact integer arithmetic or one correctly rounded float operation, so the device gives the CPU's bits.
    [Theory]
    [InlineData(1, 8, 8, false, false)]
    [InlineData(5, 70, 33, true, false)]
    [InlineData(128, 96, 64, true, true)]
    [InlineData(7, 17, 130, false, true)]
    public void TheDynamicallyQuantizedProductIsBitExact(int m, int k, int n, bool perColumnScale, bool bias)
    {
        GpuTestHost.Require();
        var rng = new Random(m * 31 + k);
        byte[] w = new byte[k * n];
        for (int i = 0; i < w.Length; i++)
        {
            w[i] = (byte)(sbyte)rng.Next(-127, 128);
        }

        var b = new ModelBuilder()
            .Input("x", 1, "M", k).Output("y", 1, "M", n)
            .RawInitializer("w", 3, [k, n], w)
            .RawInitializer("wzp", 3, [], [(byte)(sbyte)3])
            .Initializer("ws", perColumnScale ? GpuTestHost.Random(n, 4, 0.02f).Select(v => Math.Abs(v) + 0.001f).ToArray() : [0.013f], perColumnScale ? [n] : [])
            .Node("DynamicQuantizeLinear", ["x"], ["xq", "xs", "xz"])
            .Node("MatMulInteger", ["xq", "w", "xz", "wzp"], ["acc"])
            .Node("Mul", ["xs", "ws"], ["sc"])
            .Node("Cast", ["acc"], ["accf"], ("to", 1L));
        if (bias)
        {
            b = b.Initializer("bias", GpuTestHost.Random(n, 5), n).Node("Mul", ["accf", "sc"], ["scaled"]).Node("Add", ["scaled", "bias"], ["y"]);
        }
        else
        {
            b = b.Node("Mul", ["accf", "sc"], ["y"]);
        }

        byte[] model = b.Build();
        using Tensor x = Tensor.FromSpan<float>(GpuTestHost.Random(m * k, 7, 6f), Shape.Create(m, k));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            float[] cpu;
            using (Session c = GpuTestHost.Load(model, GpuTestHost.Cpu(mode)))
            using (SessionOutputs o = c.Run(feed))
            {
                cpu = o["y"].Data<float>().ToArray();
            }

            using Session g = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true));
            SessionDiagnostics d = g.GetDiagnostics(feed);
            Assert.Equal(1, d.DeviceRegions);
            for (int rep = 0; rep < 2; rep++)
            {
                using SessionOutputs o = g.Run(feed);
                Assert.Equal(GpuTestHost.Hash(cpu), GpuTestHost.Hash(o["y"].Data<float>()));
            }

            GpuTestHost.AssertValidationClean(g);
        }
    }
}
