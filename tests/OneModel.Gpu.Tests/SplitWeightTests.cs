using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tests.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu.Tests;

/// <summary>
/// Weights above one buffer binding (a vocabulary table, a logits projection) are held in pieces. The tests lower the session's binding
/// limit so a small weight takes that path on any device, then compare with the CPU and with the same model unsplit.
/// </summary>
public class SplitWeightTests
{
    private const long Limit = 100_000;   // bytes: the 256 KB tables below take three pieces

    private static Session LoadSplit(byte[] model, ArithmeticMode mode, bool require = true)
    {
        Session s = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require));
        ((GpuBackend)s.Backend!).MaxBindingBytes = Limit;
        return s;
    }

    private static float[] RunCpu(byte[] model, Dictionary<string, Tensor> feed, ArithmeticMode mode)
    {
        using Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu(mode));
        using SessionOutputs o = cpu.Run(feed);
        return o["y"].Data<float>().ToArray();
    }

    [Theory]
    [InlineData(ArithmeticMode.Fast)]
    [InlineData(ArithmeticMode.Exact)]
    public void ATableAboveTheBindingIsGatheredFromItsPieces(ArithmeticMode mode)
    {
        GpuTestHost.Require();
        const int Rows = 1000, Width = 64;
        float[] table = GpuTestHost.Random(Rows * Width, 3);
        byte[] model = new ModelBuilder().Input("ids", 7, 2, "N").Output("y", 1).Initializer("t", table, Rows, Width)
            .Node("Gather", ["t", "ids"], ["y"]).Build();
        // Rows at the start, the end, around each piece boundary, repeated, and counted from the end.
        long[] ids = [0, 999, 333, 334, 335, 666, 667, 998, 5, 5, -1, -1000, 500, 1, 0, 400];
        using Tensor t = Tensor.FromSpan<long>(ids, Shape.Create(2, 8));
        var feed = new Dictionary<string, Tensor> { ["ids"] = t };
        float[] expected = RunCpu(model, feed, mode);
        using Session gpu = LoadSplit(model, mode);
        float[] got;
        using (SessionOutputs o = gpu.Run(feed))
        {
            got = o["y"].Data<float>().ToArray();
        }

        Assert.Equal(expected, got);   // a copy: bit for bit
        SessionDiagnostics d = gpu.GetDiagnostics(feed);
        Assert.Equal(1, d.NodesOnDevice);
        long weights = OneModelGpu.GetInfo(gpu).ConstantBuffers;
        Assert.True(weights >= 3, $"{weights} weight buffers");
        GpuTestHost.AssertValidationClean(gpu);

        // A bad id is the CPU's error, and the session still runs good inputs.
        long[] bad = [.. ids];
        bad[3] = 1000;
        using Tensor tb = Tensor.FromSpan<long>(bad, Shape.Create(2, 8));
        SessionException e = Assert.Throws<SessionException>(() => gpu.Run(new Dictionary<string, Tensor> { ["ids"] = tb }));
        Assert.Equal(SessionErrorKind.InvalidInput, e.Kind);
        Assert.Contains("Gather index 1000 is out of range for an extent of 1000.", e.Message);
        using SessionOutputs again = gpu.Run(feed);
        Assert.Equal(expected, again["y"].Data<float>().ToArray());
    }

    [Theory]
    [InlineData(ArithmeticMode.Fast, false)]
    [InlineData(ArithmeticMode.Exact, false)]
    [InlineData(ArithmeticMode.Fast, true)]
    [InlineData(ArithmeticMode.Exact, true)]
    public void AProductWithAWeightAboveTheBindingIsSplitAlongItsColumns(ArithmeticMode mode, bool transposed)
    {
        GpuTestHost.Require();
        const int K = 32, N = 2000;   // 256 KB
        var builder = new ModelBuilder().Input("x", 1, "M", K).Output("y", 1);
        byte[] model = transposed
            ? builder.Initializer("w", GpuTestHost.Random(N * K, 4), N, K).Node("Gemm", ["x", "w"], ["y"], ("transB", 1L)).Build()
            : builder.Initializer("w", GpuTestHost.Random(K * N, 4), K, N).Node("MatMul", ["x", "w"], ["y"]).Build();
        foreach (int m in new[] { 1, 5, 12 })
        {
            using Tensor x = Tensor.FromSpan<float>(GpuTestHost.Random(m * K, 5), Shape.Create(m, K));
            var feed = new Dictionary<string, Tensor> { ["x"] = x };
            float[] expected = RunCpu(model, feed, mode);
            float[] whole;
            using (Session one = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true)))
            using (SessionOutputs o = one.Run(feed))
            {
                whole = o["y"].Data<float>().ToArray();
            }

            using Session gpu = LoadSplit(model, mode);
            float[] got;
            using (SessionOutputs o = gpu.Run(feed))
            {
                got = o["y"].Data<float>().ToArray();
            }

            // The split changes no element's summation order: it equals the unsplit run bit for bit.
            Assert.Equal(GpuTestHost.Hash(whole), GpuTestHost.Hash(got));
            Assert.True(GpuTestHost.ScaledError(got, expected) <= 1e-5, $"m={m}: {GpuTestHost.ScaledError(got, expected)}");
            Assert.True(OneModelGpu.GetInfo(gpu).ConstantBuffers >= 3);
            using SessionOutputs second = gpu.Run(feed);
            Assert.Equal(GpuTestHost.Hash(got), GpuTestHost.Hash(second["y"].Data<float>()));
            GpuTestHost.AssertValidationClean(gpu);
        }
    }

    [Fact]
    public void AGemmWithABiasAndAnOversizedWeightFallsBackWithTheReason()
    {
        GpuTestHost.Require();
        const int K = 32, N = 2000;
        byte[] model = new ModelBuilder().Input("x", 1, 3, K).Output("y", 1)
            .Initializer("w", GpuTestHost.Random(K * N, 4), K, N).Initializer("c", GpuTestHost.Random(N, 6), N)
            .Node("Gemm", ["x", "w", "c"], ["y"]).Build();
        using Tensor x = Tensor.FromSpan<float>(GpuTestHost.Random(3 * K, 5), Shape.Create(3, K));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        using Session gpu = LoadSplit(model, ArithmeticMode.Fast, require: false);
        SessionDiagnostics d = gpu.GetDiagnostics(feed);
        Assert.Equal(0, d.NodesOnDevice);
        Assert.Contains("larger than one buffer binding", d.Nodes[0].FallbackReason);
        using SessionOutputs o = gpu.Run(feed);
        Assert.Equal(3 * N, o["y"].Length);
    }
}
