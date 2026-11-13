using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tests.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu.Tests;

/// <summary>MatMul and Gemm on the device against the CPU session, over shapes that hit tile edges, size 1 and every batch pattern.</summary>
public class MatMulTests
{
    public static TheoryData<string, int[], int[]> MatMulShapes => new()
    {
        { "tall B is 2-D", [7, 33], [33, 19] },
        { "size one", [1, 1], [1, 1] },
        { "k one", [5, 1], [1, 9] },
        { "tile edges", [65, 17], [17, 130] },
        { "batched A, 2-D B", [3, 5, 16], [16, 8] },
        { "equal batches", [2, 4, 6, 8], [2, 4, 8, 5] },
        { "A shared", [6, 4], [3, 4, 7] },
        { "3-D by 3-D", [4, 9, 12], [4, 12, 3] },
        { "vector by matrix", [13], [13, 6] },
        { "matrix by vector", [6, 13], [13] },
        { "ones in B batch", [5, 3, 4], [1, 4, 6] },
    };

    [Theory]
    [MemberData(nameof(MatMulShapes))]
    public void FastMatchesTheCpu(string name, int[] a, int[] b) => Compare(name, a, b, ArithmeticMode.Fast, 1e-5);

    [Theory]
    [MemberData(nameof(MatMulShapes))]
    public void ExactMatchesTheCpuAndIsReproducible(string name, int[] a, int[] b) => Compare(name, a, b, ArithmeticMode.Exact, 1e-5);

    private static void Compare(string name, int[] a, int[] b, ArithmeticMode mode, double bound)
    {
        GpuTestHost.Require();
        byte[] model = new ModelBuilder()
            .Input("a", 1, a.Select(d => (object)d).ToArray()).Input("b", 1, b.Select(d => (object)d).ToArray())
            .Output("y", 1)
            .Node("MatMul", ["a", "b"], ["y"]).Build();
        float[] av = GpuTestHost.Random(a.Aggregate(1, (x, y) => x * y), 1), bv = GpuTestHost.Random(b.Aggregate(1, (x, y) => x * y), 2);
        using Tensor ta = Tensor.FromSpan<float>(av, Shape.Create(a)), tb = Tensor.FromSpan<float>(bv, Shape.Create(b));
        var inputs = new Dictionary<string, Tensor> { ["a"] = ta, ["b"] = tb };

        float[] expected, first, second;
        using (Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu(mode)))
        using (SessionOutputs o = cpu.Run(inputs))
        {
            expected = o["y"].Data<float>().ToArray();
        }

        using Session gpu = GpuTestHost.Load(model, GpuTestHost.Gpu(mode));
        using (SessionOutputs o = gpu.Run(inputs))
        {
            first = o["y"].Data<float>().ToArray();
        }

        using (SessionOutputs o = gpu.Run(inputs))
        {
            second = o["y"].Data<float>().ToArray();
        }

        SessionDiagnostics d = gpu.GetDiagnostics(inputs);
        Assert.Equal(1, d.NodesOnDevice);
        double err = GpuTestHost.ScaledError(first, expected);
        Assert.True(err <= bound, $"{name} {mode}: scaled error {err:E2} above {bound:E1}");
        if (mode == ArithmeticMode.Exact)
        {
            Assert.Equal(GpuTestHost.Hash(first), GpuTestHost.Hash(second));
        }

        GpuTestHost.AssertValidationClean(gpu);
    }

    [Theory]
    [InlineData(false, false, 1f, 1f, "none")]
    [InlineData(true, false, 1f, 1f, "n")]
    [InlineData(false, true, 0.5f, 2f, "mn")]
    [InlineData(true, true, 2f, 0.25f, "m1")]
    [InlineData(false, false, 1f, 1f, "scalar")]
    [InlineData(false, false, 1f, 0f, "mn")]
    [InlineData(true, false, 1.5f, 1f, "1n")]
    public void GemmMatchesTheCpu(bool transA, bool transB, float alpha, float beta, string cForm)
    {
        GpuTestHost.Require();
        const int M = 37, N = 21, K = 19;
        long[] sa = transA ? [K, M] : [M, K], sb = transB ? [N, K] : [K, N];
        long[]? cs = cForm switch
        {
            "n" => [N],
            "mn" => [M, N],
            "m1" => [M, 1],
            "1n" => [1, N],
            "scalar" => [],
            _ => null,
        };
        int clen = cs is null ? 0 : (int)cs.Aggregate(1L, (x, y) => x * y);
        foreach (bool constantC in new[] { true, false })
        {
            var b = new ModelBuilder().Input("a", 1, (int)sa[0], (int)sa[1]).Output("y", 1, M, N)
                .Initializer("b", GpuTestHost.Random(K * N, 3), sb);
            string[] inputs = ["a", "b"];
            if (cs is not null)
            {
                inputs = ["a", "b", "c"];
                if (constantC)
                {
                    b.Initializer("c", GpuTestHost.Random(clen, 4), cs);
                }
                else
                {
                    b.Input("c", 1, cs.Select(d => (object)(int)d).ToArray());
                }
            }

            byte[] model = b.Node("Gemm", inputs, ["y"], ("transA", transA ? 1L : 0L), ("transB", transB ? 1L : 0L), ("alpha", alpha), ("beta", beta)).Build();
            float[] av = GpuTestHost.Random(M * K, 5), cv = GpuTestHost.Random(Math.Max(clen, 1), 6);
            using Tensor ta = Tensor.FromSpan<float>(av, Shape.Create((int)sa[0], (int)sa[1]));
            using Tensor tc = cs is not null && !constantC ? Tensor.FromSpan<float>(cv.AsSpan(0, clen), Shape.Create(cs.Select(d => (int)d).ToArray())) : default;
            var feed = new Dictionary<string, Tensor> { ["a"] = ta };
            if (!tc.IsEmpty)
            {
                feed["c"] = tc;
            }

            float[] expected;
            using (Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu()))
            using (SessionOutputs o = cpu.Run(feed))
            {
                expected = o["y"].Data<float>().ToArray();
            }

            using Session gpu = GpuTestHost.Load(model, GpuTestHost.Gpu());
            using SessionOutputs g = gpu.Run(feed);
            double err = GpuTestHost.ScaledError(g["y"].Data<float>(), expected);
            Assert.True(err <= 1e-5, $"transA={transA} transB={transB} alpha={alpha} beta={beta} C={cForm} const={constantC}: {err:E2}");
            Assert.Equal(1, gpu.GetDiagnostics(feed).NodesOnDevice);
            GpuTestHost.AssertValidationClean(gpu);
        }
    }

    [Fact]
    public void HalfPrecisionProductMatchesTheCpuWithinHalfTolerance()
    {
        GpuTestHost.Require();
        const int M = 9, K = 40, N = 24;
        float[] wv = GpuTestHost.Random(K * N, 7);
        byte[] raw = new byte[wv.Length * 2];
        for (int i = 0; i < wv.Length; i++)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteHalfLittleEndian(raw.AsSpan(i * 2), (Half)wv[i]);
        }

        byte[] model = new ModelBuilder().Input("x", 10, M, K).Output("y", 10, M, N).RawInitializer("w", 10, [K, N], raw)
            .Node("MatMul", ["x", "w"], ["y"]).Build();
        Half[] xv = [.. GpuTestHost.Random(M * K, 8).Select(v => (Half)v)];
        using Tensor x = Tensor.FromSpan<Half>(xv, Shape.Create(M, K));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        float[] expected;
        using (Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu()))
        using (SessionOutputs o = cpu.Run(feed))
        {
            expected = [.. o["y"].Data<Half>().ToArray().Select(h => (float)h)];
        }

        using Session gpu = GpuTestHost.Load(model, GpuTestHost.Gpu());
        using SessionOutputs g = gpu.Run(feed);
        Assert.Equal(DType.F16, g["y"].DType);
        double err = GpuTestHost.ScaledError([.. g["y"].Data<Half>().ToArray().Select(h => (float)h)], expected);
        Assert.True(err <= 2e-3, $"float16 scaled error {err:E2}");
        Assert.Equal(1, gpu.GetDiagnostics(feed).NodesOnDevice);
        GpuTestHost.AssertValidationClean(gpu);
    }
}
