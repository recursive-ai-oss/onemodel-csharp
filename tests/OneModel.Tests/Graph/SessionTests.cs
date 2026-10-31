using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text.Json;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Operators;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Graph;

public class SessionTests
{
    private static string GraphsDir([CallerFilePath] string file = "") =>
        Path.GetFullPath(Path.Combine(Path.GetDirectoryName(file)!, "..", "..", "..", "testdata", "graphs"));

    private static string GraphPath(string name) => Path.Combine(GraphsDir(), name);

    private sealed record Case(int[] Shape, float[] Input, int[] OutShape, float[] Output);

    private static Case LoadCase(string name)
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(GraphPath("reference.json")));
        JsonElement c = doc.RootElement.GetProperty("cases").GetProperty(name);
        return new Case(
            [.. c.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32())],
            [.. c.GetProperty("input").EnumerateArray().Select(e => (float)e.GetDouble())],
            [.. c.GetProperty("outShape").EnumerateArray().Select(e => e.GetInt32())],
            [.. c.GetProperty("output").EnumerateArray().Select(e => (float)e.GetDouble())]);
    }

    internal static double ScaledError(ReadOnlySpan<float> ours, ReadOnlySpan<float> reference)
    {
        Assert.Equal(reference.Length, ours.Length);
        double max = 0, diff = 0;
        for (int i = 0; i < ours.Length; i++)
        {
            max = Math.Max(max, Math.Abs(reference[i]));
            diff = Math.Max(diff, Math.Abs((double)ours[i] - reference[i]));
        }

        return max == 0 ? diff : diff / max;
    }

    internal static ModelOptions Options(ArithmeticMode mode, int threads = 0) => new() { Mode = mode, IntraOpThreads = threads };

    internal static string Hash(ReadOnlySpan<float> data) =>
        Convert.ToHexString(SHA256.HashData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(data)));

    [Theory]
    [InlineData(ArithmeticMode.Fast, "mlp-1")]
    [InlineData(ArithmeticMode.Fast, "mlp-5")]
    [InlineData(ArithmeticMode.Exact, "mlp-1")]
    [InlineData(ArithmeticMode.Exact, "mlp-5")]
    [InlineData(ArithmeticMode.Fast, "chain-5")]
    [InlineData(ArithmeticMode.Exact, "chain-5")]
    public void ModelMatchesOnnxRuntime(ArithmeticMode mode, string caseName)
    {
        Case c = LoadCase(caseName);
        using var m = Model.Load(GraphPath(caseName.StartsWith("mlp", StringComparison.Ordinal) ? "mlp.onnx" : "chain.onnx"), Options(mode));
        float[] y = m.Run(c.Input);
        Assert.True(ScaledError(y, c.Output) <= 1e-4, $"scaled error {ScaledError(y, c.Output)}");
        float[] y2 = m.Run(c.Input, c.Shape);
        Assert.Equal(y, y2);
    }

    [Fact]
    public void SessionReportsShapesAndPlansPerInputShape()
    {
        using var s = Session.Load(GraphPath("mlp.onnx"));
        TensorInfo x = Assert.Single(s.Inputs);
        Assert.Equal("x", x.Name);
        Assert.Equal(DType.F32, x.DType);
        Assert.Equal(2, x.Rank);
        Assert.False(x.Dims[0].IsFixed);
        Assert.Equal("N", x.Dims[0].Param);
        Assert.Equal(16, x.Dims[1].Value);
        Assert.Equal(0, s.CachedPlanCount);

        using var inp = Tensor.Zeros(DType.F32, Shape.Create(3, 16));
        Span<Tensor> outs = [default];
        s.Run(["x"], [inp], ["y"], outs);
        Assert.Equal(Shape.Create(3, 2, 4), outs[0].Shape);
        outs[0].Dispose();
        outs[0] = default;
        Assert.Equal(1, s.CachedPlanCount);
        s.Run(["x"], [inp], ["y"], outs);
        outs[0].Dispose();
        outs[0] = default;
        Assert.Equal(1, s.CachedPlanCount);
        using var inp2 = Tensor.Zeros(DType.F32, Shape.Create(4, 16));
        s.Run(["x"], [inp2], ["y"], outs);
        outs[0].Dispose();
        Assert.Equal(2, s.CachedPlanCount);
    }

    [Fact]
    public void PlanCacheIsBounded()
    {
        using var s = Session.Load(GraphPath("mlp.onnx"), new ModelOptions { MaxCachedPlans = 3 });
        float[] y = new float[64 * 8];
        for (int n = 1; n <= 6; n++)
        {
            s.Run("x", new float[n * 16].AsSpan(), [n, 16], "y", y.AsSpan());
        }

        Assert.Equal(3, s.CachedPlanCount);
    }

    [Fact]
    public void ThreadCountDoesNotChangeBitsInEitherMode()
    {
        // Large enough that the product splits into several tiles.
        var rng = new Random(7);
        float[] w = [.. Enumerable.Range(0, 300 * 300).Select(_ => (float)rng.NextDouble() - 0.5f)];
        float[] x = [.. Enumerable.Range(0, 200 * 300).Select(_ => (float)rng.NextDouble() - 0.5f)];
        byte[] model = new ModelBuilder()
            .Input("x", 1, "N", 300).Output("y", 1, "N", 300).Initializer("w", w, 300, 300)
            .Node("MatMul", ["x", "w"], ["m"]).Node("Relu", ["m"], ["y"]).Build();
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            string? first = null;
            foreach (int threads in new[] { 1, 2, 5, 32 })
            {
                using var s = Session.Load(new MemoryStream(model), Options(mode, threads));
                float[] y = new float[200 * 300];
                s.Run("x", x.AsSpan(), [200, 300], "y", y.AsSpan());
                string h = Hash(y);
                first ??= h;
                Assert.Equal(first, h);
            }
        }
    }

    [Fact]
    public void ExactModeIsTheSameOnEveryKernelPath()
    {
        Case c = LoadCase("mlp-5");
        string? first = null;
        foreach (KernelPath path in KernelPaths.All.Where(KernelPaths.IsSupported))
        {
            using IrModel ir = OnnxImporter.Load(GraphPath("mlp.onnx"));
            using var s = Session.Create(ir, Options(ArithmeticMode.Exact, 1), OpRegistry.Default, path, default);
            float[] y = new float[40];
            s.Run("x", c.Input.AsSpan(), c.Shape, "y", y.AsSpan());
            first ??= Hash(y);
            Assert.True(first == Hash(y), $"path {path} differs");
        }
    }

    [Fact]
    public void ConcurrentRunsGiveIdenticalOutputs()
    {
        Case c = LoadCase("mlp-5");
        using var s = Session.Load(GraphPath("mlp.onnx"));
        float[] expected = new float[40];
        s.Run("x", c.Input.AsSpan(), c.Shape, "y", expected.AsSpan());
        var failures = new System.Collections.Concurrent.ConcurrentBag<string>();
        Parallel.For(0, 8, new ParallelOptions { MaxDegreeOfParallelism = 8 }, _ =>
        {
            float[] y = new float[40];
            for (int i = 0; i < 200; i++)
            {
                Array.Clear(y);
                s.Run("x", c.Input.AsSpan(), c.Shape, "y", y.AsSpan());
                if (!y.AsSpan().SequenceEqual(expected))
                {
                    failures.Add("output differs");
                    return;
                }
            }
        });
        Assert.Empty(failures);
    }

    [Fact]
    public void ConcurrentRunsWithDifferentShapesShareThePlanCache()
    {
        using var s = Session.Load(GraphPath("mlp.onnx"));
        var failures = new System.Collections.Concurrent.ConcurrentBag<string>();
        Parallel.For(0, 8, new ParallelOptions { MaxDegreeOfParallelism = 8 }, t =>
        {
            int n = (t % 4) + 1;
            float[] x = new float[n * 16];
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = ((i * 7) % 13) - 6;
            }

            float[] mine = new float[n * 8];
            s.Run("x", x.AsSpan(), [n, 16], "y", mine.AsSpan());
            for (int i = 0; i < 100; i++)
            {
                float[] y = new float[n * 8];
                s.Run("x", x.AsSpan(), [n, 16], "y", y.AsSpan());
                if (!y.AsSpan().SequenceEqual(mine))
                {
                    failures.Add("differs");
                }
            }
        });
        Assert.Empty(failures);
        Assert.Equal(4, s.CachedPlanCount);
    }

    private static long AllocatedBytes(Action loop)
    {
        // Best of three, as in PoolTests: runtime recompilation on the test thread is not the code under test.
        long best = long.MaxValue;
        for (int attempt = 0; attempt < 3 && best != 0; attempt++)
        {
            long before = GC.GetAllocatedBytesForCurrentThread();
            loop();
            best = Math.Min(best, GC.GetAllocatedBytesForCurrentThread() - before);
        }

        return best;
    }

    [Theory]
    [InlineData(ArithmeticMode.Fast)]
    [InlineData(ArithmeticMode.Exact)]
    public void SteadyStateRunAllocatesNothing(ArithmeticMode mode)
    {
        Case c = LoadCase("mlp-5");
        using var s = Session.Load(GraphPath("mlp.onnx"), Options(mode, 1));
        float[] y = new float[40];
        for (int i = 0; i < 20; i++)
        {
            s.Run("x", c.Input.AsSpan(), c.Shape, "y", y.AsSpan());
        }

        Assert.Equal(0, AllocatedBytes(() =>
        {
            for (int i = 0; i < 500; i++)
            {
                s.Run("x", c.Input.AsSpan(), c.Shape, "y", y.AsSpan());
            }
        }));

        // The happy path with caller-provided output is allocation-free too.
        using var m = Model.Load(GraphPath("mlp.onnx"), Options(mode, 1));
        for (int i = 0; i < 20; i++)
        {
            m.Run(c.Input, y);
        }

        Assert.Equal(0, AllocatedBytes(() =>
        {
            for (int i = 0; i < 500; i++)
            {
                m.Run(c.Input, y);
            }
        }));
    }

    [Fact]
    public void TensorRunWithCallerOutputIsAllocationFreeToo()
    {
        Case c = LoadCase("mlp-5");
        using var s = Session.Load(GraphPath("mlp.onnx"), Options(ArithmeticMode.Fast, 1));
        using Tensor x = Tensor.FromSpan<float>(c.Input, Shape.Create(c.Shape));
        using Tensor y = Tensor.Create(DType.F32, Shape.Create(5, 2, 4));
        string[] inNames = ["x"], outNames = ["y"];
        Tensor[] ins = [x], outs = [y];
        for (int i = 0; i < 20; i++)
        {
            s.Run(inNames, ins, outNames, outs);
        }

        Assert.Equal(0, AllocatedBytes(() =>
        {
            for (int i = 0; i < 500; i++)
            {
                s.Run(inNames, ins, outNames, outs);
            }
        }));
        Assert.True(ScaledError(y.Data<float>(), c.Output) <= 1e-4);
    }

    [Fact]
    public void GemmAllocatesNothingOnOneThread()
    {
        float[] a = new float[5 * 16], b = new float[16 * 32], c = new float[5 * 32];
        for (int i = 0; i < 20; i++)
        {
            Gemm.Run(KernelMode.Fast, KernelPaths.Best(), 5, 32, 16, a, 16, b, 32, c, 32, 1);
        }

        Assert.Equal(0, AllocatedBytes(() =>
        {
            for (int i = 0; i < 500; i++)
            {
                Gemm.Run(KernelMode.Fast, KernelPaths.Best(), 5, 32, 16, a, 16, b, 32, c, 32, 1);
            }
        }));
    }

    [Fact]
    public void SharedBuffersAreReusedByLiveness()
    {
        using var s = Session.Load(GraphPath("chain.onnx"));
        (long planned, long unshared, long slab) = s.PlanStatistics([DType.F32], [Shape.Create(5, 64)]);
        // Eleven intermediates of 1280 bytes each would need 14080 bytes; the chain only ever needs two live at once.
        Assert.Equal(11L * 1280, unshared);
        Assert.True(planned <= 3 * 1280, $"planned {planned}");
        Assert.True(slab >= planned);
    }

    [Fact]
    public void ExactModeRefusesAnOperatorWithoutAnExactKernel()
    {
        var defs = OpTable.All.Where(d => d.OpType != "Relu").Append(
            new OpDef("Relu", 6, int.MaxValue, static _ => new NotExactRelu())).ToArray();
        using IrModel ir = OnnxImporter.Load(GraphPath("mlp.onnx"));
        SessionException e = Assert.Throws<SessionException>(() =>
            Session.Create(ir, Options(ArithmeticMode.Exact), new OpRegistry(defs), null, default));
        Assert.Equal(SessionErrorKind.ExactModeUnavailable, e.Kind);
        Assert.Contains("Relu", e.Message);
        Assert.Equal(["Relu"], e.Operators);

        // The same kernel is fine in fast mode.
        using var s = Session.Create(ir, Options(ArithmeticMode.Fast), new OpRegistry(defs), null, default);
        Assert.Single(s.Inputs);
    }

    private sealed class NotExactRelu : OpKernel
    {
        public override bool SupportsExact => false;

        public override void Infer(InferContext ctx) => ctx.SetOutput(0, ctx.InputType(0), ctx.InputShape(0));

        public override void Run(in OpRun run) => run.Inputs[0].Data<float>().CopyTo(run.Outputs[0].MutableData<float>());
    }

    [Fact]
    public void UnsupportedOperatorsAreAllNamed()
    {
        byte[] model = new ModelBuilder()
            .Input("x", 1, 4).Output("y", 1, 4)
            .Node("DeformConv", ["x"], ["a"]).Node("FancyThing", ["a"], ["b"]).Node("Relu", ["b"], ["y"]).Build();
        SessionException e = Assert.Throws<SessionException>(() => Session.Load(new MemoryStream(model)));
        Assert.Equal(SessionErrorKind.UnsupportedOperator, e.Kind);
        Assert.Contains("DeformConv", e.Message);
        Assert.Contains("FancyThing", e.Message);
        Assert.Equal(2, e.Operators.Length);
    }

    [Fact]
    public void BadInputsAreRejectedWithTypedErrors()
    {
        using var s = Session.Load(GraphPath("mlp.onnx"));
        float[] y = new float[8];
        SessionException e = Assert.Throws<SessionException>(() => s.Run("nope", new float[16].AsSpan(), [1, 16], "y", y.AsSpan()));
        Assert.Equal(SessionErrorKind.InvalidInput, e.Kind);
        Assert.Contains("'x'", e.Message);
        e = Assert.Throws<SessionException>(() => s.Run("x", new float[16].AsSpan(), [1, 16], "nope", y.AsSpan()));
        Assert.Contains("'y'", e.Message);
        e = Assert.Throws<SessionException>(() => s.Run("x", new long[16].AsSpan(), [1, 16], "y", y.AsSpan()));
        Assert.Contains("F32", e.Message);
        e = Assert.Throws<SessionException>(() => s.Run("x", new float[16].AsSpan(), [16], "y", y.AsSpan()));
        Assert.Contains("2 dimensions", e.Message);
        e = Assert.Throws<SessionException>(() => s.Run("x", new float[15].AsSpan(), [1, 15], "y", y.AsSpan()));
        Assert.Contains("dimension 1", e.Message);
        e = Assert.Throws<SessionException>(() => s.Run("x", new float[16].AsSpan(), [2, 16], "y", y.AsSpan()));
        Assert.Contains("values", e.Message);
        e = Assert.Throws<SessionException>(() => s.Run("x", new float[32].AsSpan(), [2, 16], "y", new float[3].AsSpan()));
        Assert.Contains("buffer holds 3", e.Message);
    }

    [Fact]
    public void DisposedSessionRefusesToRun()
    {
        var s = Session.Load(GraphPath("mlp.onnx"));
        s.Dispose();
        s.Dispose();
        Assert.Throws<ObjectDisposedException>(() => s.Run("x", new float[16].AsSpan(), [1, 16], "y", new float[8].AsSpan()));
    }

    [Fact]
    public void CancellationStopsLoading()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => Session.Load(GraphPath("mlp.onnx"), null, null, cts.Token));
    }

    [Fact]
    public void HappyPathInfersTheBatchAndRejectsAmbiguity()
    {
        using var m = Model.Load(GraphPath("mlp.onnx"));
        Assert.Equal(8, m.Run(new float[16]).Length);
        Assert.Equal(40, m.Run(new float[80]).Length);
        Assert.Throws<SessionException>(() => m.Run(new float[17]));
        Assert.Throws<SessionException>(() => m.Run(new long[16]));
        float[] viaSpan = m.Run<float, float>(new float[16].AsSpan(), [1, 16]);
        Assert.Equal(8, viaSpan.Length);

        // Two open extents cannot be completed from a length alone.
        byte[] model = new ModelBuilder().Input("x", 1, "A", "B").Output("y", 1, "A", "B").Node("Relu", ["x"], ["y"]).Build();
        using var amb = Model.Load(new MemoryStream(model));
        SessionException e = Assert.Throws<SessionException>(() => amb.Run(new float[6]));
        Assert.Contains("pass the shape", e.Message);
        Assert.Equal(6, amb.Run(new float[6], [2, 3]).Length);
    }

    [Fact]
    public void HappyPathNeedsOneInput()
    {
        byte[] model = new ModelBuilder().Input("a", 1, 2).Input("b", 1, 2).Output("y", 1, 2).Node("Add", ["a", "b"], ["y"]).Build();
        using var m = Model.Load(new MemoryStream(model));
        SessionException e = Assert.Throws<SessionException>(() => m.Run(new float[2]));
        Assert.Contains("2 inputs", e.Message);
        // The tensor API handles it.
        using Tensor a = Tensor.FromSpan<float>([1, 2], Shape.Create(2)), b = Tensor.FromSpan<float>([10, 20], Shape.Create(2));
        Span<Tensor> outs = [default];
        m.Session.Run(["b", "a"], [b, a], ["y"], outs);
        Assert.Equal([11f, 22f], outs[0].ToArray<float>());
        outs[0].Dispose();
    }

    // ---- run-time shaping --------------------------------------------------------------------------------------------------

    [Fact]
    public void ReshapeWithARuntimeShapeIsShapedAtRunTime()
    {
        byte[] model = new ModelBuilder()
            .Input("x", 1, "N", 6).Input("shape", 7, 2).Output("y", 1, "A", "B")
            .Initializer("bias", [1f, 2f, 3f, 4f], 4)
            .Node("Reshape", ["x", "shape"], ["r"]).Node("Add", ["r", "bias"], ["y"]).Build();
        using var s = Session.Load(new MemoryStream(model));
        using Tensor x = Tensor.FromSpan<float>([.. Enumerable.Range(0, 12).Select(i => (float)i)], Shape.Create(2, 6));
        using Tensor shape = Tensor.FromSpan<long>([3, 4], Shape.Create(2));
        Span<Tensor> outs = [default];
        s.Run(["x", "shape"], [x, shape], ["y"], outs);
        Assert.Equal(Shape.Create(3, 4), outs[0].Shape);
        Assert.Equal([1f, 3f, 5f, 7f, 5f, 7f, 9f, 11f, 9f, 11f, 13f, 15f], outs[0].ToArray<float>());
        outs[0].Dispose();
        outs[0] = default;

        using Tensor shape2 = Tensor.FromSpan<long>([-1, 4], Shape.Create(2));
        s.Run(["x", "shape"], [x, shape2], ["y"], outs);
        Assert.Equal(Shape.Create(3, 4), outs[0].Shape);
        outs[0].Dispose();
        outs[0] = default;

        using Tensor bad = Tensor.FromSpan<long>([5, 4], Shape.Create(2));
        Tensor[] scratchOut = [default];
        Assert.Throws<SessionException>(() => s.Run(["x", "shape"], [x, bad], ["y"], scratchOut));
    }

    [Fact]
    public void DataDependentOutputsAreAllocatedByTheKernelAndShapeEverythingAfterIt()
    {
        var defs = OpTable.All.Append(new OpDef("PositiveValues", 1, int.MaxValue, static _ => new PositiveValues())).ToArray();
        byte[] model = new ModelBuilder()
            .Input("x", 1, "N").Output("y", 1, "M")
            .Initializer("one", [1f], 1)
            .Node("PositiveValues", ["x"], ["p"]).Node("Add", ["p", "one"], ["a"]).Node("Relu", ["a"], ["y"]).Build();
        using IrModel ir = OnnxImporter.Load(model);
        using var s = Session.Create(ir, Options(ArithmeticMode.Fast), new OpRegistry(defs), null, default);
        float[] y = new float[16];
        Shape r = s.Run("x", new float[] { -1, 2, -3, 4, 5 }.AsSpan(), [5], "y", y.AsSpan());
        Assert.Equal(Shape.Create(3), r);
        Assert.Equal([3f, 5f, 6f], y.AsSpan(0, 3).ToArray());
        r = s.Run("x", new float[] { 7, -1 }.AsSpan(), [2], "y", y.AsSpan());
        Assert.Equal(Shape.Create(1), r);
        Assert.Equal(8f, y[0]);
        r = s.Run("x", new float[] { -7, -1 }.AsSpan(), [2], "y", y.AsSpan());
        Assert.Equal(Shape.Create(0), r);
    }

    private sealed class PositiveValues : OpKernel
    {
        public override bool OutputsDataDependent => true;

        public override void Infer(InferContext ctx) => throw new InvalidOperationException("not called for data-dependent kernels");

        public override void Run(in OpRun run)
        {
            ReadOnlySpan<float> x = run.Inputs[0].Data<float>();
            int count = 0;
            foreach (float v in x)
            {
                count += v > 0 ? 1 : 0;
            }

            Tensor o = run.AllocateOutput(0, DType.F32, Shape.Create(count));
            Span<float> d = o.MutableData<float>();
            int k = 0;
            foreach (float v in x)
            {
                if (v > 0)
                {
                    d[k++] = v;
                }
            }
        }
    }

    [Fact]
    public void ViewsOfInputsAndConstantsComeOutRight()
    {
        // Identity of an input, reshape of an input, reshape of a weight, all as graph outputs.
        byte[] model = new ModelBuilder()
            .Input("x", 1, 2, 3).Output("same", 1, 2, 3).Output("flat", 1, 6).Output("w2", 1, 3, 2)
            .Initializer("w", [1f, 2f, 3f, 4f, 5f, 6f], 6)
            .Initializer("s1", [6L], 1).Initializer("s2", [3L, 2L], 2)
            .Node("Identity", ["x"], ["same"]).Node("Reshape", ["x", "s1"], ["flat"]).Node("Reshape", ["w", "s2"], ["w2"]).Build();
        using var s = Session.Load(new MemoryStream(model));
        using Tensor x = Tensor.FromSpan<float>([9, 8, 7, 6, 5, 4], Shape.Create(2, 3));
        Span<Tensor> outs = [default, default, default];
        s.Run(["x"], [x], ["same", "flat", "w2"], outs);
        Assert.Equal(Shape.Create(2, 3), outs[0].Shape);
        Assert.Equal([9f, 8, 7, 6, 5, 4], outs[0].ToArray<float>());
        Assert.Equal(Shape.Create(6), outs[1].Shape);
        Assert.Equal([9f, 8, 7, 6, 5, 4], outs[1].ToArray<float>());
        Assert.Equal(Shape.Create(3, 2), outs[2].Shape);
        Assert.Equal([1f, 2, 3, 4, 5, 6], outs[2].ToArray<float>());
        foreach (Tensor t in outs)
        {
            t.Dispose();
        }
    }

    [Fact]
    public void ANonContiguousInputIsAccepted()
    {
        using var s = Session.Load(GraphPath("mlp.onnx"));
        Case c = LoadCase("mlp-5");
        using Tensor all = Tensor.FromSpan<float>(c.Input, Shape.Create(c.Shape));
        // The transpose of the transpose of a 16x5 matrix has the original values but the view is strided.
        using Tensor tr = Tensor.Create(DType.F32, Shape.Create(16, 5));
        tr.CopyFrom(all.Permute(1, 0));
        Tensor view = tr.Permute(1, 0);
        Assert.False(view.IsContiguous);
        Span<Tensor> outs = [default];
        s.Run(["x"], [view], ["y"], outs);
        Assert.True(ScaledError(outs[0].Data<float>(), c.Output) <= 1e-4);
        outs[0].Dispose();
    }
}
