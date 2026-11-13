using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tests.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu.Tests;

/// <summary>
/// Each operator the table maps, on a one-operator or short-chain model, against the CPU session in the same mode: shapes that hit
/// broadcasts, odd sizes and size one; float32 and float16; fast and exact. The model must run on the device completely.
/// </summary>
public class OpTests
{
    private static object[] Dims(int[] s) => [.. s.Select(d => (object)d)];

    private static int Count(int[] s) => s.Aggregate(1, (a, b) => a * b);

    private static Tensor Make(float[] v, int[] shape, bool half) =>
        half ? Tensor.FromSpan<Half>([.. v.Select(x => (Half)x)], Shape.Create(shape)) : Tensor.FromSpan<float>(v, Shape.Create(shape));

    private static float[] Floats(in Tensor t) => t.DType == DType.F16 ? [.. t.Data<Half>().ToArray().Select(h => (float)h)] : t.Data<float>().ToArray();

    private static byte[] HalfBytes(float[] v)
    {
        byte[] raw = new byte[v.Length * 2];
        for (int i = 0; i < v.Length; i++)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteHalfLittleEndian(raw.AsSpan(i * 2), (Half)v[i]);
        }

        return raw;
    }

    /// <summary>Runs on the CPU and the device in <paramref name="mode"/>; returns the worst scaled error and checks everything ran on the device.</summary>
    private static double Compare(byte[] model, Dictionary<string, Tensor> feed, ArithmeticMode mode, string output = "y", int expectRegions = 1)
    {
        GpuTestHost.Require();
        float[] expected;
        using (Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu(mode)))
        using (SessionOutputs o = cpu.Run(feed))
        {
            expected = Floats(o[output]);
        }

        using Session gpu = GpuTestHost.Load(model, GpuTestHost.Gpu(mode));
        float[] first;
        using (SessionOutputs o = gpu.Run(feed))
        {
            first = Floats(o[output]);
        }

        using (SessionOutputs o = gpu.Run(feed))
        {
            if (mode == ArithmeticMode.Exact)
            {
                Assert.Equal(GpuTestHost.Hash(first), GpuTestHost.Hash(Floats(o[output])));
            }
        }

        SessionDiagnostics d = gpu.GetDiagnostics(feed);
        Assert.True(d.Nodes.All(n => n.FallbackReason is null), string.Join("; ", d.Nodes.Where(n => n.FallbackReason is not null).Select(n => n.OpType + ": " + n.FallbackReason)));
        Assert.Equal(expectRegions, d.DeviceRegions);
        GpuTestHost.AssertValidationClean(gpu);
        return GpuTestHost.ScaledError(first, expected);
    }

    // ---- binary ------------------------------------------------------------------------------------------------------------

    public static TheoryData<string, int[], int[]> BinaryShapes => new()
    {
        { "Add", [4, 7, 33], [33] },
        { "Add", [5, 1, 9], [1, 6, 9] },
        { "Sub", [1], [1] },
        { "Mul", [3, 129], [3, 1] },
        { "Div", [2, 3, 5, 7], [7] },
        { "Max", [17], [17] },
        { "Min", [6, 1, 4], [3, 1] },
        { "Add", [1000], [] },
    };

    [Theory]
    [MemberData(nameof(BinaryShapes))]
    public void BinaryOperatorsMatchTheCpu(string op, int[] a, int[] b)
    {
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            byte[] model = new ModelBuilder().Input("a", 1, Dims(a)).Input("b", 1, Dims(b)).Output("y", 1).Node(op, ["a", "b"], ["y"]).Build();
            // Divisors stay away from zero.
            float[] bv = GpuTestHost.Random(Count(b), 2).Select(v => v + (v >= 0 ? 0.6f : -0.6f)).ToArray();
            using Tensor ta = Make(GpuTestHost.Random(Count(a), 1), a, false), tb = Make(bv, b, false);
            double err = Compare(model, new() { ["a"] = ta, ["b"] = tb }, mode);
            // Add, Sub, Mul, Min and Max are exact in single precision; Div is correctly rounded to 1e-5 at worst.
            double bound = op is "Div" ? 1e-5 : 0;
            Assert.True(err <= bound, $"{op} {string.Join('x', a)} {string.Join('x', b)} {mode}: {err:E2}");
        }
    }

    [Fact]
    public void PowAndConstantOperandsWork()
    {
        // Pow of a positive base by a constant scalar, then a bias that is a weight.
        byte[] model = new ModelBuilder().Input("a", 1, 5, 8).Output("y", 1)
            .Initializer("p", [1.5f], 1).Initializer("bias", GpuTestHost.Random(8, 3), 8)
            .Node("Pow", ["a", "p"], ["t"]).Node("Add", ["t", "bias"], ["y"]).Build();
        using Tensor ta = Make(GpuTestHost.Random(40, 1).Select(v => Math.Abs(v) + 0.1f).ToArray(), [5, 8], false);
        Assert.True(Compare(model, new() { ["a"] = ta }, ArithmeticMode.Fast) <= 1e-5);
    }

    // ---- unary -------------------------------------------------------------------------------------------------------------

    [Theory]
    [InlineData("Erf")]
    [InlineData("Tanh")]
    [InlineData("Exp")]
    [InlineData("Sigmoid")]
    [InlineData("Relu")]
    [InlineData("Neg")]
    [InlineData("Abs")]
    [InlineData("Sqrt")]
    [InlineData("Log")]
    [InlineData("Reciprocal")]
    [InlineData("LeakyRelu")]
    [InlineData("Clip")]
    public void UnaryOperatorsMatchTheCpu(string op)
    {
        bool positive = op is "Sqrt" or "Log" or "Reciprocal";
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            var b = new ModelBuilder().Input("x", 1, 3, 41).Output("y", 1);
            string[] inputs = ["x"];
            if (op == "Clip")
            {
                b.Initializer("lo", [-0.25f]).Initializer("hi", [0.3f]);
                inputs = ["x", "lo", "hi"];
            }

            byte[] model = op == "LeakyRelu" ? b.Node(op, inputs, ["y"], ("alpha", 0.2f)).Build() : b.Node(op, inputs, ["y"]).Build();
            float[] v = GpuTestHost.Random(3 * 41, 4, 6f).Select(x => positive ? Math.Abs(x) + 0.2f : x).ToArray();
            using Tensor x = Make(v, [3, 41], false);
            double err = Compare(model, new() { ["x"] = x }, mode);
            Assert.True(err <= 1e-5, $"{op} {mode}: {err:E2}");
        }
    }

    // ---- LayerNormalization, Softmax, Reduce ---------------------------------------------------------------------------------

    [Theory]
    [InlineData(1, 8, true)]
    [InlineData(5, 37, true)]
    [InlineData(7, 384, true)]
    [InlineData(3, 1024, false)]
    [InlineData(2, 3000, true)]
    public void LayerNormalizationMatchesTheCpu(int rows, int length, bool withBias)
    {
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            var b = new ModelBuilder().Input("x", 1, 2, rows, length).Output("y", 1)
                .Initializer("s", GpuTestHost.Random(length, 5).Select(v => v + 1f).ToArray(), length);
            string[] inputs = ["x", "s"];
            if (withBias)
            {
                b.Initializer("b", GpuTestHost.Random(length, 6), length);
                inputs = ["x", "s", "b"];
            }

            byte[] model = b.Node("LayerNormalization", inputs, ["y"], ("epsilon", 1e-5f)).Build();
            using Tensor x = Make(GpuTestHost.Random(2 * rows * length, 7, 4f), [2, rows, length], false);
            double err = Compare(model, new() { ["x"] = x }, mode);
            Assert.True(err <= 1e-5, $"layernorm {rows}x{length} {mode}: {err:E2}");
        }
    }

    [Theory]
    [InlineData(-1, 128)]
    [InlineData(-1, 5)]
    [InlineData(-1, 1000)]
    [InlineData(1, 12)]
    [InlineData(0, 9)]
    public void SoftmaxMatchesTheCpu(int axis, int size)
    {
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            byte[] model = new ModelBuilder().Input("x", 1, 4, size, 6).Output("y", 1).Node("Softmax", ["x"], ["y"], ("axis", (long)axis)).Build();
            using Tensor x = Make(GpuTestHost.Random(4 * size * 6, 8, 10f), [4, size, 6], false);
            double err = Compare(model, new() { ["x"] = x }, mode);
            Assert.True(err <= 1e-5, $"softmax axis {axis} size {size} {mode}: {err:E2}");
        }
    }

    [Theory]
    [InlineData("ReduceMean", new[] { -1 }, 1)]
    [InlineData("ReduceSum", new[] { 1 }, 0)]
    [InlineData("ReduceMax", new[] { 0, 2 }, 1)]
    [InlineData("ReduceMin", new[] { 2 }, 0)]
    [InlineData("ReduceL2", new[] { 1, 2 }, 1)]
    [InlineData("ReduceSumSquare", new int[0], 1)]
    public void ReductionsMatchTheCpu(string op, int[] axes, int keepdims)
    {
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            var b = new ModelBuilder(18).Input("x", 1, 5, 17, 9).Output("y", 1);
            string[] inputs = ["x"];
            if (axes.Length > 0)
            {
                b.Initializer("axes", axes.Select(a => (long)a).ToArray(), axes.Length);
                inputs = ["x", "axes"];
            }

            byte[] model = b.Node(op, inputs, ["y"], ("keepdims", (long)keepdims)).Build();
            using Tensor x = Make(GpuTestHost.Random(5 * 17 * 9, 9).Select(v => v + 0.7f).ToArray(), [5, 17, 9], false);
            double err = Compare(model, new() { ["x"] = x }, mode);
            Assert.True(err <= 1e-5, $"{op} axes [{string.Join(',', axes)}] {mode}: {err:E2}");
        }
    }

    // ---- layout -------------------------------------------------------------------------------------------------------------

    [Theory]
    [InlineData(new[] { 2, 3, 5, 4 }, new[] { 0, 2, 1, 3 })]
    [InlineData(new[] { 2, 3, 5, 4 }, new[] { 0, 2, 3, 1 })]
    [InlineData(new[] { 6, 7 }, new[] { 1, 0 })]
    [InlineData(new[] { 3, 1, 9 }, new[] { 2, 1, 0 })]
    public void TransposeMatchesTheCpuBitForBit(int[] shape, int[] perm)
    {
        byte[] model = new ModelBuilder().Input("x", 1, Dims(shape)).Output("y", 1).Node("Transpose", ["x"], ["y"], ("perm", perm.Select(p => (long)p).ToArray())).Build();
        using Tensor x = Make(GpuTestHost.Random(Count(shape), 3), shape, false);
        Assert.Equal(0, Compare(model, new() { ["x"] = x }, ArithmeticMode.Fast));
    }

    // Reshape, Unsqueeze and Squeeze are views: they cost no kernel and keep the whole chain in one region.
    [Fact]
    public void ViewsStayInsideTheRegionAndShareTheirSourcesMemory()
    {
        byte[] model = new ModelBuilder().Input("x", 1, 4, 6, 8).Output("y", 1)
            .Initializer("shape", [24L, 8L], 2).Initializer("axes", [0L], 1)
            .Initializer("w", GpuTestHost.Random(64, 3), 8, 8)
            .Node("Relu", ["x"], ["r"]).Node("Reshape", ["r", "shape"], ["v"]).Node("Unsqueeze", ["v", "axes"], ["u"])
            .Node("Squeeze", ["u", "axes"], ["q"]).Node("MatMul", ["q", "w"], ["y"]).Build();
        using Tensor x = Make(GpuTestHost.Random(4 * 6 * 8, 4), [4, 6, 8], false);
        double err = Compare(model, new() { ["x"] = x }, ArithmeticMode.Fast);
        Assert.True(err <= 1e-5, $"{err:E2}");
    }

    [Fact]
    public void ACastBetweenFloat32AndFloat16RoundsLikeTheCpu()
    {
        byte[] model = new ModelBuilder().Input("x", 1, 3, 20).Output("y", 1)
            .Node("Cast", ["x"], ["h"], ("to", 10L)).Node("Cast", ["h"], ["y"], ("to", 1L)).Build();
        using Tensor x = Make(GpuTestHost.Random(60, 5, 40f), [3, 20], false);
        Assert.Equal(0, Compare(model, new() { ["x"] = x }, ArithmeticMode.Fast));
    }

    // ---- float16 ------------------------------------------------------------------------------------------------------------

    [Fact]
    public void AHalfPrecisionBlockMatchesTheCpuWithinHalfTolerance()
    {
        const int B = 2, S = 6, H = 32;
        byte[] model = new ModelBuilder().Input("x", 10, B, S, H).Output("y", 10)
            .RawInitializer("w", 10, [H, H], HalfBytes(GpuTestHost.Random(H * H, 3)))
            .RawInitializer("bias", 10, [H], HalfBytes(GpuTestHost.Random(H, 4)))
            .RawInitializer("g", 10, [H], HalfBytes(GpuTestHost.Random(H, 5).Select(v => v + 1f).ToArray()))
            .RawInitializer("be", 10, [H], HalfBytes(GpuTestHost.Random(H, 6)))
            .Node("MatMul", ["x", "w"], ["m"]).Node("Add", ["m", "bias"], ["a"]).Node("Erf", ["a"], ["e"])
            .Node("LayerNormalization", ["e", "g", "be"], ["l"], ("epsilon", 1e-5f)).Node("Softmax", ["l"], ["y"], ("axis", -1L)).Build();
        using Tensor x = Make(GpuTestHost.Random(B * S * H, 7), [B, S, H], true);
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            double err = Compare(model, new() { ["x"] = x }, mode);
            Assert.True(err <= 2e-3, $"float16 block {mode}: {err:E2}");
        }
    }

    // ---- a transformer layer's shape ----------------------------------------------------------------------------------------

    // x[B,S,H] -> Q,K,V projections -> split heads -> scores = Q K^T * scale + mask -> softmax -> context -> merge -> output projection
    // + residual -> layer norm. Every operator is one a real encoder uses, and the whole chain is one region.
    [Fact]
    public void AnAttentionLayerRunsInOneRegionAndMatchesTheCpu()
    {
        const int B = 2, S = 9, NH = 3, D = 8, H = NH * D;
        var m = new ModelBuilder().Input("x", 1, B, S, H).Input("mask", 1, B, 1, 1, S).Output("y", 1);
        foreach (string w in new[] { "wq", "wk", "wv", "wo" })
        {
            m.Initializer(w, GpuTestHost.Random(H * H, w.GetHashCode() & 0xff, 0.4f), H, H).Initializer("b" + w[1], GpuTestHost.Random(H, 5), H);
        }

        m.Initializer("heads", [(long)B, S, NH, D], 4).Initializer("merged", [(long)B, S, H], 3).Initializer("scale", [1f / MathF.Sqrt(D)], 1)
            .Initializer("g", GpuTestHost.Random(H, 6).Select(v => v + 1f).ToArray(), H).Initializer("be", GpuTestHost.Random(H, 7), H);
        byte[] model = m
            .Node("MatMul", ["x", "wq"], ["q0"]).Node("Add", ["q0", "bq"], ["q1"]).Node("Reshape", ["q1", "heads"], ["q2"]).Node("Transpose", ["q2"], ["q"], ("perm", new long[] { 0, 2, 1, 3 }))
            .Node("MatMul", ["x", "wk"], ["k0"]).Node("Add", ["k0", "bk"], ["k1"]).Node("Reshape", ["k1", "heads"], ["k2"]).Node("Transpose", ["k2"], ["k"], ("perm", new long[] { 0, 2, 3, 1 }))
            .Node("MatMul", ["x", "wv"], ["v0"]).Node("Add", ["v0", "bv"], ["v1"]).Node("Reshape", ["v1", "heads"], ["v2"]).Node("Transpose", ["v2"], ["v"], ("perm", new long[] { 0, 2, 1, 3 }))
            .Node("MatMul", ["q", "k"], ["s0"]).Node("Mul", ["s0", "scale"], ["s1"]).Node("Add", ["s1", "mask"], ["s2"]).Node("Softmax", ["s2"], ["p"], ("axis", -1L))
            .Node("MatMul", ["p", "v"], ["c0"]).Node("Transpose", ["c0"], ["c1"], ("perm", new long[] { 0, 2, 1, 3 })).Node("Reshape", ["c1", "merged"], ["c"])
            .Node("MatMul", ["c", "wo"], ["o0"]).Node("Add", ["o0", "bo"], ["o1"]).Node("Add", ["o1", "x"], ["r"])
            .Node("LayerNormalization", ["r", "g", "be"], ["y"], ("epsilon", 1e-12f)).Build();
        float[] maskValues = new float[B * S];
        for (int i = 0; i < maskValues.Length; i++)
        {
            maskValues[i] = i % S >= S - 2 ? -10000f : 0f;
        }

        using Tensor x = Make(GpuTestHost.Random(B * S * H, 8), [B, S, H], false), mask = Make(maskValues, [B, 1, 1, S], false);
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            double err = Compare(model, new() { ["x"] = x, ["mask"] = mask }, mode);
            Assert.True(err <= 1e-5, $"attention layer {mode}: {err:E2}");
        }
    }

    [Fact]
    public void ModelsWithManyDistinctWeightsStillMatch()
    {
        // 90 products with different weights, more sets of buffers than a small bind-group cache would hold.
        var m = new ModelBuilder().Input("x", 1, 3, 16).Output("y", 1);
        string prev = "x";
        for (int i = 0; i < 90; i++)
        {
            string w = $"w{i}", o = i == 89 ? "y" : $"t{i}";
            m.Initializer(w, GpuTestHost.Random(256, 100 + i, 0.4f), 16, 16).Node("MatMul", [prev, w], [o]);
            prev = o;
        }

        using Tensor x = Make(GpuTestHost.Random(48, 3), [3, 16], false);
        Assert.True(Compare(m.Build(), new() { ["x"] = x }, ArithmeticMode.Fast) <= 1e-4);
    }
}
