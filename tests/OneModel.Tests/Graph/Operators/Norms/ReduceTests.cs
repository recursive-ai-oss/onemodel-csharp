using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using static RecursiveAI.OneModel.Tests.Graph.Operators.Norms.NormHarness;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.Norms;

public class ReduceTests
{
    private static object[] Dims(int[] d) => [.. d.Cast<object>()];

    private static double Reduce(string op, List<double> v)
    {
        switch (op)
        {
            case "ReduceSum": return v.Sum();
            case "ReduceMean": return v.Count == 0 ? 0 : v.Sum() / v.Count;
            case "ReduceMax": return v.Count == 0 ? double.NegativeInfinity : v.Max();
            case "ReduceMin": return v.Count == 0 ? double.PositiveInfinity : v.Min();
            case "ReduceProd": return v.Aggregate(1.0, (a, b) => a * b);
            case "ReduceL1": return v.Sum(Math.Abs);
            case "ReduceL2": return Math.Sqrt(v.Sum(a => a * a));
            case "ReduceSumSquare": return v.Sum(a => a * a);
            case "ReduceLogSum": return Math.Log(v.Sum());
            default: return Math.Log(v.Sum(Math.Exp));
        }
    }

    // Naive reference: for every output tuple, collect the reduced elements.
    private static double[] Ref(string op, float[] x, int[] dims, bool[] reduced, bool keep, out int[] outDims)
    {
        outDims = dims.Select((d, i) => reduced[i] ? (keep ? 1 : -1) : d).Where(d => d >= 0).ToArray();
        var groups = new SortedDictionary<long, List<double>>();
        int[] idx = new int[dims.Length];
        for (int flat = 0; flat < x.Length; flat++)
        {
            int rem = flat;
            for (int d = dims.Length - 1; d >= 0; d--)
            {
                idx[d] = rem % dims[d];
                rem /= dims[d];
            }

            long key = 0;
            for (int d = 0; d < dims.Length; d++)
            {
                if (!reduced[d])
                {
                    key = (key * dims[d]) + idx[d];
                }
            }

            if (!groups.TryGetValue(key, out List<double>? list))
            {
                groups[key] = list = [];
            }

            list.Add(x[flat]);
        }

        long outCount = dims.Select((d, i) => reduced[i] ? 1L : d).Aggregate(1L, (a, b) => a * b);
        double[] y = new double[outCount];
        for (long k = 0; k < outCount; k++)
        {
            y[k] = Reduce(op, groups.TryGetValue(k, out List<double>? l) ? l : []);
        }

        return y;
    }

    private static byte[] Model(string op, int opset, int[] dims, int[]? axes, bool keep, int[] outDims, bool noop = false)
    {
        bool axesInput = op == "ReduceSum" ? opset >= 13 : opset >= 18;
        var mb = new ModelBuilder(opset).Input("x", 1, Dims(dims)).Output("y", 1, Dims(outDims));
        var attrs = new List<(string, object)> { ("keepdims", keep ? 1L : 0L) };
        string[] ins = ["x"];
        if (noop)
        {
            attrs.Add(("noop_with_empty_axes", 1L));
        }

        if (axes is not null && axesInput)
        {
            mb.Initializer("axes", axes.Select(a => (long)a).ToArray(), axes.Length);
            ins = ["x", "axes"];
        }
        else if (axes is not null)
        {
            attrs.Add(("axes", axes.Select(a => (long)a).ToArray()));
        }

        return mb.Node(op, ins, ["y"], [.. attrs]).Build();
    }

    public static readonly string[] Ops =
        ["ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2", "ReduceSumSquare", "ReduceLogSum", "ReduceLogSumExp"];

    public static IEnumerable<object[]> OpCases() => Ops.Select(o => new object[] { o });

    [Theory]
    [MemberData(nameof(OpCases))]
    public void EveryAxisSubsetMatchesTheNaiveReference(string op)
    {
        var rng = new Random(op.Length);
        foreach (int[] dims in new[] { new[] { 2, 3, 4, 5 }, new[] { 7, 33 }, new[] { 1, 5, 1 }, new[] { 40, 3 }, new[] { 3, 0, 2 }, new[] { 17 }, new[] { 4, 1, 6, 1, 3 } })
        {
            float[] x = Random(rng, Count(dims), 0.5, 1.5);
            for (int mask = 1; mask < 1 << dims.Length; mask++)
            {
                bool[] red = [.. Enumerable.Range(0, dims.Length).Select(i => (mask >> i & 1) != 0)];
                int[] axes = [.. Enumerable.Range(0, dims.Length).Where(i => red[i]).Select(i => rng.Next(2) == 0 ? i : i - dims.Length)];
                foreach (bool keep in new[] { true, false })
                {
                    double[] want = Ref(op, x, dims, red, keep, out int[] outDims);
                    foreach (int opset in new[] { 13, 18 })
                    {
                        byte[] model = Model(op, opset, dims, axes, keep, outDims);
                        Out[] got = Run(model, ExecutionMode.Fast, null, 1, 1, (x, dims));
                        Assert.Equal(Shape.Create(outDims), got[0].Shape);
                        double err = ScaledError(got[0].F!, want);
                        Assert.True(err < 1e-5, $"{op} opset {opset} dims {string.Join('x', dims)} mask {mask} keep {keep}: {err}");
                    }
                }
            }
        }
    }

    [Fact]
    public void AllAxesWhenNoneAreGivenAndNoopWithEmptyAxes()
    {
        var rng = new Random(1);
        float[] x = Random(rng, 24);
        int[] dims = [2, 3, 4];
        double[] all = Ref("ReduceSum", x, dims, [true, true, true], false, out _);
        Assert.True(ScaledError(Run(Model("ReduceSum", 13, dims, null, false, []), ExecutionMode.Fast, null, 1, 1, (x, dims))[0].F!, all) < 1e-6);
        Assert.True(ScaledError(Run(Model("ReduceMax", 11, dims, null, true, [1, 1, 1]), ExecutionMode.Fast, null, 1, 1, (x, dims))[0].F!, [x.Max()]) < 1e-9);

        // Empty axes input with noop_with_empty_axes: the identity.
        byte[] noop = Model("ReduceSum", 13, dims, [], true, dims, noop: true);
        AssertSameBits(Run(noop, ExecutionMode.Fast, null, 1, 1, (x, dims))[0].F!, x, "noop");
        byte[] noopAll = Model("ReduceMean", 18, dims, [], true, [1, 1, 1], noop: false);
        Assert.True(ScaledError(Run(noopAll, ExecutionMode.Fast, null, 1, 1, (x, dims))[0].F!, [x.Average(v => (double)v)]) < 1e-6);
    }

    [Fact]
    public void NaNInfinityAndEmptyInputsFollowOnnxRuntime()
    {
        float nan = float.NaN, inf = float.PositiveInfinity;
        float[] x = [1, nan, 3, nan, nan, nan, 2, 5, 4];
        foreach (var path in Paths)
        {
            foreach (ExecutionMode mode in new[] { ExecutionMode.Fast, ExecutionMode.Exact })
            {
                // A scan from the first element that only a strictly better value replaces: NaN is ignored unless it comes first.
                float[] max = Run(Model("ReduceMax", 13, [3, 3], [1], false, [3]), mode, path, 1, 1, (x, [3, 3]))[0].F!;
                Assert.True(max[0] == 3f && float.IsNaN(max[1]) && max[2] == 5f, $"max {path} {mode}");
                float[] min = Run(Model("ReduceMin", 13, [3, 3], [0], false, [3]), mode, path, 1, 1, (x, [3, 3]))[0].F!;
                Assert.True(min[0] == 1f && float.IsNaN(min[1]) && min[2] == 3f, $"min {path} {mode}");
                float[] sum = Run(Model("ReduceSum", 11, [3, 3], [1], false, [3]), mode, path, 1, 1, (x, [3, 3]))[0].F!;
                Assert.True(float.IsNaN(sum[0]) && float.IsNaN(sum[1]) && sum[2] == 11f);
            }
        }

        float[] z = [];
        Assert.Equal([float.NegativeInfinity, float.NegativeInfinity], Run(Model("ReduceMax", 13, [2, 0], [1], false, [2]), ExecutionMode.Fast, null, 1, 1, (z, [2, 0]))[0].F!);
        Assert.Equal([0f, 0f], Run(Model("ReduceSum", 11, [2, 0], [1], false, [2]), ExecutionMode.Fast, null, 1, 1, (z, [2, 0]))[0].F!);
        Assert.Equal([0f, 0f], Run(Model("ReduceMean", 13, [2, 0], [1], false, [2]), ExecutionMode.Fast, null, 1, 1, (z, [2, 0]))[0].F!);
        Assert.Equal([1f, 1f], Run(Model("ReduceProd", 13, [2, 0], [1], false, [2]), ExecutionMode.Fast, null, 1, 1, (z, [2, 0]))[0].F!);
        Assert.Equal([inf, -inf], Run(Model("ReduceMax", 13, [2, 2], [1], false, [2]), ExecutionMode.Fast, null, 1, 1, ([inf, -inf, -inf, -inf], [2, 2]))[0].F!);
    }

    [Theory]
    [InlineData(new[] { 37, 129, 31 })]
    [InlineData(new[] { 5, 4096, 3 })]
    [InlineData(new[] { 3, 70000 })]
    public void ResultsDoNotDependOnThreadsOrPathAndFastEqualsExact(int[] dims)
    {
        var rng = new Random(dims[0]);
        float[] x = Random(rng, Count(dims), -1, 1);
        foreach (string op in new[] { "ReduceSum", "ReduceMean", "ReduceMax", "ReduceProd", "ReduceL2" })
        {
            for (int mask = 1; mask < 1 << dims.Length; mask++)
            {
                int[] axes = [.. Enumerable.Range(0, dims.Length).Where(i => (mask >> i & 1) != 0)];
                bool[] red = [.. Enumerable.Range(0, dims.Length).Select(i => (mask >> i & 1) != 0)];
                int[] outDims = [.. dims.Where((_, i) => !red[i])];
                byte[] model = Model(op, 18, dims, axes, false, outDims);
                float[] one = Run(model, ExecutionMode.Fast, null, 1, 1, (x, dims))[0].F!;
                AssertSameBits(Run(model, ExecutionMode.Fast, null, 8, 1, (x, dims))[0].F!, one, $"{op} mask {mask} threads");
                foreach (var path in Paths)
                {
                    foreach (int threads in new[] { 1, 5 })
                    {
                        float[] ex = Run(model, ExecutionMode.Exact, path, threads, 1, (x, dims))[0].F!;
                        AssertSameBits(ex, one, $"{op} mask {mask} exact {path} threads {threads}");
                    }
                }
            }
        }
    }

    [Fact]
    public void IntegerTypesReduceExactly()
    {
        // int64 input is built by hand: a data input of int64 through an initializer, reduced by ReduceSum/Max/Prod.
        foreach ((string op, long want) in new[] { ("ReduceSum", 10L), ("ReduceMax", 4L), ("ReduceMin", 1L), ("ReduceProd", 24L), ("ReduceMean", 2L), ("ReduceSumSquare", 30L), ("ReduceL1", 10L) })
        {
            byte[] model = new ModelBuilder(13).Initializer("x", new long[] { 1, 2, 3, 4 }, 4).Output("y", 7, 1)
                .Node(op, ["x"], ["y"], ("keepdims", 1L)).Build();
            using var ir = RecursiveAI.OneModel.Onnx.OnnxImporter.Load(model);
            using var s = RecursiveAI.OneModel.Graph.Session.Create(ir, new RecursiveAI.OneModel.Graph.SessionOptions(), RecursiveAI.OneModel.Graph.Runtime.OpRegistry.Default, null, default);
            RecursiveAI.OneModel.Tensors.Tensor[] outs = [default];
            s.Run([], [], [s.Outputs[0].Name], outs);
            Assert.Equal(want, outs[0].ToArray<long>()[0]);
            outs[0].Dispose();
        }
    }
}
