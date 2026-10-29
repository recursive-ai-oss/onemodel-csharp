using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Graph;

/// <summary>The first operator set checked through the whole stack (model bytes, session, plan, kernel) against naive references.</summary>
public class OperatorTests
{
    private static (float[] Data, Shape Shape) Run(byte[] model, ExecutionMode mode, KernelPath? path, params (float[] Data, int[] Shape)[] inputs)
    {
        using IrModel ir = OnnxImporter.Load(model);
        using var s = Session.Create(ir, new SessionOptions { Mode = mode, IntraOpThreads = 1 }, OpRegistry.Default, path, default);
        var tensors = new Tensor[inputs.Length];
        var names = new string[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            tensors[i] = Tensor.FromSpan<float>(inputs[i].Data, Shape.Create(inputs[i].Shape));
            names[i] = s.Inputs[i].Name;
        }

        Span<Tensor> outs = [default];
        s.Run(names, tensors, [s.Outputs[0].Name], outs);
        float[] data = outs[0].ToArray<float>();
        Shape shape = outs[0].Shape;
        outs[0].Dispose();
        foreach (Tensor t in tensors)
        {
            t.Dispose();
        }

        return (data, shape);
    }

    private static float[] Random(Random rng, int n, bool special)
    {
        float[] r = new float[n];
        for (int i = 0; i < n; i++)
        {
            r[i] = (float)((rng.NextDouble() * 4) - 2);
        }

        if (special && n > 6)
        {
            r[rng.Next(n)] = float.NaN;
            r[rng.Next(n)] = float.PositiveInfinity;
            r[rng.Next(n)] = float.NegativeInfinity;
            r[rng.Next(n)] = -0f;
            r[rng.Next(n)] = BitConverter.UInt32BitsToSingle(1); // smallest subnormal
            r[rng.Next(n)] = float.MaxValue;
        }

        return r;
    }

    private static int Count(int[] s) => s.Aggregate(1, (x, y) => x * y);

    // Broadcast element (index into the output) to an operand of shape `s`.
    private static float[] NaiveBinary(Func<float, float, float> op, float[] a, int[] sa, float[] b, int[] sb, int[] so)
    {
        float[] r = new float[Count(so)];
        int[] idx = new int[so.Length];
        for (int o = 0; o < r.Length; o++)
        {
            int rem = o;
            for (int d = so.Length - 1; d >= 0; d--)
            {
                idx[d] = rem % so[d];
                rem /= so[d];
            }

            int Offset(int[] s)
            {
                int off = 0, acc = 1;
                for (int d = s.Length - 1; d >= 0; d--)
                {
                    int od = d + (so.Length - s.Length);
                    off += (s[d] == 1 ? 0 : idx[od]) * acc;
                    acc *= s[d];
                }

                return off;
            }

            r[o] = op(a[Offset(sa)], b[Offset(sb)]);
        }

        return r;
    }

    private static uint Bits(float f) => float.IsNaN(f) ? 0x7FC00000u : BitConverter.SingleToUInt32Bits(f);

    public static IEnumerable<object[]> BroadcastCases() =>
    [
        [new[] { 5 }, new[] { 5 }],
        [new[] { 3, 4 }, new[] { 4 }],
        [new[] { 3, 1 }, new[] { 1, 4 }],
        [new[] { 2, 3, 4 }, new[] { 3, 1 }],
        [new[] { 1 }, new[] { 2, 3, 5 }],
        [new[] { 2, 1, 4, 1 }, new[] { 3, 1, 5 }],
        [new[] { 17, 33 }, new[] { 17, 33 }],
        [new[] { 100 }, new[] { 1 }],
        [new[] { 0, 3 }, new[] { 3 }],
        [new[] { 4, 1, 0 }, new[] { 1, 6, 1 }],
        [new[] { 2, 3 }, Array.Empty<int>()],
    ];

    [Theory]
    [MemberData(nameof(BroadcastCases))]
    public void AddAndMulMatchTheNaiveReferenceAndAreBitExactInExactMode(int[] sa, int[] sb)
    {
        var rng = new Random(sa.Length * 31 + sb.Length);
        float[] a = Random(rng, Count(sa), true), b = Random(rng, Count(sb), true);
        Shape.TryBroadcast(Shape.Create(sa), Shape.Create(sb), out Shape so);
        int[] outDims = so.Dimensions.ToArray();
        foreach ((string op, Func<float, float, float> f) in new (string, Func<float, float, float>)[] { ("Add", (x, y) => x + y), ("Mul", (x, y) => x * y) })
        {
            byte[] model = new ModelBuilder().Input("a", 1, [.. sa.Cast<object>()]).Input("b", 1, [.. sb.Cast<object>()])
                .Output("y", 1, [.. outDims.Cast<object>()]).Node(op, ["a", "b"], ["y"]).Build();
            float[] want = NaiveBinary(f, a, sa, b, sb, outDims);

            (float[] fast, Shape fs) = Run(model, ExecutionMode.Fast, null, (a, sa), (b, sb));
            Assert.Equal(so, fs);
            for (int i = 0; i < want.Length; i++)
            {
                Assert.True(Bits(fast[i]) == Bits(want[i]) || (float.IsNaN(fast[i]) && float.IsNaN(want[i])), $"{op} fast element {i}");
            }

            foreach (KernelPath path in KernelPaths.All.Where(KernelPaths.IsSupported))
            {
                (float[] exact, _) = Run(model, ExecutionMode.Exact, path, (a, sa), (b, sb));
                for (int i = 0; i < want.Length; i++)
                {
                    Assert.True(Bits(exact[i]) == Bits(want[i]) && (!float.IsNaN(exact[i]) || BitConverter.SingleToUInt32Bits(exact[i]) == 0x7FC00000u),
                        $"{op} exact {path} element {i}: {BitConverter.SingleToUInt32Bits(exact[i]):x8} vs {Bits(want[i]):x8}");
                }
            }
        }
    }

    [Fact]
    public void ReluHandlesEveryLengthAndTheSpecialValues()
    {
        var rng = new Random(3);
        for (int n = 1; n <= 70; n++)
        {
            float[] x = Random(rng, n, true);
            byte[] model = new ModelBuilder().Input("x", 1, n).Output("y", 1, n).Node("Relu", ["x"], ["y"]).Build();
            foreach (KernelPath path in KernelPaths.All.Where(KernelPaths.IsSupported))
            {
                (float[] y, _) = Run(model, ExecutionMode.Exact, path, (x, [n]));
                for (int i = 0; i < n; i++)
                {
                    float want = float.IsNaN(x[i]) ? float.NaN : (x[i] > 0 ? x[i] : 0f);
                    Assert.True(Bits(y[i]) == Bits(want), $"n={n} {path} element {i}: {x[i]} -> {y[i]}");
                    if (float.IsNaN(y[i]))
                    {
                        Assert.Equal(0x7FC00000u, BitConverter.SingleToUInt32Bits(y[i]));
                    }
                }
            }
        }
    }

    private static float[] NaiveMatMul(float[] a, int[] sa, float[] b, int[] sb, int[] so)
    {
        int m = sa.Length == 1 ? 1 : sa[^2], k = sa[^1], n = sb.Length == 1 ? 1 : sb[^1];
        int[] ba = sa.Length > 2 ? sa[..^2] : [], bb = sb.Length > 2 ? sb[..^2] : [];
        int rank = Math.Max(ba.Length, bb.Length);
        int[] bo = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            int x = i - (rank - ba.Length) >= 0 ? ba[i - (rank - ba.Length)] : 1;
            int y = i - (rank - bb.Length) >= 0 ? bb[i - (rank - bb.Length)] : 1;
            bo[i] = Math.Max(x, y);
        }

        float[] r = new float[Count(so)];
        int batches = Count(bo);
        for (int bt = 0; bt < batches; bt++)
        {
            int rem = bt, offA = 0, offB = 0, accA = 1, accB = 1;
            int[] idx = new int[rank];
            for (int d = rank - 1; d >= 0; d--)
            {
                idx[d] = rem % bo[d];
                rem /= bo[d];
            }

            for (int d = rank - 1; d >= 0; d--)
            {
                int da = d - (rank - ba.Length), db = d - (rank - bb.Length);
                int ea = da >= 0 ? ba[da] : 1, eb = db >= 0 ? bb[db] : 1;
                offA += (ea == 1 ? 0 : idx[d]) * accA * m * k;
                offB += (eb == 1 ? 0 : idx[d]) * accB * k * n;
                accA *= ea;
                accB *= eb;
            }

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int p = 0; p < k; p++)
                    {
                        acc += (double)a[offA + (i * k) + p] * b[offB + (p * n) + j];
                    }

                    r[(bt * m * n) + (i * n) + j] = (float)acc;
                }
            }
        }

        return r;
    }

    public static IEnumerable<object[]> MatMulCases() =>
    [
        [new[] { 4, 5 }, new[] { 5, 3 }],
        [new[] { 7 }, new[] { 7, 3 }],
        [new[] { 4, 7 }, new[] { 7 }],
        [new[] { 7 }, new[] { 7 }],
        [new[] { 2, 3, 4 }, new[] { 4, 5 }],
        [new[] { 3, 4 }, new[] { 2, 4, 5 }],
        [new[] { 2, 1, 3, 4 }, new[] { 5, 4, 6 }],
        [new[] { 2, 3, 4 }, new[] { 1, 4, 2 }],
        [new[] { 130, 300 }, new[] { 300, 150 }],
        [new[] { 3, 0 }, new[] { 0, 4 }],
    ];

    [Theory]
    [MemberData(nameof(MatMulCases))]
    public void MatMulMatchesTheReferenceAndExactModeAgreesAcrossPaths(int[] sa, int[] sb)
    {
        var rng = new Random(11);
        float[] a = Random(rng, Count(sa), false), b = Random(rng, Count(sb), false);
        int[] batch = [];
        var lead = new List<int>();
        int ra = Math.Max(sa.Length - 2, 0), rb = Math.Max(sb.Length - 2, 0), r = Math.Max(ra, rb);
        for (int i = 0; i < r; i++)
        {
            int x = i - (r - ra) >= 0 ? sa[i - (r - ra)] : 1, y = i - (r - rb) >= 0 ? sb[i - (r - rb)] : 1;
            lead.Add(Math.Max(x, y));
        }

        batch = [.. lead];
        var so = new List<int>(batch);
        if (sa.Length > 1)
        {
            so.Add(sa[^2]);
        }

        if (sb.Length > 1)
        {
            so.Add(sb[^1]);
        }

        int[] outDims = [.. so];
        byte[] model = new ModelBuilder().Input("a", 1, [.. sa.Cast<object>()]).Input("b", 1, [.. sb.Cast<object>()])
            .Output("y", 1, [.. outDims.Cast<object>()]).Node("MatMul", ["a", "b"], ["y"]).Build();
        float[] want = NaiveMatMul(a, sa, b, sb, outDims);
        (float[] fast, Shape fs) = Run(model, ExecutionMode.Fast, null, (a, sa), (b, sb));
        Assert.Equal(Shape.Create(outDims), fs);
        Assert.True(SessionTests.ScaledError(fast, want) <= 1e-5);
        string? first = null;
        foreach (KernelPath path in KernelPaths.All.Where(KernelPaths.IsSupported))
        {
            (float[] exact, _) = Run(model, ExecutionMode.Exact, path, (a, sa), (b, sb));
            Assert.True(SessionTests.ScaledError(exact, want) <= 1e-5);
            first ??= SessionTests.Hash(exact);
            Assert.True(first == SessionTests.Hash(exact), $"exact {path} differs");
        }
    }

    public static IEnumerable<object[]> GemmCases() =>
    [
        [false, false, 1f, 1f, new[] { 6, 5 }],
        [true, false, 1f, 1f, new[] { 1, 5 }],
        [false, true, 0.5f, 2f, new[] { 6, 1 }],
        [true, true, 2f, 0.25f, new[] { 5 }],
        [false, true, 1f, 1f, new[] { 6, 5 }],
        [false, false, 1.5f, 1f, Array.Empty<int>()],
        [false, false, 1f, 1f, new[] { 1 }],
    ];

    [Theory]
    [MemberData(nameof(GemmCases))]
    public void GemmMatchesTheReference(bool transA, bool transB, float alpha, float beta, int[] cShape)
    {
        const int M = 6, K = 4, N = 5;
        var rng = new Random(5);
        int[] sa = transA ? [K, M] : [M, K], sb = transB ? [N, K] : [K, N];
        float[] a = Random(rng, M * K, false), b = Random(rng, K * N, false), c = Random(rng, Count(cShape), false);
        byte[] model = new ModelBuilder().Input("a", 1, [.. sa.Cast<object>()]).Input("b", 1, [.. sb.Cast<object>()])
            .Input("c", 1, [.. cShape.Cast<object>()]).Output("y", 1, M, N)
            .Node("Gemm", ["a", "b", "c"], ["y"], ("alpha", alpha), ("beta", beta), ("transA", transA ? 1L : 0L), ("transB", transB ? 1L : 0L)).Build();
        float[] want = new float[M * N];
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                double acc = 0;
                for (int p = 0; p < K; p++)
                {
                    double av = transA ? a[(p * M) + i] : a[(i * K) + p];
                    double bv = transB ? b[(j * K) + p] : b[(p * N) + j];
                    acc += av * bv;
                }

                int ci = cShape.Length == 2 ? ((cShape[0] == 1 ? 0 : i) * cShape[1]) + (cShape[1] == 1 ? 0 : j)
                    : cShape.Length == 1 ? (cShape[0] == 1 ? 0 : j) : 0;
                want[(i * N) + j] = (float)((alpha * acc) + (beta * c[ci]));
            }
        }

        string? first = null;
        (float[] fast, _) = Run(model, ExecutionMode.Fast, null, (a, sa), (b, sb), (c, cShape));
        Assert.True(SessionTests.ScaledError(fast, want) <= 1e-5);
        foreach (KernelPath path in KernelPaths.All.Where(KernelPaths.IsSupported))
        {
            (float[] exact, _) = Run(model, ExecutionMode.Exact, path, (a, sa), (b, sb), (c, cShape));
            Assert.True(SessionTests.ScaledError(exact, want) <= 1e-5);
            first ??= SessionTests.Hash(exact);
            Assert.True(first == SessionTests.Hash(exact), $"exact {path} differs");
        }
    }

    [Fact]
    public void IntegerAddWrapsAndBroadcasts()
    {
        byte[] model = new ModelBuilder().Input("a", 6, 2, 2).Input("b", 6, 2).Output("y", 6, 2, 2).Node("Add", ["a", "b"], ["y"]).Build();
        using var s = Session.Load(new MemoryStream(model));
        using Tensor a = Tensor.FromSpan<int>([int.MaxValue, 1, 2, 3], Shape.Create(2, 2));
        using Tensor b = Tensor.FromSpan<int>([1, 10], Shape.Create(2));
        Span<Tensor> outs = [default];
        s.Run(["a", "b"], [a, b], ["y"], outs);
        Assert.Equal([int.MinValue, 11, 3, 13], outs[0].ToArray<int>());
        outs[0].Dispose();
    }

    [Fact]
    public void ConstantNodesBecomeConstantsAndReshapeSupportsZeroAndAllowZero()
    {
        // Reshape target [0, -1] keeps the first extent and infers the second; the target comes from a Constant node.
        byte[] model = new ModelBuilder().Input("x", 1, 2, 3, 4).Output("y", 1, 2, 12)
            .Node("Constant", [], ["t"], ("value", ModelBuilder.Int64Tensor("t", [0, -1], 2)))
            .Node("Reshape", ["x", "t"], ["y"]).Build();
        float[] x = Random(new Random(1), 24, false);
        (float[] y, Shape s) = Run(model, ExecutionMode.Fast, null, (x, [2, 3, 4]));
        Assert.Equal(Shape.Create(2, 12), s);
        Assert.Equal(x, y);

        // With allowzero, a 0 in the target is a real zero-sized extent: [0, 24] cannot hold 24 elements... but [2, 0, 3] can hold none.
        byte[] model2 = new ModelBuilder(14).Input("x", 1, 0, 3).Output("y", 1, 2, 0, 3)
            .Node("Constant", [], ["t"], ("value", ModelBuilder.Int64Tensor("t", [2, 0, 3], 3)))
            .Node("Reshape", ["x", "t"], ["y"], ("allowzero", 1L)).Build();
        (float[] y2, Shape s2) = Run(model2, ExecutionMode.Fast, null, ([], [0, 3]));
        Assert.Equal(Shape.Create(2, 0, 3), s2);
        Assert.Empty(y2);
    }

    [Fact]
    public void ConstantAttributeForms()
    {
        // value_float scales the input; value_ints is the reshape target.
        byte[] model = new ModelBuilder(12).Input("x", 1, 2, 3).Output("y", 1, 3, 2)
            .Node("Constant", [], ["k"], ("value_float", 2.5f))
            .Node("Constant", [], ["t"], ("value_ints", new long[] { 3, 2 }))
            .Node("Mul", ["x", "k"], ["m"])
            .Node("Reshape", ["m", "t"], ["y"]).Build();
        (float[] y, Shape s) = Run(model, ExecutionMode.Fast, null, (new float[] { 1, 2, 3, 4, 5, 6 }, [2, 3]));
        Assert.Equal(Shape.Create(3, 2), s);
        Assert.Equal([2.5f, 5f, 7.5f, 10f, 12.5f, 15f], y);
    }

    [Theory]
    [InlineData(7)]
    [InlineData(9)]
    [InlineData(11)]
    [InlineData(13)]
    [InlineData(21)]
    public void TheFirstOperatorSetRunsAtEveryOpsetItSpans(int opset)
    {
        // Gemm with C (required before 11), Add, Mul and Relu (7+), MatMul (any), Reshape with a shape input (5+), Identity.
        byte[] model = new ModelBuilder(opset).Input("x", 1, 2, 3).Output("y", 1, 2, 2)
            .Initializer("w", [1f, 0, 0, 1, 1, 1], 3, 2).Initializer("c", [0.5f, -0.5f], 2).Initializer("s", [2L, 2L], 2)
            .Node("Gemm", ["x", "w", "c"], ["g"]).Node("Relu", ["g"], ["r"]).Node("MatMul", ["r", "id"], ["m"])
            .Node("Add", ["m", "c"], ["a"]).Node("Mul", ["a", "c"], ["p"]).Node("Reshape", ["p", "s"], ["q"]).Node("Identity", ["q"], ["y"])
            .Initializer("id", [1f, 0, 0, 1], 2, 2).Build();
        float[] x = [1, 2, 3, -4, -5, 6];
        // g = x*w + c: [4.5, 4.5], [2.5, 0.5]; relu keeps both rows; then + c and * c.
        float[] want = [(4.5f + 0.5f) * 0.5f, (4.5f - 0.5f) * -0.5f, (2.5f + 0.5f) * 0.5f, (0.5f - 0.5f) * -0.5f];
        (float[] y, Shape s) = Run(model, ExecutionMode.Exact, null, (x, [2, 3]));
        Assert.Equal(Shape.Create(2, 2), s);
        Assert.Equal(want, y);
    }
}
