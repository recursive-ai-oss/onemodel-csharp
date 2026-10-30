using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Tests.Kernels;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.Elementwise;

public class ElementwiseKernelTests
{
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

    [Fact]
    public void LargeBroadcastOutputsDoNotDependOnTheThreadCount()
    {
        var rng = new KernelTestData.Rng(99);
        (int[] A, int[] B)[] cases =
        [
            ([300, 1000], [300, 1000]),
            ([300, 1000], [1000]),
            ([1000, 300], [1000, 1]),
            ([1, 100000], [1]),
            ([70000], [70000]),
            ([2, 3, 20000], [3, 1]),
            ([5000, 40], [40]),
        ];
        foreach (bool exact in new[] { false, true })
        {
            foreach ((int[] da, int[] db) in cases)
            {
                Shape sa = Shape.Create(da), sb = Shape.Create(db);
                Assert.True(Shape.TryBroadcast(sa, sb, out Shape so));
                var a = new float[sa.Length];
                var b = new float[sb.Length];
                KernelTestData.Fill(ref rng, a, true);
                KernelTestData.Fill(ref rng, b, true);
                var one = new float[so.Length];
                var many = new float[so.Length];
                Broadcast.Binary<float, FloatRows<AddArith>>(KernelPaths.Best(), exact, so, sa, sb, a, b, one, 1);
                Broadcast.Binary<float, FloatRows<AddArith>>(KernelPaths.Best(), exact, so, sa, sb, a, b, many, 7);
                Assert.True(KernelTestData.SameBits(one, many), $"{sa} + {sb} exact={exact}");
                // Spot-check against the plain loop.
                for (int i = 0; i < so.Length; i += 997)
                {
                    int ra = 0, rb = 0, rest = i, strideA = 1, strideB = 1;
                    for (int d = so.Rank - 1; d >= 0; d--)
                    {
                        int idx = rest % so[d];
                        rest /= so[d];
                        int da2 = d - (so.Rank - sa.Rank), db2 = d - (so.Rank - sb.Rank);
                        if (da2 >= 0)
                        {
                            ra += sa[da2] == 1 ? 0 : idx * strideA;
                            strideA *= sa[da2];
                        }

                        if (db2 >= 0)
                        {
                            rb += sb[db2] == 1 ? 0 : idx * strideB;
                            strideB *= sb[db2];
                        }
                    }

                    float want = a[ra] + b[rb];
                    Assert.True(float.IsNaN(want) ? float.IsNaN(many[i]) : BitConverter.SingleToUInt32Bits(want) == BitConverter.SingleToUInt32Bits(many[i]), $"{sa} + {sb} at {i}");
                }
            }
        }
    }

    [Fact]
    public void LargeReluDoesNotDependOnTheThreadCountAndWorksInPlace()
    {
        var rng = new KernelTestData.Rng(5);
        var x = new float[200003];
        KernelTestData.Fill(ref rng, x, true);
        foreach (bool exact in new[] { false, true })
        {
            var one = new float[x.Length];
            var many = new float[x.Length];
            ReluParallel.Run(KernelPaths.Best(), exact, x, one, 1);
            ReluParallel.Run(KernelPaths.Best(), exact, x, many, 7);
            Assert.True(KernelTestData.SameBits(one, many));
            var inPlace = (float[])x.Clone();
            ReluParallel.Run(KernelPaths.Best(), exact, inPlace, inPlace, 7);
            Assert.True(KernelTestData.SameBits(one, inPlace));
        }
    }

    /// <summary>One graph through Exp, Sub, Greater, Where, Cast, Max, Neg, Tanh, Gelu and Mean.</summary>
    private static byte[] Chain() => new ModelBuilder(20)
        .Input("x", 1, "n")
        .Output("y", 1, "n")
        .Node("Exp", ["x"], ["e"])
        .Node("Sub", ["e", "x"], ["s"])
        .Node("Greater", ["s", "x"], ["c"])
        .Node("Where", ["c", "e", "x"], ["w"])
        .Node("Cast", ["c"], ["k"], ("to", 1L))
        .Node("Max", ["w", "k", "s"], ["m"])
        .Node("Neg", ["m"], ["ng"])
        .Node("Tanh", ["ng"], ["t"])
        .Node("Gelu", ["t"], ["g"])
        .Node("Mean", ["g", "w"], ["y"])
        .Build();

    [Theory]
    [InlineData(ExecutionMode.Fast)]
    [InlineData(ExecutionMode.Exact)]
    public void SteadyStateRunAllocatesNothingAndIsRepeatable(ExecutionMode mode)
    {
        using IrModel ir = OnnxImporter.Load(Chain());
        using var s = Session.Create(ir, new SessionOptions { Mode = mode, IntraOpThreads = 1 });
        float[] x = new float[1003];
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = ((i * 37) % 101 / 20f) - 2.5f;
        }

        float[] y = new float[x.Length], first = new float[x.Length];
        s.Run("x", x.AsSpan(), [x.Length], "y", first.AsSpan());
        for (int i = 0; i < 20; i++)
        {
            s.Run("x", x.AsSpan(), [x.Length], "y", y.AsSpan());
        }

        Assert.True(first.AsSpan().SequenceEqual(y));
        Assert.Equal(0, AllocatedBytes(() =>
        {
            for (int i = 0; i < 300; i++)
            {
                s.Run("x", x.AsSpan(), [x.Length], "y", y.AsSpan());
            }
        }));
    }

    [Fact]
    public void CastRoundsAndSaturatesAsDocumented()
    {
        // float16 and bfloat16 round to nearest even; float to integer truncates and saturates, NaN gives 0; integers wrap.
        using Tensor f = Tensor.FromSpan<float>([1.00048828125f, 1.0009765625f, 70000f, float.NaN, -3.9f, 3e9f], Shape.Create(6));
        using Tensor h = Tensor.Create(DType.F16, Shape.Create(6));
        CastKernel.Run(f, h);
        Half[] hh = h.ToArray<Half>();
        Assert.Equal((Half)1.0f, hh[0]);
        Assert.Equal((Half)1.0009765625f, hh[1]);
        Assert.Equal(Half.PositiveInfinity, hh[2]);

        using Tensor i32 = Tensor.Create(DType.I32, Shape.Create(6));
        CastKernel.Run(f, i32);
        Assert.Equal([1, 1, 70000, 0, -3, int.MaxValue], i32.ToArray<int>());

        using Tensor wide = Tensor.FromSpan<int>([300, -1, 128, 255], Shape.Create(4));
        using Tensor u8 = Tensor.Create(DType.U8, Shape.Create(4));
        CastKernel.Run(wide, u8);
        Assert.Equal(new byte[] { 44, 255, 128, 255 }, u8.ToArray<byte>());

        using Tensor bf = Tensor.Create(DType.BF16, Shape.Create(4));
        using Tensor ff = Tensor.FromSpan<float>([1f, 1.00390625f, 1.01171875f, -2.5f], Shape.Create(4));
        CastKernel.Run(ff, bf);
        using Tensor back = Tensor.Create(DType.F32, Shape.Create(4));
        CastKernel.Run(bf, back);
        // 1.00390625 is halfway between 1 and 1.0078125 in bfloat16: ties go to the even mantissa.
        Assert.Equal([1f, 1f, 1.015625f, -2.5f], back.ToArray<float>());
    }

    [Fact]
    public void IntegerDivisionAndModuloNeverThrow()
    {
        Assert.Equal(0, DivArith.Apply(5, 0));
        Assert.Equal(int.MinValue, DivArith.Apply(int.MinValue, -1));
        Assert.Equal(0, FmodArith.Apply(int.MinValue, -1));
        Assert.Equal(long.MinValue, DivArith.Apply(long.MinValue, -1L));
        Assert.Equal(2, FloorModArith.Apply(-7, 3));
        Assert.Equal(-1, FloorModArith.Apply(7, -4));
        Assert.Equal(-1, FloorModArith.Apply(-7, -3));
        Assert.Equal(-2, FloorModArith.Apply(7, -9));
    }

    [Theory]
    [InlineData(ExecutionMode.Fast)]
    [InlineData(ExecutionMode.Exact)]
    public void ExpandedGeluGraphFromTheSpecificationRuns(ExecutionMode mode)
    {
        // The shape of the Gelu function body in the ONNX backend tests: constants, CastLike, Div, Erf, Sum, Mul.
        byte[] model = new ModelBuilder(20)
            .Input("x", 1, 3)
            .Output("y", 1, 3)
            .Node("Constant", [], ["half"], ("value", ModelBuilder.FloatTensor("", [0.5f])))
            .Node("CastLike", ["half", "x"], ["halfc"])
            .Node("Constant", [], ["one"], ("value", ModelBuilder.FloatTensor("", [1f])))
            .Node("CastLike", ["one", "x"], ["onec"])
            .Node("Constant", [], ["s2"], ("value", ModelBuilder.FloatTensor("", [1.4142135f])))
            .Node("CastLike", ["s2", "x"], ["s2c"])
            .Node("Div", ["x", "s2c"], ["xs"])
            .Node("Erf", ["xs"], ["e"])
            .Node("Sum", ["onec", "e"], ["phi"])
            .Node("Mul", ["halfc", "x"], ["mx"])
            .Node("Mul", ["mx", "phi"], ["y"])
            .Build();
        using IrModel ir = OnnxImporter.Load(model);
        using var s = Session.Create(ir, new SessionOptions { Mode = mode, IntraOpThreads = 1 });
        float[] y = new float[3];
        s.Run("x", new float[] { -1f, 0f, 1f }.AsSpan(), [3], "y", y.AsSpan());
        Assert.Equal(-0.15865526f, y[0], 1e-6f);
        Assert.Equal(0f, y[1]);
        Assert.Equal(0.84134474f, y[2], 1e-6f);
    }
}
