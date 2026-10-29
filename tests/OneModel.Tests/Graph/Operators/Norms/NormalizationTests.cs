using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using static RecursiveAI.OneModel.Tests.Graph.Operators.Norms.NormHarness;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.Norms;

public class NormalizationTests
{
    private static object[] Dims(int[] d) => [.. d.Cast<object>()];

    private static double[] LayerNormRef(float[] x, int rows, int cols, float[] scale, float[]? bias, float eps, bool rms, out double[] mean, out double[] inv)
    {
        double[] y = new double[x.Length];
        mean = new double[rows];
        inv = new double[rows];
        for (int r = 0; r < rows; r++)
        {
            double m = 0;
            if (!rms)
            {
                for (int j = 0; j < cols; j++)
                {
                    m += x[(r * cols) + j];
                }

                m /= cols;
            }

            double v = 0;
            for (int j = 0; j < cols; j++)
            {
                double d = x[(r * cols) + j] - m;
                v += d * d;
            }

            v /= cols;
            double iv = 1 / Math.Sqrt(v + eps);
            mean[r] = m;
            inv[r] = iv;
            for (int j = 0; j < cols; j++)
            {
                y[(r * cols) + j] = ((x[(r * cols) + j] - m) * iv * scale[j]) + (bias?[j] ?? 0);
            }
        }

        return y;
    }

    public static IEnumerable<object[]> LayerNormCases()
    {
        foreach (int cols in new[] { 1, 2, 3, 15, 16, 17, 31, 33, 64, 100, 257, 768 })
        {
            yield return [cols, true];
            yield return [cols, false];
        }
    }

    [Theory]
    [MemberData(nameof(LayerNormCases))]
    public void LayerNormalizationMatchesTheReferenceAndAllPathsAgreeBitForBit(int cols, bool withBias)
    {
        var rng = new Random(cols);
        int[] shape = [2, 3, cols];
        int rows = 6;
        float[] x = Random(rng, rows * cols, -3, 3), scale = Random(rng, cols, 0.5, 1.5), bias = Random(rng, cols);
        var mb = new ModelBuilder(17).Input("x", 1, Dims(shape)).Initializer("s", scale, cols);
        string[] ins = ["x", "s"];
        if (withBias)
        {
            mb.Initializer("b", bias, cols);
            ins = ["x", "s", "b"];
        }

        byte[] model = mb.Output("y", 1, Dims(shape)).Output("m", 1, 2, 3, 1).Output("i", 1, 2, 3, 1)
            .Node("LayerNormalization", ins, ["y", "m", "i"], ("epsilon", 1e-5f)).Build();
        double[] want = LayerNormRef(x, rows, cols, scale, withBias ? bias : null, 1e-5f, false, out double[] wantMean, out double[] wantInv);

        Out[] fast = Run(model, ExecutionMode.Fast, null, 1, 3, (x, shape));
        Assert.True(ScaledError(fast[0].F!, want) < 2e-5, $"cols {cols}: {ScaledError(fast[0].F!, want)}");
        Assert.True(ScaledError(fast[1].F!, wantMean) < 2e-5);
        Assert.True(ScaledError(fast[2].F!, wantInv) < 2e-5);
        Assert.Equal(Shape.Create(2, 3, 1), fast[1].Shape);

        foreach (var path in Paths)
        {
            foreach (int threads in new[] { 1, 4 })
            {
                Out[] ex = Run(model, ExecutionMode.Exact, path, threads, 3, (x, shape));
                for (int k = 0; k < 3; k++)
                {
                    AssertSameBits(ex[k].F!, fast[k].F!, $"LayerNormalization output {k} {path} threads {threads}");
                }
            }
        }
    }

    [Fact]
    public void LayerNormalizationAxisAndBroadcastScale()
    {
        var rng = new Random(5);
        int[] shape = [4, 3, 5];
        float[] x = Random(rng, 60, -3, 3);
        // axis 1: normalise over [3, 5]; scale [5] broadcasts across the 3 rows, bias [3, 5] is full.
        float[] scale = Random(rng, 5, 0.5, 1.5), bias = Random(rng, 15);
        byte[] model = new ModelBuilder(17).Input("x", 1, 4, 3, 5).Initializer("s", scale, 5).Initializer("b", bias, 3, 5)
            .Output("y", 1, 4, 3, 5).Node("LayerNormalization", ["x", "s", "b"], ["y"], ("axis", 1L)).Build();
        float[] sFull = new float[15];
        for (int i = 0; i < 15; i++)
        {
            sFull[i] = scale[i % 5];
        }

        double[] want = LayerNormRef(x, 4, 15, sFull, bias, 1e-5f, false, out _, out _);
        Out[] got = Run(model, ExecutionMode.Fast, null, 1, 1, (x, shape));
        Assert.True(ScaledError(got[0].F!, want) < 2e-5);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(16)]
    [InlineData(37)]
    [InlineData(200)]
    public void RmsNormalizationMatchesTheReference(int cols)
    {
        var rng = new Random(cols + 100);
        int[] shape = [3, cols];
        float[] x = Random(rng, 3 * cols, -3, 3), scale = Random(rng, cols, 0.5, 1.5);
        byte[] model = new ModelBuilder(23).Input("x", 1, 3, cols).Initializer("s", scale, cols).Output("y", 1, 3, cols)
            .Node("RMSNormalization", ["x", "s"], ["y"], ("epsilon", 1e-5f)).Build();
        double[] want = LayerNormRef(x, 3, cols, scale, null, 1e-5f, true, out _, out _);
        Out[] fast = Run(model, ExecutionMode.Fast, null, 1, 1, (x, shape));
        Assert.True(ScaledError(fast[0].F!, want) < 2e-5);
        foreach (var path in Paths)
        {
            AssertSameBits(Run(model, ExecutionMode.Exact, path, 3, 1, (x, shape))[0].F!, fast[0].F!, $"RMSNormalization {path}");
        }
    }

    [Theory]
    [InlineData(1, 3, 1)]
    [InlineData(2, 3, 20)]
    [InlineData(2, 4, 33)]
    [InlineData(3, 2, 1)]
    public void InstanceNormalizationMatchesTheReference(int n, int c, int spatial)
    {
        var rng = new Random(n + c + spatial);
        int[] shape = [n, c, spatial];
        float[] x = Random(rng, n * c * spatial, -3, 3), scale = Random(rng, c, 0.5, 1.5), bias = Random(rng, c);
        byte[] model = new ModelBuilder(6).Input("x", 1, n, c, spatial).Initializer("s", scale, c).Initializer("b", bias, c)
            .Output("y", 1, n, c, spatial).Node("InstanceNormalization", ["x", "s", "b"], ["y"], ("epsilon", 1e-3f)).Build();
        double[] want = new double[x.Length];
        for (int r = 0; r < n * c; r++)
        {
            float[] sc = Enumerable.Repeat(scale[r % c], spatial).ToArray(), bi = Enumerable.Repeat(bias[r % c], spatial).ToArray();
            double[] row = LayerNormRef(x[(r * spatial)..((r + 1) * spatial)], 1, spatial, sc, bi, 1e-3f, false, out _, out _);
            row.CopyTo(want, r * spatial);
        }

        Out[] fast = Run(model, ExecutionMode.Fast, null, 1, 1, (x, shape));
        Assert.True(ScaledError(fast[0].F!, want) < 2e-5);
        foreach (var path in Paths)
        {
            AssertSameBits(Run(model, ExecutionMode.Exact, path, 2, 1, (x, shape))[0].F!, fast[0].F!, $"InstanceNormalization {path}");
        }
    }

    [Theory]
    [InlineData(18, 2, 6, 5)]
    [InlineData(21, 2, 6, 5)]
    [InlineData(21, 3, 8, 17)]
    [InlineData(18, 1, 4, 1)]
    public void GroupNormalizationMatchesTheReference(int opset, int n, int c, int spatial)
    {
        const int groups = 2;
        var rng = new Random(opset + c);
        int[] shape = [n, c, spatial];
        int sdim = opset >= 21 ? c : groups;
        float[] x = Random(rng, n * c * spatial, -3, 3), scale = Random(rng, sdim, 0.5, 1.5), bias = Random(rng, sdim);
        byte[] model = new ModelBuilder(opset).Input("x", 1, n, c, spatial).Initializer("s", scale, sdim).Initializer("b", bias, sdim)
            .Output("y", 1, n, c, spatial).Node("GroupNormalization", ["x", "s", "b"], ["y"], ("epsilon", 1e-5f), ("num_groups", (long)groups)).Build();
        int cpg = c / groups, cols = cpg * spatial;
        double[] want = new double[x.Length];
        for (int r = 0; r < n * groups; r++)
        {
            int g = r % groups;
            float[] sc = new float[cols], bi = new float[cols];
            for (int j = 0; j < cols; j++)
            {
                int ch = (g * cpg) + (j / spatial);
                sc[j] = opset >= 21 ? scale[ch] : scale[g];
                bi[j] = opset >= 21 ? bias[ch] : bias[g];
            }

            LayerNormRef(x[(r * cols)..((r + 1) * cols)], 1, cols, sc, bi, 1e-5f, false, out _, out _).CopyTo(want, r * cols);
        }

        Out[] fast = Run(model, ExecutionMode.Fast, null, 1, 1, (x, shape));
        Assert.True(ScaledError(fast[0].F!, want) < 2e-5, $"{ScaledError(fast[0].F!, want)}");
        foreach (var path in Paths)
        {
            AssertSameBits(Run(model, ExecutionMode.Exact, path, 2, 1, (x, shape))[0].F!, fast[0].F!, $"GroupNormalization {path}");
        }
    }

    [Fact]
    public void SpecialValuesGiveCanonicalNaNInExactMode()
    {
        float[] x = [1f, float.NaN, 3f, 4f, float.PositiveInfinity, 1f, 2f, 3f];
        byte[] model = new ModelBuilder(17).Input("x", 1, 2, 4).Initializer("s", [1f, 1f, 1f, 1f], 4).Output("y", 1, 2, 4)
            .Node("LayerNormalization", ["x", "s"], ["y"]).Build();
        foreach (var path in Paths)
        {
            float[] y = Run(model, ExecutionMode.Exact, path, 1, 1, (x, [2, 4]))[0].F!;
            foreach (float v in y)
            {
                Assert.Equal(0x7FC00000u, Bits(v));
            }
        }
    }
}
