using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Tensors;

// The lane loops of the eager element-wise operators (and their gradients) give the bits of the scalar loops, at every length and
// alignment, including NaN, infinity, signed zero and subnormal inputs.
[Collection("autograd")]
public class VectorElementwiseTests
{
    private static float[] Values(int n, int seed)
    {
        var rng = new Random(seed);
        float[] special = [0f, -0f, 1f, -1f, float.PositiveInfinity, float.NegativeInfinity, float.NaN, 1e-40f, -1e-40f, float.MaxValue, float.Epsilon, 3.5f, -2.25f];
        var v = new float[n];
        for (int i = 0; i < n; i++)
        {
            v[i] = rng.Next(5) == 0 ? special[rng.Next(special.Length)] : (float)((rng.NextDouble() * 8) - 4);
        }

        return v;
    }

    private static bool SameBits(float a, float b) => BitConverter.SingleToUInt32Bits(a) == BitConverter.SingleToUInt32Bits(b) || (float.IsNaN(a) && float.IsNaN(b));

    private static void AssertSame(ReadOnlySpan<float> a, ReadOnlySpan<float> b, string what)
    {
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
        {
            Assert.True(SameBits(a[i], b[i]), $"{what}[{i}]: {a[i]} ({BitConverter.SingleToUInt32Bits(a[i]):X8}) vs {b[i]} ({BitConverter.SingleToUInt32Bits(b[i]):X8})");
        }
    }

    [Theory]
    [InlineData(ArithmeticMode.Fast)]
    [InlineData(ArithmeticMode.Exact)]
    public void BinaryAndUnaryEqualTheScalarLoops(ArithmeticMode mode)
    {
        ArithmeticMode old = TensorCompute.Mode;
        try
        {
            TensorCompute.Mode = mode;
            for (int n = 1; n <= 150; n++)
            {
                using Tensor a = Tensor.FromSpan<float>(Values(n, n), Shape.Create(n));
                using Tensor b = Tensor.FromSpan<float>(Values(n, n + 1000), Shape.Create(n));
                using Tensor one = Tensor.FromSpan<float>([1.75f], Shape.Create(1));
                foreach (Func<Tensor, Tensor, ITensorKernels?, Tensor> op in new Func<Tensor, Tensor, ITensorKernels?, Tensor>[]
                {
                    (x, y, k) => Ops.AddWith(x, y, null, k), (x, y, k) => Ops.SubWith(x, y, null, k), (x, y, k) => Ops.MulWith(x, y, null, k), (x, y, k) => Ops.DivWith(x, y, null, k),
                })
                {
                    using Tensor got = op(a, b, null), want = op(a, b, ReferenceKernels.Instance);
                    AssertSame(got.Data<float>(), want.Data<float>(), $"{mode} n={n} binary");
                    using Tensor gotB = op(a, one, null), wantB = op(a, one, ReferenceKernels.Instance);   // broadcast scalar
                    AssertSame(gotB.Data<float>(), wantB.Data<float>(), $"{mode} n={n} broadcast");
                }

                using Tensor gotN = Ops.Neg(a), wantN = Ops.NegWith(a, null, ReferenceKernels.Instance);
                AssertSame(gotN.Data<float>(), wantN.Data<float>(), $"{mode} n={n} neg");
                using Tensor gotR = Ops.Relu(a), wantR = Ops.ReluWith(a, null, ReferenceKernels.Instance);
                AssertSame(gotR.Data<float>(), wantR.Data<float>(), $"{mode} n={n} relu");
            }

            // Strided operands: a transposed view and a column slice.
            using Tensor m = Tensor.FromSpan<float>(Values(7 * 37, 5), Shape.Create(7, 37));
            Tensor mt = m.T();
            using Tensor g1 = Ops.Add(mt, mt), w1 = Ops.AddWith(mt, mt, null, ReferenceKernels.Instance);
            AssertSame(g1.Data<float>(), w1.Data<float>(), "transposed");
            Tensor cols = m.Slice(1, 3, 30, 2);
            using Tensor g2 = Ops.Mul(cols, cols), w2 = Ops.MulWith(cols, cols, null, ReferenceKernels.Instance);
            AssertSame(g2.Data<float>(), w2.Data<float>(), "strided slice");
        }
        finally
        {
            TensorCompute.Mode = old;
        }
    }

    // Backward of an activation over n elements equals, element by element, the backward over a single element: the first n/lanes*lanes
    // go through lanes and the tail through the scalar twin, so any lane/scalar difference shows.
    [Theory]
    [InlineData(ArithmeticMode.Fast)]
    [InlineData(ArithmeticMode.Exact)]
    public void ActivationGradientsDoNotDependOnPosition(ArithmeticMode mode)
    {
        ArithmeticMode old = TensorCompute.Mode;
        try
        {
            TensorCompute.Mode = mode;
            const int N = 133;
            float[] xs = Values(N, 99).Select(v => float.IsFinite(v) ? Math.Clamp(v, -6f, 6f) : 0.5f).ToArray();
            float[] gs = Values(N, 98).Select(v => float.IsFinite(v) ? Math.Clamp(v, -3f, 3f) : 1f).ToArray();
            var ops = new (string Name, Func<Tensor, Tensor> F)[]
            {
                ("relu", t => Ops.Relu(t)), ("sigmoid", t => Ops.Sigmoid(t)), ("tanh", t => Ops.Tanh(t)), ("gelu", t => Ops.Gelu(t)), ("gelu-tanh", t => Ops.Gelu(t, true)),
            };
            foreach ((string name, Func<Tensor, Tensor> f) in ops)
            {
                float[] all = Run(f, xs, gs, out float[] yAll);
                for (int i = 0; i < N; i++)
                {
                    float[] one = Run(f, xs.AsSpan(i, 1).ToArray(), gs.AsSpan(i, 1).ToArray(), out float[] yOne);
                    Assert.True(SameBits(all[i], one[0]), $"{mode} {name} grad[{i}]: {all[i]} vs {one[0]}");
                    Assert.True(SameBits(yAll[i], yOne[0]), $"{mode} {name} y[{i}]: {yAll[i]} vs {yOne[0]}");
                }
            }
        }
        finally
        {
            TensorCompute.Mode = old;
        }

        static float[] Run(Func<Tensor, Tensor> f, float[] x, float[] g, out float[] y)
        {
            using var scope = new ActivationScope();
            Tensor t = Tensor.FromSpan<float>(x, Shape.Create(x.Length)).WithRequiresGrad();
            Tensor r = f(t);
            y = r.Data<float>().ToArray();
            using Tensor seed = Tensor.FromSpan<float>(g, Shape.Create(g.Length));
            r.Backward(seed);
            return t.Grad.Data<float>().ToArray();
        }
    }

    // erf by Simpson's rule on 2/sqrt(pi) exp(-t^2), accurate to about 1e-12 for |x| <= 5.
    private static double Erf(double x)
    {
        const int Steps = 4000;
        double h = x / Steps, sum = 1 + Math.Exp(-x * x);
        for (int i = 1; i < Steps; i++)
        {
            double t = i * h;
            sum += (i % 2 == 1 ? 4 : 2) * Math.Exp(-t * t);
        }

        return sum * h / 3 * 2 / Math.Sqrt(Math.PI);
    }

    [Fact]
    public void GeluGradientMatchesDoublePrecisionFormula()
    {
        const int N = 64;
        float[] xs = Values(N, 3).Select(v => float.IsFinite(v) ? Math.Clamp(v, -5f, 5f) : 0.25f).ToArray();
        foreach (bool tanhForm in new[] { false, true })
        {
            using var scope = new ActivationScope();
            Tensor t = Tensor.FromSpan<float>(xs, Shape.Create(N)).WithRequiresGrad();
            Tensor y = Ops.Gelu(t, tanhForm);
            using Tensor ones = Tensor.FromSpan<float>(Enumerable.Repeat(1f, N).ToArray(), Shape.Create(N));
            y.Backward(ones);
            for (int i = 0; i < N; i++)
            {
                double x = xs[i], d;
                if (tanhForm)
                {
                    double k = Math.Sqrt(2 / Math.PI), u = k * (x + (0.044715 * x * x * x)), th = Math.Tanh(u);
                    d = (0.5 * (1 + th)) + (0.5 * x * (1 - (th * th)) * k * (1 + (3 * 0.044715 * x * x)));
                }
                else
                {
                    d = (0.5 * (1 + Erf(x / Math.Sqrt(2)))) + (x * Math.Exp(-0.5 * x * x) / Math.Sqrt(2 * Math.PI));
                }

                Assert.True(Math.Abs(t.Grad.Data<float>()[i] - d) < 2e-5 * Math.Max(1, Math.Abs(d)), $"tanh={tanhForm} x={x}: {t.Grad.Data<float>()[i]} vs {d}");
            }
        }
    }
}
