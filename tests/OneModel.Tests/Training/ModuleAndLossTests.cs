using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Training;

namespace RecursiveAI.OneModel.Tests.Training;

[Collection("autograd")]
public class ModuleAndLossTests
{
    private static Tensor T(Shape s, params float[] v) => Tensor.FromSpan<float>(v, s);

    [Fact]
    public void StateDictOrderAndNamesAreStable()
    {
        var g = new Generator(1);
        using var layer = new TransformerEncoderLayer(8, 2, 16, g, dropout: 0.1f);
        string[] names = layer.StateDict().Select(p => p.Name).ToArray();
        string[] expected =
        [
            "attn.q.weight", "attn.q.bias", "attn.k.weight", "attn.k.bias", "attn.v.weight", "attn.v.bias", "attn.o.weight", "attn.o.bias",
            "norm1.weight", "norm1.bias", "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias", "norm2.weight", "norm2.bias",
            "drop1.draws", "drop2.draws",
        ];
        Assert.Equal(expected, names);
        Assert.Equal(16, layer.Parameters().Length);
        Assert.Equal(2, layer.NamedBuffers().Count);
        Assert.All(layer.Parameters(), p => Assert.True(p.RequiresGrad));

        using var bn = new Sequential(new Conv2d(1, 2, 3, g), new BatchNorm2d(2));
        Assert.Equal(["0.weight", "0.bias", "1.weight", "1.bias", "1.running_mean", "1.running_var"], bn.StateDict().Select(p => p.Name).ToArray());
    }

    [Fact]
    public void TrainAndEvalReachEveryChild()
    {
        var g = new Generator(1);
        using var seq = new Sequential(new Linear(2, 2, g), new Sequential(new Dropout(0.5f, 1), new BatchNorm2d(1)));
        Assert.True(seq.IsTraining);
        seq.Eval();
        Assert.All(seq.Children, c => Assert.False(c.Child.IsTraining));
        Assert.False(((NnModule)seq.Children[1].Child.Children[0].Child).IsTraining);
        seq.Train();
        Assert.True(seq.Children[1].Child.Children[1].Child.IsTraining);
    }

    [Fact]
    public void BatchNormUsesRunningStatisticsInEvalMode()
    {
        using var bn = new BatchNorm2d(2);
        var rng = new Generator(4);
        float[] data = Enumerable.Range(0, 2 * 2 * 3 * 3).Select(_ => rng.Uniform(-2, 3)).ToArray();
        using var scope = new ActivationScope();
        Tensor x = T(Shape.Create(2, 2, 3, 3), data);
        ActivationScope.Track(x);
        Tensor y = bn.Forward(x);
        Assert.NotEqual(0f, bn.RunningMean.Data<float>()[0]);
        float[] trainOut = y.ToArray<float>();
        bn.Eval();
        float[] evalOut = bn.Forward(x).ToArray<float>();
        Assert.NotEqual(trainOut, evalOut);   // 0.1 momentum: running statistics are far from the batch ones after one step
    }

    [Fact]
    public void DropoutIsSeededDeterministicAndCheckpointable()
    {
        using var scope = new ActivationScope();
        Tensor x = Tensor.Create(DType.F32, Shape.Create(1000));
        x.Fill(1f);
        ActivationScope.Track(x);
        using var a = new Dropout(0.3f, 42);
        using var b = new Dropout(0.3f, 42);
        float[] a1 = a.Forward(x).ToArray<float>(), b1 = b.Forward(x).ToArray<float>();
        Assert.Equal(a1, b1);
        float[] a2 = a.Forward(x).ToArray<float>();
        Assert.NotEqual(a1, a2);
        Assert.Equal(2, a.Draws);
        int zeros = a1.Count(v => v == 0f);
        Assert.InRange(zeros, 240, 360);
        Assert.All(a1.Where(v => v != 0f), v => Assert.Equal(1f / 0.7f, v));

        // Restoring the draw counter restores the stream.
        b.StateDict()[0].Tensor.MutableData<long>()[0] = 1;
        Assert.Equal(a2, b.Forward(x).ToArray<float>());

        a.Eval();
        Assert.Equal(x.ToArray<float>(), a.Forward(x).ToArray<float>());
    }

    [Fact]
    public void DropoutBackwardUsesTheMask()
    {
        using var scope = new ActivationScope();
        Tensor x = T(Shape.Create(64), Enumerable.Repeat(2f, 64).ToArray()).WithRequiresGrad();
        using var d = new Dropout(0.5f, 7);
        Tensor y = d.Forward(x);
        Ops.Sum(y).Backward();
        ReadOnlySpan<float> yv = y.Data<float>(), g = x.Grad.Data<float>();
        for (int i = 0; i < 64; i++)
        {
            Assert.Equal(yv[i] == 0f ? 0f : 2f, g[i]);
        }
    }

    [Fact]
    public void LoadStateDictCopiesInPlaceAndRejectsMismatch()
    {
        var g = new Generator(1);
        using var a = new Linear(3, 2, g);
        using var b = new Linear(3, 2, new Generator(2));
        Tensor w = b.Weight;
        b.LoadStateDict(a.StateDict());
        Assert.Equal(a.Weight.ToArray<float>(), b.Weight.ToArray<float>());
        Assert.True(w.Buffer == b.Weight.Buffer);
        using var c = new Linear(3, 4, g);
        Assert.Throws<ArgumentException>(() => b.LoadStateDict(c.StateDict()));
        Assert.Throws<ArgumentException>(() => b.LoadStateDict(a.StateDict().Take(1)));
    }

    [Fact]
    public void InitialisationIsSeededAndBounded()
    {
        using var a = new Linear(16, 8, new Generator(9));
        using var b = new Linear(16, 8, new Generator(9));
        Assert.Equal(a.Weight.ToArray<float>(), b.Weight.ToArray<float>());
        Assert.All(a.Weight.ToArray<float>(), v => Assert.InRange(v, -0.25f, 0.25f));
    }

    [Fact]
    public void TransformerLayerAndAttentionTrainEndToEnd()
    {
        var g = new Generator(3);
        using var layer = new TransformerEncoderLayer(8, 2, 16, g);
        using var scope = new ActivationScope();
        Tensor x = Tensor.Create(DType.F32, Shape.Create(2, 5, 8));
        var rng = new Generator(8);
        foreach (ref float v in x.MutableData<float>())
        {
            v = rng.Uniform(-1, 1);
        }

        ActivationScope.Track(x);
        Tensor y = layer.Forward(x);
        Assert.Equal(Shape.Create(2, 5, 8), y.Shape);
        Ops.Sum(Ops.Mul(y, y)).Backward();
        Assert.All(layer.Parameters(), p => Assert.False(p.Grad.IsEmpty));
    }

    // ---------------------------------------------------------------------------------------------------------------- losses

    private static double LogSumExp(ReadOnlySpan<float> row)
    {
        double m = row.ToArray().Max();
        return m + Math.Log(row.ToArray().Sum(v => Math.Exp(v - m)));
    }

    private static double ReferenceCrossEntropy(float[] logits, int rows, int cols, long[] targets, double eps, long ignore, bool mean)
    {
        double total = 0, smooth = 0;
        int counted = 0;
        for (int r = 0; r < rows; r++)
        {
            if (targets[r] == ignore)
            {
                continue;
            }

            counted++;
            ReadOnlySpan<float> row = logits.AsSpan(r * cols, cols);
            double lse = LogSumExp(row);
            total += lse - row[(int)targets[r]];
            double s = 0;
            for (int c = 0; c < cols; c++)
            {
                s += lse - row[c];
            }

            smooth += s / cols;
        }

        double sum = (1 - eps) * total + eps * smooth;
        return mean ? sum / counted : sum;
    }

    [Theory]
    [InlineData(0.0, true)]
    [InlineData(0.1, true)]
    [InlineData(0.25, false)]
    public void CrossEntropyMatchesReferenceWithSmoothingAndIgnore(double eps, bool mean)
    {
        var rng = new Generator(11);
        float[] logits = Enumerable.Range(0, 6 * 5).Select(_ => rng.Uniform(-3, 3)).ToArray();
        long[] targets = [1, 4, -100, 0, 2, -100];
        using var scope = new ActivationScope();
        Tensor lt = T(Shape.Create(6, 5), logits).WithRequiresGrad();
        Tensor tt = Tensor.FromSpan<long>(targets, Shape.Create(6));
        ActivationScope.Track(tt);
        float got = Losses.CrossEntropy(lt, tt, (float)eps, mean: mean).Data<float>()[0];
        double want = ReferenceCrossEntropy(logits, 6, 5, targets, (float)eps, -100, mean);
        Assert.True(Math.Abs(got - want) <= 2e-6 * Math.Abs(want), $"{got} vs {want}");
    }

    [Fact]
    public void CrossEntropyGradientIsSoftmaxMinusSmoothedTarget()
    {
        var rng = new Generator(12);
        float[] logits = Enumerable.Range(0, 4 * 3).Select(_ => rng.Uniform(-2, 2)).ToArray();
        long[] targets = [0, 2, 1, -100];
        const float eps = 0.2f;
        using var scope = new ActivationScope();
        Tensor lt = T(Shape.Create(4, 3), logits).WithRequiresGrad();
        Tensor tt = Tensor.FromSpan<long>(targets, Shape.Create(4));
        ActivationScope.Track(tt);
        Losses.CrossEntropy(lt, tt, eps).Backward();
        ReadOnlySpan<float> g = lt.Grad.Data<float>();
        for (int r = 0; r < 4; r++)
        {
            double lse = LogSumExp(logits.AsSpan(r * 3, 3));
            for (int c = 0; c < 3; c++)
            {
                double p = Math.Exp(logits[r * 3 + c] - lse);
                double want = targets[r] == -100 ? 0 : (p - ((1 - eps) * (targets[r] == c ? 1 : 0) + eps / 3)) / 3;
                Assert.True(Math.Abs(g[r * 3 + c] - want) < 1e-6, $"row {r} col {c}: {g[r * 3 + c]} vs {want}");
            }
        }
    }

    [Fact]
    public void MseL1AndBceMatchReference()
    {
        float[] x = [-120f, -3f, -0.5f, 0f, 0.7f, 4f, 90f, 2f];
        float[] t = [0f, 1f, 0.25f, 0.5f, 1f, 0f, 1f, 0.3f];
        using var scope = new ActivationScope();
        Tensor xt = T(Shape.Create(2, 4), x).WithRequiresGrad(), tt = T(Shape.Create(2, 4), t);
        double mse = x.Zip(t, (a, b) => (double)(a - b) * (a - b)).Average();
        double l1 = x.Zip(t, (a, b) => Math.Abs((double)(a - b))).Average();
        double bce = x.Zip(t, (a, b) => Math.Max(a, 0) - (double)a * b + Math.Log(1 + Math.Exp(-Math.Abs((double)a)))).Average();
        Assert.Equal(mse, Losses.Mse(xt, tt).Data<float>()[0], 1e-3);
        Assert.Equal(l1, Losses.L1(xt, tt).Data<float>()[0], 1e-4);
        Tensor b = Losses.BceWithLogits(xt, tt);
        Assert.Equal(bce, b.Data<float>()[0], 1e-4);
        Assert.Equal(bce * 8, Losses.BceWithLogits(xt, tt, mean: false).Data<float>()[0], 1e-3);
        b.Backward();
        for (int i = 0; i < 8; i++)
        {
            double want = (1 / (1 + Math.Exp(-(double)x[i])) - t[i]) / 8;
            Assert.True(Math.Abs(xt.Grad.Data<float>()[i] - want) < 1e-6, $"BCE gradient {i}");
        }
    }

    [Fact]
    public void LossGradientsPassFiniteDifferenceCheck()
    {
        var rng = new Generator(21);
        Tensor Rand(params int[] dims)
        {
            Tensor t = Tensor.Create(DType.F32, Shape.Create(dims));
            foreach (ref float v in t.MutableData<float>())
            {
                v = rng.Uniform(-1.5f, 1.5f);
            }

            return t.WithRequiresGrad();
        }

        Tensor targets = Tensor.FromSpan<long>([2, 0, 1, 2], Shape.Create(4));
        Tensor probs = Tensor.FromSpan<float>([0.2f, 0.9f, 0.5f, 0.1f, 1f, 0f], Shape.Create(2, 3));
        GradientCheckResult[] r1 = GradientCheck.Run(t => Losses.CrossEntropy(t[0], targets, 0.1f), [Rand(4, 3)]);
        GradientCheckResult[] r2 = GradientCheck.Run(t => Losses.BceWithLogits(t[0], probs), [Rand(2, 3)]);
        GradientCheckResult[] r3 = GradientCheck.Run(t => Losses.Mse(t[0], t[1]), [Rand(3, 2), Rand(3, 2)]);
        GradientCheckResult[] r4 = GradientCheck.Run(t => Losses.L1(t[0], t[1]), [Rand(3, 2), Rand(3, 2)]);
        Assert.All(r1.Concat(r2).Concat(r3).Concat(r4), r => Assert.True(r.ScaledError < 5e-3, $"scaled error {r.ScaledError}"));
    }

    [Fact]
    public void LossArgumentsAreChecked()
    {
        using var scope = new ActivationScope();
        Tensor l = T(Shape.Create(2, 3), 1, 2, 3, 4, 5, 6);
        Tensor t = Tensor.FromSpan<long>([0, 1], Shape.Create(2));
        Assert.Throws<ArgumentOutOfRangeException>(() => Losses.CrossEntropy(l, t, 1.5f));
        Assert.Throws<TensorShapeException>(() => Losses.BceWithLogits(l, T(Shape.Create(3), 1, 2, 3)));
    }
}
