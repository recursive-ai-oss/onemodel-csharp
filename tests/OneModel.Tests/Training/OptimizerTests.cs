using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Tests.Tensors;
using RecursiveAI.OneModel.Training;

namespace RecursiveAI.OneModel.Tests.Training;

[Collection("autograd")]
public class OptimizerTests
{
    private static readonly OracleFile Oracle = OracleFile.Load("optim", "training");

    private static void Log(string line)
    {
        string? path = Environment.GetEnvironmentVariable("ONEMODEL_TRAINING_LOG");
        if (path is not null)
        {
            File.AppendAllText(path, line + "\n");
        }
    }

    private static double Scaled(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double max = 0, mag = 1e-12;
        for (int i = 0; i < a.Length; i++)
        {
            max = Math.Max(max, Math.Abs((double)a[i] - b[i]));
            mag = Math.Max(mag, Math.Abs(b[i]));
        }

        return max / mag;
    }

    private static Tensor Param(string key) => Oracle.Make(key, requiresGrad: true);

    private static void SetGrad(Tensor p, float[] g)
    {
        Backprop.ClearGrad(p);
        p.Autograd!.Grad = Tensor.FromSpan<float>(g, p.Shape);
    }

    public static IEnumerable<object[]> Configs() =>
    [
        ["sgd_plain", (Func<Tensor[], Optimizer>)(p => new Sgd(p, 0.1f))],
        ["sgd_momentum", (Func<Tensor[], Optimizer>)(p => new Sgd(p, 0.1f, momentum: 0.9f))],
        ["sgd_nesterov_wd", (Func<Tensor[], Optimizer>)(p => new Sgd(p, 0.1f, momentum: 0.9f, weightDecay: 1e-2f, nesterov: true))],
        ["sgd_dampening_wd", (Func<Tensor[], Optimizer>)(p => new Sgd(p, 0.05f, momentum: 0.8f, dampening: 0.5f, weightDecay: 1e-3f))],
        ["adam", (Func<Tensor[], Optimizer>)(p => new Adam(p, 1e-2f))],
        ["adam_l2", (Func<Tensor[], Optimizer>)(p => new Adam(p, 1e-2f, weightDecay: 0.1f))],
        ["adam_amsgrad", (Func<Tensor[], Optimizer>)(p => new Adam(p, 1e-2f, amsgrad: true))],
        ["adam_betas", (Func<Tensor[], Optimizer>)(p => new Adam(p, 3e-3f, beta1: 0.8f, beta2: 0.95f, epsilon: 1e-6f))],
        ["adamw", (Func<Tensor[], Optimizer>)(p => new AdamW(p, 1e-2f, weightDecay: 0.05f))],
        ["adamw_amsgrad", (Func<Tensor[], Optimizer>)(p => new AdamW(p, 1e-2f, weightDecay: 0.05f, amsgrad: true))],
    ];

    [Theory]
    [MemberData(nameof(Configs))]
    public void StepsMatchTorch(string name, Func<Tensor[], Optimizer> make)
    {
        Tensor p = Param("in.p"), q = Param("in.q");
        using Optimizer opt = make([p, q]);
        for (int k = 0; k < 5; k++)
        {
            SetGrad(p, Oracle.Values($"in.grad.p.{k}"));
            SetGrad(q, Oracle.Values($"in.grad.q.{k}"));
            opt.Step();
            if (k is 0 or 4)
            {
                double e = Math.Max(Scaled(p.Data<float>(), Oracle.Values($"out.{name}.p.{k + 1}")), Scaled(q.Data<float>(), Oracle.Values($"out.{name}.q.{k + 1}")));
                Log($"optimizer {name} after step {k + 1}: scaled error {e:E2}");
                Assert.True(e <= 1e-6, $"{name} step {k + 1}: scaled error {e:E2}");
            }
        }

        Assert.Equal(5, opt.StepCount);
    }

    [Fact]
    public void ParametersWithoutGradientAreSkipped()
    {
        Tensor p = Param("in.p"), q = Param("in.q");
        float[] before = q.ToArray<float>();
        using var opt = new Adam([p, q], 1e-2f);
        SetGrad(p, Oracle.Values("in.grad.p.0"));
        opt.Step();
        Assert.Equal(before, q.ToArray<float>());
        Assert.NotEqual(Oracle.Values("in.p"), p.ToArray<float>());
    }

    [Theory]
    [InlineData("norm_small", 0.5f, false)]
    [InlineData("norm_large", 100f, false)]
    [InlineData("norm_inf", 0.7f, true)]
    public void ClipNormMatchesTorch(string label, float maxNorm, bool inf)
    {
        Tensor p = Param("in.p"), q = Param("in.q");
        SetGrad(p, Oracle.Values("in.grad.p.0"));
        SetGrad(q, Oracle.Values("in.grad.q.0"));
        float total = GradClip.ClipNorm([p, q], maxNorm, inf);
        double e = Math.Max(Scaled(p.Grad.Data<float>(), Oracle.Values($"out.clip.{label}.p")), Scaled(q.Grad.Data<float>(), Oracle.Values($"out.clip.{label}.q")));
        double te = Math.Abs(total - Oracle.Values($"out.clip.{label}.total")[0]) / total;
        Log($"clip {label}: total norm relative error {te:E2}, gradient scaled error {e:E2}");
        Assert.True(e <= 1e-6 && te <= 1e-6, $"{label}: gradients {e:E2}, norm {te:E2}");
    }

    [Fact]
    public void ClipValueMatchesTorch()
    {
        Tensor p = Param("in.p"), q = Param("in.q");
        SetGrad(p, Oracle.Values("in.grad.p.0"));
        SetGrad(q, Oracle.Values("in.grad.q.0"));
        GradClip.ClipValue([p, q], 0.6f);
        Assert.Equal(Oracle.Values("out.clip.value.p"), p.Grad.ToArray<float>());
        Assert.Equal(Oracle.Values("out.clip.value.q"), q.Grad.ToArray<float>());
    }

    [Theory]
    [InlineData("step")]
    [InlineData("cosine")]
    [InlineData("warmup_cosine")]
    [InlineData("onecycle")]
    public void SchedulesMatchTorch(string name)
    {
        LrSchedule s = name switch
        {
            "step" => new StepLr(0.1f, 10, 0.5f),
            "cosine" => new CosineLr(0.1f, 100, 1e-3f),
            "warmup_cosine" => new WarmupLr(new CosineLr(0.1f, 90, 1e-3f), 10, 0.01f),
            _ => new OneCycleLr(0.1f, 100),
        };
        float[] want = Oracle.Values("out.lr." + name);
        double worst = 0;
        for (int i = 0; i < want.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(s.LrAt(i) - want[i]) / want[i]);
        }

        Log($"schedule {name}: max relative error over {want.Length} steps {worst:E2}");
        Assert.True(worst <= 1e-6, $"{name}: {worst:E2}");
    }

    [Fact]
    public void CosineSeriesIsAccurateAndSymmetric()
    {
        double worst = 0;
        for (int i = 0; i <= 10000; i++)
        {
            double t = i / 10000.0;
            worst = Math.Max(worst, Math.Abs(LrSchedule.Cos(t) - Math.Cos(Math.PI * t)));
        }

        Log($"LrSchedule.Cos max abs error vs Math.Cos on 10001 points: {worst:E2}");
        Assert.True(worst < 1e-15);
        Assert.Equal(1.0, LrSchedule.Cos(0));
        Assert.Equal(-1.0, LrSchedule.Cos(1));
        Assert.Equal(0.0, LrSchedule.Cos(0.5), 15);
    }

    [Fact]
    public void BadSettingsAreRejected()
    {
        Tensor p = Param("in.p");
        Assert.Throws<ArgumentException>(() => new Sgd([p], 0.1f, nesterov: true));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Sgd([p], float.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam([p], beta1: 1f));
        Assert.Throws<ArgumentException>(() => new Sgd([], 0.1f));
        Assert.Throws<ArgumentException>(() => new Sgd([Oracle.Make("in.p")], 0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => GradClip.ClipNorm([p], 0f));
    }
}
