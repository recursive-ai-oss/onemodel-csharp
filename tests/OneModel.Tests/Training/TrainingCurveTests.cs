using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Tests.Tensors;
using RecursiveAI.OneModel.Training;

namespace RecursiveAI.OneModel.Tests.Training;

[Collection("autograd")]
public class TrainingCurveTests
{
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

    // Runs the case, compares per-step losses and the final parameters with the PyTorch run, logs the numbers.
    private static void Compare(string name, TrainCase tc, double lossRelTol, double paramTol)
    {
        float[] want = tc.Oracle.Values("out.loss");
        var got = new float[tc.Steps];
        for (int i = 0; i < tc.Steps; i++)
        {
            got[i] = tc.Step(i);
        }

        double maxAbs = 0, maxRel = 0;
        for (int i = 0; i < got.Length; i++)
        {
            double d = Math.Abs((double)got[i] - want[i]);
            maxAbs = Math.Max(maxAbs, d);
            maxRel = Math.Max(maxRel, d / Math.Abs(want[i]));
        }

        // A parameter whose true gradient is zero (a bias in front of batch norm; the key bias of attention, which softmax ignores) gets
        // only rounding noise as its gradient, and Adam turns that noise into steps of the full learning rate, so it is reported
        // separately: it says nothing about the optimizer.
        double worstParam = 0, worstNoise = 0;
        string worstName = "";
        foreach ((string p, string k) in tc.Map)
        {
            double e = Scaled(tc.Param(p).Data<float>(), tc.Oracle.Values("out." + k));
            if (p.EndsWith("attn.k.bias", StringComparison.Ordinal) || (p is "0.bias" or "4.bias" && name.StartsWith("cnn", StringComparison.Ordinal)))
            {
                worstNoise = Math.Max(worstNoise, e);
            }
            else if (e > worstParam)
            {
                worstParam = e;
                worstName = p;
            }
        }

        Log($"{name}: steps {tc.Steps}, loss first {want[0]:F5} last {want[^1]:F5} (torch), max abs diff {maxAbs:E2}, max relative diff {maxRel:E2}, final parameter scaled error {worstParam:E2} (worst {worstName}), zero-gradient biases {worstNoise:E2}");
        Assert.True(maxRel <= lossRelTol, $"{name}: loss relative difference {maxRel:E2} > {lossRelTol:E2}");
        Assert.True(worstParam <= paramTol, $"{name}: parameter scaled error {worstParam:E2} > {paramTol:E2}");
        Assert.True(want[^1] < want[0], "the reference run should learn");
        Assert.True(got[^1] < got[0], "our run should learn");
    }

    [Fact]
    public void MlpSgdMomentumMatchesTorch() => Compare("mlp sgd+momentum", TrainCase.Mlp(OracleFile.Load("mlp", "training")), 1e-3, 1e-3);

    [Fact]
    public void ConvNetAdamWCosineClipMatchesTorch()
    {
        TrainCase tc = TrainCase.Cnn(OracleFile.Load("cnn", "training"));
        Compare("cnn adamw+cosine+clip", tc, 1e-3, 1e-3);
        float[] norms = tc.Oracle.Values("out.gradnorm");
        double worst = 0;
        int clipped = 0;
        for (int i = 0; i < norms.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(tc.GradNorms[i] - norms[i]) / norms[i]);
            clipped += norms[i] > 0.5f ? 1 : 0;
        }

        Log($"cnn: gradient norm max relative diff {worst:E2}; clipping active on {clipped} of {norms.Length} steps");
        Assert.True(worst < 1e-3, $"gradient norm relative difference {worst:E2}");
        Assert.True(clipped > 5, "clipping should actually engage in this run");
        // Batch-norm running statistics are state too.
        double rm = Scaled(tc.Param("1.running_mean").Data<float>(), tc.Oracle.Values("out.rm1"));
        double rv = Scaled(tc.Param("5.running_var").Data<float>(), tc.Oracle.Values("out.rv2"));
        Log($"cnn: batch-norm running statistics scaled error: mean {rm:E2}, variance {rv:E2}");
        // The first layer's running mean includes the conv bias, which is one of the zero-gradient noise parameters above; the
        // variance is not affected by a shift, so it is the one that is gated.
        Assert.True(rv < 1e-3, $"running variance {rv:E2}");
    }

    [Fact]
    public void TransformerAdamWarmupMatchesTorch() =>
        Compare("encoder adam+warmup", TrainCase.Encoder(OracleFile.Load("encoder", "training")), 1e-3, 1e-3);
}
