using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Tests.Tensors;
using RecursiveAI.OneModel.Training;

namespace RecursiveAI.OneModel.Tests.Training;

[Collection("autograd")]
public class TrainingDeterminismTests
{
    private static void Log(string line)
    {
        string? path = Environment.GetEnvironmentVariable("ONEMODEL_TRAINING_LOG");
        if (path is not null)
        {
            File.AppendAllText(path, line + "\n");
        }
    }

    private static string Run(Func<TrainCase> make, int steps)
    {
        TrainCase tc = make();
        var losses = new List<float>();
        for (int i = 0; i < steps; i++)
        {
            losses.Add(tc.Step(i));
        }

        return TrainCase.StateHash(tc.Model, tc.Opt) + ":" + BitConverter.SingleToInt32Bits(losses[^1]).ToString("x8", System.Globalization.CultureInfo.InvariantCulture);
    }

    private static void WithMode(ExecutionMode mode, int threads, Action body)
    {
        ExecutionMode m = TensorCompute.Mode;
        int t = TensorCompute.Threads;
        try
        {
            TensorCompute.Mode = mode;
            TensorCompute.Threads = threads;
            body();
        }
        finally
        {
            TensorCompute.Mode = m;
            TensorCompute.Threads = t;
        }
    }

    private static string Cnn(int steps) => Run(() => TrainCase.Cnn(OracleFile.Load("cnn", "training")), steps);

    private static string Encoder(int steps) => Run(() => TrainCase.Encoder(OracleFile.Load("encoder", "training"), dropout: 0.1f), steps);

    [Fact]
    public void FastTrainingIsIndependentOfThreadCount()
    {
        string c1 = "", e1 = "";
        WithMode(ExecutionMode.Fast, 1, () => { c1 = Cnn(20); e1 = Encoder(20); });
        foreach (int threads in new[] { 3, 8 })
        {
            WithMode(ExecutionMode.Fast, threads, () =>
            {
                Assert.Equal(c1, Cnn(20));
                Assert.Equal(e1, Encoder(20));
            });
        }

        Log($"fast training hashes (1, 3, 8 threads equal): cnn {c1[..16]}, encoder {e1[..16]}");
    }

    // Pinned: the exact-mode state after 20 steps of the conv net (AdamW, cosine, clipping) is the same on every ISA path (run the
    // suite with DOTNET_EnableAVX512=0, DOTNET_EnableAVX2=0 and DOTNET_EnableHWIntrinsic=0), thread count and OS.
    private const string ExactCnn20 = "C5BA78DCD68109F63B558E9365B5ED2329484CF459195DBAF0C4E8D1E65AFBC1:3f9852db";
    private const string ExactEncoder20 = "1913F5D041B46AD27D1D662A9890DF98894BF62B23477BEABE37823E303E46B9:3f1412f4";

    [Fact]
    public void ExactTrainingHashesArePinnedAndIndependentOfThreads()
    {
        string c = "", e = "";
        WithMode(ExecutionMode.Exact, 1, () => { c = Cnn(20); e = Encoder(20); });
        Log($"exact training hashes: cnn {c} encoder {e}");
        WithMode(ExecutionMode.Exact, 8, () =>
        {
            Assert.Equal(c, Cnn(20));
            Assert.Equal(e, Encoder(20));
        });
        Assert.Equal(ExactCnn20, c);
        Assert.Equal(ExactEncoder20, e);
    }
}
