using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Tests.Tensors;
using RecursiveAI.OneModel.Tracing;
using RecursiveAI.OneModel.Training;
using static RecursiveAI.OneModel.Tests.Tracing.TraceHelpers;

namespace RecursiveAI.OneModel.Tests.Training;

// Conv1d against PyTorch 2.15.0 (testdata/conv1d, made by tools/conv1d_oracle.py), finite differences, thread and ISA independence,
// the module, tracing and a short training run.
[Collection("autograd")]
public class Conv1dTests
{
    private sealed record Case(string Prefix, Conv1dParams P, bool Bias, string Padding);

    private static readonly OracleFile Grid = OracleFile.Load("grid", "conv1d");

    private static List<Case> Cases()
    {
        string path = Path.Combine(OracleFile.DataDir("conv1d", ThisFile()), "grid.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        var list = new List<Case>();
        int i = 0;
        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            string prefix = $"c{i++}.";
            int k = Grid.ShapeOf(prefix + "in.w")[2];
            int stride = c.GetProperty("stride").GetInt32(), dil = c.GetProperty("dilation").GetInt32(), groups = c.GetProperty("groups").GetInt32();
            JsonElement pad = c.GetProperty("padding");
            Conv1dParams p = pad.ValueKind == JsonValueKind.Number
                ? Conv1dParams.Symmetric(stride, pad.GetInt32(), dil, groups)
                : pad.GetString() == "same" ? Conv1dParams.Same(k, dil, groups) : Conv1dParams.Symmetric(1, 0, dil, groups);
            list.Add(new Case(prefix, p, c.GetProperty("bias").GetBoolean(), pad.ToString()));
        }

        return list;
    }

    private static string ThisFile([System.Runtime.CompilerServices.CallerFilePath] string f = "") => f;

    private static (Tensor Y, Tensor X, Tensor W, Tensor B) Forward(Case c, bool grad = true)
    {
        Tensor x = Grid.Make(c.Prefix + "in.x", grad), w = Grid.Make(c.Prefix + "in.w", grad);
        Tensor b = c.Bias ? Grid.Make(c.Prefix + "in.b", grad) : default;
        return (Ops.Conv1d(x, w, b, c.P), x, w, b);
    }

    private static double ScaledErr(ReadOnlySpan<float> a, ReadOnlySpan<float> want)
    {
        double max = 0, mag = 1e-12;
        for (int i = 0; i < a.Length; i++)
        {
            max = Math.Max(max, Math.Abs((double)a[i] - want[i]));
            mag = Math.Max(mag, Math.Abs(want[i]));
        }

        return max / mag;
    }

    private static void Log(string line)
    {
        string? path = Environment.GetEnvironmentVariable("ONEMODEL_TRAINING_LOG");
        if (path is not null)
        {
            File.AppendAllText(path, line + "\n");
        }
    }

    // Runs every grid case, checks forward and the three gradients against PyTorch and returns a hash of all results.
    private static string RunGrid(bool compare, out double worst)
    {
        worst = 0;
        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (Case c in Cases())
        {
            (Tensor y, Tensor x, Tensor w, Tensor b) = Forward(c);
            Assert.Equal(Grid.ShapeOf(c.Prefix + "out.y"), y.Shape);
            Tensor loss = Ops.Sum(Ops.Mul(y, Grid.Make(c.Prefix + "in.r")));
            loss.Backward();
            sha.AppendData(MemoryMarshal.AsBytes(y.Data<float>()));
            sha.AppendData(MemoryMarshal.AsBytes(x.Grad.Data<float>()));
            sha.AppendData(MemoryMarshal.AsBytes(w.Grad.Data<float>()));
            if (c.Bias)
            {
                sha.AppendData(MemoryMarshal.AsBytes(b.Grad.Data<float>()));
            }

            if (compare)
            {
                var errs = new List<(string, double)>
                {
                    ("y", ScaledErr(y.Data<float>(), Grid.Values(c.Prefix + "out.y"))),
                    ("gx", ScaledErr(x.Grad.Data<float>(), Grid.Values(c.Prefix + "grad.x"))),
                    ("gw", ScaledErr(w.Grad.Data<float>(), Grid.Values(c.Prefix + "grad.w"))),
                };
                if (c.Bias)
                {
                    errs.Add(("gb", ScaledErr(b.Grad.Data<float>(), Grid.Values(c.Prefix + "grad.b"))));
                }

                foreach ((string what, double e) in errs)
                {
                    worst = Math.Max(worst, e);
                    Assert.True(e <= 1e-6, $"case {c.Prefix} (padding {c.Padding}, {c.P}): {what} scaled error {e:E2}");
                }
            }
        }

        return Convert.ToHexString(sha.GetHashAndReset());
    }

    private static void With(ArithmeticMode mode, int threads, Action body)
    {
        ArithmeticMode m = TensorCompute.Mode;
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

    [Fact]
    public void GridMatchesTorchInBothModes()
    {
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            With(mode, 1, () =>
            {
                RunGrid(true, out double worst);
                Log($"conv1d grid vs torch ({mode}, {Cases().Count} cases): worst scaled error {worst:E2}");
            });
        }
    }

    [Fact]
    public void FastResultsAreIdenticalAtOneAndEightThreads()
    {
        string one = "";
        With(ArithmeticMode.Fast, 1, () => one = RunGrid(false, out _));
        With(ArithmeticMode.Fast, 8, () => Assert.Equal(one, RunGrid(false, out _)));
    }

    // Pinned: the exact-mode results of the grid are the same on every ISA path (run with DOTNET_EnableAVX512=0 and
    // DOTNET_EnableHWIntrinsic=0), every thread count and every OS.
    private const string ExactGrid = "3F80FA57E3B459CFB037E225BBBBC773D8E9B17C3BA9BE3932ABB9184DFCAC4A";

    [Fact]
    public void ExactGridHashIsPinnedAndIndependentOfThreads()
    {
        string one = "";
        With(ArithmeticMode.Exact, 1, () => one = RunGrid(false, out _));
        With(ArithmeticMode.Exact, 8, () => Assert.Equal(one, RunGrid(false, out _)));
        Log($"conv1d exact grid hash: {one}");
        Assert.Equal(ExactGrid, one);
    }

    [Fact]
    public void FiniteDifferenceGradients()
    {
        GradCheck.Run("Conv1d bias", t => Ops.Conv1d(t[0], t[1], t[2]), [Shape.Create(2, 3, 7), Shape.Create(4, 3, 3), Shape.Create(4)]);
        GradCheck.Run("Conv1d stride 2 pad 1", t => Ops.Conv1d(t[0], t[1], default, Conv1dParams.Symmetric(2, 1)), [Shape.Create(2, 2, 8), Shape.Create(3, 2, 3)]);
        GradCheck.Run("Conv1d dilation 2 asymmetric pad", t => Ops.Conv1d(t[0], t[1], t[2], new Conv1dParams(1, 1, 3, 2)), [Shape.Create(1, 2, 9), Shape.Create(3, 2, 3), Shape.Create(3)]);
        GradCheck.Run("Conv1d groups 2", t => Ops.Conv1d(t[0], t[1], t[2], Conv1dParams.Symmetric(1, 1, 1, 2)), [Shape.Create(2, 4, 6), Shape.Create(6, 2, 3), Shape.Create(6)]);
        GradCheck.Run("Conv1d depthwise", t => Ops.Conv1d(t[0], t[1], t[2], Conv1dParams.Symmetric(1, 2, 1, 4)), [Shape.Create(2, 4, 7), Shape.Create(4, 1, 3), Shape.Create(4)]);
        GradCheck.Run("Conv1d same", t => Ops.Conv1d(t[0], t[1], default, Conv1dParams.Same(4)), [Shape.Create(1, 2, 6), Shape.Create(2, 2, 4)]);
    }

    [Fact]
    public void RejectsBadShapes()
    {
        Tensor x = Tensor.Zeros(DType.F32, Shape.Create(1, 4, 5));
        Assert.Throws<TensorShapeException>(() => Ops.Conv1d(x, Tensor.Zeros(DType.F32, Shape.Create(2, 3, 3))));
        Assert.Throws<TensorShapeException>(() => Ops.Conv1d(x, Tensor.Zeros(DType.F32, Shape.Create(2, 4, 3, 1))));
        Assert.Throws<TensorShapeException>(() => Ops.Conv1d(x, Tensor.Zeros(DType.F32, Shape.Create(2, 4, 7))));
        Assert.Throws<TensorShapeException>(() => Ops.Conv1d(x, Tensor.Zeros(DType.F32, Shape.Create(2, 4, 3)), Tensor.Zeros(DType.F32, Shape.Create(3))));
        Assert.Throws<ArgumentException>(() => new Conv1d(4, 4, 3, new Generator(1), "full"));
    }

    [Fact]
    public void ModuleInitialisationFollowsConv2d()
    {
        var a = new Conv1d(4, 6, 3, new Generator(9), groups: 2);
        var b = new Conv2d(4, 6, 3, new Generator(9), groups: 2);
        Assert.Equal(Shape.Create(6, 2, 3), a.Weight.Shape);
        Assert.Equal(Shape.Create(6), a.Bias.Shape);
        float bound = 1f / MathF.Sqrt(2 * 3);
        Assert.All(a.Weight.Data<float>().ToArray(), v => Assert.InRange(v, -bound, bound));
        Assert.Equal(new Conv1d(4, 6, 3, new Generator(9), groups: 2, bias: false).Weight.Data<float>().ToArray(), a.Weight.Data<float>().ToArray());
        Assert.Single(new Conv1d(4, 6, 3, new Generator(9), bias: false).Parameters());
        Assert.NotNull(b);
        // "same" keeps the length; "valid" shortens it by dilation * (kernel - 1).
        Tensor x = Tensor.Zeros(DType.F32, Shape.Create(2, 4, 10));
        Assert.Equal(Shape.Create(2, 6, 10), new Conv1d(4, 6, 4, new Generator(1), "same", dilation: 2).Forward(x).Shape);
        Assert.Equal(Shape.Create(2, 6, 6), new Conv1d(4, 6, 3, new Generator(1), "valid", dilation: 2).Forward(x).Shape);
    }

    private static Sequential Net(Generator g) => new(
        new Conv1d(3, 4, 3, g, padding: 1), Lambda.Relu(),
        new Conv1d(4, 4, 3, g, "same", dilation: 2, groups: 2), Lambda.Relu(),
        new Conv1d(4, 2, 3, g, stride: 2));

    private static Tensor Input(int n)
    {
        var rng = new Random(77 + n);
        var d = new float[n * 3 * 12];
        for (int i = 0; i < d.Length; i++)
        {
            d[i] = (float)(rng.NextDouble() - 0.5);
        }

        return Tensor.FromSpan<float>(d, Shape.Create(n, 3, 12));
    }

    [Fact]
    public void TracedConv1dLowersToOnnxConvAndMatchesEager()
    {
        Sequential net = Net(new Generator(3));
        net.Eval();
        using TracedModel tm = net.Trace(Input(2), "n", null, null);
        using Session s = tm.CreateSession();
        foreach (int n in new[] { 2, 1, 5 })
        {
            Tensor x = Input(n);
            Assert.True(Scaled(Run(s, ["input"], [x], "output"), net.Forward(x)) <= 1e-6f, $"batch {n}");
        }

        string tmp = Path.Combine(Path.GetTempPath(), "conv1d-" + Guid.NewGuid().ToString("N") + ".onnx");
        try
        {
            tm.SaveOnnx(tmp);
            byte[] bytes = File.ReadAllBytes(tmp);
            // One Conv node per layer, with 1-D attributes: no reshape to a 2-D image around them.
            Assert.True(bytes.AsSpan().IndexOf("Conv"u8) >= 0);
            Assert.True(bytes.AsSpan().IndexOf("Reshape"u8) < 0 && bytes.AsSpan().IndexOf("Unsqueeze"u8) < 0);
        }
        finally
        {
            File.Delete(tmp);
        }
    }

    private sealed class MeanOverLength : NnModule
    {
        public override Tensor Forward(Tensor input) => Ops.Mean(input, 2);
    }

    [Fact]
    public void SmallNetTrainsLikeTorch()
    {
        OracleFile o = OracleFile.Load("train", "conv1d");
        var g = new Generator(1);
        var model = new Sequential(
            new Conv1d(4, 6, 3, g, "same", dilation: 2), Lambda.Relu(),
            new Conv1d(6, 6, 3, g, stride: 2, padding: 1, groups: 2), Lambda.Relu(),
            new MeanOverLength(), new Linear(6, 3, g));
        (string, string)[] map =
        [
            ("0.weight", "c1w"), ("0.bias", "c1b"), ("2.weight", "c2w"), ("2.bias", "c2b"), ("5.weight", "fcw"), ("5.bias", "fcb"),
        ];
        foreach ((string p, string k) in map)
        {
            TrainCase.Set(model, p, o.Values("in." + k));
        }

        int steps = o.ShapeOf("in.order")[0], b = o.ShapeOf("in.order")[1];
        float[] x = o.Values("in.x"), y = o.Values("in.y"), order = o.Values("in.order"), want = o.Values("out.loss");
        var opt = new Sgd(model.Parameters(), 0.05f, momentum: 0.9f);
        double maxRel = 0;
        for (int s = 0; s < steps; s++)
        {
            using var scope = new ActivationScope();
            var xb = new float[b * 4 * 11];
            var yb = new long[b];
            for (int r = 0; r < b; r++)
            {
                int idx = (int)order[(s * b) + r];
                Array.Copy(x, idx * 44, xb, r * 44, 44);
                yb[r] = (long)y[idx];
            }

            Tensor xt = Tensor.FromSpan<float>(xb, Shape.Create(b, 4, 11)), yt = Tensor.FromSpan<long>(yb, Shape.Create(b));
            ActivationScope.Track(xt);
            ActivationScope.Track(yt);
            Tensor loss = Losses.CrossEntropy(model.Forward(xt), yt);
            float v = loss.Data<float>()[0];
            opt.ZeroGrad();
            loss.Backward();
            opt.Step();
            maxRel = Math.Max(maxRel, Math.Abs((double)v - want[s]) / Math.Abs(want[s]));
        }

        double param = 0;
        foreach ((string p, string k) in map)
        {
            param = Math.Max(param, ScaledErr(model.StateDict().First(e => e.Name == p).Tensor.Data<float>(), o.Values("out." + k)));
        }

        Log($"conv1d training vs torch ({steps} steps): loss max relative diff {maxRel:E2}, final parameter scaled error {param:E2}");
        Assert.True(maxRel <= 1e-5, $"loss relative difference {maxRel:E2}");
        Assert.True(want[^1] < want[0]);
    }
}
