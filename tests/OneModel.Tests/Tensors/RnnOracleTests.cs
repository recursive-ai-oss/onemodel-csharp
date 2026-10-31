using System.Globalization;
using System.Security.Cryptography;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Tensors;

// The models of tools/rnn_oracle.py on the eager ops, against the gradients PyTorch 2.15.0 (CPU) produced for the same weights.
[Collection("autograd")]
public class RnnOracleTests
{
    private static void Log(string line)
    {
        string? path = Environment.GetEnvironmentVariable("ONEMODEL_GRADCHECK_LOG");
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

    private static void Compare(string model, OracleFile o, Dictionary<string, Tensor> grads, double tolerance)
    {
        foreach (KeyValuePair<string, Tensor> kv in grads)
        {
            float[] expected = o.Values("grad." + kv.Key);
            Assert.Equal(o.ShapeOf("grad." + kv.Key), kv.Value.Shape);
            double err = Scaled(kv.Value.Data<float>(), expected);
            Log(string.Create(CultureInfo.InvariantCulture, $"| {model} {kv.Key} {kv.Value.Shape} | {err:E2} |"));
            Assert.True(err <= tolerance, $"{model} {kv.Key}: scaled error {err:E2} exceeds {tolerance:E1}");
        }
    }

    private static readonly string[] LstmNames = ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0",
        "weight_ih_l0_reverse", "weight_hh_l0_reverse", "bias_ih_l0_reverse", "bias_hh_l0_reverse",
        "weight_ih_l1", "weight_hh_l1", "bias_ih_l1", "bias_hh_l1",
        "weight_ih_l1_reverse", "weight_hh_l1_reverse", "bias_ih_l1_reverse", "bias_hh_l1_reverse"];

    private static RnnLayer[] LayersOf(Dictionary<string, Tensor> p, string prefix, int layers, bool bidirectional, Tensor[]? h0 = null)
    {
        int dirs = bidirectional ? 2 : 1;
        var result = new RnnLayer[layers * dirs];
        for (int l = 0; l < layers; l++)
        {
            for (int d = 0; d < dirs; d++)
            {
                string sfx = $"_l{l}" + (d == 1 ? "_reverse" : "");
                result[(l * dirs) + d] = new RnnLayer(p[prefix + "weight_ih" + sfx], p[prefix + "weight_hh" + sfx], p[prefix + "bias_ih" + sfx],
                    p[prefix + "bias_hh" + sfx], h0 is not null && l == 0 ? h0[d] : default);
            }
        }

        return result;
    }

    private static Dictionary<string, Tensor> Leaves(OracleFile o, IEnumerable<string> names) =>
        names.ToDictionary(n => n, n => o.Make("in." + n, requiresGrad: true));

    private static (Tensor Lp, Dictionary<string, Tensor> P) Crnn(OracleFile o, Tensor? lpOut = null)
    {
        var names = new List<string> { "x", "c1w", "c1b", "c2w", "c2b", "fcw", "fcb" };
        names.AddRange(LstmNames.Select(n => "lstm." + n));
        Dictionary<string, Tensor> p = Leaves(o, names);
        Tensor h = Ops.MaxPool2d(Ops.Relu(Ops.Conv2d(p["x"], p["c1w"], p["c1b"], new Conv2dParams(PadH: 1, PadW: 1))), new Pool2dParams(2, 2, 2, 2, 0, 0));
        h = Ops.MaxPool2d(Ops.Relu(Ops.Conv2d(h, p["c2w"], p["c2b"], new Conv2dParams(PadH: 1, PadW: 1))), new Pool2dParams(4, 1, 4, 1, 0, 0));
        Tensor seq = h.Reshape(2, 6, 12).Permute(2, 0, 1);
        Tensor y = Ops.Lstm(seq, LayersOf(p, "lstm.", 2, true), bidirectional: true);
        Tensor logits = Ops.Add(Ops.MatMul(y.Reshape(24, 10), p["fcw"].T()), p["fcb"]).Reshape(12, 2, 7);
        return (Ops.LogSoftmax(logits), p);
    }

    private static int[] Ints(OracleFile o, string name) => o.Values(name).Select(v => (int)v).ToArray();

    [Fact]
    public void CrnnWithCtcMatchesPyTorch()
    {
        OracleFile o = OracleFile.Load("crnn");
        int[] targets = Ints(o, "in.targets"), inLen = Ints(o, "in.input_lengths"), tgtLen = Ints(o, "in.target_lengths");
        using var scope = new ActivationScope();
        (Tensor lp, Dictionary<string, Tensor> p) = Crnn(o);
        Assert.True(Scaled(lp.Data<float>(), o.Values("out.lp")) < 1e-5);
        Tensor loss = Ops.CtcLoss(lp, targets, inLen, tgtLen, 0, CtcReduction.Mean);
        double rel = Math.Abs(loss.Data<float>()[0] - o.Values("out.loss")[0]) / Math.Abs(o.Values("out.loss")[0]);
        Log(string.Create(CultureInfo.InvariantCulture, $"| crnn ctc mean loss | {rel:E2} (relative) |"));
        Assert.True(rel < 1e-5, $"CTC mean loss {loss.Data<float>()[0]} vs {o.Values("out.loss")[0]}");
        loss.Backward();
        Compare("crnn", o, p.ToDictionary(k => k.Key, k => k.Value.Grad), 1e-4);
    }

    [Fact]
    public void CtcReductionsAndZeroInfinityMatchPyTorch()
    {
        OracleFile o = OracleFile.Load("crnn");
        int[] targets = Ints(o, "in.targets"), inLen = Ints(o, "in.input_lengths"), tgtLen = Ints(o, "in.target_lengths");
        using Tensor lp = o.Make("out.lp");
        foreach ((string key, CtcReduction red) in new[] { ("none", CtcReduction.None), ("sum", CtcReduction.Sum), ("mean", CtcReduction.Mean) })
        {
            using Tensor l = Ops.CtcLoss(lp, targets, inLen, tgtLen, 0, red);
            float[] expected = o.Values("out.ctc_" + key);
            ReadOnlySpan<float> ours = l.Data<float>();
            Assert.Equal(expected.Length, ours.Length);
            for (int i = 0; i < ours.Length; i++)
            {
                double rel = Math.Abs(ours[i] - expected[i]) / Math.Abs(expected[i]);
                Log(string.Create(CultureInfo.InvariantCulture, $"| ctc {key}[{i}] | {rel:E2} (relative) |"));
                Assert.True(rel < 1e-5, $"ctc {key}[{i}]: {ours[i]} vs {expected[i]}");
            }
        }

        int[] shortLen = [inLen[0], 2];
        using Tensor inf = Ops.CtcLoss(lp, targets, shortLen, tgtLen, 0, CtcReduction.None);
        Assert.True(float.IsPositiveInfinity(o.Values("out.ctc_inf_none")[1]) && float.IsPositiveInfinity(inf.Data<float>()[1]));
        using Tensor zero = Ops.CtcLoss(lp, targets, shortLen, tgtLen, 0, CtcReduction.None, zeroInfinity: true);
        Assert.Equal(o.Values("out.ctc_zeroinf_none")[1], zero.Data<float>()[1]);
        Assert.Equal(o.Values("out.ctc_zeroinf_none")[0], zero.Data<float>()[0], 4);
    }

    [Fact]
    public void CtcGradientDiffersFromPyTorchsByTheSoftmaxTerm()
    {
        // PyTorch's gradient with respect to log_probs carries an extra exp(lp) per used frame (it vanishes through log_softmax);
        // ours is the true partial derivative. Both give the same gradient with respect to the logits (checked above).
        OracleFile o = OracleFile.Load("crnn");
        int[] targets = Ints(o, "in.targets"), inLen = Ints(o, "in.input_lengths"), tgtLen = Ints(o, "in.target_lengths");
        Tensor lp = o.Make("out.lp", requiresGrad: true);
        Tensor loss = Ops.CtcLoss(lp, targets, inLen, tgtLen);
        loss.Backward();
        ReadOnlySpan<float> ours = lp.Grad.Data<float>(), torch = o.Values("grad.lp"), lpv = o.Values("out.lp");
        double max = 0;
        for (int t = 0; t < 12; t++)
        {
            for (int b = 0; b < 2; b++)
            {
                double w = 1.0 / Math.Max(1, tgtLen[b]) / 2;
                for (int c = 0; c < 7; c++)
                {
                    int i = (((t * 2) + b) * 7) + c;
                    double expected = ours[i] + (t < inLen[b] ? Math.Exp(lpv[i]) * w : 0);
                    max = Math.Max(max, Math.Abs(expected - torch[i]));
                }
            }
        }

        Assert.True(max < 1e-5, $"max difference {max:E2}");
    }

    [Fact]
    public void BidirectionalGruMatchesPyTorch()
    {
        OracleFile o = OracleFile.Load("gru");
        var names = new List<string> { "x", "h0" };
        string[] gn = ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0", "weight_ih_l0_reverse", "weight_hh_l0_reverse", "bias_ih_l0_reverse", "bias_hh_l0_reverse"];
        names.AddRange(gn.Select(n => "gru." + n));
        Dictionary<string, Tensor> p = Leaves(o, names);
        using Tensor proj = o.Make("in.proj");
        using var scope = new ActivationScope();
        Tensor[] h0 = [p["h0"].Select(0, 0), p["h0"].Select(0, 1)];
        Tensor y = Ops.Gru(p["x"], LayersOf(p, "gru.", 1, true, h0), bidirectional: true, batchFirst: true);
        Assert.True(Scaled(y.Contiguous().Data<float>(), o.Values("out.y")) < 1e-5);
        Tensor loss = Ops.Sum(Ops.Mul(y, proj));
        Assert.True(Math.Abs(loss.Data<float>()[0] - o.Values("out.loss")[0]) / Math.Abs(o.Values("out.loss")[0]) < 1e-5);
        loss.Backward();
        Compare("gru", o, p.ToDictionary(k => k.Key, k => k.Value.Grad), 1e-4);
    }

    private static (Tensor Loss, Dictionary<string, Tensor> P, Tensor A, Tensor C) Decoder(OracleFile o)
    {
        Dictionary<string, Tensor> p = Leaves(o, ["x", "c1w", "c1b", "t1w", "t1b", "t2w", "t2b", "c2w", "c2b"]);
        Tensor h = Ops.Relu(Ops.Conv2d(p["x"], p["c1w"], p["c1b"], new Conv2dParams(PadH: 1, PadW: 1)));
        h = Ops.Relu(Ops.ConvTranspose2d(h, p["t1w"], p["t1b"], new ConvTranspose2dParams(StrideH: 2, StrideW: 2)));
        h = Ops.Relu(Ops.ConvTranspose2d(h, p["t2w"], p["t2b"], new ConvTranspose2dParams(2, 2, 1, 1, 1, 1, Groups: 2)));
        h = Ops.Resize2d(h, 13, 30, InterpolationMode.Nearest);
        h = Ops.Conv2d(h, p["c2w"], p["c2b"], new Conv2dParams(PadH: 1, PadW: 1));
        Tensor a = Ops.Resize2d(h, 9, 17, InterpolationMode.Bilinear, alignCorners: false);
        Tensor c = Ops.Resize2d(h, 16, 11, InterpolationMode.Bilinear, alignCorners: true);
        Tensor loss = Ops.Add(Ops.Sum(Ops.Mul(a, o.Make("in.pa"))), Ops.Sum(Ops.Mul(c, o.Make("in.pc"))));
        return (loss, p, a, c);
    }

    [Fact]
    public void ConvTransposeAndResizeDecoderMatchesPyTorch()
    {
        OracleFile o = OracleFile.Load("decoder");
        using var scope = new ActivationScope();
        (Tensor loss, Dictionary<string, Tensor> p, Tensor a, Tensor c) = Decoder(o);
        Assert.True(Scaled(a.Data<float>(), o.Values("out.a")) < 1e-5);
        Assert.True(Scaled(c.Data<float>(), o.Values("out.c")) < 1e-5);
        loss.Backward();
        Compare("decoder", o, p.ToDictionary(k => k.Key, k => k.Value.Grad), 1e-4);
    }

    private static string HashOf(Dictionary<string, Tensor> grads, float loss) => GradHash.Of(grads, loss);

    // Gradients at 1 and 8 threads are bit-identical in both modes (the engine is single-threaded; threads exist only inside the GEMM
    // and convolution, whose chunking is fixed).
    [Theory]
    [InlineData(ArithmeticMode.Fast)]
    [InlineData(ArithmeticMode.Exact)]
    public void GradientsDoNotDependOnThreads(ArithmeticMode mode)
    {
        ArithmeticMode oldMode = TensorCompute.Mode;
        int oldThreads = TensorCompute.Threads;
        try
        {
            TensorCompute.Mode = mode;
            var hashes = new List<string>();
            foreach (int threads in new[] { 1, 8 })
            {
                TensorCompute.Threads = threads;
                OracleFile o = OracleFile.Load("crnn");
                using var scope = new ActivationScope();
                (Tensor lp, Dictionary<string, Tensor> p) = Crnn(o);
                Tensor loss = Ops.CtcLoss(lp, Ints(o, "in.targets"), Ints(o, "in.input_lengths"), Ints(o, "in.target_lengths"));
                loss.Backward();
                hashes.Add(HashOf(p.ToDictionary(k => k.Key, k => k.Value.Grad), loss.Data<float>()[0]));
                OracleFile d = OracleFile.Load("decoder");
                (Tensor dl, Dictionary<string, Tensor> dp, _, _) = Decoder(d);
                dl.Backward();
                hashes.Add(HashOf(dp.ToDictionary(k => k.Key, k => k.Value.Grad), dl.Data<float>()[0]));
            }

            Assert.Equal(hashes[0], hashes[2]);
            Assert.Equal(hashes[1], hashes[3]);
            if (mode == ArithmeticMode.Exact)
            {
                // Pinned: the same on every OS, CPU, instruction set and thread count (checked on x64 AVX-512/AVX2/128-bit/scalar and arm64).
                Assert.Equal("589335EBDC4E8E20BF30EFABE1C74CC0D9906C90BBEB57D1694CDC26508FD565", hashes[0]);
                Assert.Equal("0AA32D22988052EDC8688A07374C218A7232EB6C531F768D0550206F140F1468", hashes[1]);
            }

            Log($"| thread-independent {mode} | crnn {hashes[0]} decoder {hashes[1]} |");
        }
        finally
        {
            TensorCompute.Mode = oldMode;
            TensorCompute.Threads = oldThreads;
        }
    }
}
