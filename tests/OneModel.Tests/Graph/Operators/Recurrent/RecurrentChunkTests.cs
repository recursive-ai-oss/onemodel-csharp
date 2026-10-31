using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Kernels;
using static RecursiveAI.OneModel.Tests.Graph.Operators.Norms.NormHarness;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.Recurrent;

/// <summary>
/// Recurrent layers wide enough to be split into several chunks of hidden units, with batches that use the four-row product, its
/// remainder rows and the blocked GEMM: against a double-precision reference, and across kernel paths and thread counts.
/// </summary>
public class RecurrentChunkTests
{
    private const int Input = 11, Seq = 7;

    // Set by each test before it builds anything.
    private int Hidden = 70;

    private static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));

    // Layout 0, both directions, plain activations (Sigmoid/Tanh). Returns Y [seq, 2, batch, hidden] and Y_h [2, batch, hidden].
    private (double[] Y, double[] Yh) Reference(string op, bool lbr, int batch, int[]? lens, float[] x, float[] w, float[] r, float[] bias)
    {
        int g = op == "LSTM" ? 4 : op == "GRU" ? 3 : 1;
        int gh = g * Hidden;
        double[] y = new double[Seq * 2 * batch * Hidden];
        double[] yh = new double[2 * batch * Hidden];
        for (int dir = 0; dir < 2; dir++)
        {
            for (int b = 0; b < batch; b++)
            {
                int len = lens?[b] ?? Seq;
                double[] h = new double[Hidden], c = new double[Hidden];
                for (int s = 0; s < len; s++)
                {
                    int t = dir == 0 ? s : len - 1 - s;
                    double[] pre = new double[gh];
                    double[] rec = new double[gh];
                    for (int q = 0; q < gh; q++)
                    {
                        double a = 0, ar = 0;
                        for (int i = 0; i < Input; i++)
                        {
                            a += x[(((t * batch) + b) * Input) + i] * (double)w[(((dir * gh) + q) * Input) + i];
                        }

                        for (int j = 0; j < Hidden; j++)
                        {
                            ar += h[j] * (double)r[(((dir * gh) + q) * Hidden) + j];
                        }

                        pre[q] = a + (double)bias[(dir * 2 * gh) + q];
                        rec[q] = ar + (double)bias[(dir * 2 * gh) + gh + q];
                    }

                    double[] hn = new double[Hidden];
                    if (op == "LSTM")
                    {
                        for (int j = 0; j < Hidden; j++)
                        {
                            double i = Sigmoid(pre[j] + rec[j]), o = Sigmoid(pre[Hidden + j] + rec[Hidden + j]);
                            double f = Sigmoid(pre[(2 * Hidden) + j] + rec[(2 * Hidden) + j]), cc = Math.Tanh(pre[(3 * Hidden) + j] + rec[(3 * Hidden) + j]);
                            c[j] = (f * c[j]) + (i * cc);
                            hn[j] = o * Math.Tanh(c[j]);
                        }
                    }
                    else if (op == "GRU")
                    {
                        double[] rh = new double[Hidden];
                        double[] z = new double[Hidden], rr = new double[Hidden];
                        for (int j = 0; j < Hidden; j++)
                        {
                            z[j] = Sigmoid(pre[j] + rec[j]);
                            rr[j] = Sigmoid(pre[Hidden + j] + rec[Hidden + j]);
                            rh[j] = rr[j] * h[j];
                        }

                        for (int j = 0; j < Hidden; j++)
                        {
                            double cand;
                            if (lbr)
                            {
                                cand = Math.Tanh(pre[(2 * Hidden) + j] + (rr[j] * rec[(2 * Hidden) + j]));
                            }
                            else
                            {
                                double a = 0;
                                for (int k = 0; k < Hidden; k++)
                                {
                                    a += rh[k] * (double)r[(((dir * gh) + (2 * Hidden) + j) * Hidden) + k];
                                }

                                cand = Math.Tanh(pre[(2 * Hidden) + j] + a + (double)bias[(dir * 2 * gh) + gh + (2 * Hidden) + j]);
                            }

                            hn[j] = ((1 - z[j]) * cand) + (z[j] * h[j]);
                        }
                    }
                    else
                    {
                        for (int j = 0; j < Hidden; j++)
                        {
                            hn[j] = Math.Tanh(pre[j] + rec[j]);
                        }
                    }

                    h = hn;
                    for (int j = 0; j < Hidden; j++)
                    {
                        y[((((t * 2) + dir) * batch) + b) * Hidden + j] = h[j];
                    }
                }

                if (len > 0)
                {
                    Array.Copy(h, 0, yh, (((dir * batch) + b) * Hidden), Hidden);
                }
            }
        }

        return (y, yh);
    }

    private byte[] Model(string op, bool lbr, int batch, int[]? lens, float[] w, float[] r, float[] bias)
    {
        int g = op == "LSTM" ? 4 : op == "GRU" ? 3 : 1;
        var mb = new ModelBuilder(17).Input("x", 1, Seq, batch, Input)
            .Initializer("w", w, 2, g * Hidden, Input).Initializer("r", r, 2, g * Hidden, Hidden).Initializer("b", bias, 2, 2 * g * Hidden);
        string lensName = "";
        if (lens is not null)
        {
            byte[] raw = new byte[lens.Length * 4];
            for (int i = 0; i < lens.Length; i++)
            {
                BitConverter.TryWriteBytes(raw.AsSpan(i * 4), lens[i]);
            }

            mb.RawInitializer("lens", 6, [batch], raw);
            lensName = "lens";
        }

        var attrs = new List<(string, object)> { ("hidden_size", (long)Hidden), ("direction", "bidirectional") };
        if (op == "GRU")
        {
            attrs.Add(("linear_before_reset", lbr ? 1L : 0L));
        }

        return mb.Output("y", 1, Seq, 2, batch, Hidden).Output("yh", 1, 2, batch, Hidden)
            .Node(op, ["x", "w", "r", "b", lensName], ["y", "yh"], [.. attrs]).Build();
    }

    public static IEnumerable<object[]> Cases()
    {
        foreach (string op in new[] { "LSTM", "GRU", "GRU_LBR", "RNN" })
        {
            // 70 hidden units: three chunks, the last one ragged, run on the calling thread (the steps are too small to wake helpers).
            foreach (int batch in new[] { 1, 3, 6, 18 })
            {
                yield return [op, 70, batch, false];
                yield return [op, 70, batch, true];
            }

            // 192 hidden units at batch 8 are about 150 microseconds of work per step, so these run the chunks on helper threads.
            yield return [op, 192, 8, false];
            yield return [op, 192, 8, true];
            yield return [op, 192, 1, false];
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void MatchesTheReferenceAndDoesNotDependOnThreadsOrPath(string opName, int hidden, int batch, bool withLens)
    {
        Hidden = hidden;
        bool lbr = opName == "GRU_LBR";
        string op = lbr ? "GRU" : opName;
        var rng = new Random(batch + opName.Length + hidden);
        int g = op == "LSTM" ? 4 : op == "GRU" ? 3 : 1;
        float[] x = Random(rng, Seq * batch * Input, -1, 1);
        float[] w = Random(rng, 2 * g * Hidden * Input, -0.5, 0.5);
        float[] r = Random(rng, 2 * g * Hidden * Hidden, -0.3, 0.3);
        float[] bias = Random(rng, 2 * 2 * g * Hidden, -0.3, 0.3);
        int[]? lens = null;
        if (withLens)
        {
            lens = new int[batch];
            for (int i = 0; i < batch; i++)
            {
                lens[i] = (i * 3) % (Seq + 1);
            }

            lens[batch - 1] = Seq;
        }

        byte[] model = Model(op, lbr, batch, lens, w, r, bias);
        (double[] y, double[] yh) = Reference(op, lbr, batch, lens, x, w, r, bias);
        int[] shape = [Seq, batch, Input];

        Out[] fast = Run(model, ArithmeticMode.Fast, null, 1, 2, (x, shape));
        Assert.True(ScaledError(fast[0].F!, y) < 2e-4, $"Y {ScaledError(fast[0].F!, y)}");
        Assert.True(ScaledError(fast[1].F!, yh) < 2e-4, $"Y_h {ScaledError(fast[1].F!, yh)}");
        foreach (int threads in new[] { 3, 8 })
        {
            Out[] more = Run(model, ArithmeticMode.Fast, null, threads, 2, (x, shape));
            AssertSameBits(more[0].F!, fast[0].F!, $"fast Y, {threads} threads");
            AssertSameBits(more[1].F!, fast[1].F!, $"fast Y_h, {threads} threads");
        }

        Out[]? golden = null;
        foreach (KernelPath path in Paths)
        {
            foreach (int threads in new[] { 1, 6 })
            {
                Out[] ex = Run(model, ArithmeticMode.Exact, path, threads, 2, (x, shape));
                golden ??= ex;
                AssertSameBits(ex[0].F!, golden[0].F!, $"exact Y, {path}, {threads} threads");
                AssertSameBits(ex[1].F!, golden[1].F!, $"exact Y_h, {path}, {threads} threads");
            }
        }

        Assert.True(ScaledError(golden![0].F!, y) < 2e-4);
    }
}
