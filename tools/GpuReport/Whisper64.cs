using System;
using System.Collections.Generic;
using System.Numerics;
using System.Linq;
using System.Threading.Tasks;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace GpuReport;

/// <summary>
/// A float64 evaluation of the Whisper encoder graph as exported (tools/export_models.py): the same weights widened to double, the
/// same constants (including the float32 value of sqrt(2) and the layer-norm epsilon), the same order of operations at graph level,
/// and a true double erf. It is the higher-precision reference for error attribution: tools only, never part of a package.
/// Values are named like the graph's tensors so taps line up with float32 runs of a graph that exposes them as outputs.
/// </summary>
internal sealed class Emulation
{
    /// <summary>Round every operator's output to float32 (what storing tensors in float32 does).</summary>
    public bool RoundAll;

    /// <summary>"f64" (exact), "seq" (float32 fused multiply-add in increasing k), "seqnf" (the same with the product rounded before the add), "comp" (float32 sum with the rounding errors of the products and of the adds carried along, a bound on what better accumulation can give), "lane8" (eight interleaved float32 accumulators, added pairwise at the end) or "chunk64" (float32 sums of 64 products, chunks added in order).</summary>
    public string MatMul = "f64";

    /// <summary>Which products use <see cref="MatMul"/>: any of conv, q, k, v, o, fc1, fc2; empty means all.</summary>
    public HashSet<string> Which = [];

    /// <summary>Layer norm evaluated in float32 (mean, then variance, each a float32 sum in channel order).</summary>
    public bool LayerNorm32;

    /// <summary>Gelu evaluated as float32 arithmetic around a double erf rounded to float32.</summary>
    public bool Gelu32;

    public static Emulation Parse(string spec)
    {
        var e = new Emulation();
        foreach (string t in spec.Split('+'))
        {
            string[] kv = t.Split(':');
            switch (kv[0])
            {
                case "f64": break;
                case "r": e.RoundAll = true; break;
                case "seq": case "seqnf": case "comp": case "lane8": case "chunk64": e.MatMul = kv[0]; if (kv.Length > 1) { foreach (string w in kv[1].Split(',')) { e.Which.Add(w); } } break;
                case "ln32": e.LayerNorm32 = true; break;
                case "gelu32": e.Gelu32 = true; break;
                default: throw new ArgumentException("unknown emulation token " + t);
            }
        }

        return e;
    }

    public bool Uses(string label) => MatMul != "f64" && (Which.Count == 0 || Which.Contains(label));
}

internal sealed class Whisper64
{
    /// <summary>What to round to float32 and how products are summed; null evaluates everything in double.</summary>
    public Emulation? Emu { get; set; }

    private double[] R(double[] x)
    {
        if (Emu is { RoundAll: true })
        {
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = (float)x[i];
            }
        }

        return x;
    }

    private const double Sqrt2AsFloat = 1.4142135381698608;
    private readonly IrModel _model;
    private readonly Dictionary<string, IrNode> _nodes = [];
    private readonly int _heads = 12;
    private readonly int _headDim = 64;
    private readonly HashSet<string>? _wanted;

    public Whisper64(string path, IEnumerable<string>? wanted)
    {
        _model = OnnxImporter.Load(path);
        foreach (IrNode n in _model.Graph.Nodes)
        {
            _nodes[n.Name] = n;
        }

        _wanted = wanted is null ? null : [.. wanted];
    }

    private double[] Weight(string nodeName, int input)
    {
        IrValue v = _nodes[nodeName].Inputs[input]!;
        return ToDouble(v.Initializer!);
    }

    private static double[] ToDouble(TensorData t)
    {
        using Tensor tensor = t.ToTensor();
        float[] f = tensor.Data<float>().ToArray();
        double[] d = new double[f.Length];
        for (int i = 0; i < f.Length; i++)
        {
            d[i] = f[i];
        }

        return d;
    }

    private float EpsilonOf(string nodeName) => _nodes[nodeName].FindAttribute("epsilon")!.Real;

    /// <summary>Runs the encoder on a [batch, 80, 3000] mel input; returns every value that was asked for, plus <c>encoder_hidden_states</c>.</summary>
    public Dictionary<string, double[]> Run(double[] mel, int batch, int melBins = 80, int frames = 3000)
    {
        var taps = new Dictionary<string, double[]>();
        void Tap(string name, double[] v)
        {
            if (_wanted is null || _wanted.Contains(name))
            {
                taps[name] = v;
            }
        }

        // Stem: two convolutions with the exact (erf) Gelu between them.
        double[] c1 = (Conv1d(mel, batch, melBins, frames, Weight("/e/conv1/Conv", 1), Weight("/e/conv1/Conv", 2), 768, 3, 1, Emu is { } e1 && e1.Uses("conv") ? e1.MatMul : "f64"));
        c1 = R(c1);
        Tap("/e/conv1/Conv_output_0", c1);
        double[] g1 = Gelu(c1);
        int frames2 = (frames + 2 - 3) / 2 + 1;
        double[] c2 = (Conv1d(g1, batch, 768, frames, Weight("/e/conv2/Conv", 1), Weight("/e/conv2/Conv", 2), 768, 3, 2, Emu is { } e2 && e2.Uses("conv") ? e2.MatMul : "f64"));
        c2 = R(c2);
        Tap("/e/conv2/Conv_output_0", c2);
        double[] g2 = Gelu(c2);
        Tap("/e/Mul_3_output_0", g2);
        double[] pos = ToDouble(_nodes["/e/Add_2"].Inputs[1]!.Initializer!);
        int t = frames2, d = 768;
        double[] x = new double[batch * t * d];
        for (int b = 0; b < batch; b++)
        {
            for (int tt = 0; tt < t; tt++)
            {
                for (int c = 0; c < d; c++)
                {
                    x[(((b * t) + tt) * d) + c] = g2[(((b * d) + c) * t) + tt] + pos[(tt * d) + c];
                }
            }
        }

        R(x);
        Tap("/e/Add_2_output_0", (double[])x.Clone());
        for (int l = 0; l < 12; l++)
        {
            string p = $"/e/layers.{l}";
            double[] ln1 = LayerNorm(x, d, Weight($"{p}/self_attn_layer_norm/LayerNormalization", 1), Weight($"{p}/self_attn_layer_norm/LayerNormalization", 2), EpsilonOf($"{p}/self_attn_layer_norm/LayerNormalization"));
            Tap($"{p}/self_attn_layer_norm/LayerNormalization_output_0", ln1);
            int rows = batch * t;
            double[] q = MatMul("q", ln1, rows, d, Weight($"{p}/self_attn/q_proj/MatMul", 1), d);
            AddBias(q, Weight($"{p}/self_attn/q_proj/Add", 0));
            for (int i = 0; i < q.Length; i++)
            {
                q[i] *= 0.125;
            }

            double[] k = MatMul("k", ln1, rows, d, Weight($"{p}/self_attn/k_proj/MatMul", 1), d);
            double[] v = MatMul("v", ln1, rows, d, Weight($"{p}/self_attn/v_proj/MatMul", 1), d);
            AddBias(v, Weight($"{p}/self_attn/v_proj/Add", 0));
            double[] ctx = Attention(q, k, v, batch, t, p, Tap);
            double[] o = MatMul("o", ctx, rows, d, Weight($"{p}/self_attn/out_proj/MatMul", 1), d);
            AddBias(o, Weight($"{p}/self_attn/out_proj/Add", 0));
            Tap($"{p}/self_attn/out_proj/Add_output_0", o);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] += o[i];
            }

            R(x);

            Tap($"{p}/Add_output_0", (double[])x.Clone());
            double[] ln2 = LayerNorm(x, d, Weight($"{p}/final_layer_norm/LayerNormalization", 1), Weight($"{p}/final_layer_norm/LayerNormalization", 2), EpsilonOf($"{p}/final_layer_norm/LayerNormalization"));
            Tap($"{p}/final_layer_norm/LayerNormalization_output_0", ln2);
            double[] h = MatMul("fc1", ln2, rows, d, Weight($"{p}/fc1/MatMul", 1), 3072);
            AddBias(h, Weight($"{p}/fc1/Add", 0));
            Tap($"{p}/fc1/Add_output_0", h);
            h = Gelu(h);
            Tap($"{p}/activation_fn/Mul_1_output_0", h);
            double[] f = MatMul("fc2", h, rows, 3072, Weight($"{p}/fc2/MatMul", 1), d);
            AddBias(f, Weight($"{p}/fc2/Add", 0));
            Tap($"{p}/fc2/Add_output_0", f);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] += f[i];
            }

            R(x);

            Tap($"{p}/Add_1_output_0", (double[])x.Clone());
        }

        taps["encoder_hidden_states"] = LayerNorm(x, d, Weight("/e/layer_norm/LayerNormalization", 1), Weight("/e/layer_norm/LayerNormalization", 2), EpsilonOf("/e/layer_norm/LayerNormalization"));
        return taps;
    }

    /// <summary>One stage of layer <paramref name="l"/> in float64 from the given input: ln1, attn, ln2, fc1, gelu, fc2 or ffn (ln2's output to fc2's).</summary>
    public double[] Stage(string kind, int l, double[] input, int batch)
    {
        string p = $"/e/layers.{l}";
        int d = 768, t = input.Length / (batch * (kind is "gelu" or "fc2" ? 3072 : d)), rows = batch * t;
        switch (kind)
        {
            case "ln1":
                return LayerNorm(input, d, Weight($"{p}/self_attn_layer_norm/LayerNormalization", 1), Weight($"{p}/self_attn_layer_norm/LayerNormalization", 2), EpsilonOf($"{p}/self_attn_layer_norm/LayerNormalization"));
            case "ln2":
                return LayerNorm(input, d, Weight($"{p}/final_layer_norm/LayerNormalization", 1), Weight($"{p}/final_layer_norm/LayerNormalization", 2), EpsilonOf($"{p}/final_layer_norm/LayerNormalization"));
            case "fc1":
            {
                double[] h = MatMul("fc1", input, rows, d, Weight($"{p}/fc1/MatMul", 1), 3072);
                AddBias(h, Weight($"{p}/fc1/Add", 0));
                return h;
            }

            case "gelu":
                return Gelu(input);
            case "fc2":
            {
                double[] f = MatMul("fc2", input, rows, 3072, Weight($"{p}/fc2/MatMul", 1), d);
                AddBias(f, Weight($"{p}/fc2/Add", 0));
                return f;
            }

            case "ffn":
                return Stage("fc2", l, Gelu(Stage("fc1", l, input, batch)), batch);
            case "attn":
            {
                double[] q = MatMul("q", input, rows, d, Weight($"{p}/self_attn/q_proj/MatMul", 1), d);
                AddBias(q, Weight($"{p}/self_attn/q_proj/Add", 0));
                for (int i = 0; i < q.Length; i++)
                {
                    q[i] *= 0.125;
                }

                double[] k = MatMul("k", input, rows, d, Weight($"{p}/self_attn/k_proj/MatMul", 1), d);
                double[] v = MatMul("v", input, rows, d, Weight($"{p}/self_attn/v_proj/MatMul", 1), d);
                AddBias(v, Weight($"{p}/self_attn/v_proj/Add", 0));
                double[] ctx = Attention(q, k, v, batch, t, p, (_, _) => { });
                double[] o = MatMul("o", ctx, rows, d, Weight($"{p}/self_attn/out_proj/MatMul", 1), d);
                AddBias(o, Weight($"{p}/self_attn/out_proj/Add", 0));
                return o;
            }

            default:
                throw new ArgumentException(kind);
        }
    }

    private double[] Attention(double[] q, double[] k, double[] v, int batch, int t, string p, Action<string, double[]> tap)
    {
        int d = _heads * _headDim;
        double[] ctx = new double[batch * t * d];
        double[]? probsTap = _wanted is null || _wanted.Contains($"{p}/self_attn/Softmax_output_0") ? new double[batch * _heads * t * t] : null;
        double[]? scoresTap = _wanted is null || _wanted.Contains($"{p}/self_attn/Mul_1_output_0") ? new double[batch * _heads * t * t] : null;
        Parallel.For(0, batch * _heads, new ParallelOptions { MaxDegreeOfParallelism = 8 }, bh =>
        {
            int b = bh / _heads, h = bh % _heads;
            double[] row = new double[t];
            for (int i = 0; i < t; i++)
            {
                int qo = (((b * t) + i) * d) + (h * _headDim);
                double max = double.NegativeInfinity;
                for (int j = 0; j < t; j++)
                {
                    int ko = (((b * t) + j) * d) + (h * _headDim);
                    double s = 0;
                    for (int c = 0; c < _headDim; c++)
                    {
                        s += q[qo + c] * k[ko + c];
                    }

                    row[j] = s;
                    max = Math.Max(max, s);
                }

                if (scoresTap is not null)
                {
                    Array.Copy(row, 0, scoresTap, ((((b * _heads) + h) * t) + i) * t, t);
                }

                double sum = 0;
                for (int j = 0; j < t; j++)
                {
                    row[j] = Math.Exp(row[j] - max);
                    sum += row[j];
                }

                for (int j = 0; j < t; j++)
                {
                    row[j] /= sum;
                }

                if (probsTap is not null)
                {
                    Array.Copy(row, 0, probsTap, ((((b * _heads) + h) * t) + i) * t, t);
                }

                int co = (((b * t) + i) * d) + (h * _headDim);
                for (int j = 0; j < t; j++)
                {
                    int vo = (((b * t) + j) * d) + (h * _headDim);
                    double pj = row[j];
                    for (int c = 0; c < _headDim; c++)
                    {
                        ctx[co + c] += pj * v[vo + c];
                    }
                }
            }
        });
        if (probsTap is not null)
        {
            tap($"{p}/self_attn/Softmax_output_0", probsTap);
        }

        if (scoresTap is not null)
        {
            tap($"{p}/self_attn/Mul_1_output_0", scoresTap);
        }

        R(ctx);
        tap($"{p}/self_attn/Reshape_3_output_0", ctx);
        return ctx;
    }

    private static double[] Conv1d(double[] x, int batch, int cin, int len, double[] w, double[] bias, int cout, int kernel, int stride, string order)
    {
        int outLen = (len + 2 - kernel) / stride + 1;
        double[] y = new double[batch * cout * outLen];
        Parallel.For(0, batch * cout, new ParallelOptions { MaxDegreeOfParallelism = 8 }, bo =>
        {
            int b = bo / cout, o = bo % cout;
            for (int t = 0; t < outLen; t++)
            {
                if (order != "f64")
                {
                    // The same sums as MatMul32 over the flattened (channel, tap) index, bias added last.
                    float[] lane = new float[8];
                    float acc32 = 0, comp32 = 0, chunk32 = 0, chunkSum = 0;
                    int idx = 0;
                    for (int c = 0; c < cin; c++)
                    {
                        for (int kk = 0; kk < kernel; kk++, idx++)
                        {
                            int pos32 = (t * stride) + kk - 1;
                            float xv = pos32 >= 0 && pos32 < len ? (float)x[(((b * cin) + c) * len) + pos32] : 0f;
                            float wv = (float)w[(((o * cin) + c) * kernel) + kk];
                            switch (order)
                            {
                                case "lane8":
                                    lane[idx % 8] = MathF.FusedMultiplyAdd(wv, xv, lane[idx % 8]);
                                    break;
                                case "comp":
                                {
                                    float prod = wv * xv;
                                    float prodErr = MathF.FusedMultiplyAdd(wv, xv, -prod);
                                    float sNew = acc32 + prod;
                                    float bb = sNew - acc32;
                                    comp32 += ((acc32 - (sNew - bb)) + (prod - bb)) + prodErr;
                                    acc32 = sNew;
                                    break;
                                }

                                case "chunk64":
                                    chunk32 = MathF.FusedMultiplyAdd(wv, xv, chunk32);
                                    if ((idx + 1) % 64 == 0)
                                    {
                                        chunkSum += chunk32;
                                        chunk32 = 0;
                                    }

                                    break;
                                default:
                                    acc32 = MathF.FusedMultiplyAdd(wv, xv, acc32);
                                    break;
                            }
                        }
                    }

                    float total = order switch
                    {
                        "lane8" => ((lane[0] + lane[1]) + (lane[2] + lane[3])) + ((lane[4] + lane[5]) + (lane[6] + lane[7])),
                        "comp" => acc32 + comp32,
                        "chunk64" => chunkSum + chunk32,
                        _ => acc32,
                    };
                    y[(((b * cout) + o) * outLen) + t] = total + (float)bias[o];
                    continue;
                }

                double s = bias[o];
                for (int c = 0; c < cin; c++)
                {
                    for (int kk = 0; kk < kernel; kk++)
                    {
                        int pos = (t * stride) + kk - 1;
                        if (pos >= 0 && pos < len)
                        {
                            s += w[(((o * cin) + c) * kernel) + kk] * x[(((b * cin) + c) * len) + pos];
                        }
                    }
                }

                y[(((b * cout) + o) * outLen) + t] = s;
            }
        });
        return y;
    }

    private double[] MatMul(string label, double[] a, int m, int k, double[] w, int n)
    {
        if (Emu is not null && Emu.Uses(label))
        {
            return R(MatMul32(a, m, k, w, n, Emu.MatMul));
        }

        return R(MatMul(a, m, k, w, n));
    }

    // Float32 products with the accumulation order of a real implementation; the inputs are float32 values already.
    private static double[] MatMul32(double[] a, int m, int k, double[] w, int n, string order)
    {
        float[] wf = new float[w.Length];
        for (int i = 0; i < wf.Length; i++)
        {
            wf[i] = (float)w[i];
        }

        double[] c = new double[m * n];
        int lanes = 8, chunk = 64;
        Parallel.For(0, m, new ParallelOptions { MaxDegreeOfParallelism = 8 }, i =>
        {
            float[] af = new float[k];
            for (int kk = 0; kk < k; kk++)
            {
                af[kk] = (float)a[(i * k) + kk];
            }

            switch (order)
            {
                case "seq":
                case "seqnf":
                {
                    float[] acc = new float[n];
                    bool fused = order == "seq";
                    for (int kk = 0; kk < k; kk++)
                    {
                        if (fused)
                        {
                            Axpy(acc, af[kk], wf.AsSpan(kk * n, n));
                        }
                        else
                        {
                            for (int j = 0; j < n; j++)
                            {
                                acc[j] += af[kk] * wf[(kk * n) + j];
                            }
                        }
                    }

                    for (int j = 0; j < n; j++)
                    {
                        c[(i * n) + j] = acc[j];
                    }

                    break;
                }

                case "comp":
                {
                    float[] sum = new float[n], carry = new float[n];
                    for (int kk = 0; kk < k; kk++)
                    {
                        float av = af[kk];
                        for (int j = 0; j < n; j++)
                        {
                            float wv = wf[(kk * n) + j];
                            float prod = av * wv;
                            float prodErr = MathF.FusedMultiplyAdd(av, wv, -prod);
                            float acc0 = sum[j];
                            float sNew = acc0 + prod;
                            float bb = sNew - acc0;
                            float addErr = (acc0 - (sNew - bb)) + (prod - bb);
                            sum[j] = sNew;
                            carry[j] += addErr + prodErr;
                        }
                    }

                    for (int j = 0; j < n; j++)
                    {
                        c[(i * n) + j] = sum[j] + carry[j];
                    }

                    break;
                }

                case "lane8":
                {
                    float[][] acc = [.. Enumerable.Range(0, lanes).Select(_ => new float[n])];
                    for (int kk = 0; kk < k; kk++)
                    {
                        Axpy(acc[kk % lanes], af[kk], wf.AsSpan(kk * n, n));
                    }

                    for (int j = 0; j < n; j++)
                    {
                        float s01 = acc[0][j] + acc[1][j], s23 = acc[2][j] + acc[3][j], s45 = acc[4][j] + acc[5][j], s67 = acc[6][j] + acc[7][j];
                        c[(i * n) + j] = (s01 + s23) + (s45 + s67);
                    }

                    break;
                }

                default:
                {
                    float[] total = new float[n];
                    float[] part = new float[n];
                    for (int k0 = 0; k0 < k; k0 += chunk)
                    {
                        Array.Clear(part);
                        for (int kk = k0; kk < Math.Min(k, k0 + chunk); kk++)
                        {
                            Axpy(part, af[kk], wf.AsSpan(kk * n, n));
                        }

                        for (int j = 0; j < n; j++)
                        {
                            total[j] += part[j];
                        }
                    }

                    for (int j = 0; j < n; j++)
                    {
                        c[(i * n) + j] = total[j];
                    }

                    break;
                }
            }
        });
        return c;
    }

    private static void Axpy(float[] acc, float a, ReadOnlySpan<float> w)
    {
        int j = 0;
        var va = new Vector<float>(a);
        for (; j <= acc.Length - Vector<float>.Count; j += Vector<float>.Count)
        {
            Vector.FusedMultiplyAdd(va, new Vector<float>(w.Slice(j)), new Vector<float>(acc, j)).CopyTo(acc, j);
        }

        for (; j < acc.Length; j++)
        {
            acc[j] = MathF.FusedMultiplyAdd(a, w[j], acc[j]);
        }
    }

    private static double[] MatMul(double[] a, int m, int k, double[] w, int n)
    {
        double[] c = new double[m * n];
        Parallel.For(0, m, new ParallelOptions { MaxDegreeOfParallelism = 8 }, i =>
        {
            Span<double> crow = c.AsSpan(i * n, n);
            for (int kk = 0; kk < k; kk++)
            {
                double av = a[(i * k) + kk];
                ReadOnlySpan<double> wrow = w.AsSpan(kk * n, n);
                int j = 0;
                var va = new Vector<double>(av);
                for (; j <= n - Vector<double>.Count; j += Vector<double>.Count)
                {
                    (new Vector<double>(crow.Slice(j)) + (va * new Vector<double>(wrow.Slice(j)))).CopyTo(crow.Slice(j));
                }

                for (; j < n; j++)
                {
                    crow[j] += av * wrow[j];
                }
            }
        });
        return c;
    }

    private void AddBias(double[] x, double[] bias)
    {
        for (int i = 0; i < x.Length; i++)
        {
            x[i] += bias[i % bias.Length];
        }

        R(x);
    }

    private double[] LayerNorm(double[] x, int d, double[] gamma, double[] beta, float epsilon)
    {
        double[] y = new double[x.Length];
        if (Emu is { LayerNorm32: true })
        {
            for (int r = 0; r < x.Length / d; r++)
            {
                float mean32 = 0;
                for (int c = 0; c < d; c++)
                {
                    mean32 += (float)x[(r * d) + c];
                }

                mean32 /= d;
                float var32 = 0;
                for (int c = 0; c < d; c++)
                {
                    float dv = (float)x[(r * d) + c] - mean32;
                    var32 = MathF.FusedMultiplyAdd(dv, dv, var32);
                }

                var32 /= d;
                float inv32 = 1f / MathF.Sqrt(var32 + epsilon);
                for (int c = 0; c < d; c++)
                {
                    y[(r * d) + c] = ((((float)x[(r * d) + c] - mean32) * inv32) * (float)gamma[c]) + (float)beta[c];
                }
            }

            return R(y);
        }

        for (int r = 0; r < x.Length / d; r++)
        {
            double mean = 0;
            for (int c = 0; c < d; c++)
            {
                mean += x[(r * d) + c];
            }

            mean /= d;
            double var = 0;
            for (int c = 0; c < d; c++)
            {
                double dv = x[(r * d) + c] - mean;
                var += dv * dv;
            }

            var /= d;
            double inv = 1.0 / Math.Sqrt(var + (double)epsilon);
            for (int c = 0; c < d; c++)
            {
                y[(r * d) + c] = ((x[(r * d) + c] - mean) * inv * gamma[c]) + beta[c];
            }
        }

        return R(y);
    }

    // x * (erf(x / sqrt(2)) + 1) * 0.5 as the graph spells it, with sqrt(2) as float32 holds it.
    private double[] Gelu(double[] x)
    {
        double[] y = new double[x.Length];
        bool g32 = Emu is { Gelu32: true };
        Parallel.For(0, x.Length / 4096 + 1, new ParallelOptions { MaxDegreeOfParallelism = 8 }, blk =>
        {
            for (int i = blk * 4096; i < Math.Min(x.Length, (blk + 1) * 4096); i++)
            {
                y[i] = g32 ? (float)(((float)x[i] * ((float)Erf(x[i] / Sqrt2AsFloat) + 1f)) * 0.5f) : x[i] * (Erf(x[i] / Sqrt2AsFloat) + 1.0) * 0.5;
            }
        });
        return R(y);
    }

    /// <summary>erf in double to about 1e-16: the all-positive series e^(-x^2) 2/sqrt(pi) sum 2^n x^(2n+1) / (2n+1)!!, which does not cancel.</summary>
    public static double Erf(double x)
    {
        double ax = Math.Abs(x);
        if (ax > 6.5)
        {
            return Math.Sign(x);
        }

        double term = ax, sum = ax, x2 = ax * ax;
        for (int n = 1; n < 400; n++)
        {
            term *= 2 * x2 / ((2 * n) + 1);
            sum += term;
            if (term < sum * 1e-18)
            {
                break;
            }
        }

        double r = 2 / Math.Sqrt(Math.PI) * Math.Exp(-x2) * sum;
        return x < 0 ? -r : r;
    }
}
