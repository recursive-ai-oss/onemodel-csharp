using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using RecursiveAI.OneModel;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace GpuReport;

/// <summary>
/// Error attribution along a model: every executor (CPU fast, CPU exact, GPU fast, GPU exact) against a float64 evaluation of the same
/// graph, at every tapped intermediate value. The float32 and float64 files come from tools/to_fp64.py with the same --taps pattern.
/// The metric is max|x - r| / max|r| over the tensor (and over each batch item), the one the parity gate uses; the root-mean-square
/// relative error is printed as well, because the maximum of a larger batch grows with the number of elements even when no
/// element gets worse.
/// </summary>
internal static class Attribution
{
    public static int Run(string model32, string model64, int batch, string[] variants, double inputScale, int seed, bool perItem, string? worst = null, int[]? stageLayers = null, int sinkRows = 0)
    {
        Dictionary<string, Tensor> feed32 = [];
        Dictionary<string, Tensor> feed64 = [];
        using (Session probe = Session.Load(model32, new ModelOptions { Mode = ArithmeticMode.Exact }))
        {
            foreach (TensorInfo info in probe.Inputs)
            {
                int[] dims = [.. info.Dims.Select(d => d.Param == "batch" ? batch : d.IsFixed ? (int)d.Value : 1)];
                if (dims.Length > 0 && !info.Dims[0].IsFixed)
                {
                    dims[0] = batch;
                }

                int len = dims.Aggregate(1, (a, b) => a * b);
                var rng = new Random(seed + info.Name.Length);
                float[] v = new float[len];
                double[] d64 = new double[len];
                for (int i = 0; i < len; i++)
                {
                    v[i] = (float)(inputScale * Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble()));
                    d64[i] = v[i];
                }

                feed32[info.Name] = Tensor.FromSpan<float>(v, Shape.Create(dims));
                feed64[info.Name] = Tensor.FromSpan<double>(d64, Shape.Create(dims));
            }
        }

        Dictionary<string, double[]> reference;
        Dictionary<string, int[]> shapes = [];
        List<string> order = [];
        using (Session probe = Session.Load(model32, new ModelOptions { Mode = ArithmeticMode.Exact }))
        {
            foreach (TensorInfo info in probe.Outputs)
            {
                order.Add(info.Name);
            }
        }

        var oracle = new Whisper64(model64, order);
        var watch = System.Diagnostics.Stopwatch.StartNew();
        reference = oracle.Run(feed64.Values.Single().Data<double>().ToArray(), batch);
        Console.WriteLine($"float64 reference: {watch.Elapsed.TotalSeconds:F1} s");
        foreach (string name in order)
        {
            if (!reference.ContainsKey(name))
            {
                throw new InvalidOperationException("no float64 value for tap " + name);
            }

            shapes[name] = [batch];
        }

        var results = new Dictionary<string, Dictionary<string, float[]>>();
        foreach (string variant in variants)
        {
            ModelOptions options = OptionsFor(variant);
            using Session s = Session.Load(model32, options);
            using SessionOutputs o = s.Run(feed32);
            var r = new Dictionary<string, float[]>();
            foreach (string name in order)
            {
                r[name] = o[name].Data<float>().ToArray();
            }

            results[variant] = r;
        }

        Console.WriteLine($"batch={batch} inputScale={inputScale.ToString(CultureInfo.InvariantCulture)} seed={seed}; scaled error max|x-r|/max|r| (rms relative error rms(x-r)/rms(r)) against float64 of the same graph");
        Console.WriteLine("tap".PadRight(58) + string.Concat(variants.Select(v => v.PadLeft(26))));
        foreach (string name in order)
        {
            double[] r = reference[name];
            int[] dims = shapes[name];
            string shortName = name.Replace("/e/", "").Replace("_output_0", "");
            if (shortName.Length > 56)
            {
                shortName = shortName[^56..];
            }

            string line = shortName.PadRight(58);
            foreach (string variant in variants)
            {
                (double scaled, double rms) = Measure(results[variant][name], r, 0, r.Length);
                line += $"{scaled:E2} ({rms:E1})".PadLeft(26);
            }

            Console.WriteLine(line);
            if (perItem && batch > 1 && dims.Length > 0 && dims[0] == batch)
            {
                int per = r.Length / batch;
                foreach (string variant in variants)
                {
                    string items = string.Join(" ", Enumerable.Range(0, batch).Select(b => Measure(results[variant][name], r, b * per, per).Scaled.ToString("E1", CultureInfo.InvariantCulture)));
                    Console.WriteLine($"    {variant,-10} per item: {items}");
                }
            }
        }

        if (worst is not null)
        {
            var rx = new System.Text.RegularExpressions.Regex(worst);
            Console.WriteLine("worst element per tap: index (row,channel) reference value, error, [max |reference| of the tensor]");
            foreach (string name in order.Where(n => rx.IsMatch(n)))
            {
                double[] r = reference[name];
                int channels = name.Contains("fc1/") || name.Contains("activation_fn") ? 3072 : 768;
                if (name.Contains("conv") || name.Contains("Mul_3"))
                {
                    channels = 1500;
                }

                double maxAbs = r.Max(Math.Abs);
                foreach (string variant in variants)
                {
                    float[] x = results[variant][name];
                    int at = 0;
                    double bestErr = -1;
                    for (int i = 0; i < r.Length; i++)
                    {
                        double e = Math.Abs(x[i] - r[i]);
                        if (e > bestErr)
                        {
                            bestErr = e;
                            at = i;
                        }
                    }

                    Console.WriteLine($"  {name.Replace("/e/", "").Replace("_output_0", ""),-52} {variant,-10} [{at / channels},{at % channels}] ref {r[at]:G6} err {bestErr:E2} (max|ref| {maxAbs:G5})");
                }
            }
        }

        if (sinkRows > 0 && reference.TryGetValue("/e/layers.11/Add_1_output_0", out double[]? lastStream))
        {
            // The tokens that carry the huge activations are few; their errors drive the end result. Rows are those with the largest
            // |value| in the last layer's stream (batch item 0); every tap with a [rows, channels] layout is split into those rows and the rest.
            int tt = lastStream.Length / batch / 768;
            int[] rowsSel = [.. Enumerable.Range(0, tt).OrderByDescending(r => Enumerable.Range(0, 768).Max(c => Math.Abs(lastStream[(r * 768) + c]))).Take(sinkRows)];
            Console.WriteLine("sink rows (batch item 0): " + string.Join(",", rowsSel) + "; per tap: max error on those rows / max error elsewhere, both over max|reference| of the tensor");
            foreach (string name in order)
            {
                double[] r = reference[name];
                int cw = name.Contains("fc1/") || name.Contains("activation_fn") ? 3072 : 768;
                if (r.Length != batch * tt * cw)
                {
                    continue;
                }

                double maxAbs = r.Max(Math.Abs);
                string line = name.Replace("/e/", "").Replace("_output_0", "").PadRight(50);
                foreach (string variant in variants)
                {
                    double eSink = 0, eRest = 0;
                    float[] x = results[variant][name];
                    for (int row = 0; row < tt; row++)
                    {
                        double e = 0;
                        for (int c = 0; c < cw; c++)
                        {
                            e = Math.Max(e, Math.Abs(x[(row * cw) + c] - r[(row * cw) + c]));
                        }

                        if (rowsSel.Contains(row))
                        {
                            eSink = Math.Max(eSink, e);
                        }
                        else
                        {
                            eRest = Math.Max(eRest, e);
                        }
                    }

                    line += $" | {variant} {eSink / maxAbs:E1}/{eRest / maxAbs:E1}";
                }

                Console.WriteLine(line);
            }
        }

        if (stageLayers is not null)
        {
            // Each executor's stage error alone: the stage runs in float64 from that executor's own float32 input, and its own output is
            // compared with that. What is left over in the in-situ table is carried in from earlier stages.
            Console.WriteLine("stage error from each executor's own input (scaled by max|reference|; [row,channel] of the worst element and its reference value)");
            foreach (int l in stageLayers)
            {
                string pf = $"/e/layers.{l}";
                string prevTap = l == 0 ? "/e/Add_2_output_0" : $"/e/layers.{l - 1}/Add_1_output_0";
                string ln1 = $"{pf}/self_attn_layer_norm/LayerNormalization_output_0", ln2 = $"{pf}/final_layer_norm/LayerNormalization_output_0";
                string fc1 = $"{pf}/fc1/Add_output_0", act = $"{pf}/activation_fn/Mul_1_output_0";
                (string Kind, string In, string Out, int Channels)[] stages =
                [
                    ("ln1", prevTap, ln1, 768), ("attn", ln1, $"{pf}/self_attn/out_proj/Add_output_0", 768), ("ln2", $"{pf}/Add_output_0", ln2, 768),
                    ("fc1", ln2, fc1, 3072), ("gelu", fc1, act, 3072), ("fc2", act, $"{pf}/fc2/Add_output_0", 768),
                ];
                foreach ((string kind, string tin, string tout, int channels) in stages)
                {
                    string line = $"  L{l}_{kind,-5}";
                    foreach (string variant in variants)
                    {
                        double[] input = results[variant][tin].Select(x => (double)x).ToArray();
                        double[] want = oracle.Stage(kind, l, input, batch);
                        float[] got = results[variant][tout];
                        (double scaled, _) = Measure(got, want, 0, want.Length);
                        int at = 0;
                        double bestErr = -1;
                        for (int i = 0; i < want.Length; i++)
                        {
                            double e = Math.Abs(got[i] - want[i]);
                            if (e > bestErr)
                            {
                                bestErr = e;
                                at = i;
                            }
                        }

                        // The three rows with the largest error: where the error sits matters when a few tokens carry huge activations.
                        double[] rowErr = new double[want.Length / channels];
                        for (int i = 0; i < want.Length; i++)
                        {
                            rowErr[i / channels] = Math.Max(rowErr[i / channels], Math.Abs(got[i] - want[i]));
                        }

                        string rows = string.Join(",", Enumerable.Range(0, rowErr.Length).OrderByDescending(r => rowErr[r]).Take(3).Select(r => $"{r % (rowErr.Length / batch)}:{rowErr[r]:E1}"));
                        line += $" | {variant} {scaled:E2} [{at / channels},{at % channels}] {want[at]:G4} rows {rows}";
                    }

                    Console.WriteLine(line);
                }
            }
        }

        // The gate metric of the last output against the CPU's exact mode, for a direct comparison with the usual rows.
        string last = order[0];
        if (results.TryGetValue("cpu-exact", out Dictionary<string, float[]>? exactResults))
        {
            foreach (string variant in variants.Where(v => v.StartsWith("gpu", StringComparison.Ordinal)))
            {
                double[] ex = exactResults[last].Select(x => (double)x).ToArray();
                (double scaled, _) = Measure(results[variant][last], ex, 0, ex.Length);
                Console.WriteLine($"{variant} vs cpu-exact on {last}: {scaled:E2}");
            }
        }

        return 0;
    }

    /// <summary>
    /// Stage by stage: each probe model (tools/make_probes.py) runs from the correctly rounded float32 copy of the float64 value that
    /// enters the stage, so the printed error is that stage's alone, with nothing accumulated before it.
    /// </summary>
    public static int Probes(string dir, string model, int batch, string[] variants, double inputScale, int seed, bool perItem)
    {
        string[][] specs = [.. System.IO.File.ReadAllLines(System.IO.Path.Combine(dir, "manifest.txt")).Where(l => l.Length > 0).Select(l => l.Split(' '))];
        var rng = new Random(seed + "input_features".Length);
        int len = batch * 80 * 3000;
        double[] mel = new double[len];
        for (int i = 0; i < len; i++)
        {
            mel[i] = (float)(inputScale * Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble()));
        }

        var oracle = new Whisper64(model, specs.SelectMany(s => new[] { s[1], s[2] }));
        Dictionary<string, double[]> taps = oracle.Run(mel, batch);
        Console.WriteLine($"batch={batch} inputScale={inputScale.ToString(CultureInfo.InvariantCulture)} seed={seed}; one stage at a time, from the rounded float64 input; scaled error (rms relative) against float64");
        Console.WriteLine("scaled error vs the float64 stage on the rounded input (rms relative) [scaled error vs the float64 stage on the unrounded input]");
        Console.WriteLine("probe".PadRight(14) + string.Concat(variants.Select(v => v.PadLeft(36))) + "emu-seq".PadLeft(14) + "emu-lane8".PadLeft(14));
        foreach (string[] spec in specs)
        {
            double[] inD = taps[spec[1]];
            double[] want = taps[spec[2]];
            double[] wantRounded;
            int channels = spec[1].Contains("fc1/Add") || spec[1].Contains("activation_fn") ? 3072 : 768;
            if (spec[1].Contains("activation_fn"))
            {
                channels = 3072;
            }

            float[] inF = new float[inD.Length];
            for (int i = 0; i < inF.Length; i++)
            {
                inF[i] = (float)inD[i];
            }

            // The reference twice: from the float64 value (what the stage should produce), and from the float32-rounded input the
            // executors get (what they can be asked for). The second takes the input rounding out of the comparison.
            string[] fileParts = spec[0].Replace(".onnx", "").Split('_');
            wantRounded = oracle.Stage(fileParts[1], int.Parse(fileParts[0][1..], CultureInfo.InvariantCulture), inF.Select(x => (double)x).ToArray(), batch);
            var feed = new Dictionary<string, Tensor> { [spec[1]] = Tensor.FromSpan<float>(inF, Shape.Create(batch, inD.Length / (batch * channels), channels)) };
            string line = spec[0].Replace(".onnx", "").PadRight(14);
            var itemLines = new List<string>();
            var outputs = new Dictionary<string, float[]>();
            foreach (string variant in variants)
            {
                ModelOptions options = OptionsFor(variant);
                using Session s = Session.Load(System.IO.Path.Combine(dir, spec[0]), options);
                using SessionOutputs o = s.Run(feed);
                float[] got = o[spec[2]].Data<float>().ToArray();
                outputs[variant] = got;
                (double scaled, double rms) = Measure(got, wantRounded, 0, want.Length);
                (double scaledTrue, _) = Measure(got, want, 0, want.Length);
                line += $"{scaled:E2} ({rms:E1}) [{scaledTrue:E1}]".PadLeft(36);
                if (perItem && batch > 1)
                {
                    int per = want.Length / batch;
                    itemLines.Add($"    {variant,-10} per item: " + string.Join(" ", Enumerable.Range(0, batch).Select(b => Measure(got, wantRounded, b * per, per).Scaled.ToString("E1", CultureInfo.InvariantCulture))));
                }
            }

            // The same stage with the products summed in float32 as an implementation would: one chain in increasing k, or eight lanes.
            string[] stageParts = spec[0].Replace(".onnx", "").Split('_');
            int stageLayer = int.Parse(stageParts[0][1..], CultureInfo.InvariantCulture);
            foreach (string emu in new[] { "r+seq", "r+lane8" })
            {
                oracle.Emu = Emulation.Parse(emu);
                double[] emulated = oracle.Stage(stageParts[1], stageLayer, inF.Select(x => (double)x).ToArray(), batch);
                oracle.Emu = null;
                line += $"{MeasureD(emulated, wantRounded, 0, want.Length):E2}".PadLeft(14);
            }

            if (outputs.Count > 1)
            {
                string[] names = [.. outputs.Keys];
                double[] first = outputs[names[0]].Select(x => (double)x).ToArray();
                line += "  | " + string.Join(" ", names.Skip(1).Select(n => $"{names[0]}~{n}: {Measure(outputs[n], first, 0, first.Length).Scaled:E1}"));
            }

            Console.WriteLine(line);
            foreach (string il in itemLines)
            {
                Console.WriteLine(il);
            }
        }

        return 0;
    }

    /// <summary>
    /// Which float32 effects produce the model's error: the float64 run and runs that round or accumulate in float32 in chosen places
    /// (<see cref="Emulation"/>), compared with it on the final output, per batch item.
    /// </summary>
    public static int Emulate(string model, int batch, int[] seeds, string[] configs, double inputScale)
    {
        var oracle = new Whisper64(model, ["encoder_hidden_states"]);
        foreach (int seed in seeds)
        {
            var rng = new Random(seed + "input_features".Length);
            double[] mel = new double[batch * 80 * 3000];
            for (int i = 0; i < mel.Length; i++)
            {
                mel[i] = (float)(inputScale * Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble()));
            }

            oracle.Emu = null;
            double[] want = oracle.Run(mel, batch)["encoder_hidden_states"];
            Console.WriteLine($"### seed {seed} batch {batch}: scaled error of the final output against the float64 run, per item");
            foreach (string cfg in configs)
            {
                oracle.Emu = Emulation.Parse(cfg);
                double[] got = oracle.Run(mel, batch)["encoder_hidden_states"];
                int per = want.Length / batch;
                double[] items = [.. Enumerable.Range(0, batch).Select(b => MeasureD(got, want, b * per, per))];
                Console.WriteLine($"{cfg,-34} " + string.Join(" ", items.Select(v => v.ToString("E1", CultureInfo.InvariantCulture))) + $"   mean {items.Average():E1}");
            }
        }

        return 0;
    }

    private static double MeasureD(double[] x, double[] r, int start, int count)
    {
        double max = 0, diff = 0;
        for (int i = start; i < start + count; i++)
        {
            max = Math.Max(max, Math.Abs(r[i]));
            diff = Math.Max(diff, Math.Abs(x[i] - r[i]));
        }

        return diff / max;
    }

    // "gpu-fast", "gpu-exact" or "cpu-fast", "cpu-exact", optionally with "+nofuse", "+noepi", "+unfusedattn", "+unfusedgelu" after a gpu name.
    private static ModelOptions OptionsFor(string variant)
    {
        string[] parts = variant.Split('+');
        bool gpu = parts[0].StartsWith("gpu", StringComparison.Ordinal);
        ArithmeticMode mode = parts[0].EndsWith("exact", StringComparison.Ordinal) ? ArithmeticMode.Exact : ArithmeticMode.Fast;
        if (!gpu)
        {
            return new ModelOptions { Mode = mode };
        }

        RecursiveAI.OneModel.Gpu.OneModelGpu.Register(new RecursiveAI.OneModel.Gpu.GpuOptions
        {
            FuseOperators = !parts.Contains("nofuse"),
            FuseEpilogues = !parts.Contains("noepi"),
            KeepAttentionUnfused = parts.Contains("unfusedattn"),
            KeepGeluUnfused = parts.Contains("unfusedgelu"),
        });
        return new ModelOptions { Mode = mode, Device = ComputeDevice.Gpu };
    }

    private static (double Scaled, double Rms) Measure(float[] x, double[] r, int start, int count)
    {
        double max = 0, diff = 0, sumR = 0, sumD = 0;
        for (int i = start; i < start + count; i++)
        {
            double d = x[i] - r[i];
            max = Math.Max(max, Math.Abs(r[i]));
            diff = Math.Max(diff, Math.Abs(d));
            sumR += r[i] * r[i];
            sumD += d * d;
        }

        return (max == 0 ? diff : diff / max, sumR == 0 ? Math.Sqrt(sumD / count) : Math.Sqrt(sumD / sumR));
    }
}
