using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Cryptography;
using RecursiveAI.OneModel;
using RecursiveAI.OneModel.Gpu;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace GpuReport;

/// <summary>
/// A greedy decode through an encoder-decoder pair whose decoder takes the whole token sequence at each step (no key/value cache):
/// the speech model's reference export. Compares the device with the CPU's exact mode step by step, then times a free-running decode.
/// </summary>
internal static class Greedy
{
    // Whisper's start-of-transcript prompt: <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>.
    private static readonly long[] Prompt = [50258, 50259, 50359, 50363];

    public static int Run(string encoderPath, string decoderPath, int steps, int batch, int cpuThreads, double bound, int timedRuns, string? exportDir = null)
    {
        Tensor mel = Mel(batch);
        float[] hsCpu;
        using (Session enc = Session.Load(encoderPath, new ModelOptions { Mode = ArithmeticMode.Exact }))
        {
            hsCpu = Encode(enc, mel, out _);
        }

        Console.WriteLine($"greedy: {steps} steps from a {Prompt.Length}-token prompt, batch {batch} encoder, decoder {System.IO.Path.GetFileName(decoderPath)}");
        Tensor HiddenStates(float[] hs) => Tensor.FromSpan<float>(hs, Shape.Create(batch, 1500, 768));

        // The reference: the CPU's exact mode decodes greedily; every step's logits at the last position are kept, and the full logits of the
        // steps compared in detail.
        List<long> cpuTokens = [.. Prompt];
        var cpuLogits = new List<float[]>();  // the full logits of every step (sequence length times vocabulary)
        using (Session dec = Session.Load(decoderPath, new ModelOptions { Mode = ArithmeticMode.Exact }))
        using (Tensor hs = HiddenStates(hsCpu))
        {
            for (int s = 0; s < steps; s++)
            {
                Step(dec, cpuTokens, hs, batch, out long next, fullLogits: true, out float[] logits);
                cpuLogits.Add(logits);
                cpuTokens.Add(next);
            }
        }

        Console.WriteLine($"cpu exact tokens: {string.Join(" ", cpuTokens.Skip(Prompt.Length))}");
        if (exportDir is not null)
        {
            // The CPU's encoder output and free-running tokens, for a host that decodes on the device and compares (tools/GpuBrowser).
            System.IO.Directory.CreateDirectory(exportDir);
            CaseFile.Write(System.IO.Path.Combine(exportDir, "case.bin"), "whisper-greedy", batch, steps, new Dictionary<string, Tensor> { ["encoder_hidden_states"] = HiddenStates(hsCpu) },
                new Dictionary<string, float[]> { ["tokens"] = [.. cpuTokens.Select(t => (float)t)] }, new Dictionary<string, int[]> { ["tokens"] = [cpuTokens.Count] });
            Console.WriteLine($"exported {System.IO.Path.Combine(exportDir, "case.bin")}");
        }

        // The CPU's fast mode, for the drift between two correct CPU runs and for timing.
        {
            using Session dec = Session.Load(decoderPath, new ModelOptions { Mode = ArithmeticMode.Fast, IntraOpThreads = cpuThreads });
            using Tensor hs = HiddenStates(hsCpu);
            double worst = 0;
            int agree = 0;
            for (int s = 0; s < steps; s++)
            {
                Step(dec, [.. cpuTokens.Take(Prompt.Length + s)], hs, batch, out long next, fullLogits: true, out float[] logits);
                worst = Math.Max(worst, Scaled(logits, cpuLogits[s]));
                agree += next == cpuTokens[Prompt.Length + s] ? 1 : 0;
            }

            Console.WriteLine($"cpu fast vs cpu exact (teacher-forced): worst scaled error of the last-position logits {worst:E2}, same token {agree}/{steps}");
            double warm = TimeDecode(dec, hs, steps, batch, out _);
            double[] rs = Enumerable.Range(0, timedRuns).Select(r => TimeDecode(dec, hs, steps, batch, out _)).Order().ToArray();
            Console.WriteLine($"cpu fast {cpuThreads} threads: decode {steps} tokens: first pass {warm / steps:F1} ms/token, warm median of {timedRuns} {rs[rs.Length / 2] / steps:F1} ms/token (min {rs[0] / steps:F1})");
        }

        foreach (string modeName in new[] { "fast", "exact" })
        {
            ArithmeticMode mode = modeName == "exact" ? ArithmeticMode.Exact : ArithmeticMode.Fast;
            Console.WriteLine($"-- gpu {modeName}");
            float[] hsGpu;
            using (Session enc = Session.Load(encoderPath, new ModelOptions { Mode = mode, Device = ComputeDevice.Gpu }))
            {
                hsGpu = Encode(enc, mel, out _);
                Console.WriteLine($"encoder hidden states vs cpu exact: scaled error {Scaled(hsGpu, hsCpu):E2}");
            }

            Session gpu = Session.Load(decoderPath, new ModelOptions { Mode = mode, Device = ComputeDevice.Gpu });
            using (Tensor hs = HiddenStates(hsCpu))
            {
                // Teacher forced: the same prefix as the CPU at every step.
                double worst = 0, first = 0, at32 = 0;
                int agree = 0;
                for (int s = 0; s < steps; s++)
                {
                    Step(gpu, [.. cpuTokens.Take(Prompt.Length + s)], hs, batch, out long next, fullLogits: true, out float[] full);
                    double e = Scaled(full, cpuLogits[s]);
                    worst = Math.Max(worst, e);
                    if (s == 0) { first = e; }
                    if (s == 32) { at32 = e; }
                    agree += next == cpuTokens[Prompt.Length + s] ? 1 : 0;
                }

                Console.WriteLine($"decoder teacher-forced on the cpu's tokens: scaled error of the full logits first step {first:E2}, step 32 {at32:E2}, worst of {steps} {worst:E2} ({(worst <= bound ? "within" : "ABOVE")} {bound:E0}); same token as cpu {agree}/{steps}");
            }

            // Free running on the device's own encoder output and tokens: the whole pipeline.
            using (Tensor hs = HiddenStates(hsGpu))
            {
                List<long> a = [.. Prompt], b = [.. Prompt];
                var hashA = SHA256.Create();
                for (int s = 0; s < steps; s++)
                {
                    Step(gpu, a, hs, batch, out long n1, fullLogits: true, out float[] fa);
                    hashA.TransformBlock(System.Runtime.InteropServices.MemoryMarshal.AsBytes(fa.AsSpan()).ToArray(), 0, fa.Length * 4, null, 0);
                    a.Add(n1);
                }

                hashA.TransformFinalBlock([], 0, 0);
                var hashB = SHA256.Create();
                for (int s = 0; s < steps; s++)
                {
                    Step(gpu, b, hs, batch, out long n2, fullLogits: true, out float[] fb);
                    hashB.TransformBlock(System.Runtime.InteropServices.MemoryMarshal.AsBytes(fb.AsSpan()).ToArray(), 0, fb.Length * 4, null, 0);
                    b.Add(n2);
                }

                hashB.TransformFinalBlock([], 0, 0);
                bool sameSeq = a.SequenceEqual(cpuTokens);
                Console.WriteLine($"gpu free-running tokens {(sameSeq ? "equal" : "DIFFER from")} cpu exact's ({a.Zip(cpuTokens).Count(t => t.First == t.Second) - Prompt.Length}/{steps} equal positions); two runs: tokens {(a.SequenceEqual(b) ? "identical" : "DIFFER")}, logits hashes {Convert.ToHexString(hashA.Hash!)[..16]}/{Convert.ToHexString(hashB.Hash!)[..16]} ({(hashA.Hash!.AsSpan().SequenceEqual(hashB.Hash!) ? "bit-identical" : "DIFFER")})");

                double firstPass = TimeDecode(gpu, hs, steps, batch, out _);
                double[] rs = Enumerable.Range(0, timedRuns).Select(r => TimeDecode(gpu, hs, steps, batch, out _)).Order().ToArray();
                GpuSessionInfo info = OneModelGpu.GetInfo(gpu);
                Console.WriteLine($"gpu {modeName} decode {steps} tokens: first pass {firstPass / steps:F1} ms/token, warm median of {timedRuns} {rs[rs.Length / 2] / steps:F1} ms/token (min {rs[0] / steps:F1}); arena {info.ArenaBytes / 1048576.0:F1} MiB, device allocations {info.DeviceAllocations}");
                {
                    // A step with a new sequence length builds a plan (and the device constants it folds); a repeat of it only runs.
                    var fresh = new List<double>();
                    var repeat = new List<double>();
                    GpuSessionInfo before = OneModelGpu.GetInfo(gpu);
                    var gpuBackend = (GpuBackend)gpu.Backend!;
                    int groupsBefore = Counters.BindGroups(gpuBackend), groupsRepeat = 0;
                    for (int s = 0; s < steps; s++)
                    {
                        using Tensor idsFresh = Ids([.. b.Take(Prompt.Length + 1 + s)], batch);
                        var feedFresh = new Dictionary<string, Tensor> { ["input_ids"] = idsFresh, ["encoder_hidden_states"] = hs };
                        var w = Stopwatch.StartNew();
                        using (gpu.Run(feedFresh)) { }
                        fresh.Add(w.Elapsed.TotalMilliseconds);
                        int g0 = Counters.BindGroups(gpuBackend);
                        w.Restart();
                        using (gpu.Run(feedFresh)) { }
                        repeat.Add(w.Elapsed.TotalMilliseconds);
                        groupsRepeat += Counters.BindGroups(gpuBackend) - g0;
                    }

                    GpuSessionInfo after = OneModelGpu.GetInfo(gpu);
                    fresh.Sort();
                    repeat.Sort();
                    Console.WriteLine($"decoder step wall: first run of a new length median {fresh[fresh.Count / 2]:F1} ms (min {fresh[0]:F1}), repeat of the same length median {repeat[repeat.Count / 2]:F1} ms (min {repeat[0]:F1}); device allocations over {steps} new lengths {after.DeviceAllocations - before.DeviceAllocations} ({(after.DeviceAllocations - before.DeviceAllocations) / (double)steps:F1} per step); bind groups built over the new lengths {Counters.BindGroups(gpuBackend) - groupsBefore - groupsRepeat}, by the {steps} repeats of a length {groupsRepeat}; cached plans {gpu.CachedPlanCount}");
                }

                using Tensor ids = Ids(b.Take(Prompt.Length + steps).ToList(), batch);
                SessionDiagnostics d = gpu.GetDiagnostics(new Dictionary<string, Tensor> { ["input_ids"] = ids, ["encoder_hidden_states"] = hs });
                Console.WriteLine($"last step plan: {d.NodesOnDevice} of {d.Nodes.Length} nodes on gpu, {d.DeviceRegions} regions, to gpu {d.BytesToDevicePerRun / 1024.0:F1} KiB, from gpu {d.BytesFromDevicePerRun / 1024.0:F1} KiB per step");
            }

            string[] validation = ((GpuBackend)gpu.Backend!).Device.TakeValidationMessages();
            var backend = (GpuBackend)gpu.Backend!;
            gpu.Dispose();
            Console.WriteLine($"validation messages: {validation.Length}{(validation.Length > 0 ? " first: " + validation[0] : "")}; after dispose: {backend.MessagesAfterDispose.Length}");
        }

        mel.Dispose();
        return 0;
    }

    private static Tensor Ids(List<long> tokens, int batch)
    {
        long[] ids = new long[batch * tokens.Count];
        for (int b = 0; b < batch; b++)
        {
            tokens.CopyTo(ids, b * tokens.Count);
        }

        return Tensor.FromSpan<long>(ids, Shape.Create(batch, tokens.Count));
    }

    private static Tensor Mel(int batch)
    {
        float[] values = new float[batch * 80 * 3000];
        var rng = new Random(4242);
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = 0.5f * (float)(Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble()));
        }

        return Tensor.FromSpan<float>(values, Shape.Create(batch, 80, 3000));
    }

    private static float[] Encode(Session enc, Tensor mel, out double ms)
    {
        var sw = Stopwatch.StartNew();
        using SessionOutputs o = enc.Run(new Dictionary<string, Tensor> { [enc.Inputs[0].Name] = mel });
        ms = sw.Elapsed.TotalMilliseconds;
        return o[enc.Outputs[0].Name].Data<float>().ToArray();
    }

    private static float[] Step(Session dec, List<long> tokens, Tensor hs, int batch, out long next) => Step(dec, tokens, hs, batch, out next, false, out _);

    private static float[] Step(Session dec, List<long> tokens, Tensor hs, int batch, out long next, bool fullLogits, out float[] full)
    {
        using Tensor ids = Ids(tokens, batch);
        using SessionOutputs o = dec.Run(new Dictionary<string, Tensor> { ["input_ids"] = ids, ["encoder_hidden_states"] = hs });
        ReadOnlySpan<float> all = o["logits"].Data<float>();
        int vocab = all.Length / (batch * tokens.Count);
        ReadOnlySpan<float> last = all.Slice((tokens.Count - 1) * vocab, vocab);
        int best = 0;
        for (int i = 1; i < vocab; i++)
        {
            if (last[i] > last[best])
            {
                best = i;
            }
        }

        next = best;
        full = fullLogits ? all.ToArray() : [];
        return fullLogits ? full : last.ToArray();
    }

    private static double TimeDecode(Session dec, Tensor hs, int steps, int batch, out List<long> tokens)
    {
        tokens = [.. Prompt];
        var sw = Stopwatch.StartNew();
        for (int s = 0; s < steps; s++)
        {
            using Tensor ids = Ids(tokens, batch);
            using SessionOutputs o = dec.Run(new Dictionary<string, Tensor> { ["input_ids"] = ids, ["encoder_hidden_states"] = hs });
            ReadOnlySpan<float> all = o["logits"].Data<float>();
            int vocab = all.Length / (batch * tokens.Count);
            ReadOnlySpan<float> last = all.Slice((tokens.Count - 1) * vocab, vocab);
            int best = 0;
            for (int i = 1; i < vocab; i++)
            {
                if (last[i] > last[best])
                {
                    best = i;
                }
            }

            tokens.Add(best);
        }

        return sw.Elapsed.TotalMilliseconds;
    }

    private static double Scaled(float[] a, float[] r)
    {
        double max = 0, diff = 0;
        for (int i = 0; i < a.Length; i++)
        {
            max = Math.Max(max, Math.Abs(r[i]));
            diff = Math.Max(diff, Math.Abs((double)a[i] - r[i]));
        }

        return max == 0 ? diff : diff / max;
    }
}
