using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.JavaScript;
using System.Text.Json;
using System.Threading.Tasks;
using RecursiveAI.OneModel;
using RecursiveAI.OneModel.Gpu;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace GpuBrowser;

/// <summary>What the page calls. One case is one model on one adapter in one arithmetic mode.</summary>
[System.Runtime.Versioning.SupportedOSPlatform("browser")]
public static partial class Host
{
    private static bool s_registered;
    private static bool s_fuse = true;
    private static bool? s_attention;
    private static bool s_pair;
    private static int? s_pipe;

    // fuse=false turns every fusion off; attention=fused|unfused overrides the platform's choice for attention alone.
    private static void SetFusion(JsonElement cfg)
    {
        s_fuse = !cfg.TryGetProperty("fuse", out JsonElement f) || f.GetBoolean();
        s_pipe = cfg.TryGetProperty("pipe", out JsonElement pi) ? pi.GetInt32() : null;
        s_pair = cfg.TryGetProperty("pair", out JsonElement p) && p.GetBoolean();
        s_attention = cfg.TryGetProperty("attention", out JsonElement a) ? a.GetString() switch { "fused" => true, "unfused" => false, _ => null } : null;
    }

    /// <summary>
    /// Runs a model on the GPU and reports what happened as JSON. <paramref name="config"/> holds <c>mode</c> (fast or exact), <c>runs</c>
    /// (timed warm runs), <c>warmups</c> and <c>label</c>. <paramref name="model"/> is the model file, <paramref name="caseFile"/> the inputs and
    /// reference outputs the desktop wrote (see tools/GpuReport).
    /// </summary>
    [JSExport]
    public static async Task<string> RunCase(string config, byte[] model, byte[] caseFile)
    {
        using JsonDocument cfg = JsonDocument.Parse(config);
        string label = cfg.RootElement.TryGetProperty("label", out JsonElement l) ? l.GetString() ?? "" : "";
        ArithmeticMode mode = cfg.RootElement.TryGetProperty("mode", out JsonElement m) && m.GetString() == "exact" ? ArithmeticMode.Exact : ArithmeticMode.Fast;
        int runs = cfg.RootElement.TryGetProperty("runs", out JsonElement r) ? r.GetInt32() : 10;
        int warmups = cfg.RootElement.TryGetProperty("warmups", out JsonElement w) ? w.GetInt32() : 3;
        SetFusion(cfg.RootElement);
        var result = new Dictionary<string, object?> { ["label"] = label, ["mode"] = mode.ToString().ToLowerInvariant() };
        try
        {
            await RunInto(result, mode, runs, warmups, model, caseFile).ConfigureAwait(true);
        }
        catch (Exception e)
        {
            result["error"] = e.GetType().Name + ": " + e.Message;
            result["stack"] = e.StackTrace;
            if (e is SessionException se)
            {
                result["errorKind"] = se.Kind.ToString();
                result["errorOperators"] = se.Operators.ToArray();
            }
        }

        return ToJson(result);
    }

    private static async Task RunInto(Dictionary<string, object?> result, ArithmeticMode mode, int runs, int warmups, byte[] modelBytes, byte[] caseFile)
    {
        if (!s_registered)
        {
            OneModelGpu.Register(new GpuOptions { CollectTimings = true, FuseOperators = s_fuse, FusedAttention = s_attention, PipelineFirstDispatches = s_pipe });
            s_registered = true;
        }

        Case c = Case.Parse(caseFile);
        var inputs = new Dictionary<string, Tensor>();
        foreach (Case.Entry e in c.Inputs)
        {
            inputs[e.Name] = e.DType == "i64"
                ? Tensor.FromSpan<long>(MemoryMarshal.Cast<byte, long>(c.Blob(e)).ToArray(), Shape.Create(e.Dims))
                : Tensor.FromSpan<float>(MemoryMarshal.Cast<byte, float>(c.Blob(e)).ToArray(), Shape.Create(e.Dims));
        }

        // RequireGpu first: it names every operator the device cannot take. A refusal is a result, not a failure of the case.
        Session? session = null;
        var watch = Stopwatch.StartNew();
        await OneModelGpu.PrepareDeviceAsync().ConfigureAwait(true);
        try
        {
            session = Session.Load(modelBytes, new ModelOptions { Device = ComputeDevice.Gpu, RequireGpu = true, Mode = mode });
            result["requireGpu"] = "accepted at creation";
        }
        catch (SessionException e)
        {
            result["requireGpu"] = $"refused: {e.Kind}; operators [{string.Join(", ", e.Operators)}]; {Trim(e.Message)}";
        }

        if (session is null)
        {
            await OneModelGpu.PrepareDeviceAsync().ConfigureAwait(true);
            session = Session.Load(modelBytes, new ModelOptions { Device = ComputeDevice.Gpu, Mode = mode });
        }

        using (session)
        {
            result["createMs"] = Math.Round(watch.Elapsed.TotalMilliseconds, 1);
            GpuSessionInfo info = OneModelGpu.GetInfo(session);
            result["adapter"] = info.Adapter;
            result["api"] = info.Api;
            result["kernelVariants"] = info.KernelVariants.ToArray();

            watch.Restart();
            Dictionary<string, float[]> first;
            try
            {
                using SessionOutputs o = await OneModelGpu.RunAsync(session, inputs).ConfigureAwait(true);
                first = Collect(session, o);
            }
            catch (SessionException e) when (e.Kind == SessionErrorKind.DeviceUnsupported && result["requireGpu"] is string r && r.StartsWith("accepted", StringComparison.Ordinal))
            {
                // Some refusals depend on shapes and only surface when the plan is made, at the first run.
                result["requireGpu"] = $"refused at the first run: {e.Kind}; operators [{string.Join(", ", e.Operators)}]; {Trim(e.Message)}";
                throw;
            }

            result["firstRunMs"] = Math.Round(watch.Elapsed.TotalMilliseconds, 1);
            SessionDiagnostics d = session.GetDiagnostics(inputs);
            result["nodes"] = d.Nodes.Length;
            result["nodesOnDevice"] = d.NodesOnDevice;
            result["regions"] = d.DeviceRegions;
            result["toDeviceBytes"] = d.BytesToDevicePerRun;
            result["fromDeviceBytes"] = d.BytesFromDevicePerRun;
            result["fallbacks"] = d.Nodes.Where(n => n.Region < 0).Take(40).Select(n => $"{n.OpType} {n.Name}: {n.FallbackReason}").ToArray();
            result["fallbackCount"] = d.Nodes.Count(n => n.Region < 0);

            // Parity against the desktop CPU exact mode.
            var parity = new Dictionary<string, double>();
            double worst = 0;
            foreach (Case.Entry e in c.Outputs)
            {
                ReadOnlySpan<float> expected = MemoryMarshal.Cast<byte, float>(c.Blob(e));
                double s = Scaled(first[e.Name], expected);
                parity[e.Name] = s;
                worst = Math.Max(worst, s);
            }

            result["parityScaled"] = parity;
            result["worstScaled"] = worst;
            result["tokens"] = Greedy(c, first);

            // Repeatability: the second run's bits against the first's.
            Dictionary<string, float[]> second;
            using (SessionOutputs o = await OneModelGpu.RunAsync(session, inputs).ConfigureAwait(true))
            {
                second = Collect(session, o);
            }

            string h1 = Hash(first), h2 = Hash(second);
            result["hashRun1"] = h1;
            result["hashRun2"] = h2;
            result["bitIdentical"] = h1 == h2;

            // Two devices in one page: a second session of the model lives beside the first, each on its own device. Both run, alternately, and
            // every output must carry the first session's bits.
            if (s_pair)
            {
                await OneModelGpu.PrepareDeviceAsync().ConfigureAwait(true);
                using Session other = Session.Load(modelBytes, new ModelOptions { Device = ComputeDevice.Gpu, RequireGpu = true, Mode = mode });
                bool same = true;
                for (int i = 0; i < 3; i++)
                {
                    foreach (Session s in i % 2 == 0 ? new[] { other, session } : new[] { session, other })
                    {
                        using SessionOutputs o = await OneModelGpu.RunAsync(s, inputs).ConfigureAwait(true);
                        same &= Hash(Collect(s, o)) == h1;
                    }
                }

                result["pairBitIdentical"] = same;
                result["pairSessions"] = 2;
            }

            for (int i = 0; i < warmups; i++)
            {
                using SessionOutputs o = await OneModelGpu.RunAsync(session, inputs).ConfigureAwait(true);
            }

            OneModelGpu.ResetTimings(session);
            var times = new List<double>();
            for (int i = 0; i < runs; i++)
            {
                watch.Restart();
                using SessionOutputs o = await OneModelGpu.RunAsync(session, inputs).ConfigureAwait(true);
                times.Add(watch.Elapsed.TotalMilliseconds);
            }

            times.Sort();
            result["runs"] = runs;
            result["warmMedianMs"] = runs > 0 ? Math.Round(times[times.Count / 2], 2) : 0;
            result["warmMinMs"] = runs > 0 ? Math.Round(times[0], 2) : 0;
            GpuSessionInfo after = OneModelGpu.GetInfo(session);
            double per = runs > 0 ? 1.0 / runs : 0;
            result["uploadMsPerRun"] = Math.Round(after.Timings.UploadMilliseconds * per, 2);
            result["submitReadbackMsPerRun"] = Math.Round(after.Timings.SubmitAndReadbackMilliseconds * per, 2);
            result["recordMsPerRun"] = Math.Round(after.Timings.RecordMilliseconds * per, 2);
            result["submitCallsMsPerRun"] = Math.Round(after.Timings.SubmitMilliseconds * per, 2);
            result["arenaBytes"] = after.ArenaBytes;
            result["weightBytes"] = after.ConstantBytes;
            result["deviceAllocations"] = after.DeviceAllocations;
        }

        foreach (Tensor t in inputs.Values)
        {
            t.Dispose();
        }

        GC.Collect();
    }

    // Whisper's start-of-transcript prompt, as in tools/GpuReport: <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>.
    private static readonly long[] Prompt = [50258, 50259, 50359, 50363];

    /// <summary>
    /// Greedy decoding on the GPU through a decoder that takes the whole token sequence each step. The case file holds the encoder states
    /// (made by the desktop's CPU exact mode) and the CPU exact mode's free-running tokens. Reports whether the tokens are equal, whether a
    /// second decode is bit-identical, and ms per token.
    /// </summary>
    [JSExport]
    public static async Task<string> RunGreedy(string config, byte[] decoder, byte[] caseFile)
    {
        using JsonDocument cfg = JsonDocument.Parse(config);
        string label = cfg.RootElement.TryGetProperty("label", out JsonElement l) ? l.GetString() ?? "" : "";
        ArithmeticMode mode = cfg.RootElement.TryGetProperty("mode", out JsonElement m) && m.GetString() == "exact" ? ArithmeticMode.Exact : ArithmeticMode.Fast;
        SetFusion(cfg.RootElement);
        var result = new Dictionary<string, object?> { ["label"] = label, ["mode"] = mode.ToString().ToLowerInvariant() };
        try
        {
            await GreedyInto(result, mode, decoder, caseFile).ConfigureAwait(true);
        }
        catch (Exception e)
        {
            result["error"] = e.GetType().Name + ": " + e.Message;
            result["stack"] = e.StackTrace;
        }

        return ToJson(result);
    }

    private static async Task GreedyInto(Dictionary<string, object?> result, ArithmeticMode mode, byte[] decoderBytes, byte[] caseFile)
    {
        if (!s_registered)
        {
            OneModelGpu.Register(new GpuOptions { CollectTimings = true, FuseOperators = s_fuse, FusedAttention = s_attention, PipelineFirstDispatches = s_pipe });
            s_registered = true;
        }

        Case c = Case.Parse(caseFile);
        Case.Entry hsEntry = c.Inputs.Single();
        int batch = hsEntry.Dims[0];
        long[] cpuTokens = [.. MemoryMarshal.Cast<byte, float>(c.Blob(c.Outputs.Single())).ToArray().Select(f => (long)f)];
        int steps = cpuTokens.Length - Prompt.Length;
        using Tensor states = Tensor.FromSpan<float>(MemoryMarshal.Cast<byte, float>(c.Blob(hsEntry)).ToArray(), Shape.Create(hsEntry.Dims));

        await OneModelGpu.PrepareDeviceAsync().ConfigureAwait(true);
        var watch = Stopwatch.StartNew();
        using Session dec = Session.Load(decoderBytes, new ModelOptions { Device = ComputeDevice.Gpu, Mode = mode });
        result["decoderCreateMs"] = Math.Round(watch.Elapsed.TotalMilliseconds, 1);
        result["adapter"] = OneModelGpu.GetInfo(dec).Adapter;

        async Task<(List<long> Tokens, ulong Hash, double Ms)> Decode()
        {
            var tokens = new List<long>(Prompt);
            ulong h = 14695981039346656037UL;
            var sw = Stopwatch.StartNew();
            for (int s = 0; s < steps; s++)
            {
                long[] ids = new long[batch * tokens.Count];
                for (int b = 0; b < batch; b++)
                {
                    tokens.CopyTo(ids, b * tokens.Count);
                }

                using Tensor idt = Tensor.FromSpan<long>(ids, Shape.Create(batch, tokens.Count));
                using SessionOutputs o = await OneModelGpu.RunAsync(dec, new Dictionary<string, Tensor> { ["input_ids"] = idt, ["encoder_hidden_states"] = states }).ConfigureAwait(true);
                ReadOnlySpan<float> all = o["logits"].Data<float>();
                foreach (byte x in MemoryMarshal.AsBytes(all))
                {
                    h = (h ^ x) * 1099511628211UL;
                }

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

            return (tokens, h, sw.Elapsed.TotalMilliseconds);
        }

        var first = await Decode().ConfigureAwait(true);
        var second = await Decode().ConfigureAwait(true);
        result["steps"] = steps;
        result["tokensEqualCpu"] = first.Tokens.SequenceEqual(cpuTokens);
        result["tokensGpu"] = first.Tokens.Skip(Prompt.Length).ToArray();
        result["tokensCpu"] = cpuTokens.Skip(Prompt.Length).ToArray();
        result["bitIdentical"] = first.Hash == second.Hash && first.Tokens.SequenceEqual(second.Tokens);
        result["firstPassMsPerToken"] = Math.Round(first.Ms / steps, 1);
        result["warmMsPerToken"] = Math.Round(second.Ms / steps, 1);
        using Tensor lastIds = Tensor.FromSpan<long>(new long[batch * (steps + Prompt.Length)], Shape.Create(batch, steps + Prompt.Length));
        SessionDiagnostics d = dec.GetDiagnostics(new Dictionary<string, Tensor> { ["input_ids"] = lastIds, ["encoder_hidden_states"] = states });
        result["nodes"] = d.Nodes.Length;
        result["nodesOnDevice"] = d.NodesOnDevice;
        result["regions"] = d.DeviceRegions;
        GC.Collect();
    }

    // A small writer instead of reflection-based serialization, so the app also trims and compiles ahead of time.
    private static string ToJson(Dictionary<string, object?> result)
    {
        using var stream = new System.IO.MemoryStream();
        using (var w = new Utf8JsonWriter(stream))
        {
            Value(w, result);
        }

        return System.Text.Encoding.UTF8.GetString(stream.ToArray());
    }

    private static void Value(Utf8JsonWriter w, object? v)
    {
        switch (v)
        {
            case null: w.WriteNullValue(); break;
            case string s: w.WriteStringValue(s); break;
            case bool b: w.WriteBooleanValue(b); break;
            case int i: w.WriteNumberValue(i); break;
            case long l: w.WriteNumberValue(l); break;
            case double d: w.WriteNumberValue(double.IsFinite(d) ? d : 0); break;
            case Dictionary<string, double> dd:
                w.WriteStartObject();
                foreach ((string k, double x) in dd) { w.WritePropertyName(k); Value(w, x); }
                w.WriteEndObject();
                break;
            case Dictionary<string, object?> o:
                w.WriteStartObject();
                foreach ((string k, object? x) in o) { w.WritePropertyName(k); Value(w, x); }
                w.WriteEndObject();
                break;
            case System.Collections.IEnumerable e:
                w.WriteStartArray();
                foreach (object? x in e) { Value(w, x); }
                w.WriteEndArray();
                break;
            default: w.WriteStringValue(v.ToString()); break;
        }
    }

    private static string Trim(string s) => s.Length > 600 ? s[..600] + "..." : s;

    private static Dictionary<string, float[]> Collect(Session s, SessionOutputs o)
    {
        var r = new Dictionary<string, float[]>();
        foreach (TensorInfo info in s.Outputs)
        {
            Tensor t = o[info.Name];
            r[info.Name] = t.DType == DType.F16 ? [.. t.Data<Half>().ToArray().Select(h => (float)h)] : t.Data<float>().ToArray();
        }

        return r;
    }

    // Greedy CTC ids of a [batch, frames, classes] output, as the desktop tool prints them: for the speech recognizer's comparison.
    private static string Greedy(Case c, Dictionary<string, float[]> got)
    {
        Case.Entry? e = c.Outputs.FirstOrDefault(x => x.Dims.Length == 3 && c.Model.Contains("crnn", StringComparison.OrdinalIgnoreCase));
        if (e is null)
        {
            return "";
        }

        string Ids(ReadOnlySpan<float> logits)
        {
            int[] dims = e.Dims;
            var rows = new List<string>();
            for (int b = 0; b < dims[0]; b++)
            {
                var ids = new List<int>();
                int last = 0;
                for (int t = 0; t < dims[1]; t++)
                {
                    ReadOnlySpan<float> row = logits.Slice(((b * dims[1]) + t) * dims[2], dims[2]);
                    int best = 0;
                    for (int k = 1; k < row.Length; k++)
                    {
                        if (row[k] > row[best])
                        {
                            best = k;
                        }
                    }

                    if (best != last && best != 0)
                    {
                        ids.Add(best);
                    }

                    last = best;
                }

                rows.Add(string.Join(",", ids));
            }

            return string.Join(" | ", rows);
        }

        string gpu = Ids(got[e.Name]);
        string cpu = Ids(MemoryMarshal.Cast<byte, float>(c.Blob(e)));
        return gpu == cpu ? "ctc ids equal to cpu: " + gpu : $"ctc ids DIFFER: gpu [{gpu}] cpu [{cpu}]";
    }

    private static double Scaled(float[] a, ReadOnlySpan<float> r)
    {
        double max = 0, diff = 0;
        for (int i = 0; i < a.Length; i++)
        {
            max = Math.Max(max, Math.Abs(r[i]));
            diff = Math.Max(diff, Math.Abs((double)a[i] - r[i]));
        }

        return max == 0 ? diff : diff / max;
    }

    private static string Hash(Dictionary<string, float[]> outputs)
    {
        ulong h = 14695981039346656037UL;
        foreach (string k in outputs.Keys.Order(StringComparer.Ordinal))
        {
            foreach (byte b in MemoryMarshal.AsBytes<float>(outputs[k]))
            {
                h = (h ^ b) * 1099511628211UL;
            }
        }

        return h.ToString("x16", System.Globalization.CultureInfo.InvariantCulture);
    }
}

/// <summary>The case file written by tools/GpuReport --export.</summary>
internal sealed class Case
{
    public sealed record Entry(string Role, string Name, string DType, int[] Dims, int Offset, int Bytes);

    private readonly byte[] _file;
    private readonly int _blobStart;

    public string Model { get; }

    public List<Entry> Inputs { get; } = [];

    public List<Entry> Outputs { get; } = [];

    private Case(byte[] file, int blobStart, string model)
    {
        _file = file;
        _blobStart = blobStart;
        Model = model;
    }

    public ReadOnlySpan<byte> Blob(Entry e) => _file.AsSpan(_blobStart + e.Offset, e.Bytes);

    public static Case Parse(byte[] file)
    {
        int jsonLen = (int)BitConverter.ToUInt32(file, 0);
        using JsonDocument doc = JsonDocument.Parse(file.AsMemory(4, jsonLen));
        var c = new Case(file, 4 + jsonLen, doc.RootElement.GetProperty("model").GetString() ?? "");
        foreach (JsonElement e in doc.RootElement.GetProperty("entries").EnumerateArray())
        {
            var entry = new Entry(e.GetProperty("role").GetString()!, e.GetProperty("name").GetString()!, e.GetProperty("dtype").GetString()!,
                [.. e.GetProperty("dims").EnumerateArray().Select(x => x.GetInt32())], (int)e.GetProperty("offset").GetInt64(), e.GetProperty("bytes").GetInt32());
            (entry.Role == "input" ? c.Inputs : c.Outputs).Add(entry);
        }

        return c;
    }
}
