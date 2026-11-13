using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using RecursiveAI.OneGpu.Kernels;
using System.Security.Cryptography;
using RecursiveAI.OneModel;
using RecursiveAI.OneModel.Gpu;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

// --export DIR also writes DIR/case.bin: the inputs this run uses and the CPU exact mode's outputs, for a host that cannot run the CPU
// reference itself (see tools/GpuBrowser).
// GpuReport <model.onnx> [--batch N] [--seq N] [--size HxW] [--modes fast,exact] [--runs N] [--bound 1e-4] [--label text]
// Image models (float32 inputs): the first dimension is the batch, open spatial dimensions take --size (default 224x224).
// Prints one "row" line per mode. The reference is the CPU in exact mode.
string? path = null, label = "", inputFile = null, exportDir = null;
bool showSchedule = false;
int cpuThreads = 0;
bool? fusedAttention = null;
bool profileSteps = false;
bool requireCheck = false, noFuse = false, noEpilogue = false, badId = false;
int batch = 1, seq = 128, runs = 10, sizeH = 224, sizeW = 224, steps = 64, threads = 8;
string? greedyEncoder = null, unfused = null, attribute = null, probes = null, worstFilter = null, emulate = null;
int[]? stageLayers = null;
int sinkRowCount = 0;
int[] seedList = [777];
string[] variantList = ["cpu-fast", "cpu-exact", "gpu-fast", "gpu-exact"];
double inputScale = 0.5;
int inputSeed = 777;
bool perItem = false;
int noiseDraws = 0;
double bound = 1e-4;
string[] modes = ["fast", "exact"];
int? pipe = null;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--batch": batch = int.Parse(args[++i]); break;
        case "--seq": seq = int.Parse(args[++i]); break;
        case "--size": { string[] hw = args[++i].Split('x'); sizeH = int.Parse(hw[0]); sizeW = int.Parse(hw[^1]); break; }
        case "--runs": runs = int.Parse(args[++i]); break;
        case "--modes": modes = args[++i].Split(','); break;
        case "--label": label = args[++i]; break;
        case "--export": exportDir = args[++i]; break;
        case "--input": inputFile = args[++i]; break;
        case "--schedule": showSchedule = true; break;
        case "--require": requireCheck = true; break;
        case "--no-fuse": noFuse = true; if (i + 1 < args.Length && args[i + 1] is "attention" or "gelu") { unfused = args[++i]; } break;
        case "--fused-attention": fusedAttention = true; break;
        case "--profile": profileSteps = true; break;
        case "--pipe": pipe = int.Parse(args[++i]); break;
        case "--no-epilogue": noEpilogue = true; break;
        case "--greedy-encoder": greedyEncoder = args[++i]; break;
        case "--stages": stageLayers = [.. args[++i].Split(',').Select(int.Parse)]; break;
        case "--sink-rows": sinkRowCount = int.Parse(args[++i]); break;
        case "--emulate": emulate = args[++i]; break;
        case "--seeds": seedList = [.. args[++i].Split(',').Select(int.Parse)]; break;
        case "--worst": worstFilter = args[++i]; break;
        case "--probes": probes = args[++i]; break;
        case "--attribute": attribute = args[++i]; break;
        case "--variants": variantList = args[++i].Split(','); break;
        case "--input-scale": inputScale = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
        case "--input-seed": inputSeed = int.Parse(args[++i]); break;
        case "--per-item": perItem = true; break;
        case "--noise": noiseDraws = int.Parse(args[++i]); break;
        case "--steps": steps = int.Parse(args[++i]); break;
        case "--threads": threads = int.Parse(args[++i]); break;
        case "--bad-id": badId = true; break;
        case "--cpu-time": cpuThreads = int.Parse(args[++i]); break;
        case "--bound": bound = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
        default: path = args[i]; break;
    }
}

if (path is null)
{
    Console.Error.WriteLine("usage: GpuReport <model.onnx> [--batch N] [--seq N] [--modes fast,exact] [--runs N] [--bound 1e-4] [--label text]");
    return 2;
}

if (attribute is not null)
{
    // GpuReport <model32.onnx> --attribute <model64.onnx> [--batch N] [--variants a,b] [--input-scale S] [--input-seed K] [--per-item]
    OneModelGpu.Register(new GpuOptions { CollectTimings = false, FuseOperators = !noFuse || unfused is not null, FuseEpilogues = !noEpilogue, KeepAttentionUnfused = unfused == "attention", FusedAttention = fusedAttention, KeepGeluUnfused = unfused == "gelu" });
    if (emulate is not null)
    {
        // --emulate cfg1;cfg2;... with --input-seed as a comma list is not needed: seeds come from --seeds
        return GpuReport.Attribution.Emulate(attribute, batch, seedList, emulate.Split(';'), inputScale);
    }

    if (probes is not null)
    {
        return GpuReport.Attribution.Probes(probes, attribute, batch, variantList, inputScale, inputSeed, perItem);
    }

    return GpuReport.Attribution.Run(path, attribute, batch, variantList, inputScale, inputSeed, perItem, worstFilter, stageLayers, sinkRowCount);
}

if (greedyEncoder is not null)
{
    // GpuReport <decoder.onnx> --greedy-encoder <encoder.onnx> [--steps N] [--batch N] [--threads N] [--runs N] [--bound 1e-4]
    OneModelGpu.Register(new GpuOptions { CollectTimings = false, FuseOperators = !noFuse || unfused is not null, FuseEpilogues = !noEpilogue, KeepAttentionUnfused = unfused == "attention", FusedAttention = fusedAttention, ProfileSteps = profileSteps, PipelineFirstDispatches = pipe, KeepGeluUnfused = unfused == "gelu" });
    return GpuReport.Greedy.Run(greedyEncoder, path, steps, batch, threads, bound, runs, exportDir);
}

OneModelGpu.Register(new GpuOptions { CollectTimings = true, FuseOperators = !noFuse || unfused is not null, FuseEpilogues = !noEpilogue, KeepAttentionUnfused = unfused == "attention", FusedAttention = fusedAttention, ProfileSteps = profileSteps, PipelineFirstDispatches = pipe, KeepGeluUnfused = unfused == "gelu" });
Console.WriteLine($"model={Path.GetFileName(path)} batch={batch} seq={seq} {label}");

// Inputs: token ids from a fixed generator, a mask that pads the tail of each row differently, zero segment ids.
Dictionary<string, Tensor> MakeInputs(Session s)
{
    var feed = new Dictionary<string, Tensor>();
    var rng = new Random(1234);
    foreach (TensorInfo info in s.Inputs)
    {
        if (info.DType == DType.F32)
        {
            // Rank-4 image inputs: smooth values in [-1, 1] from a fixed generator, open spatial dimensions take --size.
            if (info.Dims.Length == 4)
            {
                // An image-like tensor: smooth values in [-1, 1] from a fixed generator.
                int[] imageDims = new int[info.Dims.Length];
                for (int d = 0; d < imageDims.Length; d++)
                {
                    imageDims[d] = info.Dims[d].IsFixed ? (int)info.Dims[d].Value : d == 0 ? batch : d == imageDims.Length - 2 ? sizeH : d == imageDims.Length - 1 ? sizeW : 3;
                }

                Shape shape = Shape.Create(imageDims);
                float[] pixels = new float[shape.Length];
                for (int e = 0; e < pixels.Length; e++)
                {
                    pixels[e] = (float)((rng.NextDouble() * 2) - 1);
                }

                if (inputFile is not null)
                {
                    // A float32 .npy (version 1 or 2, C order) holding the whole input, e.g. a reference input of the model.
                    byte[] raw = File.ReadAllBytes(inputFile);
                    int headerLen = raw[6] == 1 ? BitConverter.ToUInt16(raw, 8) : (int)BitConverter.ToUInt32(raw, 8);
                    int npyStart = (raw[6] == 1 ? 10 : 12) + headerLen;
                    string header = System.Text.Encoding.ASCII.GetString(raw, 0, npyStart);
                    string dims2 = header[(header.IndexOf("'shape'", StringComparison.Ordinal) + 8)..];
                    dims2 = dims2[(dims2.IndexOf('(') + 1)..dims2.IndexOf(')')];
                    int[] fileDims = [.. dims2.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse)];
                    shape = Shape.Create(fileDims);
                    pixels = new float[shape.Length];
                    Buffer.BlockCopy(raw, npyStart, pixels, 0, pixels.Length * 4);
                }

                feed[info.Name] = Tensor.FromSpan<float>(pixels, shape);
                Console.WriteLine($"input {info.Name}: {shape}");
                continue;
            }

            // Other float inputs (a log-mel spectrogram, encoder states): dimensions named batch or seq take the options, others their size.
            // The values come from a fixed generator, normally distributed with deviation 0.5, like the reference set's synthetic inputs.
            int[] dims = [.. info.Dims.Select(d => d.Param == "batch" ? batch : d.Param is "seq" or "decoder_seq" ? seq : d.IsFixed ? (int)d.Value : seq)];
            float[] values = new float[dims.Aggregate(1, (a, b) => a * b)];
            var frng = new Random(777 + info.Name.Length);
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = 0.5f * (float)(Math.Sqrt(-2 * Math.Log(1 - frng.NextDouble())) * Math.Cos(2 * Math.PI * frng.NextDouble()));
            }

            feed[info.Name] = Tensor.FromSpan<float>(values, Shape.Create(dims));
            continue;
        }

        long[] data = new long[batch * seq];
        for (int b = 0; b < batch; b++)
        {
            int valid = Math.Max(seq - 7 * b, seq / 2);
            for (int t = 0; t < seq; t++)
            {
                data[b * seq + t] = info.Name switch
                {
                    "attention_mask" => t < valid ? 1 : 0,
                    "token_type_ids" => 0,
                    _ => t < valid ? 1000 + rng.Next(28000) : 0,
                };
            }
        }

        feed[info.Name] = Tensor.FromSpan<long>(data, Shape.Create(batch, seq));
    }

    return feed;
}

static string Hash(ReadOnlySpan<float> d) => Convert.ToHexString(SHA256.HashData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(d)))[..16];

var outDims = new Dictionary<string, int[]>();
Dictionary<string, float[]> Collect(Session s, Dictionary<string, Tensor> feed)
{
    using SessionOutputs o = s.Run(feed);
    var r = new Dictionary<string, float[]>();
    foreach (TensorInfo info in s.Outputs)
    {
        Tensor t = o[info.Name];
        outDims[info.Name] = [.. Enumerable.Range(0, t.Shape.Rank).Select(i => t.Shape[i])];
        r[info.Name] = t.DType == DType.F16 ? [.. t.Data<Half>().ToArray().Select(h => (float)h)] : t.Data<float>().ToArray();
    }

    return r;
}

// Greedy CTC over [batch, frames, classes] scores: per-frame argmax (lowest index on ties), repeats merged, blank (class 0) removed.
static string Greedy(float[] logits, int[] dims)
{
    if (dims.Length != 3)
    {
        return "";
    }

    var rows = new List<string>();
    for (int b = 0; b < dims[0]; b++)
    {
        var ids = new List<int>();
        int last = 0;
        for (int t = 0; t < dims[1]; t++)
        {
            int best = 0;
            ReadOnlySpan<float> row = logits.AsSpan(((b * dims[1]) + t) * dims[2], dims[2]);
            for (int c = 1; c < row.Length; c++)
            {
                if (row[c] > row[best])
                {
                    best = c;
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

static double Scaled(float[] a, float[] r)
{
    double max = 0, diff = 0;
    for (int i = 0; i < a.Length; i++)
    {
        max = Math.Max(max, Math.Abs(r[i]));
        diff = Math.Max(diff, Math.Abs((double)a[i] - r[i]));
    }

    return max == 0 ? diff : diff / max;
}

Dictionary<string, float[]> reference;
Dictionary<string, Tensor> inputs;
using (Session cpu = Session.Load(path, new ModelOptions { Mode = ArithmeticMode.Exact }))
{
    inputs = MakeInputs(cpu);
    reference = Collect(cpu, inputs);
}

if (exportDir is not null)
{
    Directory.CreateDirectory(exportDir);
    GpuReport.CaseFile.Write(Path.Combine(exportDir, "case.bin"), Path.GetFileName(path), batch, seq, inputs, reference, outDims);
    Console.WriteLine($"exported {Path.Combine(exportDir, "case.bin")}");
}

if (noiseDraws > 0)
{
    // How far the CPU's exact mode moves when every float input changes by about one unit in the last place: the model's own noise floor
    // for this metric. A device cannot be expected to sit much below it.
    using Session cpu = Session.Load(path, new ModelOptions { Mode = ArithmeticMode.Exact });
    var rngNoise = new Random(99);
    for (int d = 0; d < noiseDraws; d++)
    {
        var noisy = new Dictionary<string, Tensor>(inputs);
        foreach ((string name, Tensor t) in inputs)
        {
            if (t.DType == DType.F32)
            {
                float[] v = t.Data<float>().ToArray();
                for (int i = 0; i < v.Length; i++)
                {
                    v[i] *= 1f + 1e-7f * (float)(rngNoise.NextDouble() * 2 - 1);
                }

                noisy[name] = Tensor.FromSpan<float>(v, t.Shape);
            }
        }

        double worstNoise = 0;
        Dictionary<string, float[]> got = Collect(cpu, noisy);
        foreach ((string name, float[] r) in reference)
        {
            worstNoise = Math.Max(worstNoise, Scaled(got[name], r));
        }

        Console.WriteLine($"noise draw {d}: cpu exact on inputs changed by 1e-7 relative vs cpu exact: scaled error {worstNoise:E2}");
    }

    return 0;
}

if (requireCheck)
{
    // RequireGpu fails session creation naming what cannot run on the device.
    try
    {
        using Session s = Session.Load(path, new ModelOptions { Device = ComputeDevice.Gpu, RequireGpu = true });
        // Some refusals depend on shapes and surface when the plan is made, so the check runs the model once.
        using SessionOutputs ran = s.Run(inputs);
        SessionDiagnostics rd = s.GetDiagnostics(inputs);
        Console.WriteLine($"RequireGpu: accepted at creation and by the first run; {rd.NodesOnDevice} of {rd.Nodes.Length} nodes on the device, {rd.DeviceRegions} regions");
    }
    catch (SessionException e)
    {
        Console.WriteLine($"RequireGpu: {e.Kind}; operators [{string.Join(", ", e.Operators)}]");
    }

    return 0;
}

if (badId)
{
    // A token id outside the embedding table: the device must raise the error the CPU raises.
    var bad = new Dictionary<string, Tensor>(inputs);
    string idName = inputs.Keys.First();
    long[] ids = inputs[idName].Data<long>().ToArray();
    ids[3] = 9_999_999;
    bad[idName] = Tensor.FromSpan<long>(ids, inputs[idName].Shape);
    string Try(Session s2)
    {
        try
        {
            using SessionOutputs o = s2.Run(bad);
            return "no error";
        }
        catch (SessionException e)
        {
            return $"{e.Kind}: {e.Message}";
        }
    }

    using Session cpuBad = Session.Load(path, new ModelOptions { Mode = ArithmeticMode.Exact });
    using Session gpuBad = Session.Load(path, new ModelOptions { Mode = ArithmeticMode.Exact, Device = ComputeDevice.Gpu });
    string onCpu = Try(cpuBad), onGpu = Try(gpuBad);
    Console.WriteLine($"bad id on cpu: {onCpu}");
    Console.WriteLine($"bad id on gpu: {onGpu}");
    using SessionOutputs after = gpuBad.Run(inputs);
    Console.WriteLine($"bad id: {(onCpu == onGpu ? "same error" : "DIFFERENT")}; the same session runs good inputs afterwards");
    return 0;
}

if (cpuThreads > 0)
{
    // The CPU's own fast mode on this machine, for comparison with the device rows.
    using Session cpuFast = Session.Load(path, new ModelOptions { Mode = ArithmeticMode.Fast, IntraOpThreads = cpuThreads });
    for (int i = 0; i < 3; i++)
    {
        using SessionOutputs warm = cpuFast.Run(inputs);
    }

    var times = new List<double>();
    for (int i = 0; i < runs; i++)
    {
        var watch = Stopwatch.StartNew();
        using SessionOutputs o = cpuFast.Run(inputs);
        times.Add(watch.Elapsed.TotalMilliseconds);
    }

    times.Sort();
    Console.WriteLine($"cpu fast mode, {cpuThreads} threads: median of {runs} runs {times[times.Count / 2]:F1} ms (min {times[0]:F1})");
    return 0;
}

// The CPU's own fast mode against its exact mode: how far two correct runs of this model drift apart, for reading the device's parity.
{
    using Session cpuFast = Session.Load(path, new ModelOptions { Mode = ArithmeticMode.Fast, IntraOpThreads = 4 });
    Dictionary<string, float[]> fast = Collect(cpuFast, inputs);
    double drift = 0;
    foreach ((string name, float[] r) in reference)
    {
        drift = Math.Max(drift, Scaled(fast[name], r));
    }

    Console.WriteLine($"reference drift: cpu fast vs cpu exact = {drift:E2}");
}

foreach (string modeName in modes)
{
    ArithmeticMode mode = modeName == "exact" ? ArithmeticMode.Exact : ArithmeticMode.Fast;
    var sw = Stopwatch.StartNew();
    Session gpu = Session.Load(path, new ModelOptions { Mode = mode, Device = ComputeDevice.Gpu });
    var backend = (GpuBackend)gpu.Backend!;
    double createMs = sw.Elapsed.TotalMilliseconds;
    GpuSessionInfo before = OneModelGpu.GetInfo(gpu);

    sw.Restart();
    Dictionary<string, float[]> first = Collect(gpu, inputs);
    double firstMs = sw.Elapsed.TotalMilliseconds;
    GpuSessionInfo afterFirst = OneModelGpu.GetInfo(gpu, inputs);
    var familiesAfterFirst = GpuReport.Counters.PerFamily(backend);
    int groupsAfterFirst = familiesAfterFirst.Sum(f => f.Count);
    Dictionary<string, float[]> second = Collect(gpu, inputs);
    GpuSessionInfo afterSecond = OneModelGpu.GetInfo(gpu);
    var familiesAfterSecond = GpuReport.Counters.PerFamily(backend);
    int groupsAfterSecond = familiesAfterSecond.Sum(f => f.Count);

    double worst = 0;
    foreach ((string name, float[] r) in reference)
    {
        worst = Math.Max(worst, Scaled(first[name], r));
    }

    string decoded = string.Join("; ", first.Keys.Where(k => outDims[k].Length == 3 && k.Contains("logit", StringComparison.OrdinalIgnoreCase))
        .Select(k => Greedy(first[k], outDims[k]) == Greedy(reference[k], outDims[k]) ? $"{k}: greedy CTC ids equal to the cpu's ({Greedy(first[k], outDims[k]).Length} chars of ids)" : $"{k}: greedy CTC ids DIFFER"));
    bool same = first.Keys.All(k => Hash(first[k]) == Hash(second[k]));
    string hashes = string.Join(",", first.Keys.Select(k => $"{k}:{Hash(first[k])}/{Hash(second[k])}"));

    SessionDiagnostics d = gpu.GetDiagnostics(inputs);
    var fallbacks = d.Nodes.Where(n => n.FallbackReason is not null).GroupBy(n => n.OpType + ": " + n.FallbackReason).Select(g => $"{g.Key} x{g.Count()}").ToList();

    // Timing: warm runs, wall per run, and the device's own accounting.
    OneModelGpu.ResetTimings(gpu);
    backend.TakeProfile();
    var wall = new List<double>();
    int changed = 0;
    string firstHash = Hash(first[first.Keys.First()]);
    for (int i = 0; i < runs; i++)
    {
        sw.Restart();
        using SessionOutputs o = gpu.Run(inputs);
        wall.Add(sw.Elapsed.TotalMilliseconds);
        if (Hash(o[first.Keys.First()].Data<float>()) != firstHash)
        {
            changed++;
        }
    }

    wall.Sort();
    if (profileSteps)
    {
        // Each step in its own command buffer: GPU time and dispatches per run, by operator.
        var prof = backend.TakeProfile().OrderByDescending(p => p.Ms).ToArray();
        double total = prof.Sum(p => p.Ms) / runs;
        Console.WriteLine($"  profile (per run, steps timed one by one, sum {total:F2} ms):");
        foreach (var p in prof)
        {
            Console.WriteLine($"    {p.Op,-24} {p.Ms / runs,7:F3} ms  {p.Dispatches / (double)runs,6:F1} dispatches  {p.Steps / runs,4} steps");
        }
    }

    GpuTimings t = OneModelGpu.GetInfo(gpu).Timings;
    double per = 1.0 / runs;

    Console.WriteLine($"row mode={modeName} adapter=\"{afterFirst.Adapter}\" api={afterFirst.Api} kernels=[{string.Join("; ", afterFirst.KernelVariants)}]");
    {
        // Where the largest difference sits and how it compares with the typical value: a few large activations dominate the scaled error.
        foreach ((string name, float[] r) in reference)
        {
            float[] g = first[name];
            double maxRef = 0, sumRef = 0, sumDiff = 0, maxDiff = 0;
            int at = 0;
            for (int i = 0; i < g.Length; i++)
            {
                maxRef = Math.Max(maxRef, Math.Abs(r[i]));
                sumRef += (double)r[i] * r[i];
                double dd = Math.Abs((double)g[i] - r[i]);
                sumDiff += dd * dd;
                if (dd > maxDiff) { maxDiff = dd; at = i; }
            }

            Console.WriteLine($"  detail {name}: max|ref| {maxRef:G4}, rms ref {Math.Sqrt(sumRef / g.Length):G4}, rms diff {Math.Sqrt(sumDiff / g.Length):G3}, max diff {maxDiff:G3} at {at} (ref {r[at]:G4}, gpu {g[at]:G4})");
        }
    }

    Console.WriteLine($"  parity: max scaled error vs cpu exact = {worst:E2} ({(worst <= bound ? "within" : "ABOVE")} the bound {bound:E0})");
    if (decoded.Length > 0)
    {
        Console.WriteLine($"  decoded: {decoded}");
    }

    Console.WriteLine($"  nodes: {d.NodesOnDevice} of {d.Nodes.Length} on gpu ({100.0 * d.NodesOnDevice / d.Nodes.Length:F1}%), {d.DeviceRegions} gpu regions; gpu flops/run = {afterFirst.Plan!.Flops / 1e9:F2} G; MatMul/Gemm nodes on gpu {d.Nodes.Count(n => n.OpType is "MatMul" or "Gemm" && n.Region >= 0)} of {d.Nodes.Count(n => n.OpType is "MatMul" or "Gemm")}");
    Console.WriteLine($"  transfers per run: {d.DeviceRegions * 2} boundary crossings (one upload + one readback per region), to gpu {d.BytesToDevicePerRun / 1048576.0:F1} MiB, from gpu {d.BytesFromDevicePerRun / 1048576.0:F1} MiB");
    Console.WriteLine($"  determinism: run1/run2 outputs {(same ? "bit-identical" : "DIFFER")} {hashes}");
    Console.WriteLine($"  arena: peak {afterSecond.ArenaBytes} B ({afterSecond.ArenaBytes / 1048576.0:F1} MiB), weights {afterSecond.ConstantBytes / 1048576.0:F1} MiB in {afterSecond.ConstantBuffers} buffers; device allocations before first run {before.DeviceAllocations}, after run 1 {afterFirst.DeviceAllocations}, after run 2 {afterSecond.DeviceAllocations} (run 2 allocated {afterSecond.DeviceAllocations - afterFirst.DeviceAllocations})");
    Console.WriteLine($"  bind groups built: {groupsAfterFirst} by the end of run 1, run 2 built {groupsAfterSecond - groupsAfterFirst}");
    if (groupsAfterSecond != groupsAfterFirst)
    {
        Console.WriteLine("  bind groups built in run 2 by family: " + string.Join(", ", familiesAfterFirst.Zip(familiesAfterSecond).Where(z => z.First.Count != z.Second.Count).Select(z => $"{z.First.Family} +{z.Second.Count - z.First.Count}")));
    }
    Console.WriteLine($"  time: create {createMs:F0} ms, first run {firstMs:F0} ms, median of {runs} warm runs {wall[wall.Count / 2]:F2} ms (min {wall[0]:F2}); per run: upload {t.UploadMilliseconds * per:F2} ms, gpu kernels {t.GpuMilliseconds * per:F2} ms, submit+wait+readback {t.SubmitAndReadbackMilliseconds * per:F2} ms (record {t.RecordMilliseconds * per:F2}, submit calls {t.SubmitMilliseconds * per:F2}, wait+readback {(t.SubmitAndReadbackMilliseconds - t.RecordMilliseconds - t.SubmitMilliseconds) * per:F2}), inside gpu regions {t.TotalMilliseconds * per:F2} ms, longest region {t.LongestRegionMilliseconds:F1} ms, dispatches/run {t.Dispatches * per:F0}, passes/run {t.ComputePasses * per:F0}; timed runs whose output differs from run 1: {changed}");
    string[] validation = ((GpuBackend)gpu.Backend!).Device.TakeValidationMessages();
    gpu.Dispose();
    string[] afterDispose = backend.MessagesAfterDispose;
    Console.WriteLine($"  validation messages: {validation.Length}{(validation.Length > 0 ? " first: " + validation[0] : "")}; after dispose: {afterDispose.Length}{(afterDispose.Length > 0 ? " first: " + afterDispose[0] : "")}");
    if (showSchedule)
    {
        // The schedule as segments of steps: where each run executes and which operators it holds.
        var segments = new List<string>();
        string? current = null;
        int count = 0;
        string ops = "";
        foreach (NodeDiagnostic n in d.Nodes)
        {
            string where = n.Region >= 0 ? $"gpu{n.Region}" : "cpu";
            if (where != current)
            {
                if (current is not null) { segments.Add($"{current} x{count} [{ops}]"); }
                current = where; count = 0; ops = "";
            }
            count++;
            ops += (ops.Length > 0 ? "," : "") + n.OpType;
        }
        if (current is not null) { segments.Add($"{current} x{count} [{ops}]"); }
        foreach (string r in segments) { Console.WriteLine("  step: " + (r.Length > 400 ? r[..400] + "..." : r)); }
    }

    foreach (string f in fallbacks)
    {
        Console.WriteLine($"  fallback: {f}");
    }
}

foreach (Tensor t in inputs.Values)
{
    t.Dispose();
}

return 0;
