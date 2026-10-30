using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace OneModel.Bar;

/// <summary>
/// Bar runner. Two commands:
///   run   times one side on one model/batch/thread-count and writes a JSON row;
///   emit  adds the peak RSS (measured by the wrapper around a separate process) and the load after the
///         run, then appends the row to a CSV and a markdown table.
/// See docs/bench-method.md; run-bar.sh drives both under the benchmark lock.
/// </summary>
internal static class Program
{
    private static readonly string[] Columns =
    [
        "time_utc", "machine", "cpu", "isa", "cores", "os", "sdk", "runtime", "side", "side_version",
        "model", "variant", "batch", "threads", "reps", "iters_per_rep", "median_ms", "min_ms", "max_ms",
        "spread_pct", "load_ms", "first_run_ms", "alloc_bytes_per_iter", "peak_rss_mb",
        "load_before", "load_during", "load_after", "check_scaled_err", "lock_label",
    ];

    private static int Main(string[] args)
    {
        try
        {
            if (args.Length == 0)
            {
                return Usage();
            }

            Dictionary<string, string> o = ParseOptions(args.Skip(1));
            return args[0] switch
            {
                "run" => Run(o),
                "emit" => Emit(o),
                "check" => Check(o),
                "convert" => ConvertModel(o),
                "load" => LoadTime(o),
                _ => Usage(),
            };
        }
        catch (Exception e) when (e is ArgumentException or IOException or InvalidDataException or NotSupportedException or KeyNotFoundException)
        {
            Console.Error.WriteLine("bar: " + e.Message);
            return 2;
        }
    }

    private static int Usage()
    {
        Console.Error.WriteLine(
            "usage: Bar run --model FILE --ref DIR --model-name N --variant V --batch B --threads T [--side ort|ours] [--mode fast|exact] [--opt none|basic|all]\n" +
            "               [--reps 3] [--warmup 3] [--target-ms 1000] --row-out FILE [--sdk X] [--lock-label L]\n" +
            "       Bar emit --row FILE --rss-mb X [--csv FILE] [--md FILE]\n" +
            "       Bar convert --model FILE.onnx --out FILE.onemodel   (ONNX to the OneModel format)\n" +
            "       Bar load --model FILE --threads T [--reps 5]   (load time of ours: first load, then the median of the later ones)\n" +
            "       Bar check --model FILE --ref DIR --threads T [--mode fast|exact] [--weights original|fp16|bf16] [--tol 1e-4] [--dump DIR]   (ours vs the stored ONNX Runtime outputs)");
        return 2;
    }

    private static Dictionary<string, string> ParseOptions(IEnumerable<string> a)
    {
        var d = new Dictionary<string, string>(StringComparer.Ordinal);
        string[] arr = [.. a];
        for (int i = 0; i < arr.Length; i++)
        {
            if (!arr[i].StartsWith("--", StringComparison.Ordinal) || i + 1 >= arr.Length)
            {
                throw new ArgumentException($"bad argument {arr[i]}");
            }

            d[arr[i][2..]] = arr[++i];
        }

        return d;
    }

    private static int ConvertModel(Dictionary<string, string> o)
    {
        var sw = Stopwatch.StartNew();
        RecursiveAI.OneModel.Format.ModelFormat.Convert(o["model"], o["out"]);
        Console.WriteLine($"converted {o["model"]} -> {o["out"]}: {new FileInfo(o["out"]).Length} bytes in {sw.Elapsed.TotalMilliseconds:F0} ms");
        return 0;
    }

    private static int LoadTime(Dictionary<string, string> o)
    {
        int threads = int.Parse(o["threads"], CultureInfo.InvariantCulture);
        int reps = int.Parse(o.GetValueOrDefault("reps", "5"), CultureInfo.InvariantCulture);
        var times = new List<double>();
        for (int i = 0; i <= reps; i++)
        {
            using var side = new OursSide("fast", "original", "all");
            long t0 = Stopwatch.GetTimestamp();
            side.Load(o["model"], threads);
            times.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
        }

        double first = times[0];
        times.RemoveAt(0);
        times.Sort();
        Console.WriteLine($"load first {first:F1} ms, median of {reps} later {times[times.Count / 2]:F1} ms");
        return 0;
    }

    private static int Run(Dictionary<string, string> o)
    {
        string modelPath = o["model"];
        string refDir = o["ref"];
        int threads = int.Parse(o["threads"], CultureInfo.InvariantCulture);
        int reps = int.Parse(o.GetValueOrDefault("reps", "3"), CultureInfo.InvariantCulture);
        int warmup = int.Parse(o.GetValueOrDefault("warmup", "3"), CultureInfo.InvariantCulture);
        double targetMs = double.Parse(o.GetValueOrDefault("target-ms", "1000"), CultureInfo.InvariantCulture);
        using ISide side = o.GetValueOrDefault("side", "ort") switch
        {
            "ort" => new OrtSide(),
            "ours" => new OursSide(o.GetValueOrDefault("mode", "fast"), o.GetValueOrDefault("weights", "original"), o.GetValueOrDefault("opt", "all")),
            var s => throw new ArgumentException($"unknown side {s}"),
        };

        var inputs = new Dictionary<string, NpyArray>();
        foreach (string f in Directory.GetFiles(Path.Combine(refDir, "inputs"), "*.npy"))
        {
            inputs[Path.GetFileNameWithoutExtension(f)] = NpyArray.Load(f);
        }

        double loadBefore = Machine.Load1();
        long t0 = Stopwatch.GetTimestamp();
        side.Load(modelPath, threads);
        double loadMs = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        side.Bind(inputs);

        TimingResult r = Timing.Measure(side.Run, warmup, reps, targetMs);
        double loadDuring = Machine.Load1();

        // Sanity: the timed engine must still produce the reference outputs (ORT matches itself;
        // this is the check that gives "ours" rows meaning).
        double err = 0;
        foreach (string f in Directory.GetFiles(Path.Combine(refDir, "outputs"), "*.npy"))
        {
            NpyArray want = NpyArray.Load(f);
            float[]? got = side.Output(Path.GetFileNameWithoutExtension(f));
            if (want.F32 is null || got is null || got.Length != want.F32.Length)
            {
                err = double.NaN;
                break;
            }

            double max = 1e-30, d = 0;
            for (int i = 0; i < got.Length; i++)
            {
                max = Math.Max(max, Math.Abs(want.F32[i]));
                d = Math.Max(d, Math.Abs((double)got[i] - want.F32[i]));
            }

            err = Math.Max(err, d / max);
        }

        string F(double v, string fmt = "R") => v.ToString(fmt, CultureInfo.InvariantCulture);
        var row = new Dictionary<string, string>
        {
            ["time_utc"] = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ", CultureInfo.InvariantCulture),
            ["machine"] = Machine.Name,
            ["cpu"] = Machine.Cpu,
            ["isa"] = Machine.IsaFlags,
            ["cores"] = Machine.Cores.ToString(CultureInfo.InvariantCulture),
            ["os"] = Machine.Os,
            ["sdk"] = o.GetValueOrDefault("sdk", "unknown"),
            ["runtime"] = Machine.Runtime,
            ["side"] = side.Name,
            ["side_version"] = side.Version,
            ["model"] = o["model-name"],
            ["variant"] = o["variant"],
            ["batch"] = o["batch"],
            ["threads"] = threads.ToString(CultureInfo.InvariantCulture),
            ["reps"] = reps.ToString(CultureInfo.InvariantCulture),
            ["iters_per_rep"] = r.ItersPerRep.ToString(CultureInfo.InvariantCulture),
            ["median_ms"] = F(r.Median, "F3"),
            ["min_ms"] = F(r.Min, "F3"),
            ["max_ms"] = F(r.Max, "F3"),
            ["spread_pct"] = F(r.SpreadPercent, "F1"),
            ["load_ms"] = F(loadMs, "F1"),
            ["first_run_ms"] = F(r.FirstMs, "F1"),
            ["alloc_bytes_per_iter"] = F(r.AllocBytesPerIter, "F0"),
            ["peak_rss_mb"] = "",
            ["load_before"] = F(loadBefore, "F2"),
            ["load_during"] = F(loadDuring, "F2"),
            ["load_after"] = "",
            ["check_scaled_err"] = err.ToString("E2", CultureInfo.InvariantCulture),
            ["lock_label"] = o.GetValueOrDefault("lock-label", ""),
        };
        File.WriteAllText(o["row-out"], JsonSerializer.Serialize(row), Encoding.UTF8);
        Console.WriteLine($"{side.Name} {o["model-name"]}/{o["variant"]} b{o["batch"]} t{threads}: " +
                          $"median {r.Median:F3} ms (min {r.Min:F3}, max {r.Max:F3}), load {loadMs:F0} ms");
        return 0;
    }

    /// <summary>
    /// Correctness of this library on one model and input set, against the stored ONNX Runtime outputs: per output the scaled error
    /// max|ours-ref|/max|ref| (the gate) and the element-wise relative error |ours-ref|/max(|ref|, 1e-6) (reported only), plus the
    /// SHA-256 of each output from two runs, so a caller can compare hashes across thread counts and across runs.
    /// Exit code 0 when every output is within --tol, 1 otherwise.
    /// </summary>
    private static int Check(Dictionary<string, string> o)
    {
        int threads = int.Parse(o["threads"], CultureInfo.InvariantCulture);
        double tol = double.Parse(o.GetValueOrDefault("tol", "1e-4"), CultureInfo.InvariantCulture);
        using var side = new OursSide(o.GetValueOrDefault("mode", "fast"), o.GetValueOrDefault("weights", "original"), o.GetValueOrDefault("opt", "all"));
        var inputs = new Dictionary<string, NpyArray>();
        foreach (string f in Directory.GetFiles(Path.Combine(o["ref"], "inputs"), "*.npy"))
        {
            inputs[Path.GetFileNameWithoutExtension(f)] = NpyArray.Load(f);
        }

        side.Load(o["model"], threads);
        side.Bind(inputs);
        bool ok = true;
        string[] first = [];
        string[] files = Directory.GetFiles(Path.Combine(o["ref"], "outputs"), "*.npy");
        Array.Sort(files, StringComparer.Ordinal);
        for (int run = 0; run < 2; run++)
        {
            side.Run();
            string[] hashes = new string[files.Length];
            for (int k = 0; k < files.Length; k++)
            {
                string name = Path.GetFileNameWithoutExtension(files[k]);
                float[]? got = side.Output(name);
                hashes[k] = got is null ? "missing" : Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(System.Runtime.InteropServices.MemoryMarshal.AsBytes<float>(got)))[..16];
                if (run > 0)
                {
                    if (hashes[k] != first[k])
                    {
                        Console.WriteLine($"  {name}: run-to-run hash differs ({first[k]} vs {hashes[k]})");
                        ok = false;
                    }

                    continue;
                }

                NpyArray want = NpyArray.Load(files[k]);
                if (got is not null && o.TryGetValue("dump", out string? dumpDir))
                {
                    Directory.CreateDirectory(dumpDir);
                    File.WriteAllBytes(Path.Combine(dumpDir, name + ".f32"), System.Runtime.InteropServices.MemoryMarshal.AsBytes<float>(got).ToArray());
                }

                if (want.F32 is null || got is null || got.Length != want.F32.Length)
                {
                    Console.WriteLine($"  {name}: FAIL missing or wrong size (want {want.F32?.Length}, got {got?.Length})");
                    ok = false;
                    continue;
                }

                double max = 1e-30, d = 0, rel = 0;
                for (int i = 0; i < got.Length; i++)
                {
                    max = Math.Max(max, Math.Abs(want.F32[i]));
                }

                for (int i = 0; i < got.Length; i++)
                {
                    double diff = Math.Abs((double)got[i] - want.F32[i]);
                    if (double.IsNaN(diff))
                    {
                        diff = double.PositiveInfinity;
                    }

                    d = Math.Max(d, diff);
                    rel = Math.Max(rel, diff / Math.Max(Math.Abs(want.F32[i]), 1e-6));
                }

                bool pass = d / max <= tol;
                ok &= pass;
                // The decision metrics used for int8 rows: how many rows have the same largest element along the last axis (class or
                // token decisions), and the smallest cosine similarity between a row of ours and the same row of the reference.
                int width = want.Shape.Length > 0 ? (int)want.Shape[^1] : got.Length;
                int rows = width > 0 ? got.Length / width : 0, agree = 0;
                double minCos = 1;
                for (int r = 0; r < rows; r++)
                {
                    int ag = 0, aw = 0;
                    double dot = 0, ng = 0, nw = 0;
                    for (int c = 0; c < width; c++)
                    {
                        float gv = got[(r * width) + c], wv = want.F32[(r * width) + c];
                        ag = gv > got[(r * width) + ag] ? c : ag;
                        aw = wv > want.F32[(r * width) + aw] ? c : aw;
                        dot += (double)gv * wv;
                        ng += (double)gv * gv;
                        nw += (double)wv * wv;
                    }

                    agree += ag == aw ? 1 : 0;
                    minCos = Math.Min(minCos, dot / Math.Max(Math.Sqrt(ng * nw), 1e-30));
                }

                Console.WriteLine($"  {name} [{string.Join('x', want.Shape)}]: {(pass ? "pass" : "FAIL")} scaled {d / max:E2} elementwise {rel:E2} top1 {agree}/{rows} mincos {minCos:F6} hash {hashes[k]}");
            }

            if (run == 0)
            {
                first = hashes;
            }
        }

        Console.WriteLine($"check {o["model"]} threads {threads} mode {o.GetValueOrDefault("mode", "fast")} path {side.KernelPath}: {(ok ? "PASS" : "FAIL")}");
        return ok ? 0 : 1;
    }

    private static int Emit(Dictionary<string, string> o)
    {
        var row = JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(o["row"]))
                  ?? throw new InvalidDataException("empty row file");
        row["peak_rss_mb"] = o["rss-mb"];
        row["load_after"] = Machine.Load1().ToString("F2", CultureInfo.InvariantCulture);
        if (o.TryGetValue("csv", out string? csv))
        {
            bool fresh = !File.Exists(csv);
            using var w = new StreamWriter(csv, append: true, new UTF8Encoding(false)) { NewLine = "\n" };
            if (fresh)
            {
                w.WriteLine(string.Join(',', Columns));
            }

            w.WriteLine(string.Join(',', Columns.Select(c => Quote(row.GetValueOrDefault(c, "")))));
        }

        if (o.TryGetValue("md", out string? md))
        {
            bool fresh = !File.Exists(md);
            string[] shown = ["machine", "side", "model", "variant", "batch", "threads", "median_ms", "min_ms", "max_ms", "load_ms", "peak_rss_mb", "alloc_bytes_per_iter", "load_before", "load_during", "load_after"];
            using var w = new StreamWriter(md, append: true, new UTF8Encoding(false)) { NewLine = "\n" };
            if (fresh)
            {
                w.WriteLine("| " + string.Join(" | ", shown) + " |");
                w.WriteLine("|" + string.Concat(shown.Select(_ => "---|")));
            }

            w.WriteLine("| " + string.Join(" | ", shown.Select(c => row.GetValueOrDefault(c, ""))) + " |");
        }

        return 0;
    }

    private static string Quote(string s) =>
        s.Contains(',', StringComparison.Ordinal) || s.Contains('"', StringComparison.Ordinal)
            ? "\"" + s.Replace("\"", "\"\"", StringComparison.Ordinal) + "\""
            : s;
}
