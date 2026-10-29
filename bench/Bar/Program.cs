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
            "usage: Bar run --model FILE --ref DIR --model-name N --variant V --batch B --threads T [--side ort|ours]\n" +
            "               [--reps 3] [--warmup 3] [--target-ms 1000] --row-out FILE [--sdk X] [--lock-label L]\n" +
            "       Bar emit --row FILE --rss-mb X [--csv FILE] [--md FILE]");
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
            "ours" => new OursSide(),
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
