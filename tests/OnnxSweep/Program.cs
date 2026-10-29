using System.Diagnostics;
using System.Globalization;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.OnnxSweep;

internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length >= 2 && args[0] == "sweep")
        {
            return Sweep(args[1], args.Length > 2 ? args[2] : null);
        }

        if (args.Length >= 3 && args[0] == "dump")
        {
            return DumpFiles(args[1], args.Skip(2));
        }

        if (args.Length == 3 && args[0] == "tensors")
        {
            return DumpTensors(args[1], args[2]);
        }

        if (args.Length >= 2 && args[0] == "load")
        {
            return LoadOne(args[1], args.Length > 2 && args[2] == "nomap");
        }

        Console.Error.WriteLine("usage: sweep <root> [<dumpdir>] | dump <dumpdir> <model.onnx>... | tensors <root> <outfile> | load <model.onnx> [nomap]");
        return 2;
    }

    private static string KeyOf(string root, string file) =>
        Path.GetRelativePath(root, file).Replace(Path.DirectorySeparatorChar, '_') + ".txt";

    private static int Sweep(string root, string? dumpDir)
    {
        if (dumpDir is not null)
        {
            Directory.CreateDirectory(dumpDir);
        }

        string[] models = Directory.EnumerateFiles(root, "model.onnx", SearchOption.AllDirectories).Order(StringComparer.Ordinal).ToArray();
        int ok = 0, tensorsOk = 0, tensorsSkipped = 0, tensorsNotStorable = 0, tensorFail = 0;
        var failures = new SortedDictionary<string, List<string>>(StringComparer.Ordinal);
        var skipped = new SortedDictionary<string, int>(StringComparer.Ordinal);
        var sw = Stopwatch.StartNew();
        foreach (string file in models)
        {
            string dir = Path.GetDirectoryName(file)!;
            IrModel? model = null;
            try
            {
                model = OnnxImporter.Load(file);
                ok++;
                if (dumpDir is not null)
                {
                    File.WriteAllText(Path.Combine(dumpDir, KeyOf(root, file)), Dump.Model(model));
                }
            }
            catch (OnnxImportException e)
            {
                Add(failures, $"{e.Kind}: {Strip(e.Message)}", Path.GetRelativePath(root, file));
                if (dumpDir is not null)
                {
                    File.WriteAllText(Path.Combine(dumpDir, KeyOf(root, file)), "ERROR\n");
                }

                continue;
            }
            catch (Exception e)
            {
                Add(failures, $"BUG {e.GetType().Name}: {e.Message}", Path.GetRelativePath(root, file));
                continue;
            }

            using (model)
            {
                foreach (string pb in Directory.EnumerateFiles(dir, "*.pb", SearchOption.AllDirectories).Order(StringComparer.Ordinal))
                {
                    string name = Path.GetFileNameWithoutExtension(pb);
                    int us = name.LastIndexOf('_');
                    bool isInput = name.StartsWith("input_", StringComparison.Ordinal);
                    if (us < 0 || !int.TryParse(name.AsSpan(us + 1), CultureInfo.InvariantCulture, out int idx))
                    {
                        continue;
                    }

                    var list = isInput ? model.Graph.Inputs : model.Graph.Outputs;
                    IrType? declared = idx < list.Length ? list[idx].Type : null;
                    if (declared is not null && declared.Kind != IrTypeKind.Tensor)
                    {
                        tensorsSkipped++;
                        skipped[declared.Kind.ToString()] = skipped.GetValueOrDefault(declared.Kind.ToString()) + 1;
                        continue;
                    }

                    try
                    {
                        TensorData t = OnnxImporter.LoadTensor(File.ReadAllBytes(pb));
                        if (!t.TryGetDType(out _))
                        {
                            tensorsNotStorable++;
                            continue;
                        }

                        using Tensor x = t.ToTensor();
                        if (x.Length != t.ElementCount || x.Rank != t.Dims.Length)
                        {
                            throw new InvalidOperationException("round trip shape mismatch");
                        }

                        tensorsOk++;
                    }
                    catch (Exception e)
                    {
                        tensorFail++;
                        Add(failures, $"tensor {e.GetType().Name}: {Strip(e.Message)}", Path.GetRelativePath(root, pb));
                    }
                }
            }
        }

        Console.WriteLine($"models: {models.Length}  imported: {ok}  failed: {models.Length - ok}  in {sw.Elapsed.TotalSeconds:F1}s");
        Console.WriteLine($"tensor .pb: converted {tensorsOk}, not storable by the tensor layer {tensorsNotStorable}, non-tensor protos skipped {tensorsSkipped} [{string.Join(", ", skipped.Select(kv => kv.Key + "=" + kv.Value))}], failed {tensorFail}");
        foreach ((string cause, List<string> where) in failures)
        {
            Console.WriteLine($"FAIL x{where.Count}: {cause}");
            foreach (string w in where.Take(5))
            {
                Console.WriteLine("   " + w);
            }
        }

        return models.Length - ok == 0 && tensorFail == 0 ? 0 : 1;
    }

    /// <summary>Writes one line per input/output .pb under the test data sets, in the format of tools/onnx_dump.py tensors.</summary>
    private static int DumpTensors(string root, string outFile)
    {
        var lines = new List<string>();
        foreach (string model in Directory.EnumerateFiles(root, "model.onnx", SearchOption.AllDirectories).Order(StringComparer.Ordinal))
        {
            string dir = Path.GetDirectoryName(model)!;
            using IrModel m = OnnxImporter.Load(model);
            foreach (string pb in Directory.EnumerateFiles(dir, "*.pb", SearchOption.AllDirectories).Order(StringComparer.Ordinal))
            {
                string n = Path.GetFileName(pb);
                if (!n.StartsWith("input_", StringComparison.Ordinal) && !n.StartsWith("output_", StringComparison.Ordinal))
                {
                    continue;
                }

                int idx = int.Parse(n[..^3][(n.LastIndexOf('_') + 1)..], CultureInfo.InvariantCulture);
                var list = n.StartsWith("input_", StringComparison.Ordinal) ? m.Graph.Inputs : m.Graph.Outputs;
                string key = Path.GetRelativePath(root, pb);
                IrType? declared = idx < list.Length ? list[idx].Type : null;
                lines.Add(declared is not null && declared.Kind != IrTypeKind.Tensor
                    ? key + ": SKIP"
                    : key + ": " + Dump.Tensor(OnnxImporter.LoadTensor(File.ReadAllBytes(pb))));
            }
        }

        File.WriteAllText(outFile, string.Join('\n', lines) + "\n");
        return 0;
    }

    private static string Strip(string message)
    {
        // Messages carry file names and offsets; group failures by the cause alone.
        int i = message.LastIndexOf(" (", StringComparison.Ordinal);
        string s = i > 0 ? message[..i] : message;
        return System.Text.RegularExpressions.Regex.Replace(s, @"'[^']*'|\d+", "#");
    }

    private static void Add(SortedDictionary<string, List<string>> map, string key, string value)
    {
        if (!map.TryGetValue(key, out List<string>? l))
        {
            map[key] = l = [];
        }

        l.Add(value);
    }

    private static int DumpFiles(string outDir, IEnumerable<string> files)
    {
        Directory.CreateDirectory(outDir);
        int bad = 0;
        foreach (string f in files)
        {
            string target = Path.Combine(outDir, Path.GetFileName(f) + ".txt");
            try
            {
                using IrModel m = OnnxImporter.Load(f);
                File.WriteAllText(target, Dump.Model(m));
            }
            catch (OnnxImportException e)
            {
                Console.WriteLine($"{f}: {e.Message}");
                File.WriteAllText(target, "ERROR\n");
                bad++;
            }
        }

        return bad == 0 ? 0 : 1;
    }

    private static int LoadOne(string file, bool noMap)
    {
        var sw = Stopwatch.StartNew();
        using IrModel m = OnnxImporter.Load(file, new OnnxLoadOptions { MemoryMap = !noMap });
        double ms = sw.Elapsed.TotalMilliseconds;
        long bytes = 0;
        int inits = 0;
        foreach (TensorData t in m.Graph.Initializers)
        {
            bytes += t.RawLength;
            inits++;
        }

        long heap = GC.GetTotalMemory(false);
        using Process p = Process.GetCurrentProcess();
        Console.WriteLine($"{Path.GetFileName(file)} ({(noMap ? "read" : "mapped")}): load {ms:F1} ms; nodes {m.Graph.Nodes.Length}; initializers {inits} ({bytes / 1048576.0:F1} MiB); managed heap {heap / 1048576.0:F1} MiB; peak RSS {p.PeakWorkingSet64 / 1048576.0:F1} MiB");
        return 0;
    }
}
