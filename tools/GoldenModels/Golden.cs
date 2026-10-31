// Prints "<name> <sha256>" lines for exact-mode runs, for tools/golden_hash.py to compare across machines and runtimes.
// Shared by the desktop console app (JIT and NativeAOT) and the browser-wasm smoke app, so every row runs the same code.
//
//   model NAME=FILE=REFDIR   run FILE on the inputs in REFDIR/inputs/*.npy; one line per output, plus one for the inputs
//   graph NAME=ONNX=JSON[#CASE]  run a committed test graph on the input stored in JSON ("shape" and "input", or cases.CASE)
//   --cases DIR [--every N]  run every N-th ONNX backend node case under DIR (model.onnx + test_data_set_0/input_*.pb);
//                            cases whose exact session cannot be created or run print nothing on stdout (a note goes to stderr)
//   --threads N              intra-op threads (default: ONEMODEL_THREADS, else 0 = every core)
//   --time REPS             also run each model REPS more times and print the best warm time to stderr (for exact-versus-fast cost)
//   --mode exact|fast        default exact; fast is only for measuring the cost of exactness
using RecursiveAI.OneModel;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using OneModel.Bar;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace OneModel.GoldenModels;

internal static class Golden
{
    // --print NAME: also print the values of that float32 output (as bit patterns), to find where two runs first differ.
    private static string PrintName = "";

    public static int Run(string[] args, TextWriter output, TextWriter error)
    {
        int threads = int.TryParse(Environment.GetEnvironmentVariable("ONEMODEL_THREADS"), NumberStyles.None, CultureInfo.InvariantCulture, out int t) ? t : 0;
        ArithmeticMode mode = ArithmeticMode.Exact;
        string? casesDir = null;
        int every = 1;
        int reps = 0;
        var models = new List<string>();
        var graphs = new List<string>();
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--threads": threads = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--mode": mode = args[++i] == "fast" ? ArithmeticMode.Fast : ArithmeticMode.Exact; break;
                case "--cases": casesDir = args[++i]; break;
                case "--print": PrintName = args[++i]; break;
                case "--time": reps = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--every": every = Math.Max(1, int.Parse(args[++i], CultureInfo.InvariantCulture)); break;
                case "model": models.Add(args[++i]); break;
                case "graph": graphs.Add(args[++i]); break;
                default: error.WriteLine($"unknown argument {args[i]}"); return 2;
            }
        }

        var options = new ModelOptions { Mode = mode, IntraOpThreads = threads };
        int failures = 0;
        foreach (string spec in models)
        {
            string[] p = spec.Split('=');
            if (p.Length != 3)
            {
                error.WriteLine($"bad model spec {spec}: want NAME=FILE=REFDIR");
                return 2;
            }

            try
            {
                RunModel(p[0], p[1], p[2], options, output, error, reps);
            }
            catch (Exception e) when (e is not OutOfMemoryException)
            {
                error.WriteLine($"{p[0]}: {e.GetType().Name}: {e.Message}");
                failures++;
            }
        }

        foreach (string spec in graphs)
        {
            string[] p = spec.Split('=');
            try
            {
                RunGraph(p[0], p[1], p[2], options, output);
            }
            catch (Exception e) when (e is not OutOfMemoryException)
            {
                error.WriteLine($"{p[0]}: {e.GetType().Name}: {e.Message}");
                failures++;
            }
        }

        if (casesDir is not null)
        {
            RunCases(casesDir, every, options, output, error);
        }

        return failures == 0 ? 0 : 1;
    }

    private static void RunModel(string name, string file, string refDir, ModelOptions options, TextWriter output, TextWriter error, int reps)
    {
        var inputs = new Dictionary<string, NpyArray>();
        foreach (string f in Directory.GetFiles(Path.Combine(refDir, "inputs"), "*.npy"))
        {
            inputs[Path.GetFileNameWithoutExtension(f)] = NpyArray.Load(f);
        }

        using Session session = Session.Load(file, options);
        string[] inNames = [.. session.Inputs.Select(i => i.Name)];
        string[] outNames = [.. session.Outputs.Select(o => o.Name)];
        var ins = new Tensor[inNames.Length];
        using var inputHash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        try
        {
            for (int i = 0; i < ins.Length; i++)
            {
                NpyArray a = inputs[inNames[i]];
                Shape shape = Shape.Create(a.Shape);
                ins[i] = a.F32 is not null ? Tensor.FromSpan<float>(a.F32, shape) : Tensor.FromSpan<long>(a.I64!, shape);
                AppendTensor(inputHash, ins[i]);
            }

            output.WriteLine($"{name}/inputs {Hex(inputHash.GetHashAndReset())}");
            var outs = new Tensor[outNames.Length];
            try
            {
                session.Run(inNames, ins, outNames, outs);
                for (int k = 0; k < outs.Length; k++)
                {
                    using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                    AppendTensor(h, outs[k]);
                    output.WriteLine($"{name}/{outNames[k]} {Hex(h.GetHashAndReset())}");
                    if (PrintName == outNames[k] && outs[k].DType == DType.F32)
                    {
                        output.WriteLine("values " + string.Join(' ', outs[k].ToArray<float>().Select(v => BitConverter.SingleToUInt32Bits(v).ToString("x8", CultureInfo.InvariantCulture))));
                    }
                }

                double best = double.MaxValue;
                for (int r = 0; r < reps; r++)
                {
                    long t0 = Stopwatch.GetTimestamp();
                    session.Run(inNames, ins, outNames, outs);
                    best = Math.Min(best, Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
                }

                if (reps > 0)
                {
                    error.WriteLine(string.Create(CultureInfo.InvariantCulture, $"time {name} {options.Mode} threads {options.IntraOpThreads} path {session.KernelPath} best {best:F2} ms"));
                }
            }
            finally
            {
                foreach (Tensor o in outs)
                {
                    if (!o.IsEmpty)
                    {
                        o.Dispose();
                    }
                }
            }
        }
        finally
        {
            foreach (Tensor i in ins)
            {
                if (!i.IsEmpty)
                {
                    i.Dispose();
                }
            }
        }
    }

    private static void RunGraph(string name, string file, string jsonSpec, ModelOptions options, TextWriter output)
    {
        string[] js = jsonSpec.Split('#');
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(js[0]));
        JsonElement e = js.Length > 1 ? doc.RootElement.GetProperty("cases").GetProperty(js[1]) : doc.RootElement;
        int[] shape = [.. e.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32())];
        float[] data = [.. e.GetProperty("input").EnumerateArray().Select(x => (float)x.GetDouble())];
        using Session session = Session.Load(file, options);
        string[] inNames = [session.Inputs[0].Name];
        string[] outNames = [.. session.Outputs.Select(o => o.Name)];
        using Tensor input = Tensor.FromSpan<float>(data, Shape.Create(shape));
        var outs = new Tensor[outNames.Length];
        try
        {
            session.Run(inNames, [input], outNames, outs);
            for (int k = 0; k < outs.Length; k++)
            {
                using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                AppendTensor(h, outs[k]);
                output.WriteLine($"{name}/{outNames[k]} {Hex(h.GetHashAndReset())}");
            }
        }
        finally
        {
            foreach (Tensor o in outs)
            {
                if (!o.IsEmpty)
                {
                    o.Dispose();
                }
            }
        }
    }

    private static void RunCases(string dir, int every, ModelOptions options, TextWriter output, TextWriter error)
    {
        string[] cases = Directory.GetDirectories(dir);
        Array.Sort(cases, StringComparer.Ordinal);
        for (int c = 0; c < cases.Length; c += every)
        {
            string name = Path.GetFileName(cases[c]);
            string data = Path.Combine(cases[c], "test_data_set_0");
            string model = Path.Combine(cases[c], "model.onnx");
            if (!File.Exists(model) || !Directory.Exists(data))
            {
                continue;
            }

            var ins = new List<Tensor>();
            var outs = Array.Empty<Tensor>();
            try
            {
                string[] files = Directory.GetFiles(data, "input_*.pb");
                Array.Sort(files, (a, b) => IndexOf(a).CompareTo(IndexOf(b)));
                using Session session = Session.Load(model, options);
                foreach (string f in files)
                {
                    ins.Add(OnnxImporter.LoadTensor(File.ReadAllBytes(f)).ToTensor());
                }

                string[] inNames = [.. session.Inputs.Select(i => i.Name)];
                string[] outNames = [.. session.Outputs.Select(o => o.Name)];
                if (inNames.Length != ins.Count)
                {
                    throw new InvalidOperationException($"{ins.Count} input files for {inNames.Length} inputs");
                }

                outs = new Tensor[outNames.Length];
                session.Run(inNames, ins.ToArray(), outNames, outs);
                using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                foreach (Tensor o in outs)
                {
                    AppendTensor(h, o);
                }

                output.WriteLine($"case/{name} {Hex(h.GetHashAndReset())}");
            }
            catch (Exception e) when (e is not OutOfMemoryException)
            {
                error.WriteLine($"skip {name}: {e.GetType().Name}: {Truncate(e.Message)}");
                output.WriteLine($"case/{name} {SkipHash}");
            }
            finally
            {
                foreach (Tensor o in outs.Concat(ins))
                {
                    if (!o.IsEmpty)
                    {
                        o.Dispose();
                    }
                }
            }
        }
    }

    // sha256("skipped"): a case that cannot run in exact mode prints this, so skipping is compared across rows like any hash.
    private const string SkipHash = "389595a44dc9dce7dee673f243e805244bae603d6a65a536ad493340ec99ced3";

    private static int IndexOf(string file) =>
        int.TryParse(Path.GetFileNameWithoutExtension(file).AsSpan(6), NumberStyles.None, CultureInfo.InvariantCulture, out int n) ? n : int.MaxValue;

    private static string Truncate(string s) => s.Length <= 160 ? s : s[..160];

    private static string Hex(byte[] h) => Convert.ToHexString(h).ToLowerInvariant();

    /// <summary>Feeds dtype, shape and the row-major element bytes (little-endian on every supported host).</summary>
    private static void AppendTensor(IncrementalHash h, Tensor t)
    {
        h.AppendData([(byte)t.DType, (byte)t.Shape.Rank]);
        Span<byte> d = stackalloc byte[8];
        for (int i = 0; i < t.Shape.Rank; i++)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(d, t.Shape[i]);
            h.AppendData(d);
        }

        bool copied = !t.IsContiguous;
        Tensor c = copied ? t.Contiguous() : t;
        try
        {
            switch (t.DType)
            {
                case DType.F32: h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(c.Data<float>())); break;
                case DType.F64: h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(c.Data<double>())); break;
                case DType.I64: h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(c.Data<long>())); break;
                case DType.U64: h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(c.Data<ulong>())); break;
                case DType.I32: h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(c.Data<int>())); break;
                case DType.U32: h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(c.Data<uint>())); break;
                case DType.I16: h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(c.Data<short>())); break;
                case DType.U16: h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(c.Data<ushort>())); break;
                case DType.I8: h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(c.Data<sbyte>())); break;
                case DType.U8: h.AppendData(c.Data<byte>()); break;
                case DType.Bool: h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(c.Data<bool>())); break;
                case DType.F16: h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(c.Data<Half>())); break;
                case DType.BF16: h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(c.Data<BFloat16>())); break;
                default: throw new NotSupportedException($"cannot hash {t.DType}");
            }
        }
        finally
        {
            if (copied && !c.IsEmpty)
            {
                c.Dispose();
            }
        }
    }
}
