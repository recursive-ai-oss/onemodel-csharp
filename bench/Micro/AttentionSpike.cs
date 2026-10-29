using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using RecursiveAI.OneModel.Kernels;

namespace RecursiveAI.OneModel.Benchmarks;

/// <summary>
/// Times fused attention, softmax and the elementwise transcendentals in their fast and exact flavours, for the shapes of the
/// reference encoders and decoders, and prints one markdown table. Run as <c>Micro attention-spike [reps] [outfile]</c>.
/// Work is split into the kernel's fixed work items; threads only decide who runs which item, never what an item computes.
/// </summary>
internal static class AttentionSpike
{
    private sealed record Shape(string Name, int Heads, int Dim, int QLen, int KLen);

    private static readonly Shape[] Shapes =
    [
        new("BERT-base seq128", 12, 64, 128, 128),
        new("BERT-base seq512", 12, 64, 512, 512),
        new("MiniLM-L6 seq128", 12, 32, 128, 128),
        new("Whisper-small encoder self 1500", 12, 64, 1500, 1500),
        new("Whisper-small decoder cross, 1 step", 12, 64, 1, 1500),
        new("Whisper-small decoder cross, 448 steps", 12, 64, 448, 1500),
    ];

    [ThreadStatic]
    private static float[]? s_scratch;

    public static int Run(string[] args)
    {
        int reps = args.Length > 0 ? int.Parse(args[0], CultureInfo.InvariantCulture) : 3;
        string? outFile = args.Length > 1 ? args[1] : null;
        var sb = new StringBuilder();
        void Line(string s)
        {
            Console.WriteLine(s);
            sb.AppendLine(s);
        }

        Line("## Machine");
        Line($"- cpu: {CpuName()}");
        Line($"- logical cores: {Environment.ProcessorCount}; os: {Environment.OSVersion}; runtime: {Environment.Version}; arch: {System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture}");
        Line($"- isa: avx2={Avx2.IsSupported} fma={Fma.IsSupported} avx512f={Avx512F.IsSupported} avx512vnni={(Avx512F.IsSupported && AvxVnni.IsSupported)} advsimd={AdvSimd.IsSupported} dotprod={Dp.IsSupported} " +
             $"accelerated: v128={Vector128.IsHardwareAccelerated} v256={Vector256.IsHardwareAccelerated} v512={Vector512.IsHardwareAccelerated}");
        string loadBefore = Load();
        Line($"- load before: {loadBefore}");

        var isas = new List<VecIsa>();
        if (Vector128.IsHardwareAccelerated)
        {
            isas.Add(VecIsa.V128);
        }

        if (Vector256.IsHardwareAccelerated && Avx2.IsSupported)
        {
            isas.Add(VecIsa.V256);
        }

        if (Vector512.IsHardwareAccelerated)
        {
            isas.Add(VecIsa.V512);
        }

        // Subsets for quick looks: ONEMODEL_SPIKE_THREADS=1,32  ONEMODEL_SPIKE_SHAPES=0,2  ONEMODEL_SPIKE_PARTS=attention,softmax,elementwise
        string? tEnv = Environment.GetEnvironmentVariable("ONEMODEL_SPIKE_THREADS");
        string? iEnv = Environment.GetEnvironmentVariable("ONEMODEL_SPIKE_ISAS");
        if (iEnv is not null)
        {
            isas.RemoveAll(i => !iEnv.Contains(i.ToString(), StringComparison.OrdinalIgnoreCase));
        }

        int[] threadCounts = tEnv is null ? [1, Environment.ProcessorCount] : Array.ConvertAll(tEnv.Split(','), x => int.Parse(x, CultureInfo.InvariantCulture));
        string? sEnv = Environment.GetEnvironmentVariable("ONEMODEL_SPIKE_SHAPES");
        Shape[] shapes = sEnv is null ? Shapes : Array.ConvertAll(sEnv.Split(','), x => Shapes[int.Parse(x, CultureInfo.InvariantCulture)]);
        string parts = Environment.GetEnvironmentVariable("ONEMODEL_SPIKE_PARTS") ?? "attention,softmax,elementwise,safety";
        void AttentionPart()
        {
            Line(string.Empty);
            Line("## Attention (batch 1, fp32); times are per call, median of reps, ms");
            Line(string.Empty);
            Line("| shape | isa | threads | fast ms | exact ms | exact/fast | exact output SHA-256 (first 16 hex) |");
            Line("|---|---|---|---|---|---|---|");
            foreach (Shape sh in shapes)
            {
                Inputs inp = MakeInputs(sh);
                foreach (VecIsa isa in isas)
                {
                    string? hash1 = null;
                    foreach (int threads in threadCounts)
                    {
                        double fast = Time(reps, () => RunAttention(inp, false, isa, threads, inp.Out));
                        double exact = Time(reps, () => RunAttention(inp, true, isa, threads, inp.Out));
                        RunAttention(inp, true, isa, threads, inp.Out);
                        string h = Convert.ToHexString(SHA256.HashData(System.Runtime.InteropServices.MemoryMarshal.AsBytes<float>(inp.Out)))[..16].ToLowerInvariant();
                        hash1 ??= h;
                        string hs = h == hash1 ? h : $"{h} MISMATCH vs 1 thread";
                        Line(string.Create(CultureInfo.InvariantCulture, $"| {sh.Name} | {isa} | {threads} | {fast * 1e3:F3} | {exact * 1e3:F3} | {exact / fast:F2} | {hs} |"));
                    }
                }
            }

        }

        void SoftmaxPart()
        {
            Line(string.Empty);
            Line("## Softmax rows (rows = heads x queries, cols = keys), 1 thread, median of reps, ms");
            Line(string.Empty);
            Line("| shape | isa | fast ms | exact ms | exact/fast |");
            Line("|---|---|---|---|---|");
            foreach (Shape sh in shapes)
            {
                int rows = sh.Heads * sh.QLen;
                float[] x = Random(rows * sh.KLen, 5, 2.0f);
                var y = new float[x.Length];
                foreach (VecIsa isa in isas)
                {
                    double fast = Time(reps, () => Softmax.Rows(false, isa, x, y, rows, sh.KLen, sh.KLen, sh.KLen));
                    double exact = Time(reps, () => Softmax.Rows(true, isa, x, y, rows, sh.KLen, sh.KLen, sh.KLen));
                    Line(string.Create(CultureInfo.InvariantCulture, $"| {sh.Name} | {isa} | {fast * 1e3:F3} | {exact * 1e3:F3} | {exact / fast:F2} |"));
                }
            }

        }

        void Elementwise()
        {
            Line(string.Empty);
            Line("## Elementwise transcendentals, 1 thread, 1M elements, ns per element (median of reps)");
            Line(string.Empty);
            Line("| function | isa | fast ns/elem | exact ns/elem | exact/fast |");
            Line("|---|---|---|---|---|");
            float[] xs = Random(1 << 20, 7, 2.0f);
            var ys = new float[xs.Length];
            foreach (ElementwiseFn fn in Enum.GetValues<ElementwiseFn>())
            {
                if (fn == ElementwiseFn.Log)
                {
                    for (int i = 0; i < xs.Length; i++)
                    {
                        xs[i] = Math.Abs(xs[i]) + 0.01f;
                    }
                }

                foreach (VecIsa isa in isas)
                {
                    double fast = Time(reps, () => ElementwiseMath.Apply(fn, false, isa, xs, ys));
                    double exact = Time(reps, () => ElementwiseMath.Apply(fn, true, isa, xs, ys));
                    Line(string.Create(CultureInfo.InvariantCulture, $"| {fn} | {isa} | {fast / xs.Length * 1e9:F2} | {exact / xs.Length * 1e9:F2} | {exact / fast:F2} |"));
                }
            }

        }

        void SafetyPart()
        {
            if (!Fma.IsSupported || !Avx2.IsSupported)
            {
                return;
            }

            Line(string.Empty);
            Line("## Bounds checks versus unchecked refs: one 4 x 16 score tile over a head dimension of 64 (AVX2 + FMA), 1 thread, ns per tile");
            Line(string.Empty);
            float[] kt = Random(64 * 64, 11, 1f);
            float[] q = Random(4 * 64, 12, 1f);
            var tile = new float[4 * 64];
            double uncheckedTime = Time(reps, () =>
            {
                for (int i = 0; i < 20000; i++)
                {
                    TileUnchecked(kt, q, tile, i & 3);
                }
            }) / 20000;
            double checkedTime = Time(reps, () =>
            {
                for (int i = 0; i < 20000; i++)
                {
                    TileChecked(kt, q, tile, i & 3);
                }
            }) / 20000;
            Line("| variant | ns per tile |");
            Line("|---|---|");
            Line(string.Create(CultureInfo.InvariantCulture, $"| unchecked refs | {uncheckedTime * 1e9:F1} |"));
            Line(string.Create(CultureInfo.InvariantCulture, $"| span indexing with bounds checks | {checkedTime * 1e9:F1} ({checkedTime / uncheckedTime:F2}x) |"));
        }

        if (parts.Contains("attention", StringComparison.Ordinal))
        {
            AttentionPart();
        }

        if (parts.Contains("safety", StringComparison.Ordinal))
        {
            SafetyPart();
        }

        if (parts.Contains("softmax", StringComparison.Ordinal))
        {
            SoftmaxPart();
        }

        if (parts.Contains("elementwise", StringComparison.Ordinal))
        {
            Elementwise();
        }

        Line(string.Empty);
        Line($"- load after: {Load()}");
        if (outFile is not null)
        {
            File.WriteAllText(outFile, sb.ToString());
        }

        return 0;
    }

    // The 4 x 16 tile of the attention score microkernel, written twice: once as the library does (refs, no bounds checks) and once
    // with ordinary span indexing, to put a number on what the bounds checks cost.
    private static void TileUnchecked(float[] kt, float[] q, float[] tile, int colOffset)
    {
        ref float k = ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(kt);
        ref float qq = ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(q);
        Vector256<float> a00 = Vector256<float>.Zero, a01 = a00, a10 = a00, a11 = a00, a20 = a00, a21 = a00, a30 = a00, a31 = a00;
        int cg = colOffset * 16;
        for (int d = 0; d < 64; d++)
        {
            Vector256<float> b0 = Vector256.LoadUnsafe(ref k, (nuint)((d * 64) + cg));
            Vector256<float> b1 = Vector256.LoadUnsafe(ref k, (nuint)((d * 64) + cg + 8));
            Vector256<float> x0 = Vector256.Create(System.Runtime.CompilerServices.Unsafe.Add(ref qq, d));
            Vector256<float> x1 = Vector256.Create(System.Runtime.CompilerServices.Unsafe.Add(ref qq, 64 + d));
            Vector256<float> x2 = Vector256.Create(System.Runtime.CompilerServices.Unsafe.Add(ref qq, 128 + d));
            Vector256<float> x3 = Vector256.Create(System.Runtime.CompilerServices.Unsafe.Add(ref qq, 192 + d));
            a00 = Fma.MultiplyAdd(x0, b0, a00);
            a01 = Fma.MultiplyAdd(x0, b1, a01);
            a10 = Fma.MultiplyAdd(x1, b0, a10);
            a11 = Fma.MultiplyAdd(x1, b1, a11);
            a20 = Fma.MultiplyAdd(x2, b0, a20);
            a21 = Fma.MultiplyAdd(x2, b1, a21);
            a30 = Fma.MultiplyAdd(x3, b0, a30);
            a31 = Fma.MultiplyAdd(x3, b1, a31);
        }

        ref float t = ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(tile);
        a00.StoreUnsafe(ref t, 0);
        a01.StoreUnsafe(ref t, 8);
        a10.StoreUnsafe(ref t, 64);
        a11.StoreUnsafe(ref t, 72);
        a20.StoreUnsafe(ref t, 128);
        a21.StoreUnsafe(ref t, 136);
        a30.StoreUnsafe(ref t, 192);
        a31.StoreUnsafe(ref t, 200);
    }

    private static void TileChecked(float[] kt, float[] q, float[] tile, int colOffset)
    {
        Vector256<float> a00 = Vector256<float>.Zero, a01 = a00, a10 = a00, a11 = a00, a20 = a00, a21 = a00, a30 = a00, a31 = a00;
        int cg = colOffset * 16;
        for (int d = 0; d < 64; d++)
        {
            Vector256<float> b0 = Vector256.Create(kt.AsSpan((d * 64) + cg, 8));
            Vector256<float> b1 = Vector256.Create(kt.AsSpan((d * 64) + cg + 8, 8));
            Vector256<float> x0 = Vector256.Create(q[d]);
            Vector256<float> x1 = Vector256.Create(q[64 + d]);
            Vector256<float> x2 = Vector256.Create(q[128 + d]);
            Vector256<float> x3 = Vector256.Create(q[192 + d]);
            a00 = Fma.MultiplyAdd(x0, b0, a00);
            a01 = Fma.MultiplyAdd(x0, b1, a01);
            a10 = Fma.MultiplyAdd(x1, b0, a10);
            a11 = Fma.MultiplyAdd(x1, b1, a11);
            a20 = Fma.MultiplyAdd(x2, b0, a20);
            a21 = Fma.MultiplyAdd(x2, b1, a21);
            a30 = Fma.MultiplyAdd(x3, b0, a30);
            a31 = Fma.MultiplyAdd(x3, b1, a31);
        }

        a00.CopyTo(tile.AsSpan(0, 8));
        a01.CopyTo(tile.AsSpan(8, 8));
        a10.CopyTo(tile.AsSpan(64, 8));
        a11.CopyTo(tile.AsSpan(72, 8));
        a20.CopyTo(tile.AsSpan(128, 8));
        a21.CopyTo(tile.AsSpan(136, 8));
        a30.CopyTo(tile.AsSpan(192, 8));
        a31.CopyTo(tile.AsSpan(200, 8));
    }

    private sealed class Inputs
    {
        public AttentionShape Shape;
        public float[] Q = [];
        public float[] K = [];
        public float[] V = [];
        public float[] Out = [];
    }

    private static Inputs MakeInputs(Shape sh)
    {
        AttentionShape s = AttentionShape.Contiguous(1, sh.Heads, sh.QLen, sh.KLen, sh.Dim, 1f / MathF.Sqrt(sh.Dim));
        return new Inputs
        {
            Shape = s,
            Q = Random(sh.Heads * sh.QLen * sh.Dim, 1, 1.5f),
            K = Random(sh.Heads * sh.KLen * sh.Dim, 2, 1.5f),
            V = Random(sh.Heads * sh.KLen * sh.Dim, 3, 1.0f),
            Out = new float[sh.Heads * sh.QLen * sh.Dim],
        };
    }

    private static float[] Random(int n, int seed, float scale)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = ((float)rng.NextDouble() - 0.5f) * 2f * scale;
        }

        return a;
    }

    private static void RunAttention(Inputs inp, bool exact, VecIsa isa, int threads, float[] output)
    {
        AttentionShape s = inp.Shape;
        int items = Attention.ItemCount(s);
        if (threads <= 1)
        {
            float[] scratch = s_scratch is { } sc && sc.Length >= Attention.ScratchLength(s.HeadDim) ? sc : (s_scratch = new float[Attention.ScratchLength(s.HeadDim)]);
            Attention.Compute(exact, isa, s, inp.Q, inp.K, inp.V, default, output, 0, items, scratch);
            return;
        }

        var po = new ParallelOptions { MaxDegreeOfParallelism = threads };
        Parallel.For(0, items, po, item =>
        {
            float[] scratch = s_scratch is { } sc && sc.Length >= Attention.ScratchLength(s.HeadDim) ? sc : (s_scratch = new float[Attention.ScratchLength(s.HeadDim)]);
            Attention.Compute(exact, isa, s, inp.Q, inp.K, inp.V, default, output, item, item + 1, scratch);
        });
    }

    /// <summary>Median over <paramref name="reps"/> repetitions of the mean time of a batch of calls lasting at least 150 ms (after warm-up), in seconds.</summary>
    private static double Time(int reps, Action action)
    {
        action();
        action();
        var times = new double[reps];
        for (int r = 0; r < reps; r++)
        {
            int n = 0;
            long start = Stopwatch.GetTimestamp();
            double elapsed;
            do
            {
                action();
                n++;
                elapsed = Stopwatch.GetElapsedTime(start).TotalSeconds;
            }
            while (elapsed < 0.15 || n < 3);
            times[r] = elapsed / n;
        }

        Array.Sort(times);
        return times[reps / 2];
    }

    private static string CpuName()
    {
        try
        {
            if (File.Exists("/proc/cpuinfo"))
            {
                foreach (string l in File.ReadLines("/proc/cpuinfo"))
                {
                    if (l.StartsWith("model name", StringComparison.Ordinal))
                    {
                        return l[(l.IndexOf(':', StringComparison.Ordinal) + 1)..].Trim();
                    }
                }
            }

            return Shell("sysctl", "-n machdep.cpu.brand_string").Trim();
        }
        catch (Exception ex) when (ex is IOException or System.ComponentModel.Win32Exception or UnauthorizedAccessException)
        {
            return "unknown";
        }
    }

    private static string Load()
    {
        try
        {
            if (File.Exists("/proc/loadavg"))
            {
                return File.ReadAllText("/proc/loadavg").Trim();
            }

            return Shell("sysctl", "-n vm.loadavg").Trim();
        }
        catch (Exception ex) when (ex is IOException or System.ComponentModel.Win32Exception or UnauthorizedAccessException)
        {
            return "unknown";
        }
    }

    private static string Shell(string file, string args)
    {
        using var p = Process.Start(new ProcessStartInfo(file, args) { RedirectStandardOutput = true, UseShellExecute = false })!;
        string s = p.StandardOutput.ReadToEnd();
        p.WaitForExit();
        return s;
    }
}
