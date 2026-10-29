using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Text;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tests.Kernels;

namespace RecursiveAI.OneModel.Benchmarks;

/// <summary>
/// The exact-versus-fast measurement for matrix products and convolutions, and the cross-path hash table.
/// <c>spike hashes</c> prints SHA-256 hashes of kernel outputs per path and thread count; <c>spike table</c> times the reference
/// shapes. See docs/kernels-gemm.md for the method.
/// </summary>
internal static class Spike
{
    private sealed record Item(string Model, string Name, int Count, bool IsConv, int M, int N, int K, ConvShape Conv)
    {
        public double Flops => IsConv ? Conv.Flops : 2.0 * M * N * K;
    }

    public static int Run(string[] args)
    {
        if (args.Length == 0)
        {
            Console.Error.WriteLine("usage: spike hashes | spike table [--ghz 4.5] [--fma256 2] [--fma512 1] [--neon-pipes 4] [--cores N] [--threads N] [--filter text] [--csv file]");
            return 2;
        }

        var opt = new Dictionary<string, string>();
        for (int i = 1; i + 1 < args.Length; i += 2)
        {
            opt[args[i]] = args[i + 1];
        }

        PrintMachine();
        return args[0] switch
        {
            "hashes" => Hashes(),
            "table" => Table(opt),
            "safe-vs-pointer" => SafeVersusPointer(),
            _ => 2,
        };
    }

    private static void PrintMachine()
    {
        string flags = string.Join(' ', new[]
        {
            Avx2.IsSupported ? "avx2" : null,
            Fma.IsSupported ? "fma" : null,
            Avx512F.IsSupported ? "avx512f" : null,
            AvxVnni.IsSupported ? "avxvnni" : null,
            AdvSimd.Arm64.IsSupported ? "neon" : null,
            Dp.IsSupported ? "dotprod" : null,
        }.Where(f => f is not null));
        Console.WriteLine($"# runtime {Environment.Version}, os {Environment.OSVersion}, logical processors {Environment.ProcessorCount}, isa [{flags}]");
        Console.WriteLine("# paths: " + string.Join(", ", KernelPaths.All.Where(KernelPaths.IsSupported)));
    }

    // ---------------------------------------------------------------- why the microkernels use pointers

    private static void SafeKernel(int kc, ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int ldc)
    {
        // The exact AVX2 6x16 microkernel written with spans and bounds checks only (accumulating from +0).
        var acc = new Vector256<float>[12];
        for (int p = 0; p < kc; p++)
        {
            Vector256<float> b0 = Vector256.Create(b.Slice(p * 16, 8));
            Vector256<float> b1 = Vector256.Create(b.Slice((p * 16) + 8, 8));
            for (int r = 0; r < 6; r++)
            {
                Vector256<float> av = Vector256.Create(a[(p * 6) + r]);
                acc[r * 2] = acc[r * 2] + (av * b0);
                acc[(r * 2) + 1] = acc[(r * 2) + 1] + (av * b1);
            }
        }

        for (int r = 0; r < 6; r++)
        {
            acc[r * 2].CopyTo(c.Slice(r * ldc, 8));
            acc[(r * 2) + 1].CopyTo(c.Slice((r * ldc) + 8, 8));
        }
    }

    private static unsafe int SafeVersusPointer()
    {
        if (!Avx2.IsSupported)
        {
            Console.WriteLine("AVX2 not supported; nothing to compare");
            return 0;
        }

        const int kc = 256;
        var rng = new KernelTestData.Rng(5);
        var a = new float[kc * 6];
        var b = new float[kc * 16];
        var cs = new float[6 * 16];
        var cp = new float[6 * 16];
        KernelTestData.Fill(ref rng, a, false);
        KernelTestData.Fill(ref rng, b, false);
        SafeKernel(kc, a, b, cs, 16);
        fixed (float* pa = a, pb = b, pc = cp)
        {
            Kernels.Exact.ExactGemmAvx2.Kernel(kc, pa, pb, pc, 16, 0);
        }

        Console.WriteLine("outputs identical: " + KernelTestData.SameBits(cs, cp));
        const int calls = 200000;
        for (int rep = 0; rep < 3; rep++)
        {
            long t0 = Stopwatch.GetTimestamp();
            for (int i = 0; i < calls; i++)
            {
                SafeKernel(kc, a, b, cs, 16);
            }

            double safe = Stopwatch.GetElapsedTime(t0).TotalSeconds;
            t0 = Stopwatch.GetTimestamp();
            fixed (float* pa = a, pb = b, pc = cp)
            {
                for (int i = 0; i < calls; i++)
                {
                    Kernels.Exact.ExactGemmAvx2.Kernel(kc, pa, pb, pc, 16, 0);
                }
            }

            double ptr = Stopwatch.GetElapsedTime(t0).TotalSeconds;
            double flops = 2.0 * 6 * 16 * kc * calls;
            Console.WriteLine($"span+bounds checks {flops / safe / 1e9:F1} GFLOP/s, pointer microkernel {flops / ptr / 1e9:F1} GFLOP/s, ratio {safe / ptr:F2}x");
        }

        return 0;
    }

    // ---------------------------------------------------------------- hashes

    private static int Hashes()
    {
        int threadsAll = Environment.ProcessorCount;
        int bad = 0;
        var paths = KernelPaths.All.Where(KernelPaths.IsSupported).ToArray();
        foreach (bool special in new[] { false, true })
        {
            foreach (KernelMode mode in new[] { KernelMode.Exact, KernelMode.Fast })
            {
                var seen = new Dictionary<string, List<string>>();
                foreach (KernelPath path in paths)
                {
                    foreach (int threads in new[] { 1, threadsAll })
                    {
                        string h = KernelTestData.SweepHash(mode, path, threads, 256, 300, special);
                        Console.WriteLine($"gemm-sweep special={special} {mode} {path} threads={threads} {h}");
                        if (!seen.TryGetValue(h, out List<string>? l))
                        {
                            seen[h] = l = [];
                        }

                        l.Add($"{path}/{threads}");
                    }
                }

                Console.WriteLine($"gemm-sweep special={special} {mode}: {seen.Count} distinct hash(es)" + (mode == KernelMode.Exact && seen.Count != 1 ? "  <-- MISMATCH" : string.Empty));
                if (mode == KernelMode.Exact && seen.Count != 1)
                {
                    bad++;
                }
            }
        }

        // Reference shapes: every distinct shape of the table, seeded data, exact mode only (that is where bit-equality is promised).
        var all = new Dictionary<string, List<string>>();
        foreach (Item it in Items().GroupBy(i => i.Name).Select(g => g.First()))
        {
            if (it.IsConv ? it.Conv.Flops > 3e9 : it.Flops > 6e9)
            {
                continue;
            }

            foreach (KernelPath path in paths)
            {
                foreach (int threads in new[] { 1, threadsAll })
                {
                    string h = HashItem(it, KernelMode.Exact, path, threads);
                    if (!all.TryGetValue(it.Name, out List<string>? l))
                    {
                        all[it.Name] = l = [];
                    }

                    l.Add(h);
                }
            }
        }

        foreach ((string name, List<string> hs) in all)
        {
            bool eq = hs.Distinct().Count() == 1;
            Console.WriteLine($"ref-shape exact {name} {hs[0]} {(eq ? "equal" : "MISMATCH")} ({hs.Count} runs)");
            if (!eq)
            {
                bad++;
            }
        }

        Console.WriteLine(bad == 0 ? "# all exact hashes equal across the paths and thread counts run here" : $"# {bad} exact MISMATCH(es)");
        return bad == 0 ? 0 : 1;
    }

    private static string HashItem(Item it, KernelMode mode, KernelPath path, int threads)
    {
        var rng = new KernelTestData.Rng(OneModelInfo.Fnv1a(System.Text.Encoding.UTF8.GetBytes(it.Name)));
        if (!it.IsConv)
        {
            var a = new float[it.M * it.K];
            var b = new float[it.K * it.N];
            var c = new float[it.M * it.N];
            KernelTestData.Fill(ref rng, a, false);
            KernelTestData.Fill(ref rng, b, false);
            Gemm.Run(mode, path, it.M, it.N, it.K, a, it.K, b, it.N, c, it.N, threads);
            return Sha(c);
        }

        ConvShape s = it.Conv;
        var input = new float[s.N * s.Cin * s.H * s.W];
        var w = new float[s.Cout * (s.Cin / s.Groups) * s.KH * s.KW];
        var o = new float[s.N * s.Cout * s.OutH * s.OutW];
        KernelTestData.Fill(ref rng, input, false);
        KernelTestData.Fill(ref rng, w, false);
        Conv.Run(mode, path, s, input, w, o, threads);
        return Sha(o);
    }

    private static string Sha(float[] data) =>
        Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(data.AsSpan()))).ToLowerInvariant();

    // ---------------------------------------------------------------- reference shapes

    private static ConvShape Conv2(int cin, int cout, int res, int k, int stride, int groups = 1, int n = 1) =>
        new(n, cin, res, res, cout, k, k, stride, stride, k / 2, k / 2, k / 2, k / 2, 1, 1, groups);

    private static List<Item> Items()
    {
        var items = new List<Item>();
        void Gemm(string model, int count, int m, int n, int k) =>
            items.Add(new Item(model, $"gemm {m}x{n}x{k}", count, false, m, n, k, default));
        void Convolution(string model, int count, ConvShape s) =>
            items.Add(new Item(model, $"conv {(s.IsDepthwise ? "dw" : "std")} {s.Cin}->{s.Cout} {s.KH}x{s.KW}/{s.StrideH} @{s.H}", count, true, 0, 0, 0, s));

        // Encoders at batch 32 x sequence 128 = 4096 rows (Whisper-small encoder: 1500 frames, batch 1). The attention score and
        // context products belong to the attention kernels and are not counted here.
        const string bert = "BERT-base";
        Gemm(bert, 48, 4096, 768, 768);
        Gemm(bert, 12, 4096, 3072, 768);
        Gemm(bert, 12, 4096, 768, 3072);
        const string minilm = "MiniLM-L6";
        Gemm(minilm, 24, 4096, 384, 384);
        Gemm(minilm, 6, 4096, 1536, 384);
        Gemm(minilm, 6, 4096, 384, 1536);
        const string whisper = "Whisper-small encoder";
        Gemm(whisper, 48, 1500, 768, 768);
        Gemm(whisper, 12, 1500, 3072, 768);
        Gemm(whisper, 12, 1500, 768, 3072);
        Convolution(whisper, 1, new ConvShape(1, 80, 1, 3000, 768, 1, 3, 1, 1, 0, 1, 0, 1, 1, 1, 1));
        Convolution(whisper, 1, new ConvShape(1, 768, 1, 3000, 768, 1, 3, 1, 2, 0, 1, 0, 1, 1, 1, 1));
        // Batch-1 rows of the same GEMMs, for the one-sequence case.
        Gemm("BERT-base (1 x 128)", 48, 128, 768, 768);
        Gemm("BERT-base (1 x 128)", 12, 128, 3072, 768);
        Gemm("BERT-base (1 x 128)", 12, 128, 768, 3072);

        // ResNet-50 v1.5 at 224 x 224, batch 1: the stride sits on the 3x3 convolution of the first block of stages 2 to 4.
        const string resnet = "ResNet-50";
        Convolution(resnet, 1, Conv2(3, 64, 224, 7, 2));
        int[] blocks = [3, 4, 6, 3];
        int[] mids = [64, 128, 256, 512];
        int res = 56;
        int cin = 64;
        for (int st = 0; st < 4; st++)
        {
            int mid = mids[st];
            int cout = mid * 4;
            int resOut = st == 0 ? res : res / 2;
            for (int b = 0; b < blocks[st]; b++)
            {
                bool first = b == 0;
                int inRes = first ? res : resOut;
                int stride = first && st > 0 ? 2 : 1;
                Convolution(resnet, 1, Conv2(first ? cin : cout, mid, inRes, 1, 1));
                Convolution(resnet, 1, Conv2(mid, mid, inRes, 3, stride));
                Convolution(resnet, 1, Conv2(mid, cout, resOut, 1, 1));
                if (first)
                {
                    Convolution(resnet, 1, Conv2(cin, cout, inRes, 1, stride));
                }
            }

            res = resOut;
            cin = cout;
        }

        // MobileNetV3-large at 224 x 224, batch 1 (squeeze-excite FCs and the classifier are tiny and left out).
        const string mobile = "MobileNetV3-large";
        Convolution(mobile, 1, Conv2(3, 16, 224, 3, 2));
        (int K, int Exp, int Out, int Stride)[] cfg =
        [
            (3, 16, 16, 1), (3, 64, 24, 2), (3, 72, 24, 1), (5, 72, 40, 2), (5, 120, 40, 1), (5, 120, 40, 1), (3, 240, 80, 2),
            (3, 200, 80, 1), (3, 184, 80, 1), (3, 184, 80, 1), (3, 480, 112, 1), (3, 672, 112, 1), (5, 672, 160, 2), (5, 960, 160, 1), (5, 960, 160, 1),
        ];
        int c = 16;
        int r = 112;
        foreach ((int k, int exp, int outC, int stride) in cfg)
        {
            if (exp != c)
            {
                Convolution(mobile, 1, Conv2(c, exp, r, 1, 1));
            }

            Convolution(mobile, 1, Conv2(exp, exp, r, k, stride, exp));
            int rOut = stride == 2 ? r / 2 : r;
            Convolution(mobile, 1, Conv2(exp, outC, rOut, 1, 1));
            c = outC;
            r = rOut;
        }

        Convolution(mobile, 1, Conv2(160, 960, 7, 1, 1));

        // Merge identical shapes inside a model.
        return items
            .GroupBy(i => (i.Model, i.Name))
            .Select(g => g.First() with { Count = g.Sum(x => x.Count) })
            .ToList();
    }

    // ---------------------------------------------------------------- table

    private sealed class Rig
    {
        public float[] A = [];
        public float[] B = [];
        public float[] C = [];
    }

    private static Action Prepare(Item it, KernelMode mode, KernelPath path, int threads)
    {
        var rng = new KernelTestData.Rng(42);
        if (!it.IsConv)
        {
            var a = new float[it.M * it.K];
            var b = new float[it.K * it.N];
            var c = new float[it.M * it.N];
            KernelTestData.Fill(ref rng, a, false);
            KernelTestData.Fill(ref rng, b, false);
            return () => Gemm.Run(mode, path, it.M, it.N, it.K, a, it.K, b, it.N, c, it.N, threads);
        }

        ConvShape s = it.Conv;
        var input = new float[s.N * s.Cin * s.H * s.W];
        var w = new float[s.Cout * (s.Cin / s.Groups) * s.KH * s.KW];
        var o = new float[s.N * s.Cout * s.OutH * s.OutW];
        KernelTestData.Fill(ref rng, input, false);
        KernelTestData.Fill(ref rng, w, false);
        return () => Conv.Run(mode, path, s, input, w, o, threads);
    }

    /// <summary>Median over three repetitions of the mean time per call, each repetition running at least 80 ms (and at least one call) after two warm-up calls.</summary>
    private static double Time(Action f)
    {
        f();
        f();
        var reps = new double[3];
        for (int rep = 0; rep < reps.Length; rep++)
        {
            int calls = 0;
            long start = Stopwatch.GetTimestamp();
            double elapsed;
            do
            {
                f();
                calls++;
                elapsed = Stopwatch.GetElapsedTime(start).TotalSeconds;
            }
            while (elapsed < 0.08);
            reps[rep] = elapsed / calls;
        }

        Array.Sort(reps);
        return reps[1];
    }

    private static int Table(Dictionary<string, string> opt)
    {
        double ghz = double.Parse(opt.GetValueOrDefault("--ghz", "0"), CultureInfo.InvariantCulture);
        double fma256 = double.Parse(opt.GetValueOrDefault("--fma256", "2"), CultureInfo.InvariantCulture);
        double fma512 = double.Parse(opt.GetValueOrDefault("--fma512", "1"), CultureInfo.InvariantCulture);
        double neonPipes = double.Parse(opt.GetValueOrDefault("--neon-pipes", "4"), CultureInfo.InvariantCulture);
        int cores = int.Parse(opt.GetValueOrDefault("--cores", Environment.ProcessorCount.ToString(CultureInfo.InvariantCulture)), CultureInfo.InvariantCulture);
        int allThreads = int.Parse(opt.GetValueOrDefault("--threads", Environment.ProcessorCount.ToString(CultureInfo.InvariantCulture)), CultureInfo.InvariantCulture);
        string filter = opt.GetValueOrDefault("--filter", string.Empty);
        string csvPath = opt.GetValueOrDefault("--csv", string.Empty);
        var paths = KernelPaths.All.Where(p => KernelPaths.IsSupported(p) && p is KernelPath.Avx2 or KernelPath.Avx512 or KernelPath.Neon).ToArray();
        if (paths.Length == 0)
        {
            paths = [KernelPaths.Best()];
        }

        double PeakPerCore(KernelPath p) => p switch
        {
            KernelPath.Avx2 => ghz * fma256 * 16,
            KernelPath.Avx512 => ghz * fma512 * 32,
            KernelPath.Neon => ghz * neonPipes * 8,
            _ => ghz * 2,
        };

        List<Item> items = Items();
        var unique = items.GroupBy(i => i.Name).Select(g => g.First()).Where(i => i.Name.Contains(filter, StringComparison.Ordinal)).ToList();

        // Sanity line: a naive triple loop against the kernels, single thread.
        {
            const int m = 128, n = 768, k = 768;
            var rng = new KernelTestData.Rng(7);
            var a = new float[m * k];
            var b = new float[k * n];
            var c = new float[m * n];
            KernelTestData.Fill(ref rng, a, false);
            KernelTestData.Fill(ref rng, b, false);
            double t = Time(() =>
            {
                Array.Clear(c);
                for (int i = 0; i < m; i++)
                {
                    for (int p = 0; p < k; p++)
                    {
                        float av = a[(i * k) + p];
                        for (int j = 0; j < n; j++)
                        {
                            c[(i * n) + j] += av * b[(p * n) + j];
                        }
                    }
                }
            });
            Console.WriteLine($"# naive i-k-j loop 128x768x768, 1 thread: {2.0 * m * n * k / t / 1e9:F2} GFLOP/s (JIT may auto-vectorise nothing; scalar code)");
        }

        var results = new Dictionary<(string Name, KernelPath Path, int Threads, KernelMode Mode), double>();
        var csv = new StringBuilder("shape,flops,path,threads,mode,seconds,gflops,peak_pct\n");
        var md = new StringBuilder();
        md.AppendLine("| shape | GFLOP | path | threads | fast GFLOP/s | exact GFLOP/s | exact/fast time | fast % peak | exact % peak |");
        md.AppendLine("|---|---|---|---|---|---|---|---|---|");
        foreach (Item it in unique)
        {
            foreach (KernelPath path in paths)
            {
                foreach (int threads in new[] { 1, allThreads })
                {
                    double tf = Time(Prepare(it, KernelMode.Fast, path, threads));
                    double te = Time(Prepare(it, KernelMode.Exact, path, threads));
                    results[(it.Name, path, threads, KernelMode.Fast)] = tf;
                    results[(it.Name, path, threads, KernelMode.Exact)] = te;
                    double peak = ghz > 0 ? PeakPerCore(path) * Math.Min(threads, cores) : 0;
                    double gf = it.Flops / tf / 1e9;
                    double ge = it.Flops / te / 1e9;
                    string pf = peak > 0 ? $"{100 * gf / peak:F0}%" : "n/a";
                    string pe = peak > 0 ? $"{100 * ge / peak:F0}%" : "n/a";
                    md.AppendLine($"| {it.Name} | {it.Flops / 1e9:F3} | {path} | {threads} | {gf:F1} | {ge:F1} | {te / tf:F2} | {pf} | {pe} |");
                    csv.AppendLine(string.Create(CultureInfo.InvariantCulture, $"{it.Name},{it.Flops},{path},{threads},fast,{tf},{gf},{(peak > 0 ? 100 * gf / peak : 0)}"));
                    csv.AppendLine(string.Create(CultureInfo.InvariantCulture, $"{it.Name},{it.Flops},{path},{threads},exact,{te},{ge},{(peak > 0 ? 100 * ge / peak : 0)}"));
                }
            }
        }

        Console.WriteLine(md.ToString());

        Console.WriteLine("| model | path | threads | GEMM/conv GFLOP | fast time (s) | exact time (s) | exact/fast |");
        Console.WriteLine("|---|---|---|---|---|---|---|");
        foreach (string model in items.Select(i => i.Model).Distinct())
        {
            foreach (KernelPath path in paths)
            {
                foreach (int threads in new[] { 1, allThreads })
                {
                    double sf = 0, se = 0, fl = 0;
                    bool complete = true;
                    foreach (Item it in items.Where(i => i.Model == model))
                    {
                        if (!results.TryGetValue((it.Name, path, threads, KernelMode.Fast), out double tf))
                        {
                            complete = false;
                            break;
                        }

                        sf += it.Count * tf;
                        se += it.Count * results[(it.Name, path, threads, KernelMode.Exact)];
                        fl += it.Count * it.Flops;
                    }

                    if (complete)
                    {
                        Console.WriteLine($"| {model} | {path} | {threads} | {fl / 1e9:F1} | {sf:F4} | {se:F4} | {se / sf:F2} |");
                    }
                }
            }
        }

        if (csvPath.Length > 0)
        {
            File.WriteAllText(csvPath, csv.ToString());
        }

        return 0;
    }
}
