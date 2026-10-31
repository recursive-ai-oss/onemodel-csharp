using System;
using System.Diagnostics;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;

namespace RecursiveAI.OneModel.Benchmarks;

/// <summary>Times element-wise float functions on one thread: <c>unary [gelu|gelufast|erf|exp|sigmoid] [length]</c>. Prints ns per element.</summary>
internal static class UnaryBench
{
    public static int Run(string[] args)
    {
        string which = args.Length > 0 ? args[0] : "gelu";
        if (which == "gelucheck")
        {
            return GeluCheck();
        }

        int n = args.Length > 1 ? int.Parse(args[1]) : 196608;
        var x = new float[n];
        var y = new float[n];
        var rng = new Random(7);
        for (int i = 0; i < n; i++)
        {
            x[i] = (float)((rng.NextDouble() * 2 - 1) * 4);
        }

        KernelPath path = args.Length > 2 ? Enum.Parse<KernelPath>(args[2]) : KernelPaths.Best();
        Console.WriteLine($"path {path}");
        for (int rep = 0; rep < 12; rep++)
        {
            int iters = 200;
            long t0 = Stopwatch.GetTimestamp();
            for (int it = 0; it < iters; it++)
            {
                switch (which)
                {
                    case "gelu": FloatUnary.Run(path, false, new GeluUnary(false), x, y); break;
                    case "erf": FloatUnary.Run(path, false, new ErfUnary(), x, y); break;
                    case "exp": FloatUnary.Run(path, false, new ExpUnary(), x, y); break;
                    case "sigmoid": FloatUnary.Run(path, false, new SigmoidUnary(), x, y); break;
                    default: throw new ArgumentException(which);
                }
            }

            double ns = (Stopwatch.GetTimestamp() - t0) * 1e9 / Stopwatch.Frequency / iters / n;
            Console.WriteLine($"{which} n={n}: {ns:F3} ns/element");
        }

        return 0;
    }

    // Fast GELU against the exact-mode kernel (about 3 ulp from the true value) on a dense grid and special values.
    private static int GeluCheck()
    {
        var xs = new System.Collections.Generic.List<float>();
        for (double v = -14; v <= 14; v += 0.0001)
        {
            xs.Add((float)v);
        }

        xs.AddRange([0f, -0f, float.PositiveInfinity, float.NegativeInfinity, float.NaN, 1e-30f, -1e-30f, 1e30f, -1e30f, float.MaxValue, float.MinValue, 5.6f, -5.6f, 1e-45f]);
        float[] x = [.. xs];
        var yf = new float[x.Length];
        var ye = new float[x.Length];
        foreach (KernelPath path in KernelPaths.All)
        {
            if (!KernelPaths.IsSupported(path))
            {
                continue;
            }

            FloatUnary.Run(path, false, new GeluUnary(false), x, yf);
            FloatUnary.Run(path, true, new GeluUnary(false), x, ye);
            double maxAbs = 0, maxRel = 0, maxScaled = 0;
            float atAbs = 0, atRel = 0;
            for (int i = 0; i < x.Length; i++)
            {
                if (float.IsNaN(ye[i]) || float.IsInfinity(ye[i]))
                {
                    if (!(float.IsNaN(ye[i]) && float.IsNaN(yf[i])) && ye[i] != yf[i])
                    {
                        Console.WriteLine($"special mismatch x={x[i]:R} exact={ye[i]:R} fast={yf[i]:R}");
                    }

                    continue;
                }

                double d = Math.Abs((double)yf[i] - ye[i]);
                if (d > maxAbs)
                {
                    maxAbs = d;
                    atAbs = x[i];
                }

                if (Math.Abs(ye[i]) > 1e-2 && d / Math.Abs(ye[i]) > maxRel)
                {
                    maxRel = d / Math.Abs(ye[i]);
                    atRel = x[i];
                }

                maxScaled = Math.Max(maxScaled, d / Math.Max(1.0, Math.Abs(x[i])));
            }

            Console.WriteLine($"{path}: max abs {maxAbs:E2} at {atAbs}; max rel (|y|>1e-2) {maxRel:E2} at {atRel}; max abs/max(1,|x|) {maxScaled:E2}");
        }

        return 0;
    }
}
