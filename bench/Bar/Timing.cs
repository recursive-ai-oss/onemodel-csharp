using System;
using System.Diagnostics;
using System.Linq;

namespace OneModel.Bar;

/// <summary>Result of <see cref="Timing.Measure"/>.</summary>
/// <param name="RepMs">Mean milliseconds per iteration, one value per repetition.</param>
/// <param name="ItersPerRep">Iterations inside each repetition.</param>
/// <param name="AllocBytesPerIter">Managed bytes allocated per iteration on the calling thread.</param>
/// <param name="FirstMs">Duration of the very first call (cold).</param>
internal sealed record TimingResult(double[] RepMs, int ItersPerRep, double AllocBytesPerIter, double FirstMs)
{
    public double Median
    {
        get
        {
            double[] s = [.. RepMs.OrderBy(x => x)];
            return s.Length % 2 == 1 ? s[s.Length / 2] : 0.5 * (s[(s.Length / 2) - 1] + s[s.Length / 2]);
        }
    }

    public double Min => RepMs.Min();

    public double Max => RepMs.Max();

    /// <summary>(max - min) / median, in percent.</summary>
    public double SpreadPercent => Median > 0 ? 100.0 * (Max - Min) / Median : 0;
}

/// <summary>
/// Warm, in-process timing: a few untimed warm-up calls, then <c>reps</c> repetitions of as many
/// iterations as fit in about <c>targetRepMs</c>. Reused by the kernel micro-rows so every number in
/// the project comes from the same method.
/// </summary>
internal static class Timing
{
    public static TimingResult Measure(Action body, int warmup = 3, int reps = 3, double targetRepMs = 1000, int maxItersPerRep = 1000)
    {
        long t0 = Stopwatch.GetTimestamp();
        body();
        double firstMs = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;

        double last = firstMs;
        for (int i = 0; i < warmup; i++)
        {
            long t = Stopwatch.GetTimestamp();
            body();
            last = Stopwatch.GetElapsedTime(t).TotalMilliseconds;
        }

        int iters = (int)Math.Clamp(Math.Ceiling(targetRepMs / Math.Max(last, 1e-6)), 1, maxItersPerRep);
        var rep = new double[reps];
        long alloc0 = GC.GetAllocatedBytesForCurrentThread();
        for (int r = 0; r < reps; r++)
        {
            long t = Stopwatch.GetTimestamp();
            for (int i = 0; i < iters; i++)
            {
                body();
            }

            rep[r] = Stopwatch.GetElapsedTime(t).TotalMilliseconds / iters;
        }

        double alloc = (GC.GetAllocatedBytesForCurrentThread() - alloc0) / (double)(reps * iters);
        return new TimingResult(rep, iters, alloc, firstMs);
    }
}
