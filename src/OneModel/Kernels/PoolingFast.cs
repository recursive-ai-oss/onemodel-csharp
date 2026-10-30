using System;
using System.Runtime.Intrinsics;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Two-dimensional float max pooling without indices, vectorised along the output row for unit column stride and split over planes.
/// A maximum is exact, so the values equal <see cref="Pooling.Max{T}"/>'s on every path and in both modes, including which of two
/// equal-comparing zeros wins and that NaN taps are ignored; only where the generic loop is needed to decide an output (every tap
/// NaN, no tap inside the input, or a true -infinity) does this fall back to the same scalar rule.
/// </summary>
internal static unsafe class MaxPoolFast
{
    // Output elements per chunk of planes: a constant, so the split depends on the shape only.
    private const int OutputsPerChunk = 16384;

    /// <summary>Whether <see cref="Run"/> can do this pooling: one spatial depth, unit dilation.</summary>
    public static bool Supports(in PoolGeom g) =>
        g.In.D == 1 && g.Kernel.D == 1 && g.Out.D == 1 && g.Dilation.H == 1 && g.Dilation.W == 1;

    /// <summary>Pools <paramref name="planes"/> volumes of <paramref name="x"/> into <paramref name="y"/>.</summary>
    /// <remarks>Bounds: both spans are checked against planes * volume before any pointer is formed; every tap read is tested against the input extent.</remarks>
    public static void Run(in PoolGeom g, long planes, ReadOnlySpan<float> x, Span<float> y, int threads)
    {
        long inVol = g.In.Volume, outVol = g.Out.Volume;
        if (x.Length < planes * inVol || y.Length < planes * outVol)
        {
            throw new ArgumentException("A buffer is smaller than the pooling geometry needs.");
        }

        if (planes == 0 || outVol == 0)
        {
            return;
        }

        int perChunk = (int)Math.Max(1, OutputsPerChunk / outVol);
        int chunks = checked((int)((planes + perChunk - 1) / perChunk));
        fixed (float* px = x, py = y)
        {
            // Kept per thread and refilled, so a steady-state call allocates nothing.
            PlaneJob job = t_job ??= new PlaneJob();
            job.G = g;
            job.X = px;
            job.Y = py;
            job.Planes = planes;
            job.PerChunk = perChunk;
            ParallelFor.Run(job, chunks, threads, planes * outVol * g.Kernel.H * g.Kernel.W / 4);
        }
    }

    [ThreadStatic]
    private static PlaneJob? t_job;

    private sealed class PlaneJob : ChunkJob
    {
        public PoolGeom G;
        public float* X;
        public float* Y;
        public long Planes;
        public int PerChunk;

        public override void RunChunk(int chunk)
        {
            long end = Math.Min(Planes, (long)(chunk + 1) * PerChunk);
            long inVol = G.In.Volume, outVol = G.Out.Volume;
            for (long p = (long)chunk * PerChunk; p < end; p++)
            {
                Plane(in G, X + (p * inVol), Y + (p * outVol));
            }
        }
    }

    private static void Plane(in PoolGeom g, float* src, float* dst)
    {
        int inH = g.In.H, inW = g.In.W, outH = g.Out.H, outW = g.Out.W;
        int kh = g.Kernel.H, kw = g.Kernel.W, sh = g.Stride.H, sw = g.Stride.W, padH = g.Pad.H, padW = g.Pad.W;
        for (int oh = 0; oh < outH; oh++)
        {
            float* o = dst + ((long)oh * outW);
            new Span<float>(o, outW).Fill(float.NegativeInfinity);
            for (int ky = 0; ky < kh; ky++)
            {
                int h = (oh * sh) - padH + ky;
                if ((uint)h >= (uint)inH)
                {
                    continue;
                }

                float* row = src + ((long)h * inW);
                for (int kx = 0; kx < kw; kx++)
                {
                    // Output columns whose tap column ow*stride - pad + kx lies inside [0, inW).
                    int shift = kx - padW;
                    int lo = shift >= 0 ? 0 : (-shift + sw - 1) / sw;
                    int hi = inW - 1 - shift < 0 ? -1 : Math.Min(outW - 1, (inW - 1 - shift) / sw);
                    int ow = lo;
                    if (sw == 1)
                    {
                        if (Vector256.IsHardwareAccelerated)
                        {
                            for (; ow + 8 <= hi + 1; ow += 8)
                            {
                                Vector256<float> v = Vector256.Load(row + ow + shift);
                                Vector256<float> a = Vector256.Load(o + ow);
                                Vector256.Store(Vector256.ConditionalSelect(Vector256.GreaterThan(v, a), v, a), o + ow);
                            }
                        }

                        for (; ow + 4 <= hi + 1; ow += 4)
                        {
                            Vector128<float> v = Vector128.Load(row + ow + shift);
                            Vector128<float> a = Vector128.Load(o + ow);
                            Vector128.Store(Vector128.ConditionalSelect(Vector128.GreaterThan(v, a), v, a), o + ow);
                        }
                    }

                    // A NaN tap compares false and is skipped, as in the generic loop.
                    for (; ow <= hi; ow++)
                    {
                        float v = row[(ow * sw) + shift];
                        if (v > o[ow])
                        {
                            o[ow] = v;
                        }
                    }
                }
            }

            // An output still at -infinity had no non-NaN tap above it: all NaN, no tap inside the input, or a true -infinity.
            for (int ow = 0; ow < outW; ow++)
            {
                if (float.IsNegativeInfinity(o[ow]))
                {
                    o[ow] = Slow(in g, src, oh, ow);
                }
            }
        }
    }

    // The generic rule for one output: the first tap that is not NaN and any strictly greater later one; else the first NaN tap;
    // else the lowest finite float when no tap lies inside the input.
    private static float Slow(in PoolGeom g, float* src, int oh, int ow)
    {
        float best = float.MinValue;
        bool any = false, sawNaN = false;
        float nan = 0f;
        for (int ky = 0; ky < g.Kernel.H; ky++)
        {
            int h = (oh * g.Stride.H) - g.Pad.H + ky;
            if ((uint)h >= (uint)g.In.H)
            {
                continue;
            }

            for (int kx = 0; kx < g.Kernel.W; kx++)
            {
                int w = (ow * g.Stride.W) - g.Pad.W + kx;
                if ((uint)w >= (uint)g.In.W)
                {
                    continue;
                }

                float v = src[(h * g.In.W) + w];
                if (float.IsNaN(v))
                {
                    if (!sawNaN)
                    {
                        sawNaN = true;
                        nan = v;
                    }

                    continue;
                }

                if (!any || v > best)
                {
                    best = v;
                    any = true;
                }
            }
        }

        return !any && sawNaN ? nan : best;
    }
}
