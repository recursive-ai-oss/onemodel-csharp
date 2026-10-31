using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Two-dimensional max pooling of 8-bit tensors without indices, separable like the float one in <see cref="MaxPoolFast"/>: the
/// maximum over the kernel columns is taken along every input row (vectorised, sixteen bytes at a time), the rows of each window are
/// combined, and every StrideW-th column is kept. int8 values are flipped by 0x80 into an unsigned order-preserving domain on the way
/// in and out, so one byte maximum serves both types. A window with no tap inside the input gives the type's lowest value, as
/// <see cref="Pooling.Max{T}"/> does. Integers have no NaN or signed zero, so the result equals the generic loop's on every path.
/// </summary>
internal static unsafe class Int8Pool
{
    private const int OutputsPerChunk = 16384;

    [ThreadStatic]
    private static PlaneJob? t_job;

    /// <summary>Whether <see cref="Run"/> can do this pooling: one spatial depth, unit dilation.</summary>
    public static bool Supports(in PoolGeom g) => MaxPoolFast.Supports(in g);

    /// <summary>Pools <paramref name="planes"/> volumes of 8-bit values.</summary>
    /// <remarks>Bounds: both spans are checked against planes * volume before any pointer is formed; every tap read is tested against the input extent.</remarks>
    public static void Run(in PoolGeom g, long planes, ReadOnlySpan<byte> x, Span<byte> y, bool signed, int threads)
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
        fixed (byte* px = x, py = y)
        {
            PlaneJob job = t_job ??= new PlaneJob();
            job.G = g;
            job.X = px;
            job.Y = py;
            job.Planes = planes;
            job.PerChunk = perChunk;
            job.Flip = signed ? (byte)0x80 : (byte)0;
            // About a quarter of a nanosecond per tap; it only decides how many threads are worth waking.
            ParallelFor.Run(job, chunks, threads, planes * outVol * g.Kernel.H * g.Kernel.W / 4);
            job.X = null;
            job.Y = null;
        }
    }

    [ThreadStatic]
    private static AverageJob? t_average;

    /// <summary>Parameters of a global average over 8-bit planes: the input's type, zero point and scale, and the output's.</summary>
    internal readonly record struct AverageParams(bool XUnsigned, int XZero, float XScale, bool YUnsigned, int YZero, float YScale);

    /// <summary>
    /// The mean of each of <paramref name="planes"/> planes of <paramref name="plane"/> values, quantized: the sum of the integers is
    /// exact, the mean is <c>(sum - plane * zero) * scale / plane</c> in float32, and the quantization is that of QuantizeLinear.
    /// </summary>
    /// <remarks>Bounds: both spans are checked against planes * plane (and planes) before any pointer is formed.</remarks>
    public static void GlobalAverage(ReadOnlySpan<byte> x, Span<byte> y, long planes, int plane, in AverageParams p, int threads)
    {
        if (planes < 0 || plane <= 0 || x.Length < planes * plane || y.Length < planes)
        {
            throw new ArgumentException("A buffer is smaller than the planes need.");
        }

        if (planes == 0)
        {
            return;
        }

        fixed (byte* px = x, py = y)
        {
            int perChunk = (int)Math.Max(1, Math.Min(planes, OutputsPerChunk / plane));
            int chunks = (int)((planes + perChunk - 1) / perChunk);
            if (threads <= 1 || chunks < 2)
            {
                Averages(px, py, 0, planes, plane, in p);
                return;
            }

            AverageJob job = t_average ??= new AverageJob();
            job.X = px;
            job.Y = py;
            job.Planes = planes;
            job.Plane = plane;
            job.PerChunk = perChunk;
            job.P = p;
            ParallelFor.Run(job, chunks, threads, planes * plane / 4);
            job.X = null;
            job.Y = null;
        }
    }

    private static void Averages(byte* x, byte* y, long first, long end, int plane, in AverageParams p)
    {
        int lo = p.YUnsigned ? 0 : -128, hi = p.YUnsigned ? 255 : 127;
        byte flip = p.XUnsigned ? (byte)0 : (byte)0x80;
        for (long i = first; i < end; i++)
        {
            // Signed bytes are summed as b ^ 0x80 (an offset of 128 each) and the offset is taken off again.
            long sum = SumBytes(x + (i * plane), plane, flip) - (p.XUnsigned ? 0 : 128L * plane);
            float mean = (float)(sum - ((long)plane * p.XZero)) * p.XScale / plane;
            y[i] = (byte)Int8Quantize.QuantizeOne(mean, p.YScale, p.YZero, lo, hi);
        }
    }

    // The sum of (b ^ flip) over n bytes. Sixteen lanes of 16-bit partial sums take at most 128 rounds of two bytes (2 * 255 * 128 < 65536).
    private static long SumBytes(byte* p, int n, byte flip)
    {
        long total = 0;
        int i = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            Vector128<byte> vflip = Vector128.Create(flip);
            Vector128<uint> wide = Vector128<uint>.Zero;
            while (i + 16 <= n)
            {
                Vector128<ushort> part = Vector128<ushort>.Zero;
                for (int r = 0; r < 128 && i + 16 <= n; r++, i += 16)
                {
                    Vector128<byte> v = Vector128.Load(p + i) ^ vflip;
                    part += Vector128.WidenLower(v) + Vector128.WidenUpper(v);
                }

                wide += Vector128.WidenLower(part) + Vector128.WidenUpper(part);
            }

            total = Vector128.Sum(wide);
        }

        for (; i < n; i++)
        {
            total += (byte)(p[i] ^ flip);
        }

        return total;
    }

    private sealed class AverageJob : ChunkJob
    {
        public byte* X;
        public byte* Y;
        public long Planes;
        public int Plane;
        public int PerChunk;
        public AverageParams P;

        public override void RunChunk(int chunk)
        {
            long first = (long)chunk * PerChunk;
            Averages(X, Y, first, Math.Min(Planes, first + PerChunk), Plane, in P);
        }
    }

    private sealed class PlaneJob : ChunkJob
    {
        public PoolGeom G;
        public byte* X;
        public byte* Y;
        public long Planes;
        public int PerChunk;
        public byte Flip;

        public override void RunChunk(int chunk)
        {
            long end = Math.Min(Planes, (long)(chunk + 1) * PerChunk);
            long inVol = G.In.Volume, outVol = G.Out.Volume;
            int span = ((G.Out.W - 1) * G.Stride.W) + 1;
            long need = (long)Math.Max(G.Pad.W + G.In.W, span + G.Kernel.W - 1) + ((long)(G.In.H + 1) * span) + 64;
            byte* scratch = (byte*)Scratch.Get(Scratch.Plane, (need + 3) / 4);
            for (long p = (long)chunk * PerChunk; p < end; p++)
            {
                Plane(in G, X + (p * inVol), Y + (p * outVol), scratch, Flip);
            }
        }
    }

    // acc[i] = max(acc[i], v[i]) for n bytes. A tail shorter than a vector is covered by one more vector that overlaps the last full one:
    // the maximum is idempotent, so redoing those lanes changes nothing, and no scalar loop is needed for rows of 16 bytes or more.
    private static void MaxInto(byte* acc, byte* v, int n)
    {
        int i = 0;
        if (Vector256.IsHardwareAccelerated && n >= 32)
        {
            for (; i + 32 <= n; i += 32)
            {
                Vector256.Max(Vector256.Load(acc + i), Vector256.Load(v + i)).Store(acc + i);
            }

            if (i < n)
            {
                Vector256.Max(Vector256.Load(acc + n - 32), Vector256.Load(v + n - 32)).Store(acc + n - 32);
            }

            return;
        }

        if (Vector128.IsHardwareAccelerated && n >= 16)
        {
            for (; i + 16 <= n; i += 16)
            {
                Vector128.Max(Vector128.Load(acc + i), Vector128.Load(v + i)).Store(acc + i);
            }

            if (i < n)
            {
                Vector128.Max(Vector128.Load(acc + n - 16), Vector128.Load(v + n - 16)).Store(acc + n - 16);
            }

            return;
        }

        for (; i < n; i++)
        {
            if (v[i] > acc[i])
            {
                acc[i] = v[i];
            }
        }
    }

    private static void Plane(in PoolGeom g, byte* src, byte* dst, byte* scratch, byte flip)
    {
        int inH = g.In.H, inW = g.In.W, outH = g.Out.H, outW = g.Out.W;
        int kh = g.Kernel.H, kw = g.Kernel.W, sh = g.Stride.H, sw = g.Stride.W, padH = g.Pad.H, padW = g.Pad.W;
        int span = ((outW - 1) * sw) + 1;
        int tmpLen = Math.Max(padW + inW, span + kw - 1);
        byte* tmp = scratch;
        byte* rows = tmp + tmpLen;
        byte* acc = rows + ((long)inH * span);

        // rows[r][c] = max over kx of input column c + kx - padW of row r, for the columns an output can start at (0 is the lowest value).
        for (int r = 0; r < inH; r++)
        {
            new Span<byte>(tmp, padW).Clear();
            byte* line = src + ((long)r * inW);
            int c = 0;
            if (Vector128.IsHardwareAccelerated && inW >= 16)
            {
                // The last vector overlaps the one before it when the width is not a multiple of 16: it stores the same bytes again.
                Vector128<byte> vflip = Vector128.Create(flip);
                for (; c + 16 <= inW; c += 16)
                {
                    (Vector128.Load(line + c) ^ vflip).Store(tmp + padW + c);
                }

                if (c < inW)
                {
                    c = inW - 16;
                    (Vector128.Load(line + c) ^ vflip).Store(tmp + padW + c);
                    c = inW;
                }
            }

            for (; c < inW; c++)
            {
                tmp[padW + c] = (byte)(line[c] ^ flip);
            }

            new Span<byte>(tmp + padW + inW, tmpLen - padW - inW).Clear();
            byte* h = rows + ((long)r * span);
            new Span<byte>(h, span).Clear();
            for (int kx = 0; kx < kw; kx++)
            {
                MaxInto(h, tmp + kx, span);
            }
        }

        for (int oh = 0; oh < outH; oh++)
        {
            byte* o = dst + ((long)oh * outW);
            new Span<byte>(acc, span).Clear();
            for (int ky = 0; ky < kh; ky++)
            {
                int r = (oh * sh) - padH + ky;
                if ((uint)r < (uint)inH)
                {
                    MaxInto(acc, rows + ((long)r * span), span);
                }
            }

            int ow = 0;
            if (Vector128.IsHardwareAccelerated && (sw == 1 || sw == 2) && (Sse2.IsSupported || AdvSimd.Arm64.IsSupported))
            {
                // Stride 1 keeps every column, stride 2 every second one (an even-byte pick; the scratch has slack past the row).
                Vector128<byte> vflip = Vector128.Create(flip);
                for (; ow + 16 <= outW; ow += 16)
                {
                    Vector128<byte> v = sw == 1 ? Vector128.Load(acc + ow) : Int8Pack.Even((sbyte*)(acc + (2 * ow))).AsByte();
                    (v ^ vflip).Store(o + ow);
                }

                if (ow < outW && outW >= 16)
                {
                    // One more vector ending at the last column; it rewrites bytes already stored with the same values.
                    ow = outW - 16;
                    Vector128<byte> v = sw == 1 ? Vector128.Load(acc + ow) : Int8Pack.Even((sbyte*)(acc + (2 * ow))).AsByte();
                    (v ^ vflip).Store(o + ow);
                    ow = outW;
                }
            }

            for (; ow < outW; ow++)
            {
                o[ow] = (byte)(acc[ow * sw] ^ flip);
            }
        }
    }
}
