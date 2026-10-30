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

    // acc[i] = max(acc[i], v[i]) for n bytes.
    private static void MaxInto(byte* acc, byte* v, int n)
    {
        int i = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            for (; i + 16 <= n; i += 16)
            {
                Vector128.Max(Vector128.Load(acc + i), Vector128.Load(v + i)).Store(acc + i);
            }
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
            if (Vector128.IsHardwareAccelerated)
            {
                Vector128<byte> vflip = Vector128.Create(flip);
                for (; c + 16 <= inW; c += 16)
                {
                    (Vector128.Load(line + c) ^ vflip).Store(tmp + padW + c);
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
            }

            for (; ow < outW; ow++)
            {
                o[ow] = (byte)(acc[ow * sw] ^ flip);
            }
        }
    }
}
