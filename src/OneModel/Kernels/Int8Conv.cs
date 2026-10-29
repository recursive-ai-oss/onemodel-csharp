using System;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>Everything one integer convolution needs. The padding value of the input is its zero point, so padded taps contribute zero.</summary>
internal ref struct Int8ConvCall
{
    public Int8Format Format;
    public ConvShape S;

    /// <summary>The input, NCHW, 8-bit.</summary>
    public ReadOnlySpan<byte> X;
    public bool XU8;

    /// <summary>The input zero point as stored (the padding byte), and in the shifted domain (uint8 minus 128).</summary>
    public int XZeroRaw;
    public int XZeroShifted;

    /// <summary>The weights [Cout, Cin/Groups, KH, KW] as raw 8-bit, or packed per group ahead of time.</summary>
    public ReadOnlySpan<byte> WRaw;
    public bool WU8;
    public Int8Packed[]? WPacked;

    /// <summary>The weight zero point, shifted domain: one per output channel (when <see cref="WZeroVec"/> is not empty) or one value.</summary>
    public int WZeroScalar;
    public ReadOnlySpan<int> WZeroVec;

    /// <summary>The output [N, Cout, OutH, OutW]: int32 sums, or 8-bit quantized values when <see cref="OutInt32"/> is empty.</summary>
    public Span<int> OutInt32;
    public Span<byte> OutQuant;
    public bool QuantUnsigned;
    public int QuantZero;

    /// <summary>Per-output-channel int32 bias (may be empty), and the scale: one value or one per output channel.</summary>
    public ReadOnlySpan<int> Bias;
    public float Mult;
    public ReadOnlySpan<float> MultVec;

    public int Threads;
}

/// <summary>
/// Integer 2-D convolution (ConvInteger, QLinearConv). Regular and grouped convolutions are lowered to <see cref="Int8Gemm"/> by im2col
/// (Chellapilla et al. 2006) exactly like the float convolution, with the weights as A and the patches as B; depthwise convolutions
/// run a direct kernel. All arithmetic is exact integer arithmetic, so the result does not depend on the path or the thread count.
/// </summary>
internal static unsafe class Int8Conv
{
    private const int Im2ColRowsPerChunk = 16;
    private const int PlanesPerChunk = 4;
    private const int MaxDirectTaps = 64;

    [ThreadStatic]
    private static Im2ColJob? t_im2col;

    [ThreadStatic]
    private static DepthwiseJob? t_depthwise;

    public static void Run(ref Int8ConvCall c)
    {
        ConvShape s = c.S;
        if (s.N <= 0 || s.Cin <= 0 || s.Cout <= 0 || s.Groups <= 0 || s.Cin % s.Groups != 0 || s.Cout % s.Groups != 0 ||
            s.KH <= 0 || s.KW <= 0 || s.OutH <= 0 || s.OutW <= 0)
        {
            throw new ArgumentException("The convolution geometry is not valid.");
        }

        int cinG = s.Cin / s.Groups, coutG = s.Cout / s.Groups;
        long k64 = (long)cinG * s.KH * s.KW;
        long nCols64 = (long)s.OutH * s.OutW;
        if (k64 > int.MaxValue || nCols64 > int.MaxValue)
        {
            throw new ArgumentException("The convolution is too large.");
        }

        int k = (int)k64, nCols = (int)nCols64;
        if (c.X.Length < (long)s.N * s.Cin * s.H * s.W)
        {
            throw new ArgumentException("The input is smaller than its shape.");
        }

        if (c.WPacked == null && c.WRaw.Length < (long)s.Cout * k)
        {
            throw new ArgumentException("The weights are smaller than their shape.");
        }

        if (c.WPacked != null && c.WPacked.Length != s.Groups)
        {
            throw new ArgumentException("There must be one packed weight matrix per group.");
        }

        bool toInt = !c.OutInt32.IsEmpty;
        long outLen = toInt ? c.OutInt32.Length : c.OutQuant.Length;
        if (outLen < (long)s.N * s.Cout * nCols)
        {
            throw new ArgumentException("The output is smaller than its shape.");
        }

        if ((!c.Bias.IsEmpty && c.Bias.Length < s.Cout) || (!c.MultVec.IsEmpty && c.MultVec.Length < s.Cout) ||
            (!c.WZeroVec.IsEmpty && c.WZeroVec.Length < s.Cout))
        {
            throw new ArgumentException("A per-channel vector is shorter than the number of output channels.");
        }

        if (cinG == 1 && coutG == 1 && s.KH * s.KW <= MaxDirectTaps)
        {
            Depthwise(ref c, nCols);
            return;
        }

        bool pointwise = s.KH == 1 && s.KW == 1 && s.StrideH == 1 && s.StrideW == 1 &&
                         s.PadTop == 0 && s.PadLeft == 0 && s.PadBottom == 0 && s.PadRight == 0;
        Int8Format f = c.Format;
        byte* cols = pointwise ? null : (byte*)Scratch.Get(Scratch.Im2Col, (((long)k * nCols) + 3) / 4);
        byte* packed = (byte*)Scratch.Get(Scratch.PackB, (f.PackedBBytes(k, nCols) + 3) / 4);
        int* sums = (int*)Scratch.Get(Scratch.Plane, nCols);
        int hw = s.H * s.W;
        fixed (byte* px = c.X)
        {
            for (int img = 0; img < s.N; img++)
            {
                for (int g = 0; g < s.Groups; g++)
                {
                    byte* src = px + ((((long)img * s.Cin) + (g * cinG)) * hw);
                    byte* b = src;
                    if (!pointwise)
                    {
                        Im2ColJob job = t_im2col ??= new Im2ColJob();
                        job.Src = src;
                        job.Dst = cols;
                        job.S = s;
                        job.CinG = cinG;
                        job.Pad = (byte)c.XZeroRaw;
                        ParallelFor.Run(job, (k + Im2ColRowsPerChunk - 1) / Im2ColRowsPerChunk, c.Threads);
                        b = cols;
                    }

                    Int8Gemm.PackB(in f, b, nCols, c.XU8, k, nCols, packed, sums, c.Threads);

                    var call = new Int8Call
                    {
                        Format = f,
                        M = coutG,
                        N = nCols,
                        K = k,
                        BPacked = new ReadOnlySpan<byte>(packed, (int)f.PackedBBytes(k, nCols)),
                        BSums = new ReadOnlySpan<int>(sums, nCols),
                        ZbScalar = c.XZeroShifted,
                        ZaScalar = c.WZeroScalar,
                        QuantUnsigned = c.QuantUnsigned,
                        QuantZero = c.QuantZero,
                        Ldc = nCols,
                        Mult = c.Mult,
                        MultPerRow = true,
                        Threads = c.Threads,
                    };
                    if (c.WPacked != null)
                    {
                        call.APacked = c.WPacked[g];
                    }
                    else
                    {
                        call.ARaw = c.WRaw.Slice(g * coutG * k, coutG * k);
                        call.Lda = k;
                        call.AU8 = c.WU8;
                    }

                    if (!c.WZeroVec.IsEmpty)
                    {
                        call.ZaVec = c.WZeroVec.Slice(g * coutG, coutG);
                    }

                    if (!c.Bias.IsEmpty)
                    {
                        call.Bias = c.Bias.Slice(g * coutG, coutG);
                    }

                    if (!c.MultVec.IsEmpty)
                    {
                        call.MultVec = c.MultVec.Slice(g * coutG, coutG);
                    }

                    int outOff = checked((int)((((long)img * s.Cout) + (g * coutG)) * nCols));
                    int outCount = coutG * nCols;
                    if (toInt)
                    {
                        call.OutInt32 = c.OutInt32.Slice(outOff, outCount);
                    }
                    else
                    {
                        call.OutQuant = c.OutQuant.Slice(outOff, outCount);
                    }

                    Int8Matmul.Run(ref call);
                }
            }
        }
    }

    private static void Depthwise(ref Int8ConvCall c, int nCols)
    {
        ConvShape s = c.S;
        fixed (byte* px = c.X, pw = c.WRaw, pq = c.OutQuant)
        fixed (int* pi = c.OutInt32, pbias = c.Bias, pzw = c.WZeroVec)
        fixed (float* pm = c.MultVec)
        {
            DepthwiseJob job = t_depthwise ??= new DepthwiseJob();
            job.S = s;
            job.X = px;
            job.XU8 = c.XU8;
            job.XZero = c.XZeroShifted;
            job.W = pw;
            job.WU8 = c.WU8;
            job.WZeroScalar = c.WZeroScalar;
            job.WZeroVec = c.WZeroVec.IsEmpty ? null : pzw;
            job.OutInt32 = c.OutInt32.IsEmpty ? null : pi;
            job.OutQuant = c.OutInt32.IsEmpty ? pq : null;
            job.QuantUnsigned = c.QuantUnsigned;
            job.QuantZero = c.QuantZero;
            job.Bias = c.Bias.IsEmpty ? null : pbias;
            job.Mult = c.Mult;
            job.MultVec = c.MultVec.IsEmpty ? null : pm;
            int planes = s.N * s.Cin;
            ParallelFor.Run(job, (planes + PlanesPerChunk - 1) / PlanesPerChunk, c.Threads);
        }
    }

    private sealed class Im2ColJob : ChunkJob
    {
        public byte* Src;
        public byte* Dst;
        public ConvShape S;
        public int CinG;
        public byte Pad;

        public override void RunChunk(int chunk)
        {
            int outH = S.OutH, outW = S.OutW;
            int rows = CinG * S.KH * S.KW;
            int r0 = chunk * Im2ColRowsPerChunk;
            int r1 = Math.Min(rows, r0 + Im2ColRowsPerChunk);
            for (int r = r0; r < r1; r++)
            {
                int kx = r % S.KW;
                int ky = (r / S.KW) % S.KH;
                int ci = r / (S.KW * S.KH);
                byte* plane = Src + ((long)ci * S.H * S.W);
                byte* dst = Dst + ((long)r * outH * outW);
                int shiftX = (kx * S.DilW) - S.PadLeft;
                int oxLo = shiftX >= 0 ? 0 : (-shiftX + S.StrideW - 1) / S.StrideW;
                int oxHi = S.W - 1 - shiftX < 0 ? -1 : Math.Min(outW - 1, (S.W - 1 - shiftX) / S.StrideW);
                for (int oy = 0; oy < outH; oy++)
                {
                    int iy = (oy * S.StrideH) - S.PadTop + (ky * S.DilH);
                    byte* row = dst + ((long)oy * outW);
                    if (iy < 0 || iy >= S.H || oxHi < oxLo)
                    {
                        new Span<byte>(row, outW).Fill(Pad);
                        continue;
                    }

                    byte* srcRow = plane + ((long)iy * S.W);
                    new Span<byte>(row, oxLo).Fill(Pad);
                    if (S.StrideW == 1)
                    {
                        new ReadOnlySpan<byte>(srcRow + oxLo + shiftX, oxHi - oxLo + 1).CopyTo(new Span<byte>(row + oxLo, oxHi - oxLo + 1));
                    }
                    else
                    {
                        for (int ox = oxLo; ox <= oxHi; ox++)
                        {
                            row[ox] = srcRow[(ox * S.StrideW) + shiftX];
                        }
                    }

                    new Span<byte>(row + oxHi + 1, outW - oxHi - 1).Fill(Pad);
                }
            }
        }
    }

    private sealed class DepthwiseJob : ChunkJob
    {
        public ConvShape S;
        public byte* X;
        public bool XU8;
        public int XZero;
        public byte* W;
        public bool WU8;
        public int WZeroScalar;
        public int* WZeroVec;
        public int* OutInt32;
        public byte* OutQuant;
        public bool QuantUnsigned;
        public int QuantZero;
        public int* Bias;
        public float Mult;
        public float* MultVec;

        public override void RunChunk(int chunk)
        {
            int planes = S.N * S.Cin;
            int p0 = chunk * PlanesPerChunk;
            int p1 = Math.Min(planes, p0 + PlanesPerChunk);
            int outH = S.OutH, outW = S.OutW, taps = S.KH * S.KW;
            int* wv = stackalloc int[MaxDirectTaps];
            int lo = QuantUnsigned ? 0 : -128, hi = QuantUnsigned ? 255 : 127;
            for (int p = p0; p < p1; p++)
            {
                int ch = p % S.Cin;
                int za = WZeroVec != null ? WZeroVec[ch] : WZeroScalar;
                for (int t = 0; t < taps; t++)
                {
                    wv[t] = Int8Gemm.Shift(W[(ch * taps) + t], WU8) - za;
                }

                byte* plane = X + ((long)p * S.H * S.W);
                long outBase = (long)p * outH * outW;
                int bias = Bias != null ? Bias[ch] : 0;
                float mult = MultVec != null ? MultVec[ch] : Mult;
                for (int oy = 0; oy < outH; oy++)
                {
                    for (int ox = 0; ox < outW; ox++)
                    {
                        int acc = 0;
                        for (int ky = 0; ky < S.KH; ky++)
                        {
                            int iy = (oy * S.StrideH) - S.PadTop + (ky * S.DilH);
                            if ((uint)iy >= (uint)S.H)
                            {
                                continue;
                            }

                            for (int kx = 0; kx < S.KW; kx++)
                            {
                                int ix = (ox * S.StrideW) - S.PadLeft + (kx * S.DilW);
                                if ((uint)ix >= (uint)S.W)
                                {
                                    continue;
                                }

                                acc += (Int8Gemm.Shift(plane[(iy * S.W) + ix], XU8) - XZero) * wv[(ky * S.KW) + kx];
                            }
                        }

                        acc += bias;
                        long o = outBase + (oy * outW) + ox;
                        if (OutInt32 != null)
                        {
                            OutInt32[o] = acc;
                        }
                        else
                        {
                            OutQuant[o] = (byte)Int8Gemm.Requant(acc, mult, QuantZero, lo, hi);
                        }
                    }
                }
            }
        }
    }
}
