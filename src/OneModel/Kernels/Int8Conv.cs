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
    private const int StripsPerChunk = 2;
    private const int PlanesPerChunk = 4;
    private const int MaxDirectTaps = 64;

    [ThreadStatic]
    private static Im2ColPackJob? t_im2col;

    [ThreadStatic]
    private static DepthwiseJob? t_depthwise;

    [ThreadStatic]
    private static PadJob? t_pad;

    [ThreadStatic]
    private static PanelSource? t_panel;

    private const int PlanesPerPadChunk = 4;

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
        int hw = s.H * s.W;

        // The column sums only matter when the weights have a nonzero zero point (symmetric weights, the usual case, need none).
        bool wantSums = c.WZeroScalar != 0;
        for (int i = 0; !wantSums && i < c.WZeroVec.Length; i++)
        {
            wantSums = c.WZeroVec[i] != 0;
        }

        // Convolutions with horizontal stride 1 or 2 gather from zero-point-bordered copies of the planes with vector loads (Int8Pack).
        int hp = s.H + s.PadTop + s.PadBottom, wp = s.W + s.PadLeft + s.PadRight;
        bool vectorGather = !pointwise && Int8Pack.Supported(in f) && s.StrideW <= 2 && s.OutW >= 4 && s.DilW >= 1 && s.DilH >= 1 &&
                            hp > 0 && wp > 0 && ((long)(s.OutH - 1) * s.StrideH) + ((long)(s.KH - 1) * s.DilH) < hp &&
                            ((long)(s.OutW - 1) * s.StrideW) + ((long)(s.KW - 1) * s.DilW) < wp && (long)cinG * hp * wp < (int.MaxValue / 2);
        long planeSize = (long)hp * wp;

        // With vector packers the patches of a panel of columns are packed by the thread that multiplies them (Int8BPanelSource);
        // otherwise the whole matrix is packed first.
        bool pointwiseVector = pointwise && Int8Pack.Supported(in f);
        // Only worth it when more than one thread will work on the product: one thread packs the whole matrix once, with no repeats.
        long work = ((long)coutG * nCols * k / 50) + ((long)coutG * nCols / 2);
        bool onDemand = (pointwiseVector || vectorGather) && c.Threads > 1 && work / ParallelFor.MinWorkPerThread > 1;
        byte* packed = null;
        int* sums = null;
        if (!onDemand)
        {
            packed = (byte*)Scratch.Get(Scratch.PackB, (f.PackedBBytes(k, nCols) + 3) / 4);
            sums = (int*)Scratch.Get(Scratch.Plane, nCols);
        }

        PanelSource panel = t_panel ??= new PanelSource();
        fixed (byte* px = c.X)
        {
            for (int img = 0; img < s.N; img++)
            {
                for (int g = 0; g < s.Groups; g++)
                {
                    byte* src = px + ((((long)img * s.Cin) + (g * cinG)) * hw);
                    if (pointwise && onDemand)
                    {
                        panel.Set(in f, src, nCols, c.XU8, k, nCols, wantSums);
                    }
                    else if (pointwise)
                    {
                        Int8Gemm.PackB(in f, src, nCols, c.XU8, k, nCols, packed, sums, c.Threads, wantSums);
                    }
                    else if (vectorGather)
                    {
                        // 32 bytes of slack on both sides of the bordered planes: a run's 16-byte load may start before its first lane.
                        sbyte* padded = (sbyte*)Scratch.Get(Scratch.Im2Col, (((long)cinG * planeSize) + 64 + 3) / 4) + 32;
                        PadJob pj = t_pad ??= new PadJob();
                        pj.Src = src;
                        pj.U8 = c.XU8;
                        pj.Pad = (sbyte)Int8Gemm.Shift((byte)c.XZeroRaw, c.XU8);
                        pj.S = s;
                        pj.Dst = padded;
                        pj.Planes = cinG;
                        ParallelFor.Run(pj, (cinG + PlanesPerPadChunk - 1) / PlanesPerPadChunk, c.Threads, (long)cinG * planeSize / 8);
                        if (onDemand)
                        {
                            panel.Set(in f, in s, padded, (int)planeSize, cinG, k, wantSums);
                        }
                        else
                        {
                            Int8Pack.Im2ColStrips(in f, in s, padded, (int)planeSize, cinG, k, (sbyte*)packed, sums, wantSums, 0, (nCols + f.Nr - 1) / f.Nr);
                        }
                    }
                    else
                    {
                        // Gather the patches straight into the packed layout: no intermediate im2col matrix.
                        int strips = (nCols + f.Nr - 1) / f.Nr;
                        Im2ColPackJob job = t_im2col ??= new Im2ColPackJob();
                        job.F = f;
                        job.Src = src;
                        job.S = s;
                        job.CinG = cinG;
                        job.Pad = (byte)c.XZeroRaw;
                        job.U8 = c.XU8;
                        job.K = k;
                        job.Dst = packed;
                        job.ColSums = sums;
                        job.Strips = strips;
                        ParallelFor.Run(job, (strips + StripsPerChunk - 1) / StripsPerChunk, c.Threads, (long)nCols * k);
                    }

                    var call = new Int8Call
                    {
                        Format = f,
                        M = coutG,
                        N = nCols,
                        K = k,
                        BPacked = onDemand ? default : new ReadOnlySpan<byte>(packed, (int)f.PackedBBytes(k, nCols)),
                        BSums = onDemand ? default : new ReadOnlySpan<int>(sums, nCols),
                        BSource = onDemand ? panel : null,
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
            ParallelFor.Run(job, (planes + PlanesPerChunk - 1) / PlanesPerChunk, c.Threads, (long)planes * nCols * s.KH * s.KW / 4);
        }
    }

    // Im2col (Chellapilla et al. 2006) fused with the packing of B: strips of Nr output positions, one packed row per kernel tap
    // (input channel, kernel row, kernel column). Taps that fall outside the input read the padding value, the input's zero point.
    private sealed class Im2ColPackJob : ChunkJob
    {
        public Int8Format F;
        public byte* Src;
        public ConvShape S;
        public int CinG;
        public byte Pad;
        public bool U8;
        public int K;
        public byte* Dst;
        public int* ColSums;
        public int Strips;

        public override void RunChunk(int chunk)
        {
            int s0 = chunk * StripsPerChunk;
            int s1 = Math.Min(Strips, s0 + StripsPerChunk);
            int nr = F.Nr, g = F.G, kp = F.KPad(K);
            int outW = S.OutW, nCols = S.OutH * outW;
            int* yBase = stackalloc int[64];
            int* xBase = stackalloc int[64];
            int* sums = stackalloc int[64];
            for (int s = s0; s < s1; s++)
            {
                byte* strip = Dst + ((long)s * nr * kp);
                int j0 = s * nr;
                int cols = Math.Min(nr, nCols - j0);
                for (int c = 0; c < nr; c++)
                {
                    sums[c] = 0;
                    int n = j0 + c;
                    // Columns past the end read nothing: put them far outside the input.
                    yBase[c] = c < cols ? ((n / outW) * S.StrideH) - S.PadTop : int.MinValue / 2;
                    xBase[c] = c < cols ? ((n % outW) * S.StrideW) - S.PadLeft : 0;
                }

                int ci = 0, ky = 0, kx = 0;
                for (int p = 0; p < kp; p++)
                {
                    sbyte* d = (sbyte*)strip + ((p / g) * nr * g) + (p % g);
                    if (p >= K)
                    {
                        for (int c = 0; c < nr; c++)
                        {
                            d[c * g] = 0;
                        }

                        continue;
                    }

                    byte* plane = Src + ((long)ci * S.H * S.W);
                    int yo = ky * S.DilH, xo = kx * S.DilW;
                    for (int c = 0; c < nr; c++)
                    {
                        int iy = yBase[c] + yo, ix = xBase[c] + xo;
                        int v = Int8Gemm.Shift((uint)iy < (uint)S.H && (uint)ix < (uint)S.W ? plane[(iy * S.W) + ix] : Pad, U8);
                        if (c >= cols)
                        {
                            v = 0;
                        }

                        sums[c] += v;
                        d[c * g] = (sbyte)v;
                    }

                    if (++kx == S.KW)
                    {
                        kx = 0;
                        if (++ky == S.KH)
                        {
                            ky = 0;
                            ci++;
                        }
                    }
                }

                for (int c = 0; c < cols; c++)
                {
                    ColSums[j0 + c] = sums[c];
                }
            }
        }
    }

    private sealed class PadJob : ChunkJob
    {
        public byte* Src;
        public bool U8;
        public sbyte Pad;
        public ConvShape S;
        public sbyte* Dst;
        public int Planes;

        public override void RunChunk(int chunk)
        {
            int p0 = chunk * PlanesPerPadChunk;
            Int8Pack.PadPlanes(Src, U8, Pad, S.H, S.W, S.PadTop, S.PadLeft, S.PadBottom, S.PadRight, Dst, p0, Math.Min(Planes, p0 + PlanesPerPadChunk));
        }
    }

    // Packs strips of B on demand: the columns of a 1x1 convolution's input as they are, or the patches of a larger one gathered from
    // the bordered planes. Strip indices are absolute; the packers address by absolute strip, so the destination is offset back.
    private sealed class PanelSource : Int8BPanelSource
    {
        private Int8Format _f;
        private ConvShape _s;
        private bool _pointwise;
        private byte* _src;
        private long _ldb;
        private bool _u8;
        private int _k;
        private int _n;
        private sbyte* _planes;
        private int _planeSize;
        private int _cinG;
        private bool _wantSums;

        public void Set(in Int8Format f, byte* src, long ldb, bool u8, int k, int n, bool wantSums)
        {
            _f = f;
            _pointwise = true;
            _src = src;
            _ldb = ldb;
            _u8 = u8;
            _k = k;
            _n = n;
            _wantSums = wantSums;
        }

        public void Set(in Int8Format f, in ConvShape s, sbyte* planes, int planeSize, int cinG, int k, bool wantSums)
        {
            _f = f;
            _s = s;
            _pointwise = false;
            _planes = planes;
            _planeSize = planeSize;
            _cinG = cinG;
            _k = k;
            _wantSums = wantSums;
        }

        public override void Pack(int firstStrip, int strips, byte* dst, int* sums)
        {
            long back = (long)firstStrip * _f.Nr * _f.KPad(_k);
            int* sumsBack = sums - ((long)firstStrip * _f.Nr);
            if (_pointwise)
            {
                Int8Pack.PackBStrips(in _f, _src, _ldb, _u8, _k, _n, dst - back, sumsBack, _wantSums, firstStrip, firstStrip + strips);
            }
            else
            {
                Int8Pack.Im2ColStrips(in _f, in _s, _planes, _planeSize, _cinG, _k, (sbyte*)(dst - back), sumsBack, _wantSums, firstStrip, firstStrip + strips);
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
            int* offsets = stackalloc int[MaxDirectTaps];
            if (Int8Depthwise.Supports(in S))
            {
                Int8Depthwise.TapOffsets(in S, offsets);
            }

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
                if (Int8Depthwise.Supports(in S))
                {
                    Int8Depthwise.Plane(in S, plane, XU8, XZero, wv, offsets, bias, mult, OutInt32 != null ? OutInt32 + outBase : null,
                        OutQuant != null ? OutQuant + outBase : null, QuantUnsigned, QuantZero);
                    continue;
                }

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
