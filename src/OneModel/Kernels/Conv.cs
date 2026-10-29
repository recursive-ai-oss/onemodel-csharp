using System;
using System.Runtime.CompilerServices;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Kernels.Fast;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>The geometry of a 2-D convolution in NCHW layout with weights [Cout, Cin/Groups, KH, KW].</summary>
internal readonly record struct ConvShape(
    int N, int Cin, int H, int W, int Cout, int KH, int KW,
    int StrideH, int StrideW, int PadTop, int PadLeft, int PadBottom, int PadRight, int DilH, int DilW, int Groups)
{
    /// <summary>Output height: floor((H + padding - dilation * (KH - 1) - 1) / stride) + 1.</summary>
    public int OutH => ((H + PadTop + PadBottom - (DilH * (KH - 1)) - 1) / StrideH) + 1;

    /// <summary>Output width.</summary>
    public int OutW => ((W + PadLeft + PadRight - (DilW * (KW - 1)) - 1) / StrideW) + 1;

    /// <summary>Whether each group has one input channel (depthwise, possibly with a channel multiplier).</summary>
    public bool IsDepthwise => Groups == Cin && Cin > 0;

    /// <summary>FLOPs of the convolution, counting a multiply-add as 2.</summary>
    public double Flops => 2.0 * N * Cout * OutH * OutW * (Cin / Groups) * KH * KW;
}

/// <summary>
/// 2-D convolution. Regular and grouped convolutions are lowered to the matrix product by im2col (Chellapilla, Puri and Simard
/// 2006): each group's input patches become the columns of a (Cin/G*KH*KW) x (OutH*OutW) matrix B, the group's weights are the
/// rows of A, and C is that group's output channels. Depthwise convolutions (one input channel per group) run a direct kernel
/// instead, because their matrix product has k = KH*KW and would be all overhead. A 1x1 stride-1 unpadded convolution reads the
/// input as B directly, with no copy.
/// </summary>
/// <remarks>
/// Both paths add taps in the order channel, then kernel row, then kernel column, starting from +0, one tap at a time. Padding
/// taps are real multiplications by zero (so an infinite or NaN weight meets padding as NaN, the same on both paths). Exact mode
/// canonicalises NaN outputs as the matrix product does.
/// </remarks>
internal static unsafe class Conv
{
    // Rows of the im2col matrix per chunk, and planes per chunk of the depthwise kernel: constants, not derived from the thread count.
    private const int Im2ColRowsPerChunk = 16;
    private const int PlanesPerChunk = 4;

    /// <summary>
    /// Computes the convolution of <paramref name="input"/> (N x Cin x H x W) with <paramref name="weights"/>
    /// (Cout x Cin/Groups x KH x KW) into <paramref name="output"/> (N x Cout x OutH x OutW). No bias.
    /// </summary>
    /// <exception cref="ArgumentException">A dimension is not positive, groups do not divide the channels, the output would be empty, or a buffer is too small.</exception>
    public static void Run(
        KernelMode mode, KernelPath path, in ConvShape s,
        ReadOnlySpan<float> input, ReadOnlySpan<float> weights, Span<float> output, int threads)
    {
        Validate(s);
        long inLen = (long)s.N * s.Cin * s.H * s.W;
        long wLen = (long)s.Cout * (s.Cin / s.Groups) * s.KH * s.KW;
        long outLen = (long)s.N * s.Cout * s.OutH * s.OutW;
        if (input.Length < inLen)
        {
            throw new ArgumentException("The input is smaller than N x Cin x H x W.", nameof(input));
        }

        if (weights.Length < wLen)
        {
            throw new ArgumentException("The weights are smaller than Cout x Cin/Groups x KH x KW.", nameof(weights));
        }

        if (output.Length < outLen)
        {
            throw new ArgumentException("The output is smaller than N x Cout x OutH x OutW.", nameof(output));
        }

        if (!KernelPaths.IsSupported(path))
        {
            throw new PlatformNotSupportedException($"Kernel path {path} is not supported on this machine.");
        }

        fixed (float* pin = input, pw = weights, pout = output)
        {
            if (s.IsDepthwise)
            {
                Depthwise(mode, path, s, pin, pw, pout, threads);
            }
            else
            {
                ViaGemm(mode, path, s, pin, pw, pout, threads);
            }
        }
    }

    private static void Validate(in ConvShape s)
    {
        if (s.N <= 0 || s.Cin <= 0 || s.H <= 0 || s.W <= 0 || s.Cout <= 0 || s.KH <= 0 || s.KW <= 0 ||
            s.StrideH <= 0 || s.StrideW <= 0 || s.DilH <= 0 || s.DilW <= 0 || s.Groups <= 0 ||
            s.PadTop < 0 || s.PadLeft < 0 || s.PadBottom < 0 || s.PadRight < 0)
        {
            throw new ArgumentException("Convolution sizes, strides, dilations and groups must be positive and paddings non-negative.");
        }

        if (s.Cin % s.Groups != 0 || s.Cout % s.Groups != 0)
        {
            throw new ArgumentException("Groups must divide both the input and the output channels.");
        }

        long span = (long)s.H + s.PadTop + s.PadBottom - ((long)s.DilH * (s.KH - 1)) - 1;
        long spanW = (long)s.W + s.PadLeft + s.PadRight - ((long)s.DilW * (s.KW - 1)) - 1;
        if (span < 0 || spanW < 0)
        {
            throw new ArgumentException("The kernel does not fit in the padded input; the output would be empty.");
        }

        // Keep every product of sizes inside 32-bit indices where the kernels use them, and inside long everywhere.
        long total = (long)s.N * Math.Max(s.Cin, s.Cout) * Math.Max((long)s.H * s.W, (long)s.OutH * s.OutW);
        if (total > int.MaxValue)
        {
            throw new ArgumentException("The convolution has more than 2^31 elements in one tensor; not supported.");
        }
    }

    private static void ViaGemm(KernelMode mode, KernelPath path, in ConvShape s, float* input, float* weights, float* output, int threads)
    {
        int cinG = s.Cin / s.Groups;
        int coutG = s.Cout / s.Groups;
        int k = cinG * s.KH * s.KW;
        int outH = s.OutH;
        int outW = s.OutW;
        int nCols = outH * outW;
        bool pointwise = s.KH == 1 && s.KW == 1 && s.StrideH == 1 && s.StrideW == 1 &&
                         s.PadTop == 0 && s.PadLeft == 0 && s.PadBottom == 0 && s.PadRight == 0;
        float* cols = pointwise ? null : Scratch.Get(Scratch.Im2Col, (long)k * nCols);

        for (int img = 0; img < s.N; img++)
        {
            for (int g = 0; g < s.Groups; g++)
            {
                float* src = input + ((((long)img * s.Cin) + (g * cinG)) * s.H * s.W);
                float* b = src;
                if (!pointwise)
                {
                    var job = new Im2ColJob { Src = src, Dst = cols, S = s, CinG = cinG, OutH = outH, OutW = outW };
                    ParallelFor.Run(job, (k + Im2ColRowsPerChunk - 1) / Im2ColRowsPerChunk, threads);
                    b = cols;
                }

                float* a = weights + ((long)g * coutG * k);
                float* c = output + ((((long)img * s.Cout) + (g * coutG)) * nCols);
                Gemm.RunCore(mode, path, coutG, nCols, k, a, k, b, nCols, c, nCols, threads);
            }
        }
    }

    private sealed class Im2ColJob : ChunkJob
    {
        public float* Src;
        public float* Dst;
        public ConvShape S;
        public int CinG;
        public int OutH;
        public int OutW;

        public override void RunChunk(int chunk)
        {
            int rows = CinG * S.KH * S.KW;
            int r0 = chunk * Im2ColRowsPerChunk;
            int r1 = Math.Min(rows, r0 + Im2ColRowsPerChunk);
            for (int r = r0; r < r1; r++)
            {
                int kx = r % S.KW;
                int ky = (r / S.KW) % S.KH;
                int ci = r / (S.KW * S.KH);
                float* plane = Src + ((long)ci * S.H * S.W);
                float* dst = Dst + ((long)r * OutH * OutW);
                // Output columns whose input column ox*stride - padLeft + kx*dilation lies inside [0, W).
                int shiftX = (kx * S.DilW) - S.PadLeft;
                int oxLo = shiftX >= 0 ? 0 : (-shiftX + S.StrideW - 1) / S.StrideW;
                int oxHi = S.W - 1 - shiftX < 0 ? -1 : Math.Min(OutW - 1, (S.W - 1 - shiftX) / S.StrideW);
                for (int oy = 0; oy < OutH; oy++)
                {
                    int iy = (oy * S.StrideH) - S.PadTop + (ky * S.DilH);
                    float* row = dst + ((long)oy * OutW);
                    if (iy < 0 || iy >= S.H || oxHi < oxLo)
                    {
                        new Span<float>(row, OutW).Clear();
                        continue;
                    }

                    float* srcRow = plane + ((long)iy * S.W);
                    new Span<float>(row, oxLo).Clear();
                    if (S.StrideW == 1)
                    {
                        Buffer.MemoryCopy(srcRow + oxLo + shiftX, row + oxLo, (oxHi - oxLo + 1) * sizeof(float), (oxHi - oxLo + 1) * sizeof(float));
                    }
                    else
                    {
                        for (int ox = oxLo; ox <= oxHi; ox++)
                        {
                            row[ox] = srcRow[(ox * S.StrideW) + shiftX];
                        }
                    }

                    new Span<float>(row + oxHi + 1, OutW - oxHi - 1).Clear();
                }
            }
        }
    }

    private static void Depthwise(KernelMode mode, KernelPath path, in ConvShape s, float* input, float* weights, float* output, int threads)
    {
        int planes = s.N * s.Cout;
        int chunks = (planes + PlanesPerChunk - 1) / PlanesPerChunk;
        bool exact = mode == KernelMode.Exact;
        bool wide = path is KernelPath.Avx2 or KernelPath.Avx512;
        bool four = path is KernelPath.Neon or KernelPath.Portable128;
        ChunkJob job = (exact, wide, four) switch
        {
            (true, true, _) => new DepthwiseJob<ExactLanes8, ExactLanes1>(s, input, weights, output, true),
            (true, _, true) => new DepthwiseJob<ExactLanes4, ExactLanes1>(s, input, weights, output, true),
            (true, _, _) => new DepthwiseJob<ExactLanes1, ExactLanes1>(s, input, weights, output, true),
            (_, true, _) => new DepthwiseJob<FastLanes8, FastLanes1>(s, input, weights, output, false),
            (_, _, true) => new DepthwiseJob<FastLanes4, FastLanes1>(s, input, weights, output, false),
            _ => new DepthwiseJob<FastLanes1, FastLanes1>(s, input, weights, output, false),
        };
        ParallelFor.Run(job, chunks, threads);
    }

    /// <summary>
    /// Direct depthwise kernel over <typeparamref name="TV"/> lanes (output columns side by side) with a one-lane tail
    /// <typeparamref name="TS"/>. Each output is the sum over taps in row-major order of weight * input, accumulated one tap at a time
    /// from +0 on a zero-padded copy of the input plane.
    /// </summary>
    private sealed class DepthwiseJob<TV, TS> : ChunkJob
        where TV : struct, ILanes<TV>
        where TS : struct, ILanes<TS>
    {
        private readonly ConvShape _s;
        private readonly float* _input;
        private readonly float* _weights;
        private readonly float* _output;
        private readonly bool _canonicalNaN;

        public DepthwiseJob(in ConvShape s, float* input, float* weights, float* output, bool canonicalNaN)
        {
            _s = s;
            _input = input;
            _weights = weights;
            _output = output;
            _canonicalNaN = canonicalNaN;
        }

        public override void RunChunk(int chunk)
        {
            ConvShape s = _s;
            int mult = s.Cout / s.Cin;
            int outH = s.OutH;
            int outW = s.OutW;
            int planes = s.N * s.Cout;
            bool padded = s.PadTop != 0 || s.PadLeft != 0 || s.PadBottom != 0 || s.PadRight != 0;
            int wp = s.W + s.PadLeft + s.PadRight;
            int hp = s.H + s.PadTop + s.PadBottom;
            float* scratch = padded ? Scratch.Get(Scratch.Plane, (long)hp * wp) : null;
            for (int q = chunk * PlanesPerChunk; q < Math.Min(planes, (chunk + 1) * PlanesPerChunk); q++)
            {
                int img = q / s.Cout;
                int oc = q % s.Cout;
                float* plane = _input + ((((long)img * s.Cin) + (oc / mult)) * s.H * s.W);
                float* src = plane;
                if (padded)
                {
                    new Span<float>(scratch, hp * wp).Clear();
                    for (int y = 0; y < s.H; y++)
                    {
                        Buffer.MemoryCopy(plane + ((long)y * s.W), scratch + ((long)(y + s.PadTop) * wp) + s.PadLeft, s.W * sizeof(float), s.W * sizeof(float));
                    }

                    src = scratch;
                }

                float* w = _weights + ((long)oc * s.KH * s.KW);
                float* dst = _output + ((long)q * outH * outW);
                int vecEnd = outW - (outW % TV.Width);
                Rows<TV>(src, wp, w, in s, dst, outH, outW, 0, vecEnd);
                Rows<TS>(src, wp, w, in s, dst, outH, outW, vecEnd, outW);
                if (_canonicalNaN)
                {
                    uint* bits = (uint*)dst;
                    for (int i = 0; i < outH * outW; i++)
                    {
                        if ((bits[i] & 0x7FFFFFFFu) > 0x7F800000u)
                        {
                            bits[i] = 0x7FC00000u;
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void Rows<T>(float* src, int wp, float* w, in ConvShape s, float* dst, int outH, int outW, int ox0, int ox1)
            where T : struct, ILanes<T>
        {
            int width = T.Width;
            nint strideW = s.StrideW;
            for (int oy = 0; oy < outH; oy++)
            {
                float* rowBase = src + ((long)oy * s.StrideH * wp);
                float* outRow = dst + ((long)oy * outW);
                for (int ox = ox0; ox + width <= ox1; ox += width)
                {
                    T acc = T.Zero;
                    float* tap = w;
                    for (int ky = 0; ky < s.KH; ky++)
                    {
                        float* line = rowBase + ((long)ky * s.DilH * wp) + ((long)ox * s.StrideW);
                        for (int kx = 0; kx < s.KW; kx++)
                        {
                            float* p = line + ((long)kx * s.DilW);
                            T v = s.StrideW == 1 ? T.Load(p) : T.Gather(p, strideW);
                            acc = T.MulAdd(T.Broadcast(*tap), v, acc);
                            tap++;
                        }
                    }

                    acc.Store(outRow + ox);
                }
            }
        }
    }
}
