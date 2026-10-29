using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
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
    // Rows of the im2col matrix per chunk, and planes per chunk of the bias add: constants, not derived from the thread count.
    private const int Im2ColRowsPerChunk = 16;
    private const int PlanesPerChunk = 4;

    /// <summary>
    /// Computes the convolution of <paramref name="input"/> (N x Cin x H x W) with <paramref name="weights"/>
    /// (Cout x Cin/Groups x KH x KW) into <paramref name="output"/> (N x Cout x OutH x OutW). When <paramref name="bias"/> is not
    /// empty (Cout values) it is added to each output channel after the taps, as one rounded addition per output.
    /// </summary>
    /// <exception cref="ArgumentException">A dimension is not positive, groups do not divide the channels, the output would be empty, or a buffer is too small.</exception>
    public static void Run(
        KernelMode mode, KernelPath path, in ConvShape s,
        ReadOnlySpan<float> input, ReadOnlySpan<float> weights, Span<float> output, int threads, ReadOnlySpan<float> bias = default)
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

        if (!bias.IsEmpty && bias.Length < s.Cout)
        {
            throw new ArgumentException("The bias is smaller than Cout.", nameof(bias));
        }

        if (!KernelPaths.IsSupported(path))
        {
            throw new PlatformNotSupportedException($"Kernel path {path} is not supported on this machine.");
        }

        fixed (float* pin = input, pw = weights, pout = output, pbias = bias)
        {
            if (s.IsDepthwise)
            {
                Depthwise.Run(mode, path, s, pin, pw, pout, threads);
            }
            else
            {
                ViaGemm(mode, path, s, pin, pw, pout, threads);
            }

            if (!bias.IsEmpty)
            {
                AddChannelBias(mode == KernelMode.Exact, pbias, pout, s.N, s.Cout, s.OutH * s.OutW, threads);
            }
        }
    }

    [ThreadStatic]
    private static BiasJob? t_biasJob;

    [ThreadStatic]
    private static Im2ColJob? t_im2colJob;

    /// <summary>
    /// output[n, c, :] += bias[c] for every plane, in place. One rounded addition per element, so fast and exact agree (exact mode
    /// also rewrites NaN results as the canonical NaN). Planes per chunk is fixed, not derived from the thread count.
    /// </summary>
    /// <remarks>Bounds: <c>output</c> covers n * channels * plane floats and <c>bias</c> covers channels, both checked by the callers.</remarks>
    internal static void AddChannelBias(bool canonicalNaN, float* bias, float* output, int n, int channels, int plane, int threads)
    {
        BiasJob job = t_biasJob ??= new BiasJob();
        job.Bias = bias;
        job.Output = output;
        job.Channels = channels;
        job.Plane = plane;
        job.Canonical = canonicalNaN;
        job.Planes = n * channels;
        ParallelFor.Run(job, (job.Planes + PlanesPerChunk - 1) / PlanesPerChunk, threads, (long)job.Planes * plane / 4);
    }

    private sealed class BiasJob : ChunkJob
    {
        public float* Bias;
        public float* Output;
        public int Channels;
        public int Plane;
        public int Planes;
        public bool Canonical;

        public override void RunChunk(int chunk)
        {
            int qEnd = Math.Min(Planes, (chunk + 1) * PlanesPerChunk);
            for (int q = chunk * PlanesPerChunk; q < qEnd; q++)
            {
                float* p = Output + ((long)q * Plane);
                float b = Bias[q % Channels];
                int i = 0;
                if (Vector128.IsHardwareAccelerated)
                {
                    var vb = Vector128.Create(b);
                    for (; i + 4 <= Plane; i += 4)
                    {
                        Vector128.Store(Vector128.Add(Vector128.Load(p + i), vb), p + i);
                    }
                }

                for (; i < Plane; i++)
                {
                    p[i] += b;
                }

                if (Canonical)
                {
                    uint* bits = (uint*)p;
                    for (int j = 0; j < Plane; j++)
                    {
                        if ((bits[j] & 0x7FFFFFFFu) > 0x7F800000u)
                        {
                            bits[j] = 0x7FC00000u;
                        }
                    }
                }
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
                    // Kept per thread and refilled: a steady-state call allocates nothing (the call is done with it before it returns).
                    Im2ColJob job = t_im2colJob ??= new Im2ColJob();
                    job.Src = src;
                    job.Dst = cols;
                    job.S = s;
                    job.CinG = cinG;
                    job.OutH = outH;
                    job.OutW = outW;
                    ParallelFor.Run(job, (k + Im2ColRowsPerChunk - 1) / Im2ColRowsPerChunk, threads, (long)k * nCols / 2);
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
}
