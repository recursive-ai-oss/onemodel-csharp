using System;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>The geometry of a 2-D transposed convolution: input N x Cin x H x W, output N x Cout x OutH x OutW, weights [Cin, Cout/Groups, KH, KW].</summary>
internal readonly record struct ConvTransposeShape(
    int N, int Cin, int H, int W, int Cout, int KH, int KW,
    int StrideH, int StrideW, int PadTop, int PadLeft, int DilH, int DilW, int Groups, int OutH, int OutW);

/// <summary>
/// Transposed convolution as the matrix product followed by col2im, the adjoint of the im2col lowering of Conv (Chellapilla et al.
/// 2006): for each group the columns C = Wg' * Xg have one row per (output channel, kernel row, kernel column) and one column per
/// input position, and each entry is added into the output pixel that tap lands on. An output pixel receives its taps in increasing
/// (kernel row, kernel column) order from +0; the channel sum is inside the matrix product. Bias is added last, once.
/// </summary>
internal static class ConvTransposeKernel
{
    /// <summary>
    /// Computes the transposed convolution. <paramref name="wt"/> is the weights rearranged as [Groups][Cout/Groups * KH * KW][Cin/Groups]; <paramref name="cols"/> is scratch of at least Cout/Groups * KH * KW * H * W floats.
    /// </summary>
    /// <exception cref="ArgumentException">A buffer is too small.</exception>
    public static void Run(
        KernelMode mode, KernelPath path, in ConvTransposeShape s,
        ReadOnlySpan<float> input, ReadOnlySpan<float> wt, ReadOnlySpan<float> bias, Span<float> output, Span<float> cols, int threads)
    {
        int cinG = s.Cin / s.Groups, coutG = s.Cout / s.Groups, taps = coutG * s.KH * s.KW;
        int inPlane = s.H * s.W, outPlane = s.OutH * s.OutW;
        if (input.Length < (long)s.N * s.Cin * inPlane || wt.Length < (long)s.Groups * taps * cinG ||
            output.Length < (long)s.N * s.Cout * outPlane || cols.Length < (long)taps * inPlane ||
            (!bias.IsEmpty && bias.Length < s.Cout))
        {
            throw new ArgumentException("A buffer is smaller than the transposed convolution needs.");
        }

        bool canon = mode == KernelMode.Exact;
        output.Slice(0, s.N * s.Cout * outPlane).Clear();
        for (int img = 0; img < s.N; img++)
        {
            for (int g = 0; g < s.Groups; g++)
            {
                ReadOnlySpan<float> x = input.Slice(((img * s.Cin) + (g * cinG)) * inPlane, cinG * inPlane);
                Gemm.Run(mode, path, taps, inPlane, cinG, wt.Slice(g * taps * cinG, taps * cinG), cinG, x, inPlane, cols, inPlane, threads);
                for (int co = 0; co < coutG; co++)
                {
                    Span<float> y = output.Slice(((img * s.Cout) + (g * coutG) + co) * outPlane, outPlane);
                    for (int ky = 0; ky < s.KH; ky++)
                    {
                        for (int kx = 0; kx < s.KW; kx++)
                        {
                            ReadOnlySpan<float> c = cols.Slice((((co * s.KH) + ky) * s.KW + kx) * inPlane, inPlane);
                            Scatter(in s, c, y, (ky * s.DilH) - s.PadTop, (kx * s.DilW) - s.PadLeft);
                        }
                    }
                }
            }
        }

        if (!bias.IsEmpty || canon)
        {
            for (int q = 0; q < s.N * s.Cout; q++)
            {
                Span<float> y = output.Slice(q * outPlane, outPlane);
                float b = bias.IsEmpty ? 0f : bias[q % s.Cout];
                for (int i = 0; i < y.Length; i++)
                {
                    float v = bias.IsEmpty ? y[i] : y[i] + b;
                    y[i] = canon && float.IsNaN(v) ? BitConverter.UInt32BitsToSingle(0x7FC00000u) : v;
                }
            }
        }
    }

    // y[oy, ox] += c[iy, ix] where oy = iy * strideH + offY and ox = ix * strideW + offX, for the positions inside the output.
    private static void Scatter(in ConvTransposeShape s, ReadOnlySpan<float> c, Span<float> y, int offY, int offX)
    {
        for (int iy = 0; iy < s.H; iy++)
        {
            int oy = (iy * s.StrideH) + offY;
            if ((uint)oy >= (uint)s.OutH)
            {
                continue;
            }

            ReadOnlySpan<float> row = c.Slice(iy * s.W, s.W);
            Span<float> yrow = y.Slice(oy * s.OutW, s.OutW);
            for (int ix = 0; ix < row.Length; ix++)
            {
                int ox = (ix * s.StrideW) + offX;
                if ((uint)ox < (uint)yrow.Length)
                {
                    yrow[ox] += row[ix];
                }
            }
        }
    }
}
