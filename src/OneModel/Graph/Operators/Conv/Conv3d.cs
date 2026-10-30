using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

/// <summary>
/// Convolution over three spatial axes, X (N, C, D, H, W) with W (M, C / group, kD, kH, kW), as plain loops: for each output the
/// products are added one at a time in the order channel, depth tap, height tap, width tap, starting from +0, then the bias, then the
/// fused activation. The order is fixed and no operation is fused, so the result is the same on every machine (exact), but this is a
/// direct loop with no blocking or vectors: it covers 3-D models and the backend cases, and a lowering onto the matrix product
/// would be the place to make it fast.
/// </summary>
internal static class Conv3d
{
    internal readonly record struct Geometry(PoolGeom Window, int N, int Cin, int Cout, int Group);

    /// <summary>Resolves strides, dilations and padding for a 5-D input and weight; the output spatial size is in <c>Window.Out</c>.</summary>
    public static Geometry Resolve(WindowAttrs attrs, Shape x, Shape w, int group, InferContext? ctx)
    {
        Span<int> kernel = stackalloc int[3], strides = stackalloc int[3], dil = stackalloc int[3], pads = stackalloc int[6];
        attrs.Resolve(ctx, 3, w.Dimensions.Slice(2), kernel, strides, dil, pads);
        Span<int> pb = stackalloc int[3], pe = stackalloc int[3], o = stackalloc int[3];
        for (int a = 0; a < 3; a++)
        {
            pb[a] = pads[a];
            pe[a] = pads[a + 3];
            o[a] = attrs.OutputExtent(ctx, a, x[2 + a], kernel[a], strides[a], dil[a], false, ref pb[a], ref pe[a]);
        }

        var window = new PoolGeom(new Dim3(x[2], x[3], x[4]), new Dim3(kernel[0], kernel[1], kernel[2]), new Dim3(strides[0], strides[1], strides[2]),
            new Dim3(dil[0], dil[1], dil[2]), new Dim3(pb[0], pb[1], pb[2]), new Dim3(pe[0], pe[1], pe[2]), new Dim3(o[0], o[1], o[2]));
        return new Geometry(window, x[0], x[1], w[0], group);
    }

    public static void Run(in Geometry g, ReadOnlySpan<float> x, ReadOnlySpan<float> w, ReadOnlySpan<float> bias, Span<float> y, FusedActivation? activation,
        KernelPath path, bool exact)
    {
        PoolGeom p = g.Window;
        int cinG = g.Cin / g.Group, coutG = g.Cout / g.Group;
        long inVol = p.In.Volume, outVol = p.Out.Volume, kVol = p.Kernel.Volume;
        for (int n = 0; n < g.N; n++)
        {
            for (int co = 0; co < g.Cout; co++)
            {
                int group = co / coutG;
                long o = (((long)n * g.Cout) + co) * outVol;
                for (int od = 0; od < p.Out.D; od++)
                {
                    for (int oh = 0; oh < p.Out.H; oh++)
                    {
                        for (int ow = 0; ow < p.Out.W; ow++, o++)
                        {
                            float acc = 0f;
                            for (int ci = 0; ci < cinG; ci++)
                            {
                                long xBase = (((long)n * g.Cin) + (group * cinG) + ci) * inVol;
                                long wBase = (((long)co * cinG) + ci) * kVol;
                                for (int kd = 0; kd < p.Kernel.D; kd++)
                                {
                                    int d = (od * p.Stride.D) - p.Pad.D + (kd * p.Dilation.D);
                                    if ((uint)d >= (uint)p.In.D)
                                    {
                                        continue;
                                    }

                                    for (int kh = 0; kh < p.Kernel.H; kh++)
                                    {
                                        int h = (oh * p.Stride.H) - p.Pad.H + (kh * p.Dilation.H);
                                        if ((uint)h >= (uint)p.In.H)
                                        {
                                            continue;
                                        }

                                        for (int kw = 0; kw < p.Kernel.W; kw++)
                                        {
                                            int wi = (ow * p.Stride.W) - p.Pad.W + (kw * p.Dilation.W);
                                            if ((uint)wi < (uint)p.In.W)
                                            {
                                                float prod = x[(int)(xBase + ((((long)d * p.In.H) + h) * p.In.W) + wi)] * w[(int)(wBase + ((((long)kd * p.Kernel.H) + kh) * p.Kernel.W) + kw)];
                                                acc += prod;
                                            }
                                        }
                                    }
                                }
                            }

                            if (!bias.IsEmpty)
                            {
                                acc += bias[co];
                            }

                            y[(int)o] = acc;
                        }
                    }
                }
            }
        }

        activation?.Apply(path, exact, y);
    }
}
