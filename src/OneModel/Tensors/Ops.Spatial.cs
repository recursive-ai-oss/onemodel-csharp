using System;
using RecursiveAI.OneModel.Kernels;
namespace RecursiveAI.OneModel.Tensors;

/// <summary>Geometry of a 2-D transposed convolution: stride, symmetric padding, output padding, dilation (each per axis) and groups.</summary>
/// <param name="StrideH">Vertical step between input positions in the output, at least 1.</param>
/// <param name="StrideW">Horizontal step, at least 1.</param>
/// <param name="PadH">Rows removed from the top and the bottom of the full output, at least 0.</param>
/// <param name="PadW">Columns removed left and right, at least 0.</param>
/// <param name="OutPadH">Extra rows added at the bottom (to resolve the output size), in <c>[0, max(StrideH, DilH))</c>.</param>
/// <param name="OutPadW">Extra columns added at the right.</param>
/// <param name="DilH">Spacing between kernel rows, at least 1.</param>
/// <param name="DilW">Spacing between kernel columns, at least 1.</param>
/// <param name="Groups">Channel groups, at least 1; both channel counts must be divisible by it.</param>
public readonly record struct ConvTranspose2dParams(int StrideH = 1, int StrideW = 1, int PadH = 0, int PadW = 0, int OutPadH = 0, int OutPadW = 0,
    int DilH = 1, int DilW = 1, int Groups = 1);

/// <summary>How <see cref="Ops.Resize2d"/> interpolates.</summary>
public enum InterpolationMode
{
    /// <summary>Nearest neighbour: source index <c>min(floor(dst · in/out), in−1)</c> (PyTorch's <c>nearest</c>).</summary>
    Nearest,
    /// <summary>Bilinear interpolation of the four surrounding pixels.</summary>
    Bilinear,
}

public static partial class Ops
{
    /// <summary>
    /// 2-D transposed convolution (the adjoint of <see cref="Conv2d"/>, "deconvolution"): every input value is multiplied by the
    /// kernel and added into the output at <c>(iy·stride − pad + ky·dil, ix·stride − pad + kx·dil)</c>. Output size per axis is
    /// <c>(in − 1)·stride − 2·pad + dil·(k − 1) + outPad + 1</c>. Runs on the blocked GEMM followed by col2im.
    /// Record: slots x and weight (copies, owned, if not contiguous); Ints = the geometry as a convolution with the roles of input and
    /// output swapped, [strideH, strideW, padTop, padLeft, padBottom, padRight, dilH, dilW, groups]. Backward:
    /// <c>dx = Conv2d(g, W)</c>, <c>dW = x · im2col(g)ᵀ</c> per group, <c>dbias = Σg</c>.
    /// </summary>
    /// <param name="x">Input <c>[N, Cin, H, W]</c>.</param>
    /// <param name="weight">Weights <c>[Cin, Cout/Groups, KH, KW]</c>.</param>
    /// <param name="bias">Bias <c>[Cout]</c>, or an empty tensor for none.</param>
    /// <param name="p">Geometry; the default (zero) fields mean stride 1, dilation 1, groups 1, no padding.</param>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    /// <exception cref="TensorShapeException">The shapes or the geometry do not fit.</exception>
    public static Tensor ConvTranspose2d(in Tensor x, in Tensor weight, in Tensor bias = default, ConvTranspose2dParams p = default, BufferPool? pool = null)
    {
        RequireFloat32(x);
        RequireFloat32(weight);
        p = p with
        {
            StrideH = p.StrideH == 0 ? 1 : p.StrideH, StrideW = p.StrideW == 0 ? 1 : p.StrideW,
            DilH = p.DilH == 0 ? 1 : p.DilH, DilW = p.DilW == 0 ? 1 : p.DilW, Groups = p.Groups == 0 ? 1 : p.Groups,
        };
        if (x.Rank != 4 || weight.Rank != 4)
        {
            throw new TensorShapeException($"ConvTranspose2d needs an (N,C,H,W) input and (Cin,Cout/G,KH,KW) weights, not {x.Shape} and {weight.Shape}.");
        }

        int cin = x.Shape[1], coutG = weight.Shape[1], kh = weight.Shape[2], kw = weight.Shape[3];
        if (p.Groups < 1 || cin % p.Groups != 0 || weight.Shape[0] != cin)
        {
            throw new TensorShapeException($"Weights {weight.Shape} do not fit input {x.Shape} with {p.Groups} groups.");
        }

        int cout = coutG * p.Groups;
        if (!bias.IsEmpty)
        {
            RequireFloat32(bias);
            if (bias.Rank != 1 || bias.Shape[0] != cout)
            {
                throw new TensorShapeException($"The bias must have shape [{cout}], not {bias.Shape}.");
            }
        }

        if (p.StrideH < 1 || p.StrideW < 1 || p.DilH < 1 || p.DilW < 1 || p.PadH < 0 || p.PadW < 0 || p.OutPadH < 0 || p.OutPadW < 0 ||
            p.OutPadH >= Math.Max(p.StrideH, p.DilH) || p.OutPadW >= Math.Max(p.StrideW, p.DilW))
        {
            throw new TensorShapeException("Invalid transposed-convolution geometry.");
        }

        long oh = ((long)(x.Shape[2] - 1) * p.StrideH) - (2L * p.PadH) + ((long)p.DilH * (kh - 1)) + p.OutPadH + 1;
        long ow = ((long)(x.Shape[3] - 1) * p.StrideW) - (2L * p.PadW) + ((long)p.DilW * (kw - 1)) + p.OutPadW + 1;
        if (oh < 1 || ow < 1 || oh > int.MaxValue || ow > int.MaxValue || x.Length == 0)
        {
            throw new TensorShapeException($"The transposed convolution of {x.Shape} with {weight.Shape} has an empty or oversized output.");
        }

        var s = new ConvTransposeShape(x.Shape[0], cin, x.Shape[2], x.Shape[3], cout, kh, kw, p.StrideH, p.StrideW, p.PadH, p.PadW,
            p.DilH, p.DilW, p.Groups, (int)oh, (int)ow);
        Tensor xc = Contig(x, out bool xcopied), wc = Contig(weight, out bool wcopied);
        Tensor bc = bias.IsEmpty ? default : Contig(bias, out _);
        int cinG = cin / p.Groups, taps = coutG * kh * kw;
        Tensor wt = Tensor.Create(DType.F32, Shape.Create(p.Groups * taps * cinG));
        Tensor cols = Tensor.Create(DType.F32, Shape.Create(taps * s.H * s.W));
        Tensor r = NewOut(Shape.Create(s.N, cout, s.OutH, s.OutW), pool);
        try
        {
            // wt[g][(co * KH + ky) * KW + kx][ci] = w[g * cinG + ci][co][ky][kx]: the group's weights as a taps x cinG matrix.
            ReadOnlySpan<float> wv = wc.Data<float>();
            Span<float> wtv = wt.MutableData<float>();
            for (int g = 0; g < p.Groups; g++)
            {
                for (int ci = 0; ci < cinG; ci++)
                {
                    for (int t = 0; t < taps; t++)
                    {
                        wtv[(((g * taps) + t) * cinG) + ci] = wv[(((g * cinG) + ci) * taps) + t];
                    }
                }
            }

            ConvTransposeKernel.Run(TensorCompute.KMode, TensorCompute.Path, s, xc.Data<float>(), wtv, bc.IsEmpty ? default : bc.Data<float>(),
                r.MutableData<float>(), cols.MutableData<float>(), TensorCompute.Threads);
        }
        finally
        {
            wt.Dispose();
            cols.Dispose();
        }

        if (!Recording(x, weight, bias))
        {
            Free(xc, xcopied);
            Free(wc, wcopied);
            return r;
        }

        BackwardRecord br = BackwardRecord.Create(OpKind.ConvTranspose2d, 2, x, weight, bias);
        SaveContig(br, 0, xc, xcopied);
        SaveContig(br, 1, wc, wcopied);
        br.SetInts(p.StrideH, p.StrideW, p.PadH, p.PadW, p.PadH, p.PadW, p.DilH, p.DilW, p.Groups);
        return br.Attach(r);
    }

    /// <summary>
    /// Resizes the spatial dimensions of an NCHW tensor to <c>outH × outW</c>. Source coordinates follow PyTorch's
    /// <c>interpolate</c>: with scale <c>in/out</c> (or <c>(in−1)/(out−1)</c>, 0 for a single output, when <paramref name="alignCorners"/>),
    /// nearest takes <c>floor(dst·scale)</c>; bilinear takes <c>scale·(dst+0.5)−0.5</c> (or <c>scale·dst</c>), clamped at 0, and blends the pixel
    /// and its right/lower neighbour (clamped at the edge) with weights <c>1−λ</c> and <c>λ</c>. Backward scatters the gradient to the
    /// same pixels with the same weights in output scan order.
    /// Record: Ints = [mode, alignCorners, outH, outW]; nothing saved (the input shape is in the record).
    /// </summary>
    /// <param name="x">Input <c>[N, C, H, W]</c>.</param>
    /// <param name="outH">Output height, at least 1.</param>
    /// <param name="outW">Output width, at least 1.</param>
    /// <param name="mode">Nearest or bilinear; default nearest.</param>
    /// <param name="alignCorners">Map the corner pixels of input and output onto each other (bilinear only; ignored for nearest). Default false.</param>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    /// <exception cref="TensorShapeException">The input is not rank 4 or a size is not positive.</exception>
    public static Tensor Resize2d(in Tensor x, int outH, int outW, InterpolationMode mode = InterpolationMode.Nearest, bool alignCorners = false, BufferPool? pool = null)
    {
        RequireFloat32(x);
        if (x.Rank != 4 || outH < 1 || outW < 1 || x.Shape[2] < 1 || x.Shape[3] < 1)
        {
            throw new TensorShapeException($"Resize2d needs an (N,C,H,W) input and a positive output size, not {x.Shape} and {outH}x{outW}.");
        }

        Tensor xc = Contig(x, out bool copied);
        int planes = x.Shape[0] * x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        Tensor r = NewOut(Shape.Create(x.Shape[0], x.Shape[1], outH, outW), pool);
        ReadOnlySpan<float> src = xc.Data<float>();
        Span<float> dst = r.MutableData<float>();
        bool align = alignCorners && mode == InterpolationMode.Bilinear;
        for (int pl = 0; pl < planes; pl++)
        {
            ReadOnlySpan<float> sp = src.Slice(pl * h * w, h * w);
            for (int oy = 0; oy < outH; oy++)
            {
                SourceCoord(mode, align, oy, h, outH, out int y0, out int y1, out float ly);
                for (int ox = 0; ox < outW; ox++)
                {
                    SourceCoord(mode, align, ox, w, outW, out int x0, out int x1, out float lx);
                    float v;
                    if (mode == InterpolationMode.Nearest)
                    {
                        v = sp[(y0 * w) + x0];
                    }
                    else
                    {
                        float top = ((1f - lx) * sp[(y0 * w) + x0]) + (lx * sp[(y0 * w) + x1]);
                        float bottom = ((1f - lx) * sp[(y1 * w) + x0]) + (lx * sp[(y1 * w) + x1]);
                        v = ((1f - ly) * top) + (ly * bottom);
                    }

                    dst[(pl * outH * outW) + (oy * outW) + ox] = v;
                }
            }
        }

        Free(xc, copied);
        if (!Recording(x))
        {
            return r;
        }

        BackwardRecord br = BackwardRecord.Create(OpKind.Resize2d, 0, x);
        br.SetInts((int)mode, align ? 1 : 0, outH, outW);
        return br.Attach(r);
    }

    /// <summary>The source pixels and the blend weight of the second one for output coordinate <paramref name="dst"/> along an axis.</summary>
    internal static void SourceCoord(InterpolationMode mode, bool alignCorners, int dst, int inSize, int outSize, out int i0, out int i1, out float lambda)
    {
        if (mode == InterpolationMode.Nearest)
        {
            float scale = (float)inSize / outSize;
            i0 = Math.Min((int)MathF.Floor(dst * scale), inSize - 1);
            i1 = i0;
            lambda = 0f;
            return;
        }

        float src;
        if (alignCorners)
        {
            float scale = outSize > 1 ? (float)(inSize - 1) / (outSize - 1) : 0f;
            src = scale * dst;
        }
        else
        {
            float scale = (float)inSize / outSize;
            src = (scale * (dst + 0.5f)) - 0.5f;
            if (src < 0f)
            {
                src = 0f;
            }
        }

        i0 = (int)src;
        if (i0 > inSize - 1)
        {
            i0 = inSize - 1;
        }

        i1 = i0 + (i0 < inSize - 1 ? 1 : 0);
        lambda = src - i0;
    }
}
