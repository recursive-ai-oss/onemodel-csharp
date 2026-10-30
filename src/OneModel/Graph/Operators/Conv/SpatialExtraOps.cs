using System;
using System.Collections.Immutable;
using System.Numerics;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

/// <summary>
/// LpPool: the p-norm of each pooling window, (sum |x|^p)^(1/p), over 1 to 3 spatial axes with the geometry attributes of AveragePool
/// (dilations from opset 18, ceil_mode from 18). Padding does not count. Uses pow, so it is not exact.
/// </summary>
internal sealed class LpPoolOp : WindowPoolKernel
{
    public static readonly OpDef Def = new("LpPool", 1, int.MaxValue, static n => new LpPoolOp(n));

    private readonly float _p;

    private LpPoolOp(NodeInfo node) : base(node, hasDilations: node.Opset >= 18)
    {
        // p is a float attribute in the first version of the operator and an integer from opset 11.
        IrAttr? attr = node.Find("p");
        _p = attr is null ? 2f : attr.Type == IrAttributeType.Real ? attr.Real : attr.Whole;
        if (_p < 1)
        {
            throw node.Error("attribute 'p' must be at least 1");
        }
    }

    public override bool SupportsExact => false;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        if (ctx.InputType(0) != DType.F32)
        {
            throw ctx.Unsupported($"element type {ctx.InputType(0)} is not supported yet (float32 only)");
        }

        _ = InferShape(ctx, out Shape o);
        ctx.SetOutput(0, DType.F32, o);
    }

    public override void Run(in OpRun run)
    {
        Tensor xt = run.Inputs[0], yt = run.Outputs[0];
        if (yt.Length == 0)
        {
            return;
        }

        PoolGeom g = Geometry(xt.Shape, null);
        long planes = (long)xt.Shape[0] * xt.Shape[1];
        long inVol = g.In.Volume, outVol = g.Out.Volume;
        ReadOnlySpan<float> x = xt.Data<float>();
        Span<float> y = yt.MutableData<float>();
        for (long pl = 0; pl < planes; pl++)
        {
            ReadOnlySpan<float> src = x.Slice((int)(pl * inVol), (int)inVol);
            long o = pl * outVol;
            for (int od = 0; od < g.Out.D; od++)
            {
                for (int oh = 0; oh < g.Out.H; oh++)
                {
                    for (int ow = 0; ow < g.Out.W; ow++, o++)
                    {
                        float sum = 0;
                        for (int kd = 0; kd < g.Kernel.D; kd++)
                        {
                            int d = (od * g.Stride.D) - g.Pad.D + (kd * g.Dilation.D);
                            for (int kh = 0; kh < g.Kernel.H; kh++)
                            {
                                int h = (oh * g.Stride.H) - g.Pad.H + (kh * g.Dilation.H);
                                for (int kw = 0; kw < g.Kernel.W; kw++)
                                {
                                    int w = (ow * g.Stride.W) - g.Pad.W + (kw * g.Dilation.W);
                                    if ((uint)d < (uint)g.In.D && (uint)h < (uint)g.In.H && (uint)w < (uint)g.In.W)
                                    {
                                        sum += MathF.Pow(MathF.Abs(src[(((d * g.In.H) + h) * g.In.W) + w]), _p);
                                    }
                                }
                            }
                        }

                        y[(int)o] = MathF.Pow(sum, 1f / _p);
                    }
                }
            }
        }
    }
}

/// <summary>
/// MaxUnpool: the partial inverse of MaxPool. Each element of X is written at the flat position (over the whole output tensor) its
/// index in I names; everything else is 0. The output shape is the optional third input, or derived from the pooling attributes:
/// (in - 1) * stride - pad_begin - pad_end + kernel on every spatial axis.
/// </summary>
internal sealed class MaxUnpoolOp : OpKernel
{
    public static readonly OpDef Def = new("MaxUnpool", 9, int.MaxValue, static n => new MaxUnpoolOp(n));

    private readonly ImmutableArray<long> _kernel, _strides, _pads;

    private MaxUnpoolOp(NodeInfo n)
    {
        _kernel = n.GetInts("kernel_shape");
        _strides = n.GetInts("strides");
        _pads = n.GetInts("pads");
        if (_kernel.IsDefaultOrEmpty)
        {
            throw n.Error("attribute kernel_shape is required");
        }
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 2 || !ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects X and I");
        }

        Shape x = ctx.InputShape(0);
        if (ctx.InputType(0) is not (DType.F32 or DType.F64) || ctx.InputType(1) != DType.I64 || x != ctx.InputShape(1) || x.Rank != _kernel.Length + 2)
        {
            throw ctx.Fail("X must be float with N, C and one axis per kernel dimension, and I an int64 tensor of the same shape");
        }

        Span<long> dims = stackalloc long[Shape.MaxRank];
        if (ctx.HasInput(2))
        {
            if (!ctx.TryGetInputValue(2, out Tensor os))
            {
                ctx.RequireRunTimeValues();
                return;
            }

            IntVec v = IntVec.Of(os);
            if (v.Length != x.Rank)
            {
                throw ctx.Fail("output_shape must have one value per dimension");
            }

            for (int i = 0; i < x.Rank; i++)
            {
                dims[i] = v[i];
            }
        }
        else
        {
            dims[0] = x[0];
            dims[1] = x[1];
            int r = _kernel.Length;
            for (int i = 0; i < r; i++)
            {
                long stride = _strides.IsDefaultOrEmpty ? 1 : _strides[i];
                long pb = _pads.IsDefaultOrEmpty ? 0 : _pads[i], pe = _pads.IsDefaultOrEmpty ? 0 : _pads[i + r];
                dims[2 + i] = ((x[2 + i] - 1) * stride) - pb - pe + _kernel[i];
            }
        }

        ctx.SetOutput(0, ctx.InputType(0), OpUtil.MakeGrownShape(ctx, dims.Slice(0, x.Rank), ctx.InputType(0)));
    }

    public override void Run(in OpRun run)
    {
        Tensor xt = run.Inputs[0], yt = run.Outputs[0];
        ReadOnlySpan<long> idx = run.Inputs[1].Data<long>();

        // The indices are positions in the tensor the pooling attributes imply; a larger output_shape only adds room around it.
        Span<int> natural = stackalloc int[Shape.MaxRank];
        natural[0] = xt.Shape[0];
        natural[1] = xt.Shape[1];
        int r = _kernel.Length;
        for (int i = 0; i < r; i++)
        {
            long stride = _strides.IsDefaultOrEmpty ? 1 : _strides[i];
            long pb = _pads.IsDefaultOrEmpty ? 0 : _pads[i], pe = _pads.IsDefaultOrEmpty ? 0 : _pads[i + r];
            natural[2 + i] = (int)(((xt.Shape[2 + i] - 1) * stride) - pb - pe + _kernel[i]);
        }

        if (xt.DType == DType.F32)
        {
            Scatter(xt.Data<float>(), idx, yt.MutableData<float>(), natural.Slice(0, xt.Rank), yt.Shape.Dimensions);
        }
        else
        {
            Scatter(xt.Data<double>(), idx, yt.MutableData<double>(), natural.Slice(0, xt.Rank), yt.Shape.Dimensions);
        }
    }

    private static void Scatter<T>(ReadOnlySpan<T> x, ReadOnlySpan<long> idx, Span<T> y, ReadOnlySpan<int> natural, ReadOnlySpan<int> outDims)
    {
        y.Clear();
        long naturalTotal = 1;
        foreach (int d in natural)
        {
            naturalTotal *= d;
        }

        for (int i = 0; i < x.Length; i++)
        {
            long k = idx[i];
            if ((ulong)k >= (ulong)naturalTotal)
            {
                throw new SessionException(SessionErrorKind.InvalidInput, $"MaxUnpool: index {k} is outside the {naturalTotal} elements the pooling geometry implies");
            }

            // Unravel in the natural shape and re-ravel in the output shape.
            long at = 0, stride = 1, rest = k;
            for (int d = natural.Length - 1; d >= 0; d--)
            {
                long c = rest % natural[d];
                rest /= natural[d];
                if (c >= outDims[d])
                {
                    throw new SessionException(SessionErrorKind.InvalidInput, "MaxUnpool: an index falls outside output_shape");
                }

                at += c * stride;
                stride *= outDims[d];
            }

            y[(int)at] = x[i];
        }
    }
}

/// <summary>
/// Col2Im (opset 18): the sum-overlapping inverse of im2col. The input is (N, C * prod(block), L) with L the number of block
/// positions; the output is (N, C, image...). A block element lands at position (block position) * stride - pad_begin +
/// (block offset) * dilation and is dropped when it falls outside the image. Defaults: dilations 1, strides 1, pads 0.
/// </summary>
internal sealed class Col2ImOp : OpKernel
{
    public static readonly OpDef Def = new("Col2Im", 18, int.MaxValue, static n => new Col2ImOp(n));

    private readonly ImmutableArray<long> _dilations, _strides, _pads;

    private Col2ImOp(NodeInfo n)
    {
        _dilations = n.GetInts("dilations");
        _strides = n.GetInts("strides");
        _pads = n.GetInts("pads");
    }

    public override bool SupportsExact => true;

    private static long At(ImmutableArray<long> a, int i, long fallback) => a.IsDefaultOrEmpty ? fallback : a[i];

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 3 || !ctx.HasInput(0) || !ctx.HasInput(1) || !ctx.HasInput(2))
        {
            throw ctx.Fail("expects the columns, the image shape and the block shape");
        }

        if (ctx.InputType(0) is not (DType.F32 or DType.F64) || ctx.InputShape(0).Rank != 3)
        {
            throw ctx.Unsupported("only float inputs of shape (N, C * block, L) are supported");
        }

        OpUtil.RequireIntegers(ctx, 1, "image_shape");
        OpUtil.RequireIntegers(ctx, 2, "block_shape");
        if (!ctx.TryGetInputValue(1, out Tensor image) || !ctx.TryGetInputValue(2, out Tensor block))
        {
            ctx.RequireRunTimeValues();
            return;
        }

        IntVec im = IntVec.Of(image), bl = IntVec.Of(block);
        if (im.Length != bl.Length || im.Length + 2 > Shape.MaxRank)
        {
            throw ctx.Fail("image_shape and block_shape must have the same length");
        }

        long blockSize = 1;
        for (int i = 0; i < bl.Length; i++)
        {
            blockSize *= bl[i];
        }

        Shape x = ctx.InputShape(0);
        if (blockSize <= 0 || x[1] % blockSize != 0)
        {
            throw ctx.Fail($"the channel axis {x[1]} is not a multiple of the block size {blockSize}");
        }

        Span<long> dims = stackalloc long[Shape.MaxRank];
        dims[0] = x[0];
        dims[1] = x[1] / blockSize;
        for (int i = 0; i < im.Length; i++)
        {
            dims[2 + i] = im[i];
        }

        ctx.SetOutput(0, ctx.InputType(0), OpUtil.MakeGrownShape(ctx, dims.Slice(0, im.Length + 2), ctx.InputType(0)));
    }

    public override void Run(in OpRun run)
    {
        Tensor xt = run.Inputs[0], yt = run.Outputs[0];
        IntVec block = IntVec.Of(run.Inputs[2]);
        int r = block.Length;
        Span<int> image = stackalloc int[Shape.MaxRank], blk = stackalloc int[Shape.MaxRank], cols = stackalloc int[Shape.MaxRank];
        Span<int> dil = stackalloc int[Shape.MaxRank], str = stackalloc int[Shape.MaxRank], pad = stackalloc int[Shape.MaxRank];
        int blockSize = 1, l = 1, imageSize = 1;
        for (int i = 0; i < r; i++)
        {
            image[i] = yt.Shape[2 + i];
            blk[i] = (int)block[i];
            dil[i] = (int)At(_dilations, i, 1);
            str[i] = (int)At(_strides, i, 1);
            pad[i] = (int)At(_pads, i, 0);
            long padEnd = At(_pads, i + r, 0);
            long span = (long)dil[i] * (blk[i] - 1) + 1;
            long count = ((image[i] + pad[i] + padEnd - span) / str[i]) + 1;
            cols[i] = (int)Math.Max(count, 0);
            blockSize *= blk[i];
            l *= cols[i];
            imageSize *= image[i];
        }

        if (l != xt.Shape[2])
        {
            throw new SessionException(SessionErrorKind.InvalidInput, $"Col2Im: the geometry gives {l} block positions but the input has {xt.Shape[2]}");
        }

        int n = xt.Shape[0], c = yt.Shape[1];
        if (xt.DType == DType.F32)
        {
            Fold(xt.Data<float>(), yt.MutableData<float>(), n, c, r, blockSize, l, imageSize, image, blk, cols, dil, str, pad);
        }
        else
        {
            Fold(xt.Data<double>(), yt.MutableData<double>(), n, c, r, blockSize, l, imageSize, image, blk, cols, dil, str, pad);
        }
    }

    private static void Fold<T>(ReadOnlySpan<T> x, Span<T> y, int n, int c, int r, int blockSize, int l, int imageSize,
        ReadOnlySpan<int> image, ReadOnlySpan<int> blk, ReadOnlySpan<int> cols, ReadOnlySpan<int> dil, ReadOnlySpan<int> str, ReadOnlySpan<int> pad)
        where T : unmanaged, INumber<T>
    {
        y.Clear();
        for (int b = 0; b < n; b++)
        {
            for (int ch = 0; ch < c; ch++)
            {
                for (int k = 0; k < blockSize; k++)
                {
                    for (int p = 0; p < l; p++)
                    {
                        int offset = 0, stride = 1, kr = k, pr = p;
                        bool inside = true;
                        for (int d = r - 1; d >= 0 && inside; d--)
                        {
                            int ko = kr % blk[d], po = pr % cols[d];
                            kr /= blk[d];
                            pr /= cols[d];
                            int pos = (po * str[d]) - pad[d] + (ko * dil[d]);
                            if ((uint)pos >= (uint)image[d])
                            {
                                inside = false;
                                break;
                            }

                            offset += pos * stride;
                            stride *= image[d];
                        }

                        if (inside)
                        {
                            int at = (((b * c) + ch) * imageSize) + offset;
                            y[at] = unchecked(y[at] + x[((((b * c * blockSize) + (ch * blockSize) + k)) * l) + p]);
                        }
                    }
                }
            }
        }
    }
}
