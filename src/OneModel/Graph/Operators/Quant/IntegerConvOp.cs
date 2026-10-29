using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Quant;

/// <summary>State of an integer convolution from <see cref="OpKernel.Prepare"/>: the constant weights packed once per group.</summary>
internal sealed class IntConvState
{
    public Int8Packed[]? Weights;
}

/// <summary>
/// ConvInteger and QLinearConv: 8-bit convolution of X [N, C, spatial...] with W [M, C/group, kernel...] for one or two spatial axes.
/// ConvInteger returns the exact int32 sums of (X - x_zero_point) * (W - w_zero_point); QLinearConv adds the optional int32 bias and
/// requantizes: <c>y = saturate(round_half_even(sum * x_scale * w_scale / y_scale) + y_zero_point)</c>, with w_scale and w_zero_point
/// per tensor or per output channel. Padding takes the input's zero point, so padded taps add nothing. Constant weights are packed
/// once per group when the plan is made.
/// </summary>
internal sealed class IntegerConvOp : OpKernel
{
    public static readonly OpDef ConvInteger = new("ConvInteger", 10, int.MaxValue, static n => new IntegerConvOp(n, false));
    public static readonly OpDef QLinearConv = new("QLinearConv", 10, int.MaxValue, static n => new IntegerConvOp(n, true));

    private readonly bool _q;
    private readonly WindowAttrs _attrs;
    private readonly int _group;

    private IntegerConvOp(NodeInfo node, bool quantized)
    {
        _q = quantized;
        _attrs = new WindowAttrs(node, hasDilations: true);
        long g = node.GetInt("group", 1);
        if (g <= 0 || g > int.MaxValue)
        {
            throw node.Error($"group must be positive, got {g}");
        }

        _group = (int)g;
    }

    public override bool SupportsExact => true;

    private static int XIn => 0;

    private int WIn => _q ? 3 : 1;

    private static int XZeroIn => 2;

    private int WZeroIn => _q ? 5 : 3;

    public override void Infer(InferContext ctx)
    {
        int required = _q ? 8 : 2;
        for (int i = 0; i < required; i++)
        {
            if (!ctx.HasInput(i))
            {
                throw ctx.Fail($"input {i} is required");
            }
        }

        _ = QuantCommon.IsU8(ctx, ctx.InputType(XIn), "X");
        _ = QuantCommon.IsU8(ctx, ctx.InputType(WIn), "W");
        Shape x = ctx.InputShape(XIn), w = ctx.InputShape(WIn);
        if (x.Rank < 3 || x.Rank != w.Rank)
        {
            throw ctx.Fail($"X {x} and W {w} must have the same rank, at least 3");
        }

        if (x.Rank > 4)
        {
            throw ctx.Unsupported($"{x.Rank - 2} spatial axes are not supported yet (1 or 2)");
        }

        if (w[1] * _group != x[1] || w[0] % _group != 0)
        {
            throw ctx.Fail($"W {w} does not fit X {x} with group {_group}");
        }

        if (ctx.HasInput(XZeroIn))
        {
            if (ctx.InputType(XZeroIn) != ctx.InputType(XIn) || ctx.InputShape(XZeroIn).Length != 1)
            {
                throw ctx.Fail("the input zero point must be a scalar of X's element type");
            }
        }

        if (ctx.HasInput(WZeroIn))
        {
            if (ctx.InputType(WZeroIn) != ctx.InputType(WIn))
            {
                throw ctx.Fail("the weight zero point must have W's element type");
            }

            _ = QuantCommon.IsPerChannel(ctx, WZeroIn, w[0], "the weight zero point");
        }

        DType outType = DType.I32;
        if (_q)
        {
            if (ctx.InputType(1) != DType.F32 || ctx.InputType(4) != DType.F32 || ctx.InputType(6) != DType.F32)
            {
                throw ctx.Unsupported("scales must be float32");
            }

            if (ctx.InputShape(1).Length != 1 || ctx.InputShape(6).Length != 1)
            {
                throw ctx.Fail("x_scale and y_scale must be scalars");
            }

            _ = QuantCommon.IsPerChannel(ctx, 4, w[0], "w_scale");
            outType = ctx.InputType(7);
            _ = QuantCommon.IsU8(ctx, outType, "y_zero_point");
            if (ctx.HasInput(8) && (ctx.InputType(8) != DType.I32 || ctx.InputShape(8).Rank != 1 || ctx.InputShape(8)[0] != w[0]))
            {
                throw ctx.Fail($"the bias must be int32 with {w[0]} values");
            }
        }

        Geometry(x, w, ctx, out ConvShape s);
        ctx.SetOutput(0, outType, x.Rank == 3 ? Shape.Create(s.N, s.Cout, s.OutW) : Shape.Create(s.N, s.Cout, s.OutH, s.OutW));
    }

    private void Geometry(Shape x, Shape w, InferContext? ctx, out ConvShape s)
    {
        int r = x.Rank - 2;
        Span<int> kernel = stackalloc int[2], strides = stackalloc int[2], dil = stackalloc int[2], pads = stackalloc int[4];
        _attrs.Resolve(ctx, r, w.Dimensions.Slice(2), kernel, strides, dil, pads);
        int h = r == 2 ? x[2] : 1, wd = x[x.Rank - 1];
        int padT = 0, padB = 0, padL, padR;
        if (r == 2)
        {
            padT = pads[0];
            padB = pads[2];
            _ = _attrs.OutputExtent(ctx, 0, h, kernel[0], strides[0], dil[0], false, ref padT, ref padB);
        }

        padL = pads[r - 1];
        padR = pads[(2 * r) - 1];
        _ = _attrs.OutputExtent(ctx, r - 1, wd, kernel[r - 1], strides[r - 1], dil[r - 1], false, ref padL, ref padR);
        s = new ConvShape(
            x[0], x[1], h, wd, w[0], r == 2 ? kernel[0] : 1, kernel[r - 1],
            r == 2 ? strides[0] : 1, strides[r - 1], padT, padL, padB, padR, r == 2 ? dil[0] : 1, dil[r - 1], _group);
    }

    public override object? Prepare(PrepareContext ctx)
    {
        if (!ctx.IsConstant(WIn))
        {
            return null;
        }

        Tensor w = ctx.Constant(WIn);
        int coutG = w.Shape[0] / _group;
        if (w.Shape[1] == 1 && coutG == 1 && w.Length / w.Shape[0] <= 64)
        {
            return null; // depthwise: the direct kernel reads the raw weights
        }

        int k = w.Length / w.Shape[0];
        Int8Format f = Int8Format.For(ctx.Path);
        ReadOnlySpan<byte> raw = QuantParams.Bytes(in w);
        var packs = new Int8Packed[_group];
        for (int g = 0; g < _group; g++)
        {
            packs[g] = Int8Packed.ForA(in f, raw.Slice(g * coutG * k, coutG * k), k, w.DType == DType.U8, coutG, k);
        }

        return new IntConvState { Weights = packs };
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[XIn], w = run.Inputs[WIn], y = run.Outputs[0];
        if (y.Length == 0)
        {
            return;
        }

        Geometry(x.Shape, w.Shape, null, out ConvShape s);
        int cout = s.Cout;
        Tensor xz = QuantParams.Optional(in run, XZeroIn), wz = QuantParams.Optional(in run, WZeroIn);
        bool xU8 = x.DType == DType.U8, wU8 = w.DType == DType.U8;
        int wZeroScalar = QuantCommon.Shifted(in wz, 0, wU8);
        ReadOnlySpan<int> wZeroVec = default;
        if (!wz.IsEmpty && wz.Length > 1)
        {
            Span<int> v = run.Scratch(DType.I32, Shape.Create(cout)).MutableData<int>();
            for (int i = 0; i < cout; i++)
            {
                v[i] = QuantCommon.Shifted(in wz, i, wU8);
            }

            wZeroVec = v;
        }

        var call = new Int8ConvCall
        {
            Format = Int8Format.For(run.Path),
            S = s,
            X = QuantParams.Bytes(in x),
            XU8 = xU8,
            XZeroRaw = QuantCommon.Raw(in xz),
            XZeroShifted = QuantCommon.Shifted(in xz, 0, xU8),
            WU8 = wU8,
            WZeroScalar = wZeroScalar,
            WZeroVec = wZeroVec,
            Threads = run.Threads,
        };
        if (run.State is IntConvState { Weights: { } packs })
        {
            call.WPacked = packs;
        }
        else
        {
            call.WRaw = QuantParams.Bytes(in w);
        }

        if (_q)
        {
            float sx = run.Inputs[1].Data<float>()[0];
            float sy = run.Inputs[6].Data<float>()[0];
            ReadOnlySpan<float> sw = run.Inputs[4].Data<float>();
            if (sw.Length == 1)
            {
                call.Mult = sx * sw[0] / sy;
            }
            else
            {
                Span<float> mv = run.Scratch(DType.F32, Shape.Create(cout)).MutableData<float>();
                for (int i = 0; i < cout; i++)
                {
                    mv[i] = sx * sw[i] / sy;
                }

                call.MultVec = mv;
            }

            call.QuantUnsigned = y.DType == DType.U8;
            call.QuantZero = QuantCommon.Raw(run.Inputs[7]);
            if (run.Inputs.Length > 8 && !run.Inputs[8].IsEmpty)
            {
                call.Bias = run.Inputs[8].Data<int>();
            }

            call.OutQuant = QuantParams.MutableBytes(in y);
        }
        else
        {
            call.OutInt32 = y.MutableData<int>();
        }

        Int8Conv.Run(ref call);
    }
}
