using System;
using System.Collections.Immutable;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

/// <summary>
/// ConvTranspose: the gradient-of-convolution upsampler, for one or two spatial axes. X [N, C, spatial...], W [C, M/group, kernel...],
/// optional bias [M]. The output size comes from <c>output_shape</c> if given, else from strides, dilations, pads and
/// <c>output_padding</c>. Three spatial axes are not supported yet.
/// </summary>
internal sealed class ConvTransposeOp : OpKernel
{
    public static readonly OpDef Def = new("ConvTranspose", 1, int.MaxValue, static n => new ConvTransposeOp(n));

    private readonly WindowAttrs _attrs;
    private readonly int _group;
    private readonly ImmutableArray<long> _outputPadding;
    private readonly ImmutableArray<long> _outputShape;

    private ConvTransposeOp(NodeInfo node)
    {
        _attrs = new WindowAttrs(node, hasDilations: true);
        long g = node.GetInt("group", 1);
        if (g <= 0 || g > int.MaxValue)
        {
            throw node.Error($"group must be positive, got {g}");
        }

        _group = (int)g;
        _outputPadding = node.GetInts("output_padding");
        _outputShape = node.GetInts("output_shape");
        foreach (long v in _outputPadding)
        {
            if (v < 0 || v > int.MaxValue)
            {
                throw node.Error($"output_padding must not be negative, got {v}");
            }
        }

        foreach (long v in _outputShape)
        {
            if (v <= 0 || v > int.MaxValue)
            {
                throw node.Error($"output_shape must be positive, got {v}");
            }
        }
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects inputs X and W");
        }

        if (ctx.InputType(0) != DType.F32 || ctx.InputType(1) != DType.F32 || (ctx.HasInput(2) && ctx.InputType(2) != DType.F32))
        {
            throw ctx.Unsupported("only float32 is supported yet");
        }

        Shape x = ctx.InputShape(0), w = ctx.InputShape(1);
        if (x.Rank < 3 || x.Rank != w.Rank)
        {
            throw ctx.Fail($"X {x} and W {w} must have the same rank, at least 3");
        }

        if (x.Rank > 4)
        {
            throw ctx.Unsupported($"{x.Rank - 2} spatial axes are not supported yet (1 or 2)");
        }

        if (w[0] != x[1] || x[1] % _group != 0)
        {
            throw ctx.Fail($"W {w} does not fit X {x} with group {_group}");
        }

        if (ctx.HasInput(2) && (ctx.InputShape(2).Rank != 1 || ctx.InputShape(2)[0] != w[1] * _group))
        {
            throw ctx.Fail($"the bias {ctx.InputShape(2)} must be 1-D with {w[1] * _group} values");
        }

        Geometry(x, w, ctx, out ConvTransposeShape s);
        ctx.SetOutput(0, DType.F32, x.Rank == 3 ? Shape.Create(s.N, s.Cout, s.OutW) : Shape.Create(s.N, s.Cout, s.OutH, s.OutW));
    }

    private void Geometry(Shape x, Shape w, InferContext? ctx, out ConvTransposeShape s)
    {
        int r = x.Rank - 2;
        Span<int> kernel = stackalloc int[2], strides = stackalloc int[2], dil = stackalloc int[2], pads = stackalloc int[4];
        Span<int> outDims = stackalloc int[2], padB = stackalloc int[2];
        _attrs.Resolve(ctx, r, w.Dimensions.Slice(2), kernel, strides, dil, pads);
        if (_outputPadding.Length != 0 && _outputPadding.Length != r)
        {
            throw Fail(ctx, "output_padding has the wrong number of values");
        }

        if (_outputShape.Length != 0 && _outputShape.Length != r)
        {
            throw Fail(ctx, "output_shape has the wrong number of values");
        }

        for (int i = 0; i < r; i++)
        {
            int input = x[2 + i];
            long outPad = _outputPadding.Length > 0 ? _outputPadding[i] : 0;
            long effective = ((long)dil[i] * (kernel[i] - 1)) + 1;
            long natural = ((long)strides[i] * (input - 1)) + outPad + effective;
            long total;
            if (_outputShape.Length > 0)
            {
                total = natural - _outputShape[i];
                if (total < 0)
                {
                    throw Fail(ctx, $"axis {i}: the output size {_outputShape[i]} is larger than the kernel can produce ({natural})");
                }

                outDims[i] = (int)_outputShape[i];
                padB[i] = Begin(total);
            }
            else if (_attrs.AutoPad is AutoPad.SameUpper or AutoPad.SameLower)
            {
                // The output is input * stride when the kernel reaches that far, and otherwise as big as it gets (no padding).
                total = Math.Max(0, natural - ((long)input * strides[i]));
                outDims[i] = (int)(natural - total);
                padB[i] = Begin(total);
            }
            else
            {
                long b = _attrs.AutoPad == AutoPad.Valid ? 0 : pads[i];
                long e = _attrs.AutoPad == AutoPad.Valid ? 0 : pads[i + r];
                long extent = natural - b - e;
                if (extent <= 0 || extent > int.MaxValue)
                {
                    throw Fail(ctx, $"axis {i}: the output size {extent} is not positive");
                }

                padB[i] = (int)b;
                outDims[i] = (int)extent;
            }
        }

        int xw = r - 1;
        s = new ConvTransposeShape(
            x[0], x[1], r == 2 ? x[2] : 1, x[x.Rank - 1], w[1] * _group, r == 2 ? kernel[0] : 1, kernel[xw],
            r == 2 ? strides[0] : 1, strides[xw], r == 2 ? padB[0] : 0, padB[xw], r == 2 ? dil[0] : 1, dil[xw], _group,
            r == 2 ? outDims[0] : 1, outDims[xw]);
    }

    // Where the total padding of an axis goes: SAME_UPPER puts the extra element at the end; everything else (SAME_LOWER, and an
    // explicit output_shape) at the beginning.
    private int Begin(long total) => (int)(_attrs.AutoPad == AutoPad.SameUpper ? total / 2 : total - (total / 2));

    private static SessionException Fail(InferContext? ctx, string message) =>
        ctx is not null ? ctx.Fail(message) : new SessionException(SessionErrorKind.InvalidModel, message);

    // A constant weight is rearranged for the matrix product once per plan.
    public override object? Prepare(PrepareContext ctx)
    {
        if (!ctx.IsConstant(1))
        {
            return null;
        }

        Tensor w = ctx.Constant(1);
        Tensor t = ctx.CreateOwned(DType.F32, Shape.Create(w.Length));
        // Geometry needs only the shapes; the input shape is not known here, so rearrange with the weight's own extents.
        int taps = w.Length / w.Shape[0];
        TransposeAll(w.Data<float>(), t.MutableData<float>(), w.Shape[0], taps);
        return t;
    }

    // Rearranges W [Cin, taps] into [group][taps][Cin/group] for every group at once: group g's rows are contiguous in W.
    private void TransposeAll(ReadOnlySpan<float> w, Span<float> wt, int cin, int taps)
    {
        int cinG = cin / _group;
        for (int g = 0; g < _group; g++)
        {
            for (int ci = 0; ci < cinG; ci++)
            {
                for (int t = 0; t < taps; t++)
                {
                    wt[(((g * taps) + t) * cinG) + ci] = w[(((g * cinG) + ci) * taps) + t];
                }
            }
        }
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], w = run.Inputs[1], y = run.Outputs[0];
        if (y.Length == 0)
        {
            return;
        }

        Geometry(x.Shape, w.Shape, null, out ConvTransposeShape s);
        ReadOnlySpan<float> wt;
        if (run.State is Tensor prepared)
        {
            wt = prepared.Data<float>();
        }
        else
        {
            Tensor tmp = run.Scratch(DType.F32, Shape.Create(w.Length));
            TransposeAll(w.Data<float>(), tmp.MutableData<float>(), w.Shape[0], w.Length / w.Shape[0]);
            wt = tmp.Data<float>();
        }

        int taps = (s.Cout / s.Groups) * s.KH * s.KW;
        Tensor cols = run.Scratch(DType.F32, Shape.Create(taps, s.H * s.W));
        ReadOnlySpan<float> bias = run.Inputs.Length > 2 && !run.Inputs[2].IsEmpty ? run.Inputs[2].Data<float>() : default;
        ConvTransposeKernel.Run(run.KernelMode, run.Path, in s, x.Data<float>(), wt, bias, y.MutableData<float>(), cols.MutableData<float>(), run.Threads);
    }
}
