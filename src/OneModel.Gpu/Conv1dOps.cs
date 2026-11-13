using System;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>
/// ONNX <c>Conv</c> over one spatial axis (<c>[N, C, L]</c>, as the speech encoders' stems use it), run as a two-dimensional convolution of
/// height 1. The window sums are in an order fixed by the shape, so the same kernel serves fast and exact mode (exact mode picks the
/// deterministic matrix product through the backend's kernel mode).
/// </summary>
internal static class Conv1dOps
{
    /// <summary>The node-level check of an adapter-run (float16) one-axis convolution: opset, operand count and <c>auto_pad</c>.</summary>
    public static string? CheckConv(NodeInfo info)
    {
        if (info.Opset < 1)
        {
            return "unknown opset";
        }

        if (info.InputCount is < 2 or > 3)
        {
            return $"Conv with {info.InputCount} inputs";
        }

        return info.GetString("auto_pad", "NOTSET") is "NOTSET" or "VALID" or "SAME_UPPER" or "SAME_LOWER" ? null : "unknown auto_pad";
    }

    public static string? BindConv(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        Shape sx = ctx.InputShape(0);
        if (sx.Rank != 3)
        {
            return $"the GPU Conv handles one spatial axis ([N, C, L]); this input has rank {sx.Rank}";
        }

        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            return "Conv needs an input and weights";
        }

        if (ctx.InputElement(0, out DeviceElement element) is { } why)
        {
            return why;
        }

        bool hasBias = ctx.HasInput(2);
        for (int i = 1; i < (hasBias ? 3 : 2); i++)
        {
            if (ctx.InputType(i) != ctx.InputType(0))
            {
                return $"input {i} is {ctx.InputType(i)} and the input is {ctx.InputType(0)}";
            }
        }

        Shape sw = ctx.InputShape(1);
        if (sw.Rank != 3)
        {
            return "the weights are not rank 3";
        }

        long groups = ctx.Info.GetInt("group", 1);
        int batch = sx[0], channels = sx[1], length = sx[2];
        int outChannels = sw[0], kernel = sw[2];
        if (groups < 1 || channels % groups != 0 || outChannels % groups != 0 || sw[1] != channels / groups)
        {
            return $"{channels} input channels, {outChannels} output channels and weights [{outChannels}, {sw[1]}, {kernel}] do not fit {groups} groups";
        }

        if (ctx.Info.GetInts("kernel_shape") is { Length: > 0 } ks && (ks.Length != 1 || ks[0] != kernel))
        {
            return "kernel_shape does not match the weights";
        }

        if (hasBias && ctx.InputShape(2).Length != outChannels)
        {
            return "the bias does not have one value per output channel";
        }

        int stride = One(ctx, "strides", 1, out string? bad);
        int dilation = bad is null ? One(ctx, "dilations", 1, out bad) : 1;
        if (bad is not null)
        {
            return bad;
        }

        int padBegin = 0, padEnd = 0;
        var pads = ctx.Info.GetInts("pads");
        if (pads.Length != 0)
        {
            if (pads.Length != 2 || pads[0] < 0 || pads[1] < 0 || pads[0] > int.MaxValue || pads[1] > int.MaxValue)
            {
                return "pads are not two non-negative values";
            }

            padBegin = (int)pads[0];
            padEnd = (int)pads[1];
        }

        GpuAutoPad auto = ctx.Info.GetString("auto_pad", "NOTSET") switch
        {
            "VALID" => GpuAutoPad.Valid,
            "SAME_UPPER" => GpuAutoPad.SameUpper,
            "SAME_LOWER" => GpuAutoPad.SameLower,
            _ => GpuAutoPad.NotSet,
        };
        ConvProblem p = ConvProblem.Conv1D(batch, channels, length, outChannels, kernel, stride, dilation, padBegin, padEnd, (int)groups).WithAutoPad(auto);
        int outLength = p.OutWidth;
        Shape so = ctx.OutputShape(0);
        if (outLength < 1 || so.Rank != 3 || so[0] != batch || so[1] != outChannels || so[2] != outLength)
        {
            return "the planned output shape does not match the convolution's";
        }

        if (ctx.OutputType(0) != ctx.InputType(0))
        {
            return $"the result type {ctx.OutputType(0)} differs from the input type";
        }

        foreach (long elements in (ReadOnlySpan<long>)[sx.Length, sw.Length, so.Length])
        {
            if (ctx.FitsBinding(elements, element) is { } big)
            {
                return big;
            }
        }

        binding = new Conv1dBinding(ctx, element, p, hasBias);
        return null;
    }

    private static int One(GpuClaimContext ctx, string name, int fallback, out string? bad)
    {
        bad = null;
        var v = ctx.Info.GetInts(name);
        if (v.Length == 0)
        {
            return fallback;
        }

        if (v.Length != 1 || v[0] < 1 || v[0] > int.MaxValue)
        {
            bad = $"{name} is not one positive value";
            return fallback;
        }

        return (int)v[0];
    }
}

internal sealed class Conv1dBinding : GpuBinding
{
    private readonly ConvKernels _conv;
    private readonly DeviceElement _element;
    private readonly ConvProblem _problem;
    private readonly Operand _x;
    private readonly Operand _w;
    private readonly Operand? _bias;
    private readonly int _outId;

    internal Conv1dBinding(GpuClaimContext ctx, DeviceElement element, ConvProblem problem, bool hasBias)
    {
        _conv = ctx.Backend.Conv;
        _element = element;
        _problem = problem;
        _x = Operand.Of(ctx, 0);
        _w = Operand.Of(ctx, 1);
        _bias = hasBias ? Operand.Of(ctx, 2) : null;
        _outId = ctx.OutputId(0);
        Flops = problem.Flops;
    }

    public override DeviceElement OutputElement(int output) => _element;

    public override void Encode(GpuRunContext ctx)
    {
        ConvProblem p = _problem;
        if (_element == DeviceElement.F32)
        {
            GpuBuffer<float> x = _x.F32(ctx, out int offX);
            GpuBuffer<float> w = _w.F32(ctx, out int offW);
            GpuBuffer<float>? bias = null;
            if (_bias is { } b)
            {
                bias = b.F32(ctx, out int offB);
                p.OffsetBias = offB;
            }

            GpuBuffer<float> y = ctx.F32(_outId, out int offY);
            p.OffsetX = offX;
            p.OffsetW = offW;
            p.OffsetY = offY;
            _conv.Encode(ctx.Encoder, p, x, w, bias, null, y);
        }
        else
        {
            GpuBuffer<Half> x = _x.F16(ctx, out int offX);
            GpuBuffer<Half> w = _w.F16(ctx, out int offW);
            GpuBuffer<Half>? bias = null;
            if (_bias is { } b)
            {
                bias = b.F16(ctx, out int offB);
                p.OffsetBias = offB;
            }

            GpuBuffer<Half> y = ctx.F16(_outId, out int offY);
            p.OffsetX = offX;
            p.OffsetW = offW;
            p.OffsetY = offY;
            _conv.Encode(ctx.Encoder, p, x, w, bias, null, y);
        }
    }
}
