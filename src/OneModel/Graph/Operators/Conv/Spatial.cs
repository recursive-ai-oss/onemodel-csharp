using System;
using System.Collections.Immutable;
using RecursiveAI.OneModel.Graph.Runtime;

namespace RecursiveAI.OneModel.Graph.Operators;

// Shared by Conv, ConvTranspose, the pools and the others that slide a window over trailing "spatial" dimensions: attribute parsing and
// the output-size / padding arithmetic. Everything is done in 64-bit and checked against int before it reaches a kernel.

/// <summary>How a window operator chooses its padding when the model does not give it.</summary>
internal enum AutoPad
{
    /// <summary>Use the <c>pads</c> attribute (zeros if absent).</summary>
    NotSet,

    /// <summary>Total padding makes the output size ceil(input / stride); the odd extra element goes at the end.</summary>
    SameUpper,

    /// <summary>As <see cref="SameUpper"/>, but the extra element goes at the beginning.</summary>
    SameLower,

    /// <summary>No padding.</summary>
    Valid,
}

/// <summary>The window attributes shared by the window operators, parsed once per node and resolved against a shape per plan.</summary>
internal sealed class WindowAttrs
{
    private readonly ImmutableArray<long> _kernel;
    private readonly ImmutableArray<long> _strides;
    private readonly ImmutableArray<long> _pads;
    private readonly ImmutableArray<long> _dilations;

    public WindowAttrs(NodeInfo node, bool hasDilations)
    {
        AutoPad = node.GetString("auto_pad", "NOTSET") switch
        {
            "NOTSET" or "" => AutoPad.NotSet,
            "SAME_UPPER" => AutoPad.SameUpper,
            "SAME_LOWER" => AutoPad.SameLower,
            "VALID" => AutoPad.Valid,
            var other => throw node.Error($"auto_pad '{other}' is not one of NOTSET, SAME_UPPER, SAME_LOWER, VALID"),
        };
        _kernel = node.GetInts("kernel_shape");
        _strides = node.GetInts("strides");
        _pads = node.GetInts("pads");
        _dilations = hasDilations ? node.GetInts("dilations") : [];
        foreach (long v in _strides)
        {
            if (v <= 0 || v > int.MaxValue)
            {
                throw node.Error($"strides must be positive, got {v}");
            }
        }

        foreach (long v in _dilations)
        {
            if (v <= 0 || v > int.MaxValue)
            {
                throw node.Error($"dilations must be positive, got {v}");
            }
        }

        foreach (long v in _kernel)
        {
            if (v <= 0 || v > int.MaxValue)
            {
                throw node.Error($"kernel_shape must be positive, got {v}");
            }
        }

        foreach (long v in _pads)
        {
            if (v < 0 || v > int.MaxValue)
            {
                throw node.Error($"pads must not be negative, got {v}");
            }
        }
    }

    public AutoPad AutoPad { get; }

    public bool HasKernelShape => _kernel.Length > 0;

    /// <summary>Fills the per-axis values for <paramref name="rank"/> spatial axes. The kernel comes from the attribute or, if absent, from <paramref name="weightKernel"/>.</summary>
    public void Resolve(InferContext? ctx, int rank, ReadOnlySpan<int> weightKernel, Span<int> kernel, Span<int> strides,
        Span<int> dilations, Span<int> pads)
    {
        CheckLength(ctx, "kernel_shape", _kernel.Length, rank, allowEmpty: !weightKernel.IsEmpty);
        CheckLength(ctx, "strides", _strides.Length, rank, allowEmpty: true);
        CheckLength(ctx, "dilations", _dilations.Length, rank, allowEmpty: true);
        CheckLength(ctx, "pads", _pads.Length, 2 * rank, allowEmpty: true);
        for (int i = 0; i < rank; i++)
        {
            kernel[i] = _kernel.Length > 0 ? (int)_kernel[i] : weightKernel[i];
            if (_kernel.Length > 0 && !weightKernel.IsEmpty && kernel[i] != weightKernel[i])
            {
                throw Fail(ctx, $"kernel_shape {kernel[i]} on axis {i} does not match the weight's extent {weightKernel[i]}");
            }

            strides[i] = _strides.Length > 0 ? (int)_strides[i] : 1;
            dilations[i] = _dilations.Length > 0 ? (int)_dilations[i] : 1;
        }

        for (int i = 0; i < 2 * rank; i++)
        {
            pads[i] = _pads.Length > 0 ? (int)_pads[i] : 0;
        }
    }

    /// <summary>
    /// Output extent of a pooling or convolution window on one axis, with the padding that gives it. <paramref name="padBegin"/> and
    /// <paramref name="padEnd"/> hold the attribute values on entry (ignored for auto_pad) and the ones used on return.
    /// </summary>
    public int OutputExtent(InferContext? ctx, int axis, int input, int kernel, int stride, int dilation, bool ceilMode, ref int padBegin,
        ref int padEnd)
    {
        long effective = ((long)dilation * (kernel - 1)) + 1;
        if (AutoPad is AutoPad.SameUpper or AutoPad.SameLower)
        {
            long outSame = ((long)input + stride - 1) / stride;
            long total = Math.Max(0, ((outSame - 1) * stride) + effective - input);
            long small = total / 2, big = total - small;
            padBegin = (int)(AutoPad == AutoPad.SameUpper ? small : big);
            padEnd = (int)(AutoPad == AutoPad.SameUpper ? big : small);
            return checked((int)outSame);
        }

        if (AutoPad == AutoPad.Valid)
        {
            padBegin = 0;
            padEnd = 0;
        }

        long span = (long)input + padBegin + padEnd - effective;
        if (span < 0)
        {
            throw Fail(ctx, $"axis {axis}: the window ({effective} wide with dilation) does not fit in the padded input ({(long)input + padBegin + padEnd})");
        }

        long extent = ceilMode ? ((span + stride - 1) / stride) + 1 : (span / stride) + 1;
        // With ceil_mode the last window must start inside the input or its left padding, never entirely in the right padding.
        if (ceilMode && (extent - 1) * stride >= (long)input + padBegin)
        {
            extent--;
        }

        return checked((int)extent);
    }

    private static void CheckLength(InferContext? ctx, string name, int have, int want, bool allowEmpty)
    {
        if (have != want && !(have == 0 && allowEmpty))
        {
            throw Fail(ctx, $"attribute {name} has {have} values; the operator has {want} spatial axes");
        }
    }

    private static SessionException Fail(InferContext? ctx, string message) =>
        ctx is not null ? ctx.Fail(message) : new SessionException(SessionErrorKind.InvalidModel, message);
}
