using System;
using System.Collections.Generic;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>
/// The entries that move data without arithmetic: Transpose, Cast between float32 and float16, and the operators that only reinterpret
/// a tensor's shape (Reshape, Flatten, Squeeze, Unsqueeze, Identity, Dropout in inference). The reinterpreting ones cost nothing on the
/// device: the result is the same memory under another shape. All are bit-exact, so one kernel serves both modes.
/// </summary>
internal static class LayoutOps
{
    public static IEnumerable<GpuOpEntry> Entries
    {
        get
        {
            yield return new GpuOpEntry("Transpose", true, _ => null, BindTranspose);
            yield return new GpuOpEntry("Cast", true, _ => null, BindCast);
            foreach (string name in new[] { "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Identity", "Dropout" })
            {
                yield return new GpuOpEntry(name, true, _ => null, BindView);
            }
        }
    }

    // ---- views -------------------------------------------------------------------------------------------------------------

    private static string? BindView(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.Step.IsView)
        {
            return "this form of the operator copies its input, and the GPU has no kernel for the copy";
        }

        int source = ctx.Step.Node.Kernel.ViewOfInput;
        if (ctx.InputAnyElement(source, out DeviceElement element) is { } why)
        {
            return why;
        }

        if (ctx.InputIsConst(source))
        {
            return "its input is a constant";
        }

        if (ctx.InputShape(source).Length != ctx.OutputShape(0).Length || ctx.OutputType(0) != ctx.InputType(source))
        {
            return "the planned output does not match the input";
        }

        binding = new ViewBinding(source, element);
        return null;
    }

    private sealed class ViewBinding : GpuBinding
    {
        private readonly int _source;
        private readonly DeviceElement _element;

        public ViewBinding(int source, DeviceElement element)
        {
            _source = source;
            _element = element;
        }

        public override int AliasOfInput => _source;

        public override DeviceElement OutputElement(int output) => _element;

        public override void Encode(GpuRunContext ctx)
        {
        }
    }

    // ---- Transpose ---------------------------------------------------------------------------------------------------------

    private static string? BindTranspose(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0))
        {
            return "needs an input";
        }

        if (ctx.InputElement(0, out DeviceElement element) is { } why)
        {
            return why;
        }

        Shape x = ctx.InputShape(0);
        if (x.Rank == 0 || x.Length == 0)
        {
            return "a scalar or empty input";
        }

        int[] perm = new int[x.Rank];
        System.Collections.Immutable.ImmutableArray<long> attr = ctx.Info.GetInts("perm");
        if (attr.IsDefaultOrEmpty)
        {
            for (int i = 0; i < perm.Length; i++)
            {
                perm[i] = perm.Length - 1 - i;
            }
        }
        else if (attr.Length != x.Rank)
        {
            return "perm does not have one entry per dimension";
        }
        else
        {
            for (int i = 0; i < perm.Length; i++)
            {
                perm[i] = (int)attr[i];
            }
        }

        if (ctx.OutputShape(0).Length != x.Length || ctx.OutputType(0) != ctx.InputType(0))
        {
            return "the planned output does not match the input";
        }

        if (ctx.FitsBinding(x.Length, element) is { } big)
        {
            return big;
        }

        binding = new TransposeBinding(ctx, element, GpuClaimContext.Dims(x), perm);
        return null;
    }

    private sealed class TransposeBinding : GpuBinding
    {
        private readonly TransposeKernels _kernels;
        private readonly DeviceElement _element;
        private readonly int[] _shape, _perm;
        private readonly Operand _x;
        private readonly int _outId;

        public TransposeBinding(GpuClaimContext ctx, DeviceElement element, int[] shape, int[] perm)
        {
            _kernels = ctx.Backend.Transpose;
            _element = element;
            _shape = shape;
            _perm = perm;
            _x = Operand.Of(ctx, 0);
            _outId = ctx.OutputId(0);
        }

        public override DeviceElement OutputElement(int output) => _element;

        public override void Encode(GpuRunContext ctx)
        {
            if (_element == DeviceElement.F32)
            {
                Encode<float>(ctx);
            }
            else
            {
                Encode<Half>(ctx);
            }
        }

        private void Encode<T>(GpuRunContext ctx) where T : unmanaged
        {
            GpuBuffer<T> x = _x.Get<T>(ctx, out int offX);
            GpuBuffer<T> y = ctx.Buffer<T>(_outId, out int offY);
            _kernels.Encode(ctx.Encoder, x, offX, _shape, _perm, y, offY);
        }
    }

    // ---- Cast --------------------------------------------------------------------------------------------------------------

    private static string? BindCast(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0))
        {
            return "needs an input";
        }

        if (ctx.InputAnyElement(0, out DeviceElement from) is { } fromWhy)
        {
            return fromWhy;
        }

        if (ctx.OutputAnyElement(0, out DeviceElement to) is { } toWhy)
        {
            return toWhy;
        }

        if (ctx.InputShape(0).Length == 0 || ctx.OutputShape(0).Length != ctx.InputShape(0).Length)
        {
            return "an empty input, or a planned output that does not match it";
        }

        if ((ctx.FitsBinding(ctx.InputShape(0).Length, from) ?? ctx.FitsBinding(ctx.InputShape(0).Length, to)) is { } big)
        {
            return big;
        }

        binding = new CastBinding(ctx, from, to);
        return null;
    }

    // Conversions between float32, float16, int32, int64 and bool. The kernel rounds, truncates, saturates and sign-extends as the CPU's
    // cast does, and is bit-exact against it.
    private sealed class CastBinding : GpuBinding
    {
        private readonly ElementwiseKernels _kernels;
        private readonly DeviceElement _from, _to;
        private readonly Operand _x;
        private readonly int _outId, _count;

        public CastBinding(GpuClaimContext ctx, DeviceElement from, DeviceElement to)
        {
            _kernels = ctx.Backend.Elementwise;
            _from = from;
            _to = to;
            _x = Operand.Of(ctx, 0);
            _outId = ctx.OutputId(0);
            _count = ctx.InputShape(0).Length;
        }

        public override DeviceElement OutputElement(int output) => _to;

        public override void Encode(GpuRunContext ctx)
        {
            switch (_from)
            {
                case DeviceElement.F32:
                    From<float>(ctx);
                    break;
                case DeviceElement.F16:
                    From<Half>(ctx);
                    break;
                case DeviceElement.I32:
                    From<int>(ctx);
                    break;
                case DeviceElement.I64:
                    From<long>(ctx);
                    break;
                default:
                    From<byte>(ctx);
                    break;
            }
        }

        private void From<TFrom>(GpuRunContext ctx) where TFrom : unmanaged
        {
            GpuBuffer<TFrom> x = _x.Get<TFrom>(ctx, out int offX);
            switch (_to)
            {
                case DeviceElement.F32:
                    _kernels.EncodeCast(ctx.Encoder, x, offX, ctx.Buffer<float>(_outId, out int o1), o1, _count);
                    break;
                case DeviceElement.F16:
                    _kernels.EncodeCast(ctx.Encoder, x, offX, ctx.Buffer<Half>(_outId, out int o2), o2, _count);
                    break;
                case DeviceElement.I32:
                    _kernels.EncodeCast(ctx.Encoder, x, offX, ctx.Buffer<int>(_outId, out int o3), o3, _count);
                    break;
                case DeviceElement.I64:
                    _kernels.EncodeCast(ctx.Encoder, x, offX, ctx.Buffer<long>(_outId, out int o4), o4, _count);
                    break;
                default:
                    _kernels.EncodeCast(ctx.Encoder, x, offX, ctx.Buffer<byte>(_outId, out int o5), o5, _count);
                    break;
            }
        }
    }
}
