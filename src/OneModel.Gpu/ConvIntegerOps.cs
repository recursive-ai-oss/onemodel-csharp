using System;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>
/// ONNX <c>ConvInteger</c>: the 8-bit convolution of X [N, C, spatial...] with W [M, C/group, kernel...], one or two spatial axes, returning
/// the exact int32 sums of (X - x_zero_point) * (W - w_zero_point). The sums are integers, so the result is the CPU's bits on every device and
/// in both modes. The device lowers it to the 8-bit matrix product through a column matrix; that matrix lives in the arena for the length of
/// the step, so the memory is used only while the convolution runs.
/// </summary>
internal static class ConvIntegerOps
{
    // The column matrix of the whole batch up to this many bytes; a bigger batch runs in chunks of images, bits unchanged.
    private const long ScratchBudgetBytes = 64L << 20;

    public static string? Check(NodeInfo info) =>
        info.GetString("auto_pad", "NOTSET") is "NOTSET" or "VALID" or "SAME_UPPER" or "SAME_LOWER" ? null : "unknown auto_pad";

    public static string? Bind(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            return "ConvInteger needs an input and weights";
        }

        DType xType = ctx.InputType(0), wType = ctx.InputType(1);
        if (xType is not (DType.U8 or DType.I8) || wType is not (DType.U8 or DType.I8))
        {
            return $"X is {xType} and W is {wType}; the GPU kernel takes uint8 and int8";
        }

        if (ctx.OutputType(0) != DType.I32)
        {
            return $"the result is {ctx.OutputType(0)}, and the GPU kernel makes int32";
        }

        Shape x = ctx.InputShape(0), wt = ctx.InputShape(1);
        if (x.Rank is not (3 or 4) || wt.Rank != x.Rank)
        {
            return $"X has rank {x.Rank} and W rank {wt.Rank}; the GPU kernel takes one or two spatial axes";
        }

        long groups = ctx.Info.GetInt("group", 1);
        if (groups < 1 || groups > int.MaxValue || wt[1] * groups != x[1] || wt[0] % groups != 0)
        {
            return $"W {wt} does not fit X {x} with group {groups}";
        }

        ReadOnlySpan<int> weightKernel = x.Rank == 4 ? [wt[2], wt[3]] : [wt[2]];
        if (VisionOps.ResolveWindow(ctx.Info, x, weightKernel, false, out VisionOps.Window win) is { } bad)
        {
            return bad;
        }

        if (!VisionOps.OutputMatches(ctx, x[0], wt[0], x.Rank, win))
        {
            return "the output shape the plan holds is not the one the window rules give";
        }

        bool xZero = ctx.HasInput(2), wZero = ctx.HasInput(3);
        if (xZero)
        {
            if (ctx.InputType(2) != xType || ctx.InputShape(2).Length != 1)
            {
                return "the input zero point is not a scalar of X's type";
            }

            if (ctx.InputMadeOnDevice(2) is { } e && e != ElementOf(xType))
            {
                return "the input zero point was made on the device as another type";
            }
        }

        bool wZeroVector = false;
        if (wZero)
        {
            if (ctx.InputType(3) != wType || (ctx.InputShape(3).Length != 1 && ctx.InputShape(3).Length != wt[0]))
            {
                return "the weight zero point is not of W's type with one value or one per output channel";
            }

            if (ctx.InputMadeOnDevice(3) is { } e && e != ElementOf(wType))
            {
                return "the weight zero point was made on the device as another type";
            }

            wZeroVector = ctx.InputShape(3).Length > 1;
        }

        foreach ((long elements, DeviceElement element) in new[] { (x.Length, ElementOf(xType)), (wt.Length, ElementOf(wType)), (ctx.OutputShape(0).Length, DeviceElement.I32) })
        {
            if (ctx.FitsBinding(elements, element) is { } big)
            {
                return big;
            }
        }

        var p = new ConvProblem(x[0], x[1], x.Rank == 4 ? x[2] : 1, x[x.Rank - 1], wt[0], x.Rank == 4 ? wt[2] : 1, wt[wt.Rank - 1])
        {
            StrideHeight = win.StrideH, StrideWidth = win.StrideW, DilationHeight = win.DilationH, DilationWidth = win.DilationW,
            PadTop = win.PadT, PadLeft = win.PadL, PadBottom = win.PadB, PadRight = win.PadR, Groups = (int)groups,
        };
        long perImage;
        try
        {
            perImage = ConvIntegerKernels.ScratchBytesPerImage(p);
        }
        catch (GpuKernelShapeException e)
        {
            return e.Message;
        }

        if (ctx.FitsBinding(perImage, ElementOf(xType)) is { } tooBig)
        {
            return tooBig;
        }

        long fit = Math.Min(ScratchBudgetBytes, ctx.MaxBindingBytes) / Math.Max(1, perImage);
        long images = Math.Max(1, Math.Min(p.Batch, fit));
        binding = new ConvIntegerBinding(ctx, p, images * perImage, wZeroVector, xType == DType.I8, wType == DType.I8);
        return null;
    }

    private static DeviceElement ElementOf(DType t) => t == DType.I8 ? DeviceElement.I8 : DeviceElement.U8;

    private sealed class ConvIntegerBinding : GpuBinding
    {
        private readonly ConvIntegerKernels _kernels;
        private readonly ConvProblem _problem;
        private readonly long _scratch;
        private readonly bool _wZeroVector, _hasXZero, _hasWZero, _xSigned, _wSigned;
        private readonly Operand _x, _w, _xZero, _wZero;
        private readonly int _outId;

        public ConvIntegerBinding(GpuClaimContext ctx, ConvProblem problem, long scratchBytes, bool wZeroVector, bool xSigned, bool wSigned)
        {
            _kernels = ctx.Backend.ConvInteger;
            _problem = problem;
            _scratch = scratchBytes;
            _wZeroVector = wZeroVector;
            _xSigned = xSigned;
            _wSigned = wSigned;
            _x = Operand.Of(ctx, 0);
            _w = Operand.Of(ctx, 1);
            _hasXZero = ctx.HasInput(2);
            _hasWZero = ctx.HasInput(3);
            if (_hasXZero)
            {
                _xZero = Operand.Of(ctx, 2);
            }

            if (_hasWZero)
            {
                _wZero = Operand.Of(ctx, 3);
            }

            _outId = ctx.OutputId(0);
            Flops = 2.0 * problem.Batch * problem.OutChannels * problem.OutHeight * problem.OutWidth * (problem.InChannels / problem.Groups) * problem.KernelHeight * problem.KernelWidth;
        }

        public override DeviceElement OutputElement(int output) => DeviceElement.I32;

        public override long ScratchBoolElements => _xSigned ? 0 : _scratch;

        public override long ScratchI8Elements => _xSigned ? _scratch : 0;

        public override void Encode(GpuRunContext ctx)
        {
            if (_xSigned)
            {
                if (_wSigned)
                {
                    Encode<sbyte, sbyte>(ctx);
                }
                else
                {
                    Encode<sbyte, byte>(ctx);
                }
            }
            else if (_wSigned)
            {
                Encode<byte, sbyte>(ctx);
            }
            else
            {
                Encode<byte, byte>(ctx);
            }
        }

        private void Encode<TX, TW>(GpuRunContext ctx) where TX : unmanaged where TW : unmanaged
        {
            GpuBuffer<TX> x = _x.Get<TX>(ctx, out int offX);
            GpuBuffer<TW> w = _w.Get<TW>(ctx, out int offW);
            GpuBuffer<int> y = ctx.Buffer<int>(_outId, out int offY);
            GpuBuffer<TX> scratch;
            int offScratch;
            if (typeof(TX) == typeof(sbyte))
            {
                scratch = (GpuBuffer<TX>)(object)ctx.ScratchI8(this, out offScratch);
            }
            else
            {
                scratch = (GpuBuffer<TX>)(object)ctx.ScratchBool(this, out offScratch);
            }

            ConvProblem p = _problem;
            p.OffsetX = offX;
            p.OffsetW = offW;
            p.OffsetY = offY;
            Int8Operand<TX>? xZero = null;
            Int8Operand<TW>? wZero = null;
            if (_hasXZero)
            {
                xZero = new Int8Operand<TX>(_xZero.Get<TX>(ctx, out int o), o);
            }

            if (_hasWZero)
            {
                wZero = new Int8Operand<TW>(_wZero.Get<TW>(ctx, out int o), o, _wZeroVector);
            }

            ctx.Encoder.ReportWork(Flops);
            _kernels.Encode(ctx.Encoder, p, x, w, y, xZero, wZero, scratch, offScratch);
        }
    }
}
