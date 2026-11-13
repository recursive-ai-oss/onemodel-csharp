using System;
using System.Collections.Generic;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>
/// ONNX <c>DequantizeLinear</c> on the device: <c>y = (float)(x - zero_point) * scale</c>, per tensor or per axis, float32 result, the
/// arithmetic of onemodel's CPU kernel bit for bit. It dequantizes the embedding rows that a <c>Gather</c> of an 8-bit table made, so a
/// dynamically quantized encoder keeps its whole body on the device. A single scale value is per tensor, more than one is per axis, as on
/// the CPU. There is no reduction, so fast and exact mode give the same bits.
/// </summary>
internal static class DequantizeOps
{
    public static IEnumerable<GpuOpEntry> Entries
    {
        get
        {
            yield return new GpuOpEntry("DequantizeLinear", true, Check, Bind);
            yield return new GpuOpEntry("QuantizeLinear", true, CheckQuantize, BindQuantize);
        }
    }

    private static string? Check(NodeInfo info) =>
        info.GetInt("block_size", 0) != 0 ? "blocked quantization (block_size) is not supported on the GPU"
        : info.Has("output_dtype") ? "the output_dtype attribute is not supported on the GPU"
        : null;

    private static string? Bind(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            return "needs x and x_scale";
        }

        DType xt = ctx.InputType(0);
        if (xt is not (DType.U8 or DType.I8 or DType.I32))
        {
            return $"the input is {xt}, and the GPU kernel takes uint8, int8 and int32";
        }

        if (ctx.InputType(1) != DType.F32)
        {
            return $"the scale is {ctx.InputType(1)}, and the GPU kernel takes float32";
        }

        bool hasZero = ctx.HasInput(2);
        if (hasZero && ctx.InputType(2) != xt)
        {
            return "the zero point's type differs from the input's";
        }

        if (ctx.OutputType(0) != DType.F32)
        {
            return "the planned output is not float32";
        }

        Shape sx = ctx.InputShape(0), ss = ctx.InputShape(1);
        if (sx.Length == 0 || sx.Rank > 8)
        {
            return "an empty tensor or one with more than eight dimensions";
        }

        bool perAxis = ss.Length > 1;
        long axisAttr = ctx.Info.GetInt("axis", 1);
        int axis = 0;
        if (perAxis)
        {
            axis = (int)(axisAttr < 0 ? axisAttr + sx.Rank : axisAttr);
            if (axis < 0 || axis >= sx.Rank || ss.Length != sx[axis])
            {
                return $"the scale has {ss.Length} values for axis {axisAttr} of {sx}";
            }
        }

        if (hasZero && ctx.InputShape(2).Length != (perAxis ? ss.Length : 1))
        {
            return "the zero point does not match the scale";
        }

        if (ctx.FitsBinding(sx.Length, DeviceElement.F32) is { } big)
        {
            return big;
        }

        if (xt != DType.I32 && ctx.FitsBinding(sx.Length, DeviceElement.U8) is { } bigBytes)
        {
            return bigBytes;
        }

        if (ctx.InputMadeOnDevice(0) is { } made && made != (xt == DType.U8 ? DeviceElement.U8 : xt == DType.I8 ? DeviceElement.I8 : DeviceElement.I32))
        {
            return "the input was made on the device as another type";
        }

        binding = new DequantizeBinding(ctx, xt, GpuClaimContext.Dims(sx), axis, perAxis, hasZero);
        return null;
    }

    private sealed class DequantizeBinding : GpuBinding
    {
        private readonly DType _type;
        private readonly int[] _shape;
        private readonly int _axis;
        private readonly bool _perAxis, _hasZero;
        private readonly Operand _x, _scale, _zero;
        private readonly int _outId;

        public DequantizeBinding(GpuClaimContext ctx, DType type, int[] shape, int axis, bool perAxis, bool hasZero)
        {
            _type = type;
            _shape = shape;
            _axis = axis;
            _perAxis = perAxis;
            _hasZero = hasZero;
            _x = Operand.Of(ctx, 0);
            _scale = Operand.Of(ctx, 1);
            _zero = hasZero ? Operand.Of(ctx, 2) : default;
            _outId = ctx.OutputId(0);
            Flops = ctx.OutputShape(0).Length;
        }

        public override DeviceElement OutputElement(int output) => DeviceElement.F32;

        public override void Encode(GpuRunContext ctx)
        {
            switch (_type)
            {
                case DType.U8:
                    Encode<byte>(ctx);
                    break;
                case DType.I8:
                    Encode<sbyte>(ctx);
                    break;
                default:
                    Encode<int>(ctx);
                    break;
            }
        }

        private void Encode<T>(GpuRunContext ctx) where T : unmanaged
        {
            GpuBuffer<T> x = _x.Get<T>(ctx, out int offX);
            GpuBuffer<float> scale = _scale.Get<float>(ctx, out int offS);
            GpuBuffer<T>? zero = null;
            int offZ = 0;
            if (_hasZero)
            {
                zero = _zero.Get<T>(ctx, out offZ);
            }

            GpuBuffer<float> y = ctx.F32(_outId, out int offY);
            ctx.Backend.Dequantize.Encode(ctx.Encoder, x, offX, _shape, _axis, _perAxis, scale, offS, zero, offZ, y, offY);
        }
    }

    // ---- QuantizeLinear ----------------------------------------------------------------------------------------------------

    private static string? CheckQuantize(NodeInfo info) =>
        info.GetInt("block_size", 0) != 0 ? "blocked quantization (block_size) is not supported on the GPU"
        : info.GetInt("output_dtype", 0) is not (0 or 2 or 3) ? "the GPU quantizes to uint8 and int8; 16-bit results are not held on the device"
        : null;

    private static string? BindQuantize(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            return "needs x and y_scale";
        }

        if (ctx.InputType(0) != DType.F32 || ctx.InputType(1) != DType.F32)
        {
            return $"the input is {ctx.InputType(0)} and the scale {ctx.InputType(1)}, and the GPU kernel takes float32 for both";
        }

        DType outType = ctx.OutputType(0);
        if (outType is not (DType.U8 or DType.I8))
        {
            return $"the result is {outType}; 16-bit values are not held on the device";
        }

        bool hasZero = ctx.HasInput(2);
        if (hasZero && ctx.InputType(2) != outType)
        {
            return "the zero point's type differs from the result's";
        }

        Shape sx = ctx.InputShape(0), ss = ctx.InputShape(1);
        if (sx.Length == 0 || sx.Rank > 8)
        {
            return "an empty tensor or one with more than eight dimensions";
        }

        bool perAxis = ss.Length > 1;
        long axisAttr = ctx.Info.GetInt("axis", 1);
        int axis = 0;
        if (perAxis)
        {
            axis = (int)(axisAttr < 0 ? axisAttr + sx.Rank : axisAttr);
            if (axis < 0 || axis >= sx.Rank || ss.Length != sx[axis])
            {
                return $"the scale has {ss.Length} values for axis {axisAttr} of {sx}";
            }
        }

        if (hasZero && ctx.InputShape(2).Length != (perAxis ? ss.Length : 1))
        {
            return "the zero point does not match the scale";
        }

        if ((ctx.FitsBinding(sx.Length, DeviceElement.F32) ?? ctx.FitsBinding(sx.Length, DeviceElement.U8)) is { } big)
        {
            return big;
        }

        if (ctx.InputMadeOnDevice(0) is { } made && made != DeviceElement.F32)
        {
            return "the input was made on the device as another type";
        }

        binding = new QuantizeBinding(ctx, outType == DType.I8, GpuClaimContext.Dims(sx), axis, perAxis, hasZero);
        return null;
    }

    private sealed class QuantizeBinding : GpuBinding
    {
        private readonly int[] _shape;
        private readonly int _axis;
        private readonly bool _perAxis, _hasZero, _signed;
        private readonly Operand _x, _scale, _zero;
        private readonly int _outId;

        public QuantizeBinding(GpuClaimContext ctx, bool signed, int[] shape, int axis, bool perAxis, bool hasZero)
        {
            _signed = signed;
            _shape = shape;
            _axis = axis;
            _perAxis = perAxis;
            _hasZero = hasZero;
            _x = Operand.Of(ctx, 0);
            _scale = Operand.Of(ctx, 1);
            _zero = hasZero ? Operand.Of(ctx, 2) : default;
            _outId = ctx.OutputId(0);
            Flops = ctx.OutputShape(0).Length;
        }

        public override DeviceElement OutputElement(int output) => _signed ? DeviceElement.I8 : DeviceElement.U8;

        public override void Encode(GpuRunContext ctx)
        {
            if (_signed)
            {
                Encode<sbyte>(ctx);
            }
            else
            {
                Encode<byte>(ctx);
            }
        }

        private void Encode<T>(GpuRunContext ctx) where T : unmanaged
        {
            GpuBuffer<float> x = _x.Get<float>(ctx, out int offX);
            GpuBuffer<float> scale = _scale.Get<float>(ctx, out int offS);
            GpuBuffer<T>? zero = null;
            int offZ = 0;
            if (_hasZero)
            {
                zero = _zero.Get<T>(ctx, out offZ);
            }

            GpuBuffer<T> y = ctx.Buffer<T>(_outId, out int offY);
            ctx.Backend.Dequantize.EncodeQuantize(ctx.Encoder, x, offX, _shape, _axis, _perAxis, scale, offS, zero, offZ, y, offY);
        }
    }
}
