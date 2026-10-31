using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Quant;

// QuantizeLinear, DequantizeLinear and DynamicQuantizeLinear: the conversions between float tensors and 8-bit (or 16-bit) integers
// that every quantized model begins and ends with. The arithmetic is in Kernels/Int8Quantize.cs; this file holds the shape and
// parameter rules. Scale and zero point are either one value for the whole tensor or one value per index along `axis`.

/// <summary>Shared parameter handling: the scale, the zero point and the axis of a quantize or dequantize node.</summary>
internal static class QuantParams
{
    /// <summary>Splits a shape around <paramref name="axis"/> into outer, axis and inner extents.</summary>
    public static void Split(Shape shape, int axis, out long outer, out int axisLen, out long inner)
    {
        outer = 1;
        inner = 1;
        axisLen = shape.Rank == 0 ? 1 : shape[axis];
        for (int i = 0; i < axis; i++)
        {
            outer *= shape[i];
        }

        for (int i = axis + 1; i < shape.Rank; i++)
        {
            inner *= shape[i];
        }
    }

    /// <summary>Checks the scale (and zero point) shapes against the input and returns the axis to use (0 when per-tensor).</summary>
    public static int Check(InferContext ctx, Shape x, int axisAttr, bool hasZero, int scaleInput, int zeroInput, out bool perAxis)
    {
        Shape s = ctx.InputShape(scaleInput);
        if (ctx.InputType(scaleInput) != DType.F32)
        {
            throw ctx.Unsupported($"scale element type {ctx.InputType(scaleInput)} is not supported (float32 only)");
        }

        perAxis = false;
        int axis = 0;
        if (s.Rank == 0 || (s.Rank == 1 && s[0] == 1))
        {
            perAxis = false;
        }
        else if (s.Rank == 1)
        {
            if (x.Rank == 0)
            {
                throw ctx.Fail("a per-axis scale needs an input with at least one dimension");
            }

            if (axisAttr < -x.Rank || axisAttr >= x.Rank)
            {
                throw ctx.Fail($"axis {axisAttr} is out of range for an input of rank {x.Rank}");
            }

            axis = Shape.NormalizeAxis(axisAttr, x.Rank);
            if (s[0] != x[axis])
            {
                throw ctx.Fail($"the scale has {s[0]} values but axis {axis} of {x} has {x[axis]}");
            }

            perAxis = true;
        }
        else
        {
            throw ctx.Unsupported($"scale shape {s} (blocked quantization) is not supported");
        }

        if (hasZero && ctx.InputShape(zeroInput).Length != s.Length)
        {
            throw ctx.Fail($"the zero point {ctx.InputShape(zeroInput)} must have as many values as the scale {s}");
        }

        return axis;
    }

    /// <summary>Reads zero point <paramref name="i"/> of a zero-point tensor of any supported 8-, 16- or 32-bit integer type.</summary>
    public static int ZeroAt(in Tensor zero, int i)
    {
        if (zero.IsEmpty)
        {
            return 0;
        }

        int idx = zero.Length == 1 ? 0 : i;
        return zero.DType switch
        {
            DType.U8 => zero.Data<byte>()[idx],
            DType.I8 => zero.Data<sbyte>()[idx],
            DType.U16 => zero.Data<ushort>()[idx],
            DType.I16 => zero.Data<short>()[idx],
            DType.I32 => zero.Data<int>()[idx],
            _ => throw new InvalidOperationException($"Zero point type {zero.DType} reached the kernel."),
        };
    }

    /// <summary>The raw bytes of an 8-bit tensor of either signedness.</summary>
    public static ReadOnlySpan<byte> Bytes(in Tensor t) => t.DType == DType.U8 ? t.Data<byte>() : Int8Quantize.AsBytes(t.Data<sbyte>());

    /// <summary>The writable raw bytes of an 8-bit tensor of either signedness.</summary>
    public static Span<byte> MutableBytes(in Tensor t) => t.DType == DType.U8 ? t.MutableData<byte>() : Int8Quantize.AsBytes(t.MutableData<sbyte>());

    /// <summary>The raw 16-bit patterns of a 16-bit tensor of either signedness.</summary>
    public static ReadOnlySpan<short> Shorts(in Tensor t) => t.DType == DType.U16 ? Int8Quantize.AsShorts(t.Data<ushort>()) : t.Data<short>();

    /// <summary>The writable raw 16-bit patterns of a 16-bit tensor of either signedness.</summary>
    public static Span<short> MutableShorts(in Tensor t) => t.DType == DType.U16 ? Int8Quantize.AsShorts(t.MutableData<ushort>()) : t.MutableData<short>();

    /// <summary>Input <paramref name="i"/> of a run, or an empty tensor when trailing optional inputs were omitted.</summary>
    public static Tensor Optional(in OpRun run, int i) => i < run.Inputs.Length ? run.Inputs[i] : default;

    public static bool IsQuantType(DType t) => t is DType.U8 or DType.I8 or DType.U16 or DType.I16;
}

/// <summary>QuantizeLinear: y = saturate(round_half_even(x / y_scale) + y_zero_point), per tensor or per axis, to uint8, int8, uint16 or int16.</summary>
internal sealed class QuantizeLinearOp : OpKernel
{
    public static readonly OpDef Def = new("QuantizeLinear", 10, int.MaxValue, static n => new QuantizeLinearOp(n));

    private readonly int _axis;
    private readonly DType _outputDType;

    private QuantizeLinearOp(NodeInfo node)
    {
        _axis = (int)node.GetInt("axis", 1);
        if (node.GetInt("block_size", 0) != 0)
        {
            throw node.Unsupported("block_size (blocked quantization) is not supported");
        }

        _outputDType = node.GetInt("output_dtype", 0) switch
        {
            0 => DType.Undefined,
            2 => DType.U8,
            3 => DType.I8,
            4 => DType.U16,
            5 => DType.I16,
            var other => throw node.Unsupported($"output_dtype {other} is not supported (uint8, int8, uint16, int16)"),
        };
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects x and y_scale");
        }

        if (ctx.InputType(0) != DType.F32)
        {
            throw ctx.Unsupported($"input element type {ctx.InputType(0)} is not supported (float32 only)");
        }

        bool hasZero = ctx.HasInput(2);
        DType t = hasZero ? ctx.InputType(2) : _outputDType == DType.Undefined ? DType.U8 : _outputDType;
        if (!QuantParams.IsQuantType(t))
        {
            throw ctx.Unsupported($"quantized element type {t} is not supported (uint8, int8, uint16, int16)");
        }

        if (hasZero && _outputDType != DType.Undefined && _outputDType != t)
        {
            throw ctx.Fail("output_dtype disagrees with the zero point's type");
        }

        _ = QuantParams.Check(ctx, ctx.InputShape(0), _axis, hasZero, 1, 2, out _);
        ctx.SetOutput(0, t, ctx.InputShape(0));
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        if (x.Length == 0)
        {
            return;
        }

        ReadOnlySpan<float> scales = run.Inputs[1].Data<float>();
        Tensor zero = QuantParams.Optional(in run, 2);
        bool perAxis = scales.Length > 1;
        int axis = perAxis ? Shape.NormalizeAxis(_axis, x.Rank) : 0;
        QuantParams.Split(x.Shape, axis, out long outer, out int axisLen, out long inner);
        bool unsigned = y.DType is DType.U8 or DType.U16;
        bool wide = y.DType is DType.U16 or DType.I16;
        ReadOnlySpan<float> xs = x.Data<float>();
        Span<byte> y8 = wide ? default : QuantParams.MutableBytes(in y);
        Span<short> y16 = wide ? QuantParams.MutableShorts(in y) : default;
        if (!perAxis)
        {
            float scale = scales[0];
            int zp = QuantParams.ZeroAt(in zero, 0);
            if (wide)
            {
                Int8Quantize.Quantize16(xs, y16, scale, zp, unsigned);
            }
            else
            {
                QuantizeParallel.Quantize8(xs, y8, scale, zp, unsigned, run.Threads);
            }

            return;
        }

        int lo = y.DType switch { DType.U8 => 0, DType.I8 => -128, DType.U16 => 0, _ => short.MinValue };
        int hi = y.DType switch { DType.U8 => 255, DType.I8 => 127, DType.U16 => ushort.MaxValue, _ => short.MaxValue };
        for (long o = 0; o < outer; o++)
        {
            for (int a = 0; a < axisLen; a++)
            {
                float scale = scales[a];
                int zp = QuantParams.ZeroAt(in zero, a);
                int off = checked((int)(((o * axisLen) + a) * inner));
                int len = (int)inner;
                if (inner >= 16 && !wide)
                {
                    Int8Quantize.Quantize8(xs.Slice(off, len), y8.Slice(off, len), scale, zp, unsigned);
                    continue;
                }

                for (int i = 0; i < len; i++)
                {
                    int q = Int8Quantize.Quantize1(xs[off + i], scale, zp, lo, hi);
                    if (wide)
                    {
                        y16[off + i] = (short)q;
                    }
                    else
                    {
                        y8[off + i] = (byte)q;
                    }
                }
            }
        }
    }
}

/// <summary>DequantizeLinear: y = (x - x_zero_point) * x_scale, per tensor or per axis; float32 output.</summary>
internal sealed class DequantizeLinearOp : OpKernel
{
    public static readonly OpDef Def = new("DequantizeLinear", 10, int.MaxValue, static n => new DequantizeLinearOp(n));

    private readonly int _axis;

    private DequantizeLinearOp(NodeInfo node)
    {
        _axis = (int)node.GetInt("axis", 1);
        if (node.GetInt("block_size", 0) != 0)
        {
            throw node.Unsupported("block_size (blocked quantization) is not supported");
        }
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects x and x_scale");
        }

        DType t = ctx.InputType(0);
        if (!QuantParams.IsQuantType(t) && t != DType.I32)
        {
            throw ctx.Unsupported($"input element type {t} is not supported (uint8, int8, uint16, int16, int32)");
        }

        bool hasZero = ctx.HasInput(2);
        if (hasZero && ctx.InputType(2) != t)
        {
            throw ctx.Fail($"the zero point's type {ctx.InputType(2)} must match the input's {t}");
        }

        _ = QuantParams.Check(ctx, ctx.InputShape(0), _axis, hasZero, 1, 2, out _);
        // A weight: everything known now, so dequantize once at plan time instead of on every run.
        if (ctx.TryGetInputValue(0, out Tensor x) && ctx.TryGetInputValue(1, out Tensor s) &&
            (!hasZero || ctx.TryGetInputValue(2, out _)))
        {
            ctx.TryGetInputValue(2, out Tensor z);
            Tensor y = Tensor.Create(DType.F32, ctx.InputShape(0));
            Compute(x, s, z, y);
            ctx.SetConstantOutput(0, y, planOwns: true);
            return;
        }

        ctx.SetOutput(0, DType.F32, ctx.InputShape(0));
    }

    public override void Run(in OpRun run) => Compute(run.Inputs[0], run.Inputs[1], QuantParams.Optional(in run, 2), run.Outputs[0]);

    private void Compute(in Tensor x, in Tensor scale, in Tensor zero, in Tensor y)
    {
        if (x.Length == 0)
        {
            return;
        }

        ReadOnlySpan<float> scales = scale.Data<float>();
        bool perAxis = scales.Length > 1;
        int axis = perAxis ? Shape.NormalizeAxis(_axis, x.Rank) : 0;
        QuantParams.Split(x.Shape, axis, out long outer, out int axisLen, out long inner);
        DType t = x.DType;
        bool unsigned = t is DType.U8 or DType.U16;
        Span<float> ys = y.MutableData<float>();
        int len = (int)inner;
        for (long o = 0; o < outer; o++)
        {
            for (int a = 0; a < axisLen; a++)
            {
                float s = scales[perAxis ? a : 0];
                int zp = QuantParams.ZeroAt(in zero, perAxis ? a : 0);
                int off = checked((int)(((o * axisLen) + a) * inner));
                Span<float> dst = ys.Slice(off, len);
                switch (t)
                {
                    case DType.U8:
                    case DType.I8:
                        Int8Quantize.Dequantize8(QuantParams.Bytes(in x).Slice(off, len), unsigned, dst, s, zp);
                        break;
                    case DType.U16:
                    case DType.I16:
                        Int8Quantize.Dequantize16(QuantParams.Shorts(in x).Slice(off, len), unsigned, dst, s, zp);
                        break;
                    default:
                        ReadOnlySpan<int> xi = x.Data<int>().Slice(off, len);
                        for (int i = 0; i < dst.Length; i++)
                        {
                            dst[i] = (float)unchecked(xi[i] - zp) * s;
                        }

                        break;
                }
            }
        }
    }
}

/// <summary>
/// DynamicQuantizeLinear: quantizes a float tensor to uint8 with a scale and zero point computed from its own range, widened to
/// include zero: <c>scale = (max(0, max x) - min(0, min x)) / 255</c>, <c>zero point = clamp(round_half_even(-min / scale), 0, 255)</c>.
/// An all-zero input gives scale 1 and zero point 0.
/// </summary>
internal sealed class DynamicQuantizeLinearOp : OpKernel
{
    public static readonly OpDef Def = new("DynamicQuantizeLinear", 11, int.MaxValue, static _ => new DynamicQuantizeLinearOp());

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        if (ctx.InputType(0) != DType.F32)
        {
            throw ctx.Unsupported($"input element type {ctx.InputType(0)} is not supported (float32 only)");
        }

        ctx.SetOutput(0, DType.U8, ctx.InputShape(0));
        ctx.SetOutput(1, DType.F32, default);
        ctx.SetOutput(2, DType.U8, default);
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0];
        Span<float> scaleOut = run.Outputs[1].MutableData<float>();
        Span<byte> zeroOut = run.Outputs[2].MutableData<byte>();
        Span<byte> y = run.Outputs[0].MutableData<byte>();
        ReadOnlySpan<float> xs = x.Data<float>();
        QuantizeParallel.MinMax(xs, out float min, out float max, run.Threads);
        // The range always contains zero, so that zero is exactly representable.
        float lo = Math.Min(0f, min), hi = Math.Max(0f, max);
        float scale = (hi - lo) / 255f;
        int zp = 0;
        if (scale != 0f && float.IsFinite(scale))
        {
            float z = MathF.Round(0f - (lo / scale));
            zp = z < 0f ? 0 : z > 255f ? 255 : (int)z;
        }

        if (scale == 0f)
        {
            // Nothing to represent: ONNX Runtime reports a unit scale and quantizes everything (all zeros) to 0.
            scaleOut[0] = 1f;
            zeroOut[0] = 0;
            y.Clear();
            return;
        }

        scaleOut[0] = scale;
        zeroOut[0] = (byte)zp;

        QuantizeParallel.Quantize8(xs, y, scale, zp, unsigned: true, run.Threads);
    }
}
