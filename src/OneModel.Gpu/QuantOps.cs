using System;
using System.Collections.Generic;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Graph.Operators.Quant;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>
/// The dynamically quantized product: <c>DynamicQuantizeLinear</c> (range, scale and zero point found on the device, then the tensor
/// quantized to uint8) and the 8-bit matrix product with its dequantization fused in, the node the rewrite makes from
/// <c>MatMulInteger, Cast, Mul [, Add]</c>. The sums are exact integers, so the product gives the same bits on every device and in both
/// modes; the scale step is onemodel's arithmetic bit for bit (the kernel divides in integer arithmetic).
/// </summary>
internal static class QuantOps
{
    /// <summary>The table key of the fused product (its node info is that of the integer product it started from).</summary>
    public const string FusedProductKey = "~MatMulIntegerToFloat";

    public static IEnumerable<GpuOpEntry> Entries
    {
        get
        {
            yield return new GpuOpEntry("DynamicQuantizeLinear", true, _ => null, BindDynamicQuantize);
            yield return new GpuOpEntry(FusedProductKey, true, _ => null, BindProduct);
            yield return new GpuOpEntry("QLinearMatMul", true, _ => null, BindRequantized);
        }
    }

    // ---- DynamicQuantizeLinear ---------------------------------------------------------------------------------------------

    private static string? BindDynamicQuantize(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0))
        {
            return "needs an input";
        }

        if (ctx.InputType(0) != DType.F32)
        {
            return $"the input is {ctx.InputType(0)}, and the GPU kernel quantizes float32";
        }

        if (ctx.Step.Node.Outputs.Length != 3 || ctx.OutputId(0) < 0 || ctx.OutputId(1) < 0 || ctx.OutputId(2) < 0)
        {
            return "the GPU kernel makes the quantized tensor, the scale and the zero point together, and one of them is unused";
        }

        long count = ctx.InputShape(0).Length;
        if (count < 1)
        {
            return "the input is empty";
        }

        if (ctx.FitsBinding(count, DeviceElement.F32) is { } big)
        {
            return big;
        }

        binding = new QuantizeBinding(ctx, (int)count);
        return null;
    }

    private sealed class QuantizeBinding : GpuBinding
    {
        private readonly Operand _x;
        private readonly int _count;
        private readonly int _y, _scale, _zero;

        public QuantizeBinding(GpuClaimContext ctx, int count)
        {
            _x = Operand.Of(ctx, 0);
            _count = count;
            _y = ctx.OutputId(0);
            _scale = ctx.OutputId(1);
            _zero = ctx.OutputId(2);
            Flops = 3.0 * count;
        }

        public override DeviceElement OutputElement(int output) => output == 1 ? DeviceElement.F32 : DeviceElement.U8;

        public override void Encode(GpuRunContext ctx)
        {
            GpuBuffer<float> x = _x.Get<float>(ctx, out int offX);
            GpuBuffer<byte> y = ctx.Buffer<byte>(_y, out int offY);
            GpuBuffer<float> scale = ctx.Buffer<float>(_scale, out int offS);
            GpuBuffer<byte> zero = ctx.Buffer<byte>(_zero, out int offZ);
            ctx.Backend.DynamicQuantize.Encode(ctx.Encoder, x, offX, _count, y, offY, scale, offS, zero, offZ);
        }
    }

    // ---- the fused product -------------------------------------------------------------------------------------------------

    // Inputs: A (uint8), B (a constant uint8 or int8 matrix), a_zero_point, b_zero_point, scale, bias. Output float32.
    private static string? BindProduct(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0) || !ctx.HasInput(1) || !ctx.HasInput(4))
        {
            return "needs A, B and the scale";
        }

        if (ctx.InputType(0) != DType.U8)
        {
            return $"A is {ctx.InputType(0)}, and the GPU kernel takes uint8 activations";
        }

        DType bt = ctx.InputType(1);
        if (bt is not (DType.U8 or DType.I8))
        {
            return $"B is {bt}, and the GPU kernel takes uint8 and int8";
        }

        if (!ctx.InputIsConst(1))
        {
            return "B is not a constant";
        }

        Shape sa = ctx.InputShape(0), sb = ctx.InputShape(1);
        if (sb.Rank != 2 || sa.Rank < 1)
        {
            return "B must be a matrix and A at least a vector";
        }

        int k = sa[sa.Rank - 1], n = sb[1];
        if (sb[0] != k)
        {
            return $"the inner dimensions differ ({k} and {sb[0]})";
        }

        long m = sa.Length / Math.Max(k, 1);
        if (sa.Length == 0 || sb.Length == 0 || m < 1)
        {
            return "an operand is empty";
        }

        if (ctx.FitsBinding(sa.Length, DeviceElement.U8) is { } bigA || ctx.FitsBinding(m * n, DeviceElement.F32) is { } bigC)
        {
            return "an operand is larger than one buffer binding on this device";
        }

        if (ctx.OutputType(0) != DType.F32 || ctx.OutputShape(0).Length != m * n)
        {
            return "the planned output does not match the product";
        }

        bool aZeroVector = false, bZeroVector = false;
        if (ctx.HasInput(2))
        {
            if (ctx.InputType(2) != DType.U8)
            {
                return "a_zero_point is not uint8";
            }

            aZeroVector = ctx.InputShape(2).Length > 1;
            if (aZeroVector && (sa.Rank > 2 || ctx.InputShape(2).Length != m))
            {
                return "a_zero_point does not fit the rows of A";
            }
        }

        if (ctx.HasInput(3))
        {
            if (ctx.InputType(3) != bt)
            {
                return "b_zero_point has another type than B";
            }

            if (!ctx.InputIsConst(3) && ctx.InputMadeOnDevice(3) is not null)
            {
                return "b_zero_point is computed on the device as a type the kernel cannot read";
            }

            bZeroVector = ctx.InputShape(3).Length > 1;
            if (bZeroVector && ctx.InputShape(3).Length != n)
            {
                return "b_zero_point does not fit the columns of B";
            }
        }

        if (ctx.InputType(4) != DType.F32 || (ctx.InputShape(4).Length != 1 && ctx.InputShape(4).Length != n))
        {
            return "the scale is not float32 with one value or one per column";
        }

        bool hasBias = ctx.HasInput(5);
        if (hasBias && (ctx.InputType(5) != DType.F32 || ctx.InputShape(5).Length != n || !ctx.InputIsConst(5)))
        {
            return "the bias is not a constant float32 vector with one value per column";
        }

        if (ctx.HasInput(2) && ctx.InputMadeOnDevice(2) is { } ea && ea != DeviceElement.U8)
        {
            return "a_zero_point was made on the device as another type";
        }

        binding = new ProductBinding(ctx, checked((int)m), n, k, bt == DType.I8, aZeroVector, bZeroVector, ctx.InputShape(4).Length > 1, hasBias);
        return null;
    }

    private sealed class ProductBinding : GpuBinding
    {
        private readonly int _m, _n, _k;
        private readonly bool _bSigned, _aZeroVector, _bZeroVector, _scaleVector;
        private readonly Operand _a, _b, _aZero, _bZero, _scale, _bias;
        private readonly bool _hasAZero, _hasBZero, _hasBias;
        private readonly int _outId;

        public ProductBinding(GpuClaimContext ctx, int m, int n, int k, bool bSigned, bool aZeroVector, bool bZeroVector, bool scaleVector, bool hasBias)
        {
            _m = m;
            _n = n;
            _k = k;
            _bSigned = bSigned;
            _aZeroVector = aZeroVector;
            _bZeroVector = bZeroVector;
            _scaleVector = scaleVector;
            _a = Operand.Of(ctx, 0);
            _b = Operand.Of(ctx, 1);
            _scale = Operand.Of(ctx, 4);
            _hasAZero = ctx.HasInput(2);
            _hasBZero = ctx.HasInput(3);
            _hasBias = hasBias;
            if (_hasAZero)
            {
                _aZero = Operand.Of(ctx, 2);
            }

            if (_hasBZero)
            {
                _bZero = Operand.Of(ctx, 3);
            }

            if (_hasBias)
            {
                _bias = Operand.Of(ctx, 5);
            }

            _outId = ctx.OutputId(0);
            Flops = 2.0 * m * n * k;
        }

        public override DeviceElement OutputElement(int output) => DeviceElement.F32;

        public override void Encode(GpuRunContext ctx)
        {
            if (_bSigned)
            {
                Encode<sbyte>(ctx);
            }
            else
            {
                Encode<byte>(ctx);
            }
        }

        private void Encode<TB>(GpuRunContext ctx) where TB : unmanaged
        {
            GpuBuffer<byte> a = _a.Get<byte>(ctx, out int offA);
            GpuBuffer<TB> b = _b.Get<TB>(ctx, out int offB);
            GpuBuffer<float> c = ctx.Buffer<float>(_outId, out int offC);
            var problem = new Int8GemmProblem(_m, _n, _k) { OffsetA = offA, OffsetB = offB, OffsetC = offC };
            GpuBuffer<float> scale = _scale.Get<float>(ctx, out int offS);
            Int8Operand<byte>? aZero = null;
            if (_hasAZero)
            {
                aZero = new Int8Operand<byte>(_aZero.Get<byte>(ctx, out int o), o, _aZeroVector);
            }

            Int8Operand<TB>? bZero = null;
            if (_hasBZero)
            {
                bZero = new Int8Operand<TB>(_bZero.Get<TB>(ctx, out int o), o, _bZeroVector);
            }

            Int8Operand<float>? bias = null;
            if (_hasBias)
            {
                bias = new Int8Operand<float>(_bias.Get<float>(ctx, out int o), o, true);
            }

            ctx.Encoder.ReportWork(Flops);
            ctx.Backend.Int8.EncodeFloat(ctx.Encoder, problem, a, b, c, new Int8Operand<float>(scale, offS, _scaleVector), aZero, bZero, bias);
        }
    }

    // ---- QLinearMatMul -----------------------------------------------------------------------------------------------------

    // Inputs: a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point, and onemodel's optional int32 bias per column.
    // Output uint8: clamp(round_half_even(float(sum + bias) * (a_scale * b_scale / y_scale)) + y_zero_point). The multiplier is made here
    // in float32 in the CPU's order, so the bits are the same.
    private static string? BindRequantized(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (ctx.InputCount < 8)
        {
            return "QLinearMatMul needs its eight inputs";
        }

        for (int i = 0; i < 8; i++)
        {
            if (!ctx.HasInput(i))
            {
                return $"input {i} is required";
            }
        }

        if (ctx.InputType(0) != DType.U8)
        {
            return $"A is {ctx.InputType(0)}, and the GPU kernel takes uint8 activations";
        }

        DType bt = ctx.InputType(3);
        if (bt is not (DType.U8 or DType.I8) || !ctx.InputIsConst(3))
        {
            return "B is not a constant uint8 or int8 matrix";
        }

        if (ctx.OutputType(0) != DType.U8)
        {
            return $"the result is {ctx.OutputType(0)}, and the device holds uint8 activations only";
        }

        Shape sa = ctx.InputShape(0), sb = ctx.InputShape(3);
        if (sb.Rank != 2 || sa.Rank < 1)
        {
            return "B must be a matrix and A at least a vector";
        }

        int k = sa[sa.Rank - 1], n = sb[1];
        if (sb[0] != k)
        {
            return $"the inner dimensions differ ({k} and {sb[0]})";
        }

        long m = sa.Length / Math.Max(k, 1);
        if (sa.Length == 0 || sb.Length == 0 || m < 1)
        {
            return "an operand is empty";
        }

        if (ctx.FitsBinding(sa.Length, DeviceElement.U8) is { } || ctx.FitsBinding(m * n, DeviceElement.U8) is { })
        {
            return "an operand is larger than one buffer binding on this device";
        }

        if (ctx.OutputShape(0).Length != m * n)
        {
            return "the planned output does not match the product";
        }

        // Scales and the output zero point must be known now: the multiplier is computed once, on the host.
        foreach (int i in new[] { 1, 4, 6, 7 })
        {
            if (!ctx.InputIsConst(i))
            {
                return $"input {i} is not a constant, and the requantization multiplier is made when the plan is";
            }
        }

        if (ctx.InputType(1) != DType.F32 || ctx.InputType(4) != DType.F32 || ctx.InputType(6) != DType.F32 || ctx.InputType(7) != DType.U8)
        {
            return "the scales are not float32 or the output zero point is not uint8";
        }

        if (ctx.InputShape(1).Length != 1 || ctx.InputShape(6).Length != 1 || (ctx.InputShape(4).Length != 1 && ctx.InputShape(4).Length != n))
        {
            return "a scale does not have one value (b_scale: one value or one per column)";
        }

        bool aZeroVector = ctx.InputShape(2).Length > 1;
        if (ctx.InputType(2) != DType.U8 || (aZeroVector && (sa.Rank > 2 || ctx.InputShape(2).Length != m)))
        {
            return "a_zero_point is not uint8 or does not fit the rows of A";
        }

        bool bZeroVector = ctx.InputShape(5).Length > 1;
        if (ctx.InputType(5) != bt || (bZeroVector && ctx.InputShape(5).Length != n))
        {
            return "b_zero_point has another type than B or does not fit the columns of B";
        }

        if (ctx.InputMadeOnDevice(2) is { } ea && ea != DeviceElement.U8)
        {
            return "a_zero_point was made on the device as another type";
        }

        bool hasBias = ctx.InputCount > 8 && ctx.HasInput(8);
        if (hasBias && (ctx.InputType(8) != DType.I32 || ctx.InputShape(8).Length != n || !ctx.InputIsConst(8)))
        {
            return "the bias is not a constant int32 vector with one value per column";
        }

        float sA = ctx.InputConst(1).Data<float>()[0], sY = ctx.InputConst(6).Data<float>()[0];
        ReadOnlySpan<float> sB = ctx.InputConst(4).Data<float>();
        float[] multiplier = new float[sB.Length];
        for (int i = 0; i < multiplier.Length; i++)
        {
            multiplier[i] = sA * sB[i] / sY;
        }

        int zeroY = ctx.InputConst(7).Data<byte>()[0];
        using Tensor multiplierTensor = Tensor.FromSpan<float>(multiplier, Shape.Create(multiplier.Length));
        DeviceConstant mult = ctx.Backend.Constants.Upload(multiplierTensor);
        ctx.OwnForPlan(mult);
        binding = new RequantizedBinding(ctx, checked((int)m), n, k, bt == DType.I8, aZeroVector, bZeroVector, multiplier.Length > 1, zeroY, mult, hasBias);
        return null;
    }

    private sealed class RequantizedBinding : GpuBinding
    {
        private readonly int _m, _n, _k, _zeroY;
        private readonly bool _bSigned, _aZeroVector, _bZeroVector, _multiplierVector, _hasBias;
        private readonly Operand _a, _b, _aZero, _bZero, _bias;
        private readonly DeviceConstant _multiplier;
        private readonly int _outId;

        public RequantizedBinding(GpuClaimContext ctx, int m, int n, int k, bool bSigned, bool aZeroVector, bool bZeroVector, bool multiplierVector, int zeroY, DeviceConstant multiplier, bool hasBias)
        {
            _m = m;
            _n = n;
            _k = k;
            _bSigned = bSigned;
            _aZeroVector = aZeroVector;
            _bZeroVector = bZeroVector;
            _multiplierVector = multiplierVector;
            _zeroY = zeroY;
            _multiplier = multiplier;
            _hasBias = hasBias;
            _a = Operand.Of(ctx, 0);
            _b = Operand.Of(ctx, 3);
            _aZero = Operand.Of(ctx, 2);
            _bZero = Operand.Of(ctx, 5);
            if (hasBias)
            {
                _bias = Operand.Of(ctx, 8);
            }

            _outId = ctx.OutputId(0);
            Flops = 2.0 * m * n * k;
        }

        public override DeviceElement OutputElement(int output) => DeviceElement.U8;

        // The sums wait in the arena between the product and the requantizing pass.
        public override long ScratchI32Elements => Int8Gemm.ScratchElements(new Int8GemmProblem(_m, _n, _k));

        public override void Encode(GpuRunContext ctx)
        {
            if (_bSigned)
            {
                Encode<sbyte>(ctx);
            }
            else
            {
                Encode<byte>(ctx);
            }
        }

        private void Encode<TB>(GpuRunContext ctx) where TB : unmanaged
        {
            GpuBuffer<byte> a = _a.Get<byte>(ctx, out int offA);
            GpuBuffer<TB> b = _b.Get<TB>(ctx, out int offB);
            GpuBuffer<byte> y = ctx.Buffer<byte>(_outId, out int offY);
            var problem = new Int8GemmProblem(_m, _n, _k) { OffsetA = offA, OffsetB = offB, OffsetC = offY };
            var aZero = new Int8Operand<byte>(_aZero.Get<byte>(ctx, out int oa), oa, _aZeroVector);
            var bZero = new Int8Operand<TB>(_bZero.Get<TB>(ctx, out int ob), ob, _bZeroVector);
            Int8Operand<int>? bias = null;
            if (_hasBias)
            {
                bias = new Int8Operand<int>(_bias.Get<int>(ctx, out int oi), oi, true);
            }

            var multiplier = new Int8Operand<float>(_multiplier.F32!, 0, _multiplierVector);
            ctx.Encoder.ReportWork(Flops);
            GpuBuffer<int> sums = ctx.ScratchI32(this, out int offSums);
            ctx.Backend.Int8.EncodeRequantized(ctx.Encoder, problem, a, b, y, multiplier, _zeroY, aZero, bZero, bias, new Int8Scratch(sums, offSums));
        }
    }
}
