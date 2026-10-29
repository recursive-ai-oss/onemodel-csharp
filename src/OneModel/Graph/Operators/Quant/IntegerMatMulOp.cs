using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Quant;

/// <summary>Helpers shared by the integer matrix-product and convolution operators.</summary>
internal static class QuantCommon
{
    /// <summary>True for uint8, false for int8; anything else is rejected.</summary>
    public static bool IsU8(InferContext ctx, DType t, string what)
    {
        if (t == DType.U8)
        {
            return true;
        }

        if (t == DType.I8)
        {
            return false;
        }

        throw ctx.Unsupported($"{what} element type {t} is not supported (uint8 or int8)");
    }

    /// <summary>A zero point in the shifted domain (uint8 minus 128). An absent zero point is 0, which is -128 shifted for uint8 data.</summary>
    public static int Shifted(in Tensor zero, int index, bool dataIsU8)
    {
        if (zero.IsEmpty)
        {
            return dataIsU8 ? -128 : 0;
        }

        int i = zero.Length == 1 ? 0 : index;
        return zero.DType == DType.U8 ? zero.Data<byte>()[i] - 128 : zero.Data<sbyte>()[i];
    }

    /// <summary>The zero point as stored (the value padding takes in a convolution).</summary>
    public static int Raw(in Tensor zero) => zero.IsEmpty ? 0 : zero.DType == DType.U8 ? zero.Data<byte>()[0] : zero.Data<sbyte>()[0];

    /// <summary>The shifted zero points of <paramref name="zero"/> as an int array of <paramref name="count"/> values, or null when it holds one value (then <paramref name="scalar"/> has it).</summary>
    public static int[]? ShiftedVector(in Tensor zero, int count, out int scalar)
    {
        scalar = Shifted(in zero, 0, zero.DType == DType.U8);
        if (zero.IsEmpty || zero.Length == 1)
        {
            return null;
        }

        int[] v = new int[count];
        for (int i = 0; i < count; i++)
        {
            v[i] = Shifted(in zero, i, zero.DType == DType.U8);
        }

        return v;
    }

    /// <summary>Checks that a zero point or scale input is a scalar, a one-element tensor, or a 1-D tensor of <paramref name="count"/> values.</summary>
    public static bool IsPerChannel(InferContext ctx, int input, int count, string what)
    {
        Shape s = ctx.InputShape(input);
        if (s.Rank == 0 || (s.Rank == 1 && s[0] == 1))
        {
            return false;
        }

        if (s.Rank == 1 && s[0] == count)
        {
            return true;
        }

        throw ctx.Fail($"{what} {s} must be a scalar or have {count} values");
    }
}

/// <summary>State a MatMulInteger or QLinearMatMul keeps from <see cref="OpKernel.Prepare"/>: the constant B, packed once.</summary>
internal sealed class IntMatMulState
{
    public Int8Packed? B;
    public int[]? ZbVec;
    public int ZbScalar;
}

/// <summary>
/// MatMulInteger and QLinearMatMul: the 8-bit matrix product Y = (A - a_zero_point) (B - b_zero_point) with batch broadcasting. MatMulInteger
/// returns the exact int32 sums; QLinearMatMul requantizes them: <c>y = saturate(round_half_even(sum * a_scale * b_scale / y_scale) + y_zero_point)</c>.
/// A constant B is packed once when the plan is made. The arithmetic is <see cref="Int8Gemm"/>'s: exact integers, identical on every path.
/// </summary>
internal sealed class IntegerMatMulOp : OpKernel
{
    public static readonly OpDef MatMulInteger = new("MatMulInteger", 10, int.MaxValue, static _ => new IntegerMatMulOp(false));
    public static readonly OpDef QLinearMatMul = new("QLinearMatMul", 10, int.MaxValue, static _ => new IntegerMatMulOp(true));

    /// <summary>
    /// The product with its dequantization fused in, made only by the graph pass that fuses MatMulInteger, Cast to float32, Mul by the
    /// scale and an optional bias Add. Inputs: A, B, a_zero_point, b_zero_point (either may be omitted), scale (a scalar or one value per
    /// column), bias (optional, one value per column). Output float32: <c>(float)sum * scale + bias</c>, the two correctly rounded
    /// operations the unfused chain performs, so the bits are the same.
    /// </summary>
    public static readonly OpDef MatMulIntegerToFloat = new("MatMulIntegerToFloat", 1, int.MaxValue, static _ => new IntegerMatMulOp(false, true));

    private readonly bool _q;
    private readonly bool _float;

    private IntegerMatMulOp(bool quantized, bool toFloat = false)
    {
        _q = quantized;
        _float = toFloat;
    }

    public override bool SupportsExact => true;

    private static int AIn => 0;

    private int BIn => _q ? 3 : 1;

    private static int AZeroIn => 2;

    private int BZeroIn => _q ? 5 : 3;

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

        if (_float && !ctx.HasInput(4))
        {
            throw ctx.Fail("input 4 (the scale) is required");
        }

        _ = QuantCommon.IsU8(ctx, ctx.InputType(AIn), "A");
        _ = QuantCommon.IsU8(ctx, ctx.InputType(BIn), "B");
        Shape a = ctx.InputShape(AIn), b = ctx.InputShape(BIn);
        if (a.Rank == 0 || b.Rank == 0)
        {
            throw ctx.Fail("operands must have at least one dimension");
        }

        int m = a.Rank == 1 ? 1 : a[a.Rank - 2];
        int k = a[a.Rank - 1];
        int kb = b.Rank == 1 ? b[0] : b[b.Rank - 2];
        int n = b.Rank == 1 ? 1 : b[b.Rank - 1];
        if (k != kb)
        {
            throw ctx.Fail($"inner dimensions do not match: {a} times {b}");
        }

        Shape batchA = a.Rank > 2 ? Shape.Create(a.Dimensions.Slice(0, a.Rank - 2)) : default;
        Shape batchB = b.Rank > 2 ? Shape.Create(b.Dimensions.Slice(0, b.Rank - 2)) : default;
        if (!Shape.TryBroadcast(batchA, batchB, out Shape batch))
        {
            throw ctx.Fail($"batch shapes {batchA} and {batchB} cannot be broadcast together");
        }

        if (ctx.HasInput(AZeroIn))
        {
            if (ctx.InputType(AZeroIn) != ctx.InputType(AIn))
            {
                throw ctx.Fail("a_zero_point must have A's element type");
            }

            if (QuantCommon.IsPerChannel(ctx, AZeroIn, m, "a_zero_point") && a.Rank > 2)
            {
                throw ctx.Unsupported("a per-row a_zero_point with a batched A is not supported");
            }
        }

        if (ctx.HasInput(BZeroIn))
        {
            if (ctx.InputType(BZeroIn) != ctx.InputType(BIn))
            {
                throw ctx.Fail("b_zero_point must have B's element type");
            }

            _ = QuantCommon.IsPerChannel(ctx, BZeroIn, n, "b_zero_point");
        }

        DType outType = _float ? DType.F32 : DType.I32;
        if (_float)
        {
            CheckColumnVector(ctx, 4, n, "the scale");
            if (ctx.HasInput(5))
            {
                CheckColumnVector(ctx, 5, n, "the bias");
            }
        }

        if (_q)
        {
            _ = QuantCommon.IsPerChannel(ctx, 1, 1, "a_scale");
            _ = QuantCommon.IsPerChannel(ctx, 4, n, "b_scale");
            _ = QuantCommon.IsPerChannel(ctx, 6, 1, "y_scale");
            if (ctx.InputType(1) != DType.F32 || ctx.InputType(4) != DType.F32 || ctx.InputType(6) != DType.F32)
            {
                throw ctx.Unsupported("scales must be float32");
            }

            outType = ctx.InputType(7);
            _ = QuantCommon.IsU8(ctx, outType, "y_zero_point");
            // An optional ninth input, not in the ONNX operator: an int32 bias per output column, added to the sums before scaling.
            // The QDQ rewrite uses it to absorb a Gemm's bias.
            if (ctx.HasInput(8) && (ctx.InputType(8) != DType.I32 || ctx.InputShape(8).Rank != 1 || ctx.InputShape(8)[0] != n))
            {
                throw ctx.Fail($"the bias must be int32 with {n} values");
            }
        }

        Span<int> d = stackalloc int[Shape.MaxRank + 2];
        int r = 0;
        for (int i = 0; i < batch.Rank; i++)
        {
            d[r++] = batch[i];
        }

        if (a.Rank > 1)
        {
            d[r++] = m;
        }

        if (b.Rank > 1)
        {
            d[r++] = n;
        }

        if (r > Shape.MaxRank)
        {
            throw ctx.Fail($"the result would have {r} dimensions; the limit is {Shape.MaxRank}");
        }

        ctx.SetOutput(0, outType, Shape.Create(d.Slice(0, r)));
    }

    // A float32 scale or bias: one value, or one per column (leading dimensions of size 1).
    private static void CheckColumnVector(InferContext ctx, int input, int n, string what)
    {
        Shape s = ctx.InputShape(input);
        if (ctx.InputType(input) != DType.F32)
        {
            throw ctx.Unsupported($"{what} must be float32");
        }

        for (int d = 0; d < s.Rank - 1; d++)
        {
            if (s[d] != 1)
            {
                throw ctx.Unsupported($"{what} {s} must be a scalar or have {n} values along the last axis only");
            }
        }

        if (s.Length != 1 && !(s.Rank >= 1 && s[s.Rank - 1] == n))
        {
            throw ctx.Unsupported($"{what} {s} must be a scalar or have {n} values along the last axis only");
        }
    }

    public override object? Prepare(PrepareContext ctx)
    {
        if (!ctx.IsConstant(BIn))
        {
            return null;
        }

        Tensor b = ctx.Constant(BIn);
        var st = new IntMatMulState();
        Shape sb = b.Shape;
        if (sb.Rank <= 2)
        {
            int k = sb[0], n = sb.Rank == 1 ? 1 : sb[1];
            Int8Format f = Int8Format.For(ctx.Path);
            st.B = Int8Packed.ForB(in f, QuantParams.Bytes(in b), n, b.DType == DType.U8, k, n);
        }

        if (ctx.IsConstant(BZeroIn))
        {
            int n = sb.Rank == 1 ? 1 : sb[sb.Rank - 1];
            st.ZbVec = QuantCommon.ShiftedVector(ctx.Constant(BZeroIn), n, out st.ZbScalar);
        }

        return st;
    }

    public override void Run(in OpRun run)
    {
        Tensor a = run.Inputs[AIn], b = run.Inputs[BIn], y = run.Outputs[0];
        if (y.Length == 0)
        {
            return;
        }

        Shape sa = a.Shape, sb = b.Shape;
        int m = sa.Rank == 1 ? 1 : sa[sa.Rank - 2];
        int k = sa[sa.Rank - 1];
        int n = sb.Rank == 1 ? 1 : sb[sb.Rank - 1];
        bool aU8 = a.DType == DType.U8, bU8 = b.DType == DType.U8;
        Int8Format f = Int8Format.For(run.Path);
        var st = run.State as IntMatMulState;

        // Zero points.
        Tensor azT = QuantParams.Optional(in run, AZeroIn), bzT = QuantParams.Optional(in run, BZeroIn);
        int za = QuantCommon.Shifted(in azT, 0, aU8);
        ReadOnlySpan<int> zaVec = default;
        if (!azT.IsEmpty && azT.Length > 1)
        {
            Span<int> v = run.Scratch(DType.I32, Shape.Create(azT.Length)).MutableData<int>();
            for (int i = 0; i < v.Length; i++)
            {
                v[i] = QuantCommon.Shifted(in azT, i, aU8);
            }

            zaVec = v;
        }

        int zb;
        ReadOnlySpan<int> zbVec;
        if (st?.ZbVec != null || (st != null && !bzT.IsEmpty && bzT.Length == 1))
        {
            zb = st!.ZbScalar;
            zbVec = st.ZbVec;
        }
        else
        {
            zb = QuantCommon.Shifted(in bzT, 0, bU8);
            zbVec = default;
            if (!bzT.IsEmpty && bzT.Length > 1)
            {
                Span<int> v = run.Scratch(DType.I32, Shape.Create(bzT.Length)).MutableData<int>();
                for (int i = 0; i < v.Length; i++)
                {
                    v[i] = QuantCommon.Shifted(in bzT, i, bU8);
                }

                zbVec = v;
            }
        }

        // Requantization scale.
        float mult = 0f;
        ReadOnlySpan<float> multVec = default;
        int quantZero = 0;
        bool quantUnsigned = false;
        ReadOnlySpan<float> biasFloat = default;
        if (_float)
        {
            Tensor sc = run.Inputs[4];
            if (sc.Length == 1)
            {
                mult = sc.Data<float>()[0];
            }
            else
            {
                multVec = sc.Data<float>();
            }

            Tensor bf = QuantParams.Optional(in run, 5);
            if (!bf.IsEmpty)
            {
                if (bf.Length == 1)
                {
                    // One bias for every column: expand it once into scratch (rare).
                    Span<float> e = run.Scratch(DType.F32, Shape.Create(n)).MutableData<float>();
                    e.Fill(bf.Data<float>()[0]);
                    biasFloat = e;
                }
                else
                {
                    biasFloat = bf.Data<float>();
                }
            }
        }

        if (_q)
        {
            float sA = run.Inputs[1].Data<float>()[0];
            float sY = run.Inputs[6].Data<float>()[0];
            ReadOnlySpan<float> sB = run.Inputs[4].Data<float>();
            if (sB.Length == 1)
            {
                mult = sA * sB[0] / sY;
            }
            else
            {
                Span<float> mv = run.Scratch(DType.F32, Shape.Create(sB.Length)).MutableData<float>();
                for (int i = 0; i < mv.Length; i++)
                {
                    mv[i] = sA * sB[i] / sY;
                }

                multVec = mv;
            }

            quantUnsigned = y.DType == DType.U8;
            quantZero = QuantCommon.Raw(run.Inputs[7]);
        }

        // Batch layout.
        long batchOut = 1;
        for (int i = 0; i < Math.Max(0, y.Rank - (sa.Rank > 1 ? 1 : 0) - (sb.Rank > 1 ? 1 : 0)); i++)
        {
            batchOut *= y.Shape[i];
        }

        ReadOnlySpan<int> bias = _q && run.Inputs.Length > 8 && !run.Inputs[8].IsEmpty ? run.Inputs[8].Data<int>() : default;
        long mk = (long)m * k, kn = (long)k * n, mn = (long)m * n;
        ReadOnlySpan<byte> aBytes = QuantParams.Bytes(in a);
        if (st?.B != null)
        {
            // B constant and unbatched: every row of every A batch is one big M.
            long mTotal = a.Length / k;
            Int8Packed pb = st.B;
            var call = new Int8Call
            {
                Format = f,
                M = checked((int)mTotal),
                N = n,
                K = k,
                ARaw = aBytes,
                Lda = k,
                AU8 = aU8,
                BPacked = pb.Data,
                BSums = pb.Sums,
                ZaScalar = za,
                ZaVec = zaVec,
                ZbScalar = zb,
                ZbVec = zbVec,
                Ldc = n,
                Mult = mult,
                MultVec = multVec,
                MultPerRow = false,
                Bias = bias,
                BiasPerCol = true,
                QuantUnsigned = quantUnsigned,
                QuantZero = quantZero,
                Threads = run.Threads,
            };
            call.BiasFloat = biasFloat;
            if (_float)
            {
                call.OutFloat = y.MutableData<float>();
            }
            else if (_q)
            {
                call.OutQuant = QuantParams.MutableBytes(in y);
            }
            else
            {
                call.OutInt32 = y.MutableData<int>();
            }

            Int8Matmul.Run(ref call);
            return;
        }

        // General case: pack B per batch into scratch.
        Span<byte> packed = run.Scratch(DType.U8, Shape.Create(checked((int)f.PackedBBytes(k, n)))).MutableData<byte>();
        Span<int> sums = run.Scratch(DType.I32, Shape.Create(n)).MutableData<int>();
        ReadOnlySpan<byte> bBytes = QuantParams.Bytes(in b);
        int batchRankA = Math.Max(0, sa.Rank - 2), batchRankB = Math.Max(0, sb.Rank - 2);
        Span<int> idx = stackalloc int[Shape.MaxRank];
        int outBatchRank = Math.Max(batchRankA, batchRankB);
        for (long bi = 0; bi < batchOut; bi++)
        {
            // Unravel the batch index into the broadcast shape and map it into each operand.
            long rem = bi;
            for (int d = outBatchRank - 1; d >= 0; d--)
            {
                int ext = y.Shape[d];
                idx[d] = (int)(rem % ext);
                rem /= ext;
            }

            long ia = BatchOffset(sa, batchRankA, outBatchRank, idx);
            long ib = BatchOffset(sb, batchRankB, outBatchRank, idx);
            Int8Matmul.PackB(in f, bBytes.Slice((int)(ib * kn), (int)kn), n, bU8, k, n, packed, sums, run.Threads);
            var call = new Int8Call
            {
                Format = f,
                M = m,
                N = n,
                K = k,
                ARaw = aBytes.Slice((int)(ia * mk), (int)mk),
                Lda = k,
                AU8 = aU8,
                BPacked = packed,
                BSums = sums,
                ZaScalar = za,
                ZaVec = zaVec,
                ZbScalar = zb,
                ZbVec = zbVec,
                Ldc = n,
                Mult = mult,
                MultVec = multVec,
                Bias = bias,
                BiasPerCol = true,
                QuantUnsigned = quantUnsigned,
                QuantZero = quantZero,
                Threads = run.Threads,
            };
            call.BiasFloat = biasFloat;
            if (_float)
            {
                call.OutFloat = y.MutableData<float>().Slice((int)(bi * mn), (int)mn);
            }
            else if (_q)
            {
                call.OutQuant = QuantParams.MutableBytes(in y).Slice((int)(bi * mn), (int)mn);
            }
            else
            {
                call.OutInt32 = y.MutableData<int>().Slice((int)(bi * mn), (int)mn);
            }

            Int8Matmul.Run(ref call);
        }
    }

    // The linear batch index of an operand with batch rank <rank> inside an output batch of rank <outRank>, broadcasting size-1 axes.
    private static long BatchOffset(Shape s, int rank, int outRank, ReadOnlySpan<int> idx)
    {
        long off = 0;
        for (int d = 0; d < rank; d++)
        {
            int outAxis = outRank - rank + d;
            int i = s[d] == 1 ? 0 : idx[outAxis];
            off = (off * s[d]) + i;
        }

        return off;
    }
}
