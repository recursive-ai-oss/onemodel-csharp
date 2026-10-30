using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Quant;

/// <summary>
/// Addition of two 8-bit quantized tensors into an 8-bit quantized tensor: inputs a, a_scale, a_zero_point, b, b_scale, b_zero_point,
/// y_scale, y_zero_point (all scales and zero points single values; the zero points are required). Built by the QDQ pass from
/// <c>DequantizeLinear, DequantizeLinear, Add, QuantizeLinear</c>; it is not an ONNX operator. The result is what that chain computes
/// (<see cref="Int8Elementwise"/>), but the float tensors between its nodes are never written.
/// </summary>
internal sealed class QLinearAddOp : OpKernel
{
    public static readonly OpDef Def = new("QLinearAdd", 1, int.MaxValue, static _ => new QLinearAddOp());

    public override void Infer(InferContext ctx) => InferBinary(ctx);

    /// <summary>The input rules and the broadcast output shape shared with <see cref="QLinearMulOp"/>.</summary>
    internal static void InferBinary(InferContext ctx)
    {
        if (ctx.InputCount < 8)
        {
            throw ctx.Fail("expects a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale and y_zero_point");
        }

        foreach (int i in new[] { 0, 3, 2, 5, 7 })
        {
            DType t = ctx.InputType(i);
            if (t is not (DType.U8 or DType.I8))
            {
                throw ctx.Unsupported($"input {i} has element type {t} (uint8 or int8 only)");
            }
        }

        if (ctx.InputType(0) != ctx.InputType(2) || ctx.InputType(3) != ctx.InputType(5))
        {
            throw ctx.Fail("a zero point's type must match its tensor's");
        }

        foreach (int i in new[] { 1, 4, 6 })
        {
            if (ctx.InputType(i) != DType.F32 || ctx.InputShape(i).Length != 1)
            {
                throw ctx.Unsupported($"input {i} must be a single float32 scale");
            }
        }

        foreach (int i in new[] { 2, 5, 7 })
        {
            if (ctx.InputShape(i).Length != 1)
            {
                throw ctx.Unsupported($"input {i} must be a single zero point");
            }
        }

        if (!Shape.TryBroadcast(ctx.InputShape(0), ctx.InputShape(3), out Shape s))
        {
            throw ctx.Fail($"shapes {ctx.InputShape(0)} and {ctx.InputShape(3)} cannot be broadcast together");
        }

        ctx.SetOutput(0, ctx.InputType(7), s);
    }

    public override void Run(in OpRun run)
    {
        Tensor a = run.Inputs[0], b = run.Inputs[3], y = run.Outputs[0];
        if (y.Length == 0)
        {
            return;
        }

        var p = new Int8AddParams(
            a.DType == DType.U8, QuantParams.ZeroAt(run.Inputs[2], 0), run.Inputs[1].Data<float>()[0],
            b.DType == DType.U8, QuantParams.ZeroAt(run.Inputs[5], 0), run.Inputs[4].Data<float>()[0],
            y.DType == DType.U8, QuantParams.ZeroAt(run.Inputs[7], 0), run.Inputs[6].Data<float>()[0]);
        ReadOnlySpan<byte> ab = QuantParams.Bytes(in a), bb = QuantParams.Bytes(in b);
        Span<byte> yb = QuantParams.MutableBytes(in y);
        if (a.Shape == y.Shape && b.Shape == y.Shape)
        {
            Int8Elementwise.Add8(ab, bb, yb, in p, run.Threads);
            return;
        }

        Broadcast(a.Shape, b.Shape, y.Shape, ab, bb, yb, in p);
    }

    // Different shapes (one side broadcast): a plain element loop with strides; not the case in the models this exists for.
    internal static void Broadcast(Shape sa, Shape sb, Shape sy, ReadOnlySpan<byte> a, ReadOnlySpan<byte> b, Span<byte> y, in Int8AddParams p, bool multiply = false)
    {
        int rank = sy.Rank;
        Span<long> strideA = stackalloc long[Shape.MaxRank];
        Span<long> strideB = stackalloc long[Shape.MaxRank];
        Span<int> index = stackalloc int[Shape.MaxRank];
        long ma = 1, mb = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            int da = d - (rank - sa.Rank), db = d - (rank - sb.Rank);
            strideA[d] = da >= 0 && sa[da] != 1 ? ma : 0;
            strideB[d] = db >= 0 && sb[db] != 1 ? mb : 0;
            ma *= da >= 0 ? sa[da] : 1;
            mb *= db >= 0 ? sb[db] : 1;
        }

        int lo = p.YUnsigned ? 0 : -128, hi = p.YUnsigned ? 255 : 127;
        long oa = 0, ob = 0;
        for (int i = 0; i < y.Length; i++)
        {
            y[i] = (byte)(multiply ? Int8Elementwise.MulOne(a[(int)oa], b[(int)ob], in p, lo, hi) : Int8Elementwise.AddOne(a[(int)oa], b[(int)ob], in p, lo, hi));
            for (int d = rank - 1; d >= 0; d--)
            {
                oa += strideA[d];
                ob += strideB[d];
                if (++index[d] < sy[d])
                {
                    break;
                }

                oa -= strideA[d] * sy[d];
                ob -= strideB[d] * sy[d];
                index[d] = 0;
            }
        }
    }
}
