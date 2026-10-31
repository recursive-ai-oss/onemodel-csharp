using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Quant;

// Internal operators the QDQ fusion pass builds from chains of DequantizeLinear, a float operator and QuantizeLinear. They are not
// ONNX operators. Each computes what its chain computes (same float32 steps in the same order, apart from the global average, whose
// sum is exact) but never writes the float tensors in between.

/// <summary>
/// A table lookup on an 8-bit tensor: inputs x (uint8 or int8) and a 256-entry table (uint8 or int8, the output type). The table is
/// built by the pass by running the chain's own kernels over all 256 input values, so <c>y[i] = table[byte(x[i])]</c> is the chain's result.
/// </summary>
internal sealed class QLutOp : OpKernel
{
    public static readonly OpDef Def = new("QLut", 1, int.MaxValue, static _ => new QLutOp());

    public override int InPlaceInput => 0;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 2 || !ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects x and a 256-entry table");
        }

        if (ctx.InputType(0) is not (DType.U8 or DType.I8) || ctx.InputType(1) is not (DType.U8 or DType.I8) || ctx.InputShape(1).Length != 256)
        {
            throw ctx.Unsupported("needs an 8-bit input and an 8-bit table of 256 entries");
        }

        ctx.SetOutput(0, ctx.InputType(1), ctx.InputShape(0));
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        if (y.Length == 0)
        {
            return;
        }

        Int8Lut.Apply(QuantParams.Bytes(in x), QuantParams.Bytes(run.Inputs[1]), QuantParams.MutableBytes(in y), run.Threads);
    }
}

/// <summary>
/// Product of two 8-bit quantized tensors into an 8-bit quantized tensor: the inputs of <see cref="QLinearAddOp"/>. The common case,
/// a gate with one value per row (<c>[N, C, 1, 1]</c> times <c>[N, C, H, W]</c>), runs the vector kernel
/// (<see cref="Int8ChannelMul"/>); other broadcasts use a plain element loop.
/// </summary>
internal sealed class QLinearMulOp : OpKernel
{
    public static readonly OpDef Def = new("QLinearMul", 1, int.MaxValue, static _ => new QLinearMulOp());

    public override void Infer(InferContext ctx) => QLinearAddOp.InferBinary(ctx);

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
        if (a.Shape == y.Shape && b.Length < a.Length && Rows(b.Shape, y.Shape, out long outer, out long inner))
        {
            Int8ChannelMul.Run(ab, bb, yb, outer, inner, in p, run.Threads);
            return;
        }

        if (b.Shape == y.Shape && a.Length < b.Length && Rows(a.Shape, y.Shape, out outer, out inner))
        {
            // The product commutes in float32, so the per-row operand can be either one; swap the parameters with it.
            var q = new Int8AddParams(p.BUnsigned, p.BZero, p.BScale, p.AUnsigned, p.AZero, p.AScale, p.YUnsigned, p.YZero, p.YScale);
            Int8ChannelMul.Run(bb, ab, yb, outer, inner, in q, run.Threads);
            return;
        }

        QLinearAddOp.Broadcast(a.Shape, b.Shape, y.Shape, ab, bb, yb, in p, multiply: true);
    }

    // Whether the small operand, broadcast over the output, has one value per row: its dimensions (right-aligned) match the output's up
    // to some axis and are 1 after it. Then the output is `outer` rows of `inner` elements and row o uses small[o].
    private static bool Rows(Shape small, Shape output, out long outer, out long inner)
    {
        outer = 1;
        inner = 1;
        int pad = output.Rank - small.Rank;
        if (pad < 0)
        {
            return false;
        }

        int last = -1;
        for (int d = 0; d < small.Rank; d++)
        {
            if (small[d] != 1)
            {
                last = d + pad;
            }
        }

        for (int d = 0; d < output.Rank; d++)
        {
            int s = d >= pad ? small[d - pad] : 1;
            if (d <= last)
            {
                if (s != output[d])
                {
                    return false;
                }

                outer *= output[d];
            }
            else
            {
                if (s != 1)
                {
                    return false;
                }

                inner *= output[d];
            }
        }

        return true;
    }
}

/// <summary>
/// Global average pooling of an 8-bit tensor into an 8-bit tensor: inputs x, x_scale, x_zero_point, y_scale, y_zero_point.
/// The mean is taken over the integers (exactly) and scaled once, so it can differ from the float chain's by the rounding of the
/// float sum, which the chain does in a different order anyway.
/// </summary>
internal sealed class QLinearGlobalAveragePoolOp : OpKernel
{
    public static readonly OpDef Def = new("QLinearGlobalAveragePool", 1, int.MaxValue, static _ => new QLinearGlobalAveragePoolOp());

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 5)
        {
            throw ctx.Fail("expects x, x_scale, x_zero_point, y_scale and y_zero_point");
        }

        if (ctx.InputType(0) is not (DType.U8 or DType.I8) || ctx.InputType(2) != ctx.InputType(0) || ctx.InputType(4) is not (DType.U8 or DType.I8))
        {
            throw ctx.Unsupported("needs an 8-bit input with a zero point of its type and an 8-bit output zero point");
        }

        foreach (int i in new[] { 1, 3 })
        {
            if (ctx.InputType(i) != DType.F32 || ctx.InputShape(i).Length != 1)
            {
                throw ctx.Unsupported($"input {i} must be a single float32 scale");
            }
        }

        if (ctx.InputShape(2).Length != 1 || ctx.InputShape(4).Length != 1)
        {
            throw ctx.Unsupported("the zero points must be single values");
        }

        Shape x = ctx.InputShape(0);
        if (x.Rank < 2)
        {
            throw ctx.Fail($"the input {x} must have at least 2 dimensions");
        }

        Span<int> d = stackalloc int[Shape.MaxRank];
        d[0] = x[0];
        d[1] = x[1];
        for (int i = 2; i < x.Rank; i++)
        {
            d[i] = 1;
            if (x[i] == 0)
            {
                throw ctx.Fail($"the input {x} has an empty spatial axis");
            }
        }

        ctx.SetOutput(0, ctx.InputType(4), Shape.Create(d[..x.Rank]));
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        if (y.Length == 0)
        {
            return;
        }

        long planes = (long)x.Shape[0] * x.Shape[1];
        int plane = x.Length / (x.Shape[0] * x.Shape[1]);
        var p = new Int8Pool.AverageParams(
            x.DType == DType.U8, QuantParams.ZeroAt(run.Inputs[2], 0), run.Inputs[1].Data<float>()[0],
            y.DType == DType.U8, QuantParams.ZeroAt(run.Inputs[4], 0), run.Inputs[3].Data<float>()[0]);
        Int8Pool.GlobalAverage(QuantParams.Bytes(in x), QuantParams.MutableBytes(in y), planes, plane, in p, run.Threads);
    }
}
