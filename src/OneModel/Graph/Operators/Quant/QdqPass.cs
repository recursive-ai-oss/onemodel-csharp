using System;
using System.Collections.Generic;
using System.Threading;
using RecursiveAI.OneModel.Graph.Passes;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Quant;

/// <summary>
/// Rewrites the quantize-dequantize (QDQ) form of a quantized model to integer operators. A convolution or Gemm written as
/// <c>DequantizeLinear (input), DequantizeLinear (weights) [, DequantizeLinear (bias)] -> Conv/Gemm -> QuantizeLinear</c> becomes one
/// QLinearConv or QLinearMatMul that reads the 8-bit input directly, accumulates in exact 32-bit integers and requantizes
/// (Jacob et al. 2018), so the activations stay 8-bit between layers and the arithmetic runs on the integer kernels.
/// </summary>
/// <remarks>
/// Not bit-preserving: the original graph sums in float32 inside the convolution, the rewrite sums exactly in integers, so
/// a value that sat close to a rounding boundary can land one step away. The pass is therefore skipped in exact mode, where the
/// QDQ graph runs as written (with the exact float kernels). A node is rewritten only when every scale and zero point is a
/// constant, the weights are constant 8-bit tensors, and the optional int32 bias carries the scale (input scale times weight
/// scale) that quantization tools give it; anything else is left as written. DequantizeLinear nodes the rewrite no longer reads
/// are removed with the dead nodes; those still read by another consumer stay.
/// </remarks>
internal sealed class QdqPass : IGraphPass
{
    public string Name => "qdq";

    public OptimizationLevel MinLevel => OptimizationLevel.All;

    public bool PreservesBits => false;

    // What one DequantizeLinear contributes to a match: the quantized value and its parameters (value ids).
    private readonly record struct Dq(int Source, int Scale, int Zero, int Axis);

    public void Run(PassGraph g, CancellationToken ct)
    {
        int[] producer = g.Producers();
        int[] uses = g.UseCounts();
        var isOutput = new HashSet<int>(g.OutputIds);
        // For every value read by exactly one node, that node's index.
        var sole = new int[g.ValueCount];
        Array.Fill(sole, -1);
        for (int i = 0; i < g.Nodes.Count; i++)
        {
            foreach (int id in g.Nodes[i].Inputs)
            {
                if (id >= 0 && uses[id] == 1)
                {
                    sole[id] = i;
                }
            }
        }

        var replaced = new Dictionary<int, CompiledNode>();
        var absorbed = new HashSet<int>();
        for (int i = 0; i < g.Nodes.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            CompiledNode n = g.Nodes[i];
            if (n.Outputs.Length != 1 || n.Outputs[0] < 0 || isOutput.Contains(n.Outputs[0]) || n.Inputs.Length < (n.Kernel is MaxPoolOp ? 1 : 2))
            {
                continue;
            }

            if (n.Kernel is not (ConvOp or GemmOp or AddOp or MaxPoolOp) || sole[n.Outputs[0]] < 0 || g.Nodes[sole[n.Outputs[0]]].Kernel is not QuantizeLinearOp)
            {
                continue;
            }

            int qIndex = sole[n.Outputs[0]];
            CompiledNode q = g.Nodes[qIndex];
            CompiledNode? repl = n.Kernel is ConvOp ? TryConv(g, n, q, producer) : n.Kernel is GemmOp ? TryGemm(g, n, q, producer) : n.Kernel is AddOp ? TryAdd(g, n, q, producer) : TryMaxPool(g, n, q, producer);
            if (repl is not null)
            {
                replaced[i] = repl;
                absorbed.Add(qIndex);
            }
        }

        if (replaced.Count == 0)
        {
            return;
        }

        var result = new List<CompiledNode>(g.Nodes.Count);
        for (int i = 0; i < g.Nodes.Count; i++)
        {
            if (!absorbed.Contains(i))
            {
                result.Add(replaced.TryGetValue(i, out CompiledNode? r) ? r : g.Nodes[i]);
            }
        }

        g.Nodes.Clear();
        g.Nodes.AddRange(result);
    }

    private static bool TryDq(PassGraph g, int[] producer, int valueId, out Dq dq)
    {
        dq = default;
        if (valueId < 0 || producer[valueId] < 0)
        {
            return false;
        }

        CompiledNode d = g.Nodes[producer[valueId]];
        if (d.Kernel is not DequantizeLinearOp || d.Inputs.Length < 2 || d.Inputs[1] < 0)
        {
            return false;
        }

        dq = new Dq(d.Inputs[0], d.Inputs[1], d.Inputs.Length > 2 ? d.Inputs[2] : -1, (int)d.Info.GetInt("axis", 1));
        return true;
    }

    private static bool IsEightBit(Tensor t) => t.DType is DType.U8 or DType.I8;

    // A per-tensor 8-bit zero point constant: its id and type.
    private static bool ScalarZero(PassGraph g, int id, out DType type)
    {
        type = DType.Undefined;
        Tensor t = id >= 0 ? g.Constant(id) : default;
        if (t.IsEmpty || t.Length != 1 || !IsEightBit(t))
        {
            return false;
        }

        type = t.DType;
        return true;
    }

    private static bool ScalarScale(PassGraph g, int id)
    {
        Tensor t = id >= 0 ? g.Constant(id) : default;
        return !t.IsEmpty && t.DType == DType.F32 && t.Length == 1;
    }

    // The weight DQ: a constant 8-bit tensor whose scale and zero point are constants of the same length (1, or one per channel of
    // the axis that is the output channel).
    private static bool WeightDq(PassGraph g, in Dq w, int channels, int channelAxis, out Tensor weights)
    {
        weights = w.Source >= 0 ? g.Constant(w.Source) : default;
        Tensor s = w.Scale >= 0 ? g.Constant(w.Scale) : default;
        Tensor z = w.Zero >= 0 ? g.Constant(w.Zero) : default;
        if (weights.IsEmpty || !IsEightBit(weights) || s.IsEmpty || s.DType != DType.F32 || z.IsEmpty || z.DType != weights.DType ||
            !z.Shape.Equals(s.Shape))
        {
            return false;
        }

        if (s.Length == 1)
        {
            return s.Rank <= 1;
        }

        return s.Rank == 1 && s.Length == channels && w.Axis == channelAxis;
    }

    // The bias DQ: a constant int32 vector, zero point absent or zero, scale = input scale * weight scale for every channel.
    private static bool BiasDq(PassGraph g, in Dq b, int channels, float xScale, ReadOnlySpan<float> wScale, out Tensor bias)
    {
        bias = b.Source >= 0 ? g.Constant(b.Source) : default;
        Tensor s = b.Scale >= 0 ? g.Constant(b.Scale) : default;
        if (bias.IsEmpty || bias.DType != DType.I32 || bias.Rank != 1 || bias.Length != channels || s.IsEmpty || s.DType != DType.F32)
        {
            return false;
        }

        if (b.Zero >= 0)
        {
            Tensor z = g.Constant(b.Zero);
            if (z.IsEmpty || z.DType != DType.I32)
            {
                return false;
            }

            foreach (int v in z.Data<int>())
            {
                if (v != 0)
                {
                    return false;
                }
            }
        }

        ReadOnlySpan<float> bs = s.Data<float>();
        if (bs.Length != 1 && bs.Length != channels)
        {
            return false;
        }

        for (int c = 0; c < channels; c++)
        {
            float want = xScale * wScale[wScale.Length == 1 ? 0 : c];
            float have = bs[bs.Length == 1 ? 0 : c];
            if (!(MathF.Abs(have - want) <= 1e-5f * MathF.Abs(want)))
            {
                return false;
            }
        }

        return true;
    }

    // DequantizeLinear, MaxPool, QuantizeLinear with the same single scale and zero point on both sides: quantizing what dequantizing
    // produced gives back the 8-bit value (a scale times a small integer, divided by the scale, rounds to that integer), and a
    // maximum commutes with a positive scale, so the pooling runs on the 8-bit values themselves.
    private static CompiledNode? TryMaxPool(PassGraph g, CompiledNode pool, CompiledNode q, int[] producer)
    {
        if (pool.Inputs.Length != 1 || !TryDq(g, producer, pool.Inputs[0], out Dq x) || x.Source < 0)
        {
            return null;
        }

        int qScale = q.Inputs.Length > 1 ? q.Inputs[1] : -1, qZero = q.Inputs.Length > 2 ? q.Inputs[2] : -1;
        if (!ScalarScale(g, x.Scale) || !ScalarZero(g, x.Zero, out DType zt) || !ScalarScale(g, qScale) || !ScalarZero(g, qZero, out DType qt) || zt != qt)
        {
            return null;
        }

        float scale = g.Constant(x.Scale).Data<float>()[0];
        if (!(scale > 0f) || !float.IsFinite(scale) || scale != g.Constant(qScale).Data<float>()[0] ||
            QuantParams.ZeroAt(g.Constant(x.Zero), 0) != QuantParams.ZeroAt(g.Constant(qZero), 0))
        {
            return null;
        }

        OpKernel kernel = g.Own(MaxPoolOp.Def.Create(pool.Info));
        return new CompiledNode(pool.Info, kernel, [x.Source], [q.Outputs[0]]);
    }

    // DequantizeLinear, DequantizeLinear, Add, QuantizeLinear with single scales and zero points: one quantized addition, which computes
    // the same float32 operations in the same order (so the pass is bit-neutral here), without writing the float tensors.
    private static CompiledNode? TryAdd(PassGraph g, CompiledNode add, CompiledNode q, int[] producer)
    {
        if (add.Inputs.Length != 2 || !TryDq(g, producer, add.Inputs[0], out Dq a) || !TryDq(g, producer, add.Inputs[1], out Dq b))
        {
            return null;
        }

        if (!ScalarScale(g, a.Scale) || !ScalarZero(g, a.Zero, out _) || !ScalarScale(g, b.Scale) || !ScalarZero(g, b.Zero, out _) ||
            !ScalarScale(g, q.Inputs.Length > 1 ? q.Inputs[1] : -1) || !ScalarZero(g, q.Inputs.Length > 2 ? q.Inputs[2] : -1, out _))
        {
            return null;
        }

        if (a.Source < 0 || b.Source < 0)
        {
            return null;
        }

        int[] inputs = [a.Source, a.Scale, a.Zero, b.Source, b.Scale, b.Zero, q.Inputs[1], q.Inputs[2]];
        OpKernel kernel = g.Own(QLinearAddOp.Def.Create(add.Info));
        return new CompiledNode(add.Info, kernel, inputs, [q.Outputs[0]]);
    }

    private static CompiledNode? TryConv(PassGraph g, CompiledNode conv, CompiledNode q, int[] producer)
    {
        if (conv.Inputs.Length > 3 || !TryDq(g, producer, conv.Inputs[0], out Dq x) || !TryDq(g, producer, conv.Inputs[1], out Dq w))
        {
            return null;
        }

        if (!ScalarScale(g, x.Scale) || !ScalarZero(g, x.Zero, out _) || !ScalarScale(g, q.Inputs.Length > 1 ? q.Inputs[1] : -1) ||
            !ScalarZero(g, q.Inputs.Length > 2 ? q.Inputs[2] : -1, out _))
        {
            return null;
        }

        Tensor wConst = w.Source >= 0 ? g.Constant(w.Source) : default;
        if (wConst.IsEmpty || wConst.Rank < 3 || !WeightDq(g, in w, wConst.Shape[0], 0, out Tensor weights))
        {
            return null;
        }

        int biasId = -1;
        if (conv.Inputs.Length == 3 && conv.Inputs[2] >= 0)
        {
            if (!TryDq(g, producer, conv.Inputs[2], out Dq b) ||
                !BiasDq(g, in b, weights.Shape[0], g.Constant(x.Scale).Data<float>()[0], g.Constant(w.Scale).Data<float>(), out _))
            {
                return null;
            }

            biasId = b.Source;
        }

        int[] inputs = biasId >= 0
            ? [x.Source, x.Scale, x.Zero, w.Source, w.Scale, w.Zero, q.Inputs[1], q.Inputs[2], biasId]
            : [x.Source, x.Scale, x.Zero, w.Source, w.Scale, w.Zero, q.Inputs[1], q.Inputs[2]];
        OpKernel kernel = g.Own(IntegerConvOp.QLinearConv.Create(conv.Info));
        return new CompiledNode(conv.Info, kernel, inputs, [q.Outputs[0]]);
    }

    private static CompiledNode? TryGemm(PassGraph g, CompiledNode gemm, CompiledNode q, int[] producer)
    {
        NodeInfo info = gemm.Info;
        if (info.GetFloat("alpha", 1f) != 1f || info.GetFloat("beta", 1f) != 1f || info.GetInt("transA", 0) != 0 || gemm.Inputs.Length > 3)
        {
            return null;
        }

        bool transB = info.GetInt("transB", 0) != 0;
        if (!TryDq(g, producer, gemm.Inputs[0], out Dq x) || !TryDq(g, producer, gemm.Inputs[1], out Dq w))
        {
            return null;
        }

        if (!ScalarScale(g, x.Scale) || !ScalarZero(g, x.Zero, out _) || !ScalarScale(g, q.Inputs.Length > 1 ? q.Inputs[1] : -1) ||
            !ScalarZero(g, q.Inputs.Length > 2 ? q.Inputs[2] : -1, out _))
        {
            return null;
        }

        Tensor wConst = w.Source >= 0 ? g.Constant(w.Source) : default;
        if (wConst.IsEmpty || wConst.Rank != 2)
        {
            return null;
        }

        int n = transB ? wConst.Shape[0] : wConst.Shape[1];
        int k = transB ? wConst.Shape[1] : wConst.Shape[0];
        if (!WeightDq(g, in w, n, transB ? 0 : 1, out Tensor weights))
        {
            return null;
        }

        int biasId = -1;
        if (gemm.Inputs.Length == 3 && gemm.Inputs[2] >= 0)
        {
            if (!TryDq(g, producer, gemm.Inputs[2], out Dq b) ||
                !BiasDq(g, in b, n, g.Constant(x.Scale).Data<float>()[0], g.Constant(w.Scale).Data<float>(), out _))
            {
                return null;
            }

            biasId = b.Source;
        }

        int wId = w.Source;
        if (transB)
        {
            // The matrix product wants B as [K, N]: transpose the 8-bit weights once, here.
            Tensor t = Tensor.Create(weights.DType, Shape.Create(k, n));
            ReadOnlySpan<byte> src = QuantParams.Bytes(in weights);
            Span<byte> dst = QuantParams.MutableBytes(in t);
            for (int r = 0; r < n; r++)
            {
                for (int c = 0; c < k; c++)
                {
                    dst[(c * n) + r] = src[(r * k) + c];
                }
            }

            wId = g.AddValue(g.NameOf(w.Source) + "_transposed", t);
        }

        int[] inputs = biasId >= 0
            ? [x.Source, x.Scale, x.Zero, wId, w.Scale, w.Zero, q.Inputs[1], q.Inputs[2], biasId]
            : [x.Source, x.Scale, x.Zero, wId, w.Scale, w.Zero, q.Inputs[1], q.Inputs[2]];
        OpKernel kernel = g.Own(IntegerMatMulOp.QLinearMatMul.Create(info));
        return new CompiledNode(info, kernel, inputs, [q.Outputs[0]]);
    }
}
