using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Transformer;

/// <summary>
/// RotaryEmbedding (opset 23): rotates pairs of features of each head by position-dependent angles. The input is (batch, heads, seq,
/// head_size), or (batch, seq, hidden) with the num_heads attribute. Only the first rotary_embedding_dim features (default all)
/// are rotated. The caches hold cos and sin for each position and each pair: (max_position, dim / 2) addressed by position_ids, or
/// (batch, seq, dim / 2) when there is no position_ids. Non-interleaved pairs feature i with i + dim / 2; interleaved pairs 2i with 2i + 1.
/// With pair values (a, b) and the angle's cos c and sin s the results are (c a - s b, s a + c b), as two rounded products and a
/// rounded difference or sum, so the operator is exact.
/// </summary>
internal sealed class RotaryEmbeddingOp : OpKernel
{
    public static readonly OpDef Def = new("RotaryEmbedding", 23, int.MaxValue, static n => new RotaryEmbeddingOp(n));

    private readonly bool _interleaved;
    private readonly int _numHeads, _rotaryDim;

    private RotaryEmbeddingOp(NodeInfo n)
    {
        _interleaved = n.GetInt("interleaved", 0) != 0;
        _numHeads = (int)n.GetInt("num_heads", 0);
        _rotaryDim = (int)n.GetInt("rotary_embedding_dim", 0);
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 3 || !ctx.HasInput(0) || !ctx.HasInput(1) || !ctx.HasInput(2))
        {
            throw ctx.Fail("expects the input and the cos and sin caches");
        }

        if (ctx.InputType(0) != DType.F32 || ctx.InputType(1) != DType.F32 || ctx.InputType(2) != DType.F32)
        {
            throw ctx.Unsupported("only float32 is supported yet");
        }

        Shape x = ctx.InputShape(0);
        if (x.Rank is not (3 or 4) || (x.Rank == 3 && _numHeads <= 0))
        {
            throw ctx.Fail("the input must be 4-D, or 3-D with the num_heads attribute");
        }

        if (ctx.HasInput(3) && ctx.InputType(3) != DType.I64)
        {
            throw ctx.Fail("position_ids must be int64");
        }

        int headSize = x.Rank == 3 ? x[2] / _numHeads : x[3];
        int dim = _rotaryDim > 0 ? _rotaryDim : headSize;
        if (dim > headSize || dim % 2 != 0 || ctx.InputShape(1)[ctx.InputShape(1).Rank - 1] != dim / 2 || ctx.InputShape(2) != ctx.InputShape(1))
        {
            throw ctx.Fail("the caches must have rotary_embedding_dim / 2 columns and equal shapes");
        }

        ctx.SetOutput(0, DType.F32, x);
    }

    public override void Run(in OpRun run)
    {
        Tensor xt = run.Inputs[0];
        bool threeD = xt.Rank == 3;
        int batch = xt.Shape[0];
        int heads = threeD ? _numHeads : xt.Shape[1];
        int seq = threeD ? xt.Shape[1] : xt.Shape[2];
        int headSize = threeD ? xt.Shape[2] / heads : xt.Shape[3];
        int dim = _rotaryDim > 0 ? _rotaryDim : headSize, half = dim / 2;
        ReadOnlySpan<float> x = xt.Data<float>(), cos = run.Inputs[1].Data<float>(), sin = run.Inputs[2].Data<float>();
        ReadOnlySpan<long> pos = run.Inputs.Length > 3 && !run.Inputs[3].IsEmpty ? run.Inputs[3].Data<long>() : default;
        int maxPos = pos.IsEmpty ? 0 : run.Inputs[1].Shape[0];
        Span<float> y = run.Outputs[0].MutableData<float>();
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seq; s++)
            {
                int cacheRow;
                if (pos.IsEmpty)
                {
                    cacheRow = (b * seq) + s;
                }
                else
                {
                    long p = pos[(b * seq) + s];
                    if ((ulong)p >= (ulong)maxPos)
                    {
                        throw new SessionException(SessionErrorKind.InvalidInput, $"RotaryEmbedding: position {p} is outside the cache of {maxPos} positions");
                    }

                    cacheRow = (int)p;
                }

                for (int h = 0; h < heads; h++)
                {
                    int at = threeD ? (((b * seq) + s) * heads * headSize) + (h * headSize) : ((((b * heads) + h) * seq) + s) * headSize;
                    ReadOnlySpan<float> xi = x.Slice(at, headSize);
                    Span<float> yi = y.Slice(at, headSize);
                    for (int i = 0; i < half; i++)
                    {
                        float c = cos[(cacheRow * half) + i], sn = sin[(cacheRow * half) + i];
                        int ia = _interleaved ? 2 * i : i, ib = _interleaved ? (2 * i) + 1 : i + half;
                        float a = xi[ia], bb = xi[ib];
                        yi[ia] = (c * a) - (sn * bb);
                        yi[ib] = (sn * a) + (c * bb);
                    }

                    // Features beyond the rotated ones pass through.
                    xi.Slice(dim).CopyTo(yi.Slice(dim));
                }
            }
        }
    }
}
