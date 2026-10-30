using System;
using System.Numerics;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

// The pattern for operators built on the blocked matrix product: the shape rules live here, the arithmetic in Kernels/Gemm.cs, which
// takes the mode (fast or exact) and the path from the session.

/// <summary>MatMul: NumPy matrix product with batch broadcasting; 1-D operands are promoted and the added dimension removed.</summary>
internal sealed class MatMulOp : OpKernel
{
    public static readonly OpDef Def = new("MatMul", 1, int.MaxValue, static _ => new MatMulOp());

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects two inputs");
        }

        DType t = ctx.InputType(0);
        if (t != ctx.InputType(1))
        {
            throw ctx.Fail($"the inputs have different element types ({t} and {ctx.InputType(1)})");
        }

        if (t is not (DType.F32 or DType.F64 or DType.I32 or DType.I64 or DType.U32 or DType.U64))
        {
            throw ctx.Unsupported($"element type {t} is not supported yet");
        }

        Dims(ctx.InputShape(0), ctx.InputShape(1), out int m, out int k, out int n, out Shape batchA, out Shape batchB, ctx);
        if (!Shape.TryBroadcast(batchA, batchB, out Shape batch))
        {
            throw ctx.Fail($"batch shapes {batchA} and {batchB} cannot be broadcast together");
        }

        Span<int> d = stackalloc int[Shape.MaxRank + 2];
        int r = 0;
        for (int i = 0; i < batch.Rank; i++)
        {
            d[r++] = batch[i];
        }

        if (ctx.InputShape(0).Rank > 1)
        {
            d[r++] = m;
        }

        if (ctx.InputShape(1).Rank > 1)
        {
            d[r++] = n;
        }

        if (r > Shape.MaxRank)
        {
            throw ctx.Fail($"the result would have {r} dimensions; the limit is {Shape.MaxRank}");
        }

        ctx.SetOutput(0, t, Shape.Create(d.Slice(0, r)));
        _ = k;
    }

    // Splits the operand shapes into [batch..., M, K] and [batch..., K, N].
    private static void Dims(Shape a, Shape b, out int m, out int k, out int n, out Shape batchA, out Shape batchB, InferContext ctx)
    {
        if (a.Rank == 0 || b.Rank == 0)
        {
            throw ctx.Fail("operands must have at least one dimension");
        }

        int ka;
        if (a.Rank == 1)
        {
            m = 1;
            ka = a[0];
            batchA = default;
        }
        else
        {
            m = a[a.Rank - 2];
            ka = a[a.Rank - 1];
            batchA = Shape.Create(a.Dimensions.Slice(0, a.Rank - 2));
        }

        int kb;
        if (b.Rank == 1)
        {
            kb = b[0];
            n = 1;
            batchB = default;
        }
        else
        {
            kb = b[b.Rank - 2];
            n = b[b.Rank - 1];
            batchB = Shape.Create(b.Dimensions.Slice(0, b.Rank - 2));
        }

        if (ka != kb)
        {
            throw ctx.Fail($"inner dimensions do not match: {a} times {b}");
        }

        k = ka;
    }

    public override void Run(in OpRun run)
    {
        Tensor a = run.Inputs[0], b = run.Inputs[1], c = run.Outputs[0];
        Shape sa = a.Shape, sb = b.Shape;
        int m = sa.Rank == 1 ? 1 : sa[sa.Rank - 2];
        int k = sa[sa.Rank - 1];
        int n = sb.Rank == 1 ? 1 : sb[sb.Rank - 1];
        int batchRankA = Math.Max(0, sa.Rank - 2), batchRankB = Math.Max(0, sb.Rank - 2);
        long batchA = 1, batchB = 1;
        for (int i = 0; i < batchRankA; i++)
        {
            batchA *= sa[i];
        }

        for (int i = 0; i < batchRankB; i++)
        {
            batchB *= sb[i];
        }

        long batchOut = (long)c.Length / Math.Max(1, (long)m * n);
        if ((long)m * n == 0)
        {
            return;
        }

        switch (c.DType)
        {
            case DType.F32:
                RunFloat(run, a.Data<float>(), b.Data<float>(), c.MutableData<float>(), sa, sb, m, k, n, batchA, batchB, batchOut);
                break;
            case DType.F64:
                RunNaive(a.Data<double>(), b.Data<double>(), c.MutableData<double>(), sa, sb, m, k, n);
                break;
            case DType.I32:
                RunNaive(a.Data<int>(), b.Data<int>(), c.MutableData<int>(), sa, sb, m, k, n);
                break;
            case DType.I64:
                RunNaive(a.Data<long>(), b.Data<long>(), c.MutableData<long>(), sa, sb, m, k, n);
                break;
            case DType.U32:
                RunNaive(a.Data<uint>(), b.Data<uint>(), c.MutableData<uint>(), sa, sb, m, k, n);
                break;
            case DType.U64:
                RunNaive(a.Data<ulong>(), b.Data<ulong>(), c.MutableData<ulong>(), sa, sb, m, k, n);
                break;
            default:
                throw new InvalidOperationException($"Element type {c.DType} reached the kernel although inference rejects it.");
        }
    }

    private static void RunFloat(in OpRun run, ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, Shape sa, Shape sb,
        int m, int k, int n, long batchA, long batchB, long batchOut)
    {
        // A shared B (no batch dimensions on B): the batch folds into the rows of A and one product does it all.
        if (batchB == 1 && sa.Rank >= 2)
        {
            Gemm.Run(run.KernelMode, run.Path, checked((int)(batchOut * m)), n, k, a, k, b, n, c, n, run.Threads);
            return;
        }

        // General case: one product per output batch, with each operand's batch index found by broadcasting.
        int rank = Math.Max(Math.Max(sa.Rank, sb.Rank) - 2, 0);
        Span<int> ba = stackalloc int[Shape.MaxRank], bb = stackalloc int[Shape.MaxRank], bo = stackalloc int[Shape.MaxRank];
        FillBatchDims(sa, rank, ba);
        FillBatchDims(sb, rank, bb);
        for (int i = 0; i < rank; i++)
        {
            bo[i] = Math.Max(ba[i], bb[i]);
        }

        Span<int> idx = stackalloc int[Shape.MaxRank];
        for (long batch = 0; batch < batchOut; batch++)
        {
            long offA = 0, offB = 0, strideA = (long)m * k, strideB = (long)k * n;
            long accA = 1, accB = 1;
            for (int d = rank - 1; d >= 0; d--)
            {
                if (ba[d] != 1)
                {
                    offA += idx[d] * accA * strideA;
                }

                if (bb[d] != 1)
                {
                    offB += idx[d] * accB * strideB;
                }

                accA *= ba[d];
                accB *= bb[d];
            }

            Gemm.Run(run.KernelMode, run.Path, m, n, k, a.Slice((int)offA, m * k), k, b.Slice((int)offB, k * n), n,
                c.Slice((int)(batch * m * n), m * n), n, run.Threads);
            for (int d = rank - 1; d >= 0; d--)
            {
                if (++idx[d] < bo[d])
                {
                    break;
                }

                idx[d] = 0;
            }
        }
    }

    // The batch extents of a shape, right-aligned into `rank` slots (1 where the operand has no such dimension).
    private static void FillBatchDims(Shape s, int rank, Span<int> dst)
    {
        int own = Math.Max(0, s.Rank - 2);
        for (int i = 0; i < rank; i++)
        {
            int j = i - (rank - own);
            dst[i] = j >= 0 ? s[j] : 1;
        }
    }

    // Reference-style product for the element types the blocked kernel does not cover. Integers wrap around.
    private static void RunNaive<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> c, Shape sa, Shape sb, int m, int k, int n)
        where T : unmanaged, INumber<T>
    {
        int rank = Math.Max(Math.Max(sa.Rank, sb.Rank) - 2, 0);
        Span<int> ba = stackalloc int[Shape.MaxRank], bb = stackalloc int[Shape.MaxRank], idx = stackalloc int[Shape.MaxRank];
        FillBatchDims(sa, rank, ba);
        FillBatchDims(sb, rank, bb);
        long batchOut = 1;
        for (int i = 0; i < rank; i++)
        {
            batchOut *= Math.Max(ba[i], bb[i]);
        }

        for (long batch = 0; batch < batchOut; batch++)
        {
            long offA = 0, offB = 0, accA = 1, accB = 1;
            for (int d = rank - 1; d >= 0; d--)
            {
                if (ba[d] != 1)
                {
                    offA += idx[d] * accA * m * k;
                }

                if (bb[d] != 1)
                {
                    offB += idx[d] * accB * k * n;
                }

                accA *= ba[d];
                accB *= bb[d];
            }

            long offC = batch * m * n;
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    T acc = T.Zero;
                    for (int p = 0; p < k; p++)
                    {
                        acc = unchecked(acc + (a[(int)(offA + (i * k) + p)] * b[(int)(offB + (p * n) + j)]));
                    }

                    c[(int)(offC + (i * n) + j)] = acc;
                }
            }

            for (int d = rank - 1; d >= 0; d--)
            {
                if (++idx[d] < Math.Max(ba[d], bb[d]))
                {
                    break;
                }

                idx[d] = 0;
            }
        }
    }
}

/// <summary>
/// Gemm: Y = alpha * op(A) * op(B) + beta * C, with op transposing when transA or transB is set and C broadcast to the result.
/// C is optional from opset 11.
/// </summary>
internal sealed class GemmOp : OpKernel
{
    public static readonly OpDef Def = new("Gemm", 7, int.MaxValue, static n => new GemmOp(n));

    private readonly float _alpha;
    private readonly float _beta;
    private readonly bool _transA;
    private readonly bool _transB;

    private GemmOp(NodeInfo node)
    {
        _alpha = node.GetFloat("alpha", 1f);
        _beta = node.GetFloat("beta", 1f);
        _transA = node.GetInt("transA", 0) != 0;
        _transB = node.GetInt("transB", 0) != 0;
        if (node.Opset < 11 && !node.HasInput(2))
        {
            throw node.Error("input C is required before opset 11");
        }
    }

    internal float Alpha => _alpha;

    internal float Beta => _beta;

    internal bool TransA => _transA;

    internal bool TransB => _transB;

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects inputs A and B");
        }

        DType t = ctx.InputType(0);
        if (t != ctx.InputType(1) || (ctx.HasInput(2) && ctx.InputType(2) != t))
        {
            throw ctx.Fail("the inputs have different element types");
        }

        if (t != DType.F32)
        {
            throw ctx.Unsupported($"element type {t} is not supported yet");
        }

        Shape a = ctx.InputShape(0), b = ctx.InputShape(1);
        if (a.Rank != 2 || b.Rank != 2)
        {
            throw ctx.Fail($"A and B must be matrices, got {a} and {b}");
        }

        int m = _transA ? a[1] : a[0], ka = _transA ? a[0] : a[1];
        int kb = _transB ? b[1] : b[0], n = _transB ? b[0] : b[1];
        if (ka != kb)
        {
            throw ctx.Fail($"inner dimensions do not match: {a}{(_transA ? "'" : "")} times {b}{(_transB ? "'" : "")}");
        }

        if (ctx.HasInput(2))
        {
            Shape c = ctx.InputShape(2);
            if (c.Rank > 2 || (c.Rank >= 1 && c[c.Rank - 1] != 1 && c[c.Rank - 1] != n) || (c.Rank == 2 && c[0] != 1 && c[0] != m))
            {
                throw ctx.Fail($"C of shape {c} cannot be broadcast to [{m}, {n}]");
            }
        }

        ctx.SetOutput(0, t, Shape.Create(m, n));
    }

    // A weight that arrives transposed is transposed once per plan instead of on every run.
    private sealed class Prepared
    {
        public Tensor A;
        public Tensor B;
    }

    public override object? Prepare(PrepareContext ctx)
    {
        bool a = _transA && ctx.IsConstant(0), b = _transB && ctx.IsConstant(1);
        if (!a && !b)
        {
            return null;
        }

        var p = new Prepared();
        if (a)
        {
            Tensor src = ctx.Constant(0);
            p.A = ctx.CreateOwned(DType.F32, Shape.Create(src.Shape[1], src.Shape[0]));
            Transpose(src.Data<float>(), p.A.MutableData<float>(), src.Shape[0], src.Shape[1]);
        }

        if (b)
        {
            Tensor src = ctx.Constant(1);
            p.B = ctx.CreateOwned(DType.F32, Shape.Create(src.Shape[1], src.Shape[0]));
            Transpose(src.Data<float>(), p.B.MutableData<float>(), src.Shape[0], src.Shape[1]);
        }

        return p;
    }

    public override void Run(in OpRun run)
    {
        Tensor a = run.Inputs[0], b = run.Inputs[1], y = run.Outputs[0];
        var prepared = run.State as Prepared;
        int m = y.Shape[0], n = y.Shape[1];
        int k = _transA ? a.Shape[0] : a.Shape[1];
        if (m == 0 || n == 0)
        {
            return;
        }

        ReadOnlySpan<float> aData = a.Data<float>(), bData = b.Data<float>();
        if (prepared is { A.IsEmpty: false })
        {
            aData = prepared.A.Data<float>();
        }
        else if (_transA)
        {
            Tensor t = run.Scratch(DType.F32, Shape.Create(m, k));
            Transpose(aData, t.MutableData<float>(), k, m);
            aData = t.Data<float>();
        }

        if (prepared is { B.IsEmpty: false })
        {
            bData = prepared.B.Data<float>();
        }
        else if (_transB)
        {
            Tensor t = run.Scratch(DType.F32, Shape.Create(k, n));
            Transpose(bData, t.MutableData<float>(), n, k);
            bData = t.Data<float>();
        }

        Span<float> yData = y.MutableData<float>();
        Gemm.Run(run.KernelMode, run.Path, m, n, k, aData, k, bData, n, yData, n, run.Threads);

        Tensor c = run.Inputs.Length > 2 ? run.Inputs[2] : default;
        if (_alpha == 1f && c.IsEmpty)
        {
            return;
        }

        // The epilogue: separate multiplies and an add, never fused, so both modes round the same way.
        bool canon = run.Exact;
        ReadOnlySpan<float> cData = c.IsEmpty ? default : c.Data<float>();
        int cRows = c.IsEmpty ? 0 : (c.Rank == 2 ? c.Shape[0] : 1);
        int cCols = c.IsEmpty ? 0 : (c.Rank == 0 ? 1 : c.Shape[c.Rank - 1]);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float v = yData[(i * n) + j];
                if (_alpha != 1f)
                {
                    v = _alpha * v;
                }

                if (!c.IsEmpty)
                {
                    float cv = cData[((cRows == 1 ? 0 : i) * cCols) + (cCols == 1 ? 0 : j)];
                    v += _beta == 1f ? cv : _beta * cv;
                }

                yData[(i * n) + j] = canon && float.IsNaN(v) ? Kernels.Exact.ExactElementwise.CanonicalNaN : v;
            }
        }
    }

    // dst[c, r] = src[r, c] for src of rows x cols (so dst is cols x rows), in square tiles so both sides stay in cache.
    private static void Transpose(ReadOnlySpan<float> src, Span<float> dst, int rows, int cols)
    {
        const int Tile = 32;
        for (int r0 = 0; r0 < rows; r0 += Tile)
        {
            int r1 = Math.Min(rows, r0 + Tile);
            for (int c0 = 0; c0 < cols; c0 += Tile)
            {
                int c1 = Math.Min(cols, c0 + Tile);
                for (int r = r0; r < r1; r++)
                {
                    for (int c = c0; c < c1; c++)
                    {
                        dst[(c * rows) + r] = src[(r * cols) + c];
                    }
                }
            }
        }
    }
}
