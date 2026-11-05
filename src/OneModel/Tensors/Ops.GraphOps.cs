using System;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
namespace RecursiveAI.OneModel.Tensors;

// Operations that imported graphs need and that the layer ops above do not cover: batched matrix products, a few elementwise
// functions, select-by-condition and gather. Each records what its backward needs; docs/autograd.md lists the formulas.
public static partial class Ops
{
    // ---------------------------------------------------------------------------------------------------------------- batched matmul

    /// <summary>
    /// Matrix product over the last two axes with the leading axes broadcast, <c>a[..., m, k] · b[..., k, n]</c> (NumPy and ONNX
    /// <c>MatMul</c> semantics for rank two and up). A rank-2 <paramref name="b"/> against a higher-rank <paramref name="a"/> is
    /// one product over the flattened rows. Record (general case): slot 0 holds <c>a</c> when <c>b</c> requires a gradient, slot 1
    /// holds <c>b</c> when <c>a</c> does; each gradient is a batched product with the other operand's transpose, summed down to the
    /// operand's own shape.
    /// </summary>
    /// <param name="a">Left operand, rank at least 2, any strides.</param>
    /// <param name="b">Right operand, rank at least 2, any strides.</param>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    /// <returns>A tensor of shape <c>broadcast(batch(a), batch(b)) + [m, n]</c>.</returns>
    /// <exception cref="TensorShapeException">A rank below 2, mismatched inner extents, or batch axes that do not broadcast.</exception>
    public static Tensor MatMulBatched(in Tensor a, in Tensor b, BufferPool? pool = null)
    {
        RequireFloat32(a, b);
        if (a.Rank < 2 || b.Rank < 2)
        {
            throw new TensorShapeException($"MatMulBatched needs rank 2 or more, not {a.Shape} and {b.Shape}.");
        }

        int m = a.Shape[a.Rank - 2], k = a.Shape[a.Rank - 1], n = b.Shape[b.Rank - 1];
        if (b.Shape[b.Rank - 2] != k)
        {
            throw new TensorShapeException($"Cannot multiply {a.Shape} by {b.Shape}.");
        }

        if (b.Rank == 2)
        {
            if (a.Rank == 2)
            {
                return MatMul(a, b, pool);
            }

            // Rows of every batch share the one right operand: a single product over the flattened rows.
            Tensor flat = MatMul(a.Reshape(Shape.Create(a.Length / Math.Max(1, k), k)), b, pool);
            Span<int> od = stackalloc int[Shape.MaxRank];
            a.Shape.Dimensions.CopyTo(od);
            od[a.Rank - 1] = n;
            return flat.Reshape(Shape.Create(od.Slice(0, a.Rank)));
        }

        int rank = Math.Max(a.Rank, b.Rank);
        Span<int> bs = stackalloc int[Shape.MaxRank];
        for (int i = 0; i < rank - 2; i++)
        {
            int ia = i - (rank - a.Rank), ib = i - (rank - b.Rank);
            int da = ia >= 0 ? a.Shape[ia] : 1, db = ib >= 0 ? b.Shape[ib] : 1;
            if (da != db && da != 1 && db != 1)
            {
                throw new TensorShapeException($"The batch axes of {a.Shape} and {b.Shape} do not broadcast.");
            }

            bs[i] = da == 1 ? db : da;
        }

        Span<int> full = stackalloc int[Shape.MaxRank];
        bs.Slice(0, rank - 2).CopyTo(full);
        full[rank - 2] = m;
        full[rank - 1] = n;
        Tensor r = NewOut(Shape.Create(full.Slice(0, rank)), pool);
        using (GradMode.NoGradInternal())
        {
            bs[rank - 2] = m;
            bs[rank - 1] = k;
            Tensor ae = a.Expand(Shape.Create(bs.Slice(0, rank)));
            bs[rank - 2] = k;
            bs[rank - 1] = n;
            Tensor be = b.Expand(Shape.Create(bs.Slice(0, rank)));
            BatchLoop(ae, be, r, rank - 2);
        }

        if (!Recording(a, b))
        {
            return r;
        }

        BackwardRecord rec = BackwardRecord.Create(OpKind.BatchMatMul, 2, a, b);
        if (b.RequiresGrad)
        {
            rec.Save(0, a);
        }

        if (a.RequiresGrad)
        {
            rec.Save(1, b);
        }

        return rec.Attach(r);
    }

    private static void BatchLoop(in Tensor a, in Tensor b, in Tensor r, int batchAxes)
    {
        if (batchAxes == 0)
        {
            KernelSet.Default.MatMul(a, b, r);
            return;
        }

        int count = r.Shape[r.Rank - 2 - batchAxes];
        for (int i = 0; i < count; i++)
        {
            BatchLoop(a.Select(0, i), b.Select(0, i), r.Select(0, i), batchAxes - 1);
        }
    }

    // ---------------------------------------------------------------------------------------------------------------- elementwise

    /// <summary>
    /// The error function <c>erf(x)</c>, from our fixed polynomial in both modes (max error 2.76 ulp). Record: slot 0 = the input;
    /// the gradient is <c>g · 2/√π · e^(−x²)</c>.
    /// </summary>
    /// <param name="x">Float32 input of any shape and strides.</param>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    public static Tensor Erf(in Tensor x, BufferPool? pool = null)
    {
        Tensor xc = Contig(x, out bool copied);
        Tensor r = NewOut(x.Shape, pool);
        VectorMaps.Map1(new ErfUnary(), xc.InternalData<float>(), r.MutableData<float>());
        return FinishUnary(OpKind.Erf, x, xc, copied, r, 0, saveOutput: false);
    }

    /// <summary>
    /// The natural exponential <c>e^x</c>: our polynomial in exact mode, the lane loop of the fast kernel otherwise. Record: slot 0 = the output.
    /// </summary>
    /// <param name="x">Float32 input of any shape and strides.</param>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    public static Tensor Exp(in Tensor x, BufferPool? pool = null)
    {
        Tensor xc = Contig(x, out bool copied);
        Tensor r = NewOut(x.Shape, pool);
        VectorMaps.Map1(new ExpUnary(), xc.InternalData<float>(), r.MutableData<float>());
        return FinishUnary(OpKind.Exp, x, xc, copied, r, 0, saveOutput: true);
    }

    /// <summary>
    /// The square root, correctly rounded (IEEE 754) so it is the same in both modes; negative inputs give NaN. Record: slot 0 = the
    /// output; the gradient is <c>g / (2·y)</c>.
    /// </summary>
    /// <param name="x">Float32 input of any shape and strides.</param>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    public static Tensor Sqrt(in Tensor x, BufferPool? pool = null)
    {
        Tensor xc = Contig(x, out bool copied);
        Tensor r = NewOut(x.Shape, pool);
        ReadOnlySpan<float> s = xc.InternalData<float>();
        Span<float> d = r.MutableData<float>();
        for (int i = 0; i < s.Length; i++)
        {
            d[i] = MathF.Sqrt(s[i]);
        }

        return FinishUnary(OpKind.Sqrt, x, xc, copied, r, 0, saveOutput: true);
    }

    /// <summary>
    /// <c>x^p</c> for a constant exponent. Exponents 1, 2, 0.5 and −1 use IEEE multiplication, square root and division, so they are
    /// the same in both modes; any other exponent uses <see cref="MathF.Pow(float, float)"/> and is not exact. Record: slot 0 = the
    /// input, Floats = [p]; the gradient is <c>g · p · x^(p−1)</c>.
    /// </summary>
    /// <param name="x">Float32 input of any shape and strides.</param>
    /// <param name="exponent">The exponent.</param>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    /// <exception cref="NotSupportedException">The exponent is not one of the exact ones and the compute mode is exact.</exception>
    public static Tensor Pow(in Tensor x, float exponent, BufferPool? pool = null)
    {
        if (TensorCompute.Exact && !IsExactPower(exponent))
        {
            throw new NotSupportedException($"Pow with exponent {exponent} has no exact kernel; exact mode supports 1, 2, 0.5 and -1.");
        }

        Tensor xc = Contig(x, out bool copied);
        Tensor r = NewOut(x.Shape, pool);
        ReadOnlySpan<float> s = xc.InternalData<float>();
        Span<float> d = r.MutableData<float>();
        for (int i = 0; i < s.Length; i++)
        {
            d[i] = PowScalar(s[i], exponent);
        }

        return FinishUnary(OpKind.Pow, x, xc, copied, r, exponent, saveOutput: false);
    }

    internal static bool IsExactPower(float p) => p is 1f or 2f or 0.5f or -1f;

    internal static float PowScalar(float x, float p) => p switch
    {
        1f => x,
        2f => x * x,
        0.5f => MathF.Sqrt(x),
        -1f => 1f / x,
        _ => MathF.Pow(x, p),
    };

    private static Tensor FinishUnary(OpKind op, in Tensor x, in Tensor xc, bool copied, in Tensor r, float parameter, bool saveOutput)
    {
        if (!Recording(x))
        {
            Free(xc, copied);
            return r;
        }

        BackwardRecord rec = BackwardRecord.Create(op, 1, x);
        if (saveOutput)
        {
            rec.Save(0, r);
            Free(xc, copied);
        }
        else
        {
            SaveContig(rec, 0, xc, copied);
        }

        if (op == OpKind.Pow)
        {
            rec.SetFloats(parameter);
        }

        return rec.Attach(r);
    }

    // ---------------------------------------------------------------------------------------------------------------- where, gather

    /// <summary>
    /// <c>condition ? x : y</c> elementwise with broadcasting of all three. Record: slot 0 = the condition (a contiguous copy at the
    /// broadcast shape); each input's gradient is the output gradient where it was selected and zero elsewhere, summed down to the
    /// input's shape. The condition gets no gradient.
    /// </summary>
    /// <param name="condition">A <see cref="DType.Bool"/> tensor.</param>
    /// <param name="x">Values where the condition is true (float32).</param>
    /// <param name="y">Values where the condition is false (float32).</param>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    /// <exception cref="TensorShapeException">The three shapes do not broadcast.</exception>
    public static Tensor Where(in Tensor condition, in Tensor x, in Tensor y, BufferPool? pool = null)
    {
        RequireFloat32(x, y);
        if (condition.IsEmpty || condition.DType != DType.Bool)
        {
            throw new ArgumentException("Where needs a Bool condition.", nameof(condition));
        }

        Shape s = Shape.Broadcast(Shape.Broadcast(condition.Shape, x.Shape), y.Shape);
        Tensor cc, xc, yc;
        using (GradMode.NoGradInternal())
        {
            cc = condition.Expand(s).Contiguous();
            xc = x.Expand(s).Contiguous();
            yc = y.Expand(s).Contiguous();
        }

        Tensor r = NewOut(s, pool);
        ReadOnlySpan<bool> c = cc.InternalData<bool>();
        ReadOnlySpan<float> xv = xc.InternalData<float>(), yv = yc.InternalData<float>();
        Span<float> d = r.MutableData<float>();
        for (int i = 0; i < d.Length; i++)
        {
            d[i] = c[i] ? xv[i] : yv[i];
        }

        if (!ReferenceEquals(xc.Buffer, x.Buffer))
        {
            xc.Dispose();
        }

        if (!ReferenceEquals(yc.Buffer, y.Buffer))
        {
            yc.Dispose();
        }

        if (!Recording(x, y))
        {
            if (!ReferenceEquals(cc.Buffer, condition.Buffer))
            {
                cc.Dispose();
            }

            return r;
        }

        BackwardRecord rec = BackwardRecord.Create(OpKind.Where, 1, condition, x, y);
        rec.Save(0, cc, owned: !ReferenceEquals(cc.Buffer, condition.Buffer));
        return rec.Attach(r);
    }

    /// <summary>
    /// Gathers slices of <paramref name="data"/> along <paramref name="axis"/> by <paramref name="indices"/> (ONNX <c>Gather</c>): the
    /// result has shape <c>data[:axis] + indices.shape + data[axis+1:]</c>. Negative indices count from the end of the axis. Record:
    /// slot 0 = the indices (contiguous), Ints = [axis]; the gradient adds each output slice into its source row, in index order.
    /// </summary>
    /// <param name="data">Float32 source of any strides.</param>
    /// <param name="indices">Int32 or Int64 indices of any shape (rank 0 allowed).</param>
    /// <param name="axis">Axis of <paramref name="data"/>; negative counts from the end.</param>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    /// <exception cref="ArgumentOutOfRangeException">An index is outside the axis.</exception>
    /// <exception cref="TensorShapeException">The result would exceed the maximum rank.</exception>
    public static Tensor Gather(in Tensor data, in Tensor indices, int axis, BufferPool? pool = null)
    {
        RequireFloat32(data);
        if (indices.IsEmpty || (indices.DType != DType.I32 && indices.DType != DType.I64))
        {
            throw new ArgumentException("Gather needs Int32 or Int64 indices.", nameof(indices));
        }

        int ax = Shape.NormalizeAxis(axis, data.Rank);
        if (data.Rank - 1 + indices.Rank > Shape.MaxRank)
        {
            throw new TensorShapeException("The gathered tensor would exceed the maximum rank.");
        }

        Span<int> od = stackalloc int[Shape.MaxRank];
        int rank = 0;
        int outer = 1, inner = 1, n = data.Shape[ax];
        for (int i = 0; i < ax; i++)
        {
            od[rank++] = data.Shape[i];
            outer *= data.Shape[i];
        }

        for (int i = 0; i < indices.Rank; i++)
        {
            od[rank++] = indices.Shape[i];
        }

        for (int i = ax + 1; i < data.Rank; i++)
        {
            od[rank++] = data.Shape[i];
            inner *= data.Shape[i];
        }

        Tensor dc = Contig(data, out bool copied);
        Tensor ic = indices.IsContiguous ? indices : indices.Clone();
        Tensor r = NewOut(Shape.Create(od.Slice(0, rank)), pool);
        ReadOnlySpan<float> src = dc.InternalData<float>();
        Span<float> dst = r.MutableData<float>();
        int count = indices.Length;
        for (int o = 0; o < outer; o++)
        {
            for (int j = 0; j < count; j++)
            {
                long id = GatherIndex(ic, j, n);
                if (id < 0)
                {
                    Free(dc, copied);
                    if (!indices.IsContiguous)
                    {
                        ic.Dispose();
                    }

                    r.Dispose();
                    throw new ArgumentOutOfRangeException(nameof(indices), $"Index at position {j} is outside the axis of {n}.");
                }

                src.Slice(((o * n) + (int)id) * inner, inner).CopyTo(dst.Slice(((o * count) + j) * inner, inner));
            }
        }

        Free(dc, copied);
        if (!Recording(data))
        {
            if (!indices.IsContiguous)
            {
                ic.Dispose();
            }

            return r;
        }

        BackwardRecord rec = BackwardRecord.Create(OpKind.Gather, 1, data);
        rec.Save(0, ic, owned: !indices.IsContiguous);
        rec.SetInts(ax);
        return rec.Attach(r);
    }

    // The index at position j wrapped into [0, n), or -1 when it is out of range.
    internal static long GatherIndex(in Tensor indices, int j, int n)
    {
        long id = BackwardOps.TargetAt(indices, j);
        if (id < 0)
        {
            id += n;
        }

        return (ulong)id < (ulong)n ? id : -1;
    }
}
