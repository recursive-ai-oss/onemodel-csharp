using System;
namespace RecursiveAI.OneModel.Tensors;

/// <summary>
/// Matrix products for the eager operations and their gradients, on the blocked GEMM of the kernel layer. The kernel takes
/// row-major operands; a transposed operand (the usual case in gradients: dW = xᵀ·g, dX = g·Wᵀ) is copied to a pooled scratch
/// first. The copy is O(rows·cols) against an O(m·n·k) product, and it keeps one accumulation order whatever the operand layout.
/// </summary>
internal static class Dense
{
    /// <summary>C[m,n] = op(A)·op(B). <paramref name="a"/> is stored row-major with row stride <paramref name="lda"/> as [m,k], or as [k,m] when <paramref name="transA"/>; likewise B as [k,n] or [n,k].</summary>
    public static void Gemm(bool transA, bool transB, int m, int n, int k,
        ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc)
    {
        Tensor ta = default, tb = default;
        try
        {
            if (transA && m > 0 && k > 0)
            {
                ta = Tensor.Create(DType.F32, Shape.Create(m, k));
                Transpose(a, lda, k, m, ta.MutableData<float>());
                a = ta.Data<float>();
                lda = k;
            }

            if (transB && n > 0 && k > 0)
            {
                tb = Tensor.Create(DType.F32, Shape.Create(k, n));
                Transpose(b, ldb, n, k, tb.MutableData<float>());
                b = tb.Data<float>();
                ldb = n;
            }

            Kernels.Gemm.Run(TensorCompute.KMode, TensorCompute.Path, m, n, k, a, lda, b, ldb, c, ldc, TensorCompute.Threads);
        }
        finally
        {
            if (!ta.IsEmpty)
            {
                ta.Dispose();
            }

            if (!tb.IsEmpty)
            {
                tb.Dispose();
            }
        }
    }

    /// <summary>dst[c, r] = src[r, c] for a [rows, cols] source with row stride <paramref name="ld"/>; dst is [cols, rows] contiguous.</summary>
    public static void Transpose(ReadOnlySpan<float> src, int ld, int rows, int cols, Span<float> dst)
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
                        dst[(c * rows) + r] = src[(r * ld) + c];
                    }
                }
            }
        }
    }
}

/// <summary>
/// The default tensor kernels: the reference semantics for everything but <see cref="MatMul"/>, which runs on the blocked GEMM
/// (fast or exact per <see cref="TensorCompute.Mode"/>) instead of a double-accumulating triple loop.
/// </summary>
public sealed class DenseKernels : ITensorKernels
{
    /// <summary>The shared instance.</summary>
    public static DenseKernels Instance { get; } = new();

    private DenseKernels() { }

    /// <inheritdoc/>
    public string Name => "dense";

    /// <inheritdoc/>
    public void Binary(BinaryOp op, in Tensor a, in Tensor b, in Tensor dst) => ReferenceKernels.Instance.Binary(op, a, b, dst);

    /// <inheritdoc/>
    public void Unary(UnaryOp op, in Tensor x, in Tensor dst) => ReferenceKernels.Instance.Unary(op, x, dst);

    /// <inheritdoc/>
    public void MaxPool2d(in Tensor x, in Pool2dParams p, in Tensor dst, in Tensor indices) =>
        ReferenceKernels.Instance.MaxPool2d(x, p, dst, indices);

    /// <inheritdoc/>
    public void MatMul(in Tensor a, in Tensor b, in Tensor dst)
    {
        Tensor am = a, bm = b;
        int m = a.Shape[0], k = a.Shape[1], n = b.Shape[1];
        if (!dst.IsContiguous)
        {
            ReferenceKernels.Instance.MatMul(a, b, dst);
            return;
        }

        ReadOnlySpan<float> sa = a.Storage<float>(), sb = b.Storage<float>();
        bool transA = false, transB = false;
        int lda, ldb;
        Tensor ca = default, cb = default;
        try
        {
            if (am.Strides[1] == 1 && am.Strides[0] >= k)
            {
                lda = am.Strides[0];
            }
            else if (am.Strides[0] == 1 && am.Strides[1] >= m)
            {
                transA = true;
                lda = am.Strides[1];
            }
            else
            {
                ca = am.Clone();
                sa = ca.Data<float>();
                lda = k;
                am = ca;
            }

            if (bm.Strides[1] == 1 && bm.Strides[0] >= n)
            {
                ldb = bm.Strides[0];
            }
            else if (bm.Strides[0] == 1 && bm.Strides[1] >= k)
            {
                transB = true;
                ldb = bm.Strides[1];
            }
            else
            {
                cb = bm.Clone();
                sb = cb.Data<float>();
                ldb = n;
                bm = cb;
            }

            // Strided operands index the whole storage from their offset; a clone starts at 0.
            Dense.Gemm(transA, transB, m, n, k, sa.Slice(am.Offset), lda, sb.Slice(bm.Offset), ldb, dst.MutableData<float>(), n);
        }
        finally
        {
            if (!ca.IsEmpty)
            {
                ca.Dispose();
            }

            if (!cb.IsEmpty)
            {
                cb.Dispose();
            }
        }
    }
}
