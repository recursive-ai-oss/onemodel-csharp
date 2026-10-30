using System;
using System.Collections.Generic;
using System.Threading;
namespace RecursiveAI.OneModel.Tensors;

/// <summary>Elementwise binary operations.</summary>
public enum BinaryOp
{
    /// <summary><c>a + b</c>.</summary>
    Add,
    /// <summary><c>a - b</c>.</summary>
    Sub,
    /// <summary><c>a * b</c>.</summary>
    Mul,
    /// <summary><c>a / b</c>.</summary>
    Div,
}

/// <summary>Elementwise unary operations.</summary>
public enum UnaryOp
{
    /// <summary><c>-x</c>.</summary>
    Neg,
    /// <summary><c>max(x, 0)</c>; NaN stays NaN.</summary>
    Relu,
}

/// <summary>Window geometry of a 2-D pooling operation (no dilation): kernel, stride and symmetric zero-padding in height and width.</summary>
/// <param name="KernelH">Window height, at least 1.</param>
/// <param name="KernelW">Window width, at least 1.</param>
/// <param name="StrideH">Vertical step, at least 1.</param>
/// <param name="StrideW">Horizontal step, at least 1.</param>
/// <param name="PadH">Rows of padding on top and bottom, 0 to KernelH-1.</param>
/// <param name="PadW">Columns of padding left and right, 0 to KernelW-1.</param>
public readonly record struct Pool2dParams(int KernelH, int KernelW, int StrideH, int StrideW, int PadH, int PadW)
{
    /// <summary>Output extent for an input extent: <c>floor((size + 2·pad − kernel) / stride) + 1</c>.</summary>
    /// <exception cref="TensorShapeException">The parameters are invalid or the window does not fit.</exception>
    public static int OutputSize(int size, int kernel, int stride, int pad)
    {
        if (kernel < 1 || stride < 1 || pad < 0 || pad >= kernel)
        {
            throw new TensorShapeException($"Invalid pooling window (kernel {kernel}, stride {stride}, pad {pad}).");
        }

        long span = (long)size + (2L * pad) - kernel;
        if (span < 0)
        {
            throw new TensorShapeException($"A window of {kernel} does not fit an input of {size} with padding {pad}.");
        }

        return (int)((span / stride) + 1);
    }
}

/// <summary>
/// The seam where the kernel registry plugs into the eager operations. An implementation is chosen once, up front (fast or
/// exact, per instruction set), and every eager operation calls through it. All methods work on float32 tensors with arbitrary
/// strides (including negative and zero), and must produce results that depend only on the inputs and the implementation, never on
/// the thread count. Shapes are validated by the caller; implementations may assume them.
/// </summary>
public interface ITensorKernels
{
    /// <summary>A short name for diagnostics, for example <c>reference</c>.</summary>
    string Name { get; }

    /// <summary><c>dst = a op b</c> elementwise. All three have the same shape (the caller expanded broadcasts to stride 0). <paramref name="dst"/> must not alias an input unless it is the same view.</summary>
    void Binary(BinaryOp op, in Tensor a, in Tensor b, in Tensor dst);

    /// <summary><c>dst = op(x)</c> elementwise, same shape. <paramref name="dst"/> may be <paramref name="x"/>.</summary>
    void Unary(UnaryOp op, in Tensor x, in Tensor dst);

    /// <summary><c>dst[m,n] = Σ_k a[m,k] · b[k,n]</c>. Operands may be transposed views.</summary>
    void MatMul(in Tensor a, in Tensor b, in Tensor dst);

    /// <summary>
    /// Max pooling of an (N,C,H,W) input into <paramref name="dst"/> (N,C,OH,OW). When <paramref name="indices"/> is not empty
    /// (Int32, shape of dst) it receives, for each output, the flat row-major index into the logical (N,C,H,W) input of the
    /// element that won. Ties go to the first element in window order; a NaN in the window wins and the first NaN is reported.
    /// </summary>
    void MaxPool2d(in Tensor x, in Pool2dParams p, in Tensor dst, in Tensor indices);
}

/// <summary>Holds the kernel set the eager operations use when the caller names none.</summary>
public static class KernelSet
{
    private static ITensorKernels s_default = DenseKernels.Instance;

    /// <summary>The default kernels: <see cref="DenseKernels"/> (reference semantics, with the matrix product on the blocked GEMM) until <see cref="SetDefault"/> is called.</summary>
    public static ITensorKernels Default => Volatile.Read(ref s_default);

    /// <summary>Replaces the default. Call once at startup, before running anything; sessions capture the set they were created with.</summary>
    public static void SetDefault(ITensorKernels kernels)
    {
        ArgumentNullException.ThrowIfNull(kernels);
        Volatile.Write(ref s_default, kernels);
    }
}

/// <summary>
/// Plain scalar loops that define what each eager operation means: double-precision accumulation in the matrix product, no
/// SIMD, no threads. Slow by design; the optimized kernels are tested against it.
/// </summary>
public sealed class ReferenceKernels : ITensorKernels
{
    /// <summary>The shared instance.</summary>
    public static ReferenceKernels Instance { get; } = new();

    private ReferenceKernels() { }

    /// <inheritdoc/>
    public string Name => "reference";

    /// <inheritdoc/>
    public void Binary(BinaryOp op, in Tensor a, in Tensor b, in Tensor dst)
    {
        ReadOnlySpan<float> sa = a.InternalStorage<float>(), sb = b.InternalStorage<float>();
        Span<float> d = dst.MutableStorage<float>();
        Shape shape = dst.Shape;
        if (shape.Length == 0)
        {
            return;
        }

        int rank = shape.Rank;
        Strides stA = a.Strides, stB = b.Strides, stD = dst.Strides;
        Span<int> idx = stackalloc int[Shape.MaxRank];
        int pa = a.Offset, pb = b.Offset, pd = dst.Offset;
        while (true)
        {
            // Innermost dimension (or the single element of a scalar).
            int inner = rank == 0 ? 1 : shape[rank - 1];
            int ia = rank == 0 ? 0 : stA[rank - 1], ib = rank == 0 ? 0 : stB[rank - 1], id = rank == 0 ? 0 : stD[rank - 1];
            for (int i = 0, x = pa, y = pb, z = pd; i < inner; i++, x += ia, y += ib, z += id)
            {
                float u = sa[x], v = sb[y];
                d[z] = op switch
                {
                    BinaryOp.Add => u + v,
                    BinaryOp.Sub => u - v,
                    BinaryOp.Mul => u * v,
                    _ => u / v,
                };
            }

            int ax = rank - 2;
            for (; ax >= 0; ax--)
            {
                idx[ax]++;
                pa += stA[ax];
                pb += stB[ax];
                pd += stD[ax];
                if (idx[ax] < shape[ax])
                {
                    break;
                }

                pa -= stA[ax] * shape[ax];
                pb -= stB[ax] * shape[ax];
                pd -= stD[ax] * shape[ax];
                idx[ax] = 0;
            }

            if (ax < 0)
            {
                return;
            }
        }
    }

    /// <inheritdoc/>
    public void Unary(UnaryOp op, in Tensor x, in Tensor dst)
    {
        ReadOnlySpan<float> sx = x.InternalStorage<float>();
        Span<float> d = dst.MutableStorage<float>();
        Shape shape = dst.Shape;
        if (shape.Length == 0)
        {
            return;
        }

        int rank = shape.Rank;
        Strides stX = x.Strides, stD = dst.Strides;
        Span<int> idx = stackalloc int[Shape.MaxRank];
        int px = x.Offset, pd = dst.Offset;
        while (true)
        {
            int inner = rank == 0 ? 1 : shape[rank - 1];
            int ix = rank == 0 ? 0 : stX[rank - 1], id = rank == 0 ? 0 : stD[rank - 1];
            for (int i = 0, a = px, z = pd; i < inner; i++, a += ix, z += id)
            {
                float v = sx[a];
                d[z] = op == UnaryOp.Neg ? -v : (v > 0f || float.IsNaN(v) ? v : 0f);
            }

            int ax = rank - 2;
            for (; ax >= 0; ax--)
            {
                idx[ax]++;
                px += stX[ax];
                pd += stD[ax];
                if (idx[ax] < shape[ax])
                {
                    break;
                }

                px -= stX[ax] * shape[ax];
                pd -= stD[ax] * shape[ax];
                idx[ax] = 0;
            }

            if (ax < 0)
            {
                return;
            }
        }
    }

    /// <inheritdoc/>
    public void MatMul(in Tensor a, in Tensor b, in Tensor dst)
    {
        ReadOnlySpan<float> sa = a.InternalStorage<float>(), sb = b.InternalStorage<float>();
        Span<float> d = dst.MutableStorage<float>();
        int m = a.Shape[0], k = a.Shape[1], n = b.Shape[1];
        int a0 = a.Strides[0], a1 = a.Strides[1], b0 = b.Strides[0], b1 = b.Strides[1], d0 = dst.Strides[0], d1 = dst.Strides[1];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double acc = 0;
                for (int p = 0; p < k; p++)
                {
                    acc += (double)sa[a.Offset + (i * a0) + (p * a1)] * sb[b.Offset + (p * b0) + (j * b1)];
                }

                d[dst.Offset + (i * d0) + (j * d1)] = (float)acc;
            }
        }
    }

    /// <inheritdoc/>
    public void MaxPool2d(in Tensor x, in Pool2dParams p, in Tensor dst, in Tensor indices)
    {
        ReadOnlySpan<float> sx = x.InternalStorage<float>();
        Span<float> d = dst.MutableStorage<float>();
        Span<int> ind = indices.IsEmpty ? default : indices.MutableStorage<int>();
        int n = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        int oh = dst.Shape[2], ow = dst.Shape[3];
        Strides sx4 = x.Strides, sd = dst.Strides;
        Strides si = indices.IsEmpty ? default : indices.Strides;
        for (int in_ = 0; in_ < n; in_++)
        {
            for (int ic = 0; ic < c; ic++)
            {
                for (int oy = 0; oy < oh; oy++)
                {
                    for (int ox = 0; ox < ow; ox++)
                    {
                        float best = float.NegativeInfinity;
                        int bestIdx = -1;
                        for (int ky = 0; ky < p.KernelH; ky++)
                        {
                            int y = (oy * p.StrideH) - p.PadH + ky;
                            if ((uint)y >= (uint)h)
                            {
                                continue;
                            }

                            for (int kx = 0; kx < p.KernelW; kx++)
                            {
                                int xx = (ox * p.StrideW) - p.PadW + kx;
                                if ((uint)xx >= (uint)w)
                                {
                                    continue;
                                }

                                float v = sx[x.Offset + (in_ * sx4[0]) + (ic * sx4[1]) + (y * sx4[2]) + (xx * sx4[3])];
                                bool wins = bestIdx < 0 || (v > best && !float.IsNaN(best)) || (float.IsNaN(v) && !float.IsNaN(best));
                                if (wins)
                                {
                                    best = v;
                                    bestIdx = (((((in_ * c) + ic) * h) + y) * w) + xx;
                                }
                            }
                        }

                        d[dst.Offset + (in_ * sd[0]) + (ic * sd[1]) + (oy * sd[2]) + (ox * sd[3])] = best;
                        if (!ind.IsEmpty)
                        {
                            ind[indices.Offset + (in_ * si[0]) + (ic * si[1]) + (oy * si[2]) + (ox * si[3])] = bestIdx;
                        }
                    }
                }
            }
        }
    }
}
