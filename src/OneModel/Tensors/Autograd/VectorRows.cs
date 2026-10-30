using System;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
namespace RecursiveAI.OneModel.Tensors;

/// <summary>
/// The default <c>Binary</c> and <c>Unary</c> kernels: runs of unit-stride elements go through the lane loops of
/// <see cref="TrainElementwise"/> (one flat run for contiguous tensors, one run per innermost row otherwise); an operand with innermost
/// stride 0 (a broadcast scalar per row) is expanded into a pooled scratch row first; anything else falls back to the scalar reference
/// loop. Every element is one IEEE operation, so the lanes give the reference loop's bits. Exact mode rewrites NaN results to the
/// canonical NaN.
/// </summary>
internal static class VectorRows
{
    public static void Binary(BinaryOp op, in Tensor a, in Tensor b, in Tensor dst)
    {
        Shape shape = dst.Shape;
        if (shape.Length == 0)
        {
            return;
        }

        if (a.IsContiguous && b.IsContiguous && dst.IsContiguous)
        {
            Run(op, a.Storage<float>().Slice(a.Offset), b.Storage<float>().Slice(b.Offset), dst.MutableStorage<float>().Slice(dst.Offset), shape.Length);
            return;
        }

        int rank = shape.Rank;
        if (rank == 0 || a.Strides[rank - 1] is not (0 or 1) || b.Strides[rank - 1] is not (0 or 1) || dst.Strides[rank - 1] != 1)
        {
            ReferenceKernels.Instance.Binary(op, a, b, dst);
            return;
        }

        int inner = shape[rank - 1];
        bool splatA = a.Strides[rank - 1] == 0 && inner > 1, splatB = b.Strides[rank - 1] == 0 && inner > 1;
        Tensor scratchA = splatA ? Tensor.Create(DType.F32, Shape.Create(inner)) : default;
        Tensor scratchB = splatB ? Tensor.Create(DType.F32, Shape.Create(inner)) : default;
        try
        {
            ReadOnlySpan<float> sa = a.Storage<float>(), sb = b.Storage<float>();
            Span<float> d = dst.MutableStorage<float>();
            Strides stA = a.Strides, stB = b.Strides, stD = dst.Strides;
            Span<int> idx = stackalloc int[Shape.MaxRank];
            int pa = a.Offset, pb = b.Offset, pd = dst.Offset;
            while (true)
            {
                ReadOnlySpan<float> ra, rb;
                if (splatA)
                {
                    scratchA.MutableData<float>().Fill(sa[pa]);
                    ra = scratchA.Data<float>();
                }
                else
                {
                    ra = sa.Slice(pa, inner);
                }

                if (splatB)
                {
                    scratchB.MutableData<float>().Fill(sb[pb]);
                    rb = scratchB.Data<float>();
                }
                else
                {
                    rb = sb.Slice(pb, inner);
                }

                Run(op, ra, rb, d.Slice(pd, inner), inner);
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
        finally
        {
            if (!scratchA.IsEmpty)
            {
                scratchA.Dispose();
            }

            if (!scratchB.IsEmpty)
            {
                scratchB.Dispose();
            }
        }
    }

    public static void Unary(UnaryOp op, in Tensor x, in Tensor dst)
    {
        Shape shape = dst.Shape;
        if (shape.Length == 0)
        {
            return;
        }

        if (x.IsContiguous && dst.IsContiguous)
        {
            RunUnary(op, x.Storage<float>().Slice(x.Offset), dst.MutableStorage<float>().Slice(dst.Offset), shape.Length);
            return;
        }

        int rank = shape.Rank;
        if (rank == 0 || x.Strides[rank - 1] != 1 || dst.Strides[rank - 1] != 1)
        {
            ReferenceKernels.Instance.Unary(op, x, dst);
            return;
        }

        int inner = shape[rank - 1];
        ReadOnlySpan<float> sx = x.Storage<float>();
        Span<float> d = dst.MutableStorage<float>();
        Strides stX = x.Strides, stD = dst.Strides;
        Span<int> idx = stackalloc int[Shape.MaxRank];
        int px = x.Offset, pd = dst.Offset;
        while (true)
        {
            RunUnary(op, sx.Slice(px, inner), d.Slice(pd, inner), inner);
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

    private static void Run(BinaryOp op, ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> d, int n)
    {
        KernelPath path = TensorCompute.Path;
        bool exact = TensorCompute.Exact;
        switch (op)
        {
            case BinaryOp.Add:
                TrainElementwise.Binary(path, exact, new AddBin(), a, b, d, n);
                break;
            case BinaryOp.Sub:
                TrainElementwise.Binary(path, exact, new SubBin(), a, b, d, n);
                break;
            case BinaryOp.Mul:
                TrainElementwise.Binary(path, exact, new MulBin(), a, b, d, n);
                break;
            default:
                TrainElementwise.Binary(path, exact, new DivBin(), a, b, d, n);
                break;
        }
    }

    private static void RunUnary(UnaryOp op, ReadOnlySpan<float> x, Span<float> d, int n)
    {
        x = x.Slice(0, n);
        if (op == UnaryOp.Neg)
        {
            FloatUnary.Run(TensorCompute.Path, TensorCompute.Exact, new NegUn(), x, d);
        }
        else
        {
            FloatUnary.Run(TensorCompute.Path, TensorCompute.Exact, new ReluUn(), x, d);
        }
    }
}
