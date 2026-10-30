using System;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

/// <summary>
/// Reductions the main reduce kernel does not cover: ReduceMax and ReduceMin on bool (any, all) and ReduceLogSum and ReduceLogSumExp
/// on float64. They are plain loops over the reduced axes in increasing index order. The float64 log forms call the platform's
/// math library, so they are not exact (the kernel refuses exact mode for them when it prepares).
/// </summary>
internal static class ReduceExtra
{
    /// <summary>Reduces a bool tensor: any (ReduceMax) or all (ReduceMin) over the marked axes; an empty set gives false / true.</summary>
    public static void BoolMaxMin(ReadOnlySpan<byte> x, Span<byte> y, ReadOnlySpan<int> dims, ReadOnlySpan<bool> reduced, bool max)
    {
        Span<int> buf = stackalloc int[4 * Shape.MaxRank];
        Walk(dims, reduced, buf, out int outputs, out int reducedCount, out Span<int> keptDims, out Span<int> keptStrides, out Span<int> redDims, out Span<int> redStrides);
        for (int o = 0; o < outputs; o++)
        {
            int b = Base(o, keptDims, keptStrides);
            byte acc = max ? (byte)0 : (byte)1;
            for (int r = 0; r < reducedCount; r++)
            {
                byte v = x[b + Offset(r, redDims, redStrides)];
                acc = max ? (byte)(acc | v) : (byte)(acc & v);
            }

            y[o] = acc;
        }
    }

    /// <summary>log(sum(x)) or log(sum(exp(x))) over the marked axes of a float64 tensor. The exponential form subtracts the maximum first.</summary>
    public static void LogDouble(ReadOnlySpan<double> x, Span<double> y, ReadOnlySpan<int> dims, ReadOnlySpan<bool> reduced, bool exp)
    {
        Span<int> buf = stackalloc int[4 * Shape.MaxRank];
        Walk(dims, reduced, buf, out int outputs, out int reducedCount, out Span<int> keptDims, out Span<int> keptStrides, out Span<int> redDims, out Span<int> redStrides);
        for (int o = 0; o < outputs; o++)
        {
            int b = Base(o, keptDims, keptStrides);
            double max = 0;
            if (exp)
            {
                max = double.NegativeInfinity;
                for (int r = 0; r < reducedCount; r++)
                {
                    max = Math.Max(max, x[b + Offset(r, redDims, redStrides)]);
                }
            }

            double sum = 0;
            for (int r = 0; r < reducedCount; r++)
            {
                double v = x[b + Offset(r, redDims, redStrides)];
                sum += exp ? Math.Exp(v - max) : v;
            }

            y[o] = exp ? Math.Log(sum) + max : Math.Log(sum);
        }
    }

    // Splits the axes into kept and reduced ones with the element strides of a contiguous input.
    private static void Walk(ReadOnlySpan<int> dims, ReadOnlySpan<bool> reduced, Span<int> buf, out int outputs, out int reducedCount,
        out Span<int> keptDims, out Span<int> keptStrides, out Span<int> redDims, out Span<int> redStrides)
    {
        int n = Shape.MaxRank;
        Span<int> kd = buf[..n], ks = buf[n..(2 * n)], rd = buf[(2 * n)..(3 * n)], rs = buf[(3 * n)..];
        int nk = 0, nr = 0, stride = 1;
        outputs = 1;
        reducedCount = 1;
        for (int d = dims.Length - 1; d >= 0; d--)
        {
            if (reduced[d])
            {
                rd[nr] = dims[d];
                rs[nr++] = stride;
                reducedCount *= dims[d];
            }
            else
            {
                kd[nk] = dims[d];
                ks[nk++] = stride;
                outputs *= dims[d];
            }

            stride *= dims[d];
        }

        keptDims = kd[..nk];
        keptStrides = ks[..nk];
        redDims = rd[..nr];
        redStrides = rs[..nr];
    }

    // Axes were stored innermost first, so the least significant digit of an index is element 0.
    private static int Base(int o, ReadOnlySpan<int> dims, ReadOnlySpan<int> strides)
    {
        int off = 0;
        for (int i = 0; i < dims.Length; i++)
        {
            off += (o % dims[i]) * strides[i];
            o /= dims[i];
        }

        return off;
    }

    private static int Offset(int r, ReadOnlySpan<int> dims, ReadOnlySpan<int> strides) => Base(r, dims, strides);
}
