using System;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
namespace RecursiveAI.OneModel.Tensors;

/// <summary>Whole-tensor element-wise maps for the eager operators and their gradients, on the lane loops of <see cref="TrainElementwise"/>.</summary>
internal static class VectorMaps
{
    /// <summary>A new tensor <c>op(a[i], b[i])</c> over two contiguous float32 tensors of the same shape, in the current compute mode.</summary>
    public static Tensor Map2<TOp>(TOp op, in Tensor a, in Tensor b, BufferPool? pool = null, bool? exact = null) where TOp : struct, IFloatBinary
    {
        Tensor r = Tensor.Create(DType.F32, a.Shape, pool);
        TrainElementwise.Binary(TensorCompute.Path, exact ?? TensorCompute.Exact, op, a.Data<float>(), b.Data<float>(), r.MutableData<float>(), r.Length);
        return r;
    }

    /// <summary>Writes <c>op(x[i])</c> into <paramref name="dst"/> (same length, may alias <paramref name="x"/>).</summary>
    public static void Map1<TOp>(TOp op, ReadOnlySpan<float> x, Span<float> dst, bool? exact = null) where TOp : struct, IFloatUnary =>
        FloatUnary.Run(TensorCompute.Path, exact ?? TensorCompute.Exact, op, x, dst);
}
