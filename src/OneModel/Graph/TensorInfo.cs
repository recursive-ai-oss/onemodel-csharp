using System.Collections.Immutable;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph;

/// <summary>The name, element type and declared shape of a model input or output.</summary>
/// <param name="Name">The name in the model.</param>
/// <param name="DType">The element type.</param>
/// <param name="Dims">The declared extents; empty when <paramref name="HasShape"/> is false. A symbolic or unknown extent has
/// <see cref="IrDim.IsFixed"/> false (<see cref="IrDim.Param"/> names it when the model does).</param>
/// <param name="HasShape">Whether the model declares a shape at all.</param>
public sealed record TensorInfo(string Name, DType DType, ImmutableArray<IrDim> Dims, bool HasShape)
{
    /// <summary>The number of dimensions, or -1 when the model declares no shape.</summary>
    public int Rank => HasShape ? Dims.Length : -1;
}
