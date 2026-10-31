using System.Collections.Immutable;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph;

/// <summary>The name, element type and declared shape of a model input or output.</summary>
/// <param name="Name">The name in the model.</param>
/// <param name="DType">The element type.</param>
/// <param name="Dims">The declared extents; empty when <paramref name="HasShape"/> is false. A symbolic or unknown extent has
/// <see cref="IrDim.IsFixed"/> false (<see cref="IrDim.Param"/> names it when the model does).</param>
/// <param name="HasShape">Whether the model declares a shape at all.</param>
/// <param name="Kind">Tensor, or Sequence or Optional, which a session takes and returns as a rank-1 U8 <see cref="PackedValue"/> (then
/// <see cref="DType"/> is U8 and there is no shape).</param>
public sealed record TensorInfo(string Name, DType DType, ImmutableArray<IrDim> Dims, bool HasShape, TensorInfoKind Kind = TensorInfoKind.Tensor)
{
    /// <summary>The number of dimensions, or -1 when the model declares no shape.</summary>
    public int Rank => HasShape ? Dims.Length : -1;

    // The generated text would print the dimension array's type name; this prints `x: F32[N, 16]` (an open extent is its name, or ?).
    private bool PrintMembers(System.Text.StringBuilder builder)
    {
        builder.Append(Name).Append(": ").Append(DType);
        if (Kind != TensorInfoKind.Tensor)
        {
            builder.Append(' ').Append(Kind);
        }

        if (HasShape)
        {
            builder.Append('[').AppendJoin(", ", Dims).Append(']');
        }

        return true;
    }
}
