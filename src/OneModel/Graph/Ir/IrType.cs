using System;
using System.Collections.Immutable;
using System.Globalization;
using System.Text;

namespace RecursiveAI.OneModel.Graph;

/// <summary>One extent of a declared shape: a fixed size, a named symbol, or unknown.</summary>
/// <param name="Value">The fixed size, or -1 when the dimension is symbolic or unknown. A negative size read from a file is kept as read.</param>
/// <param name="Param">The symbol (for example <c>batch</c>) when the dimension is symbolic; otherwise null.</param>
public readonly record struct IrDim(long Value, string? Param)
{
    /// <summary>A dimension with no size and no symbol.</summary>
    public static IrDim Unknown => new(-1, null);

    /// <summary>True when the dimension has a non-negative fixed size.</summary>
    public bool IsFixed => Param is null && Value >= 0;

    /// <inheritdoc/>
    public override string ToString() => Param ?? (Value >= 0 || Value < -1 ? Value.ToString(CultureInfo.InvariantCulture) : "?");
}

/// <summary>Which kind of value a <see cref="IrType"/> describes.</summary>
public enum IrTypeKind
{
    /// <summary>A dense tensor.</summary>
    Tensor,
    /// <summary>A sparse tensor.</summary>
    SparseTensor,
    /// <summary>A sequence of values of one type.</summary>
    Sequence,
    /// <summary>A map from a scalar key type to values of one type.</summary>
    Map,
    /// <summary>A value that may be absent.</summary>
    Optional,
    /// <summary>A type known only by domain and name.</summary>
    Opaque,
}

/// <summary>A declared type: ONNX's <c>TypeProto</c>. Immutable.</summary>
public sealed class IrType
{
    internal IrType(IrTypeKind kind)
    {
        Kind = kind;
    }

    /// <summary>The kind of type.</summary>
    public IrTypeKind Kind { get; }

    /// <summary>For tensors and sparse tensors, the ONNX element type code (<c>TensorProto.DataType</c>); for maps, the key type code; otherwise 0.</summary>
    public int ElemType { get; internal set; }

    /// <summary>For tensors and sparse tensors, the declared dimensions, or default when the rank is unknown. A shape with rank 0 is a scalar.</summary>
    public ImmutableArray<IrDim> Dims { get; internal set; }

    /// <summary>True when a tensor type declares a shape (even a scalar's empty one).</summary>
    public bool HasShape { get; internal set; }

    /// <summary>For sequences, optionals and maps, the element (or map value) type; otherwise null.</summary>
    public IrType? Inner { get; internal set; }

    /// <summary>For opaque types, the domain; otherwise null.</summary>
    public string? OpaqueDomain { get; internal set; }

    /// <summary>For opaque types, the name; otherwise null.</summary>
    public string? OpaqueName { get; internal set; }

    /// <summary>Renders the type compactly, for example <c>tensor(1)[batch,3,?]</c>.</summary>
    public override string ToString()
    {
        var sb = new StringBuilder();
        Append(sb);
        return sb.ToString();
    }

    private void Append(StringBuilder sb)
    {
        switch (Kind)
        {
            case IrTypeKind.Tensor:
            case IrTypeKind.SparseTensor:
                sb.Append(Kind == IrTypeKind.Tensor ? "tensor(" : "sparse(").Append(ElemType.ToString(CultureInfo.InvariantCulture)).Append(')');
                if (HasShape)
                {
                    sb.Append('[');
                    for (int i = 0; i < Dims.Length; i++)
                    {
                        if (i > 0)
                        {
                            sb.Append(',');
                        }

                        sb.Append(Dims[i].ToString());
                    }

                    sb.Append(']');
                }

                break;
            case IrTypeKind.Sequence:
            case IrTypeKind.Optional:
                sb.Append(Kind == IrTypeKind.Sequence ? "seq<" : "opt<");
                Inner?.Append(sb);
                sb.Append('>');
                break;
            case IrTypeKind.Map:
                sb.Append("map<").Append(ElemType.ToString(CultureInfo.InvariantCulture)).Append(',');
                Inner?.Append(sb);
                sb.Append('>');
                break;
            default:
                sb.Append("opaque(").Append(OpaqueDomain).Append(':').Append(OpaqueName).Append(')');
                break;
        }
    }
}
