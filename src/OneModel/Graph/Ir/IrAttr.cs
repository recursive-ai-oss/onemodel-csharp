using System;
using System.Collections.Immutable;
using System.Text;

namespace RecursiveAI.OneModel.Graph;

/// <summary>The kind of value an attribute holds: ONNX's <c>AttributeProto.AttributeType</c>, with the same numeric codes.</summary>
public enum IrAttributeType
{
    /// <summary>Not set.</summary>
    Undefined = 0,
    /// <summary>A 32-bit float.</summary>
    Real = 1,
    /// <summary>A 64-bit integer.</summary>
    Whole = 2,
    /// <summary>A byte string (UTF-8 text by convention).</summary>
    Text = 3,
    /// <summary>A tensor.</summary>
    Tensor = 4,
    /// <summary>A subgraph.</summary>
    Graph = 5,
    /// <summary>A list of floats.</summary>
    Reals = 6,
    /// <summary>A list of integers.</summary>
    Wholes = 7,
    /// <summary>A list of byte strings.</summary>
    Texts = 8,
    /// <summary>A list of tensors.</summary>
    Tensors = 9,
    /// <summary>A list of subgraphs.</summary>
    Graphs = 10,
    /// <summary>A sparse tensor.</summary>
    SparseTensor = 11,
    /// <summary>A list of sparse tensors.</summary>
    SparseTensors = 12,
    /// <summary>A type.</summary>
    TypeProto = 13,
    /// <summary>A list of types.</summary>
    TypeProtos = 14,
}

/// <summary>
/// One attribute of a node. Read the property that matches <see cref="Type"/>; the others throw <see cref="InvalidOperationException"/>.
/// In a function body an attribute may instead refer to one of the function's own attributes (<see cref="RefAttrName"/>), and then holds no value.
/// </summary>
public sealed class IrAttr
{
    private readonly object? _value;
    private readonly long _scalar;

    internal IrAttr(string name, IrAttributeType type, string refAttrName, string docString, object? value, long scalar)
    {
        Name = name;
        Type = type;
        RefAttrName = refAttrName;
        DocString = docString;
        _value = value;
        _scalar = scalar;
    }

    /// <summary>The attribute's name.</summary>
    public string Name { get; }

    /// <summary>The kind of value. Models that omit it (older exporters) get it inferred from the field that is set.</summary>
    public IrAttributeType Type { get; }

    /// <summary>For a function body, the name of the function attribute this one takes its value from; otherwise empty.</summary>
    public string RefAttrName { get; }

    /// <summary>Documentation text, or empty.</summary>
    public string DocString { get; }

    /// <summary>The value of a <see cref="IrAttributeType.Real"/> attribute.</summary>
    public float Real => Expect(IrAttributeType.Real) ? BitConverter.Int32BitsToSingle((int)_scalar) : 0;

    /// <summary>The value of an <see cref="IrAttributeType.Whole"/> attribute.</summary>
    public long Whole => Expect(IrAttributeType.Whole) ? _scalar : 0;

    /// <summary>The bytes of a <see cref="IrAttributeType.Text"/> attribute.</summary>
    public ReadOnlyMemory<byte> Text => Expect(IrAttributeType.Text) ? (byte[])_value! : default;

    /// <summary>The value of a <see cref="IrAttributeType.Tensor"/> attribute.</summary>
    public TensorData Tensor => Expect(IrAttributeType.Tensor) ? (TensorData)_value! : null!;

    /// <summary>The value of a <see cref="IrAttributeType.Graph"/> attribute.</summary>
    public IrGraph Graph => Expect(IrAttributeType.Graph) ? (IrGraph)_value! : null!;

    /// <summary>The value of a <see cref="IrAttributeType.SparseTensor"/> attribute.</summary>
    public SparseTensorData SparseTensor => Expect(IrAttributeType.SparseTensor) ? (SparseTensorData)_value! : null!;

    /// <summary>The value of a <see cref="IrAttributeType.TypeProto"/> attribute.</summary>
    public IrType TypeValue => Expect(IrAttributeType.TypeProto) ? (IrType)_value! : null!;

    /// <summary>The values of a <see cref="IrAttributeType.Reals"/> attribute.</summary>
    public ImmutableArray<float> Reals => Expect(IrAttributeType.Reals) ? (ImmutableArray<float>)_value! : default;

    /// <summary>The values of an <see cref="IrAttributeType.Wholes"/> attribute.</summary>
    public ImmutableArray<long> Wholes => Expect(IrAttributeType.Wholes) ? (ImmutableArray<long>)_value! : default;

    /// <summary>The values of a <see cref="IrAttributeType.Texts"/> attribute.</summary>
    public ImmutableArray<ReadOnlyMemory<byte>> Texts => Expect(IrAttributeType.Texts) ? (ImmutableArray<ReadOnlyMemory<byte>>)_value! : default;

    /// <summary>The values of a <see cref="IrAttributeType.Tensors"/> attribute.</summary>
    public ImmutableArray<TensorData> Tensors => Expect(IrAttributeType.Tensors) ? (ImmutableArray<TensorData>)_value! : default;

    /// <summary>The values of a <see cref="IrAttributeType.Graphs"/> attribute.</summary>
    public ImmutableArray<IrGraph> Graphs => Expect(IrAttributeType.Graphs) ? (ImmutableArray<IrGraph>)_value! : default;

    /// <summary>The values of a <see cref="IrAttributeType.SparseTensors"/> attribute.</summary>
    public ImmutableArray<SparseTensorData> SparseTensors => Expect(IrAttributeType.SparseTensors) ? (ImmutableArray<SparseTensorData>)_value! : default;

    /// <summary>The values of a <see cref="IrAttributeType.TypeProtos"/> attribute.</summary>
    public ImmutableArray<IrType> TypeValues => Expect(IrAttributeType.TypeProtos) ? (ImmutableArray<IrType>)_value! : default;

    /// <summary>Decodes a <see cref="IrAttributeType.Text"/> attribute as UTF-8, replacing invalid bytes.</summary>
    public string GetText() => Encoding.UTF8.GetString(Text.Span);

    private bool Expect(IrAttributeType type) =>
        Type == type ? true : throw new InvalidOperationException($"Attribute '{Name}' is {Type}, not {type}.");
}
