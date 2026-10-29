using System;
using System.Collections.Immutable;
using System.Runtime.InteropServices;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph;

/// <summary>
/// A constant tensor from a model: ONNX's <c>TensorProto</c>, with its values in little-endian bytes. The bytes are not copied out of
/// the model file at import: when the file is memory-mapped, an initializer is a view into the mapping and costs no managed memory
/// until <see cref="ToTensor"/> copies it. Any element type ONNX defines can be held, including ones <see cref="DType"/> cannot store
/// (strings, complex, 8-bit floats, sub-byte integers); <see cref="ToTensor"/> converts the stored ones.
/// A <c>TensorData</c> is valid until the <see cref="IrModel"/> it came from is disposed.
/// </summary>
public sealed class TensorData
{
    private readonly ByteSource? _source;
    private readonly long _offset;
    private readonly int _length;
    private readonly ImmutableArray<ReadOnlyMemory<byte>> _strings;

    internal TensorData(string name, int dataType, ImmutableArray<long> dims, long elementCount, ByteSource? source, long offset, int length,
        ImmutableArray<ReadOnlyMemory<byte>> strings, bool isExternal, string docString)
    {
        Name = name;
        DataType = dataType;
        Dims = dims;
        ElementCount = elementCount;
        _source = source;
        _offset = offset;
        _length = length;
        _strings = strings;
        IsExternal = isExternal;
        DocString = docString;
    }

    /// <summary>The tensor's name; empty when it has none (for example an attribute value).</summary>
    public string Name { get; }

    /// <summary>The ONNX element type code (<c>TensorProto.DataType</c>), 1 to 28.</summary>
    public int DataType { get; }

    /// <summary>The dimensions; empty for a scalar. Every extent is at least 0.</summary>
    public ImmutableArray<long> Dims { get; }

    /// <summary>The number of elements: the product of <see cref="Dims"/> (1 for a scalar).</summary>
    public long ElementCount { get; }

    /// <summary>True when the data came from an external file.</summary>
    public bool IsExternal { get; }

    /// <summary>The tensor's documentation string, or empty.</summary>
    public string DocString { get; }

    /// <summary>The elements of a string tensor as UTF-8 bytes, in row-major order; empty for other types.</summary>
    public ImmutableArray<ReadOnlyMemory<byte>> Strings => _strings;

    /// <summary>
    /// The values as little-endian bytes in row-major order, in ONNX's raw layout (sub-byte types packed as ONNX defines them);
    /// empty for string tensors and for tensors with no elements.
    /// </summary>
    /// <exception cref="ObjectDisposedException">The model that owns the data was disposed.</exception>
    public ReadOnlySpan<byte> RawData => _source is null ? default : _source.GetSpan(_offset, _length);

    /// <summary>The number of bytes in <see cref="RawData"/>.</summary>
    public int RawLength => _length;

    /// <summary>Gets the stored type for <see cref="DataType"/>, if the tensor layer stores it.</summary>
    public bool TryGetDType(out DType type) => DTypes.TryFromOnnx(DataType, out type);

    /// <summary>
    /// Copies the values into a new contiguous tensor.
    /// </summary>
    /// <param name="pool">The pool to take storage from; <see cref="BufferPool.Shared"/> when null.</param>
    /// <returns>A tensor the caller disposes.</returns>
    /// <exception cref="NotSupportedException">The element type is not one the tensor layer stores (strings, complex, 8-bit floats, 4-bit and smaller types).</exception>
    /// <exception cref="TensorShapeException">The rank exceeds 8 or the element count or byte size exceeds 2^31-1.</exception>
    /// <exception cref="ObjectDisposedException">The model that owns the data was disposed.</exception>
    public Tensor ToTensor(BufferPool? pool = null)
    {
        DType dtype = DTypes.FromOnnx(DataType);
        Shape shape = Shape.Create(Dims.AsSpan());
        Tensor t = Tensor.Create(dtype, shape, pool);
        try
        {
            ReadOnlySpan<byte> raw = RawData;
            switch (dtype)
            {
                case DType.F32: raw.CopyTo(MemoryMarshal.AsBytes(t.MutableData<float>())); break;
                case DType.U8: raw.CopyTo(t.MutableData<byte>()); break;
                case DType.I8: raw.CopyTo(MemoryMarshal.AsBytes(t.MutableData<sbyte>())); break;
                case DType.U16: raw.CopyTo(MemoryMarshal.AsBytes(t.MutableData<ushort>())); break;
                case DType.I16: raw.CopyTo(MemoryMarshal.AsBytes(t.MutableData<short>())); break;
                case DType.I32: raw.CopyTo(MemoryMarshal.AsBytes(t.MutableData<int>())); break;
                case DType.I64: raw.CopyTo(MemoryMarshal.AsBytes(t.MutableData<long>())); break;
                case DType.Bool: raw.CopyTo(MemoryMarshal.AsBytes(t.MutableData<bool>())); break;
                case DType.F16: raw.CopyTo(MemoryMarshal.AsBytes(t.MutableData<Half>())); break;
                case DType.F64: raw.CopyTo(MemoryMarshal.AsBytes(t.MutableData<double>())); break;
                case DType.U32: raw.CopyTo(MemoryMarshal.AsBytes(t.MutableData<uint>())); break;
                case DType.U64: raw.CopyTo(MemoryMarshal.AsBytes(t.MutableData<ulong>())); break;
                case DType.BF16: raw.CopyTo(MemoryMarshal.AsBytes(t.MutableData<BFloat16>())); break;
                default: throw new NotSupportedException($"ONNX element type {DTypes.OnnxName(DataType)} cannot be converted to a tensor.");
            }

            return t;
        }
        catch
        {
            t.Dispose();
            throw;
        }
    }

    /// <summary>The size in bytes of the raw layout of <paramref name="elementCount"/> elements of an ONNX type, or -1 for strings and unknown codes.</summary>
    /// <param name="dataType">The ONNX element type code.</param>
    /// <param name="elementCount">The number of elements; at most 2^60.</param>
    public static long RawByteLength(int dataType, long elementCount)
    {
        long bits = BitsPerElement(dataType);
        if (bits < 0)
        {
            return -1;
        }

        // elementCount is bounded by the caller (the product of checked dimensions), and 2^60 * 128 would overflow, so cap it.
        if (elementCount > (1L << 55))
        {
            return long.MaxValue;
        }

        return (elementCount * bits + 7) / 8;
    }

    /// <summary>Bits per element of an ONNX type's raw layout; -1 for strings and unknown codes.</summary>
    internal static int BitsPerElement(int dataType) => dataType switch
    {
        1 or 6 or 12 => 32,
        2 or 3 or 9 or 17 or 18 or 19 or 20 or 24 => 8,
        4 or 5 or 10 or 16 => 16,
        7 or 11 or 13 or 14 => 64,
        15 => 128,
        21 or 22 or 23 => 4,
        25 or 26 => 2,
        27 or 28 => 6,
        _ => -1,
    };
}

/// <summary>A sparse tensor in coordinate form: ONNX's <c>SparseTensorProto</c>. The importer keeps it as read and does not densify it.</summary>
public sealed class SparseTensorData
{
    internal SparseTensorData(TensorData values, TensorData indices, ImmutableArray<long> dims)
    {
        Values = values;
        Indices = indices;
        Dims = dims;
    }

    /// <summary>The non-default values, a tensor of shape [NNZ]; its name names the sparse tensor.</summary>
    public TensorData Values { get; }

    /// <summary>The indices: int64, shape [NNZ] (linearized) or [NNZ, rank] (coordinates).</summary>
    public TensorData Indices { get; }

    /// <summary>The dense shape.</summary>
    public ImmutableArray<long> Dims { get; }
}
