using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph;

/// <summary>What a graph input, output or value holds.</summary>
public enum TensorInfoKind
{
    /// <summary>A tensor.</summary>
    Tensor = 0,

    /// <summary>An ONNX sequence of tensors, carried as a <see cref="PackedValue"/>.</summary>
    Sequence = 1,

    /// <summary>An ONNX optional (a tensor, a sequence, or nothing), carried as a <see cref="PackedValue"/>.</summary>
    Optional = 2,
}

/// <summary>What a packed value holds.</summary>
public enum PackedKind
{
    /// <summary>A sequence of tensors (possibly none).</summary>
    Sequence = 0,

    /// <summary>An optional with no value.</summary>
    OptionalEmpty = 1,

    /// <summary>An optional holding one tensor.</summary>
    OptionalTensor = 2,

    /// <summary>An optional holding a sequence of tensors.</summary>
    OptionalSequence = 3,
}

/// <summary>
/// ONNX sequences and optionals as the session sees them: one rank-1 <see cref="DType.U8"/> tensor that holds the value's elements
/// back to back, so that sessions, plans and kernels keep a single value kind. A session input or output declared as a sequence or
/// an optional (<see cref="TensorInfo.Kind"/>) takes and returns such a tensor; use these methods to make and read them.
/// </summary>
/// <remarks>
/// Layout (little-endian): magic "OMV1", one tag byte (a <see cref="PackedKind"/>), then for a sequence an int32 count, and for each
/// tensor one byte of <see cref="DType"/>, one byte of rank, rank int32 extents and the row-major element bytes. An optional holding
/// a tensor has the same tensor record without the count. The layout is internal to the library and may change.
/// </remarks>
public static class PackedValue
{
    private static ReadOnlySpan<byte> Magic => "OMV1"u8;

    /// <summary>Packs tensors as a sequence.</summary>
    /// <param name="items">The elements, in order. Read only; non-contiguous tensors are copied.</param>
    /// <returns>A new rank-1 U8 tensor the caller disposes.</returns>
    public static Tensor Sequence(ReadOnlySpan<Tensor> items) => Pack(PackedKind.Sequence, items);

    /// <summary>An optional with no value.</summary>
    /// <returns>A new tensor the caller disposes.</returns>
    public static Tensor EmptyOptional() => Pack(PackedKind.OptionalEmpty, default);

    /// <summary>An optional holding <paramref name="item"/>.</summary>
    /// <returns>A new tensor the caller disposes.</returns>
    public static Tensor Optional(Tensor item) => Pack(PackedKind.OptionalTensor, new ReadOnlySpan<Tensor>(in item));

    /// <summary>An optional holding a sequence.</summary>
    /// <returns>A new tensor the caller disposes.</returns>
    public static Tensor OptionalSequence(ReadOnlySpan<Tensor> items) => Pack(PackedKind.OptionalSequence, items);

    /// <summary>True when <paramref name="value"/> is a packed value made by this class.</summary>
    public static bool IsPacked(Tensor value)
    {
        if (value.IsEmpty || value.DType != DType.U8 || value.Rank != 1 || value.Shape[0] < 5)
        {
            return false;
        }

        ReadOnlySpan<byte> b = value.Data<byte>();
        return b.Slice(0, 4).SequenceEqual(Magic) && b[4] <= (byte)PackedKind.OptionalSequence;
    }

    /// <summary>What <paramref name="value"/> holds.</summary>
    /// <exception cref="ArgumentException">The tensor is not a packed value.</exception>
    public static PackedKind Kind(Tensor value)
    {
        if (!IsPacked(value))
        {
            throw new ArgumentException("The tensor is not a packed sequence or optional.", nameof(value));
        }

        return (PackedKind)value.Data<byte>()[4];
    }

    /// <summary>The tensors a packed value holds: all elements of a sequence, the one of an optional tensor, none for an empty optional.</summary>
    /// <returns>New tensors; the caller disposes each.</returns>
    /// <exception cref="ArgumentException">The tensor is not a packed value.</exception>
    /// <exception cref="FormatException">The bytes are cut short or inconsistent.</exception>
    public static Tensor[] Items(Tensor value)
    {
        PackedKind kind = Kind(value);
        ReadOnlySpan<byte> b = value.Data<byte>();
        int at = 5;
        int count;
        switch (kind)
        {
            case PackedKind.OptionalEmpty:
                return [];
            case PackedKind.OptionalTensor:
                count = 1;
                break;
            default:
                if (b.Length < at + 4)
                {
                    throw new FormatException("The packed value is cut short.");
                }

                count = BinaryPrimitives.ReadInt32LittleEndian(b.Slice(at));
                at += 4;
                if (count < 0 || count > b.Length)
                {
                    throw new FormatException("The packed value has an impossible element count.");
                }

                break;
        }

        var items = new List<Tensor>(count);
        try
        {
            for (int i = 0; i < count; i++)
            {
                items.Add(ReadTensor(b, ref at));
            }
        }
        catch
        {
            foreach (Tensor t in items)
            {
                t.Dispose();
            }

            throw;
        }

        return [.. items];
    }

    private static Tensor Pack(PackedKind kind, ReadOnlySpan<Tensor> items)
    {
        long size = 5 + (kind is PackedKind.Sequence or PackedKind.OptionalSequence ? 4 : 0);
        foreach (Tensor t in items)
        {
            size += 2 + (4L * t.Rank) + ((long)t.Length * t.ElementSize);
        }

        if (size > int.MaxValue)
        {
            throw new ArgumentException("The value is larger than 2 GiB.");
        }

        Tensor result = Tensor.Create(DType.U8, Shape.Create((int)size));
        try
        {
            Span<byte> b = result.MutableData<byte>();
            Magic.CopyTo(b);
            b[4] = (byte)kind;
            int at = 5;
            if (kind is PackedKind.Sequence or PackedKind.OptionalSequence)
            {
                BinaryPrimitives.WriteInt32LittleEndian(b.Slice(at), items.Length);
                at += 4;
            }

            foreach (Tensor t in items)
            {
                WriteTensor(b, ref at, t);
            }

            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
    }

    private static void WriteTensor(Span<byte> b, ref int at, Tensor t)
    {
        b[at++] = (byte)t.DType;
        b[at++] = (byte)t.Rank;
        for (int d = 0; d < t.Rank; d++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(b.Slice(at), t.Shape[d]);
            at += 4;
        }

        int bytes = t.Length * t.ElementSize;
        if (bytes > 0)
        {
            bool copy = !t.IsContiguous || t.IsBlocked;
            Tensor c = copy ? t.Contiguous() : t;
            try
            {
                Bytes(c).CopyTo(b.Slice(at, bytes));
            }
            finally
            {
                if (copy)
                {
                    c.Dispose();
                }
            }
        }

        at += bytes;
    }

    private static Tensor ReadTensor(ReadOnlySpan<byte> b, ref int at)
    {
        if (b.Length < at + 2)
        {
            throw new FormatException("The packed value is cut short.");
        }

        var type = (DType)b[at++];
        int rank = b[at++];
        if (!Enum.IsDefined(type) || type == DType.Undefined || rank > Shape.MaxRank || b.Length < at + (4 * rank))
        {
            throw new FormatException("The packed value has a bad tensor header.");
        }

        Span<int> dims = stackalloc int[Shape.MaxRank];
        for (int d = 0; d < rank; d++)
        {
            dims[d] = BinaryPrimitives.ReadInt32LittleEndian(b.Slice(at));
            at += 4;
            if (dims[d] < 0)
            {
                throw new FormatException("The packed value has a negative extent.");
            }
        }

        Shape shape = Shape.Create(dims.Slice(0, rank));
        Tensor t = Tensor.Create(type, shape);
        long bytes = (long)t.Length * t.ElementSize;
        if (bytes > b.Length - at)
        {
            t.Dispose();
            throw new FormatException("The packed value is cut short.");
        }

        if (bytes > 0)
        {
            b.Slice(at, (int)bytes).CopyTo(MutableBytes(t));
        }

        at += (int)bytes;
        return t;
    }

    /// <summary>The element bytes of a contiguous tensor.</summary>
    internal static ReadOnlySpan<byte> Bytes(Tensor t) => t.DType switch
    {
        DType.F32 => MemoryMarshal.AsBytes(t.Data<float>()),
        DType.F64 => MemoryMarshal.AsBytes(t.Data<double>()),
        DType.U8 => t.Data<byte>(),
        DType.I8 => MemoryMarshal.AsBytes(t.Data<sbyte>()),
        DType.U16 => MemoryMarshal.AsBytes(t.Data<ushort>()),
        DType.I16 => MemoryMarshal.AsBytes(t.Data<short>()),
        DType.I32 => MemoryMarshal.AsBytes(t.Data<int>()),
        DType.U32 => MemoryMarshal.AsBytes(t.Data<uint>()),
        DType.I64 => MemoryMarshal.AsBytes(t.Data<long>()),
        DType.U64 => MemoryMarshal.AsBytes(t.Data<ulong>()),
        DType.Bool => MemoryMarshal.AsBytes(t.Data<bool>()),
        DType.F16 => MemoryMarshal.AsBytes(t.Data<Half>()),
        DType.BF16 => MemoryMarshal.AsBytes(t.Data<BFloat16>()),
        _ => throw new ArgumentException($"Element type {t.DType} cannot be packed."),
    };

    /// <summary>The writable element bytes of a contiguous tensor.</summary>
    internal static Span<byte> MutableBytes(Tensor t) => t.DType switch
    {
        DType.F32 => MemoryMarshal.AsBytes(t.MutableData<float>()),
        DType.F64 => MemoryMarshal.AsBytes(t.MutableData<double>()),
        DType.U8 => t.MutableData<byte>(),
        DType.I8 => MemoryMarshal.AsBytes(t.MutableData<sbyte>()),
        DType.U16 => MemoryMarshal.AsBytes(t.MutableData<ushort>()),
        DType.I16 => MemoryMarshal.AsBytes(t.MutableData<short>()),
        DType.I32 => MemoryMarshal.AsBytes(t.MutableData<int>()),
        DType.U32 => MemoryMarshal.AsBytes(t.MutableData<uint>()),
        DType.I64 => MemoryMarshal.AsBytes(t.MutableData<long>()),
        DType.U64 => MemoryMarshal.AsBytes(t.MutableData<ulong>()),
        DType.Bool => MemoryMarshal.AsBytes(t.MutableData<bool>()),
        DType.F16 => MemoryMarshal.AsBytes(t.MutableData<Half>()),
        DType.BF16 => MemoryMarshal.AsBytes(t.MutableData<BFloat16>()),
        _ => throw new ArgumentException($"Element type {t.DType} cannot be packed."),
    };
}
