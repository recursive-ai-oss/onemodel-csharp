using System;
using System.Buffers.Binary;

namespace RecursiveAI.OneModel.Onnx;

/// <summary>A growable byte buffer for assembling little-endian tensor data from typed protobuf fields.</summary>
internal struct ByteList
{
    private byte[]? _array;
    private int _count;

    public readonly int Count => _count;

    public readonly bool IsEmpty => _count == 0;

    public readonly ReadOnlySpan<byte> Span => _array is null ? default : _array.AsSpan(0, _count);

    public byte[] ToArray() => Span.ToArray();

    public Span<byte> Grow(int extra, ParseContext ctx)
    {
        long need = (long)_count + extra;
        if (need > Array.MaxLength)
        {
            throw ctx.Fail(OnnxErrorKind.Unsupported, "Tensor data is larger than 2 GB.");
        }

        if (_array is null || need > _array.Length)
        {
            long cap = Math.Max(need, Math.Max(64, _array is null ? 0 : (long)_array.Length * 2));
            Array.Resize(ref _array, (int)Math.Min(cap, Array.MaxLength));
        }

        Span<byte> s = _array.AsSpan(_count, extra);
        _count += extra;
        return s;
    }

    public void AppendUInt32(uint v, ParseContext ctx) => BinaryPrimitives.WriteUInt32LittleEndian(Grow(4, ctx), v);

    public void AppendUInt64(ulong v, ParseContext ctx) => BinaryPrimitives.WriteUInt64LittleEndian(Grow(8, ctx), v);
}
