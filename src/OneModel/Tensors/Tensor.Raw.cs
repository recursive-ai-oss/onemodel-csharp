using System;

namespace RecursiveAI.OneModel.Tensors;

// Element-type-blind access for data-movement kernels: a gather or a transpose moves 4-byte elements whether they are floats, ints
// or the bits of a half pair, so those kernels are written once per element size (byte, ushort, uint, ulong) instead of once per type.
public readonly partial struct Tensor
{
    private static int SizeOfRaw<T>() where T : unmanaged =>
        typeof(T) == typeof(byte) ? 1 : typeof(T) == typeof(ushort) ? 2 : typeof(T) == typeof(uint) ? 4 : typeof(T) == typeof(ulong) ? 8 :
        throw new ArgumentException($"{typeof(T).Name} is not an element-size carrier (byte, ushort, uint, ulong).");

    /// <summary>The elements of a contiguous tensor as raw carriers of the tensor's element size.</summary>
    internal ReadOnlySpan<T> RawData<T>() where T : unmanaged
    {
        ThrowIfEmptyOrBlocked();
        CheckRaw<T>();
        ReadOnlySpan<T> all = _buffer!.GetSpan<T>(_generation);
        int n = Length;
        return n == 0 ? default : all.Slice(_offset, n);
    }

    /// <summary>Writable form of <see cref="RawData{T}"/>; bumps the storage version.</summary>
    internal Span<T> RawMutableData<T>() where T : unmanaged
    {
        ThrowIfEmptyOrBlocked();
        CheckRaw<T>();
        Span<T> all = _buffer!.GetSpan<T>(_generation);
        _buffer.Touch();
        int n = Length;
        return n == 0 ? default : all.Slice(_offset, n);
    }

    /// <summary>The bytes of a contiguous tensor, writable; bumps the storage version.</summary>
    internal Span<byte> RawMutableBytes()
    {
        ThrowIfEmptyOrBlocked();
        if (!IsContiguous)
        {
            throw new InvalidOperationException("The tensor is not contiguous.");
        }

        Span<byte> all = _buffer!.GetSpan<byte>(_generation);
        _buffer.Touch();
        int size = ElementSize;
        return Length == 0 ? default : all.Slice(_offset * size, Length * size);
    }

    private void CheckRaw<T>() where T : unmanaged
    {
        if (SizeOfRaw<T>() != ElementSize)
        {
            throw new ArgumentException($"A {_dtype} element is {ElementSize} bytes, not {SizeOfRaw<T>()}.");
        }

        if (!IsContiguous)
        {
            throw new InvalidOperationException("The tensor is not contiguous.");
        }
    }
}
