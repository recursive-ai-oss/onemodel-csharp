using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Win32.SafeHandles;

namespace RecursiveAI.OneModel.Training.Data;

/// <summary>An indexable collection of samples. With background workers, <see cref="GetSample"/> is called from several threads at once and must be safe for that.</summary>
/// <typeparam name="T">The sample type.</typeparam>
public interface IDataset<out T>
{
    /// <summary>The number of samples, at least 0.</summary>
    int Count { get; }

    /// <summary>Returns sample <paramref name="index"/>. The same index always gives the same sample.</summary>
    /// <param name="index">In [0, <see cref="Count"/>).</param>
    T GetSample(int index);
}

/// <summary>A dataset over an in-memory list. Reads are thread-safe as long as nobody modifies the list.</summary>
/// <typeparam name="T">The sample type.</typeparam>
public sealed class ListDataset<T> : IDataset<T>
{
    private readonly IReadOnlyList<T> _items;

    /// <summary>Wraps <paramref name="items"/> without copying.</summary>
    /// <exception cref="ArgumentNullException"><paramref name="items"/> is null.</exception>
    public ListDataset(IReadOnlyList<T> items) => _items = items ?? throw new ArgumentNullException(nameof(items));

    /// <inheritdoc/>
    public int Count => _items.Count;

    /// <inheritdoc/>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is outside the list.</exception>
    public T GetSample(int index) => _items[index];
}

/// <summary>Several datasets presented as one, in order: indices [0, n₀) go to the first, [n₀, n₀+n₁) to the second, and so on.</summary>
/// <typeparam name="T">The sample type.</typeparam>
public sealed class ConcatDataset<T> : IDataset<T>
{
    private readonly IDataset<T>[] _parts;
    private readonly int[] _starts;

    /// <summary>Joins <paramref name="parts"/>.</summary>
    /// <exception cref="ArgumentException">There are no parts, or the total exceeds 2^31-1 samples.</exception>
    public ConcatDataset(params IDataset<T>[] parts)
    {
        ArgumentNullException.ThrowIfNull(parts);
        if (parts.Length == 0) throw new ArgumentException("At least one dataset is needed.", nameof(parts));
        _parts = (IDataset<T>[])parts.Clone();
        _starts = new int[parts.Length + 1];
        long total = 0;
        for (int i = 0; i < parts.Length; i++)
        {
            ArgumentNullException.ThrowIfNull(parts[i]);
            _starts[i] = (int)total;
            total += parts[i].Count;
            if (total > int.MaxValue) throw new ArgumentException("The datasets together hold more than 2^31-1 samples.", nameof(parts));
        }
        _starts[parts.Length] = (int)total;
    }

    /// <summary>The first global index of each part, followed by the total count.</summary>
    public ReadOnlySpan<int> Starts => _starts;

    /// <inheritdoc/>
    public int Count => _starts[^1];

    /// <inheritdoc/>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is outside [0, <see cref="Count"/>).</exception>
    public T GetSample(int index)
    {
        if ((uint)index >= (uint)Count) throw new ArgumentOutOfRangeException(nameof(index));
        int lo = 0, hi = _parts.Length - 1;
        while (lo < hi) // the last part whose start is <= index; empty parts are skipped because the next start is equal
        {
            int mid = (lo + hi + 1) >> 1;
            if (_starts[mid] <= index) lo = mid; else hi = mid - 1;
        }
        return _parts[lo].GetSample(index - _starts[lo]);
    }
}

/// <summary>A dataset that applies a function to every sample of another. The function must be pure and thread-safe.</summary>
/// <typeparam name="TIn">The source sample type.</typeparam>
/// <typeparam name="TOut">The result sample type.</typeparam>
public sealed class MapDataset<TIn, TOut> : IDataset<TOut>
{
    private readonly IDataset<TIn> _source;
    private readonly Func<TIn, TOut> _map;

    /// <summary>Maps <paramref name="source"/> through <paramref name="map"/>.</summary>
    public MapDataset(IDataset<TIn> source, Func<TIn, TOut> map)
    {
        _source = source ?? throw new ArgumentNullException(nameof(source));
        _map = map ?? throw new ArgumentNullException(nameof(map));
    }

    /// <inheritdoc/>
    public int Count => _source.Count;

    /// <inheritdoc/>
    public TOut GetSample(int index) => _map(_source.GetSample(index));
}

/// <summary>
/// A dataset of fixed-size records in a file, read on demand with positional reads (safe from many threads, nothing cached).
/// Record <c>i</c> occupies bytes [<c>headerBytes + i·recordSize</c>, <c>headerBytes + (i+1)·recordSize</c>).
/// </summary>
/// <typeparam name="T">The sample type the decoder produces.</typeparam>
public sealed class RecordFileDataset<T> : IDataset<T>, IDisposable
{
    private readonly SafeFileHandle _handle;
    private readonly int _recordSize;
    private readonly long _header;
    private readonly Decoder _decode;

    /// <summary>Decodes one record. The span is valid only during the call.</summary>
    /// <param name="record">The record bytes.</param>
    /// <param name="index">The record's index.</param>
    public delegate T Decoder(ReadOnlySpan<byte> record, int index);

    /// <summary>Opens <paramref name="path"/> for reading.</summary>
    /// <param name="path">The file.</param>
    /// <param name="recordSize">Bytes per record, at least 1.</param>
    /// <param name="decode">Turns record bytes into a sample.</param>
    /// <param name="headerBytes">Bytes to skip at the start, default 0.</param>
    /// <exception cref="IOException">The file cannot be opened.</exception>
    /// <exception cref="InvalidDataException">The data after the header is not a whole number of records, or holds more than 2^31-1.</exception>
    public RecordFileDataset(string path, int recordSize, Decoder decode, long headerBytes = 0)
    {
        ArgumentNullException.ThrowIfNull(path);
        ArgumentNullException.ThrowIfNull(decode);
        ArgumentOutOfRangeException.ThrowIfLessThan(recordSize, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(headerBytes);
        _handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read, FileOptions.RandomAccess);
        try
        {
            long length = RandomAccess.GetLength(_handle) - headerBytes;
            if (length < 0 || length % recordSize != 0 || length / recordSize > int.MaxValue)
                throw new InvalidDataException($"'{path}' holds {length} bytes after the header, which is not a whole number (at most 2^31-1) of {recordSize}-byte records.");
            Count = (int)(length / recordSize);
        }
        catch
        {
            _handle.Dispose();
            throw;
        }
        _recordSize = recordSize;
        _header = headerBytes;
        _decode = decode;
    }

    /// <inheritdoc/>
    public int Count { get; }

    /// <inheritdoc/>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is outside [0, <see cref="Count"/>).</exception>
    /// <exception cref="EndOfStreamException">The file got shorter after it was opened.</exception>
    public T GetSample(int index)
    {
        if ((uint)index >= (uint)Count) throw new ArgumentOutOfRangeException(nameof(index));
        // Records up to 1 KiB are read into the stack; larger ones take a per-call array (the decoder usually allocates its sample anyway).
        if (_recordSize <= 1024)
        {
            Span<byte> buf = stackalloc byte[1024];
            Span<byte> rec = buf.Slice(0, _recordSize);
            Read(rec, index);
            return _decode(rec, index);
        }
        byte[] big = new byte[_recordSize];
        Read(big, index);
        return _decode(big, index);
    }

    private void Read(Span<byte> dst, int index)
    {
        int got = RandomAccess.Read(_handle, dst, _header + (long)index * _recordSize);
        if (got != dst.Length) throw new EndOfStreamException($"Record {index} is truncated: {got} of {dst.Length} bytes.");
    }

    /// <summary>Closes the file.</summary>
    public void Dispose() => _handle.Dispose();
}
