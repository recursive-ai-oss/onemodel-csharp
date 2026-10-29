using System;
using System.IO;
using System.IO.MemoryMappedFiles;

namespace RecursiveAI.OneModel.Graph;

/// <summary>
/// Read-only bytes that tensor data points into: an array, or a file mapped into memory. This is one of the two places in the
/// importer that uses pointers: a mapped view is addressed through its base pointer.
/// Bounds: every span handed out comes from <see cref="GetSpan"/>, which checks <c>0 &lt;= offset</c>, <c>0 &lt;= length</c> and
/// <c>offset + length &lt;= Length</c> in 64-bit arithmetic against the mapped length before building the span, and refuses once
/// the source is disposed. A span must not be used after the source is disposed.
/// </summary>
internal sealed unsafe class ByteSource : IDisposable
{
    private readonly byte[]? _array;
    private readonly MemoryMappedFile? _file;
    private readonly MemoryMappedViewAccessor? _view;
    private readonly byte* _pointer;
    private volatile bool _disposed;

    private ByteSource(byte[] array)
    {
        _array = array;
        Length = array.Length;
    }

    private ByteSource(MemoryMappedFile file, MemoryMappedViewAccessor view, byte* pointer, long length)
    {
        _file = file;
        _view = view;
        _pointer = pointer;
        Length = length;
    }

    /// <summary>The number of bytes.</summary>
    public long Length { get; }

    /// <summary>True when the bytes are a mapped file.</summary>
    public bool IsMapped => _file is not null;

    /// <summary>Wraps an array without copying it.</summary>
    public static ByteSource FromArray(byte[] array) => new(array);

    /// <summary>
    /// Opens a file for reading. With <paramref name="allowMap"/> the file is mapped when it is at least <paramref name="mapThreshold"/>
    /// bytes and the platform supports mapping; otherwise, and when mapping fails, it is read into an array (files over 2^31-1 bytes
    /// are then refused by the caller's range check, since an array cannot hold them).
    /// </summary>
    public static ByteSource OpenFile(string path, bool allowMap, long mapThreshold, bool readWholeIfUnmapped)
    {
        var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete, 1, FileOptions.None);
        try
        {
            long length = fs.Length;
            if (allowMap && length >= mapThreshold && length > 0 && !OperatingSystem.IsBrowser())
            {
                try
                {
                    // The mapping outlives the stream: the stream is closed below once the view exists.
                    MemoryMappedFile mmf = MemoryMappedFile.CreateFromFile(fs, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.None, leaveOpen: true);
                    MemoryMappedViewAccessor? view = null;
                    try
                    {
                        view = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
                        byte* p = null;
                        view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
                        return new ByteSource(mmf, view, p + view.PointerOffset, length);
                    }
                    catch
                    {
                        view?.Dispose();
                        mmf.Dispose();
                        throw;
                    }
                }
                catch (Exception e) when (e is IOException or UnauthorizedAccessException or NotSupportedException or PlatformNotSupportedException or ArgumentException)
                {
                    // Fall through and read instead; start again from the beginning of the file.
                }
            }

            if (!readWholeIfUnmapped)
            {
                throw new NotSupportedException("The file could not be mapped.");
            }

            if (length > Array.MaxLength)
            {
                throw new NotSupportedException($"The file is {length} bytes; at most {Array.MaxLength} can be read into memory.");
            }

            var bytes = new byte[length];
            fs.Position = 0;
            fs.ReadExactly(bytes);
            return new ByteSource(bytes);
        }
        finally
        {
            fs.Dispose();
        }
    }

    /// <summary>Returns <paramref name="length"/> bytes starting at <paramref name="offset"/>.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The range is not inside the source.</exception>
    /// <exception cref="ObjectDisposedException">The source was disposed.</exception>
    public ReadOnlySpan<byte> GetSpan(long offset, int length)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(ByteSource), "The model was disposed; tensor data it owned is gone.");
        }

        if (offset < 0 || length < 0 || offset > Length - length)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), $"Range {offset}+{length} is outside the {Length} bytes of the source.");
        }

        if (_array is not null)
        {
            return new ReadOnlySpan<byte>(_array, (int)offset, length);
        }

        return new ReadOnlySpan<byte>(_pointer + offset, length);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        if (_view is not null)
        {
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
            _view.Dispose();
            _file?.Dispose();
        }
    }
}
