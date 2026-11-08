using System;
using System.IO;
using System.IO.MemoryMappedFiles;

namespace RecursiveAI.OneModel.Graph;

/// <summary>
/// Read-only bytes that tensor data points into: caller-owned memory, an unmanaged stream or a mapped file.
/// Unmanaged sources are addressed through a captured base pointer and kept alive by the source.
/// Bounds: every span handed out comes from <see cref="GetSpan"/>, which checks <c>0 &lt;= offset</c>, <c>0 &lt;= length</c> and
/// <c>offset + length &lt;= Length</c> in 64-bit arithmetic against the source length before building the span, and refuses once
/// the source is disposed. A span must not be used after the source is disposed.
/// </summary>
internal sealed unsafe class ByteSource : IDisposable
{
    private readonly ReadOnlyMemory<byte>? _memory;
    private readonly UnmanagedMemoryStream? _stream;
    private readonly MemoryMappedFile? _file;
    private readonly MemoryMappedViewAccessor? _view;
    private readonly byte* _pointer;
    private volatile bool _disposed;

    private ByteSource(ReadOnlyMemory<byte> memory)
    {
        _memory = memory;
        Length = memory.Length;
    }

    private ByteSource(UnmanagedMemoryStream stream)
    {
        _stream = stream;
        _pointer = stream.PositionPointer;
        Length = stream.Length - stream.Position;
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

    /// <summary>Wraps caller-owned memory without copying or pinning it. Its owner must keep it valid and unchanged.</summary>
    public static ByteSource FromMemory(ReadOnlyMemory<byte> memory) => new(memory);

    /// <summary>Views the remaining unmanaged bytes without copying. The caller must keep the stream open and its storage unchanged.</summary>
    public static ByteSource? FromUnmanagedStream(Stream stream)
    {
        if (stream is UnmanagedMemoryStream unmanaged && unmanaged.CanRead && unmanaged.CanSeek)
        {
            if (unmanaged.Position >= unmanaged.Length)
            {
                return FromMemory(ReadOnlyMemory<byte>.Empty);
            }

            try
            {
                return new ByteSource(unmanaged);
            }
            catch (NotSupportedException)
            {
                // Some subclasses (notably mapped view streams) do not expose PositionPointer.
            }
        }

        return null;
    }

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
                throw new NotSupportedException($"The file is {length} bytes; at most {Array.MaxLength} can be read into memory. Load from a file path with MemoryMap enabled on a platform supporting memory mapping.");
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

        if (_memory is ReadOnlyMemory<byte> memory)
        {
            return memory.Span.Slice((int)offset, length);
        }

        if (_stream is not null && !_stream.CanRead)
        {
            throw new ObjectDisposedException(nameof(UnmanagedMemoryStream), "The caller-owned model stream was closed.");
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
