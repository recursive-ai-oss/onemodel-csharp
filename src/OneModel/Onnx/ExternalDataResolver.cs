using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using RecursiveAI.OneModel.Graph;

namespace RecursiveAI.OneModel.Onnx;

/// <summary>
/// Finds the files that hold external tensor data (<c>TensorProto.external_data</c>, ONNX IR.md "External Data"). A location is
/// accepted only if it is a relative path with no <c>..</c> segment whose fully resolved target, after following every symbolic link,
/// is a file inside the model's directory. The check runs when the file is opened; a link swapped in afterwards is not defended
/// against.
/// </summary>
internal sealed class ExternalDataResolver : IDisposable
{
    private readonly string _directory;
    private readonly string _realDirectory;
    private readonly bool _allowMap;
    private readonly Dictionary<string, ByteSource?> _files = new(StringComparer.Ordinal);
    private readonly List<IDisposable> _owned = [];

    public ExternalDataResolver(string directory, bool allowMap)
    {
        _directory = Path.GetFullPath(directory);
        _realDirectory = ResolveReal(_directory, 0);
        _allowMap = allowMap;
    }

    public IReadOnlyList<IDisposable> Owned => _owned;

    /// <summary>The arguments are the tensor's <c>location</c>, <c>offset</c> and <c>length</c> entries (null when absent) and the byte size its shape and type need.</summary>
    public ByteSource Resolve(string? location, string? offsetText, string? lengthText, long expectedBytes, ParseContext ctx, out long offset, out int length)
    {
        if (string.IsNullOrEmpty(location))
        {
            throw ctx.Fail(OnnxErrorKind.ExternalData, "External data has no 'location' entry.");
        }

        long off = 0;
        if (offsetText is not null && (!long.TryParse(offsetText, NumberStyles.None, CultureInfo.InvariantCulture, out off)))
        {
            throw ctx.Fail(OnnxErrorKind.ExternalData, $"External data offset '{Shorten(offsetText)}' is not a non-negative integer.");
        }

        long len = expectedBytes;
        if (lengthText is not null)
        {
            if (!long.TryParse(lengthText, NumberStyles.None, CultureInfo.InvariantCulture, out len))
            {
                throw ctx.Fail(OnnxErrorKind.ExternalData, $"External data length '{Shorten(lengthText)}' is not a non-negative integer.");
            }

            if (len != expectedBytes)
            {
                throw ctx.Fail(OnnxErrorKind.ExternalData, $"External data length is {len} bytes but the tensor's shape and type need {expectedBytes}.");
            }
        }

        if (len > int.MaxValue)
        {
            throw ctx.Fail(OnnxErrorKind.Unsupported, $"External tensor of {len} bytes is larger than the 2^31-1 bytes a tensor can hold.");
        }

        string real = ConfinedPath(location, ctx);
        if (!_files.TryGetValue(real, out ByteSource? source))
        {
            source = Open(real, ctx, off, len, out bool mapped);
            if (mapped)
            {
                _files[real] = source;
                _owned.Add(source!);
            }
            else
            {
                // A fallback source holds only the requested range, so it is not shared.
                _owned.Add(source!);
                offset = 0;
                length = (int)len;
                return source!;
            }
        }

        if (off > source!.Length - len)
        {
            throw ctx.Fail(OnnxErrorKind.ExternalData,
                $"External data range {off}+{len} is outside '{location}', which has {source.Length} bytes.");
        }

        offset = off;
        length = (int)len;
        return source;
    }

    private ByteSource Open(string real, ParseContext ctx, long off, long len, out bool mapped)
    {
        try
        {
            if (_allowMap)
            {
                try
                {
                    ByteSource s = ByteSource.OpenFile(real, allowMap: true, mapThreshold: 1, readWholeIfUnmapped: false);
                    mapped = s.IsMapped;
                    if (mapped)
                    {
                        return s;
                    }

                    s.Dispose();
                }
                catch (NotSupportedException)
                {
                    // Mapping is unavailable here; read the range below.
                }
            }

            mapped = false;
            using SafeHandleScope h = new(real);
            long fileLength = h.Length;
            if (off > fileLength - len)
            {
                throw ctx.Fail(OnnxErrorKind.ExternalData, $"External data range {off}+{len} is outside the {fileLength}-byte file.");
            }

            var bytes = new byte[len];
            int done = 0;
            while (done < bytes.Length)
            {
                int n = h.Read(bytes.AsSpan(done), off + done);
                if (n <= 0)
                {
                    throw ctx.Fail(OnnxErrorKind.ExternalData, "External data file ended while reading.");
                }

                done += n;
            }

            return ByteSource.FromArray(bytes);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            throw new OnnxImportException(OnnxErrorKind.ExternalData, "External data file cannot be read: " + e.Message, ctx.InputName, -1, ctx.Path(), e);
        }
    }

    private readonly struct SafeHandleScope : IDisposable
    {
        private readonly Microsoft.Win32.SafeHandles.SafeFileHandle _handle;

        public SafeHandleScope(string path) => _handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete);

        public long Length => RandomAccess.GetLength(_handle);

        public int Read(Span<byte> buffer, long offset) => RandomAccess.Read(_handle, buffer, offset);

        public void Dispose() => _handle.Dispose();
    }

    /// <summary>Validates <paramref name="location"/> and returns the fully resolved path of the file it names.</summary>
    private string ConfinedPath(string location, ParseContext ctx)
    {
        string shown = Shorten(location);
        if (location.Contains('\0', StringComparison.Ordinal))
        {
            throw ctx.Fail(OnnxErrorKind.ExternalData, "External data location contains a NUL character.");
        }

        if (location[0] is '/' or '\\' || (location.Length >= 2 && location[1] == ':') || Path.IsPathRooted(location))
        {
            throw ctx.Fail(OnnxErrorKind.ExternalData, $"External data location '{shown}' is an absolute path; it must be relative to the model's directory.");
        }

        foreach (string segment in location.Split('/', '\\'))
        {
            if (segment == "..")
            {
                throw ctx.Fail(OnnxErrorKind.ExternalData, $"External data location '{shown}' contains '..'.");
            }
        }

        string full = Path.GetFullPath(Path.Combine(_directory, location));
        string real = ResolveReal(full, 0);
        StringComparison cmp = OperatingSystem.IsWindows() || OperatingSystem.IsMacOS() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        string prefix = _realDirectory.EndsWith(Path.DirectorySeparatorChar) ? _realDirectory : _realDirectory + Path.DirectorySeparatorChar;
        if (!real.StartsWith(prefix, cmp))
        {
            throw ctx.Fail(OnnxErrorKind.ExternalData, $"External data location '{shown}' resolves outside the model's directory.");
        }

        if (!File.Exists(real))
        {
            throw ctx.Fail(OnnxErrorKind.ExternalData, $"External data file '{shown}' does not exist.");
        }

        return real;
    }

    /// <summary>Resolves symbolic links in every component of an absolute path.</summary>
    private static string ResolveReal(string fullPath, int hops)
    {
        if (hops > 16)
        {
            throw new IOException("Too many levels of symbolic links.");
        }

        string root = Path.GetPathRoot(fullPath) ?? "";
        string current = root;
        foreach (string segment in fullPath[root.Length..].Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))
        {
            if (segment.Length == 0 || segment == ".")
            {
                continue;
            }

            string next = Path.Combine(current, segment);
            FileSystemInfo info = new FileInfo(next);
            if (info.LinkTarget is not null)
            {
                FileSystemInfo? target = info.ResolveLinkTarget(returnFinalTarget: true);
                next = target is null ? next : ResolveReal(Path.GetFullPath(target.FullName), hops + 1);
            }

            current = next;
        }

        return current;
    }

    private static string Shorten(string s) => s.Length <= 120 ? s : string.Concat(s.AsSpan(0, 117), "...");

    public void Dispose()
    {
        foreach (IDisposable d in _owned)
        {
            d.Dispose();
        }

        _owned.Clear();
    }
}
