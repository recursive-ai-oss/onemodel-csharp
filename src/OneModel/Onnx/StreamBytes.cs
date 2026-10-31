using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;

namespace RecursiveAI.OneModel.Onnx;

/// <summary>Reads a whole stream with a size limit, so an endless or enormous stream ends in a typed error rather than in memory exhaustion.</summary>
internal static class StreamBytes
{
    private const int FirstChunk = 1 << 16;
    private const int LastChunk = 1 << 23;

    /// <summary>
    /// Reads <paramref name="stream"/> to its end. Fails through <paramref name="tooLarge"/> (given the message and the first bytes
    /// of the stream, so the caller can pick the exception type for the format) as soon as more than
    /// <paramref name="maxBytes"/> have arrived (or the stream says it holds more), and when the memory for the bytes cannot be had.
    /// Peak memory is about twice the bytes read (the pieces, then the one array).
    /// </summary>
    public static byte[] ReadAll(Stream stream, long maxBytes, Func<string, byte[], Exception> tooLarge, CancellationToken cancellationToken)
    {
        long limit = Math.Min(maxBytes, Array.MaxLength);
        string over = $"The stream holds more than {limit} bytes (ModelLoadOptions.MaxModelBytes); keep large weights in external data files.";
        try
        {
            if (stream.CanSeek)
            {
                long remaining = stream.Length - stream.Position;
                if (remaining > limit)
                {
                    throw tooLarge(over, []);
                }

                var exact = new byte[Math.Max(remaining, 0)];
                int filled = stream.ReadAtLeast(exact, exact.Length, throwOnEndOfStream: false);
                if (filled == exact.Length)
                {
                    // A stream may have grown since Length was read; one more byte means it did.
                    int extra = stream.ReadByte();
                    if (extra < 0)
                    {
                        return exact;
                    }

                    var grown = new List<byte[]> { exact, new byte[] { (byte)extra } };
                    return Collect(stream, grown, exact.Length + 1L, limit, tooLarge, over, cancellationToken);
                }

                return exact.AsSpan(0, filled).ToArray();
            }

            return Collect(stream, [], 0, limit, tooLarge, over, cancellationToken);
        }
        catch (OutOfMemoryException)
        {
            throw tooLarge($"There is not enough memory to hold the stream (at most {limit} bytes allowed); use a path so the file can be mapped, or set a smaller ModelLoadOptions.MaxModelBytes.", []);
        }
    }

    private static byte[] Collect(Stream stream, List<byte[]> pieces, long total, long limit, Func<string, byte[], Exception> tooLarge, string over, CancellationToken ct)
    {
        int chunk = FirstChunk;
        while (true)
        {
            ct.ThrowIfCancellationRequested();
            var buffer = new byte[chunk];
            int got = stream.ReadAtLeast(buffer, buffer.Length, throwOnEndOfStream: false);
            if (got == 0)
            {
                break;
            }

            total += got;
            if (total > limit)
            {
                byte[] first = pieces.Count > 0 ? pieces[0] : buffer;
                throw tooLarge(over, first.AsSpan(0, Math.Min(8, first.Length)).ToArray());
            }

            pieces.Add(got == buffer.Length ? buffer : buffer.AsSpan(0, got).ToArray());
            if (got < buffer.Length)
            {
                break;
            }

            chunk = Math.Min(chunk * 2, LastChunk);
        }

        var all = new byte[total];
        long at = 0;
        foreach (byte[] p in pieces)
        {
            p.CopyTo(all, at);
            at += p.Length;
        }

        return all;
    }
}
