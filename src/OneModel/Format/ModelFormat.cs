using System;
using System.IO;
using System.Threading;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;

namespace RecursiveAI.OneModel.Format;

/// <summary>
/// The OneModel model file: a 64-byte header, a JSON manifest describing the graph, and the weights as raw little-endian bytes
/// aligned to 64 bytes, so a large file can be memory-mapped and each weight used in place. Format version 1; the layout is
/// unstable until 1.0. The normative description is <c>docs/format.md</c>.
/// <para>
/// What is saved is the imported graph, before optimization passes: the same model every session builds from, so a saved file
/// optimizes and runs exactly as the original ONNX file does. Weights stored in ONNX external data files are copied into the
/// one file.
/// </para>
/// <para>
/// <see cref="Model.Load(string, SessionOptions?, CancellationToken)"/> and <see cref="Session.Load(string, SessionOptions?, CancellationToken)"/>
/// recognise these files by their first eight bytes, so a saved model loads through the same entry points as an ONNX file.
/// </para>
/// </summary>
public static class ModelFormat
{
    /// <summary>The format name written into the manifest.</summary>
    public const string Name = "onemodel";

    /// <summary>The format version this library writes and reads.</summary>
    public const int Version = 1;

    /// <summary>Files larger than this are memory-mapped when loaded from a path; smaller files are read into memory.</summary>
    private const long MapThreshold = 1 << 20;

    /// <summary>True when <paramref name="start"/> begins with the format's magic bytes (<c>ONEMODEL</c>). Pass at least the first eight bytes of the file.</summary>
    public static bool IsModelFormat(ReadOnlySpan<byte> start) => start.StartsWith(FormatWriter.Magic);

    /// <summary>True when the file at <paramref name="path"/> begins with the format's magic bytes. A missing or unreadable file gives false.</summary>
    /// <param name="path">The file to look at.</param>
    public static bool IsModelFormatFile(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        try
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete, 1);
            Span<byte> head = new byte[FormatWriter.Magic.Length];
            return fs.ReadAtLeast(head, head.Length, throwOnEndOfStream: false) == head.Length && IsModelFormat(head);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            return false;
        }
    }

    /// <summary>
    /// Saves an imported model, as it was imported (before optimization passes). Tensor bytes are read from the model's own views,
    /// so the model must not be disposed. The same model always produces the same bytes.
    /// </summary>
    /// <param name="model">The model to save.</param>
    /// <param name="path">The file to create or overwrite. If saving fails the partial file is deleted.</param>
    /// <param name="cancellationToken">Checked between graphs and tensors.</param>
    /// <exception cref="ModelFormatException">The file cannot be written (<see cref="ModelFormatErrorKind.Io"/>) or the model has an attribute without a type.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public static void Save(IrModel model, string path, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(path);
        FileStream fs;
        try
        {
            fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            throw new ModelFormatException(ModelFormatErrorKind.Io, "The file cannot be created: " + e.Message, path, -1, null, e);
        }

        bool ok = false;
        try
        {
            using (fs)
            {
                FormatWriter.Write(model, fs, cancellationToken);
            }

            ok = true;
        }
        catch (IOException e)
        {
            throw new ModelFormatException(ModelFormatErrorKind.Io, "The file cannot be written: " + e.Message, path, -1, null, e);
        }
        finally
        {
            if (!ok)
            {
                try
                {
                    File.Delete(path);
                }
                catch (Exception e) when (e is IOException or UnauthorizedAccessException)
                {
                    // The partial file stays; the original error is the one worth reporting.
                }
            }
        }
    }

    /// <summary>Saves an imported model to a stream, which is written sequentially and need not be seekable. See <see cref="Save(IrModel, string, CancellationToken)"/>.</summary>
    /// <param name="model">The model to save.</param>
    /// <param name="output">Where to write.</param>
    /// <param name="cancellationToken">Checked between graphs and tensors.</param>
    /// <exception cref="ModelFormatException">The model has an attribute without a type.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public static void Save(IrModel model, Stream output, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(output);
        FormatWriter.Write(model, output, cancellationToken);
    }

    /// <summary>Converts an ONNX file to the OneModel format: imports it (reading external data next to it) and saves it.</summary>
    /// <param name="onnxPath">The <c>.onnx</c> file.</param>
    /// <param name="outputPath">The file to create.</param>
    /// <param name="cancellationToken">Checked while importing and saving.</param>
    /// <exception cref="OnnxImportException">The ONNX file is not a valid model.</exception>
    /// <exception cref="ModelFormatException">The output cannot be written.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public static void Convert(string onnxPath, string outputPath, CancellationToken cancellationToken = default)
    {
        using IrModel model = OnnxImporter.Load(onnxPath, null, cancellationToken);
        Save(model, outputPath, cancellationToken);
    }

    /// <summary>
    /// Loads a model file. Large files are memory-mapped and weights stay views of the file until a session copies them, as with ONNX
    /// files. Dispose the model when done.
    /// </summary>
    /// <param name="path">The file.</param>
    /// <param name="cancellationToken">Checked while reading the manifest.</param>
    /// <returns>The model, ready for <see cref="Session.Create(IrModel, SessionOptions?, CancellationToken)"/>.</returns>
    /// <exception cref="ModelFormatException">The file is truncated, corrupt, of a newer version or inconsistent; the message names the field.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public static IrModel Load(string path, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(path);
        ByteSource source;
        try
        {
            source = ByteSource.OpenFile(path, allowMap: true, MapThreshold, readWholeIfUnmapped: true);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or NotSupportedException or ArgumentException)
        {
            throw new ModelFormatException(ModelFormatErrorKind.Io, "The model file cannot be read: " + e.Message, path, -1, null, e);
        }

        return ReadOwning(source, path, cancellationToken);
    }

    /// <summary>Loads a model from a stream, read to its end.</summary>
    /// <param name="stream">The stream, positioned at the start of the file.</param>
    /// <param name="cancellationToken">Checked while reading.</param>
    /// <exception cref="ModelFormatException">As <see cref="Load(string, CancellationToken)"/>.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public static IrModel Load(Stream stream, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(stream);
        return Load(ReadAll(stream, cancellationToken), cancellationToken);
    }

    /// <summary>Loads a model from bytes already in memory. The array is used in place and must not change while the model is in use.</summary>
    /// <param name="data">The whole file.</param>
    /// <param name="cancellationToken">Checked while reading the manifest.</param>
    /// <exception cref="ModelFormatException">As <see cref="Load(string, CancellationToken)"/>.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public static IrModel Load(byte[] data, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(data);
        return ReadOwning(ByteSource.FromArray(data), "<memory>", cancellationToken);
    }

    /// <summary>Loads an ONNX or OneModel file, whichever <paramref name="path"/> is, chosen by the first eight bytes.</summary>
    internal static IrModel LoadAny(string path, CancellationToken ct) =>
        IsModelFormatFile(path) ? Load(path, ct) : OnnxImporter.Load(path, null, ct);

    /// <summary>Loads an ONNX or OneModel stream, whichever it is, chosen by the first eight bytes.</summary>
    internal static IrModel LoadAny(Stream stream, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (stream.CanSeek)
        {
            long start = stream.Position;
            Span<byte> head = new byte[FormatWriter.Magic.Length];
            int n = stream.ReadAtLeast(head, head.Length, throwOnEndOfStream: false);
            stream.Position = start;
            return n == head.Length && IsModelFormat(head) ? Load(stream, ct) : OnnxImporter.Load(stream, null, ct);
        }

        byte[] bytes = ReadAll(stream, ct);
        return IsModelFormat(bytes) ? Load(bytes, ct) : OnnxImporter.Load(bytes, null, ct);
    }

    private static IrModel ReadOwning(ByteSource source, string name, CancellationToken ct)
    {
        try
        {
            return FormatReader.Read(source, name, ct);
        }
        catch
        {
            source.Dispose();
            throw;
        }
    }

    private static byte[] ReadAll(Stream stream, CancellationToken ct)
    {
        try
        {
            using var ms = new MemoryStream();
            var buffer = new byte[81920];
            int n;
            while ((n = stream.Read(buffer, 0, buffer.Length)) > 0)
            {
                ct.ThrowIfCancellationRequested();
                if (ms.Length + n > Array.MaxLength)
                {
                    throw new ModelFormatException(ModelFormatErrorKind.Unsupported, "The stream is larger than 2 GB; load large models from a path.", "<stream>");
                }

                ms.Write(buffer, 0, n);
            }

            return ms.ToArray();
        }
        catch (Exception e) when (e is IOException or NotSupportedException or ObjectDisposedException)
        {
            throw new ModelFormatException(ModelFormatErrorKind.Io, "The stream cannot be read: " + e.Message, "<stream>", -1, null, e);
        }
    }
}
