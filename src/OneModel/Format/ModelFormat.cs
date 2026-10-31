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
/// <see cref="Model.Load(string, ModelOptions?, ModelLoadOptions?, CancellationToken)"/> and <see cref="Session.Load(string, ModelOptions?, ModelLoadOptions?, CancellationToken)"/>
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
    /// <exception cref="ModelFormatException">The model has an attribute without a type.</exception>
    /// <exception cref="IOException">The file cannot be created or written (<see cref="DirectoryNotFoundException"/>, a full disk, ...); <see cref="UnauthorizedAccessException"/> when access is denied.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public static void Save(IrModel model, string path, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(path);
        FileStream fs = new(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16);
        bool ok = false;
        try
        {
            using (fs)
            {
                FormatWriter.Write(model, fs, cancellationToken);
            }

            ok = true;
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

    /// <summary>
    /// Converts a model file to the OneModel format: imports it (an ONNX file, reading external data next to it; a file already in the
    /// OneModel format is read and rewritten, which normalizes it) and saves it.
    /// </summary>
    /// <param name="inputPath">The <c>.onnx</c> or OneModel file.</param>
    /// <param name="outputPath">The file to create. It must not be the input file: the input is read lazily and overwriting it would destroy the model.</param>
    /// <param name="loadOptions">Limits for reading the input; defaults when null.</param>
    /// <param name="cancellationToken">Checked while importing and saving.</param>
    /// <exception cref="ArgumentException"><paramref name="outputPath"/> is the same file as <paramref name="inputPath"/>.</exception>
    /// <exception cref="OnnxImportException">The input is not a valid ONNX model.</exception>
    /// <exception cref="ModelFormatException">The input in the OneModel format is invalid.</exception>
    /// <exception cref="IOException">A file cannot be read or written.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public static void Convert(string inputPath, string outputPath, ModelLoadOptions? loadOptions = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputPath);
        ArgumentNullException.ThrowIfNull(outputPath);
        if (string.Equals(Path.GetFullPath(inputPath), Path.GetFullPath(outputPath), OperatingSystem.IsWindows() || OperatingSystem.IsMacOS() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal)
            || (File.Exists(inputPath) && File.Exists(outputPath) && SameFile(inputPath, outputPath)))
        {
            throw new ArgumentException("The output file is the input file; choose another path.", nameof(outputPath));
        }

        using IrModel model = LoadAny(inputPath, loadOptions, cancellationToken);
        Save(model, outputPath, cancellationToken);
    }

    // Catches the same file reached through a link or a different spelling.
    private static bool SameFile(string a, string b)
    {
        FileSystemInfo fa = new FileInfo(a), fb = new FileInfo(b);
        string ta = fa.ResolveLinkTarget(returnFinalTarget: true)?.FullName ?? fa.FullName;
        string tb = fb.ResolveLinkTarget(returnFinalTarget: true)?.FullName ?? fb.FullName;
        return string.Equals(ta, tb, OperatingSystem.IsWindows() || OperatingSystem.IsMacOS() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
    }

    /// <summary>
    /// Loads a model file. Large files are memory-mapped and weights stay views of the file until a session copies them, as with ONNX
    /// files. Dispose the model when done.
    /// </summary>
    /// <param name="path">The file.</param>
    /// <param name="loadOptions">Limits (<see cref="ModelLoadOptions.MaxModelBytes"/>) and <see cref="ModelLoadOptions.MemoryMap"/>; defaults when null.</param>
    /// <param name="cancellationToken">Checked while reading the manifest.</param>
    /// <returns>The model, ready for <see cref="Session.Create(IrModel, ModelOptions?, CancellationToken)"/>.</returns>
    /// <exception cref="ModelFormatException">The file is truncated, corrupt, of a newer version, inconsistent or over the size limit; the message names the field.</exception>
    /// <exception cref="IOException">The file is missing or cannot be read; <see cref="UnauthorizedAccessException"/> when access is denied.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public static IrModel Load(string path, ModelLoadOptions? loadOptions = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(path);
        loadOptions ??= new ModelLoadOptions();
        ByteSource source = ByteSource.OpenFile(path, loadOptions.MemoryMap, MapThreshold, readWholeIfUnmapped: true);
        return ReadOwning(source, path, loadOptions, cancellationToken);
    }

    /// <summary>Loads a model from a stream, read to its end (or until <see cref="ModelLoadOptions.MaxModelBytes"/> is passed, which is an error).</summary>
    /// <param name="stream">The stream, positioned at the start of the file.</param>
    /// <param name="loadOptions">Limits; defaults when null.</param>
    /// <param name="cancellationToken">Checked while reading.</param>
    /// <exception cref="ModelFormatException">As <see cref="Load(string, ModelLoadOptions?, CancellationToken)"/>.</exception>
    /// <exception cref="IOException">The stream cannot be read.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public static IrModel Load(Stream stream, ModelLoadOptions? loadOptions = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(stream);
        loadOptions ??= new ModelLoadOptions();
        return Load(ReadAll(stream, loadOptions, cancellationToken), loadOptions, cancellationToken);
    }

    /// <summary>Loads a model from bytes already in memory. The array is used in place and must not change while the model is in use.</summary>
    /// <param name="data">The whole file.</param>
    /// <param name="loadOptions">Limits; defaults when null.</param>
    /// <param name="cancellationToken">Checked while reading the manifest.</param>
    /// <exception cref="ModelFormatException">As <see cref="Load(string, ModelLoadOptions?, CancellationToken)"/>.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public static IrModel Load(byte[] data, ModelLoadOptions? loadOptions = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(data);
        return ReadOwning(ByteSource.FromArray(data), "<memory>", loadOptions ?? new ModelLoadOptions(), cancellationToken);
    }

    /// <summary>
    /// Says what kind of model file the first bytes of a file are, from the format's magic or the first protobuf key of an ONNX
    /// <c>ModelProto</c>. A look at the first bytes only: a file can be detected as ONNX and still fail to import.
    /// </summary>
    /// <param name="start">At least the first eight bytes of the file (fewer give <see cref="ModelFileKind.Unknown"/> unless they are a recognisable ONNX start).</param>
    public static ModelFileKind Detect(ReadOnlySpan<byte> start)
    {
        if (IsModelFormat(start))
        {
            return ModelFileKind.OneModel;
        }

        // ModelProto fields: ir_version(1) varint, producer_name(2), producer_version(3), domain(4), model_version(5) varint,
        // doc_string(6), graph(7), opset_import(8), metadata_props(14), training_info(20), functions(25).
        if (start.IsEmpty)
        {
            return ModelFileKind.Unknown;
        }

        return start[0] is 0x08 or 0x12 or 0x1a or 0x22 or 0x28 or 0x32 or 0x3a or 0x42 or 0x72
            || (start.Length > 1 && start[1] == 0x01 && start[0] is 0xa2 or 0xca)
            ? ModelFileKind.Onnx
            : ModelFileKind.Unknown;
    }

    /// <summary>Says what kind of model file <paramref name="path"/> is, by its first bytes (see <see cref="Detect(ReadOnlySpan{byte})"/>).</summary>
    /// <param name="path">The file.</param>
    /// <exception cref="IOException">The file is missing or cannot be read.</exception>
    public static ModelFileKind Detect(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete, 1);
        Span<byte> head = stackalloc byte[8];
        int n = fs.ReadAtLeast(head, head.Length, throwOnEndOfStream: false);
        return Detect(head[..n]);
    }

    /// <summary>Loads an ONNX or OneModel file, whichever <paramref name="path"/> is, chosen by the first eight bytes.</summary>
    internal static IrModel LoadAny(string path, ModelLoadOptions? options, CancellationToken ct) =>
        IsModelFormatFile(path) ? Load(path, options, ct) : OnnxImporter.Load(path, options, ct);

    /// <summary>Loads an ONNX or OneModel stream, whichever it is, chosen by the first eight bytes.</summary>
    internal static IrModel LoadAny(Stream stream, ModelLoadOptions? options, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(stream);
        options ??= new ModelLoadOptions();
        byte[] bytes = ReadAll(stream, options, ct, forAny: true);
        return IsModelFormat(bytes) ? Load(bytes, options, ct) : OnnxImporter.Load(bytes, options, ct);
    }

    private static IrModel ReadOwning(ByteSource source, string name, ModelLoadOptions options, CancellationToken ct)
    {
        try
        {
            if (source.Length > options.MaxModelBytes)
            {
                throw new ModelFormatException(ModelFormatErrorKind.Unsupported,
                    $"The model is {source.Length} bytes; the limit is {options.MaxModelBytes} (ModelLoadOptions.MaxModelBytes).", name);
            }

            return FormatReader.Read(source, name, ct);
        }
        catch
        {
            source.Dispose();
            throw;
        }
    }

    private static byte[] ReadAll(Stream stream, ModelLoadOptions options, CancellationToken ct, bool forAny = false) =>
        StreamBytes.ReadAll(stream, options.MaxModelBytes, (m, head) => !forAny || IsModelFormat(head)
            ? new ModelFormatException(ModelFormatErrorKind.Unsupported, m, "<stream>")
            : new OnnxImportException(OnnxErrorKind.Unsupported, m, "<stream>"), ct);
}
