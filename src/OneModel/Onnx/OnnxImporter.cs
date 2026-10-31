using System;
using System.IO;
using System.Threading;
using RecursiveAI.OneModel.Graph;

namespace RecursiveAI.OneModel.Onnx;

/// <summary>
/// Reads ONNX models (IR version 3 and later, any opset) into the <see cref="IrModel"/> graph representation. The protobuf reader,
/// the ONNX message reader and the external-data resolver are this library's own; nothing is generated and nothing is loaded
/// that was not checked against the bytes that remain. Bad content produces <see cref="OnnxImportException"/> and nothing else; a
/// file that cannot be opened or read produces the BCL's own <see cref="IOException"/> family, unwrapped.
/// </summary>
public static class OnnxImporter
{
    /// <summary>The files at least this large are memory-mapped when <see cref="ModelLoadOptions.MemoryMap"/> is set; smaller ones are read.</summary>
    private const long MapThreshold = 1 << 20;

    /// <summary>Imports a model from a file. External data is looked up in the file's directory unless the options say otherwise.</summary>
    /// <param name="path">The <c>.onnx</c> file. It may be at most 2^31-1 bytes (protobuf's limit); larger models keep their weights in external files.</param>
    /// <param name="options">Import options; defaults when null.</param>
    /// <param name="cancellationToken">Checked between nodes and tensors.</param>
    /// <returns>The model. The caller disposes it, which unmaps the file.</returns>
    /// <exception cref="OnnxImportException">The file is not a valid, supported ONNX model, or is larger than <see cref="ModelLoadOptions.MaxModelBytes"/>.</exception>
    /// <exception cref="IOException">The file is missing or cannot be read (<see cref="FileNotFoundException"/>, <see cref="DirectoryNotFoundException"/>, ...); <see cref="UnauthorizedAccessException"/> when access is denied.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public static IrModel Load(string path, ModelLoadOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(path);
        options ??= new ModelLoadOptions();
        ByteSource source = ByteSource.OpenFile(path, options.MemoryMap, MapThreshold, readWholeIfUnmapped: true);

        string? directory = options.ExternalDataDirectory ?? Path.GetDirectoryName(Path.GetFullPath(path));
        return LoadCore(source, path, directory, options, cancellationToken);
    }

    /// <summary>Imports a model from a stream, which is read to its end. External data is an error unless <see cref="ModelLoadOptions.ExternalDataDirectory"/> is set.</summary>
    /// <param name="stream">A readable stream holding at most 2^31-1 bytes from its current position.</param>
    /// <param name="options">Import options; defaults when null.</param>
    /// <param name="cancellationToken">Checked between nodes and tensors, and while reading.</param>
    /// <returns>The model.</returns>
    /// <exception cref="OnnxImportException">The stream does not hold a valid, supported model, or holds more than <see cref="ModelLoadOptions.MaxModelBytes"/> (reading stops there).</exception>
    /// <exception cref="IOException">The stream cannot be read.</exception>
    public static IrModel Load(Stream stream, ModelLoadOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(stream);
        options ??= new ModelLoadOptions();
        byte[] bytes = StreamBytes.ReadAll(stream, options.MaxModelBytes,
            (m, _) => new OnnxImportException(OnnxErrorKind.Unsupported, m, "<stream>"), cancellationToken);

        return LoadCore(ByteSource.FromArray(bytes), "<stream>", options.ExternalDataDirectory, options, cancellationToken);
    }

    /// <summary>Imports a model from memory. The array is used in place, not copied: do not change it while the model is alive.</summary>
    /// <param name="data">The bytes of an <c>.onnx</c> file.</param>
    /// <param name="options">Import options; defaults when null.</param>
    /// <param name="cancellationToken">Checked between nodes and tensors.</param>
    /// <returns>The model.</returns>
    /// <exception cref="OnnxImportException">The bytes are not a valid, supported model.</exception>
    public static IrModel Load(byte[] data, ModelLoadOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(data);
        options ??= new ModelLoadOptions();
        return LoadCore(ByteSource.FromArray(data), "<memory>", options.ExternalDataDirectory, options, cancellationToken);
    }

    /// <summary>Reads one serialized <c>TensorProto</c> (the format of the <c>.pb</c> files in ONNX's test data). External data is not followed.</summary>
    /// <param name="data">The bytes of the message. They are copied only if tensor data is stored in typed fields; raw data points into <paramref name="data"/>, which must not change.</param>
    /// <returns>The tensor.</returns>
    /// <exception cref="OnnxImportException">The bytes are not a valid tensor.</exception>
    public static TensorData LoadTensor(byte[] data)
    {
        ArgumentNullException.ThrowIfNull(data);
        var ctx = new ParseContext("<tensor>", 128, null, CancellationToken.None) { Main = ByteSource.FromArray(data), Root = "TensorProto" };
        var r = new ProtoReader(data, 0, ctx, 0);
        return OnnxParser.ParseTensor(ref r);
    }

    private static IrModel LoadCore(ByteSource source, string inputName, string? externalDirectory, ModelLoadOptions options, CancellationToken ct)
    {
        ExternalDataResolver? resolver = null;
        try
        {
            if (source.Length > Math.Min(options.MaxModelBytes, Array.MaxLength))
            {
                throw new OnnxImportException(OnnxErrorKind.Unsupported,
                    $"The model is {source.Length} bytes; the limit is {Math.Min(options.MaxModelBytes, Array.MaxLength)} (a protobuf message holds at most 2^31-1). Keep large weights in external data.", inputName);
            }

            try
            {
                resolver = externalDirectory is null ? null : new ExternalDataResolver(externalDirectory, options.MemoryMap);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
            {
                throw new OnnxImportException(OnnxErrorKind.ExternalData, "The external data directory cannot be used: " + e.Message, inputName, -1, null, e);
            }

            var ctx = new ParseContext(inputName, options.MaxNesting, resolver, ct) { Main = source };
            var r = new ProtoReader(source.GetSpan(0, (int)source.Length), 0, ctx, 0);
            IrModel model = OnnxParser.ParseModel(ref r);
            GraphResolver.ResolveMain(model, ctx);

            var owned = new System.Collections.Generic.List<IDisposable> { source };
            if (resolver is not null)
            {
                owned.AddRange(resolver.Owned);
            }

            model.SetOwned(owned);
            return model;
        }
        catch
        {
            resolver?.Dispose();
            source.Dispose();
            throw;
        }
    }
}
