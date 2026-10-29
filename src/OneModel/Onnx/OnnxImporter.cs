using System;
using System.IO;
using System.Threading;
using RecursiveAI.OneModel.Graph;

namespace RecursiveAI.OneModel.Onnx;

/// <summary>
/// Reads ONNX models (IR version 3 and later, any opset) into the <see cref="IrModel"/> graph representation. The protobuf reader,
/// the ONNX message reader and the external-data resolver are this library's own; nothing is generated and nothing is loaded
/// that was not checked against the bytes that remain. Bad input produces <see cref="OnnxImportException"/> and nothing else.
/// </summary>
public static class OnnxImporter
{
    /// <summary>The files at least this large are memory-mapped when <see cref="OnnxLoadOptions.MemoryMap"/> is set; smaller ones are read.</summary>
    private const long MapThreshold = 1 << 20;

    /// <summary>Imports a model from a file. External data is looked up in the file's directory unless the options say otherwise.</summary>
    /// <param name="path">The <c>.onnx</c> file. It may be at most 2^31-1 bytes (protobuf's limit); larger models keep their weights in external files.</param>
    /// <param name="options">Import options; defaults when null.</param>
    /// <param name="cancellationToken">Checked between nodes and tensors.</param>
    /// <returns>The model. The caller disposes it, which unmaps the file.</returns>
    /// <exception cref="OnnxImportException">The file cannot be read or is not a valid, supported ONNX model.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public static IrModel Load(string path, OnnxLoadOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(path);
        options ??= new OnnxLoadOptions();
        ByteSource source;
        try
        {
            source = ByteSource.OpenFile(path, options.MemoryMap, MapThreshold, readWholeIfUnmapped: true);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or NotSupportedException or ArgumentException)
        {
            throw new OnnxImportException(OnnxErrorKind.Io, "The model file cannot be read: " + e.Message, path, -1, null, e);
        }

        string? directory = options.ExternalDataDirectory ?? Path.GetDirectoryName(Path.GetFullPath(path));
        return LoadCore(source, path, directory, options, cancellationToken);
    }

    /// <summary>Imports a model from a stream, which is read to its end. External data is an error unless <see cref="OnnxLoadOptions.ExternalDataDirectory"/> is set.</summary>
    /// <param name="stream">A readable stream holding at most 2^31-1 bytes from its current position.</param>
    /// <param name="options">Import options; defaults when null.</param>
    /// <param name="cancellationToken">Checked between nodes and tensors, and while reading.</param>
    /// <returns>The model.</returns>
    /// <exception cref="OnnxImportException">The stream cannot be read or does not hold a valid, supported model.</exception>
    public static IrModel Load(Stream stream, OnnxLoadOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(stream);
        options ??= new OnnxLoadOptions();
        byte[] bytes;
        try
        {
            using var ms = new MemoryStream();
            var buffer = new byte[81920];
            int n;
            while ((n = stream.Read(buffer, 0, buffer.Length)) > 0)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (ms.Length + n > Array.MaxLength)
                {
                    throw new OnnxImportException(OnnxErrorKind.Unsupported, "The stream is larger than 2 GB, the most a protobuf message can hold.", "<stream>");
                }

                ms.Write(buffer, 0, n);
            }

            bytes = ms.ToArray();
        }
        catch (Exception e) when (e is IOException or NotSupportedException or ObjectDisposedException)
        {
            throw new OnnxImportException(OnnxErrorKind.Io, "The stream cannot be read: " + e.Message, "<stream>", -1, null, e);
        }

        return LoadCore(ByteSource.FromArray(bytes), "<stream>", options.ExternalDataDirectory, options, cancellationToken);
    }

    /// <summary>Imports a model from memory. The array is used in place, not copied: do not change it while the model is alive.</summary>
    /// <param name="data">The bytes of an <c>.onnx</c> file.</param>
    /// <param name="options">Import options; defaults when null.</param>
    /// <param name="cancellationToken">Checked between nodes and tensors.</param>
    /// <returns>The model.</returns>
    /// <exception cref="OnnxImportException">The bytes are not a valid, supported model.</exception>
    public static IrModel Load(byte[] data, OnnxLoadOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(data);
        options ??= new OnnxLoadOptions();
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

    private static IrModel LoadCore(ByteSource source, string inputName, string? externalDirectory, OnnxLoadOptions options, CancellationToken ct)
    {
        ExternalDataResolver? resolver = null;
        try
        {
            if (source.Length > Array.MaxLength)
            {
                throw new OnnxImportException(OnnxErrorKind.Unsupported,
                    $"The model file is {source.Length} bytes; a protobuf message can hold at most 2^31-1. Keep large weights in external data.", inputName);
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
