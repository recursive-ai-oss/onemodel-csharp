using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using RecursiveAI.OneModel.Format;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;

namespace RecursiveAI.OneModel.Tracing;

/// <summary>
/// A model recorded by <see cref="Tracer"/>: an ordinary graph. Run it with <see cref="CreateSession"/>, save it in the library's format
/// with <see cref="Save(string, CancellationToken)"/> (load it back with <c>Model.Load</c> or <c>Session.Load</c>), or write it as ONNX
/// with <see cref="SaveOnnx(string)"/>. Dispose it when done; sessions made from it stay valid.
/// </summary>
public sealed class TracedModel : IDisposable
{
    private readonly byte[] _onnx;
    private IrModel? _ir;

    internal TracedModel(byte[] onnx, string[] inputNames, string[] outputNames, int nodeCount)
    {
        _onnx = onnx;
        InputNames = inputNames;
        OutputNames = outputNames;
        NodeCount = nodeCount;
    }

    /// <summary>The input names in declaration order.</summary>
    public IReadOnlyList<string> InputNames { get; }

    /// <summary>The output names in the order given to <c>Finish</c>.</summary>
    public IReadOnlyList<string> OutputNames { get; }

    /// <summary>Number of operators in the recorded graph (before the session's optimizations).</summary>
    public int NodeCount { get; }

    /// <summary>The recorded graph as an ONNX model (opset 20), as a new array.</summary>
    public byte[] ToOnnx() => (byte[])_onnx.Clone();

    /// <summary>Writes the recorded graph as an ONNX file, constants embedded.</summary>
    /// <param name="path">The file to create or overwrite.</param>
    public void SaveOnnx(string path) => File.WriteAllBytes(path, _onnx);

    /// <summary>The recorded graph as the library's IR; owned by this object.</summary>
    public IrModel Ir => _ir ??= OnnxImporter.Load(_onnx);

    /// <summary>Saves the model in the library's format (<see cref="ModelFormat"/>).</summary>
    /// <param name="path">The file to create or overwrite.</param>
    /// <param name="cancellationToken">Checked while writing.</param>
    public void Save(string path, CancellationToken cancellationToken = default) => ModelFormat.Save(Ir, path, cancellationToken);

    /// <summary>Saves the model in the library's format to a stream.</summary>
    /// <param name="output">A writable stream; left open.</param>
    /// <param name="cancellationToken">Checked while writing.</param>
    public void Save(Stream output, CancellationToken cancellationToken = default) => ModelFormat.Save(Ir, output, cancellationToken);

    /// <summary>Creates a session over the recorded graph, with the optimizations the options ask for.</summary>
    /// <param name="options">Defaults to fast mode.</param>
    /// <param name="cancellationToken">Checked between operators.</param>
    /// <exception cref="SessionException">An operator is unsupported in the requested mode.</exception>
    public Session CreateSession(SessionOptions? options = null, CancellationToken cancellationToken = default) =>
        Session.Create(Ir, options, cancellationToken);

    /// <summary>Releases the imported graph. Sessions created earlier keep working.</summary>
    public void Dispose() => _ir?.Dispose();
}
