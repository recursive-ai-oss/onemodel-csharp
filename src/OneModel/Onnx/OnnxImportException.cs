using System;

namespace RecursiveAI.OneModel.Onnx;

/// <summary>What went wrong while importing an ONNX model.</summary>
public enum OnnxErrorKind
{
    /// <summary>The bytes are not valid protobuf: a truncated or overlong varint, a length past the end, a group, a bad tag, bad UTF-8, or nesting beyond the limit.</summary>
    Malformed,
    /// <summary>The bytes are valid protobuf but not a valid ONNX model: a missing required field, mismatched sizes, a dangling or duplicate name, a cycle.</summary>
    InvalidModel,
    /// <summary>The model uses something this library does not import: a tensor segment, an unstored element type for a tensor conversion, a size past a limit.</summary>
    Unsupported,
    /// <summary>An external-data reference is unsafe or cannot be satisfied: a path outside the model's directory, a missing file, a range outside the file.</summary>
    ExternalData,
}

/// <summary>
/// The one exception the ONNX importer throws for bad input. It names the input, the byte offset when one is known, and the field
/// path inside the model (for example <c>ModelProto.graph.node[3].attribute[1].t</c>), so a report can point at the fault.
/// Malformed input never surfaces as an <see cref="IndexOutOfRangeException"/>, <see cref="NullReferenceException"/> or
/// <see cref="OverflowException"/>.
/// </summary>
public sealed class OnnxImportException : OneModelException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="kind">What went wrong.</param>
    /// <param name="message">What was expected and what was found.</param>
    /// <param name="source">The file path or a description of the stream, or null.</param>
    /// <param name="offset">The byte offset in <paramref name="source"/>, or -1 when unknown.</param>
    /// <param name="fieldPath">The path of the message field being read, or null.</param>
    /// <param name="inner">The underlying exception, if any.</param>
    public OnnxImportException(OnnxErrorKind kind, string message, string? source = null, long offset = -1, string? fieldPath = null, Exception? inner = null)
        : base(OneModelErrorCategory.OnnxImport, message, source, offset, fieldPath, inner)
    {
        Kind = kind;
    }

    /// <summary>The category of the failure.</summary>
    public OnnxErrorKind Kind { get; }
}
