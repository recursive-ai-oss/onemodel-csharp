using System;
using System.Globalization;

namespace RecursiveAI.OneModel;

/// <summary>Which part of the library rejected something. See <see cref="OneModelException.Category"/>.</summary>
public enum OneModelErrorCategory
{
    /// <summary>An ONNX file or stream could not be imported (<c>OnnxImportException</c>).</summary>
    OnnxImport,

    /// <summary>A file in the OneModel format could not be read or written (<c>ModelFormatException</c>).</summary>
    ModelFormat,

    /// <summary>A session could not be created or run (<c>SessionException</c>).</summary>
    Session,

    /// <summary>A model could not be traced (<c>TraceException</c>).</summary>
    Tracing,

    /// <summary>A training checkpoint could not be read or applied (<c>CheckpointException</c>).</summary>
    Checkpoint,
}

/// <summary>
/// The base of every exception the library throws because of the data it was given (a model, a checkpoint, a request): catch this
/// to handle "the library rejected this" in one clause. Each derived type adds its own <c>Kind</c> enum with the specific cause.
/// </summary>
/// <remarks>
/// <para>What is not derived from it: argument mistakes are the BCL's <see cref="ArgumentException"/> family (a
/// <c>TensorShapeException</c> is one), use after dispose is an <see cref="ObjectDisposedException"/>, and file-system failures are
/// the BCL's own <see cref="System.IO.IOException"/> family (<see cref="System.IO.FileNotFoundException"/>,
/// <see cref="UnauthorizedAccessException"/>, ...), thrown unwrapped from every method that opens a file or reads a stream, so
/// existing I/O handling keeps working.</para>
/// <para>Where the fault is in a file, <see cref="InputName"/>, <see cref="Offset"/> and <see cref="FieldPath"/> say so; the message
/// contains them too, shortened when the field path is very long.</para>
/// </remarks>
public abstract class OneModelException : Exception
{
    private const int MaxPathInMessage = 160;

    /// <summary>Creates the exception.</summary>
    /// <param name="category">Which part of the library rejected the input.</param>
    /// <param name="message">What was expected and what was found.</param>
    /// <param name="inputName">The file path, or a description such as <c>&lt;stream&gt;</c>; null when not known.</param>
    /// <param name="offset">A byte offset in the input, or -1 when not known.</param>
    /// <param name="fieldPath">The field the problem is at, or null.</param>
    /// <param name="inner">The exception that caused this one, if any.</param>
    protected OneModelException(OneModelErrorCategory category, string message, string? inputName = null, long offset = -1,
        string? fieldPath = null, Exception? inner = null)
        : base(Compose(message, inputName, offset, fieldPath), inner)
    {
        Category = category;
        InputName = inputName;
        Offset = offset;
        FieldPath = fieldPath;
    }

    /// <summary>Which part of the library rejected the input.</summary>
    public OneModelErrorCategory Category { get; }

    /// <summary>The file path or stream description the fault is in, or null when not known.</summary>
    public string? InputName { get; }

    /// <summary>The byte offset into <see cref="InputName"/>, or -1 when not known.</summary>
    public long Offset { get; }

    /// <summary>The field path inside the input, or null when not known. The full path, even when the message shortens it.</summary>
    public string? FieldPath { get; }

    private static string Compose(string message, string? source, long offset, string? path)
    {
        string where = source ?? "";
        if (offset >= 0)
        {
            where += (where.Length > 0 ? " " : "") + "at byte " + offset.ToString(CultureInfo.InvariantCulture);
        }

        if (path is not null)
        {
            if (path.Length > MaxPathInMessage)
            {
                path = string.Concat(path.AsSpan(0, 60), " ... ", path.AsSpan(path.Length - 90));
            }

            where += (where.Length > 0 ? ", " : "") + path;
        }

        return where.Length > 0 ? message + " (" + where + ")" : message;
    }
}
