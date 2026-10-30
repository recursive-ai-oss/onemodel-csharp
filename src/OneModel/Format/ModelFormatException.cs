using System;
using System.Globalization;

namespace RecursiveAI.OneModel.Format;

/// <summary>What was wrong with a model file in the OneModel format.</summary>
public enum ModelFormatErrorKind
{
    /// <summary>The bytes are not a well-formed file: bad magic, truncated, inconsistent header, invalid JSON, a field of the wrong JSON type, or a range outside the file.</summary>
    Malformed,

    /// <summary>The file is well formed but describes an impossible model: duplicate or undefined names, a tensor whose byte length does not match its shape.</summary>
    InvalidModel,

    /// <summary>The file uses something this reader does not support: a newer format version, unknown flags, an unknown element type, a tensor over 2^31-1 bytes.</summary>
    Unsupported,

    /// <summary>The file could not be read or written.</summary>
    Io,
}

/// <summary>
/// The one exception the OneModel format reader and writer throw for bad input, bad files or unusable models.
/// The message names what was expected and where: <see cref="FieldPath"/> is a path into the manifest such as
/// <c>graph.nodes[3].attributes[0].value</c>, and <see cref="Offset"/> is a byte offset in the file when one applies.
/// </summary>
public sealed class ModelFormatException : Exception
{
    /// <summary>Creates the exception.</summary>
    /// <param name="kind">What went wrong.</param>
    /// <param name="message">What was expected and what was found.</param>
    /// <param name="source">The file path, or <c>&lt;stream&gt;</c> or <c>&lt;memory&gt;</c>.</param>
    /// <param name="offset">A byte offset in the file, or -1.</param>
    /// <param name="fieldPath">A manifest path or header field name, or null.</param>
    /// <param name="inner">The exception that caused this one.</param>
    public ModelFormatException(ModelFormatErrorKind kind, string message, string? source = null, long offset = -1, string? fieldPath = null, Exception? inner = null)
        : base(Compose(message, source, offset, fieldPath), inner)
    {
        Kind = kind;
        InputName = source;
        Offset = offset;
        FieldPath = fieldPath;
    }

    /// <summary>What went wrong.</summary>
    public ModelFormatErrorKind Kind { get; }

    /// <summary>The file path, or <c>&lt;stream&gt;</c> or <c>&lt;memory&gt;</c>; null when unknown.</summary>
    public string? InputName { get; }

    /// <summary>A byte offset in the file, or -1.</summary>
    public long Offset { get; }

    /// <summary>The header field or manifest path the problem is at, or null.</summary>
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
            where += (where.Length > 0 ? ", " : "") + path;
        }

        return where.Length > 0 ? message + " (" + where + ")" : message;
    }
}
