using System;

namespace RecursiveAI.OneModel;

/// <summary>
/// Limits and file handling for loading a model (ONNX or the OneModel format) from a path, a stream or memory. The defaults suit
/// trusted and untrusted files alike; tighten them for input you do not control.
/// </summary>
public sealed class ModelLoadOptions
{
    private readonly int _maxNesting = 128;
    private readonly long _maxModelBytes = int.MaxValue;

    /// <summary>
    /// Map files into memory instead of reading them. Tensor data then stays in the page cache and costs no managed memory until it
    /// is copied into a <see cref="Tensors.Tensor"/>. Default true. Where mapping is unavailable (browser WASM) or fails, files are read
    /// instead. A mapped file must not be truncated or rewritten while the model is alive.
    /// </summary>
    public bool MemoryMap { get; init; } = true;

    /// <summary>
    /// The directory ONNX external-data locations are relative to and confined to. For a path it defaults to the model file's
    /// directory. For a stream or byte array it defaults to null, which makes any external-data reference an error.
    /// </summary>
    public string? ExternalDataDirectory { get; init; }

    /// <summary>
    /// The deepest nesting of protobuf messages accepted when importing ONNX (a graph inside a node inside a graph counts three).
    /// Default 128; the range is 8 to 250.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The value is outside 8 to 250.</exception>
    public int MaxNesting
    {
        get => _maxNesting;
        init => _maxNesting = value is >= 8 and <= 250
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "MaxNesting must be between 8 and 250.");
    }

    /// <summary>
    /// The most bytes a model file, stream or array may hold. A longer input fails with a typed error (an <c>Unsupported</c> kind of
    /// <c>OnnxImportException</c> or <c>ModelFormatException</c>) as soon as the limit is passed, so an endless stream cannot exhaust
    /// memory. Default and maximum 2^31-1, the most one protobuf message can hold. Weights beyond that belong in external data files.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The value is below 1 or above 2^31-1.</exception>
    public long MaxModelBytes
    {
        get => _maxModelBytes;
        init => _maxModelBytes = value is >= 1 and <= int.MaxValue
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "MaxModelBytes must be between 1 and 2^31-1.");
    }
}
