using System;

namespace RecursiveAI.OneModel.Onnx;

/// <summary>Options for <see cref="OnnxImporter"/>. The defaults suit trusted and untrusted files alike.</summary>
public sealed class OnnxLoadOptions
{
    /// <summary>
    /// Map files into memory instead of reading them. Tensor data then stays in the page cache and costs no managed memory until it
    /// is copied into a <see cref="Tensors.Tensor"/>. Default true. Where mapping is unavailable (browser WASM) or fails, files are read
    /// instead. A mapped file must not be truncated or rewritten while the model is alive.
    /// </summary>
    public bool MemoryMap { get; init; } = true;

    /// <summary>
    /// The directory external-data locations are relative to and confined to. For <c>Load(path)</c> it defaults to the model file's
    /// directory. For a stream or byte array it defaults to null, which makes any external-data reference an error.
    /// </summary>
    public string? ExternalDataDirectory { get; init; }

    /// <summary>The deepest nesting of protobuf messages accepted (a graph inside a node inside a graph counts three). Default 128; the minimum is 8.</summary>
    public int MaxNesting { get; init; } = 128;
}
