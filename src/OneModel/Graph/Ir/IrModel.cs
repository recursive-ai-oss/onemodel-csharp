using System;
using System.Collections.Generic;
using System.Collections.Immutable;

namespace RecursiveAI.OneModel.Graph;

/// <summary>An operator set the model imports: a domain and the version of it.</summary>
/// <param name="Domain">The domain; the empty string (or <c>ai.onnx</c>) is the default domain.</param>
/// <param name="Version">The opset version.</param>
public readonly record struct OpsetImport(string Domain, long Version);

/// <summary>A function defined in a model: ONNX's <c>FunctionProto</c>. It is imported as written; calls are not expanded.</summary>
public sealed class IrFunction
{
    internal IrFunction()
    {
    }

    /// <summary>The function's name.</summary>
    public string Name { get; internal set; } = "";

    /// <summary>The function's domain.</summary>
    public string Domain { get; internal set; } = "";

    /// <summary>The overload name, or empty.</summary>
    public string Overload { get; internal set; } = "";

    /// <summary>The names of the formal inputs.</summary>
    public ImmutableArray<string> Inputs { get; internal set; } = [];

    /// <summary>The names of the formal outputs.</summary>
    public ImmutableArray<string> Outputs { get; internal set; } = [];

    /// <summary>The names of attributes with no default, which a caller must supply.</summary>
    public ImmutableArray<string> AttributeNames { get; internal set; } = [];

    /// <summary>Attributes with a default value.</summary>
    public ImmutableArray<IrAttr> AttributeDefaults { get; internal set; } = [];

    /// <summary>The body. Nodes here refer to values by name only: their <see cref="IrNode.Inputs"/> are empty.</summary>
    public ImmutableArray<IrNode> Nodes { get; internal set; } = [];

    /// <summary>The operator sets the body uses.</summary>
    public ImmutableArray<OpsetImport> OpsetImports { get; internal set; } = [];

    /// <summary>Documentation text, or empty.</summary>
    public string DocString { get; internal set; } = "";
}

/// <summary>
/// An imported ONNX model: ONNX's <c>ModelProto</c> resolved into an immutable <see cref="IrGraph"/>. Initializers are views into the
/// model file (memory-mapped when possible), so the model owns the file's memory: dispose it when no tensor data is needed any more.
/// Reading from an immutable model is safe from many threads.
/// </summary>
public sealed class IrModel : IDisposable
{
    private ImmutableArray<IDisposable> _owned = [];

    internal IrModel()
    {
    }

    internal void SetOwned(IEnumerable<IDisposable> owned) => _owned = [.. owned];

    /// <summary>The ONNX IR version (<c>ir_version</c>), at least 3.</summary>
    public long IrVersion { get; internal set; }

    /// <summary>The tool that wrote the model, or empty.</summary>
    public string ProducerName { get; internal set; } = "";

    /// <summary>The producer's version, or empty.</summary>
    public string ProducerVersion { get; internal set; } = "";

    /// <summary>The model's domain, or empty.</summary>
    public string Domain { get; internal set; } = "";

    /// <summary>The model's own version number.</summary>
    public long ModelVersion { get; internal set; }

    /// <summary>Documentation text, or empty.</summary>
    public string DocString { get; internal set; } = "";

    /// <summary>The imported operator sets in file order.</summary>
    public ImmutableArray<OpsetImport> OpsetImports { get; internal set; } = [];

    /// <summary>The main graph.</summary>
    public IrGraph Graph { get; internal set; } = null!;

    /// <summary>The functions the model defines.</summary>
    public ImmutableArray<IrFunction> Functions { get; internal set; } = [];

    /// <summary>The model's metadata key/value pairs in file order.</summary>
    public ImmutableArray<KeyValuePair<string, string>> Metadata { get; internal set; } = [];

    /// <summary>The version of the default domain that the model imports, or 0 if it imports none.</summary>
    public long DefaultOpset
    {
        get
        {
            foreach (OpsetImport o in OpsetImports)
            {
                if (o.Domain.Length == 0 || o.Domain == "ai.onnx")
                {
                    return o.Version;
                }
            }

            return 0;
        }
    }

    /// <summary>Releases the memory-mapped files behind the initializers. <see cref="TensorData"/> from this model must not be read afterwards.</summary>
    public void Dispose()
    {
        foreach (IDisposable d in _owned)
        {
            d.Dispose();
        }
    }
}
