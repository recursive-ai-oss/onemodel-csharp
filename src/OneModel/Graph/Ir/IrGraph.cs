using System;
using System.Collections.Immutable;

namespace RecursiveAI.OneModel.Graph;

/// <summary>
/// A named value in a graph: a graph input, an initializer, or a node output. Names are unique within a graph (SSA).
/// Immutable once the model is returned.
/// </summary>
public sealed class IrValue
{
    internal IrValue(string name, IrGraph graph)
    {
        Name = name;
        Graph = graph;
    }

    /// <summary>The value's name, never empty.</summary>
    public string Name { get; }

    /// <summary>The graph that defines the value.</summary>
    public IrGraph Graph { get; }

    /// <summary>The declared type from the graph's inputs, outputs or <c>value_info</c>, or null if none was declared.</summary>
    public IrType? Type { get; internal set; }

    /// <summary>The node that produces the value, or null for graph inputs and initializers.</summary>
    public IrNode? Producer { get; internal set; }

    /// <summary>The position of the value among <see cref="Producer"/>'s outputs.</summary>
    public int ProducerIndex { get; internal set; }

    /// <summary>True when the value is listed in the graph's inputs.</summary>
    public bool IsGraphInput { get; internal set; }

    /// <summary>The constant tensor that is the value (or, for an input, its default), or null.</summary>
    public TensorData? Initializer { get; internal set; }

    /// <summary>The sparse constant that is the value, or null.</summary>
    public SparseTensorData? SparseInitializer { get; internal set; }

    /// <summary>
    /// The nodes that read the value, in graph order. A node whose subgraph reads the value (from the enclosing scope) counts as a
    /// consumer; so does a node that lists it twice (once).
    /// </summary>
    public ImmutableArray<IrNode> Consumers { get; internal set; } = [];

    /// <summary>True when the value is read by no node and is not a graph output.</summary>
    public bool IsDead => Consumers.IsEmpty && !Graph.Outputs.Contains(this);

    /// <inheritdoc/>
    public override string ToString() => Name;
}

/// <summary>One operator application. Immutable once the model is returned.</summary>
public sealed class IrNode
{
    internal IrNode()
    {
    }

    /// <summary>The node's name (optional in ONNX; may be empty and need not be unique).</summary>
    public string Name { get; internal set; } = "";

    /// <summary>The operator, for example <c>Conv</c>.</summary>
    public string OpType { get; internal set; } = "";

    /// <summary>The operator domain as written in the file. The empty string and <c>ai.onnx</c> both name the default domain; see <see cref="IsDefaultDomain"/>.</summary>
    public string Domain { get; internal set; } = "";

    /// <summary>The overload name of a function call, or empty.</summary>
    public string Overload { get; internal set; } = "";

    /// <summary>Documentation text, or empty.</summary>
    public string DocString { get; internal set; } = "";

    /// <summary>The opset version the model imports for <see cref="Domain"/>; 0 for nodes inside a function body that are not resolved.</summary>
    public int OpsetVersion { get; internal set; }

    /// <summary>The attributes in file order.</summary>
    public ImmutableArray<IrAttr> Attributes { get; internal set; } = [];

    /// <summary>The input names as written; an empty string marks an omitted optional input.</summary>
    public ImmutableArray<string> InputNames { get; internal set; } = [];

    /// <summary>The output names as written; an empty string marks an omitted optional output.</summary>
    public ImmutableArray<string> OutputNames { get; internal set; } = [];

    /// <summary>The input values, with null for an omitted optional input. Empty for nodes in a function body.</summary>
    public ImmutableArray<IrValue?> Inputs { get; internal set; } = [];

    /// <summary>The output values, with null for an omitted optional output. Empty for nodes in a function body.</summary>
    public ImmutableArray<IrValue?> Outputs { get; internal set; } = [];

    /// <summary>The graph that holds the node; null inside a function body.</summary>
    public IrGraph? Graph { get; internal set; }

    /// <summary>True when the node belongs to the default operator domain.</summary>
    public bool IsDefaultDomain => Domain.Length == 0 || Domain == "ai.onnx";

    /// <summary>Finds an attribute by name, or null.</summary>
    public IrAttr? FindAttribute(string name)
    {
        foreach (IrAttr a in Attributes)
        {
            if (a.Name == name)
            {
                return a;
            }
        }

        return null;
    }

    /// <inheritdoc/>
    public override string ToString() => $"{(Domain.Length == 0 ? "" : Domain + ".")}{OpType} '{Name}'";
}

/// <summary>
/// A graph: ONNX's <c>GraphProto</c>, resolved so that every node input and output is an <see cref="IrValue"/>. The nodes are in
/// topological order. A subgraph (the body of <c>If</c>, <c>Loop</c> or <c>Scan</c>) is an <see cref="IrGraph"/> with a <see cref="Parent"/>
/// and may read values of the enclosing graphs; those are listed in <see cref="Captures"/>. Immutable once the model is returned.
/// </summary>
public sealed class IrGraph
{
    internal IrGraph()
    {
    }

    /// <summary>The parsed, not yet resolved form; dropped when the importer finishes the graph.</summary>
    internal RecursiveAI.OneModel.Onnx.RawGraph? Raw { get; set; }

    /// <summary>The graph's name, possibly empty.</summary>
    public string Name { get; internal set; } = "";

    /// <summary>Documentation text, or empty.</summary>
    public string DocString { get; internal set; } = "";

    /// <summary>The enclosing graph for a subgraph; null for the main graph.</summary>
    public IrGraph? Parent { get; internal set; }

    /// <summary>The declared inputs in file order. In models older than IR version 4 this includes the initializers.</summary>
    public ImmutableArray<IrValue> Inputs { get; internal set; } = [];

    /// <summary>The declared outputs in file order.</summary>
    public ImmutableArray<IrValue> Outputs { get; internal set; } = [];

    /// <summary>The dense initializers in file order.</summary>
    public ImmutableArray<TensorData> Initializers { get; internal set; } = [];

    /// <summary>The sparse initializers in file order.</summary>
    public ImmutableArray<SparseTensorData> SparseInitializers { get; internal set; } = [];

    /// <summary>The nodes in topological order. When the file was already ordered this is the file's order.</summary>
    public ImmutableArray<IrNode> Nodes { get; internal set; } = [];

    /// <summary>Every value the graph defines: inputs, initializers, then node outputs, in definition order.</summary>
    public ImmutableArray<IrValue> Values { get; internal set; } = [];

    /// <summary>For a subgraph, the values of enclosing graphs that it (or its own subgraphs) reads, each once, in first-use order.</summary>
    public ImmutableArray<IrValue> Captures { get; internal set; } = [];

    /// <summary>True when the file listed nodes out of dependency order and the importer reordered them.</summary>
    public bool NodesWereReordered { get; internal set; }

    /// <summary>Finds a value defined by this graph by name, or null (enclosing graphs are not searched).</summary>
    public IrValue? FindValue(string name)
    {
        foreach (IrValue v in Values)
        {
            if (v.Name == name)
            {
                return v;
            }
        }

        return null;
    }
}
