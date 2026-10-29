using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using RecursiveAI.OneModel.Graph;

namespace RecursiveAI.OneModel.Onnx;

/// <summary>
/// Turns the parsed form of a graph into the resolved <see cref="IrGraph"/>: defines every value, links node inputs and outputs to
/// values (looking outward through enclosing graphs for subgraphs), puts nodes in dependency order, and rejects dangling and
/// duplicate names and cycles. ONNX requires nodes to arrive already sorted (IR.md); a file that breaks that is reordered, and
/// <see cref="IrGraph.NodesWereReordered"/> says so.
/// </summary>
internal static class GraphResolver
{
    private sealed class Scope
    {
        public readonly Dictionary<string, IrValue> Map = new(StringComparer.Ordinal);
        public readonly Scope? Parent;

        public Scope(Scope? parent) => Parent = parent;

        public IrValue? Lookup(string name)
        {
            for (Scope? s = this; s is not null; s = s.Parent)
            {
                if (s.Map.TryGetValue(name, out IrValue? v))
                {
                    return v;
                }
            }

            return null;
        }
    }

    public static void ResolveMain(IrModel model, ParseContext ctx)
    {
        var opsets = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (OpsetImport o in model.OpsetImports)
        {
            opsets[Normalize(o.Domain)] = o.Version;
        }

        ctx.Push("graph");
        Resolve(model.Graph, null, null, opsets, ctx);
        ctx.Pop();
    }

    private static string Normalize(string domain) => domain == "ai.onnx" ? "" : domain;

    private static void Resolve(IrGraph g, Scope? parentScope, IrGraph? parent, Dictionary<string, long> opsets, ParseContext ctx)
    {
        ctx.Cancel.ThrowIfCancellationRequested();
        RawGraph raw = g.Raw ?? throw new InvalidOperationException("Graph resolved twice.");
        g.Raw = null;
        g.Parent = parent;
        var scope = new Scope(parentScope);
        var values = new List<IrValue>();

        IrValue Define(string name, string what)
        {
            if (name.Length == 0)
            {
                throw ctx.Fail(OnnxErrorKind.InvalidModel, $"A {what} has an empty name.");
            }

            if (scope.Map.ContainsKey(name))
            {
                throw ctx.Fail(OnnxErrorKind.InvalidModel, $"The name '{name}' is defined more than once in the graph ({what}).");
            }

            var v = new IrValue(name, g);
            scope.Map[name] = v;
            values.Add(v);
            return v;
        }

        var inputs = ImmutableArray.CreateBuilder<IrValue>(raw.Inputs.Count);
        foreach ((string name, IrType? type) in raw.Inputs)
        {
            IrValue v = Define(name, "graph input");
            v.IsGraphInput = true;
            v.Type = type;
            inputs.Add(v);
        }

        foreach (TensorData t in raw.Initializers)
        {
            if (scope.Map.TryGetValue(t.Name, out IrValue? existing) && existing.IsGraphInput && existing.Initializer is null)
            {
                existing.Initializer = t;
            }
            else
            {
                Define(t.Name, "initializer").Initializer = t;
            }
        }

        foreach (SparseTensorData sp in raw.SparseInitializers)
        {
            Define(sp.Values.Name, "sparse initializer").SparseInitializer = sp;
        }

        for (int i = 0; i < raw.Nodes.Count; i++)
        {
            IrNode node = raw.Nodes[i];
            node.Graph = g;
            ctx.Push("node", i);
            if (!opsets.TryGetValue(Normalize(node.Domain), out long version))
            {
                throw ctx.Fail(OnnxErrorKind.InvalidModel,
                    $"Node '{node.Name}' ({node.OpType}) uses the domain '{node.Domain}', which the model does not import.");
            }

            node.OpsetVersion = (int)Math.Clamp(version, 0, int.MaxValue);
            var outs = new IrValue?[node.OutputNames.Length];
            for (int k = 0; k < outs.Length; k++)
            {
                string on = node.OutputNames[k];
                if (on.Length == 0)
                {
                    continue;
                }

                IrValue v = Define(on, $"output {k} of node '{node.Name}' ({node.OpType})");
                v.Producer = node;
                v.ProducerIndex = k;
                outs[k] = v;
            }

            node.Outputs = [.. outs];
            ctx.Pop();
        }

        var outputs = ImmutableArray.CreateBuilder<IrValue>(raw.Outputs.Count);
        var captured = new List<IrValue>();
        var capturedSet = new HashSet<IrValue>();
        void Capture(IrValue v)
        {
            if (v.Graph != g && capturedSet.Add(v))
            {
                captured.Add(v);
            }
        }

        foreach ((string name, IrType? type) in raw.Outputs)
        {
            IrValue v = (name.Length > 0 ? scope.Lookup(name) : null)
                ?? throw ctx.Fail(OnnxErrorKind.InvalidModel, $"Graph output '{name}' is not defined by any input, initializer or node.");
            Capture(v);
            if (type is not null && v.Graph == g)
            {
                v.Type = type;
            }

            outputs.Add(v);
        }

        foreach ((string name, IrType? type) in raw.ValueInfos)
        {
            if (type is not null && scope.Map.TryGetValue(name, out IrValue? v) && v.Type is null)
            {
                v.Type = type;
            }
        }

        // Node inputs, and subgraphs (which can read everything this graph defines).
        var extra = new List<IrValue>?[raw.Nodes.Count];
        for (int i = 0; i < raw.Nodes.Count; i++)
        {
            IrNode node = raw.Nodes[i];
            ctx.Push("node", i);
            var ins = new IrValue?[node.InputNames.Length];
            for (int k = 0; k < ins.Length; k++)
            {
                string n = node.InputNames[k];
                if (n.Length == 0)
                {
                    continue;
                }

                IrValue v = scope.Lookup(n)
                    ?? throw ctx.Fail(OnnxErrorKind.InvalidModel, $"Input {k} of node '{node.Name}' ({node.OpType}) is '{n}', which nothing defines.");
                Capture(v);
                ins[k] = v;
            }

            node.Inputs = [.. ins];

            foreach (IrAttr a in node.Attributes)
            {
                if (a.RefAttrName.Length > 0)
                {
                    continue;
                }

                if (a.Type == IrAttributeType.Graph)
                {
                    ctx.Push("attribute '" + a.Name + "'");
                    ResolveChild(a.Graph, i);
                    ctx.Pop();
                }
                else if (a.Type == IrAttributeType.Graphs)
                {
                    ctx.Push("attribute '" + a.Name + "'");
                    foreach (IrGraph sub in a.Graphs)
                    {
                        ResolveChild(sub, i);
                    }

                    ctx.Pop();
                }
            }

            void ResolveChild(IrGraph child, int at)
            {
                Resolve(child, scope, g, opsets, ctx);
                foreach (IrValue c in child.Captures)
                {
                    Capture(c);
                    if (c.Graph == g)
                    {
                        (extra[at] ??= []).Add(c);
                    }
                }
            }

            ctx.Pop();
        }

        // Dependency order. Edges run from the producer of each value a node reads (directly, or through a subgraph) to the node.
        int count = raw.Nodes.Count;
        var index = new Dictionary<IrNode, int>(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < count; i++)
        {
            index[raw.Nodes[i]] = i;
        }

        var dependents = new List<int>?[count];
        var waiting = new int[count];
        var seen = new HashSet<int>();
        for (int i = 0; i < count; i++)
        {
            seen.Clear();
            foreach (IrValue? v in EffectiveInputs(raw.Nodes[i], extra[i]))
            {
                if (v is { Producer: { } p } && v.Graph == g && seen.Add(index[p]))
                {
                    (dependents[index[p]] ??= []).Add(i);
                    waiting[i]++;
                }
            }
        }

        var ready = new PriorityQueue<int, int>();
        for (int i = 0; i < count; i++)
        {
            if (waiting[i] == 0)
            {
                ready.Enqueue(i, i);
            }
        }

        var sorted = ImmutableArray.CreateBuilder<IrNode>(count);
        bool reordered = false;
        while (ready.TryDequeue(out int next, out _))
        {
            reordered |= next != sorted.Count;
            sorted.Add(raw.Nodes[next]);
            if (dependents[next] is { } list)
            {
                foreach (int d in list)
                {
                    if (--waiting[d] == 0)
                    {
                        ready.Enqueue(d, d);
                    }
                }
            }
        }

        if (sorted.Count != count)
        {
            for (int i = 0; i < count; i++)
            {
                if (waiting[i] > 0)
                {
                    throw ctx.Fail(OnnxErrorKind.InvalidModel, $"The graph has a cycle through node '{raw.Nodes[i].Name}' ({raw.Nodes[i].OpType}).");
                }
            }
        }

        // Consumers, in the sorted order.
        var consumers = new Dictionary<IrValue, List<IrNode>>();
        foreach (IrNode node in sorted)
        {
            var once = new HashSet<IrValue>();
            foreach (IrValue? v in EffectiveInputs(node, extra[index[node]]))
            {
                if (v is not null && v.Graph == g && once.Add(v))
                {
                    if (!consumers.TryGetValue(v, out List<IrNode>? l))
                    {
                        consumers[v] = l = [];
                    }

                    l.Add(node);
                }
            }
        }

        foreach (IrValue v in values)
        {
            if (consumers.TryGetValue(v, out List<IrNode>? l))
            {
                v.Consumers = [.. l];
            }
        }

        g.Inputs = inputs.ToImmutable();
        g.Outputs = outputs.ToImmutable();
        g.Initializers = [.. raw.Initializers];
        g.SparseInitializers = [.. raw.SparseInitializers];
        g.Nodes = sorted.ToImmutable();
        g.Values = [.. values];
        g.Captures = [.. captured];
        g.NodesWereReordered = reordered;
    }

    private static IEnumerable<IrValue?> EffectiveInputs(IrNode node, List<IrValue>? extra)
    {
        foreach (IrValue? v in node.Inputs)
        {
            yield return v;
        }

        if (extra is not null)
        {
            foreach (IrValue v in extra)
            {
                yield return v;
            }
        }
    }
}
