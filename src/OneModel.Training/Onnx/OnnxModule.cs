using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Threading;
using RecursiveAI.OneModel.Format;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Training;

/// <summary>Choices for loading a graph as a trainable module.</summary>
public sealed class OnnxModuleOptions
{
    /// <summary>
    /// Decides whether a float32 initializer (by its name in the file) is a trainable parameter. Called only for initializers the
    /// default rule would train; returning false makes it a frozen buffer. Default: null, every such initializer trains.
    /// </summary>
    public Func<string, bool>? IsTrainable { get; init; }

    /// <summary>Limits and external-data settings for reading the file. Default: the library defaults.</summary>
    public ModelLoadOptions? Load { get; init; }
}

/// <summary>
/// A model loaded from ONNX (or the OneModel format) as a trainable module: its float32 weights are parameters, and
/// <see cref="Forward(IReadOnlyDictionary{string, Tensor})"/> runs the graph's nodes on the eager tensor operators, so a backward
/// pass reaches every parameter. The trained weights can be written back (<see cref="Save(string, CancellationToken)"/>, <see cref="CreateSession"/>) so the
/// model is served by the ordinary graph <see cref="Session"/>.
/// </summary>
/// <remarks>
/// <para><b>Parameters.</b> A float32 initializer with more than one element is a parameter, named exactly as in the file (names may
/// contain dots), in the order the file lists the initializers. Other float32 initializers (scalars, and every initializer that feeds
/// a <c>Sin</c> or <c>Cos</c>, such as rotary-embedding frequency tables) are frozen <em>buffers</em> in the same order; initializers
/// of other element types are graph constants and are not part of the module's state. <see cref="OnnxModuleOptions.IsTrainable"/>
/// freezes more.</para>
/// <para><b>What runs with gradients.</b> Operators in <c>docs/training.md</c> section 11. Any operator whose inputs do not depend on a
/// parameter (shape arithmetic, masks, position tables, integer work) runs on the graph runtime's own kernel and carries no gradient.
/// A graph that needs an operator outside both sets fails in <see cref="Load(string, OnnxModuleOptions?, CancellationToken)"/> with a
/// <see cref="SessionException"/> naming every such operator.</para>
/// <para><b>Dropout</b> nodes run as the identity in both modes. Intermediate tensors belong to the open <see cref="ActivationScope"/>;
/// run the forward and backward inside one, as for any other module. Not thread-safe.</para>
/// </remarks>
public sealed class OnnxModule : NnModule
{
    private sealed class Step
    {
        public IrNode Node = null!;
        public int[] In = [];
        public int[] Out = [];
        public DiffHandler? Handler;
        public EagerNode? Eager;
    }

    private readonly IrModel _model;
    private readonly Step[] _steps;
    private readonly Tensor[] _base;
    private readonly int[] _inputIds;
    private readonly string[] _inputNames;
    private readonly int[] _outputIds;
    private readonly string[] _outputNames;
    private readonly string[] _valueNames;
    private readonly List<(string Name, Tensor Parameter)> _written = [];
    private readonly Arena _scratch = new();
    private readonly List<EagerNode> _folded = [];
    private readonly bool _anyNotExact;

    private OnnxModule(IrModel model, OnnxModuleOptions options)
    {
        _model = model;
        IrGraph graph = model.Graph;
        if (!graph.SparseInitializers.IsEmpty)
        {
            throw new SessionException(SessionErrorKind.Unsupported, "Sparse initializers are not supported in a trainable graph.");
        }

        var ids = new Dictionary<IrValue, int>(ReferenceEqualityComparer.Instance);
        var byName = new Dictionary<string, IrValue>(StringComparer.Ordinal);
        foreach (IrValue v in graph.Values)
        {
            ids[v] = ids.Count;
            byName[v.Name] = v;
        }

        _valueNames = new string[ids.Count];
        foreach ((IrValue v, int id) in ids)
        {
            _valueNames[id] = v.Name;
        }

        _base = new Tensor[ids.Count];
        var diff = new bool[ids.Count];
        var isConst = new bool[ids.Count];

        // Initializers: parameters, frozen buffers, constants. Suspend any open activation scope so they outlive it.
        HashSet<string> tables = PositionTableInitializers(graph);
        ActivationScope? suspended = ActivationScope.Suspend();
        try
        {
            foreach (TensorData t in graph.Initializers)
            {
                if (!byName.TryGetValue(t.Name, out IrValue? value))
                {
                    continue;
                }

                int id = ids[value];
                Tensor tensor = t.ToTensor();
                isConst[id] = true;
                if (t.TryGetDType(out DType dt) && dt == DType.F32)
                {
                    bool train = t.ElementCount > 1 && !tables.Contains(t.Name) && (options.IsTrainable?.Invoke(t.Name) ?? true);
                    if (train)
                    {
                        _base[id] = RegisterVerbatim(t.Name, tensor, parameter: true);
                        diff[id] = true;
                        isConst[id] = false;
                        _written.Add((t.Name, _base[id]));
                    }
                    else
                    {
                        _base[id] = RegisterVerbatim(t.Name, tensor, parameter: false);
                    }
                }
                else
                {
                    _base[id] = tensor;
                }
            }

            // Nodes in order: fold what depends on constants only, route the rest.
            var steps = new List<Step>();
            var unsupported = new SortedSet<string>(StringComparer.Ordinal);
            foreach (IrNode node in graph.Nodes)
            {
                var step = new Step { Node = node, In = IdsOf(node.Inputs, ids), Out = IdsOf(node.Outputs, ids) };
                bool anyDiff = false, allConst = true;
                foreach (int i in step.In)
                {
                    if (i >= 0)
                    {
                        anyDiff |= diff[i];
                        allConst &= isConst[i];
                    }
                }

                string op = node.OpType;
                if (!node.IsDefaultDomain || node.Attributes.AsSpan().ContainsGraph())
                {
                    unsupported.Add(Describe(node) + (node.IsDefaultDomain ? " (subgraph)" : " (custom domain)"));
                    continue;
                }

                bool differentiable = anyDiff && !OnnxDiffOps.NoGradient.Contains(op) && !(op == "Cast" && !CastsToFloat32(node));
                if (differentiable)
                {
                    // Mark the outputs first: a missing operator must not hide the ones after it from the check.
                    foreach (int o in step.Out)
                    {
                        if (o >= 0)
                        {
                            diff[o] = true;
                        }
                    }

                    if (!OnnxDiffOps.Table.TryGetValue(op, out step.Handler))
                    {
                        unsupported.Add(op);
                        continue;
                    }

                    if (op is "LayerNormalization" && HasExtraOutputs(node, graph))
                    {
                        unsupported.Add("LayerNormalization (Mean/InvStdDev outputs)");
                        continue;
                    }

                    steps.Add(step);
                    continue;
                }

                step.Eager = EagerNode.TryCreate(node);
                if (step.Eager is null)
                {
                    unsupported.Add(anyDiff ? op : op + " (no kernel)");
                    continue;
                }

                if (allConst && TryFold(step, _base, isConst))
                {
                    // A constant the kernel produced stays owned by the kernel, so the kernel lives as long as the module.
                    _folded.Add(step.Eager);
                    continue;
                }

                steps.Add(step);
                _anyNotExact |= !step.Eager.SupportsExact;
            }

            if (unsupported.Count > 0)
            {
                throw new SessionException(SessionErrorKind.UnsupportedOperator,
                    $"The graph cannot be trained: it uses operators with no gradient support on a path that reaches a parameter, or no kernel: {string.Join(", ", unsupported)}.",
                    null, [.. unsupported]);
            }

            _steps = [.. steps];
        }
        finally
        {
            ActivationScope.Resume(suspended);
        }

        var inNames = new List<string>();
        var inIds = new List<int>();
        foreach (IrValue v in graph.Inputs)
        {
            if (v.Initializer is null)
            {
                inNames.Add(v.Name);
                inIds.Add(ids[v]);
            }
        }

        _inputNames = [.. inNames];
        _inputIds = [.. inIds];
        _outputNames = [.. Names(graph.Outputs)];
        var outIds = new int[graph.Outputs.Length];
        for (int i = 0; i < outIds.Length; i++)
        {
            outIds[i] = ids[graph.Outputs[i]];
        }

        _outputIds = outIds;
    }

    private protected override void OnDispose()
    {
        foreach (Step s in _steps)
        {
            s.Eager?.Dispose();
        }

        foreach (EagerNode k in _folded)
        {
            k.Dispose();
        }

        _scratch.Dispose();
        _model.Dispose();
    }

    /// <summary>Names of the graph's inputs that the caller supplies (not initializers), in graph order.</summary>
    public IReadOnlyList<string> InputNames => _inputNames;

    /// <summary>Names of the graph's outputs, in graph order.</summary>
    public IReadOnlyList<string> OutputNames => _outputNames;

    /// <summary>
    /// Loads a model file, ONNX or OneModel format (told apart by the first bytes), as a trainable module.
    /// </summary>
    /// <param name="path">The model file; external data is read from its directory.</param>
    /// <param name="options">Loading choices; default null (all trainable float32 weights train).</param>
    /// <param name="cancellationToken">Cancels reading the file.</param>
    /// <returns>A module in training mode; dispose it to release the weights.</returns>
    /// <exception cref="OnnxImportException">The file is not a valid ONNX model.</exception>
    /// <exception cref="ModelFormatException">The file is not a valid OneModel file.</exception>
    /// <exception cref="SessionException">The graph uses operators this path cannot run with gradients (kind
    /// <see cref="SessionErrorKind.UnsupportedOperator"/>, listing all of them), or sparse initializers.</exception>
    public static OnnxModule Load(string path, OnnxModuleOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(path);
        options ??= new OnnxModuleOptions();
        IrModel model = ModelFormat.IsModelFormatFile(path)
            ? ModelFormat.Load(path, options.Load, cancellationToken)
            : OnnxImporter.Load(path, options.Load, cancellationToken);
        return FromModel(model, options);
    }

    /// <summary>Makes a trainable module from an imported model. The module takes ownership of <paramref name="model"/> and disposes it.</summary>
    /// <param name="model">A model from <see cref="OnnxImporter"/> or <see cref="ModelFormat"/>.</param>
    /// <param name="options">Loading choices; default null.</param>
    /// <returns>A module in training mode.</returns>
    /// <exception cref="SessionException">As for <see cref="Load(string, OnnxModuleOptions?, CancellationToken)"/>; the model is disposed.</exception>
    public static OnnxModule FromModel(IrModel model, OnnxModuleOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        try
        {
            return new OnnxModule(model, options ?? new OnnxModuleOptions());
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    /// <summary>Runs a graph with exactly one caller-supplied input and returns its first output.</summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The first output, carrying a backward record when it depends on a parameter and recording is on.</returns>
    /// <exception cref="InvalidOperationException">The graph has other than one input.</exception>
    public override Tensor Forward(Tensor input)
    {
        if (_inputNames.Length != 1)
        {
            throw new InvalidOperationException($"The graph has {_inputNames.Length} inputs ({string.Join(", ", _inputNames)}); use Forward with a dictionary.");
        }

        return Forward(new Dictionary<string, Tensor> { [_inputNames[0]] = input })[0];
    }

    /// <summary>Runs the graph and returns all outputs.</summary>
    /// <param name="inputs">One tensor per name in <see cref="InputNames"/>, of the element types the graph declares (token ids are Int64).</param>
    /// <returns>The outputs in <see cref="OutputNames"/> order. Float outputs that depend on a parameter record a backward pass.</returns>
    /// <exception cref="ArgumentException">An input is missing or empty.</exception>
    /// <exception cref="SessionException">A node fails at run time, or exact mode is selected and a node outside the gradient path has no exact kernel.</exception>
    public Tensor[] Forward(IReadOnlyDictionary<string, Tensor> inputs)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        ArithmeticMode mode = TensorCompute.Mode;
        if (mode == ArithmeticMode.Exact && _anyNotExact)
        {
            var bad = new SortedSet<string>(StringComparer.Ordinal);
            foreach (Step s in _steps)
            {
                if (s.Eager is { SupportsExact: false })
                {
                    bad.Add(s.Node.OpType);
                }
            }

            throw new SessionException(SessionErrorKind.ExactModeUnavailable,
                $"Exact mode is not available: no exact kernel exists for {string.Join(", ", bad)}.", null, [.. bad]);
        }

        var v = new Tensor[_base.Length];
        Array.Copy(_base, v, v.Length);
        for (int i = 0; i < _inputIds.Length; i++)
        {
            if (!inputs.TryGetValue(_inputNames[i], out Tensor t) || t.IsEmpty)
            {
                throw new ArgumentException($"Input '{_inputNames[i]}' is missing.", nameof(inputs));
            }

            v[_inputIds[i]] = t;
        }

        KernelPath path = TensorCompute.Path;
        int threads = TensorCompute.Threads;
        var ins = new Tensor[8];
        foreach (Step s in _steps)
        {
            if (ins.Length < s.In.Length)
            {
                ins = new Tensor[s.In.Length];
            }

            Span<Tensor> x = ins.AsSpan(0, s.In.Length);
            for (int i = 0; i < x.Length; i++)
            {
                int id = s.In[i];
                if (id < 0)
                {
                    x[i] = default;
                    continue;
                }

                x[i] = v[id].IsEmpty ? throw new SessionException(SessionErrorKind.InvalidModel,
                    $"{s.Node.OpType} '{s.Node.Name}': input '{_valueNames[id]}' was never produced.", s.Node.Name) : v[id];
            }

            Tensor[] outs = s.Handler is not null ? s.Handler(s.Node, x) : s.Eager!.Run(x, mode, path, threads, _scratch, null);
            for (int o = 0; o < s.Out.Length && o < outs.Length; o++)
            {
                if (s.Out[o] >= 0)
                {
                    v[s.Out[o]] = outs[o];
                }
            }
        }

        var result = new Tensor[_outputIds.Length];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = v[_outputIds[i]].IsEmpty
                ? throw new SessionException(SessionErrorKind.InvalidModel, $"Output '{_outputNames[i]}' was never produced.")
                : v[_outputIds[i]];
        }

        return result;
    }

    /// <summary>
    /// Writes the current parameter values into the module's copy of the graph and returns it. The model stays owned by this module and
    /// is disposed with it; to serve the trained model use <see cref="CreateSession"/> or <see cref="Save(string, CancellationToken)"/>.
    /// </summary>
    /// <returns>The graph with the trained weights.</returns>
    public IrModel ToModel()
    {
        foreach ((string name, Tensor p) in _written)
        {
            float[] values = p.Detach().ToArray<float>();
            var raw = new byte[checked(values.Length * sizeof(float))];
            Buffer.BlockCopy(values, 0, raw, 0, raw.Length);
            IrEdit.ReplaceInitializer(_model, name, raw);
        }

        return _model;
    }

    /// <summary>Creates a graph session that runs the model with the current (trained) weights.</summary>
    /// <param name="options">Session options; default null (fast mode, all optimizations).</param>
    /// <param name="cancellationToken">Cancels session creation.</param>
    /// <returns>A session independent of later training steps (it holds its own copy of the weights).</returns>
    public Session CreateSession(ModelOptions? options = null, CancellationToken cancellationToken = default) =>
        Session.Create(ToModel(), options, cancellationToken);

    /// <summary>Saves the model with the current (trained) weights in the OneModel format; <c>Model.Load</c> and <c>Session.Load</c> read it.</summary>
    /// <param name="path">The file to write; replaced if it exists.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public void Save(string path, CancellationToken cancellationToken = default) =>
        ModelFormat.Save(ToModel(), path, cancellationToken);

    /// <summary>Saves the model with the current (trained) weights in the OneModel format to <paramref name="output"/>.</summary>
    /// <param name="output">A writable stream; left open.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public void Save(Stream output, CancellationToken cancellationToken = default) =>
        ModelFormat.Save(ToModel(), output, cancellationToken);

    // ------------------------------------------------------------------------------------------------------------ loading helpers

    private static string[] Names(ImmutableArray<IrValue> values)
    {
        var r = new string[values.Length];
        for (int i = 0; i < r.Length; i++)
        {
            r[i] = values[i].Name;
        }

        return r;
    }

    private static string Describe(IrNode n) => n.IsDefaultDomain ? n.OpType : n.Domain + "." + n.OpType;

    private static int[] IdsOf(ImmutableArray<IrValue?> values, Dictionary<IrValue, int> ids)
    {
        var r = new int[values.Length];
        for (int i = 0; i < r.Length; i++)
        {
            r[i] = values[i] is { } v ? ids[v] : -1;
        }

        return r;
    }

    private static bool CastsToFloat32(IrNode n) => n.FindAttribute("to") is { } a && a.Whole == 1;

    private static bool HasExtraOutputs(IrNode n, IrGraph g)
    {
        for (int i = 1; i < n.Outputs.Length; i++)
        {
            if (n.Outputs[i] is { } v && (!v.Consumers.IsEmpty || g.Outputs.Contains(v)))
            {
                return true;
            }
        }

        return false;
    }

    // Float initializers that feed a Sin or Cos, directly or through other nodes: position-encoding tables, which stay fixed.
    private static HashSet<string> PositionTableInitializers(IrGraph graph)
    {
        var result = new HashSet<string>(StringComparer.Ordinal);
        var seen = new HashSet<IrValue>(ReferenceEqualityComparer.Instance);
        var stack = new Stack<IrValue>();
        foreach (IrNode n in graph.Nodes)
        {
            if (n.OpType is "Sin" or "Cos")
            {
                foreach (IrValue? v in n.Inputs)
                {
                    if (v is not null)
                    {
                        stack.Push(v);
                    }
                }
            }
        }

        while (stack.Count > 0)
        {
            IrValue v = stack.Pop();
            if (!seen.Add(v))
            {
                continue;
            }

            if (v.Initializer is not null)
            {
                result.Add(v.Name);
            }
            else if (v.Producer is { } p && p.OpType is not ("Shape" or "Size"))
            {
                // Shape and Size pass sizes, not values, so the weights behind a shape never count as feeding the table.
                foreach (IrValue? i in p.Inputs)
                {
                    if (i is not null)
                    {
                        stack.Push(i);
                    }
                }
            }
        }

        return result;
    }

    private const int FoldLimitElements = 1 << 20;

    private static readonly HashSet<string> DataMovement = new(StringComparer.Ordinal)
    {
        "Constant", "Identity", "Reshape", "Squeeze", "Unsqueeze", "Flatten", "Transpose", "Slice", "Concat", "Gather", "Expand", "Cast",
    };

    // Evaluates a node whose inputs are all constants once, at load, when the result is small. Returns false to leave it for run time.
    private static bool TryFold(Step step, Tensor[] values, bool[] isConst)
    {
        if (step.Node.OpType is "RandomNormal" or "RandomUniform" or "Multinomial" or "Bernoulli")
        {
            return false;
        }

        var ins = new Tensor[step.In.Length];
        for (int i = 0; i < ins.Length; i++)
        {
            ins[i] = step.In[i] >= 0 ? values[step.In[i]] : default;
        }

        Tensor[] outs;
        try
        {
            using var scratch = new Arena();
            outs = step.Eager!.Run(ins, ArithmeticMode.Fast, TensorCompute.Path, 1, scratch, null);
        }
        catch (Exception e) when (e is SessionException or NotSupportedException or ArgumentException or InvalidOperationException)
        {
            return false;
        }

        // Only results that do not depend on the arithmetic mode are kept: integers, booleans, and float data movement. Float
        // arithmetic on constants (rotary tables) is redone per call so exact mode gets its own bits.
        long total = 0;
        bool keep = true;
        foreach (Tensor o in outs)
        {
            total += o.IsEmpty ? 0 : o.Length;
            keep &= o.IsEmpty || o.DType is not (DType.F32 or DType.F16 or DType.BF16 or DType.F64) || DataMovement.Contains(step.Node.OpType);
        }

        if (total > FoldLimitElements || !keep)
        {
            return false;
        }

        for (int o = 0; o < step.Out.Length && o < outs.Length; o++)
        {
            if (step.Out[o] >= 0)
            {
                values[step.Out[o]] = outs[o];
                isConst[step.Out[o]] = true;
            }
        }

        return true;
    }
}

internal static class AttributeSpanExtensions
{
    public static bool ContainsGraph(this ReadOnlySpan<IrAttr> attributes)
    {
        foreach (IrAttr a in attributes)
        {
            if (a.Type is IrAttributeType.Graph or IrAttributeType.Graphs)
            {
                return true;
            }
        }

        return false;
    }
}
