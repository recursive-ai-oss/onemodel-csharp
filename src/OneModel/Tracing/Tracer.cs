using System;
using System.Collections.Generic;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tracing;

/// <summary>One symbolic extent: a size that varies between runs, named for a declared input axis or derived from a traced value's axis.</summary>
internal sealed class Atom
{
    public required int Id { get; init; }

    /// <summary>The user's name for a declared symbol; null for an extent derived by an operation (a convolution's output width, a concatenation).</summary>
    public string? Name { get; init; }

    /// <summary>The extent in the example run.</summary>
    public required int Size { get; init; }

    /// <summary>The value whose <see cref="Axis"/> holds this extent at run time (read with a Shape node).</summary>
    public required TraceValue Source { get; init; }

    public required int Axis { get; init; }
}

/// <summary>A symbolic extent: <c>Coef × Π Atoms</c>. Never constructed without atoms (a plain number is a static extent).</summary>
internal sealed class SymDim
{
    public SymDim(long coef, Atom[] atoms)
    {
        Coef = coef;
        Atoms = atoms;
    }

    public long Coef { get; }

    /// <summary>The atoms in increasing id, repeated for a power.</summary>
    public Atom[] Atoms { get; }

    public static SymDim Of(Atom a) => new(1, [a]);
}

/// <summary>A value of the graph being recorded: a graph input or the output of an emitted node.</summary>
internal sealed class TraceValue
{
    public TraceValue(Tracer owner, string name, DType dtype, int[] shape, SymDim?[] sym)
    {
        Owner = owner;
        Name = name;
        DType = dtype;
        Shape = shape;
        Sym = sym;
    }

    public Tracer Owner { get; }

    public string Name { get; set; }

    public DType DType { get; }

    public int[] Shape { get; }

    public SymDim?[] Sym { get; }

    public bool IsInput { get; set; }
}

/// <summary>
/// Records a model written on the tensor layer into the graph layer. Run the model once, eagerly, on example inputs while a tracer is
/// active; every operation that depends on an input is recorded as the ONNX operator(s) with the same semantics, and
/// <see cref="Finish(ReadOnlySpan{string}, ReadOnlySpan{Tensor})"/> returns a <see cref="TracedModel"/> that runs through
/// <see cref="Graph.Session"/> with the usual optimizations and saves to the library's format or to ONNX.
/// </summary>
/// <remarks>
/// <para>Only static graphs are recorded. Whatever the model computes from tensors that do not depend on an input (weights, constants)
/// is captured as constants; what depends on an input becomes graph nodes. The model's own C# control flow runs once, at trace time,
/// on values that do not come from traced tensors (sizes, flags, loop counts over layers).</para>
/// <para>Reading the data of a traced tensor (<c>ToArray</c>, <c>Get</c>, <c>Data</c>, <c>Storage</c>, copying out of it), changing one in
/// place, or entering <see cref="GradMode.NoGrad"/> throws <see cref="TraceException"/>, because the result of such a step would be
/// frozen into the graph. Operations without a graph equivalent also throw <see cref="TraceException"/> (<c>docs/tracing.md</c>).</para>
/// <para>A tracer is active on the thread that called <see cref="Begin"/>; dispose it (or call Finish) on that thread.
/// It is not thread-safe.</para>
/// </remarks>
public sealed partial class Tracer : IDisposable
{
    private const int OutputOpset = 20;

    private readonly Sink _sink;
    private readonly List<OnnxNodeSpec> _nodes = [];
    private readonly List<OnnxInitializerSpec> _initializers = [];
    private readonly List<TraceValue> _inputs = [];
    private readonly Dictionary<string, Atom> _declared = new(StringComparer.Ordinal);
    private readonly List<Atom> _atoms = [];
    private readonly HashSet<string> _usedNames = new(StringComparer.Ordinal);
    private readonly Dictionary<ConstKey, string> _constants = [];
    private readonly Dictionary<ConstKey, string> _parameterNames = [];
    private readonly Dictionary<(TensorBuffer? Buffer, int Generation, int Offset), string> _parameterStorage = [];
    private readonly List<(Tensor Tensor, long Version, string Name)> _captured = [];
    private readonly Dictionary<string, TraceValue> _byName = new(StringComparer.Ordinal);
    private int _counter;
    private bool _active;

    private Tracer()
    {
        _sink = new Sink(this);
    }

    /// <summary>Starts recording on the calling thread.</summary>
    /// <returns>The tracer. Dispose it (or call Finish) to stop recording.</returns>
    /// <exception cref="TraceException">A trace is already recording on this thread.</exception>
    public static Tracer Begin()
    {
        var t = new Tracer();
        TraceContext.Enter(t._sink);
        t._active = true;
        return t;
    }

    /// <summary>True while any tracer is recording on the calling thread. Layers whose behaviour a graph cannot hold (random dropout masks) check it.</summary>
    public static bool IsTracing => TraceContext.Current is not null;

    /// <summary>True until <see cref="Finish(ReadOnlySpan{string}, ReadOnlySpan{Tensor})"/> or <see cref="Dispose"/>.</summary>
    public bool IsRecording => _active;

    /// <summary>
    /// Declares a graph input and returns the traced tensor to feed the model. The example's values are used to run the model eagerly while
    /// tracing; they do not enter the graph.
    /// </summary>
    /// <param name="name">The input's name in the traced model; unique.</param>
    /// <param name="example">An example tensor of the dtype and a typical shape (float32, int32 or int64 and so on). Not modified, not kept.</param>
    /// <param name="symbolicAxes">Either empty (every axis fixed) or one entry per axis: a name for an axis whose size may differ at run time
    /// (the same name on several inputs means the same size), or null for a fixed axis. A symbolic axis needs an example size of at least 2, and
    /// different names need different example sizes, so the tracer can tell the sizes apart in reshapes.</param>
    /// <returns>A tensor with the example's data that the tracer follows through the model.</returns>
    /// <exception cref="TraceException">The tracer is not recording, the name is taken, the example is empty or blocked, or the symbolic axes are invalid.</exception>
    public Tensor Input(string name, in Tensor example, params ReadOnlySpan<string?> symbolicAxes)
    {
        RequireActive();
        if (string.IsNullOrEmpty(name) || !_usedNames.Add(name))
        {
            throw Fail(TraceErrorKind.InvalidUsage, $"The input name '{name}' is empty or already used.");
        }

        if (example.IsEmpty || example.IsBlocked)
        {
            throw Fail(TraceErrorKind.InvalidUsage, $"Input '{name}': the example must be a non-empty plain tensor.");
        }

        int rank = example.Rank;
        if (symbolicAxes.Length != 0 && symbolicAxes.Length != rank)
        {
            throw Fail(TraceErrorKind.InvalidUsage, $"Input '{name}': {symbolicAxes.Length} axis names for a rank-{rank} example.");
        }

        var shape = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            shape[i] = example.Shape[i];
        }

        var sym = new SymDim?[rank];
        var v = new TraceValue(this, name, example.DType, shape, sym) { IsInput = true };
        for (int i = 0; i < symbolicAxes.Length; i++)
        {
            if (symbolicAxes[i] is not { } axisName)
            {
                continue;
            }

            if (shape[i] < 2)
            {
                throw Fail(TraceErrorKind.InvalidUsage, $"Input '{name}': axis {i} is symbolic ('{axisName}') but its example size is {shape[i]}; use a size of at least 2.");
            }

            if (_declared.TryGetValue(axisName, out Atom? existing))
            {
                if (existing.Size != shape[i])
                {
                    throw Fail(TraceErrorKind.InvalidUsage, $"Symbol '{axisName}' has example size {existing.Size} on '{existing.Source.Name}' but {shape[i]} on '{name}'.");
                }

                sym[i] = SymDim.Of(existing);
                continue;
            }

            foreach (Atom other in _atoms)
            {
                if (other.Size == shape[i])
                {
                    throw Fail(TraceErrorKind.InvalidUsage,
                        $"Symbols '{other.Name ?? "(derived)"}' and '{axisName}' both have example size {shape[i]}; give them different example sizes so reshapes can tell them apart.");
                }
            }

            var atom = new Atom { Id = _atoms.Count, Name = axisName, Size = shape[i], Source = v, Axis = i };
            _atoms.Add(atom);
            _declared[axisName] = atom;
            sym[i] = SymDim.Of(atom);
        }

        _inputs.Add(v);
        _byName[name] = v;
        return example.WithAutograd(new AutogradMeta(true, null, 0) { Trace = v });
    }

    /// <summary>Names a constant tensor (a weight) so the traced model's initializer carries that name. Optional; unnamed constants get generated names.</summary>
    /// <param name="name">A name that is not an input name or another parameter's.</param>
    /// <param name="tensor">The weight, as it will be used by the model (the same storage and view).</param>
    /// <exception cref="TraceException">The name is taken or the tensor is traced (it depends on an input, so it is not a constant).</exception>
    public void Parameter(string name, in Tensor tensor)
    {
        RequireActive();
        if (tensor.IsEmpty || tensor.Autograd?.Trace is not null)
        {
            throw Fail(TraceErrorKind.InvalidUsage, $"Parameter '{name}' must be a non-empty tensor that does not depend on a traced input.");
        }

        if (string.IsNullOrEmpty(name) || !_usedNames.Add(name))
        {
            throw Fail(TraceErrorKind.InvalidUsage, $"The parameter name '{name}' is empty or already used.");
        }

        _parameterNames[KeyOf(tensor)] = name;
        _parameterStorage[(tensor.Buffer, tensor.Generation, tensor.Offset)] = name;
    }

    /// <summary>Stops recording and returns the traced model with one output.</summary>
    /// <param name="outputName">The output's name.</param>
    /// <param name="output">The model's result; it must depend on an input.</param>
    /// <exception cref="TraceException">The output does not depend on a traced input, or a captured constant was changed during the trace.</exception>
    public TracedModel Finish(string outputName, in Tensor output) => Finish([outputName], [output]);

    /// <summary>Stops recording and returns the traced model.</summary>
    /// <param name="outputNames">Output names, unique and different from the input names.</param>
    /// <param name="outputs">The model's results, one per name; each must depend on an input.</param>
    /// <exception cref="TraceException">An output does not depend on a traced input, names clash, or a captured constant was changed during the trace.</exception>
    public TracedModel Finish(ReadOnlySpan<string> outputNames, ReadOnlySpan<Tensor> outputs)
    {
        RequireActive();
        try
        {
            if (outputNames.Length != outputs.Length || outputs.Length == 0)
            {
                throw Fail(TraceErrorKind.InvalidUsage, "Finish needs one name per output and at least one output.");
            }

            if (_inputs.Count == 0)
            {
                throw Fail(TraceErrorKind.InvalidUsage, "The trace has no inputs; declare them with Input().");
            }

            var outVals = new TraceValue[outputs.Length];
            for (int i = 0; i < outputs.Length; i++)
            {
                if (outputs[i].Autograd?.Trace is not TraceValue tv || tv.Owner != this)
                {
                    throw Fail(TraceErrorKind.InvalidUsage,
                        $"Output '{outputNames[i]}' does not depend on any traced input: it is a constant of the model, or it was computed from data read out of a traced tensor.");
                }

                outVals[i] = tv;
            }

            foreach ((Tensor t, long version, string name) in _captured)
            {
                if (t.Version != version)
                {
                    throw Fail(TraceErrorKind.InPlaceMutation,
                        $"The constant '{name}' was modified in place during the trace (storage version {version} at capture, {t.Version} now); the graph holds the earlier values.");
                }
            }

            return Build(outputNames, outVals);
        }
        finally
        {
            Stop();
        }
    }

    /// <summary>Stops recording without producing a model. Safe to call more than once.</summary>
    public void Dispose() => Stop();

    private void Stop()
    {
        _active = false;
        TraceContext.Exit(_sink);
    }

    private void RequireActive()
    {
        if (!_active)
        {
            throw Fail(TraceErrorKind.InvalidUsage, "The tracer has finished.");
        }
    }

    internal static TraceException Fail(TraceErrorKind kind, string message) => new(kind, message);

    // ---- the sink the eager operations call ---------------------------------------------------------------------------------

    private sealed class Sink : TraceSink
    {
        private readonly Tracer _t;

        public Sink(Tracer t) => _t = t;

        public override bool Owns(object trace) => _t._active && trace is TraceValue v && v.Owner == _t;

        public override Tensor Attach(BackwardRecord record, in Tensor output, int outputIndex)
        {
            if (!_t.DependsOnTrace(record))
            {
                // Only weights and constants went in: this is ordinary eager arithmetic (a transposed weight, say), kept as a constant.
                return output.WithAutograd(new AutogradMeta(true, record, outputIndex));
            }

            TraceValue v = _t.Lower(record, output);
            return output.WithAutograd(new AutogradMeta(true, record, outputIndex) { Trace = v });
        }

        public override void Reject(TraceErrorKind kind, in Tensor tensor, string what)
        {
            string producer = tensor.Autograd?.GradFn is { } r ? $"the result of {r.Op}" : "a traced input";
            throw kind == TraceErrorKind.DataDependent
                ? Fail(kind,
                    $"{what} read the data of a traced tensor ({producer}, shape {tensor.Shape}). Anything decided from it (a branch, a loop count, a copied value) would be frozen into the graph; " +
                    "keep control flow on sizes and flags, not on tensor values.")
                : Fail(kind,
                    $"{what} would change a traced tensor in place ({producer}, shape {tensor.Shape}). The graph records operations, not stores; build a new tensor instead.");
        }
    }

    private bool DependsOnTrace(BackwardRecord record)
    {
        if (record.TraceInputs is { } ins)
        {
            foreach (Tensor t in ins)
            {
                if (IsTracedHere(t))
                {
                    return true;
                }
            }
        }

        if (record.TraceExtras is { } ex)
        {
            foreach (Tensor t in ex)
            {
                if (IsTracedHere(t))
                {
                    return true;
                }
            }
        }

        return false;
    }

    private bool IsTracedHere(in Tensor t) => !t.IsEmpty && t.Autograd?.Trace is TraceValue v && v.Owner == this;

    // ---- constants -----------------------------------------------------------------------------------------------------------

    private readonly record struct ConstKey(TensorBuffer? Buffer, int Generation, int Offset, DType DType, string Shape, string Strides);

    private static ConstKey KeyOf(in Tensor t) =>
        new(t.Buffer, t.Generation, t.Offset, t.DType, t.Shape.ToString(), t.IsBlocked ? "blocked" : t.Strides.ToString());

    private string Constant(in Tensor t)
    {
        if (t.IsBlocked)
        {
            throw Fail(TraceErrorKind.UnsupportedOperation, "A blocked-layout tensor cannot be captured as a constant; convert it to NCHW first.");
        }

        ConstKey key = KeyOf(t);
        if (_constants.TryGetValue(key, out string? existing))
        {
            return existing;
        }

        // A named weight seen through another view (a transposed copy for a matrix product) keeps its name as a prefix.
        string name = _parameterNames.TryGetValue(key, out string? given) ? given
            : _parameterStorage.TryGetValue((t.Buffer, t.Generation, t.Offset), out string? baseName) ? FreshName(baseName + ".view")
            : FreshName("const");
        using Tensor copy = t.Clone();
        if ((long)copy.Length * copy.ElementSize > int.MaxValue)
        {
            throw Fail(TraceErrorKind.UnsupportedOperation, $"The constant '{name}' ({t.Shape}, {t.DType}) is larger than 2 GiB.");
        }

        byte[] raw = copy.RawMutableBytes().ToArray();
        var dims = new long[t.Rank];
        for (int i = 0; i < dims.Length; i++)
        {
            dims[i] = t.Shape[i];
        }

        _initializers.Add(new OnnxInitializerSpec { Name = name, DataType = DTypes.ToOnnx(t.DType), Dims = dims, Raw = raw });
        _constants[key] = name;
        _captured.Add((t, t.Version, name));
        return name;
    }

    private string FreshName(string prefix)
    {
        string n;
        do
        {
            n = prefix + "_" + (_counter++).ToString(System.Globalization.CultureInfo.InvariantCulture);
        }
        while (!_usedNames.Add(n));

        return n;
    }
}
