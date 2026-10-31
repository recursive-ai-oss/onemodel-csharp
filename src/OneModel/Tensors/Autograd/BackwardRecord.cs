using System;
using System.Collections.Generic;
using System.Threading;
namespace RecursiveAI.OneModel.Tensors;

/// <summary>
/// What one eager operation keeps so its backward pass can run later: which inputs need gradients and where to send them, the
/// input shapes and types (to undo broadcasting and reshapes), tensors saved from the forward pass (inputs, outputs, or
/// intermediates such as pooling argmax indices, normalized rows or the tanh of GELU) and small attribute vectors.
/// A record exists only when gradients are on and an input requires them, so inference pays nothing. There is no engine yet: a
/// backward function reads the record and computes input gradients from the output gradient.
/// </summary>
/// <remarks>
/// Layout is fixed per <see cref="OpKind"/> (the table is in <c>docs/tensors.md</c>): <c>Saved</c> slots are positional, empty where
/// the operation did not need them (for example <c>Mul</c> saves operand <c>a</c> only when <c>b</c> requires a gradient).
/// </remarks>
internal sealed class BackwardRecord
{
    private readonly AutogradMeta?[] _inputs;
    private readonly Shape[] _inputShapes;
    private readonly DType[] _inputTypes;
    private readonly SavedTensor[] _saved;
    private long[] _ints = [];
    private double[] _floats = [];

    // Engine state for one backward pass: the gradient that has arrived at this record's output, and the pass that last visited it.
    internal Tensor PendingGrad;
    internal int VisitMark;
    internal int Pending;

    /// <summary>True after <see cref="Release"/>: the saved tensors are gone, so backward through this record is no longer possible.</summary>
    public bool IsReleased { get; private set; }

    private BackwardRecord(OpKind op, AutogradMeta?[] inputs, Shape[] shapes, DType[] types, int savedSlots)
    {
        Op = op;
        _inputs = inputs;
        _inputShapes = shapes;
        _inputTypes = types;
        _saved = savedSlots == 0 ? [] : new SavedTensor[savedSlots];
    }

    /// <summary>Starts a record for <paramref name="op"/> over the given inputs.</summary>
    /// <param name="op">The operation.</param>
    /// <param name="savedSlots">Number of positional slots for saved tensors.</param>
    /// <param name="inputs">The operation's inputs in order. An input that does not require gradients gets no edge.</param>
    public static BackwardRecord Create(OpKind op, int savedSlots, params ReadOnlySpan<Tensor> inputs)
    {
        var metas = new AutogradMeta?[inputs.Length];
        var shapes = new Shape[inputs.Length];
        var types = new DType[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            metas[i] = inputs[i].RequiresGrad ? inputs[i].Autograd : null;
            shapes[i] = inputs[i].Shape;
            types[i] = inputs[i].DType;
        }

        var rec = new BackwardRecord(op, metas, shapes, types, savedSlots);
        if (TraceContext.Current is not null)
        {
            rec.TraceInputs = inputs.ToArray();
        }

        return rec;
    }

    /// <summary>The inputs as passed to <see cref="Create"/>, kept only while a trace records (the tracer needs the tensors, not just their gradient edges).</summary>
    internal Tensor[]? TraceInputs { get; private set; }

    /// <summary>Inputs the operation does not differentiate through (index tensors, a mask, running statistics), for the tracer. No effect outside a trace.</summary>
    internal Tensor[]? TraceExtras { get; private set; }

    internal void AddTraceExtras(params ReadOnlySpan<Tensor> extras)
    {
        if (TraceContext.Current is not null)
        {
            TraceExtras = extras.ToArray();
        }
    }

    /// <summary>The operation.</summary>
    public OpKind Op { get; }

    /// <summary>Number of inputs.</summary>
    public int InputCount => _inputs.Length;

    /// <summary>True if input <paramref name="i"/> requires a gradient (so backward must produce one for it).</summary>
    public bool NeedsInputGrad(int i) => _inputs[i] is not null;

    /// <summary>The autograd node of input <paramref name="i"/> (where its gradient goes), or null when it needs none.</summary>
    public AutogradMeta? InputMeta(int i) => _inputs[i];

    /// <summary>The logical shape input <paramref name="i"/> had (needed to sum a broadcast gradient back down).</summary>
    public Shape InputShape(int i) => _inputShapes[i];

    /// <summary>The element type input <paramref name="i"/> had.</summary>
    public DType InputDType(int i) => _inputTypes[i];

    /// <summary>Number of saved-tensor slots.</summary>
    public int SavedCount => _saved.Length;

    /// <summary>Integer attributes (axes, window sizes, strides, ...), meaning fixed per <see cref="OpKind"/>.</summary>
    public ReadOnlySpan<long> Ints => _ints;

    /// <summary>Floating-point attributes (scales, epsilons, ...), meaning fixed per <see cref="OpKind"/>.</summary>
    public ReadOnlySpan<double> Floats => _floats;

    /// <summary>Sets the integer attributes.</summary>
    public void SetInts(params ReadOnlySpan<long> values) => _ints = values.ToArray();

    /// <summary>Sets the floating-point attributes.</summary>
    public void SetFloats(params ReadOnlySpan<double> values) => _floats = values.ToArray();

    /// <summary>
    /// Saves <paramref name="value"/> in <paramref name="slot"/>, remembering its storage version. The saved tensor shares storage
    /// with the original, so no data is copied.
    /// </summary>
    /// <param name="slot">The positional slot.</param>
    /// <param name="value">The tensor (or view) to keep.</param>
    /// <param name="owned">True if the record owns <paramref name="value"/> and frees it in <see cref="Release"/>.</param>
    public void Save(int slot, in Tensor value, bool owned = false) => _saved[slot] = new SavedTensor(value, owned);

    /// <summary>The slot's saved tensor (version checked on <see cref="SavedTensor.Unpack"/>).</summary>
    public SavedTensor Saved(int slot) => _saved[slot];

    /// <summary>Makes <paramref name="output"/> the result of this record: returns a handle to the same data that requires a gradient and points here.</summary>
    public Tensor Attach(in Tensor output, int outputIndex = 0) =>
        TraceContext.Current is { } sink
            ? sink.Attach(this, output, outputIndex)
            : output.WithAutograd(new AutogradMeta(true, this, outputIndex));

    /// <summary>The record's saved tensors, or null if none, as a span for the engine and for composite operations.</summary>
    internal ReadOnlySpan<SavedTensor> SavedSlots => _saved;

    /// <summary>Frees the tensors the record owns. Call after backward has run (or when the graph is dropped).</summary>
    public void Release()
    {
        IsReleased = true;
        for (int i = 0; i < _saved.Length; i++)
        {
            if (_saved[i].Owned)
            {
                _saved[i].Value.Dispose();
            }

            _saved[i] = default;
        }
    }
}
