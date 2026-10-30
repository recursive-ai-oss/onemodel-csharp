using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Text;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Runtime;

// The operator contract. docs/graph-runtime.md is the normative text; this file is the code form of it.
//
// One class per operator (or per family pattern) in its own file under Graph/Operators/, one OpDef line in OpTable.cs.
//   1. The OpDef names the operator type, its domain and the opset range one class serves.
//   2. Its factory runs once per node when the session is created, parses the attributes, and returns an OpKernel.
//   3. Infer runs when a plan is made for a set of concrete input shapes: it states each output's element type and shape.
//   4. Run runs on every Run call, on spans and tensors, and allocates no managed memory.
// A kernel object is shared by all plans and all concurrent runs, so it holds only what its attributes fixed: no per-run state.

/// <summary>Creates the kernel for one node.</summary>
internal delegate OpKernel OpFactory(NodeInfo node);

/// <summary>An operator type, the opsets one kernel class serves, and how to build it.</summary>
internal sealed class OpDef
{
    /// <param name="opType">The ONNX operator type, such as "Add".</param>
    /// <param name="minOpset">The first opset version this class serves (inclusive).</param>
    /// <param name="maxOpset">The last opset version this class serves (inclusive); <see cref="int.MaxValue"/> for no end.</param>
    /// <param name="create">Builds the kernel; static lambdas and method groups only.</param>
    /// <param name="domain">The operator domain; "" is the default ONNX domain.</param>
    public OpDef(string opType, int minOpset, int maxOpset, OpFactory create, string domain = "")
    {
        OpType = opType;
        MinOpset = minOpset;
        MaxOpset = maxOpset;
        Create = create;
        Domain = domain == "ai.onnx" ? "" : domain;
    }

    public string OpType { get; }

    public string Domain { get; }

    public int MinOpset { get; }

    public int MaxOpset { get; }

    public OpFactory Create { get; }
}

/// <summary>What a factory may read about its node: attributes, arity, name, opset.</summary>
internal sealed class NodeInfo
{
    private readonly IrNode _node;

    internal NodeInfo(IrNode node, int opset)
    {
        _node = node;
        Opset = opset;
    }

    public string Name => _node.Name;

    public string OpType => _node.OpType;

    public int Opset { get; }

    /// <summary>Declared input slots, including empty optional ones.</summary>
    public int InputCount => _node.InputNames.Length;

    /// <summary>Declared output slots, including unused ones.</summary>
    public int OutputCount => _node.OutputNames.Length;

    /// <summary>Whether input <paramref name="i"/> is present (an empty name marks an omitted optional input).</summary>
    public bool HasInput(int i) => i < _node.Inputs.Length && _node.Inputs[i] is not null;

    /// <summary>The underlying IR node, for operators that read attributes this class has no helper for (graphs, for If/Loop/Scan).</summary>
    public IrNode Ir => _node;

    public bool Has(string attribute) => _node.FindAttribute(attribute) is not null;

    public long GetInt(string attribute, long fallback) => _node.FindAttribute(attribute) is { } a ? Expect(a, IrAttributeType.Whole).Whole : fallback;

    public float GetFloat(string attribute, float fallback) => _node.FindAttribute(attribute) is { } a ? Expect(a, IrAttributeType.Real).Real : fallback;

    public string GetString(string attribute, string fallback) => _node.FindAttribute(attribute) is { } a ? Expect(a, IrAttributeType.Text).GetText() : fallback;

    public ImmutableArray<long> GetInts(string attribute) => _node.FindAttribute(attribute) is { } a ? Expect(a, IrAttributeType.Wholes).Wholes : [];

    public ImmutableArray<float> GetFloats(string attribute) => _node.FindAttribute(attribute) is { } a ? Expect(a, IrAttributeType.Reals).Reals : [];

    public IrAttr? Find(string attribute) => _node.FindAttribute(attribute);

    /// <summary>A model error attributed to this node.</summary>
    public SessionException Error(string message) => new(SessionErrorKind.InvalidModel, $"{Describe()}: {message}", _node.Name);

    /// <summary>A "not implemented yet" error attributed to this node.</summary>
    public SessionException Unsupported(string message) => new(SessionErrorKind.Unsupported, $"{Describe()}: {message}", _node.Name);

    internal string Describe()
    {
        var sb = new StringBuilder();
        sb.Append(_node.OpType);
        if (_node.Name.Length > 0)
        {
            sb.Append(" '").Append(_node.Name).Append('\'');
        }

        return sb.ToString();
    }

    private IrAttr Expect(IrAttr a, IrAttributeType type) =>
        a.Type == type ? a : throw Error($"attribute '{a.Name}' is {a.Type}, expected {type}");
}

/// <summary>
/// The behaviour of one node. Instances are immutable after construction and shared by every plan and every concurrent run.
/// </summary>
internal abstract class OpKernel : IDisposable
{
    /// <summary>
    /// Whether this kernel has an exact flavour: with <see cref="ExecutionMode.Exact"/> its <see cref="Run"/> must be bit-identical on
    /// every machine (docs/exact-mode.md). A kernel that returns false is "not exact"; a session in exact mode refuses to be created
    /// with it.
    /// </summary>
    public virtual bool SupportsExact => false;

    /// <summary>
    /// True when the output shapes depend on the input values (NonZero, TopK with a computed k, ...). Such a kernel's outputs are
    /// allocated by the kernel itself during <see cref="Run"/>, and everything downstream is shaped at run time.
    /// </summary>
    public virtual bool OutputsDataDependent => false;

    /// <summary>
    /// Index of an input whose storage output 0 may reuse when that input is not needed afterwards (same element type and size). The
    /// kernel must then be correct with output and input being the same memory. -1: never.
    /// </summary>
    public virtual int InPlaceInput => -1;

    /// <summary>
    /// Index of an input of which output 0 is a reshaped view (Reshape, Identity, Squeeze, ...). The session builds the view itself;
    /// <see cref="Run"/> is not called. -1: not a view.
    /// </summary>
    public virtual int ViewOfInput => -1;

    /// <summary>States the element type and shape of every output for the input types and shapes in <paramref name="ctx"/>.</summary>
    public abstract void Infer(InferContext ctx);

    /// <summary>
    /// Called once when a plan is made, after shape inference, for a node that will run. A kernel can turn its constant inputs into a
    /// form it runs faster on (a transposed or packed weight) and return it; the plan hands it back as <see cref="OpRun.State"/> on
    /// every run. The state is shared by concurrent runs and must not be written during <see cref="Run"/>.
    /// </summary>
    public virtual object? Prepare(PrepareContext ctx) => null;

    /// <summary>Computes the outputs. Allocates no managed memory in steady state.</summary>
    public abstract void Run(in OpRun run);

    /// <summary>Releases anything the kernel built when it was created (a parsed constant tensor, say).</summary>
    public virtual void Dispose()
    {
    }
}

/// <summary>What <see cref="OpKernel.Infer"/> sees and fills in. Reused from node to node: read it during the call only.</summary>
internal sealed class InferContext
{
    private DType[] _inType = new DType[8];
    private Shape[] _inShape = new Shape[8];
    private Tensor[] _inValue = new Tensor[8];
    private bool[] _inPresent = new bool[8];

    private DType[] _outType = new DType[4];
    private Shape[] _outShape = new Shape[4];
    private bool[] _outSet = new bool[4];
    private Tensor[] _outConst = new Tensor[4];
    private bool[] _outConstOwned = new bool[4];

    private NodeInfo? _node;

    /// <summary>The kernel's node, for error messages.</summary>
    public string Describe => _node?.Describe() ?? "node";

    public int InputCount { get; private set; }

    public int OutputCount { get; private set; }

    /// <summary>Set by the kernel when it needs input values it cannot see now: the node is then shaped again at run time.</summary>
    public bool NeedsRunTimeValues { get; private set; }

    internal void Begin(NodeInfo node, int inputs, int outputs)
    {
        if (inputs > _inType.Length)
        {
            int n = Math.Max(inputs, _inType.Length * 2);
            _inType = new DType[n];
            _inShape = new Shape[n];
            _inValue = new Tensor[n];
            _inPresent = new bool[n];
        }

        if (outputs > _outType.Length)
        {
            int n = Math.Max(outputs, _outType.Length * 2);
            _outType = new DType[n];
            _outShape = new Shape[n];
            _outSet = new bool[n];
            _outConst = new Tensor[n];
            _outConstOwned = new bool[n];
        }

        _node = node;
        InputCount = inputs;
        OutputCount = outputs;
        NeedsRunTimeValues = false;
        Array.Clear(_inPresent, 0, inputs);
        Array.Clear(_outSet, 0, outputs);
        Array.Clear(_outConst, 0, outputs);
        Array.Clear(_outConstOwned, 0, outputs);
    }

    internal void SetInput(int i, DType type, Shape shape, Tensor value)
    {
        _inPresent[i] = true;
        _inType[i] = type;
        _inShape[i] = shape;
        _inValue[i] = value;
    }

    internal bool OutputIsSet(int i) => _outSet[i];

    internal DType OutputType(int i) => _outType[i];

    internal Shape OutputShape(int i) => _outShape[i];

    internal Tensor OutputConstant(int i) => _outConst[i];

    internal bool OutputConstantOwned(int i) => _outConstOwned[i];

    /// <summary>Whether input <paramref name="i"/> is present (optional inputs may be omitted).</summary>
    public bool HasInput(int i) => i < InputCount && _inPresent[i];

    public DType InputType(int i) => _inType[i];

    public Shape InputShape(int i) => _inShape[i];

    /// <summary>
    /// The values of input <paramref name="i"/> when they are known now (a constant, or any value when shaping at run time). Operators
    /// whose output shape depends on a value (Reshape's shape input) read it here.
    /// </summary>
    public bool TryGetInputValue(int i, out Tensor value)
    {
        value = _inValue[i];
        return _inPresent[i] && !value.IsEmpty;
    }

    /// <summary>
    /// For the half-precision adapter: treats every binary16 and bfloat16 input as binary32 (their values are hidden, so a kernel cannot
    /// read them as float) and returns the element type the outputs should be stored in: the type of the first such input that is not
    /// marked in <paramref name="keepFloat"/>, or <see cref="DType.F32"/> when there is none.
    /// </summary>
    internal DType WidenHalfInputs(ReadOnlySpan<bool> keepFloat)
    {
        DType storage = DType.F32;
        bool found = false;
        for (int i = 0; i < InputCount; i++)
        {
            if (_inPresent[i] && _inType[i] is DType.F16 or DType.BF16)
            {
                if (!found && !(i < keepFloat.Length && keepFloat[i]))
                {
                    storage = _inType[i];
                    found = true;
                }

                _inType[i] = DType.F32;
                _inValue[i] = default;
            }
        }

        return storage;
    }

    /// <summary>For the half-precision adapter: outputs the kernel stated as binary32 are stored as <paramref name="storage"/> instead.</summary>
    internal void RetypeFloatOutputs(DType storage)
    {
        for (int i = 0; i < OutputCount; i++)
        {
            if (_outSet[i] && _outType[i] == DType.F32 && storage != DType.F32)
            {
                if (!_outConst[i].IsEmpty)
                {
                    throw Unsupported("a constant binary32 result of a half-precision node is not supported");
                }

                _outType[i] = storage;
            }
        }
    }

    /// <summary>Asks for the node to be shaped at run time, when the values it needs exist. Call it and return.</summary>
    public void RequireRunTimeValues() => NeedsRunTimeValues = true;

    /// <summary>States the element type and shape of output <paramref name="i"/>.</summary>
    public void SetOutput(int i, DType type, Shape shape)
    {
        _outSet[i] = true;
        _outType[i] = type;
        _outShape[i] = shape;
    }

    /// <summary>
    /// States that output <paramref name="i"/> is a constant for these input shapes, so the node needs no execution (Constant, Shape,
    /// Size). With <paramref name="planOwns"/> the plan disposes the tensor; otherwise the kernel keeps it alive.
    /// </summary>
    public void SetConstantOutput(int i, Tensor value, bool planOwns)
    {
        _outSet[i] = true;
        _outType[i] = value.DType;
        _outShape[i] = value.Shape;
        _outConst[i] = value;
        _outConstOwned[i] = planOwns;
    }

    /// <summary>A model error attributed to the node being shaped.</summary>
    public SessionException Fail(string message) => new(SessionErrorKind.InvalidModel, $"{Describe}: {message}", _node?.Name);

    /// <summary>A "not implemented yet" error attributed to the node being shaped.</summary>
    public SessionException Unsupported(string message) => new(SessionErrorKind.Unsupported, $"{Describe}: {message}", _node?.Name);
}

/// <summary>What <see cref="OpKernel.Prepare"/> sees: which inputs are constants, their values, and a way to create plan-owned tensors.</summary>
internal sealed class PrepareContext
{
    private readonly Tensor[] _constants;
    private readonly List<Tensor> _owned;

    internal PrepareContext(Tensor[] constants, List<Tensor> owned, ExecutionMode mode, KernelPath path, DType[]? inputTypes = null)
    {
        InputTypes = inputTypes ?? [];
        _constants = constants;
        _owned = owned;
        Mode = mode;
        Path = path;
    }

    public ExecutionMode Mode { get; }

    public KernelPath Path { get; }

    /// <summary>The same context with other constants (the half-precision adapter hides or widens some).</summary>
    internal PrepareContext WithConstants(Tensor[] constants) => new(constants, _owned, Mode, Path, InputTypes);

    /// <summary>The element type of each input slot in this plan (undefined for omitted optional inputs).</summary>
    public DType[] InputTypes { get; }

    /// <summary>Whether input <paramref name="i"/> is a constant (a weight, or the result of constant evaluation).</summary>
    public bool IsConstant(int i) => i < _constants.Length && !_constants[i].IsEmpty;

    /// <summary>The value of constant input <paramref name="i"/>.</summary>
    public Tensor Constant(int i) => _constants[i];

    /// <summary>A new tensor that lives as long as the plan; the kernel fills it once, now.</summary>
    public Tensor CreateOwned(DType type, Shape shape)
    {
        Tensor t = Tensor.Create(type, shape);
        _owned.Add(t);
        return t;
    }
}

/// <summary>What <see cref="OpKernel.Run"/> sees: the input tensors, the output tensors to fill, the mode and the resources.</summary>
internal readonly ref struct OpRun
{
    private readonly Arena _arena;

    internal OpRun(ReadOnlySpan<Tensor> inputs, Span<Tensor> outputs, ExecutionMode mode, KernelPath path, int threads, Arena arena,
        object? state = null)
    {
        State = state;
        Inputs = inputs;
        Outputs = outputs;
        Mode = mode;
        Path = path;
        Threads = threads;
        _arena = arena;
    }

    /// <summary>The same run with other tensors and state (the half-precision adapter hands a kernel binary32 copies).</summary>
    internal OpRun With(ReadOnlySpan<Tensor> inputs, Span<Tensor> outputs, object? state) =>
        new(inputs, outputs, Mode, Path, Threads, _arena, state);

    /// <summary>Input tensors, contiguous; an omitted optional input is an empty tensor. Never written.</summary>
    public ReadOnlySpan<Tensor> Inputs { get; }

    /// <summary>
    /// Output tensors, contiguous and already allocated for the shapes <see cref="OpKernel.Infer"/> gave, to be overwritten. For a
    /// data-dependent kernel they are empty until <see cref="AllocateOutput"/> is called.
    /// </summary>
    public Span<Tensor> Outputs { get; }

    /// <summary>What <see cref="OpKernel.Prepare"/> returned for this node in this plan, or null.</summary>
    public object? State { get; }

    /// <summary>The session's mode.</summary>
    public ExecutionMode Mode { get; }

    /// <summary>True in exact mode.</summary>
    public bool Exact => Mode == ExecutionMode.Exact;

    /// <summary>The kernel mode for the shared kernels (<see cref="Gemm"/> and friends).</summary>
    public KernelMode KernelMode => Exact ? KernelMode.Exact : KernelMode.Fast;

    /// <summary>The instruction set chosen once when the session was created.</summary>
    public KernelPath Path { get; }

    /// <summary>The most threads the kernel may use. Changes speed only.</summary>
    public int Threads { get; }

    /// <summary>Scratch storage that lives until the end of this run, allocation-free after warm-up. Contents are undefined.</summary>
    public Tensor Scratch(DType type, Shape shape) => Tensor.Create(type, shape, _arena);

    /// <summary>Allocates output <paramref name="i"/> of a data-dependent kernel; it lives until the end of this run.</summary>
    public Tensor AllocateOutput(int i, DType type, Shape shape)
    {
        Tensor t = Tensor.Create(type, shape, _arena);
        Outputs[i] = t;
        return t;
    }
}
