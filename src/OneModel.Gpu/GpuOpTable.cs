using System;
using System.Collections.Generic;
using System.Collections.Frozen;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>
/// One operator the GPU backend can run. The table of entries (<see cref="GpuOpTable"/>) is all there is to extend: a new kernel family
/// is a new entry, with no change to partitioning, memory planning or execution.
/// </summary>
/// <param name="OpType">The ONNX operator type in the default domain.</param>
/// <param name="Deterministic">Whether a kernel with a fixed summation order and no atomics exists, so that exact mode may run the operator here.</param>
/// <param name="Check">
/// Called once per node when a session is created, before any shape is known: null when the node's attributes and opset are ones the
/// kernel handles, else the reason it cannot run on the GPU.
/// </param>
/// <param name="Bind">
/// Called once per node and execution plan, when element types and shapes are known. Returns null and a binding when the kernel can run
/// the node with these inputs, else the reason in plain words. May throw <see cref="GpuUnsupportedException"/> when the device lacks a
/// capability; the backend turns that into a reason.
/// </param>
internal sealed record GpuOpEntry(string OpType, bool Deterministic, Func<NodeInfo, string?> Check, GpuBind Bind);

internal delegate string? GpuBind(GpuClaimContext ctx, out GpuBinding? binding);

/// <summary>The operators the backend knows, by type.</summary>
internal static class GpuOpTable
{
    /// <summary>The table every session uses. Adding a kernel family is adding its entries here.</summary>
    public static readonly FrozenDictionary<string, GpuOpEntry> Default = ((GpuOpEntry[])
    [
        MatMulOps.MatMul,
        MatMulOps.Gemm,
        .. ElementwiseOps.Entries,
        .. NormOps.Entries,
        .. LayoutOps.Entries,
        .. IntegerOps.Entries,
        .. GatherOps.Entries,
        .. QuantOps.Entries,
        .. DequantizeOps.Entries,
        .. VisionOps.Entries,
        .. RecurrentOps.Entries,
        // The nodes the attention and Gelu rewrites make are identified by their kernel, not by an ONNX operator type.
        AttentionOps.Entry,
        ElementwiseOps.FusedGelu,
    ]).ToFrozenDictionary(e => e.OpType, StringComparer.Ordinal);
}

/// <summary>
/// What the backend knows about one step when it decides whether to run it: the node, the element types and shapes the plan fixed, and
/// whether each input is a weight or the result of another device step.
/// </summary>
internal sealed class GpuClaimContext
{
    private readonly PlanClaims _claims;

    internal GpuClaimContext(GpuBackend backend, ExecutionPlan plan, PlanStep step, PlanClaims claims)
    {
        Backend = backend;
        Plan = plan;
        Step = step;
        _claims = claims;
    }

    public GpuBackend Backend { get; }

    public ExecutionPlan Plan { get; }

    public PlanStep Step { get; }

    public NodeInfo Info => Step.Node.Info;

    public ArithmeticMode Mode => Plan.Env.Mode;

    public int InputCount => Step.Node.Inputs.Length;

    public bool HasInput(int i) => i < InputCount && Step.Node.Inputs[i] >= 0;

    public int InputId(int i) => Step.Node.Inputs[i];

    public DType InputType(int i) => Plan.Types[InputId(i)];

    /// <summary>The device element type for input <paramref name="i"/> (float32 or float16), or the reason it has none.</summary>
    public string? InputElement(int i, out DeviceElement element) => ElementOf(InputType(i), $"input {i}", out element);

    /// <summary>
    /// The device element type for input <paramref name="i"/> when it is any type the integer, bool and cast kernels hold: float32,
    /// float16, int32, int64 or bool. Returns the reason when it is another type.
    /// </summary>
    public string? InputAnyElement(int i, out DeviceElement element) => AnyElementOf(InputType(i), $"input {i}", out element);

    /// <summary>As <see cref="InputAnyElement"/> for output <paramref name="o"/>.</summary>
    public string? OutputAnyElement(int o, out DeviceElement element) => AnyElementOf(OutputType(o), $"output {o}", out element);

    private static string? AnyElementOf(DType type, string what, out DeviceElement element)
    {
        element = DeviceElement.F32;
        switch (type)
        {
            case DType.F32:
                return null;
            case DType.F16:
                element = DeviceElement.F16;
                return null;
            case DType.I32:
                element = DeviceElement.I32;
                return null;
            case DType.I64:
                element = DeviceElement.I64;
                return null;
            case DType.Bool:
                element = DeviceElement.Bool;
                return null;
            default:
                return $"{what} is {type}, and the GPU kernels take float32, float16, int32, int64 and bool here";
        }
    }

    /// <summary>The device element type of output <paramref name="o"/>, or the reason it has none.</summary>
    public string? OutputElement(int o, out DeviceElement element) => ElementOf(OutputType(o), $"output {o}", out element);

    private static string? ElementOf(DType type, string what, out DeviceElement element)
    {
        element = DeviceElement.F32;
        switch (type)
        {
            case DType.F32:
                return null;
            case DType.F16:
                element = DeviceElement.F16;
                return null;
            default:
                return $"{what} is {type}, and the GPU kernels take float32 and float16";
        }
    }

    /// <summary>Why a device buffer of <paramref name="elements"/> elements of <paramref name="element"/> cannot be bound on this device, or null.</summary>
    public string? FitsBinding(long elements, DeviceElement element) =>
        elements * DeviceArena.ElementBytes(element) > MaxBindingBytes ? $"a tensor of {elements} elements is larger than one buffer binding on this device ({MaxBindingBytes} bytes)"
        : elements >= int.MaxValue ? "a tensor has more elements than a kernel can index" : null;

    /// <summary>A shape as the kernels take it.</summary>
    public static int[] Dims(Shape shape)
    {
        var d = new int[shape.Rank];
        for (int i = 0; i < d.Length; i++)
        {
            d[i] = shape[i];
        }

        return d;
    }

    public Shape InputShape(int i) => Plan.Shapes[InputId(i)];

    public bool InputIsConst(int i) => Plan.Kinds[InputId(i)] == ValueKind.Const;

    public Tensor InputConst(int i) => Plan.Consts[InputId(i)];

    /// <summary>The device element type of input <paramref name="i"/> when another step of this plan made it on the device, else null.</summary>
    public DeviceElement? InputMadeOnDevice(int i) => _claims.Produced.TryGetValue(InputId(i), out DeviceElement e) ? e : null;

    public int OutputId(int o) => Step.Node.Outputs[o];

    public DType OutputType(int o) => Step.OutTypes[o];

    public Shape OutputShape(int o) => Step.OutShapes[o];

    /// <summary>Most bytes a kernel can bind in one buffer on this device.</summary>
    public long MaxBindingBytes => Backend.MaxBindingBytes;

    /// <summary>Hands a constant the binding made for this plan to the plan's program, which disposes it.</summary>
    public void OwnForPlan(DeviceConstant constant) => _claims.Owned.Add(constant);

    /// <summary>
    /// A constant input larger than one buffer binding, as several device buffers (see <see cref="ConstantStore.GetSplit"/>). The pieces of a
    /// plan-local constant belong to the plan's program.
    /// </summary>
    public ConstantChunks SplitConstantInput(int i, int rows, int columns, bool byColumns)
    {
        ConstantChunks c = Backend.Constants.GetSplit(Plan, InputId(i), rows, columns, byColumns, MaxBindingBytes, out bool planLocal);
        if (planLocal)
        {
            _claims.Owned.AddRange(c.Pieces);
        }

        return c;
    }

    /// <summary>Device copy of a constant input; weights are shared by all plans, a constant made by the plan belongs to its program.</summary>
    public DeviceConstant ConstantInput(int i)
    {
        DeviceConstant c = Backend.Constants.Get(Plan, InputId(i), out bool planLocal);
        if (planLocal)
        {
            _claims.Owned.Add(c);
        }

        return c;
    }
}

/// <summary>What a plan's earlier steps have claimed so far.</summary>
internal sealed class PlanClaims
{
    /// <summary>Value ids made by claimed steps, with their element type on the device.</summary>
    public Dictionary<int, DeviceElement> Produced { get; } = [];

    /// <summary>Plan-local constants uploaded for claimed steps; the plan's program disposes them.</summary>
    public List<DeviceConstant> Owned { get; } = [];

}

/// <summary>A claimed step, ready to run: shapes and strides are fixed, constants are on the device, kernels are chosen.</summary>
internal abstract class GpuBinding
{
    /// <summary>Floating-point operations one run of the step does.</summary>
    public double Flops { get; protected init; }

    /// <summary>Bytes the step uploads each run beyond its region inputs (an expanded bias that is not a constant, for instance).</summary>
    public virtual long ExtraUploadBytes => 0;

    /// <summary>The element type output <paramref name="output"/> has on the device. May differ from the host's (a half-precision result is held as binary32 and rounded when it is read back).</summary>
    public abstract DeviceElement OutputElement(int output);

    /// <summary>True when the step writes its first output before its kernel runs, so that output's memory must be reserved from the start of the region.</summary>
    public virtual bool InitializesOutput => false;

    /// <summary>
    /// When the step's first output only reinterprets the storage of input <c>AliasOfInput</c> (a view), that input's index, else -1. The
    /// output then shares the input's arena slot, and the step records nothing.
    /// </summary>
    public virtual int AliasOfInput => -1;

    /// <summary>Float32 elements of scratch the step needs while it runs (0 for none). Lives only during the step, or from the start of the region when <see cref="InitializesOutput"/>.</summary>
    public virtual long ScratchF32Elements => 0;

    /// <summary>Where the program placed the scratch; set once the plan's memory is laid out.</summary>
    public ValueSlot ScratchSlot { get; set; }

    /// <summary>
    /// Float32 elements of a second scratch, for a step that reads one float32 scratch while writing another in the same dispatch (a device
    /// that refuses one buffer bound for reading and writing needs them apart).
    /// </summary>
    public virtual long ScratchF32bElements => 0;

    /// <summary>Where the program placed the second float32 scratch.</summary>
    public ValueSlot ScratchF32bSlot { get; set; }

    /// <summary>Int32 elements of scratch the step needs while it runs (0 for none).</summary>
    public virtual long ScratchI32Elements => 0;

    /// <summary>Where the program placed the int32 scratch.</summary>
    public ValueSlot ScratchI32Slot { get; set; }

    /// <summary>
    /// Int32 elements of a second scratch, for a step that reads one int32 scratch while writing another in the same dispatch (a device that
    /// refuses one buffer bound for reading and writing needs them apart).
    /// </summary>
    public virtual long ScratchI32bElements => 0;

    /// <summary>Where the program placed the second int32 scratch.</summary>
    public ValueSlot ScratchI32bSlot { get; set; }

    /// <summary>Float16 elements of scratch the step needs while it runs (0 for none).</summary>
    public virtual long ScratchF16Elements => 0;

    /// <summary>Where the program placed the float16 scratch.</summary>
    public ValueSlot ScratchF16Slot { get; set; }

    /// <summary>Bool (byte) elements of scratch the step needs while it runs (0 for none).</summary>
    public virtual long ScratchBoolElements => 0;

    /// <summary>Where the program placed the bool scratch.</summary>
    public ValueSlot ScratchBoolSlot { get; set; }

    /// <summary>Signed 8-bit elements of scratch the step needs while it runs (0 for none).</summary>
    public virtual long ScratchI8Elements => 0;

    /// <summary>Where the program placed the signed 8-bit scratch.</summary>
    public ValueSlot ScratchI8Slot { get; set; }

    /// <summary>True when the step passes the backend's index status to a kernel, so a run that includes it reads the status after the submission.</summary>
    public virtual bool UsesIndexStatus => false;

    /// <summary>
    /// The value ids the step reads when fusion changed them, else null for the node's own inputs. A step another step absorbed reads and
    /// writes nothing.
    /// </summary>
    public int[]? InputIds { get; set; }

    /// <summary>The value ids the step writes when fusion changed them, else null for the node's own outputs.</summary>
    public int[]? OutputIds { get; set; }

    /// <summary>Queues host-to-device copies this step needs before the region's commands; called with the run's value table.</summary>
    public virtual void Upload(GpuRunContext ctx, Tensor[] values)
    {
    }

    /// <summary>Records the step's commands.</summary>
    public abstract void Encode(GpuRunContext ctx);
}

/// <summary>A step whose work a matrix product did in its own dispatch instead: it records nothing and touches no value.</summary>
internal sealed class AbsorbedBinding : GpuBinding
{
    public AbsorbedBinding()
    {
        InputIds = [];
        OutputIds = [];
    }

    public override DeviceElement OutputElement(int output) => DeviceElement.F32;

    public override void Encode(GpuRunContext ctx)
    {
    }
}

/// <summary>An addition that can be folded into the product that makes one of its operands, because its other operand is a bias vector.</summary>
internal interface IBiasAdd
{
    /// <summary>
    /// When one operand is the constant vector that adds to every row of the other (which has the result's shape), returns it: the id of the
    /// value being biased, the constant, its value id and the result's id.
    /// </summary>
    bool TryBias(out int valueId, out DeviceConstant bias, out int biasId, out int resultId, out int vectorLength);
}

/// <summary>An elementwise activation that a matrix product can apply to its own result.</summary>
internal interface IProductActivation
{
    bool TryActivation(out GemmActivation activation, out int valueId, out int resultId);
}
