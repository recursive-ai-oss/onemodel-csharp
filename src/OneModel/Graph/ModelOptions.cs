using System;

namespace RecursiveAI.OneModel.Graph;

/// <summary>How much the graph is rewritten before it is planned.</summary>
public enum OptimizationLevel
{
    /// <summary>The graph runs as imported (apart from evaluating constant subgraphs when a plan is made).</summary>
    None = 0,

    /// <summary>Rewrites that never change a result: constant subgraphs are evaluated once when the session is created.</summary>
    Basic = 1,

    /// <summary>
    /// The default. Everything in <see cref="Basic"/>, plus constant weights packed once for the matrix kernels and operator fusion
    /// (bias and activation applied to each finished tile, Gelu and attention patterns, convolution with its batch normalization).
    /// In <see cref="ArithmeticMode.Exact"/> only the passes that keep every bit run; see docs/graph-passes.md.
    /// </summary>
    All = 2,
}

/// <summary>How a session stores the large constant weights of float32 models.</summary>
public enum WeightStorage
{
    /// <summary>Weights stay as the model has them. The default.</summary>
    Original = 0,

    /// <summary>
    /// Weights of matrix products and convolutions are rounded to IEEE binary16 and kept that way: half the memory, widened to float32
    /// for each use. The model's results change by the rounding of its weights.
    /// </summary>
    Binary16 = 1,

    /// <summary>As <see cref="Binary16"/> with bfloat16 (same range as float32, 8 bits of precision).</summary>
    BFloat16 = 2,
}

/// <summary>Where a session runs the operators it can.</summary>
public enum ComputeDevice
{
    /// <summary>The built-in CPU kernels. The default.</summary>
    Cpu = 0,

    /// <summary>
    /// A graphics processor, through the <c>RecursiveAI.OneModel.Gpu</c> package, which must be registered first
    /// (<c>OneModelGpu.Register()</c>). Operators it has no kernel for run on the CPU unless <see cref="ModelOptions.RequireGpu"/> is set.
    /// </summary>
    Gpu = 1,
}

/// <summary>Settings fixed when a <see cref="Session"/> is created.</summary>
public sealed class ModelOptions
{
    /// <summary>Fast or exact arithmetic. Default <see cref="ArithmeticMode.Fast"/>.</summary>
    public ArithmeticMode Mode { get; init; } = ArithmeticMode.Fast;

    /// <summary>
    /// The most threads one <c>Run</c> may use inside an operator. 0 (the default) means one per logical core, up to 256. 1 runs
    /// everything on the calling thread. Changes speed only: results never depend on it.
    /// </summary>
    public int IntraOpThreads { get; init; }

    /// <summary>Graph rewriting level. Default <see cref="OptimizationLevel.All"/>.</summary>
    public OptimizationLevel OptimizationLevel { get; init; } = OptimizationLevel.All;

    /// <summary>
    /// Whether large float32 weights are stored in half precision to save memory. Default <see cref="WeightStorage.Original"/>. Applies to
    /// constant weights of MatMul, Gemm and Conv with at least 1024 values; each use widens them to float32 first, which costs time, so it
    /// is a memory-for-speed trade. Models that are already stored in half precision run in any case, with their weights widened once
    /// when the session plans a run.
    /// </summary>
    public WeightStorage WeightStorage { get; init; } = WeightStorage.Original;

    /// <summary>
    /// How many execution plans (one per distinct set of input shapes) a session keeps. When more distinct shapes arrive, the least
    /// recently created plan is dropped. Default 16, at least 1.
    /// </summary>
    public int MaxCachedPlans { get; init; } = 16;

    /// <summary>
    /// Where the session runs operators. Default <see cref="ComputeDevice.Cpu"/>. With <see cref="ComputeDevice.Gpu"/> every operator
    /// the GPU package has a kernel for, with the model's element type and attributes, runs there and the rest stay on the CPU; tensors
    /// cross between the two only at the borders of runs of GPU operators. <see cref="Session.GetDiagnostics"/> lists each operator's
    /// placement and the reason for each one left on the CPU.
    /// </summary>
    /// <remarks>
    /// Results on the GPU are not the CPU's bits. <see cref="ArithmeticMode.Fast"/> uses the fastest kernels. <see cref="ArithmeticMode.Exact"/>
    /// uses kernels with a fixed summation order and no atomics, so two runs on the same GPU and driver give the same bits, but another
    /// GPU or driver may differ in the last bits: the exact mode's promise of identical bits on every machine holds for the CPU only.
    /// An operator with no such kernel stays on the CPU in exact mode. With <see cref="ComputeDevice.Gpu"/> only the graph
    /// rewrites whose fused operators the device runs are applied; the CPU's packed-weight rewrites are not.
    /// </remarks>
    public ComputeDevice Device { get; init; } = ComputeDevice.Cpu;

    /// <summary>
    /// Makes <see cref="ComputeDevice.Gpu"/> all or nothing: creating the session fails with <see cref="SessionErrorKind.DeviceUnsupported"/>,
    /// naming every operator that cannot run on the GPU and why, instead of leaving those on the CPU. Default false. Requires
    /// <see cref="Device"/> to be <see cref="ComputeDevice.Gpu"/>. A reason that depends on input shapes (a size above a device limit)
    /// is only known when a shape is first run, and fails that run the same way.
    /// </summary>
    public bool RequireGpu { get; init; }

    internal void Validate()
    {
        if (!Enum.IsDefined(Device))
        {
            throw new ArgumentOutOfRangeException(nameof(Device), Device, "Unknown device.");
        }

        if (RequireGpu && Device != ComputeDevice.Gpu)
        {
            throw new ArgumentException("RequireGpu needs Device = ComputeDevice.Gpu.", nameof(RequireGpu));
        }

        if (!Enum.IsDefined(Mode))
        {
            throw new ArgumentOutOfRangeException(nameof(Mode), Mode, "Unknown execution mode.");
        }

        if (IntraOpThreads is < 0 or > 256)
        {
            throw new ArgumentOutOfRangeException(nameof(IntraOpThreads), IntraOpThreads, "Threads must be between 0 and 256.");
        }

        if (!Enum.IsDefined(OptimizationLevel))
        {
            throw new ArgumentOutOfRangeException(nameof(OptimizationLevel), OptimizationLevel, "Unknown optimization level.");
        }

        if (!Enum.IsDefined(WeightStorage))
        {
            throw new ArgumentOutOfRangeException(nameof(WeightStorage), WeightStorage, "Unknown weight storage.");
        }

        if (MaxCachedPlans < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxCachedPlans), MaxCachedPlans, "At least one plan must be cached.");
        }
    }
}
