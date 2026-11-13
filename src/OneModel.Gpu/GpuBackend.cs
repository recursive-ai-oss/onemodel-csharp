using System;
using System.Collections.Generic;
using System.Collections.Frozen;
using System.Collections.Immutable;
using System.Runtime.CompilerServices;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Passes;
using RecursiveAI.OneModel.Graph.Operators.Quant;
using RecursiveAI.OneModel.Graph.Operators.Elementwise;
using RecursiveAI.OneModel.Graph.Runtime;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>Devices a browser made ahead of the sessions that will own them.</summary>
internal static class PreparedDevices
{
    private static readonly System.Collections.Concurrent.ConcurrentQueue<GpuDevice> s_ready = new();

    public static void Add(GpuDevice device) => s_ready.Enqueue(device);

    public static GpuDevice Take() =>
        s_ready.TryDequeue(out GpuDevice? d)
            ? d
            : throw new GpuUnsupportedException("A browser makes its GPU device asynchronously: await OneModelGpu.PrepareDeviceAsync() before creating each GPU session.");
}

/// <summary>
/// The GPU execution backend of one session: a device, the kernels it has created, the weights it holds and the arena for intermediate
/// tensors. Every use of the device goes through <see cref="Gate"/>, so concurrent runs of a session take turns on their GPU regions.
/// </summary>
internal sealed class GpuBackend : IExecutionBackend
{
    private readonly GpuKernelMode _kernelMode;
    private readonly Dictionary<GemmElementType, Gemm> _gemms = [];
    private ElementwiseKernels? _elementwise;
    private NormKernels? _norm;
    private SoftmaxKernels? _softmax;
    private ReduceKernels? _reduce;
    private TransposeKernels? _transpose;
    private IntegerKernels? _integer;
    private GatherKernels? _gather;
    private Int8Gemm? _int8;
    private DynamicQuantizeKernels? _dynamicQuantize;
    private DequantizeKernels? _dequantize;
    private AttentionKernels? _attention;
    private ConvKernels? _conv;
    private PoolKernels? _pool;
    private LayoutKernels? _layout;
    private ResizeKernels? _resize;
    private RecurrentKernels? _recurrent;
    private GpuIndexStatus? _indexStatus;
    private readonly ConditionalWeakTable<PlanStep, GpuBinding> _bindings = [];
    private readonly ConditionalWeakTable<ExecutionPlan, PlanClaims> _claims = [];
    private readonly GpuOptions _options;
    private GpuTimestampSet? _timestamps;
    private GpuTimings _timings;

    private readonly FrozenDictionary<string, GpuOpEntry> _ops;

    private GpuBackend(GpuDevice device, ModelOptions model, GpuOptions options, FrozenDictionary<string, GpuOpEntry> ops)
    {
        Device = device;
        _ops = ops;
        _options = options;
        _kernelMode = model.Mode == ArithmeticMode.Exact ? GpuKernelMode.Deterministic : GpuKernelMode.Fast;
        Arena = new DeviceArena(device, ReleaseKernelBindings);
        Constants = new ConstantStore(device);
        PipelineFirstDispatches = options.PipelineFirstDispatches
            ?? (device.Backend == RecursiveAI.OneGpu.GpuBackend.Metal ? 0 : 6);
        ulong binding = device.Capabilities.Compute.MaxStorageBufferBindingBytes;
        MaxBindingBytes = binding == 0 ? long.MaxValue : (long)Math.Min(binding, long.MaxValue);
        if (options.CollectTimings && device.Capabilities.Timestamps)
        {
            _timestamps = device.CreateTimestampSet(2 * MaxTimedPasses);
        }
    }

    /// <summary>Most compute passes of one region that are timed; later ones run untimed. Two timestamps each, and Metal allows 4096 samples in a set.</summary>
    public const int MaxTimedPasses = 2048;

    /// <summary>
    /// How many sets of buffers a kernel family remembers bind groups for. A model's matrix products alone touch one set per weight, so the
    /// bound is well above the weights of the largest encoders (a few hundred), and a warm run builds none.
    /// </summary>
    public const int BindGroupBound = 4096;

    public string Name => "gpu";

    public GpuDevice Device { get; }

    public DeviceArena Arena { get; }

    public ConstantStore Constants { get; }

    public long MaxBindingBytes { get; internal set; }

    /// <summary>
    /// Arithmetic operations (a multiply-add counts two) after which a long region's recording goes to the device as one submission and
    /// continues in the next. A slow or shared device (the integrated GPU of a desktop processor runs a speech encoder at tens of GFLOP/s)
    /// would otherwise spend seconds in one submission, and the driver resets the queue of a submission that runs for ten.
    /// About 34 GFLOP is a few milliseconds on a discrete card, so the cut costs nothing measurable there.
    /// </summary>
    public double SplitWorkBudget { get; internal set; } = 34_359_738_368.0;

    /// <summary>
    /// Dispatches in the first piece of a region that is handed to the device before the whole region is recorded (each later piece twice as
    /// large); 0 hands the region over whole. The device then starts while the host is still recording, which hides the host's cost behind
    /// the kernels. Measured on Vulkan and in a browser, where it saves 0.5 to 2 ms per run; a Metal commit costs enough that splitting loses (the 24 ms BERT run grows to 27 ms), so Metal hands over whole.
    /// </summary>
    public int PipelineFirstDispatches { get; internal set; }

    public GpuTimestampSet? Timestamps => _timestamps;

    /// <summary>Times each step of a region on its own (for tools that attribute a run's GPU time to operators); slower than a normal run.</summary>
    public bool ProfileSteps => _options.ProfileSteps;

    private readonly Dictionary<string, (double Ms, int Dispatches, int Steps)> _profile = [];

    public void AddProfile(string op, double ms, int dispatches)
    {
        _profile.TryGetValue(op, out var v);
        _profile[op] = (v.Ms + ms, v.Dispatches + dispatches, v.Steps + 1);
    }

    /// <summary>GPU time, dispatches and step count by operator type, summed over the runs since the last reset.</summary>
    public (string Op, double Ms, int Dispatches, int Steps)[] TakeProfile()
    {
        lock (Gate)
        {
            var r = new (string, double, int, int)[_profile.Count];
            int i = 0;
            foreach (var kv in _profile)
            {
                r[i++] = (kv.Key, kv.Value.Ms, kv.Value.Dispatches, kv.Value.Steps);
            }

            _profile.Clear();
            return r;
        }
    }

    public bool TimingsEnabled => _options.CollectTimings;

    /// <summary>Whether a matrix product may take over the bias addition and activation that follow it.</summary>
    public bool FuseEpilogues => _options.FuseOperators && _options.FuseEpilogues;

    /// <summary>Serialises everything that touches the device, the arena, the constants or the kernels' caches.</summary>
    public object Gate { get; } = new();

    /// <summary>
    /// Whether the arena keeps an operator's inputs, outputs and scratch in different buffers. A browser refuses one buffer bound for reading
    /// and for writing in a single dispatch; elsewhere it is off (set <c>ONEMODEL_GPU_SEPARATE_SLABS=1</c> to try the layout on any device).
    /// </summary>
    public bool SeparateReadWriteSlabs { get; internal set; } = OperatingSystem.IsBrowser() || Environment.GetEnvironmentVariable("ONEMODEL_GPU_SEPARATE_SLABS") == "1";

    /// <summary>Makes awaited runs take turns (a lock cannot be held across an await).</summary>
    public System.Threading.SemaphoreSlim AsyncGate { get; } = new(1, 1);

    public static GpuBackend Create(ModelOptions model, GpuOptions options) => Create(model, options, GpuOpTable.Default);

    /// <summary>As above with another operator table, so tests can exercise the rules around entries the real table does not have.</summary>
    internal static GpuBackend Create(ModelOptions model, GpuOptions options, FrozenDictionary<string, GpuOpEntry> ops)
    {
        GpuDevice device;
        try
        {
            // A browser makes its device asynchronously, ahead of the session (OneModelGpu.PrepareDeviceAsync); elsewhere it is made here.
            device = OperatingSystem.IsBrowser()
                ? PreparedDevices.Take()
                : GpuBackends.CreateDevice(new GpuDeviceOptions { Validation = options.Validation, Adapter = options.Adapter, ApplicationName = "RecursiveAI.OneModel" });
        }
        catch (Exception e) when (e is GpuException or DllNotFoundException or EntryPointNotFoundException or PlatformNotSupportedException or InvalidOperationException or NotSupportedException)
        {
            throw new SessionException(SessionErrorKind.DeviceUnsupported, $"No usable GPU device: {e.Message}", null, default, e);
        }

        try
        {
            return new GpuBackend(device, model, options, ops);
        }
        catch (Exception e) when (e is GpuException)
        {
            device.Dispose();
            throw new SessionException(SessionErrorKind.DeviceUnsupported, $"The GPU device could not be set up: {e.Message}", null, default, e);
        }
        catch
        {
            device.Dispose();
            throw;
        }
    }

    /// <summary>The matrix-product kernel for an element type, created on first use in the session's mode.</summary>
    public Gemm GetGemm(DeviceElement element)
    {
        GemmElementType type = element == DeviceElement.F32 ? GemmElementType.F32 : GemmElementType.F16;
        lock (Gate)
        {
            if (!_gemms.TryGetValue(type, out Gemm? g))
            {
                g = new Gemm(Device, type, _kernelMode, maxCachedBindGroups: BindGroupBound);
                _gemms[type] = g;
            }

            return g;
        }
    }

    /// <summary>The kernel mode every kernel of the session runs in: deterministic for exact mode.</summary>
    public GpuKernelMode KernelMode => _kernelMode;

    // The other kernel families are created when the first operator that needs them is bound. Callers hold the gate (binding and encoding do).
    public ElementwiseKernels Elementwise => _elementwise ??= new ElementwiseKernels(Device, BindGroupBound);

    public NormKernels Norm => _norm ??= new NormKernels(Device, maxCachedBindGroups: BindGroupBound);

    public SoftmaxKernels Softmax => _softmax ??= new SoftmaxKernels(Device, maxCachedBindGroups: BindGroupBound);

    public ReduceKernels Reduce => _reduce ??= new ReduceKernels(Device, maxCachedBindGroups: BindGroupBound);

    public TransposeKernels Transpose => _transpose ??= new TransposeKernels(Device, BindGroupBound);

    public IntegerKernels Integer => _integer ??= new IntegerKernels(Device, BindGroupBound);

    public Int8Gemm Int8 => _int8 ??= new Int8Gemm(Device, maxCachedBindGroups: BindGroupBound);

    public DynamicQuantizeKernels DynamicQuantize => _dynamicQuantize ??= new DynamicQuantizeKernels(Device, BindGroupBound);

    public DequantizeKernels Dequantize => _dequantize ??= new DequantizeKernels(Device, BindGroupBound);

    public GatherKernels Gather => _gather ??= new GatherKernels(Device, BindGroupBound);

    public AttentionKernels Attention => _attention ??= new AttentionKernels(Device, maxCachedBindGroups: BindGroupBound);

    public ConvKernels Conv => _conv ??= new ConvKernels(Device, _kernelMode, null, BindGroupBound);

    private ConvIntegerKernels? _convInteger;

    public ConvIntegerKernels ConvInteger => _convInteger ??= new ConvIntegerKernels(Device, maxCachedBindGroups: BindGroupBound);

    public LayoutKernels Layout => _layout ??= new LayoutKernels(Device, BindGroupBound);

    public PoolKernels Pool => _pool ??= new PoolKernels(Device, BindGroupBound);

    /// <summary>The recurrent layers' kernels, in the session's mode.</summary>
    public RecurrentKernels Recurrent => _recurrent ??= new RecurrentKernels(Device, _kernelMode, null, BindGroupBound);

    public ResizeKernels Resize => _resize ??= new ResizeKernels(Device, BindGroupBound);

    /// <summary>Marks that a binding records invalid indices, so the status word exists and runs that use it read it back.</summary>
    public void IndexStatusNeeded() => _indexStatus ??= new GpuIndexStatus(Device);

    /// <summary>The word the gather kernels record an out-of-range index in; created when the first binding needs it.</summary>
    public GpuIndexStatus? IndexStatus => _indexStatus;

    private void ReleaseKernelBindings()
    {
        foreach (Gemm g in _gemms.Values)
        {
            g.ReleaseBindings();
        }

        // Every other family holds bind groups for the old slabs; dropping them lets the old buffers go.
        _elementwise?.ReleaseBindings();
        _norm?.ReleaseBindings();
        _softmax?.ReleaseBindings();
        _reduce?.ReleaseBindings();
        _transpose?.ReleaseBindings();
        _integer?.ReleaseBindings();
        _gather?.ReleaseBindings();
        _attention?.ReleaseBindings();
        _int8?.ReleaseBindings();
        _convInteger?.ReleaseBindings();
        _dynamicQuantize?.ReleaseBindings();
        _dequantize?.ReleaseBindings();
        _conv?.ReleaseBindings();
        _pool?.ReleaseBindings();
        _layout?.ReleaseBindings();
        _resize?.ReleaseBindings();
        _recurrent?.ReleaseBindings();
    }

    /// <summary>What the device reported after the session released everything, to show leaks; empty until the backend is disposed.</summary>
    public string[] MessagesAfterDispose { get; private set; } = [];

    /// <summary>True once <see cref="Dispose"/> has released the device.</summary>
    public bool IsDisposed { get; private set; }

    // ---- IExecutionBackend -------------------------------------------------------------------------------------------------

    // Reshape and its kin cost nothing on the device (the result is the same memory), and claiming them keeps a region in one piece.
    public bool ClaimsViews => true;

    /// <summary>
    /// The graph rewrites the device can use the results of: the fused attention and the one-pass Gelu, which have kernels here. The other
    /// fused forms of the highest level (packed-weight products, convolutions) exist as CPU kernels only. Both fused forms run in exact mode
    /// too, because their device kernels are deterministic.
    /// </summary>
    public bool? ChoosePass(string passName, OptimizationLevel minLevel) => passName switch
    {
        "attention" => _options.FuseOperators && !_options.KeepAttentionUnfused && (_options.FusedAttention ?? true),
        "gelu" => _options.FuseOperators && !_options.KeepGeluUnfused,
        // Convolution rewrites: the batch-norm fold (also in exact mode: the device's result is deterministic, not the CPU's bits) and
        // the activation and residual fused into the kernel's last pass. The CPU's weight packing is of no use here.
        "conv-batchnorm-fold" or "conv-activation" or "conv-residual" => _options.FuseOperators ? true : false,
        _ => minLevel > OptimizationLevel.Basic ? false : null,
    };

    // A fused node keeps the node info of the operator the rewrite ended on, so the kernel says what it really is.
    private static string KeyOf(CompiledNode node) => node.Kernel switch
    {
        AttentionOp => AttentionOps.Key,
        FloatFunctionOp<RecursiveAI.OneModel.Kernels.Exact.GeluUnary> => ElementwiseOps.FusedGeluKey,
        IntegerMatMulOp { ToFloat: true } => QuantOps.FusedProductKey,
        _ => node.Info.OpType,
    };

    public string? CheckNode(CompiledNode node, ArithmeticMode mode)
    {
        NodeInfo info = node.Info;
        if (!info.Ir.IsDefaultDomain)
        {
            return $"{info.Ir.Domain} is not an ONNX-domain operator, and the GPU has kernels for the ONNX domain only";
        }

        if (!_ops.TryGetValue(KeyOf(node), out GpuOpEntry? entry))
        {
            return $"no GPU kernel for {info.OpType}";
        }

        if (mode == ArithmeticMode.Exact && !entry.Deterministic)
        {
            return $"no deterministic GPU kernel for {info.OpType}, so exact mode keeps it on the CPU";
        }

        return entry.Check(info);
    }

    public string? ClaimStep(ExecutionPlan plan, PlanStep step)
    {
        if (CheckNode(step.Node, plan.Env.Mode) is { } declined)
        {
            return declined;
        }

        GpuOpEntry entry = _ops[KeyOf(step.Node)];
        PlanClaims claims = _claims.GetOrCreateValue(plan);
        var ctx = new GpuClaimContext(this, plan, step, claims);
        string? reason;
        GpuBinding? binding;
        try
        {
            lock (Gate)
            {
                reason = entry.Bind(ctx, out binding);
            }
        }
        catch (GpuUnsupportedException e)
        {
            return $"the device lacks what the kernel needs: {e.Message}";
        }

        if (reason is not null || binding is null)
        {
            return reason ?? "the kernel declined the step";
        }

        _bindings.Add(step, binding);
        for (int o = 0; o < step.Node.Outputs.Length; o++)
        {
            if (step.Node.Outputs[o] >= 0)
            {
                claims.Produced[step.Node.Outputs[o]] = binding.OutputElement(o);
            }
        }

        return null;
    }

    public IBackendProgram Compile(ExecutionPlan plan, IReadOnlyList<BackendRegion> regions)
    {
        PlanClaims claims = _claims.GetOrCreateValue(plan);
        try
        {
            lock (Gate)
            {
                return GpuProgram.Create(this, plan, regions, claims, _bindings);
            }
        }
        catch (Exception e) when (e is GpuException or GpuKernelShapeException)
        {
            throw GpuErrors.Wrap("preparing the device for this model", e);
        }
    }

    public void Dispose()
    {
        lock (Gate)
        {
            if (IsDisposed)
            {
                return;
            }

            foreach (Gemm g in _gemms.Values)
            {
                g.Dispose();
            }

            _gemms.Clear();
            _elementwise?.Dispose();
            _norm?.Dispose();
            _softmax?.Dispose();
            _reduce?.Dispose();
            _transpose?.Dispose();
            _integer?.Dispose();
            _gather?.Dispose();
            _int8?.Dispose();
            _dynamicQuantize?.Dispose();
            _convInteger?.Dispose();
            _dequantize?.Dispose();
            _int8 = null;
            _dynamicQuantize = null;
            _convInteger = null;
            _dequantize = null;
            _attention?.Dispose();
            _conv?.Dispose();
            _pool?.Dispose();
            _layout?.Dispose();
            _layout = null;
            _resize?.Dispose();
            _recurrent?.Dispose();
            _recurrent = null;
            _conv = null;
            _pool = null;
            _resize = null;
            _indexStatus?.Dispose();
            _gather = null;
            _attention = null;
            _indexStatus = null;
            _elementwise = null;
            _norm = null;
            _softmax = null;
            _reduce = null;
            _transpose = null;
            _integer = null;
            Constants.Dispose();
            Arena.Dispose();
            _timestamps?.Dispose();
            _timestamps = null;
            Device.Dispose();
            MessagesAfterDispose = Device.TakeValidationMessages();
            IsDisposed = true;
        }
    }

    // ---- statistics -------------------------------------------------------------------------------------------------------

    public void AddTimings(double uploadMs, double gpuMs, double submitReadbackMs, double totalMs, long dispatches, double recordMs = 0, double submitMs = 0, long passes = 0)
    {
        _timings = new GpuTimings(_timings.Regions + 1, _timings.UploadMilliseconds + uploadMs, _timings.GpuMilliseconds + gpuMs,
            _timings.SubmitAndReadbackMilliseconds + submitReadbackMs, _timings.TotalMilliseconds + totalMs, Math.Max(_timings.LongestRegionMilliseconds, totalMs), _timings.Dispatches + dispatches,
            _timings.RecordMilliseconds + recordMs, _timings.SubmitMilliseconds + submitMs, _timings.ComputePasses + passes);
    }

    public void ResetTimings()
    {
        lock (Gate)
        {
            _timings = default;
        }
    }

    public GpuSessionInfo Snapshot(GpuProgram? program)
    {
        lock (Gate)
        {
            var variants = ImmutableArray.CreateBuilder<string>();
            foreach (KeyValuePair<GemmElementType, Gemm> kv in _gemms)
            {
                variants.Add($"{kv.Key.ToString().ToLowerInvariant()} {kv.Value.Mode.ToString().ToLowerInvariant()}: {kv.Value.Variant}");
            }

            string mode = _kernelMode.ToString().ToLowerInvariant();
            if (_norm is not null)
            {
                variants.Add($"norm {mode}: {_norm.Variant}");
            }

            if (_softmax is not null)
            {
                variants.Add($"softmax {mode}: {_softmax.Variant}");
            }

            if (_reduce is not null)
            {
                variants.Add($"reduce {mode}: {_reduce.Variant}");
            }

            if (_conv is not null)
            {
                variants.Add($"conv {mode}: {_conv.GemmVariantFloat32}");
            }

            GpuPlanInfo? plan = program is null ? null
                : new GpuPlanInfo(program.RegionCount, program.FlopsPerRun, program.BytesToBackendPerRun, program.BytesFromBackendPerRun, program.ArenaBytes);
            return new GpuSessionInfo(Device.Adapter.Name, Device.Backend.ToString(), variants.ToImmutable(), Arena.Bytes, Arena.Allocations,
                Constants.Bytes, Constants.Buffers, Arena.Allocations + Constants.Buffers, _timings, plan);
        }
    }
}

/// <summary>Turns device failures into the library's typed errors, naming what was being done.</summary>
internal static class GpuErrors
{
    public static SessionException Wrap(string doing, Exception e) => e switch
    {
        GpuOutOfMemoryException oom => new SessionException(SessionErrorKind.DeviceFailure,
            $"The device ran out of memory while {doing}{(oom.RequestedBytes > 0 ? $" (asked for {oom.RequestedBytes} bytes)" : "")}: {e.Message}", null, default, e),
        GpuDeviceLostException lost => new SessionException(SessionErrorKind.DeviceFailure,
            $"The device was lost while {doing}{(lost.Operation is { } op ? $" ({op})" : "")}: {e.Message}. Create a new session to continue.", null, default, e),
        GpuUnsupportedException un => new SessionException(SessionErrorKind.DeviceFailure,
            $"The device cannot do what was asked while {doing}: {e.Message}{(un.Capability is { } cap ? $" [capability {cap}: expected {un.Expected}, found {un.Actual ?? "nothing"}]" : "")}", null, default, e),
        GpuKernelShapeException shape => new SessionException(SessionErrorKind.Unsupported,
            $"A GPU kernel rejected the shapes while {doing}: {shape.Operation} {shape.Argument}: {e.Message}", null, default, e),
        _ => new SessionException(SessionErrorKind.DeviceFailure, $"The device failed while {doing}: {e.Message}", null, default, e),
    };
}
