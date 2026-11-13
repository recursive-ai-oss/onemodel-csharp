using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>
/// The compiled regions of one execution plan: for every value that lives on the device, its slot in the session's arena, and for every
/// region the steps to record. A run uploads a region's inputs, records and submits its steps once, and reads its outputs back.
/// </summary>
internal sealed class GpuProgram : IBackendProgram
{
    private readonly GpuBackend _backend;
    private readonly BackendRegion[] _regions;
    private readonly GpuBinding[][] _bindings;
    private readonly Dictionary<int, ValueSlot>[] _slots;
    private readonly GpuRunContext _context;
    private readonly List<DeviceConstant> _owned;
    private readonly DeviceArena.Layout _layout;
    private ulong[] _stamps = [];

    private GpuProgram(GpuBackend backend, BackendRegion[] regions, GpuBinding[][] bindings, DeviceArena.Layout layout, Dictionary<int, ValueSlot>[] slots,
        List<DeviceConstant> owned, long toBackend, long fromBackend, double flops)
    {
        _backend = backend;
        _regions = regions;
        _bindings = bindings;
        _layout = layout;
        _slots = slots;
        _owned = owned;
        _context = new GpuRunContext(backend);
        BytesToBackendPerRun = toBackend;
        BytesFromBackendPerRun = fromBackend;
        FlopsPerRun = flops;
    }

    public long BytesToBackendPerRun { get; }

    public long BytesFromBackendPerRun { get; }

    public double FlopsPerRun { get; }

    public int RegionCount => _regions.Length;

    public long ArenaBytes => _layout.Bytes;

    /// <summary>Plans the arena for every region of the plan and makes sure the session's slabs are big enough. Caller holds the gate.</summary>
    public static GpuProgram Create(GpuBackend backend, ExecutionPlan plan, IReadOnlyList<BackendRegion> regions, PlanClaims claims,
        ConditionalWeakTable<PlanStep, GpuBinding> bindingOf)
    {
        var all = new BackendRegion[regions.Count];
        var bindings = new GpuBinding[regions.Count][];
        var live = new List<LiveValue>();
        var keys = new Dictionary<int, int>[regions.Count];
        var aliases = new Dictionary<int, int>[regions.Count];
        var scratchKeys = new List<(GpuBinding Binding, int Key, DeviceElement Element, bool Second)>();
        long toBackend = 0, fromBackend = 0;
        double flops = 0;
        int clock = 0;
        var fusedConstants = new List<DeviceConstant>();
        for (int r = 0; r < all.Length; r++)
        {
            BackendRegion region = regions[r];
            all[r] = region;
            bindings[r] = new GpuBinding[region.Steps.Length];
            int begin = clock, end = clock + region.Steps.Length;
            var lastUse = new Dictionary<int, int>();
            var born = new Dictionary<int, int>();
            var keyOf = keys[r] = [];
            // A view shares its source's memory, so every use of the view is a use of the root value it ends up aliasing.
            var alias = aliases[r] = [];
            int Root(int id) => alias.TryGetValue(id, out int root) ? root : id;
            var scratchAt = new List<(int Step, GpuBinding Binding)>();
            for (int s = 0; s < region.Steps.Length; s++)
            {
                PlanStep step = region.Steps[s];
                if (!bindingOf.TryGetValue(step, out GpuBinding? binding))
                {
                    throw new InvalidOperationException($"{step.Node.Info.Describe()} was placed on the device without being claimed.");
                }

                bindings[r][s] = binding;
            }

            if (backend.FuseEpilogues)
            {
                FoldEpilogues(backend, plan, region, bindings[r], fusedConstants);
            }

            for (int s = 0; s < region.Steps.Length; s++)
            {
                PlanStep step = region.Steps[s];
                GpuBinding binding = bindings[r][s];
                flops += binding.Flops;
                toBackend += binding.ExtraUploadBytes;
                foreach (int id in binding.InputIds ?? step.Node.Inputs)
                {
                    if (id >= 0 && plan.Kinds[id] != ValueKind.Const)
                    {
                        lastUse[Root(id)] = begin + s;
                    }
                }

                int[] stepOutputs = binding.OutputIds ?? step.Node.Outputs;
                if (binding.AliasOfInput >= 0)
                {
                    alias[step.Node.Outputs[0]] = Root(step.Node.Inputs[binding.AliasOfInput]);
                    continue;
                }

                if (binding.ScratchF32Elements > 0 || binding.ScratchF32bElements > 0 || binding.ScratchF16Elements > 0 || binding.ScratchI32Elements > 0 || binding.ScratchI32bElements > 0 || binding.ScratchBoolElements > 0 || binding.ScratchI8Elements > 0)
                {
                    scratchAt.Add((s, binding));
                }

                for (int o = 0; o < stepOutputs.Length; o++)
                {
                    int id = stepOutputs[o];
                    if (id >= 0)
                    {
                        born[id] = o == 0 && binding.InitializesOutput ? begin : begin + s;
                        lastUse[id] = Math.Max(lastUse.GetValueOrDefault(id, -1), begin + s);
                    }
                }
            }

            foreach (int id in region.Inputs)
            {
                DeviceElement e = ElementOf(plan.Types[id]);
                keyOf[id] = live.Count;
                live.Add(new LiveValue(live.Count, e, plan.Shapes[id].Length, begin, lastUse[id]));
                toBackend += (long)plan.Shapes[id].Length * DeviceArena.ElementBytes(e);
            }

            foreach ((int id, int start) in born)
            {
                DeviceElement e = DeviceElementOfOutput(region, bindings[r], id);
                // A value that leaves the region is read back after the last step.
                int stop = Array.IndexOf(region.Outputs, id) >= 0 ? end : lastUse[id];
                keyOf[id] = live.Count;
                live.Add(new LiveValue(live.Count, e, plan.Shapes[id].Length, start, stop));
            }

            foreach ((int step, GpuBinding binding) in scratchAt)
            {
                int from = binding.InitializesOutput ? begin : begin + step;
                if (binding.ScratchF32Elements > 0)
                {
                    scratchKeys.Add((binding, live.Count, DeviceElement.F32, false));
                    live.Add(new LiveValue(live.Count, DeviceElement.F32, binding.ScratchF32Elements, from, begin + step));
                }

                if (binding.ScratchF32bElements > 0)
                {
                    scratchKeys.Add((binding, live.Count, DeviceElement.F32, true));
                    live.Add(new LiveValue(live.Count, DeviceElement.F32, binding.ScratchF32bElements, from, begin + step));
                }

                if (binding.ScratchF16Elements > 0)
                {
                    scratchKeys.Add((binding, live.Count, DeviceElement.F16, false));
                    live.Add(new LiveValue(live.Count, DeviceElement.F16, binding.ScratchF16Elements, from, begin + step));
                }

                if (binding.ScratchI32Elements > 0)
                {
                    scratchKeys.Add((binding, live.Count, DeviceElement.I32, false));
                    live.Add(new LiveValue(live.Count, DeviceElement.I32, binding.ScratchI32Elements, from, begin + step));
                }

                if (binding.ScratchI32bElements > 0)
                {
                    scratchKeys.Add((binding, live.Count, DeviceElement.I32, true));
                    live.Add(new LiveValue(live.Count, DeviceElement.I32, binding.ScratchI32bElements, from, begin + step));
                }

                if (binding.ScratchBoolElements > 0)
                {
                    scratchKeys.Add((binding, live.Count, DeviceElement.Bool, false));
                    live.Add(new LiveValue(live.Count, DeviceElement.Bool, binding.ScratchBoolElements, from, begin + step));
                }

                if (binding.ScratchI8Elements > 0)
                {
                    scratchKeys.Add((binding, live.Count, DeviceElement.I8, false));
                    live.Add(new LiveValue(live.Count, DeviceElement.I8, binding.ScratchI8Elements, from, begin + step));
                }
            }

            foreach (int id in region.Outputs)
            {
                fromBackend += (long)plan.Shapes[id].Length * DeviceArena.ElementBytes(DeviceElementOfOutput(region, bindings[r], id));
            }

            clock = end + 1;
        }

        // The planner reorders its list; slots come back by key.
        DeviceArena.Layout layout = backend.Arena.Plan([.. live], backend.SeparateReadWriteSlabs ? ReadWriteConflicts(regions, bindings, keys, aliases, scratchKeys) : null);
        backend.Arena.Ensure(layout);
        var slots = new Dictionary<int, ValueSlot>[keys.Length];
        for (int r = 0; r < slots.Length; r++)
        {
            slots[r] = [];
            foreach ((int id, int key) in keys[r])
            {
                slots[r][id] = layout.Slots[key];
            }

            foreach ((int view, int root) in aliases[r])
            {
                slots[r][view] = slots[r][root];
            }
        }

        foreach ((GpuBinding binding, int key, DeviceElement element, bool second) in scratchKeys)
        {
            switch (element)
            {
                case DeviceElement.F32 when second:
                    binding.ScratchF32bSlot = layout.Slots[key];
                    break;
                case DeviceElement.F32:
                    binding.ScratchSlot = layout.Slots[key];
                    break;
                case DeviceElement.I32 when second:
                    binding.ScratchI32bSlot = layout.Slots[key];
                    break;
                case DeviceElement.F16:
                    binding.ScratchF16Slot = layout.Slots[key];
                    break;
                case DeviceElement.I32:
                    binding.ScratchI32Slot = layout.Slots[key];
                    break;
                case DeviceElement.I8:
                    binding.ScratchI8Slot = layout.Slots[key];
                    break;
                default:
                    binding.ScratchBoolSlot = layout.Slots[key];
                    break;
            }
        }

        var owned = new List<DeviceConstant>(claims.Owned);
        owned.AddRange(fusedConstants);
        claims.Owned.Clear();
        return new GpuProgram(backend, all, bindings, layout, slots, owned, toBackend, fromBackend, flops);
    }

    // Pairs of arena values that one dispatch touches: its inputs, its outputs and its scratch. A browser refuses one buffer bound both for
    // reading and for writing in a dispatch, so no two of these may share a slab.
    private static List<(int A, int B)> ReadWriteConflicts(IReadOnlyList<BackendRegion> regions, GpuBinding[][] bindings, Dictionary<int, int>[] keys,
        Dictionary<int, int>[] aliases, List<(GpuBinding Binding, int Key, DeviceElement Element, bool Second)> scratchKeys)
    {
        var pairs = new List<(int, int)>();
        var touched = new List<int>();
        for (int r = 0; r < regions.Count; r++)
        {
            for (int s = 0; s < regions[r].Steps.Length; s++)
            {
                GpuBinding binding = bindings[r][s];
                if (binding.AliasOfInput >= 0)
                {
                    continue;
                }

                PlanStep step = regions[r].Steps[s];
                var reads = new List<int>();
                foreach (int id in binding.InputIds ?? step.Node.Inputs)
                {
                    int root = id >= 0 && aliases[r].TryGetValue(id, out int a) ? a : id;
                    if (root >= 0 && keys[r].TryGetValue(root, out int k))
                    {
                        reads.Add(k);
                    }
                }

                touched.Clear();
                foreach (int id in binding.OutputIds ?? step.Node.Outputs)
                {
                    if (id >= 0 && keys[r].TryGetValue(id, out int k))
                    {
                        touched.Add(k);
                    }
                }

                foreach ((GpuBinding owner, int key, _, _) in scratchKeys)
                {
                    if (ReferenceEquals(owner, binding))
                    {
                        touched.Add(key);
                    }
                }

                foreach (int w in touched)
                {
                    foreach (int rd in reads)
                    {
                        pairs.Add((rd, w));
                    }

                    foreach (int other in touched)
                    {
                        if (other != w)
                        {
                            pairs.Add((w, other));
                        }
                    }
                }
            }
        }

        return pairs;
    }

    /// <summary>
    /// Folds a bias addition and an activation into the matrix product that makes their input, when nothing else reads the intermediate
    /// results: the product's dispatch then does all three. A step folded away is replaced by <see cref="AbsorbedBinding"/>. Float32
    /// products take it in both modes (the kernel's epilogue is the elementwise kernels' own add and activation, bit for bit, and has no order
    /// to fix); float16 ones in fast mode only, because there the fused result skips the rounding of the intermediate to half precision.
    /// </summary>
    private static void FoldEpilogues(GpuBackend backend, ExecutionPlan plan, BackendRegion region, GpuBinding[] bindings, List<DeviceConstant> made)
    {
        // How many steps of the region read each value; a value the region returns or a later region reads is also used.
        var uses = new Dictionary<int, int>();
        var user = new Dictionary<int, int>();
        for (int s = 0; s < region.Steps.Length; s++)
        {
            foreach (int id in region.Steps[s].Node.Inputs)
            {
                if (id >= 0)
                {
                    uses[id] = uses.GetValueOrDefault(id) + 1;
                    user[id] = s;
                }
            }
        }

        bool SoleReader(int id, out int step) => (step = user.GetValueOrDefault(id, -1)) >= 0 && uses[id] == 1 && Array.IndexOf(region.Outputs, id) < 0;

        for (int s = 0; s < bindings.Length; s++)
        {
            if (bindings[s] is not GemmBinding gemm || !gemm.CanTakeEpilogue || region.Steps[s].Node.Outputs[0] < 0)
            {
                continue;
            }

            if (gemm.Element == DeviceElement.F16 && backend.KernelMode == GpuKernelMode.Deterministic)
            {
                continue;
            }

            int result = region.Steps[s].Node.Outputs[0];
            GpuBuffer<float>? biasBuffer = null;
            GemmActivation activation = GemmActivation.None;
            var absorbed = new List<int>();
            if (SoleReader(result, out int t) && bindings[t] is IBiasAdd add && add.TryBias(out int valueId, out DeviceConstant bias, out int biasId, out int addResult, out int length)
                && valueId == result && length == gemm.Columns && addResult >= 0 && plan.Shapes[addResult].Length == plan.Shapes[result].Length)
            {
                GpuBuffer<float>? f32 = bias.F32;
                if (f32 is null && bias.F16 is not null)
                {
                    DeviceConstant converted = ConvertToF32(backend, plan.Consts[biasId]);
                    made.Add(converted);
                    f32 = converted.F32;
                }

                if (f32 is not null)
                {
                    biasBuffer = f32;
                    absorbed.Add(t);
                    result = addResult;
                }
            }

            if (SoleReader(result, out int u) && bindings[u] is IProductActivation act && act.TryActivation(out GemmActivation a, out int actIn, out int actOut)
                && actIn == result && actOut >= 0 && plan.Shapes[actOut].Length == plan.Shapes[result].Length && u > s)
            {
                activation = a;
                absorbed.Add(u);
                result = actOut;
            }

            if (absorbed.Count == 0)
            {
                continue;
            }

            gemm.TakeEpilogue(biasBuffer, activation, result, region.Steps[s].Node.Inputs);
            foreach (int i in absorbed)
            {
                bindings[i] = new AbsorbedBinding();
            }
        }
    }

    // A half-precision bias as a float32 constant, which is what the product's epilogue adds.
    private static DeviceConstant ConvertToF32(GpuBackend backend, in Tensor half)
    {
        Tensor src = half.IsContiguous ? half : half.Contiguous();
        try
        {
            using Tensor wide = Tensor.Create(DType.F32, src.Shape);
            Span<float> dst = wide.MutableData<float>();
            ReadOnlySpan<Half> from = src.Data<Half>();
            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] = (float)from[i];
            }

            return backend.Constants.Upload(wide);
        }
        finally
        {
            if (!ReferenceEquals(src.Buffer, half.Buffer))
            {
                src.Dispose();
            }
        }
    }

    private static DeviceElement ElementOf(DType t) => t switch
    {
        DType.F16 => DeviceElement.F16,
        DType.I64 => DeviceElement.I64,
        DType.I32 => DeviceElement.I32,
        DType.U8 => DeviceElement.U8,
        DType.I8 => DeviceElement.I8,
        DType.Bool => DeviceElement.Bool,
        _ => DeviceElement.F32,
    };

    private static DeviceElement DeviceElementOfOutput(BackendRegion region, GpuBinding[] bindings, int id)
    {
        for (int s = 0; s < region.Steps.Length; s++)
        {
            int[] outs = bindings[s].OutputIds ?? region.Steps[s].Node.Outputs;
            for (int o = 0; o < outs.Length; o++)
            {
                if (outs[o] == id)
                {
                    return bindings[s].OutputElement(o);
                }
            }
        }

        throw new InvalidOperationException("A region output has no producer in the region.");
    }

    // ---- running ----------------------------------------------------------------------------------------------------------

    public void Run(int region, Tensor[] values)
    {
        BackendRegion r = _regions[region];
        try
        {
            lock (_backend.Gate)
            {
                RunLocked(r, _bindings[region], values);
            }
        }
        catch (Exception e) when (e is GpuException or GpuKernelShapeException)
        {
            throw GpuErrors.Wrap($"running {r.Steps.Length} operator(s) starting at {r.Steps[0].Node.Info.Describe()}", e);
        }
    }

    public async ValueTask RunAsync(int region, Tensor[] values)
    {
        BackendRegion r = _regions[region];
        // Async runs of a session take turns; the gate cannot be a lock because it is held across awaits.
        await _backend.AsyncGate.WaitAsync().ConfigureAwait(true);
        try
        {
            RunTimes times;
            lock (_backend.Gate)
            {
                times = Record(r, _bindings[region], values);
            }

            await DownloadAllAsync(r, values).ConfigureAwait(true);
            if (times.StatusUsed)
            {
                try
                {
                    await _backend.IndexStatus!.ThrowIfFailedAsync().ConfigureAwait(true);
                }
                catch (GpuKernelIndexException e)
                {
                    throw IndexError(e);
                }
            }

            lock (_backend.Gate)
            {
                Finish(r, times);
            }
        }
        catch (Exception e) when (e is GpuException or GpuKernelShapeException)
        {
            throw GpuErrors.Wrap($"running {r.Steps.Length} operator(s) starting at {r.Steps[0].Node.Info.Describe()}", e);
        }
        finally
        {
            _backend.AsyncGate.Release();
        }
    }

    // Starts every output's read before waiting for any, so a browser pays one round trip rather than one per output.
    private async ValueTask DownloadAllAsync(BackendRegion region, Tensor[] values)
    {
        int n = region.Outputs.Length;
        var pending = new (System.Threading.Tasks.Task Read, byte[] Bytes, int Length, int Id)[n];
        for (int k = 0; k < n; k++)
        {
            int id = region.Outputs[k];
            Tensor host = values[id];
            (GpuBuffer buffer, int offsetElements, int elementBytes) = DeviceBufferOf(id, host.DType);
            int bytes = checked((int)host.Shape.Length * elementBytes);
            byte[] scratch = System.Buffers.ArrayPool<byte>.Shared.Rent(Math.Max(bytes, 4));
            pending[k] = (_backend.Device.ReadBufferAsync(buffer, (ulong)offsetElements * (ulong)elementBytes, scratch.AsMemory(0, bytes)).AsTask(), scratch, bytes, id);
        }

        try
        {
            for (int k = 0; k < n; k++)
            {
                await pending[k].Read.ConfigureAwait(true);
                Tensor host = values[pending[k].Id];
                pending[k].Bytes.AsSpan(0, pending[k].Length).CopyTo(System.Runtime.InteropServices.MemoryMarshal.AsBytes(HostBytes(host)));
            }
        }
        finally
        {
            for (int k = 0; k < n; k++)
            {
                // A read still in flight owns its scratch; only finished ones go back to the pool.
                if (pending[k].Read.IsCompleted)
                {
                    System.Buffers.ArrayPool<byte>.Shared.Return(pending[k].Bytes);
                }
            }
        }
    }

    private static Span<byte> HostBytes(in Tensor host) => host.DType switch
    {
        DType.F32 => System.Runtime.InteropServices.MemoryMarshal.AsBytes(host.MutableData<float>()),
        DType.F16 => System.Runtime.InteropServices.MemoryMarshal.AsBytes(host.MutableData<Half>()),
        DType.I64 => System.Runtime.InteropServices.MemoryMarshal.AsBytes(host.MutableData<long>()),
        DType.I32 => System.Runtime.InteropServices.MemoryMarshal.AsBytes(host.MutableData<int>()),
        DType.I8 => System.Runtime.InteropServices.MemoryMarshal.AsBytes(host.MutableData<sbyte>()),
        DType.Bool => System.Runtime.InteropServices.MemoryMarshal.AsBytes(host.MutableData<bool>()),
        _ => host.MutableData<byte>(),
    };

    // The device buffer behind a value, its offset in elements and its element size, mirroring Download.
    private (GpuBuffer Buffer, int Offset, int ElementBytes) DeviceBufferOf(int id, DType type)
    {
        switch (type)
        {
            case DType.F32: { GpuBuffer<float> b = _context.F32(id, out int o); return (b.Buffer, o, 4); }
            case DType.F16: { GpuBuffer<Half> b = _context.F16(id, out int o); return (b.Buffer, o, 2); }
            case DType.I64: { GpuBuffer<long> b = _context.Buffer<long>(id, out int o); return (b.Buffer, o, 8); }
            case DType.I32: { GpuBuffer<int> b = _context.Buffer<int>(id, out int o); return (b.Buffer, o, 4); }
            case DType.I8: { GpuBuffer<sbyte> b = _context.Buffer<sbyte>(id, out int o); return (b.Buffer, o, 1); }
            default: { GpuBuffer<byte> b = _context.Buffer<byte>(id, out int o); return (b.Buffer, o, 1); }
        }
    }

    private void RunLocked(BackendRegion region, GpuBinding[] bindings, Tensor[] values)
    {
        RunTimes times = Record(region, bindings, values);
        foreach (int id in region.Outputs)
        {
            Download(region.Index, id, values[id]);
        }

        if (times.StatusUsed)
        {
            try
            {
                _backend.IndexStatus!.ThrowIfFailed();
            }
            catch (GpuKernelIndexException e)
            {
                throw IndexError(e);
            }
        }

        Finish(region, times);
    }

    private static SessionException IndexError(GpuKernelIndexException e) =>
        new(SessionErrorKind.InvalidInput, $"{e.Operation} index at position {e.Position} of the indices is out of range for its axis.", null, default, e);

    private readonly record struct RunTimes(bool Timed, bool Stamps, bool StatusUsed, uint TimedPasses, long T0, long T1,
        uint[]? StepFirst = null, uint[]? StepCount = null, bool Span = false, long SubmitTicks = 0, long SubmittedAt = 0, long Dispatches = 0, long Passes = 0);

    // Uploads the region's inputs, records its steps and hands them to the device. The outputs are still on the device when this returns.
    private RunTimes Record(BackendRegion region, GpuBinding[] bindings, Tensor[] values)
    {
        bool timed = _backend.TimingsEnabled;
        long t0 = timed ? Stopwatch.GetTimestamp() : 0;
        _context.Use(_slots[region.Index]);
        foreach (int id in region.Inputs)
        {
            Upload(region.Index, id, values[id]);
        }

        foreach (GpuBinding b in bindings)
        {
            b.Upload(_context, values);
        }

        long t1 = timed ? Stopwatch.GetTimestamp() : 0;
        GpuDevice device = _backend.Device;
        GpuCommandEncoder encoder = device.BeginCommands();
        bool stamps = timed && _backend.Timestamps is not null;

        // Profiling: each step gets its own command buffer and timing range, so its dispatches' GPU time can be told apart.
        bool profile = stamps && _backend.ProfileSteps;

        // Metal samples its counters only at the boundaries of a compute encoder, so timing a pass splits the encoder around it and costs
        // about 20 microseconds. There the run is timed as one span instead: only the first operators with a pass and the last two are bracketed, and the
        // time is from the start of the first timed pass to the end of the last one. Every other pass shares the encoder untimed.
        bool span = stamps && !profile && device.Backend == RecursiveAI.OneGpu.GpuBackend.Metal;
        uint[]? stepFirst = profile ? new uint[bindings.Length] : null, stepCount = profile ? new uint[bindings.Length] : null;

        // Where the platform names a work budget the kernels cut their own submissions; elsewhere the region is cut here.
        bool cut = device.Capabilities.Compute.SubmissionWorkBudget == 0;
        if (stamps && !profile && !span)
        {
            encoder.BeginPassTiming(new PassTiming(_backend.Timestamps!, 0, GpuBackend.MaxTimedPasses));
        }

        bool statusUsed = false;
        foreach (GpuBinding b in bindings)
        {
            statusUsed |= b.UsesIndexStatus;
        }

        if (statusUsed)
        {
            _backend.IndexStatus!.Reset();
        }

        _context.Encoder = encoder;
        uint timedPasses = 0;
        long submitTicks = 0, dispatchCount = 0, passCount = 0;
        bool timing = stamps && !profile && !span;
        double sinceSubmit = 0;
        bool firstOpen = true;
        int pipeline = _backend.PipelineFirstDispatches;
        uint chunk = (uint)Math.Max(pipeline, 1);
        for (int i = 0; i < bindings.Length; i++)
        {
            bool spanEdge = span && ((firstOpen && i < 8) || i >= bindings.Length - 2);
            if (spanEdge)
            {
                encoder.BeginPassTiming(new PassTiming(_backend.Timestamps!, 2 * timedPasses, GpuBackend.MaxTimedPasses - timedPasses));
            }

            if (profile)
            {
                if (i > 0)
                {
                    dispatchCount += encoder.DispatchCount;
                    passCount += encoder.ComputePassCount;
                    long p0 = timed ? Stopwatch.GetTimestamp() : 0;
                    device.Submit(encoder);
                    submitTicks += timed ? Stopwatch.GetTimestamp() - p0 : 0;
                    encoder = device.BeginCommands();
                    _context.Encoder = encoder;
                }

                encoder.BeginPassTiming(new PassTiming(_backend.Timestamps!, 2 * timedPasses, GpuBackend.MaxTimedPasses - timedPasses));
            }

            bindings[i].Encode(_context);
            if (spanEdge)
            {
                uint edge = encoder.EndPassTiming();
                timedPasses += edge;
                firstOpen &= edge == 0;
            }

            if (profile)
            {
                uint n = encoder.EndPassTiming();
                stepFirst![i] = timedPasses;
                stepCount![i] = n;
                timedPasses += n;
            }

            sinceSubmit += bindings[i].Flops;
            // Pipelining: hand the first few dispatches to the device early and every following piece twice as large, so the device starts
            // while the host is still recording and the host's cost hides behind the kernels instead of adding to them.
            bool early = pipeline > 0 && !profile && encoder.DispatchCount >= chunk && bindings.Length - i > 4;
            if (!profile && i + 1 < bindings.Length && ((cut && sinceSubmit >= _backend.SplitWorkBudget) || early))
            {
                // Hand the work so far to the device and keep recording in a new command buffer. Buffers carry their own synchronization
                // state, so a later pass still gets the barrier for what an earlier submission wrote.
                if (timing)
                {
                    timedPasses += encoder.EndPassTiming();
                }

                dispatchCount += encoder.DispatchCount;
                passCount += encoder.ComputePassCount;
                long c0 = timed ? Stopwatch.GetTimestamp() : 0;
                device.Submit(encoder);
                submitTicks += timed ? Stopwatch.GetTimestamp() - c0 : 0;
                encoder = device.BeginCommands();
                _context.Encoder = encoder;
                timing = stamps && timedPasses < GpuBackend.MaxTimedPasses;
                if (timing)
                {
                    encoder.BeginPassTiming(new PassTiming(_backend.Timestamps!, 2 * timedPasses, GpuBackend.MaxTimedPasses - timedPasses));
                }

                sinceSubmit = 0;
                chunk = Math.Min(chunk * 2, 256);
            }
        }

        timedPasses += timing ? encoder.EndPassTiming() : 0;
        dispatchCount += encoder.DispatchCount;
        passCount += encoder.ComputePassCount;
        long s0 = timed ? Stopwatch.GetTimestamp() : 0;
        device.Submit(encoder);
        long s1 = timed ? Stopwatch.GetTimestamp() : 0;
        return new RunTimes(timed, stamps, statusUsed, timedPasses, t0, t1, stepFirst, stepCount, span, s1 - s0 + submitTicks, s1, dispatchCount, passCount);
    }

    // Checks what the device reported, and with timings on, adds the run's times.
    private void Finish(BackendRegion region, RunTimes times)
    {
        bool timed = times.Timed, stamps = times.Stamps;
        uint timedPasses = times.TimedPasses;
        long t0 = times.T0, t1 = times.T1;
        uint[]? stepFirst = times.StepFirst, stepCount = times.StepCount;
        bool profile = stepFirst is not null;
        GpuDevice device = _backend.Device;
        // A command buffer the driver aborted leaves the outputs undefined, and the backends report it as a message rather than an
        // exception, so a run that produced one must not return its (possibly wrong) results.
        string[] reported = device.TakeValidationMessages();
        if (reported.Length > 0)
        {
            throw new SessionException(SessionErrorKind.DeviceFailure,
                $"The device reported {reported.Length} problem(s) while running {region.Steps.Length} operator(s) starting at {region.Steps[0].Node.Info.Describe()}, so the results were discarded: {string.Join(" | ", reported)}");
        }

        if (timed)
        {
            long t2 = Stopwatch.GetTimestamp();
            double gpuMs = 0;
            if (stamps && timedPasses > 0)
            {
                int n = (int)timedPasses * 2;
                if (_stamps.Length < n)
                {
                    _stamps = new ulong[n];
                }

                if (device.TryReadTimestamps(_backend.Timestamps!, 0, _stamps.AsSpan(0, n)))
                {
                    if (times.Span)
                    {
                        gpuMs = (_stamps[n - 1] - _stamps[0]) / 1e6;
                    }
                    else
                    {
                        for (int i = 0; i < n; i += 2)
                        {
                            gpuMs += (_stamps[i + 1] - _stamps[i]) / 1e6;
                        }
                    }

                    if (profile)
                    {
                        for (int i = 0; i < region.Steps.Length; i++)
                        {
                            double ms = 0;
                            for (uint k = stepFirst![i]; k < stepFirst[i] + stepCount![i]; k++)
                            {
                                ms += (_stamps[2 * k + 1] - _stamps[2 * k]) / 1e6;
                            }

                            _backend.AddProfile(region.Steps[i].Node.Info.OpType, ms, (int)stepCount[i]);
                        }
                    }
                }
            }

            _backend.AddTimings(Ms(t1 - t0), gpuMs, Ms(t2 - t1), Ms(t2 - t0), times.Dispatches, Ms(times.SubmittedAt - t1 - times.SubmitTicks), Ms(times.SubmitTicks), times.Passes);
        }
    }

    private static double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

    private void Upload(int region, int id, in Tensor host)
    {
        ValueSlot slot = _slots[region][id];
        Tensor src = host.IsContiguous ? host : host.Contiguous();
        try
        {
            switch (slot.Element)
            {
                case DeviceElement.F32:
                    _context.F32(id, out int o32).Write(src.Data<float>(), o32);
                    break;
                case DeviceElement.F16:
                    _context.F16(id, out int o16).Write(src.Data<Half>(), o16);
                    break;
                case DeviceElement.I64:
                    _context.Buffer<long>(id, out int o64).Write(src.Data<long>(), o64);
                    break;
                case DeviceElement.I32:
                    _context.Buffer<int>(id, out int oi).Write(src.Data<int>(), oi);
                    break;
                case DeviceElement.I8:
                    _context.Buffer<sbyte>(id, out int o8).Write(src.Data<sbyte>(), o8);
                    break;
                case DeviceElement.Bool:
                    _context.Buffer<byte>(id, out int obl).Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(src.Data<bool>()), obl);
                    break;
                default:
                    _context.Buffer<byte>(id, out int ob).Write(src.Data<byte>(), ob);
                    break;
            }
        }
        finally
        {
            if (!ReferenceEquals(src.Buffer, host.Buffer))
            {
                src.Dispose();
            }
        }
    }

    private void Download(int region, int id, in Tensor host)
    {
        switch (host.DType)
        {
            case DType.F32:
                _context.F32(id, out int o32).Read(host.MutableData<float>(), o32);
                break;
            case DType.F16:
                _context.F16(id, out int o16).Read(host.MutableData<Half>(), o16);
                break;
            case DType.I64:
                _context.Buffer<long>(id, out int o64).Read(host.MutableData<long>(), o64);
                break;
            case DType.I32:
                _context.Buffer<int>(id, out int oi).Read(host.MutableData<int>(), oi);
                break;
            case DType.I8:
                _context.Buffer<sbyte>(id, out int o8).Read(host.MutableData<sbyte>(), o8);
                break;
            case DType.Bool:
                _context.Buffer<byte>(id, out int obl).Read(System.Runtime.InteropServices.MemoryMarshal.AsBytes(host.MutableData<bool>()), obl);
                break;
            default:
                _context.Buffer<byte>(id, out int ob).Read(host.MutableData<byte>(), ob);
                break;
        }
    }

    public void Dispose()
    {
        lock (_backend.Gate)
        {
            foreach (DeviceConstant c in _owned)
            {
                c.Dispose();
            }

            _owned.Clear();
        }
    }
}
