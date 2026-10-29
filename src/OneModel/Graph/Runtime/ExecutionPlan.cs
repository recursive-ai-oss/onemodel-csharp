using System;
using System.Collections.Generic;
using System.Threading;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Runtime;

/// <summary>What a session fixes once: the mode, the instruction set, the thread budget and where memory comes from.</summary>
internal sealed record SessionEnv(ExecutionMode Mode, KernelPath Path, int Threads, BufferPool Pool);

internal enum ValueKind : byte
{
    /// <summary>Not defined yet (a node reads it before any producer): a malformed graph.</summary>
    Unset,

    /// <summary>Known before the run: weights, folded constants, constants a kernel declared.</summary>
    Const,

    /// <summary>Supplied by the caller for each run.</summary>
    Input,

    /// <summary>Produced by a node; type, shape and slab location are fixed by the plan.</summary>
    Static,

    /// <summary>Produced by a node whose shape is only known at run time; lives in the run's arena.</summary>
    Dynamic,
}

/// <summary>One scheduled node.</summary>
internal sealed class PlanStep
{
    public required CompiledNode Node { get; init; }

    /// <summary>True when the node is shaped and its outputs allocated at run time.</summary>
    public required bool Dynamic { get; init; }

    public bool IsView => Node.Kernel.ViewOfInput >= 0;

    /// <summary>For a static step, the type of each output slot (also slots no value reads).</summary>
    public DType[] OutTypes { get; set; } = [];

    public Shape[] OutShapes { get; set; } = [];

    /// <summary>What the kernel's <c>Prepare</c> made from constant inputs for this plan.</summary>
    public object? State { get; set; }

    /// <summary>Slab ranges that become free after this step (used to poison them in checked builds).</summary>
    public (long Offset, long Size)[] Freed { get; set; } = [];
}

/// <summary>
/// Everything about running a model for one set of input element types and shapes: which values are constants, the schedule, and
/// where every intermediate lives in one slab. Immutable after <see cref="Build"/>, shared by any number of concurrent runs. Each run
/// takes a <see cref="RunState"/> (its own slab, arena and value table) from the plan's idle list.
/// </summary>
internal sealed class ExecutionPlan : IDisposable
{
    private const long Align = 64;

    // A node whose inputs are all constants is evaluated when the plan is made, unless that would create a tensor much bigger than
    // what it read (Tile, ConstantOfShape) or bigger than this.
    private const long FoldAlwaysBytes = 1 << 16;

    // States kept for reuse; a burst of more concurrent runs creates and frees extras.
    private const int MaxIdleStates = 8;

    private readonly List<Tensor> _owned = [];
    private readonly Stack<RunState> _idle = new();
    private int _refs = 1;

    private ExecutionPlan(CompiledGraph graph, SessionEnv env, DType[] inTypes, Shape[] inShapes)
    {
        graph.AddRef();
        Graph = graph;
        Env = env;
        InTypes = inTypes;
        InShapes = inShapes;
        int n = graph.ValueCount;
        Kinds = new ValueKind[n];
        Types = new DType[n];
        Shapes = new Shape[n];
        Offsets = new long[n];
        Consts = new Tensor[n];
    }

    public CompiledGraph Graph { get; }

    public SessionEnv Env { get; }

    public DType[] InTypes { get; }

    public Shape[] InShapes { get; }

    public ValueKind[] Kinds { get; }

    public DType[] Types { get; }

    public Shape[] Shapes { get; }

    /// <summary>Slab byte offset of each static value (undefined for the others, and for views).</summary>
    public long[] Offsets { get; }

    public Tensor[] Consts { get; }

    public PlanStep[] Steps { get; private set; } = [];

    /// <summary>Slab bytes for the planned values plus one staging area per input.</summary>
    public long SlabBytes { get; private set; }

    /// <summary>Slab byte offset of each input's staging area (where a span input is copied to).</summary>
    public long[] StagingOffsets { get; private set; } = [];

    public int MaxNodeInputs { get; private set; }

    public int MaxNodeOutputs { get; private set; }

    public bool HasDynamic { get; private set; }

    /// <summary>Slab bytes used by planned values alone, for tests and diagnostics.</summary>
    public long PlannedBytes { get; private set; }

    /// <summary>Bytes the same values would need if none shared storage.</summary>
    public long UnsharedBytes { get; private set; }

    public bool Matches(ReadOnlySpan<DType> types, ReadOnlySpan<Shape> shapes)
    {
        if (types.Length != InTypes.Length)
        {
            return false;
        }

        for (int i = 0; i < InTypes.Length; i++)
        {
            if (types[i] != InTypes[i] || !shapes[i].Equals(InShapes[i]))
            {
                return false;
            }
        }

        return true;
    }

    // ---- reference counting: the cache holds one reference, every run another ----------------------------------------------

    public bool TryAcquire()
    {
        while (true)
        {
            int r = Volatile.Read(ref _refs);
            if (r <= 0)
            {
                return false;
            }

            if (Interlocked.CompareExchange(ref _refs, r + 1, r) == r)
            {
                return true;
            }
        }
    }

    public void Release()
    {
        if (Interlocked.Decrement(ref _refs) == 0)
        {
            Dispose();
        }
    }

    public RunState RentState()
    {
        lock (_idle)
        {
            if (_idle.Count > 0)
            {
                return _idle.Pop();
            }
        }

        return new RunState(this);
    }

    public void ReturnState(RunState s)
    {
        lock (_idle)
        {
            if (_idle.Count < MaxIdleStates && Volatile.Read(ref _refs) > 0)
            {
                _idle.Push(s);
                return;
            }
        }

        s.Dispose();
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        lock (_idle)
        {
            while (_idle.Count > 0)
            {
                _idle.Pop().Dispose();
            }
        }

        foreach (Tensor t in _owned)
        {
            t.Dispose();
        }

        _owned.Clear();
        Graph.Dispose();
    }

    private bool _disposed;

    // ---- building ---------------------------------------------------------------------------------------------------------

    /// <summary>Shapes every node for the given input types and shapes, folds constants, and assigns slab space.</summary>
    /// <exception cref="SessionException">A node cannot be shaped for these inputs.</exception>
    public static ExecutionPlan Build(CompiledGraph graph, SessionEnv env, ReadOnlySpan<DType> inTypes, ReadOnlySpan<Shape> inShapes, CancellationToken ct)
    {
        var p = new ExecutionPlan(graph, env, inTypes.ToArray(), inShapes.ToArray());
        try
        {
            p.Shape(ct);
            p.Schedule();
            return p;
        }
        catch
        {
            p.Dispose();
            throw;
        }
    }

    private void Shape(CancellationToken ct)
    {
        CompiledGraph g = Graph;
        for (int v = 0; v < g.ValueCount; v++)
        {
            if (!g.Constants[v].IsEmpty)
            {
                Kinds[v] = ValueKind.Const;
                Consts[v] = g.Constants[v];
                Types[v] = Consts[v].DType;
                Shapes[v] = Consts[v].Shape;
            }
        }

        for (int i = 0; i < g.InputIds.Length; i++)
        {
            int id = g.InputIds[i];
            Kinds[id] = ValueKind.Input;
            Types[id] = InTypes[i];
            Shapes[id] = InShapes[i];
        }

        var steps = new List<PlanStep>();
        var ictx = new InferContext();
        using var scratch = new Arena(Env.Pool);
        foreach (CompiledNode node in g.Nodes)
        {
            ct.ThrowIfCancellationRequested();
            MaxNodeInputs = Math.Max(MaxNodeInputs, node.Inputs.Length);
            MaxNodeOutputs = Math.Max(MaxNodeOutputs, node.Outputs.Length);
            ShapeNode(node, ictx, scratch, steps);
        }

        foreach (int id in g.OutputIds)
        {
            if (Kinds[id] == ValueKind.Unset)
            {
                throw new SessionException(SessionErrorKind.InvalidModel, $"Graph output '{g.ValueNames[id]}' is never produced.");
            }
        }

        Steps = [.. steps];
    }

    private void ShapeNode(CompiledNode node, InferContext ictx, Arena scratch, List<PlanStep> steps)
    {
        OpKernel k = node.Kernel;
        bool anyDynamic = false;
        foreach (int id in node.Inputs)
        {
            if (id < 0)
            {
                continue;
            }

            if (Kinds[id] == ValueKind.Unset)
            {
                throw new SessionException(SessionErrorKind.InvalidModel,
                    $"{node.Info.Describe()}: input '{Graph.ValueNames[id]}' has no producer, initializer or graph input.", node.Info.Name);
            }

            anyDynamic |= Kinds[id] == ValueKind.Dynamic;
        }

        if (!anyDynamic && !k.OutputsDataDependent)
        {
            ictx.Begin(node.Info, node.Inputs.Length, node.Outputs.Length);
            for (int i = 0; i < node.Inputs.Length; i++)
            {
                int id = node.Inputs[i];
                if (id >= 0)
                {
                    ictx.SetInput(i, Types[id], Shapes[id], Kinds[id] == ValueKind.Const ? Consts[id] : default);
                }
            }

            try
            {
                k.Infer(ictx);
            }
            catch (Exception e) when (e is ArgumentException or InvalidOperationException)
            {
                throw new SessionException(SessionErrorKind.InvalidModel, $"{node.Info.Describe()}: {e.Message}", node.Info.Name, default, e);
            }

            if (!ictx.NeedsRunTimeValues)
            {
                PlaceStatic(node, ictx, scratch, steps);
                return;
            }
        }

        // Shaped at run time.
        HasDynamic = true;
        foreach (int id in node.Outputs)
        {
            if (id >= 0)
            {
                Kinds[id] = ValueKind.Dynamic;
            }
        }

        steps.Add(new PlanStep { Node = node, Dynamic = true });
    }

    private void PlaceStatic(CompiledNode node, InferContext ictx, Arena scratch, List<PlanStep> steps)
    {
        OpKernel k = node.Kernel;
        int nOut = node.Outputs.Length;
        bool anyConst = false;
        for (int o = 0; o < nOut; o++)
        {
            if (node.Outputs[o] >= 0 && !ictx.OutputIsSet(o))
            {
                throw new SessionException(SessionErrorKind.InvalidModel,
                    $"{node.Info.Describe()}: shape inference did not state output {o} ('{Graph.ValueNames[node.Outputs[o]]}').", node.Info.Name);
            }

            anyConst |= ictx.OutputIsSet(o) && !ictx.OutputConstant(o).IsEmpty;
        }

        if (anyConst)
        {
            for (int o = 0; o < nOut; o++)
            {
                int id = node.Outputs[o];
                Tensor c = ictx.OutputConstant(o);
                if (id < 0)
                {
                    if (!c.IsEmpty && ictx.OutputConstantOwned(o))
                    {
                        c.Dispose();
                    }

                    continue;
                }

                if (c.IsEmpty)
                {
                    throw new SessionException(SessionErrorKind.InvalidModel,
                        $"{node.Info.Describe()}: some outputs are constant and some are not, which the operator contract does not allow.", node.Info.Name);
                }

                SetConst(id, c, ictx.OutputConstantOwned(o));
            }

            return;
        }

        // A view of a constant is a constant.
        if (k.ViewOfInput >= 0 && node.Inputs[k.ViewOfInput] is int vin and >= 0 && Kinds[vin] == ValueKind.Const)
        {
            SetConst(node.Outputs[0], Consts[vin].Reshape(ictx.OutputShape(0)), false);
            return;
        }

        // Evaluate it now if every input is constant and the result is not a blow-up.
        if (k.ViewOfInput < 0 && TryFold(node, ictx, scratch))
        {
            return;
        }

        var step = new PlanStep { Node = node, Dynamic = false, OutTypes = new DType[nOut], OutShapes = new Shape[nOut] };
        for (int o = 0; o < nOut; o++)
        {
            if (ictx.OutputIsSet(o))
            {
                step.OutTypes[o] = ictx.OutputType(o);
                step.OutShapes[o] = ictx.OutputShape(o);
            }

            int id = node.Outputs[o];
            if (id >= 0)
            {
                Kinds[id] = ValueKind.Static;
                Types[id] = step.OutTypes[o];
                Shapes[id] = step.OutShapes[o];
            }
        }

        var consts = new Tensor[node.Inputs.Length];
        for (int i = 0; i < consts.Length; i++)
        {
            if (node.Inputs[i] >= 0 && Kinds[node.Inputs[i]] == ValueKind.Const)
            {
                consts[i] = Consts[node.Inputs[i]];
            }
        }

        step.State = k.Prepare(new PrepareContext(consts, _owned, Env.Mode, Env.Path));
        steps.Add(step);
    }

    private void SetConst(int id, Tensor value, bool owned)
    {
        Kinds[id] = ValueKind.Const;
        Consts[id] = value;
        Types[id] = value.DType;
        Shapes[id] = value.Shape;
        if (owned)
        {
            _owned.Add(value);
        }
    }

    private bool TryFold(CompiledNode node, InferContext ictx, Arena scratch)
    {
        long inBytes = 0;
        foreach (int id in node.Inputs)
        {
            if (id < 0)
            {
                continue;
            }

            if (Kinds[id] != ValueKind.Const)
            {
                return false;
            }

            inBytes += (long)Shapes[id].Length * DTypes.SizeOf(Types[id]);
        }

        int nOut = node.Outputs.Length;
        long outBytes = 0;
        for (int o = 0; o < nOut; o++)
        {
            if (ictx.OutputIsSet(o))
            {
                outBytes += (long)ictx.OutputShape(o).Length * DTypes.SizeOf(ictx.OutputType(o));
            }
        }

        if (outBytes > Math.Max(inBytes, FoldAlwaysBytes))
        {
            return false;
        }

        var ins = new Tensor[node.Inputs.Length];
        for (int i = 0; i < ins.Length; i++)
        {
            ins[i] = node.Inputs[i] >= 0 ? Consts[node.Inputs[i]] : default;
        }

        var outs = new Tensor[nOut];
        for (int o = 0; o < nOut; o++)
        {
            if (ictx.OutputIsSet(o))
            {
                outs[o] = Tensor.Create(ictx.OutputType(o), ictx.OutputShape(o), Env.Pool);
            }
        }

        try
        {
            node.Kernel.Run(new OpRun(ins, outs, Env.Mode, Env.Path, Env.Threads, scratch));
        }
        catch
        {
            foreach (Tensor t in outs)
            {
                if (!t.IsEmpty)
                {
                    t.Dispose();
                }
            }

            throw;
        }
        finally
        {
            scratch.Reset();
        }

        for (int o = 0; o < nOut; o++)
        {
            if (outs[o].IsEmpty)
            {
                continue;
            }

            if (node.Outputs[o] >= 0)
            {
                SetConst(node.Outputs[o], outs[o], true);
            }
            else
            {
                outs[o].Dispose();
            }
        }

        return true;
    }

    // ---- scheduling and memory planning -----------------------------------------------------------------------------------

    private sealed class Block
    {
        public long Offset;
        public long Size;
        public int Refs;
    }

    private void Schedule()
    {
        int n = Graph.ValueCount;
        int s = Steps.Length;

        // Last step that reads each value; a value nobody reads dies where it is made; outputs live to the end.
        var lastUse = new int[n];
        Array.Fill(lastUse, -1);
        for (int i = 0; i < s; i++)
        {
            PlanStep st = Steps[i];
            foreach (int id in st.Node.Inputs)
            {
                if (id >= 0)
                {
                    lastUse[id] = Math.Max(lastUse[id], i);
                }
            }

            foreach (int id in st.Node.Outputs)
            {
                if (id >= 0)
                {
                    lastUse[id] = Math.Max(lastUse[id], i);
                }
            }
        }

        foreach (int id in Graph.OutputIds)
        {
            lastUse[id] = s;
        }

        // A view keeps its source's storage alive as long as the view lives.
        for (int i = s - 1; i >= 0; i--)
        {
            PlanStep st = Steps[i];
            if (st.IsView && st.Node.Inputs[st.Node.Kernel.ViewOfInput] is int src and >= 0 && st.Node.Outputs[0] is int dst and >= 0)
            {
                lastUse[src] = Math.Max(lastUse[src], lastUse[dst]);
            }
        }

        var dying = new List<int>[s + 1];
        for (int v = 0; v < n; v++)
        {
            if (lastUse[v] >= 0 && lastUse[v] < s)
            {
                (dying[lastUse[v]] ??= []).Add(v);
            }
        }

        var viewOut = new bool[n];
        foreach (PlanStep st in Steps)
        {
            if (st.IsView && st.Node.Outputs[0] >= 0)
            {
                viewOut[st.Node.Outputs[0]] = true;
            }
        }

        var blockOf = new Block?[n];
        var free = new List<Block>();
        long high = 0;
        long unshared = 0;
        var freedNow = new List<(long, long)>();

        for (int i = 0; i < s; i++)
        {
            PlanStep st = Steps[i];
            OpKernel k = st.Node.Kernel;
            freedNow.Clear();
            if (!st.Dynamic)
            {
                for (int o = 0; o < st.Node.Outputs.Length; o++)
                {
                    int id = st.Node.Outputs[o];
                    if (id < 0)
                    {
                        continue;
                    }

                    if (st.IsView)
                    {
                        int src = st.Node.Inputs[k.ViewOfInput];
                        if (blockOf[src] is { } sb)
                        {
                            blockOf[id] = sb;
                            sb.Refs++;
                        }

                        continue;
                    }

                    long bytes = RoundUp((long)Shapes[id].Length * DTypes.SizeOf(Types[id]));
                    unshared += bytes;
                    int inPlace = o == 0 ? k.InPlaceInput : -1;
                    if (inPlace >= 0 && st.Node.Inputs[inPlace] is int pin and >= 0 && Kinds[pin] == ValueKind.Static &&
                        blockOf[pin] is { Refs: 1 } pb && lastUse[pin] == i && pb.Size >= bytes && Types[pin] == Types[id] &&
                        Shapes[pin].Length == Shapes[id].Length)
                    {
                        blockOf[id] = pb;
                        pb.Refs++;
                        continue;
                    }

                    Block? best = null;
                    foreach (Block f in free)
                    {
                        if (f.Size >= bytes && (best is null || f.Size < best.Size))
                        {
                            best = f;
                        }
                    }

                    Block b;
                    if (best is not null)
                    {
                        free.Remove(best);
                        b = new Block { Offset = best.Offset, Size = bytes };
                        if (best.Size > bytes)
                        {
                            free.Add(new Block { Offset = best.Offset + bytes, Size = best.Size - bytes });
                        }
                    }
                    else
                    {
                        b = new Block { Offset = high, Size = bytes };
                        high += bytes;
                    }

                    b.Refs = 1;
                    blockOf[id] = b;
                    Offsets[id] = b.Offset;
                }
            }

            if (dying[i] is { } d)
            {
                foreach (int v in d)
                {
                    if (blockOf[v] is { } b && --b.Refs == 0)
                    {
                        freedNow.Add((b.Offset, b.Size));
                        InsertFree(free, b);
                    }
                }
            }

            st.Freed = [.. freedNow];
        }

        // In-place values share their source's offset.
        for (int v = 0; v < n; v++)
        {
            if (blockOf[v] is { } b && Kinds[v] == ValueKind.Static && !viewOut[v])
            {
                Offsets[v] = b.Offset;
            }
        }

        PlannedBytes = high;
        UnsharedBytes = unshared;
        long total = high;
        StagingOffsets = new long[InTypes.Length];
        for (int i = 0; i < InTypes.Length; i++)
        {
            StagingOffsets[i] = total;
            total += RoundUp((long)InShapes[i].Length * DTypes.SizeOf(InTypes[i]));
        }

        if (total > int.MaxValue - 1024)
        {
            throw new SessionException(SessionErrorKind.Unsupported,
                $"The intermediate values for these input shapes need {total} bytes at once; a single run is limited to 2 GiB.");
        }

        SlabBytes = total;
    }

    private static long RoundUp(long bytes) => Math.Max(Align, (bytes + Align - 1) & ~(Align - 1));

    // Keeps the free list sorted by offset and merges neighbours.
    private static void InsertFree(List<Block> free, Block b)
    {
        var nb = new Block { Offset = b.Offset, Size = b.Size };
        int at = 0;
        while (at < free.Count && free[at].Offset < nb.Offset)
        {
            at++;
        }

        free.Insert(at, nb);
        if (at + 1 < free.Count && free[at].Offset + free[at].Size == free[at + 1].Offset)
        {
            free[at].Size += free[at + 1].Size;
            free.RemoveAt(at + 1);
        }

        if (at > 0 && free[at - 1].Offset + free[at - 1].Size == free[at].Offset)
        {
            free[at - 1].Size += free[at].Size;
            free.RemoveAt(at);
        }
    }
}
