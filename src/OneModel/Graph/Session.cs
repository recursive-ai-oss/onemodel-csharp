using System;
using System.Collections.Immutable;
using System.IO;
using System.Threading;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Graph;

/// <summary>
/// A model ready to run: operators resolved, weights loaded, kernels chosen. Created once, then <c>Run</c> from any number of threads
/// at the same time. The model graph never changes after creation; per-run memory comes from state the session keeps for reuse, so
/// a steady-state run allocates no managed memory.
/// </summary>
/// <remarks>
/// <para>Shapes: a model may declare symbolic or unknown extents. The first run with a given set of input shapes builds an execution
/// plan for exactly those shapes (shape inference, constant evaluation, buffer reuse) and caches it, up to
/// <see cref="SessionOptions.MaxCachedPlans"/> plans; later runs with the same shapes reuse it.</para>
/// <para>Determinism: in <see cref="ExecutionMode.Fast"/> results depend on the inputs, the options and the machine's kernel path,
/// never on the thread count or on how concurrent runs interleave. <see cref="ExecutionMode.Exact"/> is bit-identical everywhere.</para>
/// </remarks>
public sealed class Session : IDisposable
{
    private const int StackInputs = 8;

    private readonly CompiledGraph _graph;
    private readonly SessionEnv _env;
    private readonly object _planLock = new();
    private ExecutionPlan[] _plans = [];
    private int _disposed;

    private Session(CompiledGraph graph, SessionOptions options, SessionEnv env)
    {
        _graph = graph;
        Options = options;
        _env = env;
        Inputs = [.. graph.Inputs];
        Outputs = [.. graph.Outputs];
    }

    /// <summary>The options the session was created with.</summary>
    public SessionOptions Options { get; }

    /// <summary>The inputs a caller must supply, in graph order. Weights stored as initializers are not listed.</summary>
    public ImmutableArray<TensorInfo> Inputs { get; }

    /// <summary>The outputs the model produces, in graph order.</summary>
    public ImmutableArray<TensorInfo> Outputs { get; }

    /// <summary>Which instruction set the kernels use (for example <c>Avx2</c>), chosen once at creation.</summary>
    public string KernelPath => _env.Path.ToString();

    /// <summary>The number of execution plans currently cached (one per distinct set of input shapes).</summary>
    public int CachedPlanCount => Volatile.Read(ref _plans).Length;

    // ---- creation ---------------------------------------------------------------------------------------------------------

    /// <summary>
    /// Creates a session from an imported model. The weights are copied, so <paramref name="model"/> may be disposed afterwards.
    /// </summary>
    /// <param name="model">The imported model.</param>
    /// <param name="options">Defaults to fast mode, one thread per core.</param>
    /// <param name="cancellationToken">Checked between operators.</param>
    /// <exception cref="SessionException">An operator is not implemented (<see cref="SessionErrorKind.UnsupportedOperator"/>, listing every one), exact mode is
    /// requested but an operator has no exact kernel (<see cref="SessionErrorKind.ExactModeUnavailable"/>, listing them), or the model is inconsistent.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public static Session Create(IrModel model, SessionOptions? options = null, CancellationToken cancellationToken = default) =>
        Create(model, options, OpRegistry.Default, null, cancellationToken);

    internal static Session Create(IrModel model, SessionOptions? options, OpRegistry registry, KernelPath? path, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(model);
        options ??= new SessionOptions();
        options.Validate();
        KernelPath chosen = path ?? KernelPaths.Best();
        int threads = options.IntraOpThreads == 0 ? ParallelFor.DefaultThreads : options.IntraOpThreads;
        var env = new SessionEnv(options.Mode, chosen, threads, BufferPool.Shared);
        CompiledGraph graph = CompiledGraph.Create(model.Graph, registry, options.Mode, options.WeightStorage, cancellationToken);
        return new Session(graph, options, env);
    }

    /// <summary>Imports an ONNX file and creates a session from it.</summary>
    /// <param name="path">The model file; external weight files are read from its directory.</param>
    /// <param name="options">Defaults to fast mode, one thread per core.</param>
    /// <param name="cancellationToken">Checked while importing and between operators.</param>
    /// <exception cref="OnnxImportException">The file is not a valid model.</exception>
    /// <exception cref="SessionException">As <see cref="Create(IrModel, SessionOptions?, CancellationToken)"/>.</exception>
    public static Session Load(string path, SessionOptions? options = null, CancellationToken cancellationToken = default)
    {
        using IrModel model = OnnxImporter.Load(path, null, cancellationToken);
        return Create(model, options, cancellationToken);
    }

    /// <summary>Imports an ONNX model from a stream (read to the end; external data is not available) and creates a session from it.</summary>
    /// <inheritdoc cref="Load(string, SessionOptions?, CancellationToken)"/>
    public static Session Load(Stream stream, SessionOptions? options = null, CancellationToken cancellationToken = default)
    {
        using IrModel model = OnnxImporter.Load(stream, null, cancellationToken);
        return Create(model, options, cancellationToken);
    }

    // ---- running ----------------------------------------------------------------------------------------------------------

    /// <summary>
    /// Runs the model on tensors. Every model input must be supplied once, by name, in any order. Each element of
    /// <paramref name="outputs"/> receives the output named at the same position in <paramref name="outputNames"/>: an empty tensor is
    /// replaced by a new one the caller owns and must dispose; a tensor of the right type and shape is overwritten in place.
    /// </summary>
    /// <param name="inputNames">Input names, one per model input.</param>
    /// <param name="inputs">Input tensors, same length. Read, never changed, and not kept after the call. Non-contiguous tensors are copied first.</param>
    /// <param name="outputNames">The outputs wanted.</param>
    /// <param name="outputs">Same length as <paramref name="outputNames"/>.</param>
    /// <exception cref="SessionException"><see cref="SessionErrorKind.InvalidInput"/> for an unknown or missing name, a wrong element type or shape, or a bad output
    /// tensor; <see cref="SessionErrorKind.InvalidModel"/> if an operator cannot handle these shapes; <see cref="SessionErrorKind.Disposed"/>.</exception>
    public void Run(ReadOnlySpan<string> inputNames, ReadOnlySpan<Tensor> inputs, ReadOnlySpan<string> outputNames, Span<Tensor> outputs)
    {
        ThrowIfDisposed();
        if (inputNames.Length != inputs.Length || outputNames.Length != outputs.Length)
        {
            throw Invalid("The name and tensor lists must have the same length.");
        }

        int nIn = _graph.InputIds.Length;
        if (inputs.Length != nIn)
        {
            throw Invalid($"The model has {nIn} input(s) ({InputNameList()}) but {inputs.Length} were supplied.");
        }

        Span<int> order = nIn <= StackInputs ? stackalloc int[StackInputs] : new int[nIn];
        Span<DType> types = nIn <= StackInputs ? stackalloc DType[StackInputs] : new DType[nIn];
        Span<Shape> shapes = nIn <= StackInputs ? stackalloc Shape[StackInputs] : new Shape[nIn];
        Tensor[]? temps = null;
        try
        {
            order.Slice(0, nIn).Fill(-1);
            for (int i = 0; i < nIn; i++)
            {
                if (!_graph.InputIndex.TryGetValue(inputNames[i], out int slot))
                {
                    throw Invalid($"The model has no input named '{inputNames[i]}'. Its inputs are {InputNameList()}.");
                }

                if (order[slot] >= 0)
                {
                    throw Invalid($"Input '{inputNames[i]}' was supplied twice.");
                }

                order[slot] = i;
                Tensor t = inputs[i];
                if (t.IsEmpty)
                {
                    throw Invalid($"Input '{inputNames[i]}' is an empty tensor.");
                }

                if (!t.IsContiguous || t.IsBlocked)
                {
                    temps ??= new Tensor[nIn];
                    t = temps[i] = t.Contiguous(_env.Pool);
                }

                CheckInput(slot, t.DType, t.Shape);
                types[slot] = t.DType;
                shapes[slot] = t.Shape;
            }

            ExecutionPlan plan = AcquirePlan(types.Slice(0, nIn), shapes.Slice(0, nIn));
            RunState state = plan.RentState();
            try
            {
                for (int slot = 0; slot < nIn; slot++)
                {
                    int i = order[slot];
                    state.Values[_graph.InputIds[slot]] = temps?[i].IsEmpty == false ? temps[i] : inputs[i];
                }

                state.Execute();
                Deliver(state, outputNames, outputs);
            }
            finally
            {
                state.Clear();
                plan.ReturnState(state);
                plan.Release();
            }
        }
        finally
        {
            if (temps is not null)
            {
                foreach (Tensor t in temps)
                {
                    if (!t.IsEmpty)
                    {
                        t.Dispose();
                    }
                }
            }
        }
    }

    /// <summary>
    /// Runs a model that has exactly one input on a span of values, and copies one output into a span. Allocates nothing once the
    /// plan for <paramref name="inputShape"/> exists and the session has warmed up.
    /// </summary>
    /// <typeparam name="TIn">The input element type; must equal the model input's type.</typeparam>
    /// <typeparam name="TOut">The output element type; must equal the model output's type.</typeparam>
    /// <param name="inputName">The input's name.</param>
    /// <param name="input">The values in row-major order; its length must equal the product of <paramref name="inputShape"/>.</param>
    /// <param name="inputShape">The input's shape.</param>
    /// <param name="outputName">The output wanted.</param>
    /// <param name="output">Receives the output in row-major order; must be at least as long as the output.</param>
    /// <returns>The output's shape.</returns>
    /// <exception cref="SessionException">As for the tensor overload; also <see cref="SessionErrorKind.InvalidInput"/> if the model has more than one input
    /// or <paramref name="output"/> is too short (the message gives the length needed).</exception>
    public Shape Run<TIn, TOut>(string inputName, ReadOnlySpan<TIn> input, ReadOnlySpan<int> inputShape, string outputName, Span<TOut> output)
        where TIn : unmanaged
        where TOut : unmanaged
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(inputName);
        ArgumentNullException.ThrowIfNull(outputName);
        if (_graph.InputIds.Length != 1)
        {
            throw Invalid($"The model has {_graph.InputIds.Length} inputs ({InputNameList()}); this overload is for a single input. Use the tensor overload.");
        }

        if (!_graph.InputIndex.TryGetValue(inputName, out _))
        {
            throw Invalid($"The model has no input named '{inputName}'. Its inputs are {InputNameList()}.");
        }

        if (!_graph.OutputIndex.TryGetValue(outputName, out int outSlot))
        {
            throw Invalid($"The model has no output named '{outputName}'. Its outputs are {OutputNameList()}.");
        }

        Shape shape;
        try
        {
            shape = Shape.Create(inputShape);
        }
        catch (TensorShapeException e)
        {
            throw Invalid($"Input '{inputName}': {e.Message}");
        }

        if (shape.Length != input.Length)
        {
            throw Invalid($"Input '{inputName}' has shape {shape} ({shape.Length} values) but {input.Length} values were supplied.");
        }

        DType dt = DTypes.Of<TIn>();
        CheckInput(0, dt, shape);
        Span<DType> types = [dt];
        Span<Shape> shapes = [shape];
        ExecutionPlan plan = AcquirePlan(types, shapes);
        RunState state = plan.RentState();
        try
        {
            Tensor staged = state.Staging[0];
            input.CopyTo(staged.MutableData<TIn>());
            state.Values[_graph.InputIds[0]] = staged;
            state.Execute();
            Tensor result = state.Values[_graph.OutputIds[outSlot]];
            if (result.DType != DTypes.Of<TOut>())
            {
                throw Invalid($"Output '{outputName}' holds {result.DType}, not {DTypes.Of<TOut>()}.");
            }

            if (result.Length > output.Length)
            {
                throw Invalid($"Output '{outputName}' has shape {result.Shape} ({result.Length} values) but the buffer holds {output.Length}.");
            }

            ReadOnlySpan<TOut> data = result.Data<TOut>();
            data.CopyTo(output);
            return result.Shape;
        }
        finally
        {
            state.Clear();
            plan.ReturnState(state);
            plan.Release();
        }
    }

    private void Deliver(RunState state, ReadOnlySpan<string> outputNames, Span<Tensor> outputs)
    {
        int made = 0;
        try
        {
            for (int k = 0; k < outputNames.Length; k++)
            {
                if (!_graph.OutputIndex.TryGetValue(outputNames[k], out int slot))
                {
                    throw Invalid($"The model has no output named '{outputNames[k]}'. Its outputs are {OutputNameList()}.");
                }

                Tensor v = state.Values[_graph.OutputIds[slot]];
                if (outputs[k].IsEmpty)
                {
                    Tensor t = Tensor.Create(v.DType, v.Shape, _env.Pool);
                    t.CopyFrom(v);
                    outputs[k] = t;
                    made = k + 1;
                }
                else
                {
                    if (outputs[k].DType != v.DType || outputs[k].Shape != v.Shape)
                    {
                        throw Invalid($"Output tensor for '{outputNames[k]}' is {outputs[k].DType}{outputs[k].Shape} but the model produces {v.DType}{v.Shape}.");
                    }

                    outputs[k].CopyFrom(v);
                }
            }
        }
        catch
        {
            // Do not hand back tensors the caller will never see a result for.
            for (int k = 0; k < made; k++)
            {
                if (!outputs[k].IsEmpty && !ReferenceEquals(outputs[k].Buffer, null))
                {
                    outputs[k].Dispose();
                    outputs[k] = default;
                }
            }

            throw;
        }
    }

    // ---- plans ------------------------------------------------------------------------------------------------------------

    private ExecutionPlan AcquirePlan(ReadOnlySpan<DType> types, ReadOnlySpan<Shape> shapes)
    {
        ExecutionPlan[] snapshot = Volatile.Read(ref _plans);
        foreach (ExecutionPlan p in snapshot)
        {
            if (p.Matches(types, shapes) && p.TryAcquire())
            {
                return p;
            }
        }

        lock (_planLock)
        {
            ThrowIfDisposed();
            snapshot = _plans;
            foreach (ExecutionPlan p in snapshot)
            {
                if (p.Matches(types, shapes) && p.TryAcquire())
                {
                    return p;
                }
            }

            ExecutionPlan plan = ExecutionPlan.Build(_graph, _env, types, shapes, CancellationToken.None);
            int keep = Math.Min(snapshot.Length, Options.MaxCachedPlans - 1);
            var next = new ExecutionPlan[keep + 1];
            Array.Copy(snapshot, snapshot.Length - keep, next, 0, keep);
            next[keep] = plan;
            plan.TryAcquire();
            Volatile.Write(ref _plans, next);
            for (int i = 0; i < snapshot.Length - keep; i++)
            {
                snapshot[i].Release();
            }

            return plan;
        }
    }

    /// <summary>Statistics of the plan for the given input shapes, building it if needed. For tests and diagnostics.</summary>
    internal (long PlannedBytes, long UnsharedBytes, long SlabBytes) PlanStatistics(ReadOnlySpan<DType> types, ReadOnlySpan<Shape> shapes)
    {
        ExecutionPlan p = AcquirePlan(types, shapes);
        try
        {
            return (p.PlannedBytes, p.UnsharedBytes, p.SlabBytes);
        }
        finally
        {
            p.Release();
        }
    }

    // ---- checks -----------------------------------------------------------------------------------------------------------

    private void CheckInput(int slot, DType type, Shape shape)
    {
        TensorInfo info = _graph.Inputs[slot];
        if (type != info.DType)
        {
            throw Invalid($"Input '{info.Name}' must be {info.DType} but {type} was supplied.");
        }

        if (!info.HasShape)
        {
            return;
        }

        if (shape.Rank != info.Dims.Length)
        {
            throw Invalid($"Input '{info.Name}' must have {info.Dims.Length} dimensions ({FormatDims(info)}) but the shape {shape} has {shape.Rank}.");
        }

        for (int d = 0; d < info.Dims.Length; d++)
        {
            if (info.Dims[d].IsFixed && info.Dims[d].Value != shape[d])
            {
                throw Invalid($"Input '{info.Name}' must have shape {FormatDims(info)} but {shape} was supplied (dimension {d}).");
            }
        }
    }

    private static string FormatDims(TensorInfo info) => "[" + string.Join(", ", info.Dims) + "]";

    private string InputNameList() => string.Join(", ", System.Linq.Enumerable.Select(_graph.Inputs, i => "'" + i.Name + "'"));

    private string OutputNameList() => string.Join(", ", System.Linq.Enumerable.Select(_graph.Outputs, i => "'" + i.Name + "'"));

    private static SessionException Invalid(string message) => new(SessionErrorKind.InvalidInput, message);

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
        {
            throw new SessionException(SessionErrorKind.Disposed, "The session was disposed.");
        }
    }

    /// <summary>
    /// Releases the weights and cached plans once every run in progress has finished. Calling <c>Run</c> afterwards throws
    /// <see cref="SessionException"/> with <see cref="SessionErrorKind.Disposed"/>.
    /// </summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        lock (_planLock)
        {
            foreach (ExecutionPlan p in _plans)
            {
                p.Release();
            }

            _plans = [];
        }

        _graph.Dispose();
    }
}
