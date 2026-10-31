using System;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Runtime;

/// <summary>
/// Runs a kernel that only knows binary32 on tensors stored as binary16 or bfloat16: each half-precision input is widened to binary32
/// (exactly), the kernel runs, and each result it produced as binary32 is rounded back to the storage type (nearest even). So a model
/// stored in half precision runs with the full-precision arithmetic of the kernels and half-precision storage between operators, which
/// is what half-precision storage means. docs/fp16.md has the rules.
/// </summary>
/// <remarks>
/// Weights that are constants are widened once when the plan is made and kept as binary32, unless the session asked to keep them in
/// half precision (<see cref="ModelOptions.WeightStorage"/>): those are widened for each run, from the compact copy, into run scratch.
/// A node with no half-precision input is passed straight to the kernel, as are the few operators (<c>Add, Sub, Mul, Div</c> on
/// binary16) that already compute in half precision.
/// </remarks>
internal sealed class HalfAdapterKernel : OpKernel
{
    private readonly OpKernel _inner;
    private readonly string _opType;
    private readonly bool[] _keepCompact;

    [ThreadStatic]
    private static Tensor[]? t_inputs;

    [ThreadStatic]
    private static Tensor[]? t_outputs;

    [ThreadStatic]
    private static Tensor[]? t_temps;

    private HalfAdapterKernel(OpKernel inner, string opType, bool[] keepCompact)
    {
        _inner = inner;
        _opType = opType;
        _keepCompact = keepCompact;
    }

    /// <summary>
    /// Whether a kernel can be wrapped. Kernels that build views, produce values the plan knows without running (shape, constants), cast,
    /// or shape their output from values at run time must see the real element types themselves.
    /// </summary>
    public static bool CanWrap(string opType, OpKernel kernel) =>
        kernel.ViewOfInput < 0 && !kernel.OutputsDataDependent &&
        opType is not ("Cast" or "CastLike" or "Constant" or "ConstantOfShape" or "Shape" or "Size" or "Range" or "If" or "Loop" or "Scan");

    /// <summary>Wraps <paramref name="kernel"/>; <paramref name="keepCompact"/> marks the inputs whose half-precision constants stay compact.</summary>
    public static OpKernel Wrap(string opType, OpKernel kernel, bool[] keepCompact) => new HalfAdapterKernel(kernel, opType, keepCompact);

    public override bool SupportsExact => _inner.SupportsExact;

    public override int InPlaceInput => _inner.InPlaceInput;

    // The inner kernel is owned (and disposed) by the compiled graph that created it.
    public override void Dispose() => base.Dispose();

    private static bool IsHalf(DType t) => t is DType.F16 or DType.BF16;

    private bool NativeOnHalf => _opType is "Add" or "Sub" or "Mul" or "Div";

    // Whether a node is adapted, from the types of its present inputs: some input is half precision, and the node is not one of the
    // operators that compute in binary16 themselves with every input binary16.
    private struct Scan
    {
        private bool _any;
        private bool _allBinary16;

        public static Scan Start() => new() { _allBinary16 = true };

        public void Add(DType t)
        {
            _any |= IsHalf(t);
            _allBinary16 &= t == DType.F16;
        }

        public readonly bool Adapted(bool nativeOnBinary16) => _any && !(nativeOnBinary16 && _allBinary16);
    }

    public override void Infer(InferContext ctx)
    {
        Scan scan = Scan.Start();
        for (int i = 0; i < ctx.InputCount; i++)
        {
            if (ctx.HasInput(i))
            {
                scan.Add(ctx.InputType(i));
            }
        }

        if (!scan.Adapted(NativeOnHalf))
        {
            _inner.Infer(ctx);
            return;
        }

        DType storage = ctx.WidenHalfInputs(_keepCompact);
        _inner.Infer(ctx);
        ctx.RetypeFloatOutputs(storage);
    }

    private sealed class State
    {
        public required bool Adapted { get; init; }

        public object? Inner { get; init; }

        /// <summary>Per input: the binary32 copy of a half-precision constant made when the plan was made, or empty.</summary>
        public Tensor[] Widened { get; init; } = [];
    }

    public override object? Prepare(PrepareContext ctx)
    {
        int n = ctx.InputTypes.Length;
        Scan scan = Scan.Start();
        for (int i = 0; i < n; i++)
        {
            if (ctx.InputTypes[i] != DType.Undefined)
            {
                scan.Add(ctx.InputTypes[i]);
            }
        }

        if (!scan.Adapted(NativeOnHalf))
        {
            return new State { Adapted = false, Inner = _inner.Prepare(ctx) };
        }

        var consts = new Tensor[n];
        var widened = new Tensor[n];
        for (int i = 0; i < n; i++)
        {
            if (!ctx.IsConstant(i))
            {
                continue;
            }

            Tensor c = ctx.Constant(i);
            if (!IsHalf(c.DType))
            {
                consts[i] = c;
            }
            else if (!(i < _keepCompact.Length && _keepCompact[i]))
            {
                Tensor f = ctx.CreateOwned(DType.F32, c.Shape);
                HalfConvert.Widen(c, f);
                consts[i] = f;
                widened[i] = f;
            }
        }

        return new State { Adapted = true, Inner = _inner.Prepare(ctx.WithConstants(consts)), Widened = widened };
    }

    public override void Run(in OpRun run)
    {
        var state = run.State as State;
        ReadOnlySpan<Tensor> inputs = run.Inputs;
        bool adapted;
        if (state is not null)
        {
            adapted = state.Adapted;
        }
        else
        {
            // Run without a prepared state (a node folded while the plan is made): decide from the tensors.
            Scan scan = Scan.Start();
            for (int i = 0; i < inputs.Length; i++)
            {
                if (!inputs[i].IsEmpty)
                {
                    scan.Add(inputs[i].DType);
                }
            }

            adapted = scan.Adapted(NativeOnHalf);
        }

        if (!adapted)
        {
            _inner.Run(state is null ? run : run.With(run.Inputs, run.Outputs, state.Inner));
            return;
        }

        int nIn = inputs.Length;
        int nOut = run.Outputs.Length;
        Tensor[] ins = t_inputs is { } a && a.Length >= nIn ? a : (t_inputs = new Tensor[Math.Max(nIn, 16)]);
        Tensor[] outs = t_outputs is { } b && b.Length >= nOut ? b : (t_outputs = new Tensor[Math.Max(nOut, 8)]);
        Tensor[] temps = t_temps is { } c && c.Length >= nIn + nOut ? c : (t_temps = new Tensor[Math.Max(nIn + nOut, 24)]);
        int nTemps = 0;
        try
        {
            // Temporary binary32 copies come from the buffer pool and go back to it as soon as this node is done, so a model's
            // half-precision values never exist as binary32 all at once.
            for (int i = 0; i < nIn; i++)
            {
                Tensor t = inputs[i];
                if (!t.IsEmpty && IsHalf(t.DType))
                {
                    Tensor w = state is not null && i < state.Widened.Length ? state.Widened[i] : default;
                    if (w.IsEmpty)
                    {
                        w = Tensor.Create(DType.F32, t.Shape);
                        temps[nTemps++] = w;
                        HalfConvert.Widen(t, w);
                    }

                    t = w;
                }

                ins[i] = t;
            }

            for (int o = 0; o < nOut; o++)
            {
                Tensor t = run.Outputs[o];
                if (!t.IsEmpty && IsHalf(t.DType))
                {
                    Tensor w = Tensor.Create(DType.F32, t.Shape);
                    temps[nTemps++] = w;
                    t = w;
                }

                outs[o] = t;
            }

            _inner.Run(run.With(ins.AsSpan(0, nIn), outs.AsSpan(0, nOut), state?.Inner));
            for (int o = 0; o < nOut; o++)
            {
                Tensor t = run.Outputs[o];
                if (!t.IsEmpty && IsHalf(t.DType))
                {
                    HalfConvert.Narrow(outs[o], t);
                }
            }
        }
        finally
        {
            for (int i = 0; i < nTemps; i++)
            {
                temps[i].Dispose();
                temps[i] = default;
            }
        }
    }
}
