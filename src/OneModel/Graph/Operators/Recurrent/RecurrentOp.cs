using System;
using System.Collections.Immutable;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

// Recurrent operators: RNN, GRU and LSTM share their attribute handling, shape rules and weight preparation here; the arithmetic of
// the cells is in Kernels/Recurrent.cs. The semantics are Operators.md's, with the choices it leaves open written in
// docs/ops-recurrent.md.

/// <summary>What <see cref="RecurrentOp.Prepare"/> keeps: the weights transposed once so the GEMM can read them as its B operand.</summary>
internal sealed class RecurrentWeights
{
    public Tensor WT { get; init; }

    public Tensor RT { get; init; }
}

/// <summary>Base of RNN, GRU and LSTM: forward, reverse and bidirectional, layouts 0 and 1, sequence lengths, initial state, clipping.</summary>
internal abstract class RecurrentOp : OpKernel
{
    private readonly RecurrentCell _cell;
    private readonly int _dirs;
    private readonly bool _reverseFirst;
    private readonly long _hiddenAttr;
    private readonly bool _layout1;
    private readonly float _clip;
    private readonly bool _inputForget;
    private readonly bool _linearBeforeReset;
    private readonly RecurrentActivation[] _acts; // per direction, ActsPerDirection each
    private readonly int _actsPer;

    protected RecurrentOp(NodeInfo node, RecurrentCell cell)
    {
        _cell = cell;
        string dir = node.GetString("direction", "forward");
        switch (dir)
        {
            case "forward": _dirs = 1; break;
            case "reverse": _dirs = 1; _reverseFirst = true; break;
            case "bidirectional": _dirs = 2; break;
            default: throw node.Error($"direction '{dir}' is not forward, reverse or bidirectional");
        }

        _hiddenAttr = node.GetInt("hidden_size", -1);
        long layout = node.GetInt("layout", 0);
        if (layout is not (0 or 1))
        {
            throw node.Error($"layout {layout} is not 0 or 1");
        }

        _layout1 = layout == 1;
        _clip = node.Has("clip") ? node.GetFloat("clip", 0f) : 0f;
        if (_clip < 0f || float.IsNaN(_clip))
        {
            throw node.Error("clip must not be negative");
        }

        _inputForget = cell == RecurrentCell.Lstm && node.GetInt("input_forget", 0) != 0;
        _linearBeforeReset = cell == RecurrentCell.Gru && node.GetInt("linear_before_reset", 0) != 0;
        _actsPer = cell switch { RecurrentCell.Lstm => 3, RecurrentCell.Gru => 2, _ => 1 };
        _acts = ParseActivations(node, cell, _dirs, _actsPer);
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        int g = RecurrentKernel.Gates(_cell);
        if (!ctx.HasInput(0) || !ctx.HasInput(1) || !ctx.HasInput(2))
        {
            throw ctx.Fail("expects at least X, W and R");
        }

        for (int i = 0; i < 3; i++)
        {
            if (ctx.InputType(i) != DType.F32)
            {
                throw ctx.Unsupported($"element type {ctx.InputType(i)} is not supported yet (float32 only)");
            }
        }

        Shape x = ctx.InputShape(0), w = ctx.InputShape(1), r = ctx.InputShape(2);
        if (x.Rank != 3)
        {
            throw ctx.Fail($"X must have 3 dimensions, got {x}");
        }

        if (w.Rank != 3 || r.Rank != 3)
        {
            throw ctx.Fail($"W and R must have 3 dimensions, got {w} and {r}");
        }

        int seq = _layout1 ? x[1] : x[0];
        int batch = _layout1 ? x[0] : x[1];
        int input = x[2];
        int hid = r[2];
        if (_hiddenAttr >= 0 && _hiddenAttr != hid)
        {
            throw ctx.Fail($"hidden_size {_hiddenAttr} does not match R's last dimension {hid}");
        }

        if (hid <= 0)
        {
            throw ctx.Fail("the hidden size must be positive");
        }

        if (w[0] != _dirs || r[0] != _dirs || w[1] != g * hid || r[1] != g * hid || w[2] != input)
        {
            throw ctx.Fail($"W {w} and R {r} do not fit X {x}: expected W [{_dirs}, {g * hid}, {input}] and R [{_dirs}, {g * hid}, {hid}]");
        }

        if (ctx.HasInput(3) && !(ctx.InputType(3) == DType.F32 && ctx.InputShape(3) == Shape.Create(_dirs, 2 * g * hid)))
        {
            throw ctx.Fail($"B must be float32 [{_dirs}, {2 * g * hid}], got {ctx.InputShape(3)}");
        }

        if (ctx.HasInput(4) && !(ctx.InputType(4) == DType.I32 && ctx.InputShape(4) == Shape.Create(batch)))
        {
            throw ctx.Fail($"sequence_lens must be int32 [{batch}], got {ctx.InputShape(4)}");
        }

        Shape state = _layout1 ? Shape.Create(batch, _dirs, hid) : Shape.Create(_dirs, batch, hid);
        for (int i = 5; i <= (_cell == RecurrentCell.Lstm ? 6 : 5); i++)
        {
            if (ctx.HasInput(i) && !(ctx.InputType(i) == DType.F32 && ctx.InputShape(i) == state))
            {
                throw ctx.Fail($"input {i} (initial state) must be float32 {state}, got {ctx.InputShape(i)}");
            }
        }

        if (_cell == RecurrentCell.Lstm && ctx.HasInput(7) && !(ctx.InputType(7) == DType.F32 && ctx.InputShape(7) == Shape.Create(_dirs, 3 * hid)))
        {
            throw ctx.Fail($"P must be float32 [{_dirs}, {3 * hid}], got {ctx.InputShape(7)}");
        }

        long scratch = RecurrentKernel.ScratchFloats(_cell, seq, batch, hid) + ((long)input * g * hid) + ((long)hid * g * hid);
        if (scratch >= int.MaxValue / 2)
        {
            throw ctx.Unsupported("the recurrent layer is too large for one scratch buffer");
        }

        Shape y = _layout1 ? Shape.Create(batch, seq, _dirs, hid) : Shape.Create(seq, _dirs, batch, hid);
        ctx.SetOutput(0, DType.F32, y);
        if (ctx.OutputCount > 1)
        {
            ctx.SetOutput(1, DType.F32, state);
        }

        if (ctx.OutputCount > 2)
        {
            ctx.SetOutput(2, DType.F32, state);
        }
    }

    public override object? Prepare(PrepareContext ctx)
    {
        Tensor wt = default, rt = default;
        if (ctx.IsConstant(1))
        {
            Tensor w = ctx.Constant(1);
            wt = ctx.CreateOwned(DType.F32, Shape.Create(_dirs, w.Shape[2], w.Shape[1]));
            for (int d = 0; d < _dirs; d++)
            {
                Transpose(w.Data<float>().Slice(d * w.Shape[1] * w.Shape[2], w.Shape[1] * w.Shape[2]),
                    wt.MutableData<float>().Slice(d * w.Shape[1] * w.Shape[2], w.Shape[1] * w.Shape[2]), w.Shape[1], w.Shape[2]);
            }
        }

        if (ctx.IsConstant(2))
        {
            Tensor r = ctx.Constant(2);
            rt = ctx.CreateOwned(DType.F32, Shape.Create(_dirs, r.Shape[2], r.Shape[1]));
            for (int d = 0; d < _dirs; d++)
            {
                Transpose(r.Data<float>().Slice(d * r.Shape[1] * r.Shape[2], r.Shape[1] * r.Shape[2]),
                    rt.MutableData<float>().Slice(d * r.Shape[1] * r.Shape[2], r.Shape[1] * r.Shape[2]), r.Shape[1], r.Shape[2]);
            }
        }

        return wt.IsEmpty && rt.IsEmpty ? null : new RecurrentWeights { WT = wt, RT = rt };
    }

    public override void Run(in OpRun run)
    {
        Tensor xT = run.Inputs[0], wTensor = run.Inputs[1], rTensor = run.Inputs[2];
        int seq = _layout1 ? xT.Shape[1] : xT.Shape[0];
        int batch = _layout1 ? xT.Shape[0] : xT.Shape[1];
        int input = xT.Shape[2];
        int hid = rTensor.Shape[2];
        int g = RecurrentKernel.Gates(_cell);
        int gh = g * hid;
        ReadOnlySpan<int> lens = default;
        if (run.Inputs.Length > 4 && !run.Inputs[4].IsEmpty)
        {
            lens = run.Inputs[4].Data<int>();
            foreach (int l in lens)
            {
                if (l < 0 || l > seq)
                {
                    throw new SessionException(SessionErrorKind.InvalidInput, $"sequence_lens contains {l}, outside [0, {seq}].");
                }
            }
        }

        int nIn = run.Inputs.Length;
        Tensor bT = nIn > 3 ? run.Inputs[3] : default, h0T = nIn > 5 ? run.Inputs[5] : default;
        Tensor c0T = _cell == RecurrentCell.Lstm && nIn > 6 ? run.Inputs[6] : default, pT = _cell == RecurrentCell.Lstm && nIn > 7 ? run.Inputs[7] : default;
        Tensor yT = run.Outputs[0];
        Tensor yhT = run.Outputs.Length > 1 ? run.Outputs[1] : default;
        Tensor ycT = run.Outputs.Length > 2 ? run.Outputs[2] : default;
        if (!lens.IsEmpty && !yT.IsEmpty)
        {
            // Steps past a row's length are defined as zeros in Y.
            yT.MutableData<float>().Clear();
        }

        var weights = run.State as RecurrentWeights;
        Tensor wtScratch = default, rtScratch = default;
        if (weights is null || weights.WT.IsEmpty)
        {
            wtScratch = run.Scratch(DType.F32, Shape.Create(input, gh));
        }

        if (weights is null || weights.RT.IsEmpty)
        {
            rtScratch = run.Scratch(DType.F32, Shape.Create(hid, gh));
        }

        long total = RecurrentKernel.ScratchFloats(_cell, seq, batch, hid);
        Span<float> scratch = batch == 0 || seq == 0 ? default : run.Scratch(DType.F32, Shape.Create((int)total)).MutableData<float>();

        for (int dir = 0; dir < _dirs; dir++)
        {
            ReadOnlySpan<float> wt, rt;
            if (wtScratch.IsEmpty)
            {
                wt = weights!.WT.Data<float>().Slice(dir * input * gh, input * gh);
            }
            else
            {
                Span<float> dst = wtScratch.MutableData<float>();
                Transpose(wTensor.Data<float>().Slice(dir * gh * input, gh * input), dst, gh, input);
                wt = dst;
            }

            if (rtScratch.IsEmpty)
            {
                rt = weights!.RT.Data<float>().Slice(dir * hid * gh, hid * gh);
            }
            else
            {
                Span<float> dst = rtScratch.MutableData<float>();
                Transpose(rTensor.Data<float>().Slice(dir * gh * hid, gh * hid), dst, gh, hid);
                rt = dst;
            }

            var d = new RecurrentDirection
            {
                Cell = _cell,
                Exact = run.Exact,
                Path = run.Path,
                Threads = run.Threads,
                Seq = seq,
                Batch = batch,
                Input = input,
                Hidden = hid,
                Reverse = _reverseFirst || dir == 1,
                X = xT.Data<float>(),
                XTimeStep = _layout1 ? 1 : batch,
                XBatchStep = _layout1 ? seq : 1,
                WT = wt,
                RT = rt,
                Lens = lens,
                Clip = _clip,
                InputForget = _inputForget,
                LinearBeforeReset = _linearBeforeReset,
            };

            RecurrentActivation[] acts = _acts;
            int a0 = dir * _actsPer;
            d.F = acts[a0];
            d.G = _actsPer > 1 ? acts[a0 + 1] : d.F;
            d.Hact = _actsPer > 2 ? acts[a0 + 2] : d.G;

            if (!bT.IsEmpty)
            {
                ReadOnlySpan<float> b = bT.Data<float>().Slice(dir * 2 * gh, 2 * gh);
                d.Wb = b.Slice(0, gh);
                d.Rb = b.Slice(gh, gh);
            }

            if (!pT.IsEmpty)
            {
                d.Peep = pT.Data<float>().Slice(dir * 3 * hid, 3 * hid);
            }

            d.StateBatchStride = _layout1 ? _dirs * hid : hid;
            d.StateOffset = _layout1 ? dir * hid : dir * batch * hid;
            if (!h0T.IsEmpty)
            {
                d.H0 = h0T.Data<float>().Slice(d.StateOffset);
            }

            if (!c0T.IsEmpty)
            {
                d.C0 = c0T.Data<float>().Slice(d.StateOffset);
            }

            if (!yT.IsEmpty)
            {
                d.Y = yT.MutableData<float>();
                d.YOffset = _layout1 ? dir * hid : dir * batch * hid;
                d.YTimeStep = _layout1 ? _dirs * hid : _dirs * batch * hid;
                d.YBatchStep = _layout1 ? seq * _dirs * hid : hid;
            }

            if (!yhT.IsEmpty)
            {
                d.YH = yhT.MutableData<float>();
            }

            if (!ycT.IsEmpty)
            {
                d.YC = ycT.MutableData<float>();
            }

            if (batch == 0 || seq == 0)
            {
                // Nothing to step through: the final states are the initial states (or zeros).
                FinalFromInitial(ref d, batch, hid);
                continue;
            }

            RecurrentKernel.Bind(ref d, scratch);
            RecurrentKernel.Run(ref d);
        }
    }

    private static void FinalFromInitial(ref RecurrentDirection d, int batch, int hid)
    {
        for (int b = 0; b < batch; b++)
        {
            if (!d.YH.IsEmpty)
            {
                Span<float> dst = d.YH.Slice(d.StateOffset + (b * d.StateBatchStride), hid);
                if (d.H0.IsEmpty) { dst.Clear(); } else { d.H0.Slice(b * d.StateBatchStride, hid).CopyTo(dst); }
            }

            if (!d.YC.IsEmpty)
            {
                Span<float> dst = d.YC.Slice(d.StateOffset + (b * d.StateBatchStride), hid);
                if (d.C0.IsEmpty) { dst.Clear(); } else { d.C0.Slice(b * d.StateBatchStride, hid).CopyTo(dst); }
            }
        }
    }

    // dst[c, r] = src[r, c] for src of rows x cols, in square tiles so both sides stay in cache.
    private static void Transpose(ReadOnlySpan<float> src, Span<float> dst, int rows, int cols)
    {
        const int Tile = 32;
        for (int r0 = 0; r0 < rows; r0 += Tile)
        {
            int r1 = Math.Min(rows, r0 + Tile);
            for (int c0 = 0; c0 < cols; c0 += Tile)
            {
                int c1 = Math.Min(cols, c0 + Tile);
                for (int r = r0; r < r1; r++)
                {
                    for (int c = c0; c < c1; c++)
                    {
                        dst[(c * rows) + r] = src[(r * cols) + c];
                    }
                }
            }
        }
    }

    private static RecurrentActivation[] ParseActivations(NodeInfo node, RecurrentCell cell, int dirs, int per)
    {
        ImmutableArray<float> alphas = node.GetFloats("activation_alpha");
        ImmutableArray<float> betas = node.GetFloats("activation_beta");
        var names = new string[per * dirs];
        IrAttr? attr = node.Find("activations");
        if (attr is null)
        {
            for (int i = 0; i < names.Length; i++)
            {
                int k = i % per;
                names[i] = cell switch
                {
                    RecurrentCell.Lstm => k == 0 ? "Sigmoid" : "Tanh",
                    RecurrentCell.Gru => k == 0 ? "Sigmoid" : "Tanh",
                    _ => "Tanh",
                };
            }
        }
        else
        {
            if (attr.Type != IrAttributeType.Texts)
            {
                throw node.Error("attribute 'activations' must be a list of strings");
            }

            ImmutableArray<ReadOnlyMemory<byte>> given = attr.Texts;
            if (given.Length != per && given.Length != per * dirs)
            {
                throw node.Error($"activations has {given.Length} entries; expected {per}" + (dirs > 1 ? $" or {per * dirs}" : string.Empty));
            }

            for (int i = 0; i < names.Length; i++)
            {
                names[i] = System.Text.Encoding.UTF8.GetString(given[i % given.Length].Span);
            }
        }

        // Alpha and beta values are handed out in activation order, one per activation that takes the parameter (docs/ops-recurrent.md).
        var result = new RecurrentActivation[names.Length];
        int ai = 0, bi = 0;
        for (int i = 0; i < names.Length; i++)
        {
            RecurrentActivationKind kind;
            float defAlpha, defBeta = 0f;
            bool usesAlpha, usesBeta = false;
            switch (names[i].ToLowerInvariant())
            {
                case "sigmoid": kind = RecurrentActivationKind.Sigmoid; defAlpha = 0; usesAlpha = false; break;
                case "tanh": kind = RecurrentActivationKind.Tanh; defAlpha = 0; usesAlpha = false; break;
                case "relu": kind = RecurrentActivationKind.Relu; defAlpha = 0; usesAlpha = false; break;
                case "affine": kind = RecurrentActivationKind.Affine; defAlpha = 1f; usesAlpha = true; usesBeta = true; break;
                case "leakyrelu": kind = RecurrentActivationKind.LeakyRelu; defAlpha = 0.01f; usesAlpha = true; break;
                case "thresholdedrelu": kind = RecurrentActivationKind.ThresholdedRelu; defAlpha = 1f; usesAlpha = true; break;
                case "scaledtanh": kind = RecurrentActivationKind.ScaledTanh; defAlpha = 1f; usesAlpha = true; usesBeta = true; break;
                case "hardsigmoid": kind = RecurrentActivationKind.HardSigmoid; defAlpha = 0.2f; defBeta = 0.5f; usesAlpha = true; usesBeta = true; break;
                case "elu": kind = RecurrentActivationKind.Elu; defAlpha = 1f; usesAlpha = true; break;
                case "softsign": kind = RecurrentActivationKind.Softsign; defAlpha = 0; usesAlpha = false; break;
                case "softplus": kind = RecurrentActivationKind.Softplus; defAlpha = 0; usesAlpha = false; break;
                default: throw node.Unsupported($"activation '{names[i]}' is not supported");
            }

            float alpha = defAlpha, beta = defBeta;
            if (usesAlpha && ai < alphas.Length)
            {
                alpha = alphas[ai++];
            }

            if (usesBeta && bi < betas.Length)
            {
                beta = betas[bi++];
            }

            result[i] = new RecurrentActivation(kind, alpha, beta);
        }

        return result;
    }
}

/// <summary>LSTM (opsets 1, 7, 14, 22): inputs X, W, R, B, sequence_lens, initial_h, initial_c, P; outputs Y, Y_h, Y_c.</summary>
internal sealed class LstmOp : RecurrentOp
{
    public static readonly OpDef Def = new("LSTM", 7, int.MaxValue, static n => new LstmOp(n));

    private LstmOp(NodeInfo node) : base(node, RecurrentCell.Lstm)
    {
    }
}

/// <summary>GRU (opsets 7, 14, 22): inputs X, W, R, B, sequence_lens, initial_h; outputs Y, Y_h.</summary>
internal sealed class GruOp : RecurrentOp
{
    public static readonly OpDef Def = new("GRU", 7, int.MaxValue, static n => new GruOp(n));

    private GruOp(NodeInfo node) : base(node, RecurrentCell.Gru)
    {
    }
}

/// <summary>RNN (opsets 7, 14, 22): inputs X, W, R, B, sequence_lens, initial_h; outputs Y, Y_h.</summary>
internal sealed class RnnOp : RecurrentOp
{
    public static readonly OpDef Def = new("RNN", 7, int.MaxValue, static n => new RnnOp(n));

    private RnnOp(NodeInfo node) : base(node, RecurrentCell.Rnn)
    {
    }
}
