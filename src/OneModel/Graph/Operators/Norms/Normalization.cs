using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

// Normalisation operators. The arithmetic is in Kernels/Exact/NormKernels.cs; this file holds the shape rules and the chunking. Fast and
// exact mode run the same code (single IEEE operations in a fixed order), so all of these are exact. docs/ops-norms.md has the rules.

/// <summary>Shared pieces of the normalisation operators.</summary>
internal static class NormCommon
{
    /// <summary>Elements of work per chunk; the rows per chunk follow from the row length alone.</summary>
    public const int ChunkElements = 16384;

    public static int RowsPerChunk(int cols) => Math.Max(1, ChunkElements / Math.Max(1, cols));

    public static void RequireFloat(InferContext ctx, int i)
    {
        DType t = ctx.InputType(i);
        if (t != DType.F32)
        {
            throw ctx.Unsupported($"element type {t} is not supported yet; only float32 is");
        }
    }

    public static int Axis(InferContext ctx, long axis, int rank)
    {
        if (axis < -rank || axis >= rank)
        {
            throw ctx.Fail($"axis {axis} is outside [-{rank}, {rank})");
        }

        return (int)(axis < 0 ? axis + rank : axis);
    }

    /// <summary>True when <paramref name="src"/> broadcasts to <paramref name="dst"/> (right-aligned; missing leading dimensions count as one).</summary>
    public static bool Broadcastable(ReadOnlySpan<int> src, ReadOnlySpan<int> dst)
    {
        if (src.Length > dst.Length)
        {
            for (int i = 0; i < src.Length - dst.Length; i++)
            {
                if (src[i] != 1)
                {
                    return false;
                }
            }
        }

        int n = Math.Min(src.Length, dst.Length);
        for (int i = 1; i <= n; i++)
        {
            int s = src[src.Length - i];
            if (s != dst[dst.Length - i] && s != 1)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// The values of <paramref name="t"/> as one array of the normalized extent: itself when it already has that many elements,
    /// otherwise a broadcast copy in scratch.
    /// </summary>
    public static Tensor Expand(in OpRun run, Tensor t, ReadOnlySpan<int> normalized, int cols)
    {
        if (t.Length == cols)
        {
            return t;
        }

        Tensor e = run.Scratch(DType.F32, Shape.Create(cols));
        ReadOnlySpan<float> src = t.Data<float>();
        Span<float> dst = e.MutableData<float>();
        ReadOnlySpan<int> sd = t.Shape.Dimensions;
        for (int o = 0; o < cols; o++)
        {
            int rem = o, off = 0, stride = 1;
            for (int d = 1; d <= sd.Length; d++)
            {
                int dstDim = normalized[normalized.Length - d];
                int i = rem % dstDim;
                rem /= dstDim;
                int s = sd[sd.Length - d];
                off += (s == 1 ? 0 : i) * stride;
                stride *= s;
            }

            dst[o] = src[off];
        }

        return e;
    }
}

/// <summary>Normalises <c>rows</c> rows of <c>cols</c> elements: LayerNormalization and RMSNormalization.</summary>
internal readonly struct RowNormWork : IChunkWork
{
    public required Tensor X { get; init; }

    public required Tensor Y { get; init; }

    public Tensor Scale { get; init; }

    public Tensor Bias { get; init; }

    public Tensor Mean { get; init; }

    public Tensor InvStdDev { get; init; }

    public required int Rows { get; init; }

    public required int Cols { get; init; }

    public required bool Rms { get; init; }

    public required float Epsilon { get; init; }

    public required VecIsa Isa { get; init; }

    public int Chunks => (Rows + NormCommon.RowsPerChunk(Cols) - 1) / NormCommon.RowsPerChunk(Cols);

    public void RunChunk(int chunk)
    {
        int per = NormCommon.RowsPerChunk(Cols);
        int r0 = chunk * per, r1 = Math.Min(Rows, r0 + per);
        ReadOnlySpan<float> bias = Bias.IsEmpty ? default : Bias.Data<float>();
        NormKernels.Rows(Isa, X.Data<float>(), Y.MutableData<float>(), r0, r1, Cols, Rms, Epsilon, Scale.Data<float>(), 1f, bias,
            !Bias.IsEmpty, 0f, Mean.IsEmpty ? default : Mean.MutableData<float>(), InvStdDev.IsEmpty ? default : InvStdDev.MutableData<float>());
    }
}

/// <summary>LayerNormalization (opset 17): <c>Y = (X - mean) / sqrt(var + epsilon) * Scale + B</c> over the axes from <c>axis</c> on.</summary>
internal sealed class LayerNormalizationOp : OpKernel
{
    public static readonly OpDef Def = new("LayerNormalization", 17, int.MaxValue, static n => new LayerNormalizationOp(n));

    private readonly long _axis;
    private readonly float _epsilon;
    private readonly bool _wantMean, _wantInv;

    private LayerNormalizationOp(NodeInfo node)
    {
        _axis = node.GetInt("axis", -1);
        _epsilon = node.GetFloat("epsilon", 1e-5f);
        _wantMean = node.Ir.OutputNames.Length > 1 && node.Ir.OutputNames[1].Length > 0;
        _wantInv = node.Ir.OutputNames.Length > 2 && node.Ir.OutputNames[2].Length > 0;
    }

    public override bool SupportsExact => true;

    public override int InPlaceInput => 0;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects X and Scale (and optionally B)");
        }

        NormCommon.RequireFloat(ctx, 0);
        NormCommon.RequireFloat(ctx, 1);
        if (ctx.HasInput(2))
        {
            NormCommon.RequireFloat(ctx, 2);
        }

        Shape x = ctx.InputShape(0);
        if (x.Rank == 0)
        {
            throw ctx.Fail("X must have at least one dimension");
        }

        int axis = NormCommon.Axis(ctx, _axis, x.Rank);
        ReadOnlySpan<int> normalized = x.Dimensions.Slice(axis);
        if (!NormCommon.Broadcastable(ctx.InputShape(1).Dimensions, normalized) ||
            (ctx.HasInput(2) && !NormCommon.Broadcastable(ctx.InputShape(2).Dimensions, normalized)))
        {
            throw ctx.Fail($"Scale and B must broadcast to the normalized shape {Shape.Create(normalized)}");
        }

        ctx.SetOutput(0, DType.F32, x);
        Span<int> stat = stackalloc int[Shape.MaxRank];
        x.Dimensions.CopyTo(stat);
        stat.Slice(axis, x.Rank - axis).Fill(1);
        Shape statShape = Shape.Create(stat.Slice(0, x.Rank));
        if (_wantMean)
        {
            ctx.SetOutput(1, DType.F32, statShape);
        }

        if (_wantInv)
        {
            ctx.SetOutput(2, DType.F32, statShape);
        }
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0];
        int axis = (int)(_axis < 0 ? _axis + x.Rank : _axis);
        int rows = 1, cols = 1;
        for (int i = 0; i < x.Rank; i++)
        {
            if (i < axis)
            {
                rows *= x.Shape[i];
            }
            else
            {
                cols *= x.Shape[i];
            }
        }

        if (rows == 0 || cols == 0)
        {
            return;
        }

        ReadOnlySpan<int> normalized = x.Shape.Dimensions.Slice(axis);
        var work = new RowNormWork
        {
            X = x,
            Y = run.Outputs[0],
            Scale = NormCommon.Expand(run, run.Inputs[1], normalized, cols),
            Bias = run.Inputs.Length > 2 && !run.Inputs[2].IsEmpty ? NormCommon.Expand(run, run.Inputs[2], normalized, cols) : default,
            Mean = run.Outputs.Length > 1 ? run.Outputs[1] : default,
            InvStdDev = run.Outputs.Length > 2 ? run.Outputs[2] : default,
            Rows = rows,
            Cols = cols,
            Rms = false,
            Epsilon = _epsilon,
            Isa = RowIsa.Of(run.Path),
        };
        Chunked.Run(work, work.Chunks, run.Threads);
    }
}

/// <summary>RMSNormalization (opset 23): <c>Y = X / sqrt(mean(X^2) + epsilon) * Scale</c> over the axes from <c>axis</c> on.</summary>
internal sealed class RmsNormalizationOp : OpKernel
{
    public static readonly OpDef Def = new("RMSNormalization", 23, int.MaxValue, static n => new RmsNormalizationOp(n));

    private readonly long _axis;
    private readonly float _epsilon;

    private RmsNormalizationOp(NodeInfo node)
    {
        _axis = node.GetInt("axis", -1);
        _epsilon = node.GetFloat("epsilon", 1e-5f);
    }

    public override bool SupportsExact => true;

    public override int InPlaceInput => 0;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects X and Scale");
        }

        NormCommon.RequireFloat(ctx, 0);
        NormCommon.RequireFloat(ctx, 1);
        Shape x = ctx.InputShape(0);
        if (x.Rank == 0)
        {
            throw ctx.Fail("X must have at least one dimension");
        }

        int axis = NormCommon.Axis(ctx, _axis, x.Rank);
        if (!NormCommon.Broadcastable(ctx.InputShape(1).Dimensions, x.Dimensions.Slice(axis)))
        {
            throw ctx.Fail($"Scale must broadcast to the normalized shape {Shape.Create(x.Dimensions.Slice(axis))}");
        }

        ctx.SetOutput(0, DType.F32, x);
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0];
        int axis = (int)(_axis < 0 ? _axis + x.Rank : _axis);
        int rows = 1, cols = 1;
        for (int i = 0; i < x.Rank; i++)
        {
            if (i < axis)
            {
                rows *= x.Shape[i];
            }
            else
            {
                cols *= x.Shape[i];
            }
        }

        if (rows == 0 || cols == 0)
        {
            return;
        }

        var work = new RowNormWork
        {
            X = x,
            Y = run.Outputs[0],
            Scale = NormCommon.Expand(run, run.Inputs[1], x.Shape.Dimensions.Slice(axis), cols),
            Rows = rows,
            Cols = cols,
            Rms = true,
            Epsilon = _epsilon,
            Isa = RowIsa.Of(run.Path),
        };
        Chunked.Run(work, work.Chunks, run.Threads);
    }
}

/// <summary>
/// Shared work of InstanceNormalization and GroupNormalization: every row is one (sample, channel) or (sample, group) block of a
/// contiguous input, with the scale and bias given per channel (expanded across the spatial positions) or per row.
/// </summary>
internal readonly struct BlockNormWork : IChunkWork
{
    public required Tensor X { get; init; }

    public required Tensor Y { get; init; }

    public required Tensor Scale { get; init; }

    public required Tensor Bias { get; init; }

    public required int Rows { get; init; }

    public required int Cols { get; init; }

    /// <summary>Scale and bias are indexed by <c>row % Period</c> and apply as scalars to the whole row.</summary>
    public required int Period { get; init; }

    public required float Epsilon { get; init; }

    public required VecIsa Isa { get; init; }

    public int Chunks => (Rows + NormCommon.RowsPerChunk(Cols) - 1) / NormCommon.RowsPerChunk(Cols);

    public void RunChunk(int chunk)
    {
        int per = NormCommon.RowsPerChunk(Cols);
        int r0 = chunk * per, r1 = Math.Min(Rows, r0 + per);
        ReadOnlySpan<float> x = X.Data<float>();
        Span<float> y = Y.MutableData<float>();
        ReadOnlySpan<float> scale = Scale.Data<float>(), bias = Bias.Data<float>();
        for (int r = r0; r < r1; r++)
        {
            NormKernels.Rows(Isa, x, y, r, r + 1, Cols, false, Epsilon, default, scale[r % Period], default, true, bias[r % Period], default, default);
        }
    }
}

/// <summary>InstanceNormalization (opset 6+): each (sample, channel) plane is normalised on its own; Scale and B are per channel.</summary>
internal sealed class InstanceNormalizationOp : OpKernel
{
    public static readonly OpDef Def = new("InstanceNormalization", 6, int.MaxValue, static n => new InstanceNormalizationOp(n));

    private readonly float _epsilon;

    private InstanceNormalizationOp(NodeInfo node) => _epsilon = node.GetFloat("epsilon", 1e-5f);

    public override bool SupportsExact => true;

    public override int InPlaceInput => 0;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0) || !ctx.HasInput(1) || !ctx.HasInput(2))
        {
            throw ctx.Fail("expects input, scale and B");
        }

        for (int i = 0; i < 3; i++)
        {
            NormCommon.RequireFloat(ctx, i);
        }

        Shape x = ctx.InputShape(0);
        if (x.Rank < 2)
        {
            throw ctx.Fail("the input must have at least two dimensions (N and C)");
        }

        if (ctx.InputShape(1) != Shape.Create(x[1]) || ctx.InputShape(2) != Shape.Create(x[1]))
        {
            throw ctx.Fail($"scale and B must have shape [{x[1]}], the channel count");
        }

        ctx.SetOutput(0, DType.F32, x);
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0];
        int n = x.Shape[0], c = x.Shape[1];
        int spatial = 1;
        for (int i = 2; i < x.Rank; i++)
        {
            spatial *= x.Shape[i];
        }

        if (n == 0 || c == 0 || spatial == 0)
        {
            return;
        }

        var work = new BlockNormWork
        {
            X = x,
            Y = run.Outputs[0],
            Scale = run.Inputs[1],
            Bias = run.Inputs[2],
            Rows = n * c,
            Cols = spatial,
            Period = c,
            Epsilon = _epsilon,
            Isa = RowIsa.Of(run.Path),
        };
        Chunked.Run(work, work.Chunks, run.Threads);
    }
}

/// <summary>
/// GroupNormalization. Opset 18 to 20: Scale and B have one value per group. Opset 21 and later: one value per channel (the
/// definition was corrected in 21). Each (sample, group) block of <c>C / num_groups</c> channels times the spatial positions is
/// normalised together.
/// </summary>
internal sealed class GroupNormalizationOp : OpKernel
{
    public static readonly OpDef DefPerGroup = new("GroupNormalization", 18, 20, static n => new GroupNormalizationOp(n, true));

    public static readonly OpDef DefPerChannel = new("GroupNormalization", 21, int.MaxValue, static n => new GroupNormalizationOp(n, false));

    private readonly float _epsilon;
    private readonly long _groups;
    private readonly bool _perGroup;

    private GroupNormalizationOp(NodeInfo node, bool perGroup)
    {
        _epsilon = node.GetFloat("epsilon", 1e-5f);
        _groups = node.GetInt("num_groups", 0);
        _perGroup = perGroup;
        if (_groups <= 0)
        {
            throw node.Error("attribute num_groups is required and must be positive");
        }
    }

    public override bool SupportsExact => true;

    public override int InPlaceInput => 0;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0) || !ctx.HasInput(1) || !ctx.HasInput(2))
        {
            throw ctx.Fail("expects X, scale and bias");
        }

        for (int i = 0; i < 3; i++)
        {
            NormCommon.RequireFloat(ctx, i);
        }

        Shape x = ctx.InputShape(0);
        if (x.Rank < 2)
        {
            throw ctx.Fail("X must have at least two dimensions (N and C)");
        }

        if (x[1] % _groups != 0)
        {
            throw ctx.Fail($"the channel count {x[1]} is not divisible by num_groups {_groups}");
        }

        Shape want = Shape.Create(_perGroup ? (int)_groups : x[1]);
        if (ctx.InputShape(1) != want || ctx.InputShape(2) != want)
        {
            throw ctx.Fail($"scale and bias must have shape {want}");
        }

        ctx.SetOutput(0, DType.F32, x);
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0];
        int n = x.Shape[0], c = x.Shape[1];
        int groups = (int)_groups;
        int spatial = 1;
        for (int i = 2; i < x.Rank; i++)
        {
            spatial *= x.Shape[i];
        }

        if (n == 0 || c == 0 || spatial == 0)
        {
            return;
        }

        int cpg = c / groups;
        int cols = cpg * spatial;
        Tensor scaleRow = default, biasRow = default;
        if (!_perGroup)
        {
            // Each group's per-channel parameters, spread across the spatial positions so a block is one row of scales and biases.
            scaleRow = run.Scratch(DType.F32, Shape.Create(groups, cols));
            biasRow = run.Scratch(DType.F32, Shape.Create(groups, cols));
            Span<float> s = scaleRow.MutableData<float>(), b = biasRow.MutableData<float>();
            ReadOnlySpan<float> sc = run.Inputs[1].Data<float>(), bi = run.Inputs[2].Data<float>();
            for (int g = 0; g < groups; g++)
            {
                for (int k = 0; k < cpg; k++)
                {
                    s.Slice((g * cols) + (k * spatial), spatial).Fill(sc[(g * cpg) + k]);
                    b.Slice((g * cols) + (k * spatial), spatial).Fill(bi[(g * cpg) + k]);
                }
            }

            GroupWork gw = new()
            {
                X = x,
                Y = run.Outputs[0],
                ScaleRows = scaleRow,
                BiasRows = biasRow,
                Rows = n * groups,
                Groups = groups,
                Cols = cols,
                Epsilon = _epsilon,
                Isa = RowIsa.Of(run.Path),
            };
            Chunked.Run(gw, gw.Chunks, run.Threads);
            return;
        }

        var work = new BlockNormWork
        {
            X = x,
            Y = run.Outputs[0],
            Scale = run.Inputs[1],
            Bias = run.Inputs[2],
            Rows = n * groups,
            Cols = cols,
            Period = groups,
            Epsilon = _epsilon,
            Isa = RowIsa.Of(run.Path),
        };
        Chunked.Run(work, work.Chunks, run.Threads);
    }

    /// <summary>Per-channel parameters, one expanded row of scale and bias per group.</summary>
    private readonly struct GroupWork : IChunkWork
    {
        public required Tensor X { get; init; }

        public required Tensor Y { get; init; }

        public required Tensor ScaleRows { get; init; }

        public required Tensor BiasRows { get; init; }

        public required int Rows { get; init; }

        public required int Groups { get; init; }

        public required int Cols { get; init; }

        public required float Epsilon { get; init; }

        public required VecIsa Isa { get; init; }

        public int Chunks => (Rows + NormCommon.RowsPerChunk(Cols) - 1) / NormCommon.RowsPerChunk(Cols);

        public void RunChunk(int chunk)
        {
            int per = NormCommon.RowsPerChunk(Cols);
            int r0 = chunk * per, r1 = Math.Min(Rows, r0 + per);
            ReadOnlySpan<float> x = X.Data<float>();
            Span<float> y = Y.MutableData<float>();
            ReadOnlySpan<float> s = ScaleRows.Data<float>(), b = BiasRows.Data<float>();
            for (int r = r0; r < r1; r++)
            {
                int g = r % Groups;
                NormKernels.Rows(Isa, x, y, r, r + 1, Cols, false, Epsilon, s.Slice(g * Cols, Cols), 1f, b.Slice(g * Cols, Cols), true, 0f, default, default);
            }
        }
    }
}
