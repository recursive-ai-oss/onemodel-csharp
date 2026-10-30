using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

/// <summary>
/// Conv: N-D convolution (cross-correlation) of X [N, C, spatial...] with W [M, C/group, kernel...] and an optional bias [M], for one
/// or two spatial axes. One-dimensional convolutions run as two-dimensional ones of height 1. The bias is the epilogue of the kernel
/// (one addition per output). Three spatial axes are not supported yet.
/// </summary>
internal sealed class ConvOp : OpKernel
{
    public static readonly OpDef Def = new("Conv", 1, int.MaxValue, static n => new ConvOp(n));

    private readonly WindowAttrs _attrs;
    private readonly int _group;
    private readonly FusedActivation? _activation;

    // Set by ConvPackPass: the weights packed for one instruction set (the kernels' panel layout) and their original shape. The
    // node then has no weights input; the tensor belongs to the graph that made it.
    private readonly Tensor _packed;
    private readonly Shape _packedShape;
    private readonly KernelPath _packedPath;

    private ConvOp(ConvOp source, FusedActivation activation)
    {
        _attrs = source._attrs;
        _group = source._group;
        _activation = activation;
        _packed = source._packed;
        _packedShape = source._packedShape;
        _packedPath = source._packedPath;
    }

    private ConvOp(ConvOp source, Tensor packed, Shape weightsShape, KernelPath path)
    {
        _attrs = source._attrs;
        _group = source._group;
        _activation = source._activation;
        _packed = packed;
        _packedShape = weightsShape;
        _packedPath = path;
    }

    /// <summary>The same convolution reading its weights from <paramref name="packed"/> (see ConvPackPass); input 1 is then absent.</summary>
    internal ConvOp WithPackedWeights(Tensor packed, Shape weightsShape, KernelPath path) => new(this, packed, weightsShape, path);

    /// <summary>The group count.</summary>
    internal int Group => _group;

    /// <summary>The same convolution with <paramref name="activation"/> applied to every output after the bias (see ConvActivationFusionPass).</summary>
    internal ConvOp WithActivation(FusedActivation activation) => new(this, activation);

    /// <summary>True when an activation is already fused in.</summary>
    internal bool HasActivation => _activation is not null;

    private ConvOp(NodeInfo node)
    {
        _attrs = new WindowAttrs(node, hasDilations: true);
        long g = node.GetInt("group", 1);
        if (g <= 0 || g > int.MaxValue)
        {
            throw node.Error($"group must be positive, got {g}");
        }

        _group = (int)g;
    }

    public override bool SupportsExact => true;

    // A packed convolution owns its packed weights (the copy made by WithActivation, which shares them, is dropped by its pass).
    public override void Dispose()
    {
        if (!_packed.IsEmpty)
        {
            _packed.Dispose();
        }

        base.Dispose();
    }

    public override void Infer(InferContext ctx)
    {
        bool packed = !_packed.IsEmpty;
        if (!ctx.HasInput(0) || (!packed && !ctx.HasInput(1)))
        {
            throw ctx.Fail("expects inputs X and W");
        }

        if (ctx.InputType(0) != DType.F32 || (!packed && ctx.InputType(1) != DType.F32) || (ctx.HasInput(2) && ctx.InputType(2) != DType.F32))
        {
            throw ctx.Unsupported($"element types {ctx.InputType(0)}/{ctx.InputType(1)} are not supported yet (float32 only)");
        }

        Shape x = ctx.InputShape(0), w = packed ? _packedShape : ctx.InputShape(1);
        if (x.Rank < 3 || x.Rank != w.Rank)
        {
            throw ctx.Fail($"X {x} and W {w} must have the same rank, at least 3");
        }

        if (x.Rank > 5)
        {
            throw ctx.Unsupported($"{x.Rank - 2} spatial axes are not supported yet (1 to 3)");
        }

        if (w[1] * _group != x[1] || w[0] % _group != 0)
        {
            throw ctx.Fail($"W {w} does not fit X {x} with group {_group}");
        }

        if (ctx.HasInput(2) && (ctx.InputShape(2).Rank != 1 || ctx.InputShape(2)[0] != w[0]))
        {
            throw ctx.Fail($"the bias {ctx.InputShape(2)} must be 1-D with {w[0]} values");
        }

        if (x.Rank == 5)
        {
            Dim3 o = Conv3d.Resolve(_attrs, x, w, _group, ctx).Window.Out;
            ctx.SetOutput(0, DType.F32, Shape.Create(x[0], w[0], o.D, o.H, o.W));
            return;
        }

        Geometry(x, w, ctx, out ConvShape s);
        ctx.SetOutput(0, DType.F32, x.Rank == 3 ? Shape.Create(s.N, s.Cout, s.OutW) : Shape.Create(s.N, s.Cout, s.OutH, s.OutW));
    }

    // Lifts one-dimensional convolutions to height 1 and resolves strides, dilations, pads and output size.
    private void Geometry(Shape x, Shape w, InferContext? ctx, out ConvShape s)
    {
        int r = x.Rank - 2;
        Span<int> kernel = stackalloc int[2], strides = stackalloc int[2], dil = stackalloc int[2], pads = stackalloc int[4];
        ReadOnlySpan<int> wk = w.Dimensions.Slice(2);
        _attrs.Resolve(ctx, r, wk, kernel, strides, dil, pads);
        int h = r == 2 ? x[2] : 1, wd = x[x.Rank - 1];
        int padT = 0, padB = 0, padL, padR;
        int outW;
        if (r == 2)
        {
            padT = pads[0];
            padB = pads[2];
            _ = _attrs.OutputExtent(ctx, 0, h, kernel[0], strides[0], dil[0], false, ref padT, ref padB);
        }

        padL = pads[r - 1];
        padR = pads[(2 * r) - 1];
        outW = _attrs.OutputExtent(ctx, r - 1, wd, kernel[r - 1], strides[r - 1], dil[r - 1], false, ref padL, ref padR);
        s = new ConvShape(
            x[0], x[1], h, wd, w[0], r == 2 ? kernel[0] : 1, kernel[r - 1],
            r == 2 ? strides[0] : 1, strides[r - 1], padT, padL, padB, padR, r == 2 ? dil[0] : 1, dil[r - 1], _group);
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        if (y.Length == 0)
        {
            return;
        }

        ReadOnlySpan<float> bias = run.Inputs.Length > 2 && !run.Inputs[2].IsEmpty ? run.Inputs[2].Data<float>() : default;
        if (!_packed.IsEmpty)
        {
            if (run.Path != _packedPath || run.KernelMode != KernelMode.Fast)
            {
                throw new InvalidOperationException("Weights were packed for a different instruction set or mode than this run uses.");
            }

            Geometry(x.Shape, _packedShape, null, out ConvShape ps);
            Conv.RunPacked(run.Path, in ps, x.Data<float>(), _packed.Data<float>(), y.MutableData<float>(), run.Threads, bias, _activation);
            return;
        }

        Tensor w = run.Inputs[1];
        if (x.Rank == 5)
        {
            Conv3d.Run(Conv3d.Resolve(_attrs, x.Shape, w.Shape, _group, null), x.Data<float>(), w.Data<float>(), bias, y.MutableData<float>(), _activation, run.Path, run.Exact);
            return;
        }

        Geometry(x.Shape, w.Shape, null, out ConvShape s);
        Conv.Run(run.KernelMode, run.Path, in s, x.Data<float>(), w.Data<float>(), y.MutableData<float>(), run.Threads, bias, _activation);
    }
}
