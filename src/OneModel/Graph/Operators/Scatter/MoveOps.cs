using System;
using System.Numerics;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Scatter;

// Data-movement operators of the default domain that rearrange, pad or place elements without arithmetic: every one is exact. They
// move elements of any type through same-size carriers, so one loop serves float, integer and bool tensors alike.

/// <summary>DepthToSpace (mode DCR or CRD) and SpaceToDepth: blocks of channels become blocks of pixels and back, for (N, C, H, W) tensors.</summary>
internal sealed class BlockShuffleOp : OpKernel
{
    public static readonly OpDef DepthToSpace = new("DepthToSpace", 1, int.MaxValue, static n => new BlockShuffleOp(n, true));
    public static readonly OpDef SpaceToDepth = new("SpaceToDepth", 1, int.MaxValue, static n => new BlockShuffleOp(n, false));

    private readonly bool _depthToSpace;
    private readonly int _block;
    private readonly bool _crd;

    private BlockShuffleOp(NodeInfo n, bool depthToSpace)
    {
        _depthToSpace = depthToSpace;
        long b = n.GetInt("blocksize", 0);
        if (b < 1 || b > 1 << 15)
        {
            throw n.Error("attribute 'blocksize' is required and must be at least 1");
        }

        _block = (int)b;
        string mode = n.GetString("mode", "DCR");
        _crd = mode == "CRD";
        if (mode is not ("DCR" or "CRD"))
        {
            throw n.Error($"attribute 'mode' is '{mode}', expected 'DCR' or 'CRD'");
        }
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        Shape x = ctx.InputShape(0);
        if (x.Rank != 4)
        {
            throw ctx.Fail($"the input {x} must be (N, C, H, W)");
        }

        long b = _block;
        if (_depthToSpace)
        {
            if (x[1] % (b * b) != 0)
            {
                throw ctx.Fail($"the channel count {x[1]} is not divisible by blocksize squared");
            }

            ctx.SetOutput(0, ctx.InputType(0), OpUtil.MakeGrownShape(ctx, [x[0], x[1] / (b * b), x[2] * b, x[3] * b], ctx.InputType(0)));
        }
        else
        {
            if (x[2] % b != 0 || x[3] % b != 0)
            {
                throw ctx.Fail($"the spatial size {x[2]}x{x[3]} is not divisible by blocksize {b}");
            }

            ctx.SetOutput(0, ctx.InputType(0), OpUtil.MakeGrownShape(ctx, [x[0], x[1] * b * b, x[2] / b, x[3] / b], ctx.InputType(0)));
        }
    }

    public override void Run(in OpRun run)
    {
        var v = new Visitor { X = run.Inputs[0], Y = run.Outputs[0], B = _block, DepthToSpace = _depthToSpace, Crd = _crd };
        Carriers.Run(v.X.ElementSize, ref v);
    }

    private struct Visitor : ICarrierVisitor
    {
        public Tensor X, Y;
        public int B;
        public bool DepthToSpace, Crd;

        public void Visit<TC>() where TC : unmanaged
        {
            ReadOnlySpan<TC> x = X.RawData<TC>();
            Span<TC> y = Y.RawMutableData<TC>();
            int n = X.Shape[0], c = X.Shape[1], h = X.Shape[2], w = X.Shape[3], b = B;
            if (DepthToSpace)
            {
                // Output (N, C/b^2, H*b, W*b); an output pixel (oh, ow) is block row oh % b, column ow % b of input pixel (oh / b, ow / b).
                int oc = c / (b * b), oh = h * b, ow = w * b;
                for (int i = 0; i < n; i++)
                {
                    for (int k = 0; k < oc; k++)
                    {
                        for (int r = 0; r < oh; r++)
                        {
                            for (int s = 0; s < ow; s++)
                            {
                                int bh = r % b, bw = s % b;
                                int ic = Crd ? (((k * b) + bh) * b) + bw : (((bh * b) + bw) * oc) + k;
                                y[((((i * oc) + k) * oh) + r) * ow + s] = x[((((i * c) + ic) * h) + (r / b)) * w + (s / b)];
                            }
                        }
                    }
                }
            }
            else
            {
                int oc = c * b * b, oh = h / b, ow = w / b;
                for (int i = 0; i < n; i++)
                {
                    for (int k = 0; k < c; k++)
                    {
                        for (int r = 0; r < h; r++)
                        {
                            for (int s = 0; s < w; s++)
                            {
                                int bh = r % b, bw = s % b;
                                int oc2 = (((bh * b) + bw) * c) + k;
                                y[((((i * oc) + oc2) * oh) + (r / b)) * ow + (s / b)] = x[((((i * c) + k) * h) + r) * w + s];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// <summary>EyeLike: a 2-D tensor of the input's shape with ones on the diagonal k and zeros elsewhere, in the type of the dtype attribute or the input.</summary>
internal sealed class EyeLikeOp : OpKernel
{
    public static readonly OpDef Def = new("EyeLike", 9, int.MaxValue, static n => new EyeLikeOp(n));

    private readonly DType? _dtype;
    private readonly long _k;

    private EyeLikeOp(NodeInfo n)
    {
        _k = n.GetInt("k", 0);
        if (n.Has("dtype"))
        {
            long code = n.GetInt("dtype", 0);
            if (code < 0 || code > int.MaxValue || !DTypes.TryFromOnnx((int)code, out DType t))
            {
                throw n.Unsupported($"dtype {DTypes.OnnxName((int)Math.Clamp(code, 0, int.MaxValue))} is not supported");
            }

            _dtype = t;
        }
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        if (ctx.InputShape(0).Rank != 2)
        {
            throw ctx.Fail($"the input {ctx.InputShape(0)} must be 2-D");
        }

        DType t = _dtype ?? ctx.InputType(0);
        if (!TypeDispatch.IsNumber(t))
        {
            throw ctx.Unsupported($"element type {t} is not supported yet");
        }

        ctx.SetOutput(0, t, ctx.InputShape(0));
    }

    public override void Run(in OpRun run)
    {
        var v = new Visitor { Y = run.Outputs[0], K = _k };
        TypeDispatch.Number(v.Y.DType, ref v);
    }

    private struct Visitor : INumberVisitor
    {
        public Tensor Y;
        public long K;

        public void Visit<T>() where T : unmanaged, INumber<T>
        {
            Span<T> y = Y.MutableData<T>();
            y.Clear();
            int rows = Y.Shape[0], cols = Y.Shape[1];
            for (int i = 0; i < rows; i++)
            {
                long j = i + K;
                if (j >= 0 && j < cols)
                {
                    y[(i * cols) + (int)j] = T.One;
                }
            }
        }
    }
}

/// <summary>ReverseSequence: along time_axis, the first sequence_lens[b] elements of each batch entry b (along batch_axis) are reversed.</summary>
internal sealed class ReverseSequenceOp : OpKernel
{
    public static readonly OpDef Def = new("ReverseSequence", 10, int.MaxValue, static n => new ReverseSequenceOp(n));

    private readonly int _batchAxis, _timeAxis;

    private ReverseSequenceOp(NodeInfo n)
    {
        _batchAxis = (int)n.GetInt("batch_axis", 1);
        _timeAxis = (int)n.GetInt("time_axis", 0);
        if (_batchAxis is not (0 or 1) || _timeAxis is not (0 or 1) || _batchAxis == _timeAxis)
        {
            throw n.Error("batch_axis and time_axis must be 0 and 1 in some order");
        }
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 2 || !ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects the input and sequence_lens");
        }

        if (ctx.InputShape(0).Rank < 2 || ctx.InputType(1) != DType.I64 || ctx.InputShape(1).Rank != 1 || ctx.InputShape(1)[0] != ctx.InputShape(0)[_batchAxis])
        {
            throw ctx.Fail("sequence_lens must be an int64 vector with one length per batch entry, and the input at least 2-D");
        }

        ctx.SetOutput(0, ctx.InputType(0), ctx.InputShape(0));
    }

    public override void Run(in OpRun run)
    {
        var v = new Visitor { X = run.Inputs[0], Y = run.Outputs[0], Lens = run.Inputs[1].Data<long>(), BatchAxis = _batchAxis, TimeAxis = _timeAxis };
        Carriers.Run(v.X.ElementSize, ref v);
    }

    private ref struct Visitor : ICarrierVisitor
    {
        public Tensor X, Y;
        public ReadOnlySpan<long> Lens;
        public int BatchAxis, TimeAxis;

        public void Visit<TC>() where TC : unmanaged
        {
            ReadOnlySpan<TC> x = X.RawData<TC>();
            Span<TC> y = Y.RawMutableData<TC>();
            int a0 = X.Shape[0], a1 = X.Shape[1];
            int inner = OpUtil.Product(X.Shape.Dimensions, 2, X.Rank);
            int time = X.Shape[TimeAxis];
            for (int i = 0; i < a0; i++)
            {
                for (int j = 0; j < a1; j++)
                {
                    int b = BatchAxis == 0 ? i : j, t = TimeAxis == 0 ? i : j;
                    long len = Lens[b];
                    if (len < 0 || len > time)
                    {
                        throw new SessionException(SessionErrorKind.InvalidInput, $"ReverseSequence: sequence length {len} is outside [0, {time}]");
                    }

                    int srcT = t < len ? (int)len - 1 - t : t;
                    int si = TimeAxis == 0 ? srcT : i, sj = TimeAxis == 0 ? j : srcT;
                    x.Slice(((si * a1) + sj) * inner, inner).CopyTo(y.Slice(((i * a1) + j) * inner, inner));
                }
            }
        }
    }
}

/// <summary>CenterCropPad: each listed axis is cropped or zero-padded to the size in the shape input, centred (the odd element goes to the end).</summary>
internal sealed class CenterCropPadOp : OpKernel
{
    public static readonly OpDef Def = new("CenterCropPad", 18, int.MaxValue, static n => new CenterCropPadOp(n));

    private readonly System.Collections.Immutable.ImmutableArray<long> _axes;

    private CenterCropPadOp(NodeInfo n)
    {
        _axes = n.GetInts("axes");
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 2 || !ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects the input and the target shape");
        }

        OpUtil.RequireIntegers(ctx, 1, "shape");
        if (!ctx.TryGetInputValue(1, out Tensor shape))
        {
            ctx.RequireRunTimeValues();
            return;
        }

        Shape x = ctx.InputShape(0);
        Span<long> dims = stackalloc long[Shape.MaxRank];
        Target(x, IntVec.Of(shape), dims, ctx);
        ctx.SetOutput(0, ctx.InputType(0), OpUtil.MakeGrownShape(ctx, dims.Slice(0, x.Rank), ctx.InputType(0)));
    }

    private void Target(Shape x, IntVec shape, Span<long> dims, InferContext ctx)
    {
        for (int i = 0; i < x.Rank; i++)
        {
            dims[i] = x[i];
        }

        int count = _axes.IsDefaultOrEmpty ? x.Rank : _axes.Length;
        if (shape.Length != count)
        {
            throw ctx.Fail($"the shape input has {shape.Length} values for {count} axes");
        }

        for (int i = 0; i < count; i++)
        {
            int axis = OpUtil.Axis(ctx, _axes.IsDefaultOrEmpty ? i : _axes[i], x.Rank);
            if (shape[i] < 0)
            {
                throw ctx.Fail("target sizes cannot be negative");
            }

            dims[axis] = shape[i];
        }
    }

    public override void Run(in OpRun run)
    {
        var v = new Visitor { X = run.Inputs[0], Y = run.Outputs[0] };
        Carriers.Run(v.X.ElementSize, ref v);
    }

    private struct Visitor : ICarrierVisitor
    {
        public Tensor X, Y;

        public void Visit<TC>() where TC : unmanaged
        {
            ReadOnlySpan<TC> x = X.RawData<TC>();
            Span<TC> y = Y.RawMutableData<TC>();
            y.Clear();
            int rank = X.Rank;
            if (y.Length == 0 || x.Length == 0)
            {
                return;
            }

            ReadOnlySpan<int> id = X.Shape.Dimensions, od = Y.Shape.Dimensions;
            Span<int> xs = stackalloc int[Shape.MaxRank];
            OpUtil.RowMajor(id, xs);
            Span<int> pos = stackalloc int[Shape.MaxRank];
            for (int p = 0; p < y.Length; p++)
            {
                int src = 0;
                bool inside = true;
                for (int d = 0; d < rank; d++)
                {
                    // Cropping starts (in - out) / 2 into the input; padding leaves (out - in) / 2 empty elements before it.
                    int shift = id[d] >= od[d] ? (id[d] - od[d]) / 2 : -((od[d] - id[d]) / 2);
                    int c = pos[d] + shift;
                    if (c < 0 || c >= id[d])
                    {
                        inside = false;
                        break;
                    }

                    src += c * xs[d];
                }

                if (inside)
                {
                    y[p] = x[src];
                }

                for (int d = rank - 1; d >= 0; d--)
                {
                    if (++pos[d] < od[d])
                    {
                        break;
                    }

                    pos[d] = 0;
                }
            }
        }
    }
}

/// <summary>BitCast: the bytes of each element reinterpreted as the type of the 'to' attribute, which must have the same size.</summary>
internal sealed class BitCastOp : OpKernel
{
    public static readonly OpDef Def = new("BitCast", 26, int.MaxValue, static n => new BitCastOp(n));

    private readonly DType _to;

    private BitCastOp(NodeInfo n)
    {
        long code = n.GetInt("to", 0);
        if (code <= 0 || code > int.MaxValue || !DTypes.TryFromOnnx((int)code, out _to))
        {
            throw n.Unsupported($"bit cast to ONNX element type {DTypes.OnnxName((int)Math.Clamp(code, 0, int.MaxValue))} is not supported");
        }
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        DType t = ctx.InputType(0);
        if (DTypes.SizeOf(t) != DTypes.SizeOf(_to))
        {
            throw ctx.Fail($"cannot reinterpret {t} as {_to}: the element sizes differ");
        }

        ctx.SetOutput(0, _to, ctx.InputShape(0));
    }

    public override void Run(in OpRun run)
    {
        var v = new Visitor { X = run.Inputs[0], Y = run.Outputs[0] };
        Carriers.Run(v.X.ElementSize, ref v);
    }

    private struct Visitor : ICarrierVisitor
    {
        public Tensor X, Y;

        public void Visit<TC>() where TC : unmanaged => X.RawData<TC>().CopyTo(Y.RawMutableData<TC>());
    }
}

/// <summary>
/// TensorScatter (opset 24): a copy of past_cache in which, for batch entry b, the update's positions along the axis are written at
/// write_indices[b] + t (0 when there is no write_indices); mode 'circular' wraps around the end of the cache, 'linear' requires
/// the update to fit.
/// </summary>
internal sealed class TensorScatterOp : OpKernel
{
    public static readonly OpDef Def = new("TensorScatter", 24, int.MaxValue, static n => new TensorScatterOp(n));

    private readonly long _axis;
    private readonly bool _circular;

    private TensorScatterOp(NodeInfo n)
    {
        _axis = n.GetInt("axis", -2);
        string mode = n.GetString("mode", "linear");
        _circular = mode == "circular";
        if (mode is not ("linear" or "circular"))
        {
            throw n.Error($"attribute 'mode' is '{mode}', expected 'linear' or 'circular'");
        }
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 2 || !ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects past_cache and update");
        }

        Shape c = ctx.InputShape(0), u = ctx.InputShape(1);
        if (ctx.InputType(0) != ctx.InputType(1) || c.Rank != u.Rank || c.Rank < 2)
        {
            throw ctx.Fail("past_cache and update must have the same type and rank (at least 2)");
        }

        int axis = OpUtil.Axis(ctx, _axis, c.Rank);
        if (axis == 0)
        {
            throw ctx.Fail("the axis cannot be the batch axis");
        }

        for (int i = 0; i < c.Rank; i++)
        {
            if (i != axis && c[i] != u[i])
            {
                throw ctx.Fail($"update {u} and past_cache {c} differ in axis {i}");
            }
        }

        if (ctx.HasInput(2) && (ctx.InputType(2) != DType.I64 || ctx.InputShape(2).Rank != 1 || ctx.InputShape(2)[0] != c[0]))
        {
            throw ctx.Fail("write_indices must be an int64 vector with one entry per batch element");
        }

        ctx.SetOutput(0, ctx.InputType(0), c);
    }

    public override void Run(in OpRun run)
    {
        var v = new Visitor
        {
            Cache = run.Inputs[0], Update = run.Inputs[1], Out = run.Outputs[0], Circular = _circular,
            Axis = (int)(_axis < 0 ? _axis + run.Inputs[0].Rank : _axis),
            Write = run.Inputs.Length > 2 && !run.Inputs[2].IsEmpty ? run.Inputs[2].Data<long>() : default,
        };
        Carriers.Run(v.Cache.ElementSize, ref v);
    }

    private ref struct Visitor : ICarrierVisitor
    {
        public Tensor Cache, Update, Out;
        public ReadOnlySpan<long> Write;
        public int Axis;
        public bool Circular;

        public void Visit<TC>() where TC : unmanaged
        {
            Cache.RawData<TC>().CopyTo(Out.RawMutableData<TC>());
            Span<TC> y = Out.RawMutableData<TC>();
            ReadOnlySpan<TC> u = Update.RawData<TC>();
            ReadOnlySpan<int> d = Cache.Shape.Dimensions;
            int mid = OpUtil.Product(d, 1, Axis), len = d[Axis], inner = OpUtil.Product(d, Axis + 1, d.Length);
            int ulen = Update.Shape[Axis], batch = d[0];
            for (int b = 0; b < batch; b++)
            {
                long start = Write.IsEmpty ? 0 : Write[b];
                if (start < 0 || start >= len || (!Circular && start + ulen > len))
                {
                    throw new SessionException(SessionErrorKind.InvalidInput, $"TensorScatter: write index {start} with {ulen} update positions does not fit a cache of {len}");
                }

                for (int m = 0; m < mid; m++)
                {
                    int outer = (b * mid) + m;
                    for (int t = 0; t < ulen; t++)
                    {
                        long pos = Circular ? (start + t) % len : start + t;
                        u.Slice(((outer * ulen) + t) * inner, inner).CopyTo(y.Slice(((outer * len) + (int)pos) * inner, inner));
                    }
                }
            }
        }
    }
}
