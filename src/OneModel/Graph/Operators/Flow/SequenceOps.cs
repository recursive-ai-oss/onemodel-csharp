using System;
using System.Collections.Generic;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Flow;

/// <summary>Base of the sequence and optional operators: their results are packed values whose size depends on the data.</summary>
internal abstract class FlowKernel : OpKernel
{
    protected FlowKernel(NodeInfo info) => Info = info;

    protected NodeInfo Info { get; }

    public sealed override bool SupportsExact => true;

    public sealed override bool OutputsDataDependent => true;

    public sealed override void Infer(InferContext ctx) => throw new InvalidOperationException($"{Info.OpType} is shaped when it runs.");

    /// <summary>The position input (an optional int32 or int64 scalar) as an index into <paramref name="count"/> elements; negative counts from the end.</summary>
    protected int Position(in OpRun run, int input, int count, int fallback, bool insert)
    {
        if (input >= run.Inputs.Length || run.Inputs[input].IsEmpty)
        {
            return fallback;
        }

        long p = FlowUtil.ScalarInt(run.Inputs[input]);
        long max = insert ? count : count - 1;
        if (p < -count || p > max)
        {
            throw new SessionException(SessionErrorKind.InvalidInput, $"{Info.OpType}: position {p} is outside [{-count}, {max}] for a sequence of {count}");
        }

        return (int)(p < 0 ? p + count : p);
    }
}

internal sealed class SequenceEmptyOp : FlowKernel
{
    public static readonly OpDef Def = new("SequenceEmpty", 11, int.MaxValue, static n => new SequenceEmptyOp(n));

    private SequenceEmptyOp(NodeInfo n) : base(n)
    {
    }

    public override void Run(in OpRun run) => FlowUtil.EmitPacked(run, 0, PackedValue.Sequence([]));
}

internal sealed class SequenceConstructOp : FlowKernel
{
    public static readonly OpDef Def = new("SequenceConstruct", 11, int.MaxValue, static n => new SequenceConstructOp(n));

    private SequenceConstructOp(NodeInfo n) : base(n)
    {
    }

    public override void Run(in OpRun run) => FlowUtil.EmitPacked(run, 0, PackedValue.Sequence(run.Inputs.Slice(0, Info.InputCount)));
}

internal sealed class SequenceLengthOp : FlowKernel
{
    public static readonly OpDef Def = new("SequenceLength", 11, int.MaxValue, static n => new SequenceLengthOp(n));

    private SequenceLengthOp(NodeInfo n) : base(n)
    {
    }

    public override void Run(in OpRun run)
    {
        Tensor[] items = FlowUtil.Sequence(run.Inputs[0]);
        long count = items.Length;
        FlowUtil.DisposeAll(items);
        run.AllocateOutput(0, DType.I64, FlowUtil.ScalarShape).MutableData<long>()[0] = count;
    }
}

internal sealed class SequenceAtOp : FlowKernel
{
    public static readonly OpDef Def = new("SequenceAt", 11, int.MaxValue, static n => new SequenceAtOp(n));

    private SequenceAtOp(NodeInfo n) : base(n)
    {
    }

    public override void Run(in OpRun run)
    {
        Tensor[] items = FlowUtil.Sequence(run.Inputs[0]);
        try
        {
            FlowUtil.Emit(run, 0, items[Position(run, 1, items.Length, 0, insert: false)]);
        }
        finally
        {
            FlowUtil.DisposeAll(items);
        }
    }
}

internal sealed class SequenceInsertOp : FlowKernel
{
    public static readonly OpDef Def = new("SequenceInsert", 11, int.MaxValue, static n => new SequenceInsertOp(n));

    private SequenceInsertOp(NodeInfo n) : base(n)
    {
    }

    public override void Run(in OpRun run)
    {
        Tensor[] items = FlowUtil.Sequence(run.Inputs[0]);
        try
        {
            int at = Position(run, 2, items.Length, items.Length, insert: true);
            var list = new List<Tensor>(items);
            list.Insert(at, run.Inputs[1]);
            FlowUtil.EmitPacked(run, 0, PackedValue.Sequence(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(list)));
        }
        finally
        {
            FlowUtil.DisposeAll(items);
        }
    }
}

internal sealed class SequenceEraseOp : FlowKernel
{
    public static readonly OpDef Def = new("SequenceErase", 11, int.MaxValue, static n => new SequenceEraseOp(n));

    private SequenceEraseOp(NodeInfo n) : base(n)
    {
    }

    public override void Run(in OpRun run)
    {
        Tensor[] items = FlowUtil.Sequence(run.Inputs[0]);
        try
        {
            int at = Position(run, 1, items.Length, items.Length - 1, insert: false);
            var list = new List<Tensor>(items);
            list.RemoveAt(at);
            FlowUtil.EmitPacked(run, 0, PackedValue.Sequence(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(list)));
        }
        finally
        {
            FlowUtil.DisposeAll(items);
        }
    }
}

/// <summary>SplitToSequence: cuts a tensor along an axis into a sequence; without <c>split</c> every element becomes a slice (keepdims keeps the axis with extent 1).</summary>
internal sealed class SplitToSequenceOp : FlowKernel
{
    public static readonly OpDef Def = new("SplitToSequence", 11, int.MaxValue, static n => new SplitToSequenceOp(n));

    private readonly long _axis;
    private readonly bool _keepDims;

    private SplitToSequenceOp(NodeInfo n) : base(n)
    {
        _axis = n.GetInt("axis", 0);
        _keepDims = n.GetInt("keepdims", 1) != 0;
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0];
        int axis = FlowUtil.Axis(_axis, x.Rank, "SplitToSequence"), dim = x.Shape[axis];
        var parts = new List<Tensor>();
        try
        {
            if (run.Inputs.Length < 2 || run.Inputs[1].IsEmpty)
            {
                for (int i = 0; i < dim; i++)
                {
                    parts.Add(FlowUtil.Narrow(x, axis, i, 1, drop: !_keepDims));
                }
            }
            else
            {
                Tensor split = run.Inputs[1];
                if (split.Rank == 0)
                {
                    long s = FlowUtil.ScalarInt(split);
                    if (s <= 0)
                    {
                        throw new SessionException(SessionErrorKind.InvalidInput, "SplitToSequence: the split size must be positive");
                    }

                    for (long at = 0; at < dim; at += s)
                    {
                        parts.Add(FlowUtil.Narrow(x, axis, (int)at, (int)Math.Min(s, dim - at), drop: false));
                    }
                }
                else
                {
                    long sum = 0, at = 0;
                    for (int i = 0; i < split.Length; i++)
                    {
                        long s = split.DType == DType.I64 ? split.Data<long>()[i] : split.Data<int>()[i];
                        if (s < 0 || at + s > dim)
                        {
                            throw new SessionException(SessionErrorKind.InvalidInput, "SplitToSequence: the split sizes do not fit the axis");
                        }

                        parts.Add(FlowUtil.Narrow(x, axis, (int)at, (int)s, drop: false));
                        at += s;
                        sum += s;
                    }

                    if (sum != dim)
                    {
                        throw new SessionException(SessionErrorKind.InvalidInput, "SplitToSequence: the split sizes do not add up to the axis length");
                    }
                }
            }

            FlowUtil.EmitPacked(run, 0, PackedValue.Sequence(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(parts)));
        }
        finally
        {
            FlowUtil.DisposeAll(parts);
        }
    }
}

/// <summary>ConcatFromSequence: joins the elements along an axis, or (new_axis = 1) stacks them on a new one.</summary>
internal sealed class ConcatFromSequenceOp : FlowKernel
{
    public static readonly OpDef Def = new("ConcatFromSequence", 11, int.MaxValue, static n => new ConcatFromSequenceOp(n));

    private readonly long _axis;
    private readonly bool _newAxis;

    private ConcatFromSequenceOp(NodeInfo n) : base(n)
    {
        if (!n.Has("axis"))
        {
            throw n.Error("attribute 'axis' is required");
        }

        _axis = n.GetInt("axis", 0);
        _newAxis = n.GetInt("new_axis", 0) != 0;
    }

    public override void Run(in OpRun run)
    {
        Tensor[] items = FlowUtil.Sequence(run.Inputs[0]);
        try
        {
            if (items.Length == 0)
            {
                throw new SessionException(SessionErrorKind.InvalidInput, "ConcatFromSequence: the sequence is empty");
            }

            Tensor first = items[0];
            int rank = first.Rank + (_newAxis ? 1 : 0);
            int axis = FlowUtil.Axis(_axis, rank, "ConcatFromSequence");
            // Each element viewed with the new axis (extent 1) in place.
            Span<int> vd = stackalloc int[Shape.MaxRank];
            Span<int> od = stackalloc int[Shape.MaxRank];
            long total = 0;
            foreach (Tensor t in items)
            {
                if (t.DType != first.DType || t.Rank != first.Rank)
                {
                    throw new SessionException(SessionErrorKind.InvalidInput, "ConcatFromSequence: the elements differ in type or rank");
                }

                Virtual(t, axis, vd);
                total += vd[axis];
                for (int d = 0; d < rank; d++)
                {
                    Virtual(first, axis, od);
                    if (d != axis && vd[d] != od[d])
                    {
                        throw new SessionException(SessionErrorKind.InvalidInput, "ConcatFromSequence: the elements differ outside the concatenation axis");
                    }
                }
            }

            Virtual(first, axis, od);
            od[axis] = (int)total;
            Tensor y = run.AllocateOutput(0, first.DType, Shape.Create(od.Slice(0, rank)));
            int outer = FlowUtil.Product(od, 0, axis), inner = FlowUtil.Product(od, axis + 1, rank), es = first.ElementSize;
            Span<byte> o = PackedValue.MutableBytes(y);
            int at = 0;
            foreach (Tensor t in items)
            {
                Virtual(t, axis, vd);
                int block = vd[axis] * inner * es;
                if (block > 0)
                {
                    ReadOnlySpan<byte> s = PackedValue.Bytes(t);
                    for (int a = 0; a < outer; a++)
                    {
                        s.Slice(a * block, block).CopyTo(o.Slice(((a * (int)total * inner) + (at * inner)) * es, block));
                    }
                }

                at += vd[axis];
            }
        }
        finally
        {
            FlowUtil.DisposeAll(items);
        }

        void Virtual(Tensor t, int axis, Span<int> dims)
        {
            int r = 0;
            for (int d = 0; d <= t.Rank; d++)
            {
                if (_newAxis && d == axis)
                {
                    dims[r++] = 1;
                }

                if (d < t.Rank)
                {
                    dims[r++] = t.Shape[d];
                }
            }
        }
    }
}

/// <summary>SequenceMap: applies the body to each element of the input sequences (other inputs are passed whole each time); every body output becomes a sequence.</summary>
internal sealed class SequenceMapOp : OpKernel, ISubgraphKernel
{
    public static readonly OpDef Def = new("SequenceMap", 17, int.MaxValue, static n => new SequenceMapOp(n));

    private readonly NodeInfo _info;
    private NestedGraph? _body;

    private SequenceMapOp(NodeInfo n) => _info = n;

    public override bool SupportsExact => true;

    public override bool OutputsDataDependent => true;

    public void Bind(SubgraphBinder b)
    {
        IrGraph ir = IfOp.Body(_info, "body");
        if (ir.Inputs.Length != _info.InputCount)
        {
            throw _info.Error("the body must take one input per node input");
        }

        _body = b.Compile(ir);
    }

    public override void Infer(InferContext ctx) => throw new InvalidOperationException("SequenceMap is shaped when it runs.");

    public override void Run(in OpRun run)
    {
        int n = _info.InputCount, k = _body!.OutputCount;
        var seqs = new Tensor[n][];
        var isSeq = new bool[n];
        var results = new List<Tensor>[k];
        for (int q = 0; q < k; q++)
        {
            results[q] = [];
        }

        try
        {
            int len = -1;
            for (int i = 0; i < n; i++)
            {
                isSeq[i] = i == 0 || FlowUtil.IsSequenceInput(_info, i, run.Inputs[i]);
                if (!isSeq[i])
                {
                    continue;
                }

                seqs[i] = FlowUtil.Sequence(run.Inputs[i]);
                if (len >= 0 && len != seqs[i].Length)
                {
                    throw new SessionException(SessionErrorKind.InvalidInput, "SequenceMap: the input sequences differ in length");
                }

                len = seqs[i].Length;
            }

            var declared = new Tensor[n];
            for (int l = 0; l < len; l++)
            {
                for (int i = 0; i < n; i++)
                {
                    declared[i] = isSeq[i] ? seqs[i][l] : run.Inputs[i];
                }

                Tensor[] outs = _body.Run(declared, run.Inputs.Slice(n));
                for (int q = 0; q < k; q++)
                {
                    results[q].Add(outs[q]);
                }
            }

            for (int q = 0; q < k; q++)
            {
                FlowUtil.EmitPacked(run, q, PackedValue.Sequence(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(results[q])));
            }
        }
        finally
        {
            foreach (Tensor[]? s in seqs)
            {
                if (s is not null)
                {
                    FlowUtil.DisposeAll(s);
                }
            }

            foreach (List<Tensor> r in results)
            {
                FlowUtil.DisposeAll(r);
            }
        }
    }

    public override void Dispose()
    {
        _body?.Dispose();
        base.Dispose();
    }
}

/// <summary>Optional: wraps its input (a tensor or a sequence) as an optional, or makes an empty one when the input is omitted.</summary>
internal sealed class OptionalOp : FlowKernel
{
    public static readonly OpDef Def = new("Optional", 15, int.MaxValue, static n => new OptionalOp(n));

    private OptionalOp(NodeInfo n) : base(n)
    {
    }

    public override void Run(in OpRun run)
    {
        if (run.Inputs.Length == 0 || run.Inputs[0].IsEmpty)
        {
            FlowUtil.EmitPacked(run, 0, PackedValue.EmptyOptional());
            return;
        }

        Tensor x = run.Inputs[0];
        bool seq = FlowUtil.IsSequenceInput(Info, 0, x);
        if (!seq)
        {
            FlowUtil.EmitPacked(run, 0, PackedValue.Optional(x));
            return;
        }

        Tensor[] items = FlowUtil.Sequence(x);
        try
        {
            FlowUtil.EmitPacked(run, 0, PackedValue.OptionalSequence(items));
        }
        finally
        {
            FlowUtil.DisposeAll(items);
        }
    }
}

/// <summary>OptionalHasElement: whether the optional holds a value; a plain tensor or sequence always does, an omitted input never.</summary>
internal sealed class OptionalHasElementOp : FlowKernel
{
    public static readonly OpDef Def = new("OptionalHasElement", 15, int.MaxValue, static n => new OptionalHasElementOp(n));

    private OptionalHasElementOp(NodeInfo n) : base(n)
    {
    }

    public override void Run(in OpRun run)
    {
        bool has;
        if (run.Inputs.Length == 0 || run.Inputs[0].IsEmpty)
        {
            has = false;
        }
        else if (FlowUtil.InputKind(Info, 0) == IrTypeKind.Optional || (FlowUtil.InputKind(Info, 0) is null && PackedValue.IsPacked(run.Inputs[0])))
        {
            has = PackedValue.Kind(run.Inputs[0]) != PackedKind.OptionalEmpty;
        }
        else
        {
            has = true;
        }

        run.AllocateOutput(0, DType.Bool, FlowUtil.ScalarShape).MutableData<bool>()[0] = has;
    }
}

/// <summary>OptionalGetElement: the value an optional holds (an error if empty); a plain tensor or sequence passes through.</summary>
internal sealed class OptionalGetElementOp : FlowKernel
{
    public static readonly OpDef Def = new("OptionalGetElement", 15, int.MaxValue, static n => new OptionalGetElementOp(n));

    private OptionalGetElementOp(NodeInfo n) : base(n)
    {
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0];
        IrTypeKind? kind = FlowUtil.InputKind(Info, 0);
        if (kind == IrTypeKind.Tensor || (kind is null && !PackedValue.IsPacked(x)) || kind == IrTypeKind.Sequence)
        {
            FlowUtil.Emit(run, 0, x);
            return;
        }

        switch (PackedValue.Kind(x))
        {
            case PackedKind.OptionalEmpty:
                throw new SessionException(SessionErrorKind.InvalidInput, "OptionalGetElement: the optional is empty");
            case PackedKind.OptionalTensor:
            {
                Tensor[] one = PackedValue.Items(x);
                try
                {
                    FlowUtil.Emit(run, 0, one[0]);
                }
                finally
                {
                    FlowUtil.DisposeAll(one);
                }

                break;
            }

            default:
            {
                Tensor[] items = PackedValue.Items(x);
                try
                {
                    FlowUtil.EmitPacked(run, 0, PackedValue.Sequence(items));
                }
                finally
                {
                    FlowUtil.DisposeAll(items);
                }

                break;
            }
        }
    }
}
