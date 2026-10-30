using System;
using System.Collections.Generic;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Flow;

/// <summary>
/// Helpers for the control-flow and sequence operators: byte-level copies of contiguous tensors, slicing and stacking along an axis,
/// and the packed form of sequences and optionals. These operators move whole tensors between iterations, so none of this is hot.
/// </summary>
internal static class FlowUtil
{
    public static int Product(ReadOnlySpan<int> d, int from, int to)
    {
        int p = 1;
        for (int i = from; i < to; i++)
        {
            p *= d[i];
        }

        return p;
    }

    /// <summary>A contiguous copy owned by the caller.</summary>
    public static Tensor Clone(Tensor t)
    {
        Tensor c = Tensor.Create(t.DType, t.Shape);
        Copy(t, c);
        return c;
    }

    public static void Copy(Tensor src, Tensor dst)
    {
        if (src.Length == 0)
        {
            return;
        }

        PackedValue.Bytes(src).CopyTo(PackedValue.MutableBytes(dst));
    }

    /// <summary>Writes a copy of <paramref name="src"/> as data-dependent output <paramref name="i"/>.</summary>
    public static void Emit(in OpRun run, int i, Tensor src)
    {
        if (i >= run.Outputs.Length)
        {
            return;
        }

        Copy(src, run.AllocateOutput(i, src.DType, src.Shape));
    }

    /// <summary>Writes a packed value as output <paramref name="i"/> and disposes it.</summary>
    public static void EmitPacked(in OpRun run, int i, Tensor packed)
    {
        try
        {
            Emit(run, i, packed);
        }
        finally
        {
            packed.Dispose();
        }
    }

    public static void DisposeAll(IEnumerable<Tensor> tensors)
    {
        foreach (Tensor t in tensors)
        {
            if (!t.IsEmpty)
            {
                t.Dispose();
            }
        }
    }

    /// <summary>The elements <c>start .. start+count</c> along <paramref name="axis"/>; with <paramref name="drop"/> the axis is removed (count must be 1).</summary>
    public static Tensor Narrow(Tensor src, int axis, int start, int count, bool drop)
    {
        ReadOnlySpan<int> d = src.Shape.Dimensions;
        int outer = Product(d, 0, axis), inner = Product(d, axis + 1, d.Length), size = d[axis] * inner;
        Span<int> nd = stackalloc int[Shape.MaxRank];
        int r = 0;
        for (int i = 0; i < d.Length; i++)
        {
            if (i == axis)
            {
                if (!drop)
                {
                    nd[r++] = count;
                }
            }
            else
            {
                nd[r++] = d[i];
            }
        }

        Tensor t = Tensor.Create(src.DType, Shape.Create(nd.Slice(0, r)));
        if (t.Length == 0)
        {
            return t;
        }

        int es = src.ElementSize, block = count * inner * es;
        ReadOnlySpan<byte> s = PackedValue.Bytes(src);
        Span<byte> o = PackedValue.MutableBytes(t);
        for (int a = 0; a < outer; a++)
        {
            s.Slice(((a * size) + (start * inner)) * es, block).CopyTo(o.Slice(a * block, block));
        }

        return t;
    }

    /// <summary>
    /// Writes <paramref name="slice"/> into <paramref name="dst"/> at <paramref name="index"/> along <paramref name="axis"/>, where
    /// <paramref name="dst"/> has the slice's shape with one extra axis there.
    /// </summary>
    public static void Place(Tensor dst, int axis, int index, Tensor slice)
    {
        if (slice.Length == 0)
        {
            return;
        }

        ReadOnlySpan<int> d = dst.Shape.Dimensions;
        int outer = Product(d, 0, axis), inner = Product(d, axis + 1, d.Length), n = d[axis], es = dst.ElementSize;
        int block = inner * es;
        ReadOnlySpan<byte> s = PackedValue.Bytes(slice);
        Span<byte> o = PackedValue.MutableBytes(dst);
        for (int a = 0; a < outer; a++)
        {
            s.Slice(a * block, block).CopyTo(o.Slice(((a * n) + index) * block, block));
        }
    }

    /// <summary>A tensor of shape <paramref name="slice"/> with <paramref name="count"/> inserted at <paramref name="axis"/>, as an output of the run.</summary>
    public static Tensor AllocateStacked(in OpRun run, int outIndex, DType type, ReadOnlySpan<int> slice, int axis, int count)
    {
        Span<int> nd = stackalloc int[Shape.MaxRank];
        int r = 0;
        for (int i = 0; i <= slice.Length; i++)
        {
            if (i == axis)
            {
                nd[r++] = count;
            }

            if (i < slice.Length)
            {
                nd[r++] = slice[i];
            }
        }

        Tensor t = run.AllocateOutput(outIndex, type, Shape.Create(nd.Slice(0, r)));
        if (t.Length > 0)
        {
            PackedValue.MutableBytes(t).Clear();
        }

        return t;
    }

    public static long ScalarInt(Tensor t) => t.DType switch
    {
        DType.I64 => t.Data<long>()[0],
        DType.I32 => t.Data<int>()[0],
        _ => throw new SessionException(SessionErrorKind.InvalidInput, $"expected an integer scalar but got {t.DType}"),
    };

    public static int Axis(long axis, int rank, string what)
    {
        if (axis < -rank || axis >= rank)
        {
            throw new SessionException(SessionErrorKind.InvalidInput, $"{what}: axis {axis} is outside [{-rank}, {rank})");
        }

        return (int)(axis < 0 ? axis + rank : axis);
    }

    /// <summary>What the model says the node's input <paramref name="i"/> is, or null when it does not say.</summary>
    public static IrTypeKind? InputKind(NodeInfo n, int i) =>
        i < n.Ir.Inputs.Length && n.Ir.Inputs[i]?.Type is { } t ? t.Kind : null;

    /// <summary>True if the input is a packed sequence or optional rather than a plain tensor, by the model's type or, failing that, by its bytes.</summary>
    public static bool IsPackedInput(NodeInfo n, int i, Tensor value) => InputKind(n, i) switch
    {
        IrTypeKind.Sequence or IrTypeKind.Optional => true,
        IrTypeKind.Tensor => false,
        _ => PackedValue.IsPacked(value),
    };

    /// <summary>True if the input is a sequence, by the model's type or, failing that, by its bytes.</summary>
    public static bool IsSequenceInput(NodeInfo n, int i, Tensor value) => InputKind(n, i) switch
    {
        IrTypeKind.Sequence => true,
        null => PackedValue.IsPacked(value) && PackedValue.Kind(value) == PackedKind.Sequence,
        _ => false,
    };

    public static Tensor[] Sequence(Tensor packed)
    {
        PackedKind k = PackedValue.Kind(packed);
        if (k is not (PackedKind.Sequence or PackedKind.OptionalSequence))
        {
            throw new SessionException(SessionErrorKind.InvalidInput, "expected a sequence");
        }

        return PackedValue.Items(packed);
    }

    public static Shape ScalarShape => Shape.Create();
}
