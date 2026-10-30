using System;
using System.Collections.Generic;
using System.Numerics;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Scatter;

/// <summary>
/// Unique (opset 11): the distinct elements of the flattened input, or of its slices along an axis. Outputs: Y the distinct values or
/// slices; indices the first occurrence of each; inverse_indices the position in Y of each input element or slice; counts how
/// often each occurs. With sorted = 1 (default) Y is in ascending order (slices compare element by element); with 0 it is in order of
/// first occurrence. NaN elements are never equal to each other.
/// </summary>
internal sealed class UniqueOp : OpKernel
{
    public static readonly OpDef Def = new("Unique", 11, int.MaxValue, static n => new UniqueOp(n));

    private readonly bool _hasAxis, _sorted;
    private readonly long _axis;

    private UniqueOp(NodeInfo n)
    {
        _hasAxis = n.Has("axis");
        _axis = n.GetInt("axis", 0);
        _sorted = n.GetInt("sorted", 1) != 0;
    }

    public override bool SupportsExact => true;

    public override bool OutputsDataDependent => true;

    public override void Infer(InferContext ctx) => throw new InvalidOperationException("Unique is shaped when it runs.");

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0];
        if (!TypeDispatch.IsNumber(x.DType))
        {
            throw new SessionException(SessionErrorKind.Unsupported, $"Unique: element type {x.DType} is not supported yet");
        }

        int axis = 0;
        if (_hasAxis)
        {
            if (_axis < -x.Rank || _axis >= x.Rank)
            {
                throw new SessionException(SessionErrorKind.InvalidInput, $"Unique: axis {_axis} is outside the {x.Rank} dimensions of the input");
            }

            axis = (int)(_axis < 0 ? _axis + x.Rank : _axis);
        }

        switch (x.DType)
        {
            case DType.F32: Compute<float>(run, x, axis); break;
            case DType.F64: Compute<double>(run, x, axis); break;
            case DType.I8: Compute<sbyte>(run, x, axis); break;
            case DType.I16: Compute<short>(run, x, axis); break;
            case DType.I32: Compute<int>(run, x, axis); break;
            case DType.I64: Compute<long>(run, x, axis); break;
            case DType.U8: Compute<byte>(run, x, axis); break;
            case DType.U16: Compute<ushort>(run, x, axis); break;
            case DType.U32: Compute<uint>(run, x, axis); break;
            case DType.U64: Compute<ulong>(run, x, axis); break;
            default: throw new SessionException(SessionErrorKind.Unsupported, $"Unique: element type {x.DType} is not supported yet");
        }
    }

    private void Compute<T>(in OpRun run, Tensor X, int Axis) where T : unmanaged, INumber<T>
    {
        bool HasAxis = _hasAxis, Sorted = _sorted;
        {
            T[] x = X.Data<T>().ToArray();
            ReadOnlySpan<int> d = X.Shape.Dimensions;
            // Items are slices along the axis; without an axis each element is an item.
            int outer = HasAxis ? OpUtil.Product(d, 0, Axis) : 1;
            int items = HasAxis ? d[Axis] : x.Length;
            int inner = HasAxis ? OpUtil.Product(d, Axis + 1, d.Length) : 1;
            // Compare item a with item b: the elements are visited outer-block by outer-block, as the slice's own order.
            int Compare(int a, int b)
            {
                for (int o = 0; o < outer; o++)
                {
                    for (int i = 0; i < inner; i++)
                    {
                        T p = x[(((o * items) + a) * inner) + i], q = x[(((o * items) + b) * inner) + i];
                        if (p < q)
                        {
                            return -1;
                        }

                        if (p > q)
                        {
                            return 1;
                        }
                    }
                }

                return 0;
            }

            // Group equal items: sort positions (stable) so equal items are adjacent, then number the groups.
            var order = new int[items];
            for (int i = 0; i < items; i++)
            {
                order[i] = i;
            }

            Array.Sort(order, (p, q) => Compare(p, q) is var c && c != 0 ? c : p.CompareTo(q));
            var firstOf = new List<int>();
            var count = new List<long>();
            var group = new int[items];
            for (int i = 0; i < items; i++)
            {
                if (i == 0 || Compare(order[i - 1], order[i]) != 0)
                {
                    firstOf.Add(order[i]);
                    count.Add(0);
                }

                group[order[i]] = firstOf.Count - 1;
                count[^1]++;
            }

            // Output order: ascending (the sorted order) or by first occurrence.
            int u = firstOf.Count;
            var rank = new int[u];
            for (int i = 0; i < u; i++)
            {
                rank[i] = i;
            }

            if (!Sorted)
            {
                Array.Sort(rank, (p, q) => firstOf[p].CompareTo(firstOf[q]));
            }

            var position = new int[u];
            for (int i = 0; i < u; i++)
            {
                position[rank[i]] = i;
            }

            Shape ys;
            if (HasAxis)
            {
                Span<int> dims = stackalloc int[Shape.MaxRank];
                d.CopyTo(dims);
                dims[Axis] = u;
                ys = Shape.Create(dims.Slice(0, d.Length));
            }
            else
            {
                ys = Shape.Create(u);
            }

            Tensor y = run.AllocateOutput(0, X.DType, ys);
            Span<T> yv = y.MutableData<T>();
            for (int i = 0; i < u; i++)
            {
                int src = firstOf[rank[i]];
                for (int o = 0; o < outer; o++)
                {
                    x.AsSpan(((o * items) + src) * inner, inner).CopyTo(yv.Slice(((o * u) + i) * inner, inner));
                }
            }

            Span<long> idx = run.AllocateOutput(1, DType.I64, Shape.Create(u)).MutableData<long>();
            Span<long> inv = run.AllocateOutput(2, DType.I64, Shape.Create(items)).MutableData<long>();
            Span<long> cnt = run.AllocateOutput(3, DType.I64, Shape.Create(u)).MutableData<long>();
            for (int i = 0; i < u; i++)
            {
                idx[i] = firstOf[rank[i]];
                cnt[i] = count[rank[i]];
            }

            for (int i = 0; i < items; i++)
            {
                inv[i] = position[group[i]];
            }
        }
    }
}
