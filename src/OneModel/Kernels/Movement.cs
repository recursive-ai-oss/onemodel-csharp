using System;
using System.Numerics;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Data-movement kernels: strided copies (transpose, slice, concat, split, expand, tile), gathers, scatters, padding, triangular masks,
/// one-hot, non-zero and top-k. They only move or compare elements, so they are exact by construction: every path gives the same bits.
/// Written on spans; the moving kernels are generic over the element size (byte, ushort, uint, ulong) so one body serves every type.
/// </summary>
internal static class Movement
{
    // A tile is a block of the transpose small enough that its source and destination lines stay in L1 while it is copied (cache
    // blocking as in Goto & van de Geijn 2008 §4 for packing): 8 rows by 64 columns, so at most 8 destination lines share a cache set.
    private const int TileJ = 8;
    private const int TileL = 64;

    // Tile's rank doubles the input's: it splits each axis into (repeat, extent).
    private const int MaxDims = Shape.MaxRank * 2;

    // ---- strided copy ---------------------------------------------------------------------------------------------------------

    /// <summary>
    /// dst[dstBase + sum(i_k * dstStr[k])] = src[srcBase + sum(i_k * srcStr[k])] for every index i inside <paramref name="dims"/>.
    /// Strides are in elements and may be zero (broadcast) or negative (reversed slice). Axes that line up in both operands are merged
    /// first, so a contiguous copy is one memmove; when the source's contiguous axis is not the destination's, the copy is done in
    /// 32x32 tiles. Both tensors must be contiguous and of the same element size, and the strided accesses must stay inside them
    /// (callers derive strides from shapes they validated; a bad stride throws from the span indexer, never reads outside).
    /// </summary>
    public static void Move(in Tensor src, int srcBase, ReadOnlySpan<int> srcStr, in Tensor dst, int dstBase, ReadOnlySpan<int> dstStr,
        ReadOnlySpan<int> dims)
    {
        if (srcStr.Length != dims.Length || dstStr.Length != dims.Length || dims.Length > MaxDims)
        {
            throw new ArgumentException("Strides and dims must have the same length (at most 16).");
        }

        for (int i = 0; i < dims.Length; i++)
        {
            if (dims[i] == 0)
            {
                return;
            }
        }

        switch (src.ElementSize)
        {
            case 1: MoveCore(src.RawData<byte>(), srcBase, srcStr, dst.RawMutableData<byte>(), dstBase, dstStr, dims); break;
            case 2: MoveCore(src.RawData<ushort>(), srcBase, srcStr, dst.RawMutableData<ushort>(), dstBase, dstStr, dims); break;
            case 4: MoveCore(src.RawData<uint>(), srcBase, srcStr, dst.RawMutableData<uint>(), dstBase, dstStr, dims); break;
            default: MoveCore(src.RawData<ulong>(), srcBase, srcStr, dst.RawMutableData<ulong>(), dstBase, dstStr, dims); break;
        }
    }

    private static void MoveCore<T>(ReadOnlySpan<T> s, int sb, ReadOnlySpan<int> srcStr, Span<T> d, int db, ReadOnlySpan<int> dstStr,
        ReadOnlySpan<int> dims) where T : unmanaged
    {
        Span<int> dm = stackalloc int[MaxDims];
        Span<int> sa = stackalloc int[MaxDims];
        Span<int> sd = stackalloc int[MaxDims];

        // Drop extent-1 axes and merge an outer axis into the inner one when both operands step across it exactly one inner extent.
        int r = 0;
        for (int i = 0; i < dims.Length; i++)
        {
            if (dims[i] == 1)
            {
                continue;
            }

            if (r > 0 && sa[r - 1] == (long)srcStr[i] * dims[i] && sd[r - 1] == (long)dstStr[i] * dims[i])
            {
                dm[r - 1] *= dims[i];
                sa[r - 1] = srcStr[i];
                sd[r - 1] = dstStr[i];
            }
            else
            {
                dm[r] = dims[i];
                sa[r] = srcStr[i];
                sd[r] = dstStr[i];
                r++;
            }
        }

        if (r == 0)
        {
            d[db] = s[sb];
            return;
        }

        int last = r - 1;
        int tiled = -1;
        if (r >= 2 && sa[last] != 1)
        {
            for (int j = last - 1; j >= 0; j--)
            {
                if (sa[j] == 1)
                {
                    tiled = j;
                    break;
                }
            }
        }

        int nL = dm[last];
        int nJ = 1;
        if (tiled >= 0)
        {
            nJ = dm[tiled];
            dm[tiled] = 1; // the tile covers this axis; the outer loop must not step it
        }

        Span<int> idx = stackalloc int[MaxDims];
        int so = sb, dd = db;
        while (true)
        {
            if (tiled >= 0)
            {
                TileBlock(s, so, sa[tiled], sa[last], d, dd, sd[tiled], sd[last], nJ, nL);
            }
            else
            {
                Row(s, so, sa[last], d, dd, sd[last], nL);
            }

            int k = last - 1;
            for (; k >= 0; k--)
            {
                so += sa[k];
                dd += sd[k];
                if (++idx[k] < dm[k])
                {
                    break;
                }

                so -= sa[k] * dm[k];
                dd -= sd[k] * dm[k];
                idx[k] = 0;
            }

            if (k < 0)
            {
                break;
            }
        }
    }

    private static void Row<T>(ReadOnlySpan<T> s, int so, int sa, Span<T> d, int dd, int sd, int n) where T : unmanaged
    {
        if (sa == 1 && sd == 1)
        {
            s.Slice(so, n).CopyTo(d.Slice(dd, n));
        }
        else if (sa == 0 && sd == 1)
        {
            d.Slice(dd, n).Fill(s[so]);
        }
        else
        {
            for (int i = 0; i < n; i++, so += sa, dd += sd)
            {
                d[dd] = s[so];
            }
        }
    }

    private static void TileBlock<T>(ReadOnlySpan<T> s, int so, int sJ, int sL, Span<T> d, int dd, int dJ, int dL, int nJ, int nL)
        where T : unmanaged
    {
        // J is the axis that is contiguous in the source, L the one that is contiguous in the destination. Walking L in the outer loop
        // reads the source in runs along J and writes TileJ destination rows that stay in L1 while their TileL columns fill. A power-of-two
        // row length maps all rows to one cache set, so the number of rows touched at once (TileJ) must not exceed the associativity (8).
        for (int j0 = 0; j0 < nJ; j0 += TileJ)
        {
            int jn = Math.Min(TileJ, nJ - j0);
            for (int l0 = 0; l0 < nL; l0 += TileL)
            {
                int ln = Math.Min(TileL, nL - l0);
                for (int l = 0; l < ln; l++)
                {
                    int sp = so + (j0 * sJ) + ((l0 + l) * sL);
                    int dp = dd + (j0 * dJ) + ((l0 + l) * dL);
                    for (int j = 0; j < jn; j++, sp += sJ, dp += dJ)
                    {
                        d[dp] = s[sp];
                    }
                }
            }
        }
    }

    /// <summary>Sets every element of a contiguous tensor to the element at index 0 of <paramref name="pattern"/> (any type of the same size).</summary>
    public static void FillWith(in Tensor dst, in Tensor pattern)
    {
        switch (dst.ElementSize)
        {
            case 1: dst.RawMutableData<byte>().Fill(pattern.RawData<byte>()[0]); break;
            case 2: dst.RawMutableData<ushort>().Fill(pattern.RawData<ushort>()[0]); break;
            case 4: dst.RawMutableData<uint>().Fill(pattern.RawData<uint>()[0]); break;
            default: dst.RawMutableData<ulong>().Fill(pattern.RawData<ulong>()[0]); break;
        }
    }

    // ---- gather ---------------------------------------------------------------------------------------------------------------

    private static SessionException IndexError(string what, long index, long extent) =>
        new(SessionErrorKind.InvalidInput, $"{what} index {index} is out of range for an extent of {extent}.");

    /// <summary>
    /// Gather: out[o, i, :] = data[o, indices[i], :] with data viewed as [outer, axisDim, inner]. Negative indices count from the end.
    /// </summary>
    public static void Gather(in Tensor data, in Tensor indices, in Tensor output, int outer, int axisDim, int inner)
    {
        if (indices.DType == DType.I64)
        {
            GatherDispatch(data, indices.Data<long>(), output, outer, axisDim, inner);
        }
        else
        {
            GatherDispatch(data, indices.Data<int>(), output, outer, axisDim, inner);
        }
    }

    private static void GatherDispatch<TI>(in Tensor data, ReadOnlySpan<TI> idx, in Tensor output, int outer, int axisDim, int inner)
        where TI : unmanaged, IBinaryInteger<TI>
    {
        switch (data.ElementSize)
        {
            case 1: GatherCore(data.RawData<byte>(), idx, output.RawMutableData<byte>(), outer, axisDim, inner); break;
            case 2: GatherCore(data.RawData<ushort>(), idx, output.RawMutableData<ushort>(), outer, axisDim, inner); break;
            case 4: GatherCore(data.RawData<uint>(), idx, output.RawMutableData<uint>(), outer, axisDim, inner); break;
            default: GatherCore(data.RawData<ulong>(), idx, output.RawMutableData<ulong>(), outer, axisDim, inner); break;
        }
    }

    private static void GatherCore<T, TI>(ReadOnlySpan<T> data, ReadOnlySpan<TI> idx, Span<T> output, int outer, int axisDim, int inner)
        where T : unmanaged where TI : unmanaged, IBinaryInteger<TI>
    {
        int ni = idx.Length;
        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < ni; i++)
            {
                long k = long.CreateTruncating(idx[i]);
                if (k < 0)
                {
                    k += axisDim;
                }

                if ((ulong)k >= (ulong)axisDim)
                {
                    throw IndexError("Gather", long.CreateTruncating(idx[i]), axisDim);
                }

                int src = (int)((((long)o * axisDim) + k) * inner);
                int dst = ((o * ni) + i) * inner;
                if (inner == 1)
                {
                    output[dst] = data[src];
                }
                else
                {
                    data.Slice(src, inner).CopyTo(output.Slice(dst, inner));
                }
            }
        }
    }

    /// <summary>GatherElements: out[p] = data[p with its <paramref name="axis"/> coordinate replaced by indices[p]]; the output has the indices' shape.</summary>
    public static void GatherElements(in Tensor data, in Tensor indices, in Tensor output, int axis)
    {
        if (indices.DType == DType.I64)
        {
            GatherElementsDispatch(data, indices.Data<long>(), indices.Shape, output, axis);
        }
        else
        {
            GatherElementsDispatch(data, indices.Data<int>(), indices.Shape, output, axis);
        }
    }

    private static void GatherElementsDispatch<TI>(in Tensor data, ReadOnlySpan<TI> idx, Shape idxShape, in Tensor output, int axis)
        where TI : unmanaged, IBinaryInteger<TI>
    {
        switch (data.ElementSize)
        {
            case 1: GatherElementsCore(data.RawData<byte>(), data.Shape, idx, idxShape, output.RawMutableData<byte>(), axis); break;
            case 2: GatherElementsCore(data.RawData<ushort>(), data.Shape, idx, idxShape, output.RawMutableData<ushort>(), axis); break;
            case 4: GatherElementsCore(data.RawData<uint>(), data.Shape, idx, idxShape, output.RawMutableData<uint>(), axis); break;
            default: GatherElementsCore(data.RawData<ulong>(), data.Shape, idx, idxShape, output.RawMutableData<ulong>(), axis); break;
        }
    }

    private static void GatherElementsCore<T, TI>(ReadOnlySpan<T> data, Shape dataShape, ReadOnlySpan<TI> idx, Shape idxShape,
        Span<T> output, int axis) where T : unmanaged where TI : unmanaged, IBinaryInteger<TI>
    {
        int rank = idxShape.Rank;
        if (idx.Length == 0)
        {
            return;
        }

        Span<int> ds = stackalloc int[Shape.MaxRank];
        Span<int> pos = stackalloc int[Shape.MaxRank];
        long acc = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            ds[d] = (int)acc;
            acc *= dataShape[d];
        }

        int axisDim = dataShape[axis];
        int axisStride = ds[axis];
        int baseOff = 0; // contribution of every coordinate except the axis one
        for (int p = 0; p < idx.Length; p++)
        {
            long k = long.CreateTruncating(idx[p]);
            long kk = k < 0 ? k + axisDim : k;
            if ((ulong)kk >= (ulong)axisDim)
            {
                throw IndexError("GatherElements", k, axisDim);
            }

            output[p] = data[baseOff + (int)(kk * axisStride)];
            for (int d = rank - 1; d >= 0; d--)
            {
                pos[d]++;
                if (d != axis)
                {
                    baseOff += ds[d];
                }

                if (pos[d] < idxShape[d])
                {
                    break;
                }

                if (d != axis)
                {
                    baseOff -= ds[d] * idxShape[d];
                }

                pos[d] = 0;
            }
        }
    }

    // Offset (in elements, relative to the batch's start) of the slice that the index tuple t of [.., k] indices selects; the first k
    // dimensions of the batch's data are indexed, negative values count from the end.
    private static int NdOffset<TI>(ReadOnlySpan<TI> idx, long tupleStart, int k, ReadOnlySpan<int> dims, ReadOnlySpan<int> strides, string what)
        where TI : unmanaged, IBinaryInteger<TI>
    {
        long off = 0;
        for (int j = 0; j < k; j++)
        {
            long v = long.CreateTruncating(idx[(int)(tupleStart + j)]);
            long w = v < 0 ? v + dims[j] : v;
            if ((ulong)w >= (ulong)dims[j])
            {
                throw IndexError(what, v, dims[j]);
            }

            off += w * strides[j];
        }

        return (int)off;
    }

    /// <summary>
    /// GatherND: <paramref name="indices"/> has shape [b..., n..., k]; each k-tuple selects a slice of data's dimensions after the
    /// batch dimensions and the first k of the rest. The output is [b..., n..., data.shape[batchDims + k:]].
    /// </summary>
    public static void GatherND(in Tensor data, in Tensor indices, in Tensor output, int batchDims)
    {
        if (indices.DType == DType.I64)
        {
            GatherNDDispatch(data, indices.Data<long>(), indices.Shape, output, batchDims);
        }
        else
        {
            GatherNDDispatch(data, indices.Data<int>(), indices.Shape, output, batchDims);
        }
    }

    private static void GatherNDDispatch<TI>(in Tensor data, ReadOnlySpan<TI> idx, Shape idxShape, in Tensor output, int batchDims)
        where TI : unmanaged, IBinaryInteger<TI>
    {
        switch (data.ElementSize)
        {
            case 1: GatherNDCore(data.RawData<byte>(), data.Shape, idx, idxShape, output.RawMutableData<byte>(), batchDims); break;
            case 2: GatherNDCore(data.RawData<ushort>(), data.Shape, idx, idxShape, output.RawMutableData<ushort>(), batchDims); break;
            case 4: GatherNDCore(data.RawData<uint>(), data.Shape, idx, idxShape, output.RawMutableData<uint>(), batchDims); break;
            default: GatherNDCore(data.RawData<ulong>(), data.Shape, idx, idxShape, output.RawMutableData<ulong>(), batchDims); break;
        }
    }

    private static void NdGeometry(Shape dataShape, Shape idxShape, int batchDims, out int batch, out int tuples, out int k,
        out int sliceSize, out int dataBatchSize, Span<int> strides)
    {
        k = idxShape[idxShape.Rank - 1];
        batch = 1;
        for (int i = 0; i < batchDims; i++)
        {
            batch *= dataShape[i];
        }

        tuples = 1;
        for (int i = batchDims; i < idxShape.Rank - 1; i++)
        {
            tuples *= idxShape[i];
        }

        sliceSize = 1;
        for (int i = batchDims + k; i < dataShape.Rank; i++)
        {
            sliceSize *= dataShape[i];
        }

        long acc = sliceSize;
        for (int j = k - 1; j >= 0; j--)
        {
            strides[j] = (int)acc;
            acc *= dataShape[batchDims + j];
        }

        dataBatchSize = (int)acc;
    }

    private static void GatherNDCore<T, TI>(ReadOnlySpan<T> data, Shape dataShape, ReadOnlySpan<TI> idx, Shape idxShape, Span<T> output,
        int batchDims) where T : unmanaged where TI : unmanaged, IBinaryInteger<TI>
    {
        Span<int> strides = stackalloc int[Shape.MaxRank];
        NdGeometry(dataShape, idxShape, batchDims, out int batch, out int tuples, out int k, out int sliceSize, out int dataBatchSize, strides);
        ReadOnlySpan<int> dims = dataShape.Dimensions.Slice(batchDims);
        if (idx.Length == 0 || output.Length == 0)
        {
            return;
        }

        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < tuples; t++)
            {
                long tuple = (((long)b * tuples) + t) * k;
                int off = NdOffset(idx, tuple, k, dims, strides, "GatherND");
                data.Slice((b * dataBatchSize) + off, sliceSize).CopyTo(output.Slice(((b * tuples) + t) * sliceSize, sliceSize));
            }
        }
    }

    // ---- scatter --------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// ScatterND: output = data, then for each k-tuple of <paramref name="indices"/> the matching slice of the output is replaced by
    /// (reduction 0) or combined with (1 add, 2 mul, 3 max, 4 min) the matching slice of <paramref name="updates"/>. Duplicate
    /// indices apply in index order.
    /// </summary>
    public static void ScatterND(in Tensor data, in Tensor indices, in Tensor updates, in Tensor output, int reduction)
    {
        if (reduction == 0)
        {
            if (indices.DType == DType.I64)
            {
                ScatterNDDispatch(data, indices.Data<long>(), indices.Shape, updates, output);
            }
            else
            {
                ScatterNDDispatch(data, indices.Data<int>(), indices.Shape, updates, output);
            }

            return;
        }

        switch (data.DType)
        {
            case DType.F32: ScatterNDReduceDispatch<float>(data, indices, updates, output, reduction); break;
            case DType.F64: ScatterNDReduceDispatch<double>(data, indices, updates, output, reduction); break;
            case DType.I32: ScatterNDReduceDispatch<int>(data, indices, updates, output, reduction); break;
            case DType.I64: ScatterNDReduceDispatch<long>(data, indices, updates, output, reduction); break;
            default: throw new NotSupportedException($"ScatterND reduction on {data.DType}");
        }
    }

    private static void ScatterNDDispatch<TI>(in Tensor data, ReadOnlySpan<TI> idx, Shape idxShape, in Tensor updates, in Tensor output)
        where TI : unmanaged, IBinaryInteger<TI>
    {
        switch (data.ElementSize)
        {
            case 1: ScatterNDCore(data.RawData<byte>(), data.Shape, idx, idxShape, updates.RawData<byte>(), output.RawMutableData<byte>()); break;
            case 2: ScatterNDCore(data.RawData<ushort>(), data.Shape, idx, idxShape, updates.RawData<ushort>(), output.RawMutableData<ushort>()); break;
            case 4: ScatterNDCore(data.RawData<uint>(), data.Shape, idx, idxShape, updates.RawData<uint>(), output.RawMutableData<uint>()); break;
            default: ScatterNDCore(data.RawData<ulong>(), data.Shape, idx, idxShape, updates.RawData<ulong>(), output.RawMutableData<ulong>()); break;
        }
    }

    private static void ScatterNDCore<T, TI>(ReadOnlySpan<T> data, Shape dataShape, ReadOnlySpan<TI> idx, Shape idxShape,
        ReadOnlySpan<T> updates, Span<T> output) where T : unmanaged where TI : unmanaged, IBinaryInteger<TI>
    {
        data.CopyTo(output);
        Span<int> strides = stackalloc int[Shape.MaxRank];
        NdGeometry(dataShape, idxShape, 0, out _, out int tuples, out int k, out int sliceSize, out _, strides);
        if (idx.Length == 0 && tuples > 0)
        {
            return;
        }

        for (int t = 0; t < tuples; t++)
        {
            int off = NdOffset(idx, (long)t * k, k, dataShape.Dimensions, strides, "ScatterND");
            updates.Slice(t * sliceSize, sliceSize).CopyTo(output.Slice(off, sliceSize));
        }
    }

    private static void ScatterNDReduceDispatch<T>(in Tensor data, in Tensor indices, in Tensor updates, in Tensor output, int reduction)
        where T : unmanaged, INumber<T>
    {
        if (indices.DType == DType.I64)
        {
            ScatterNDReduceCore(data.Data<T>(), data.Shape, indices.Data<long>(), indices.Shape, updates.Data<T>(), output.MutableData<T>(), reduction);
        }
        else
        {
            ScatterNDReduceCore(data.Data<T>(), data.Shape, indices.Data<int>(), indices.Shape, updates.Data<T>(), output.MutableData<T>(), reduction);
        }
    }

    private static void ScatterNDReduceCore<T, TI>(ReadOnlySpan<T> data, Shape dataShape, ReadOnlySpan<TI> idx, Shape idxShape,
        ReadOnlySpan<T> updates, Span<T> output, int reduction) where T : unmanaged, INumber<T> where TI : unmanaged, IBinaryInteger<TI>
    {
        data.CopyTo(output);
        Span<int> strides = stackalloc int[Shape.MaxRank];
        NdGeometry(dataShape, idxShape, 0, out _, out int tuples, out int k, out int sliceSize, out _, strides);
        if (idx.Length == 0 && tuples > 0)
        {
            return;
        }

        for (int t = 0; t < tuples; t++)
        {
            int off = NdOffset(idx, (long)t * k, k, dataShape.Dimensions, strides, "ScatterND");
            for (int i = 0; i < sliceSize; i++)
            {
                T u = updates[(t * sliceSize) + i];
                ref T o = ref output[off + i];
                o = reduction switch
                {
                    1 => o + u,
                    2 => o * u,
                    3 => u < o ? o : u, // NaN: the update wins when either is NaN (black box: ONNX Runtime)
                    _ => o < u ? o : u,
                };
            }
        }
    }

    // ---- pad ------------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// For one axis of a non-constant pad: map[j] is the input index that output index j reads, for an input extent of
    /// <paramref name="inDim"/> padded by <paramref name="before"/> in front (negative: cropped). Mode 1 = reflect (the edge is not
    /// repeated; period 2*(n-1)), 2 = edge (clamp), 3 = wrap (period n).
    /// </summary>
    public static void PadMap(Span<int> map, int inDim, long before, int mode)
    {
        for (int j = 0; j < map.Length; j++)
        {
            long p = j - before;
            if ((ulong)p < (ulong)inDim)
            {
                map[j] = (int)p;
            }
            else if (mode == 2)
            {
                map[j] = p < 0 ? 0 : inDim - 1;
            }
            else if (mode == 3)
            {
                map[j] = (int)(((p % inDim) + inDim) % inDim);
            }
            else if (inDim == 1)
            {
                map[j] = 0;
            }
            else
            {
                long period = 2L * (inDim - 1);
                long q = ((p % period) + period) % period;
                map[j] = (int)(q < inDim ? q : period - q);
            }
        }
    }

    /// <summary>Pads by reading through per-axis index maps (see <see cref="PadMap"/>), concatenated in <paramref name="maps"/>.</summary>
    public static void PadMapped(in Tensor input, in Tensor output, ReadOnlySpan<int> maps)
    {
        switch (input.ElementSize)
        {
            case 1: PadMappedCore(input.RawData<byte>(), input.Shape, output.RawMutableData<byte>(), output.Shape, maps); break;
            case 2: PadMappedCore(input.RawData<ushort>(), input.Shape, output.RawMutableData<ushort>(), output.Shape, maps); break;
            case 4: PadMappedCore(input.RawData<uint>(), input.Shape, output.RawMutableData<uint>(), output.Shape, maps); break;
            default: PadMappedCore(input.RawData<ulong>(), input.Shape, output.RawMutableData<ulong>(), output.Shape, maps); break;
        }
    }

    private static void PadMappedCore<T>(ReadOnlySpan<T> s, Shape inShape, Span<T> d, Shape outShape, ReadOnlySpan<int> maps) where T : unmanaged
    {
        int rank = outShape.Rank;
        if (d.Length == 0)
        {
            return;
        }

        Span<int> mapStart = stackalloc int[Shape.MaxRank];
        Span<int> ist = stackalloc int[Shape.MaxRank];
        Span<int> pos = stackalloc int[Shape.MaxRank];
        int acc = 0;
        long a2 = 1;
        for (int a = 0; a < rank; a++)
        {
            mapStart[a] = acc;
            acc += outShape[a];
        }

        for (int a = rank - 1; a >= 0; a--)
        {
            ist[a] = (int)a2;
            a2 *= inShape[a];
        }

        int n = outShape[rank - 1];
        ReadOnlySpan<int> lastMap = maps.Slice(mapStart[rank - 1], n);
        int rows = d.Length / n;
        for (int row = 0; row < rows; row++)
        {
            int srcRow = 0;
            for (int a = 0; a < rank - 1; a++)
            {
                srcRow += maps[mapStart[a] + pos[a]] * ist[a];
            }

            Span<T> dr = d.Slice(row * n, n);
            for (int j = 0; j < n; j++)
            {
                dr[j] = s[srcRow + lastMap[j]];
            }

            for (int a = rank - 2; a >= 0; a--)
            {
                if (++pos[a] < outShape[a])
                {
                    break;
                }

                pos[a] = 0;
            }
        }
    }

    // ---- trilu ----------------------------------------------------------------------------------------------------------------

    /// <summary>Keeps the upper (or lower) triangle relative to diagonal <paramref name="k"/> of every [rows, cols] matrix and zeroes the rest.</summary>
    public static void Trilu(in Tensor input, in Tensor output, int batch, int rows, int cols, long k, bool upper)
    {
        switch (input.ElementSize)
        {
            case 1: TriluCore(input.RawData<byte>(), output.RawMutableData<byte>(), batch, rows, cols, k, upper); break;
            case 2: TriluCore(input.RawData<ushort>(), output.RawMutableData<ushort>(), batch, rows, cols, k, upper); break;
            case 4: TriluCore(input.RawData<uint>(), output.RawMutableData<uint>(), batch, rows, cols, k, upper); break;
            default: TriluCore(input.RawData<ulong>(), output.RawMutableData<ulong>(), batch, rows, cols, k, upper); break;
        }
    }

    private static void TriluCore<T>(ReadOnlySpan<T> s, Span<T> d, int batch, int rows, int cols, long k, bool upper) where T : unmanaged
    {
        for (int b = 0; b < batch; b++)
        {
            for (int r = 0; r < rows; r++)
            {
                int at = ((b * rows) + r) * cols;
                ReadOnlySpan<T> src = s.Slice(at, cols);
                Span<T> dst = d.Slice(at, cols);
                if (upper)
                {
                    int from = (int)Math.Clamp(r + k, 0, cols); // keep columns c with c - r >= k
                    dst.Slice(0, from).Clear();
                    src.Slice(from).CopyTo(dst.Slice(from));
                }
                else
                {
                    int to = (int)Math.Clamp(r + k + 1, 0, cols); // keep columns c with c - r <= k
                    src.Slice(0, to).CopyTo(dst.Slice(0, to));
                    dst.Slice(to).Clear();
                }
            }
        }
    }

    // ---- one-hot --------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// OneHot: output [pre, depth, post] where indices is [pre, post]. Element (p, c, q) is <paramref name="values"/>[1] when
    /// index[p, q] (negative: plus depth) equals c, else values[0]. <paramref name="values"/> is the 2-element [off, on] tensor.
    /// </summary>
    public static void OneHot(in Tensor indices, in Tensor values, in Tensor output, int pre, int depth, int post)
    {
        switch (indices.DType)
        {
            case DType.I64: OneHotDispatch(indices.Data<long>(), values, output, pre, depth, post); break;
            case DType.I32: OneHotDispatch(indices.Data<int>(), values, output, pre, depth, post); break;
            case DType.F32: OneHotDispatch(indices.Data<float>(), values, output, pre, depth, post); break;
            case DType.F64: OneHotDispatch(indices.Data<double>(), values, output, pre, depth, post); break;
            case DType.I16: OneHotDispatch(indices.Data<short>(), values, output, pre, depth, post); break;
            case DType.I8: OneHotDispatch(indices.Data<sbyte>(), values, output, pre, depth, post); break;
            case DType.U8: OneHotDispatch(indices.Data<byte>(), values, output, pre, depth, post); break;
            default: throw new NotSupportedException($"OneHot indices of type {indices.DType}");
        }
    }

    private static void OneHotDispatch<TI>(ReadOnlySpan<TI> idx, in Tensor values, in Tensor output, int pre, int depth, int post)
        where TI : unmanaged, INumber<TI>
    {
        switch (values.ElementSize)
        {
            case 1: OneHotCore(idx, values.RawData<byte>(), output.RawMutableData<byte>(), pre, depth, post); break;
            case 2: OneHotCore(idx, values.RawData<ushort>(), output.RawMutableData<ushort>(), pre, depth, post); break;
            case 4: OneHotCore(idx, values.RawData<uint>(), output.RawMutableData<uint>(), pre, depth, post); break;
            default: OneHotCore(idx, values.RawData<ulong>(), output.RawMutableData<ulong>(), pre, depth, post); break;
        }
    }

    private static void OneHotCore<TI, T>(ReadOnlySpan<TI> idx, ReadOnlySpan<T> values, Span<T> d, int pre, int depth, int post)
        where TI : unmanaged, INumber<TI> where T : unmanaged
    {
        d.Fill(values[0]);
        T on = values[1];
        for (int p = 0; p < pre; p++)
        {
            for (int q = 0; q < post; q++)
            {
                TI raw = idx[(p * post) + q];
                if (TI.IsNaN(raw))
                {
                    continue;
                }

                long v = long.CreateSaturating(raw);
                if (v < 0)
                {
                    v += depth;
                }

                if ((ulong)v < (ulong)depth)
                {
                    d[((((p * depth) + (int)v)) * post) + q] = on;
                }
            }
        }
    }

    // ---- non-zero -------------------------------------------------------------------------------------------------------------

    /// <summary>Counts the elements of <paramref name="input"/> that are not zero (-0.0 is zero, NaN is not).</summary>
    public static int CountNonZero(in Tensor input) => input.DType switch
    {
        DType.F32 => CountNz(input.Data<float>()),
        DType.F64 => CountNz(input.Data<double>()),
        DType.I8 => CountNz(input.Data<sbyte>()),
        DType.I16 => CountNz(input.Data<short>()),
        DType.I32 => CountNz(input.Data<int>()),
        DType.I64 => CountNz(input.Data<long>()),
        DType.U8 => CountNz(input.Data<byte>()),
        DType.U16 => CountNz(input.Data<ushort>()),
        DType.U32 => CountNz(input.Data<uint>()),
        DType.U64 => CountNz(input.Data<ulong>()),
        DType.Bool => CountNz(input.RawData<byte>()),
        _ => throw new NotSupportedException($"NonZero of {input.DType}"),
    };

    private static int CountNz<T>(ReadOnlySpan<T> s) where T : unmanaged, INumber<T>
    {
        int n = 0;
        foreach (T v in s)
        {
            if (v != T.Zero)
            {
                n++;
            }
        }

        return n;
    }

    /// <summary>
    /// Writes the coordinates of the non-zero elements, in row-major order, as an int64 tensor [max(rank, 1), count]: column c holds
    /// the coordinates of the c-th non-zero element. A scalar input is treated as one element of a 1-D tensor.
    /// </summary>
    public static void NonZero(in Tensor input, in Tensor output)
    {
        switch (input.DType)
        {
            case DType.F32: NonZeroCore(input.Data<float>(), input.Shape, output.MutableData<long>()); break;
            case DType.F64: NonZeroCore(input.Data<double>(), input.Shape, output.MutableData<long>()); break;
            case DType.I8: NonZeroCore(input.Data<sbyte>(), input.Shape, output.MutableData<long>()); break;
            case DType.I16: NonZeroCore(input.Data<short>(), input.Shape, output.MutableData<long>()); break;
            case DType.I32: NonZeroCore(input.Data<int>(), input.Shape, output.MutableData<long>()); break;
            case DType.I64: NonZeroCore(input.Data<long>(), input.Shape, output.MutableData<long>()); break;
            case DType.U8: NonZeroCore(input.Data<byte>(), input.Shape, output.MutableData<long>()); break;
            case DType.U16: NonZeroCore(input.Data<ushort>(), input.Shape, output.MutableData<long>()); break;
            case DType.U32: NonZeroCore(input.Data<uint>(), input.Shape, output.MutableData<long>()); break;
            case DType.U64: NonZeroCore(input.Data<ulong>(), input.Shape, output.MutableData<long>()); break;
            case DType.Bool: NonZeroCore(input.RawData<byte>(), input.Shape, output.MutableData<long>()); break;
            default: throw new NotSupportedException($"NonZero of {input.DType}");
        }
    }

    private static void NonZeroCore<T>(ReadOnlySpan<T> s, Shape shape, Span<long> o) where T : unmanaged, INumber<T>
    {
        int rank = Math.Max(shape.Rank, 1);
        int count = o.Length / rank;
        Span<int> pos = stackalloc int[Shape.MaxRank];
        int c = 0;
        for (int i = 0; i < s.Length && c < count; i++)
        {
            if (s[i] != T.Zero)
            {
                for (int r = 0; r < rank; r++)
                {
                    o[(r * count) + c] = pos[r];
                }

                c++;
            }

            for (int a = shape.Rank - 1; a >= 0; a--)
            {
                if (++pos[a] < shape[a])
                {
                    break;
                }

                pos[a] = 0;
            }
        }
    }

    // ---- top-k ----------------------------------------------------------------------------------------------------------------

    // Up to this k a row is selected by insertion into a sorted list of k (O(n*k)); beyond it the row's indices are merge-sorted.
    private const int InsertionTopKLimit = 16;

    /// <summary>
    /// TopK along an axis: the viewed [outer, n, inner] input gives values and int64 indices [outer, k, inner], each column sorted
    /// best first (largest first, or smallest first when <paramref name="largest"/> is false). Equal values keep their input order
    /// (lower index first); NaN counts as greater than every number. <paramref name="scratch"/> needs 2*n ints.
    /// </summary>
    public static void TopK(in Tensor input, in Tensor values, in Tensor indices, int outer, int n, int inner, int k, bool largest,
        Span<int> scratch)
    {
        switch (input.DType)
        {
            case DType.F32: TopKCore(input.Data<float>(), values.MutableData<float>(), indices.MutableData<long>(), outer, n, inner, k, largest, scratch); break;
            case DType.F64: TopKCore(input.Data<double>(), values.MutableData<double>(), indices.MutableData<long>(), outer, n, inner, k, largest, scratch); break;
            case DType.I8: TopKCore(input.Data<sbyte>(), values.MutableData<sbyte>(), indices.MutableData<long>(), outer, n, inner, k, largest, scratch); break;
            case DType.I16: TopKCore(input.Data<short>(), values.MutableData<short>(), indices.MutableData<long>(), outer, n, inner, k, largest, scratch); break;
            case DType.I32: TopKCore(input.Data<int>(), values.MutableData<int>(), indices.MutableData<long>(), outer, n, inner, k, largest, scratch); break;
            case DType.I64: TopKCore(input.Data<long>(), values.MutableData<long>(), indices.MutableData<long>(), outer, n, inner, k, largest, scratch); break;
            case DType.U8: TopKCore(input.Data<byte>(), values.MutableData<byte>(), indices.MutableData<long>(), outer, n, inner, k, largest, scratch); break;
            case DType.U16: TopKCore(input.Data<ushort>(), values.MutableData<ushort>(), indices.MutableData<long>(), outer, n, inner, k, largest, scratch); break;
            case DType.U32: TopKCore(input.Data<uint>(), values.MutableData<uint>(), indices.MutableData<long>(), outer, n, inner, k, largest, scratch); break;
            case DType.U64: TopKCore(input.Data<ulong>(), values.MutableData<ulong>(), indices.MutableData<long>(), outer, n, inner, k, largest, scratch); break;
            default: throw new NotSupportedException($"TopK of {input.DType}");
        }
    }

    // True when x ranks strictly before y.
    private static bool Before<T>(T x, T y, bool largest) where T : unmanaged, INumber<T>
    {
        // NaN is the greatest value: first when largest, last when smallest.
        bool xn = T.IsNaN(x), yn = T.IsNaN(y);
        if (xn || yn)
        {
            return largest ? xn && !yn : yn && !xn;
        }

        return largest ? x > y : x < y;
    }

    private static void TopKCore<T>(ReadOnlySpan<T> s, Span<T> vOut, Span<long> iOut, int outer, int n, int inner, int k, bool largest,
        Span<int> scratch) where T : unmanaged, INumber<T>
    {
        if (k == 0 || outer == 0 || inner == 0)
        {
            return;
        }

        Span<int> top = scratch.Slice(0, n);
        Span<int> tmp = scratch.Slice(n, n);
        for (int o = 0; o < outer; o++)
        {
            for (int q = 0; q < inner; q++)
            {
                int colBase = (o * n * inner) + q;
                if (k <= InsertionTopKLimit)
                {
                    int count = 0;
                    for (int i = 0; i < n; i++)
                    {
                        T v = s[colBase + (i * inner)];
                        if (count == k && !Before(v, s[colBase + (top[k - 1] * inner)], largest))
                        {
                            continue;
                        }

                        int p = count < k ? count++ : k - 1;
                        // Shift worse entries down; an equal earlier entry stays ahead because Before is strict.
                        while (p > 0 && Before(v, s[colBase + (top[p - 1] * inner)], largest))
                        {
                            top[p] = top[p - 1];
                            p--;
                        }

                        top[p] = i;
                    }
                }
                else
                {
                    for (int i = 0; i < n; i++)
                    {
                        top[i] = i;
                    }

                    MergeSort(s, colBase, inner, top, tmp, largest);
                }

                for (int j = 0; j < k; j++)
                {
                    int at = (((o * k) + j) * inner) + q;
                    vOut[at] = s[colBase + (top[j] * inner)];
                    iOut[at] = top[j];
                }
            }
        }
    }

    // Bottom-up stable merge sort of the indices in a by the values they name.
    private static void MergeSort<T>(ReadOnlySpan<T> s, int colBase, int inner, Span<int> a, Span<int> b, bool largest) where T : unmanaged, INumber<T>
    {
        int n = a.Length;
        Span<int> src = a, dst = b;
        for (int width = 1; width < n; width *= 2)
        {
            for (int lo = 0; lo < n; lo += 2 * width)
            {
                int mid = Math.Min(lo + width, n), hi = Math.Min(lo + (2 * width), n);
                int i = lo, j = mid, o = lo;
                while (i < mid && j < hi)
                {
                    // Take from the right run only if it ranks strictly before: ties keep the left (earlier) element.
                    dst[o++] = Before(s[colBase + (src[j] * inner)], s[colBase + (src[i] * inner)], largest) ? src[j++] : src[i++];
                }

                while (i < mid)
                {
                    dst[o++] = src[i++];
                }

                while (j < hi)
                {
                    dst[o++] = src[j++];
                }
            }

            Span<int> swap = src;
            src = dst;
            dst = swap;
        }

        if (!src.Overlaps(a))
        {
            src.CopyTo(a);
        }
    }

    // ---- range ----------------------------------------------------------------------------------------------------------------

    /// <summary>Range: out[i] = start + i * delta.</summary>
    public static void Range<T>(Span<T> output, T start, T delta) where T : unmanaged, INumber<T>
    {
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = start + (T.CreateTruncating(i) * delta);
        }
    }
}
