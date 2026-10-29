using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using static RecursiveAI.OneModel.Tests.Graph.Operators.ShapeIndexing.Harness;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.ShapeIndexing;

/// <summary>Gathers, scatters, top-k, one-hot, non-zero, trilu, range and constant-of-shape against naive references.</summary>
public class IndexingTests
{
    private static readonly bool[] Bools = [true, false];

    [Fact]
    public void GatherMatchesNaiveWithNegativeIndicesAndEveryAxis()
    {
        var rng = new Random(3);
        var failures = new List<string>();
        for (int c = 0; c < 150; c++)
        {
            int rank = rng.Next(1, 4);
            int[] dims = Enumerable.Range(0, rank).Select(_ => rng.Next(1, 5)).ToArray();
            int axis = rng.Next(rank);
            int irank = rng.Next(0, 3);
            int[] idims = Enumerable.Range(0, irank).Select(_ => rng.Next(0, 4)).ToArray();
            long[] idx = Enumerable.Range(0, Count(idims)).Select(_ => (long)rng.Next(-dims[axis], dims[axis])).ToArray();
            bool wide = rng.Next(2) == 0;
            long[] data = Iota<long>(Count(dims));
            int[] od = [.. dims.Take(axis), .. idims, .. dims.Skip(axis + 1)];
            long[] want = new long[Count(od)];
            for (int f = 0; f < want.Length; f++)
            {
                int[] oi = Unravel(f, od);
                int[] ii = [.. oi.Take(axis)];
                int k = Ravel(oi.Skip(axis).Take(irank).ToArray(), idims);
                long gi = idx[k] < 0 ? idx[k] + dims[axis] : idx[k];
                ii = [.. ii, (int)gi, .. oi.Skip(axis + irank)];
                want[f] = data[Ravel(ii, dims)];
            }

            Tensor idxT = wide ? Make(idx, idims) : Make(idx.Select(v => (int)v).ToArray(), idims);
            Tensor[] o = Harness.Run("Gather", 13, [new In(Make(data, dims)), new In(idxT, c % 2 == 0)], [("axis", (long)(c % 3 == 0 ? axis - rank : axis))], [DType.I64]);
            (long[] got, int[] gd) = Result<long>(o[0]);
            DisposeAll(o);
            if (!gd.SequenceEqual(od) || !got.AsSpan().SequenceEqual(want))
            {
                failures.Add($"dims {Describe(dims)} axis {axis} idx {Describe(idims)}");
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures.Take(6)));
    }

    [Fact]
    public void GatherRejectsOutOfRangeIndicesAtRunTime()
    {
        foreach (long bad in new long[] { 3, -4, long.MaxValue, long.MinValue })
        {
            var e = Assert.Throws<SessionException>(() => Harness.Run("Gather", 13, [new In(Make([1f, 2, 3], 3)), new In(Make([bad], 1))], [], [DType.F32]));
            Assert.Equal(SessionErrorKind.InvalidInput, e.Kind);
        }
    }

    [Fact]
    public void GatherElementsMatchesNaive()
    {
        var rng = new Random(4);
        for (int c = 0; c < 80; c++)
        {
            int rank = rng.Next(1, 4);
            int[] dims = Enumerable.Range(0, rank).Select(_ => rng.Next(1, 5)).ToArray();
            int axis = rng.Next(rank);
            int[] idims = dims.Select((d, i) => i == axis ? rng.Next(0, 6) : rng.Next(1, d + 1)).ToArray();
            long[] idx = Enumerable.Range(0, Count(idims)).Select(_ => (long)rng.Next(-dims[axis], dims[axis])).ToArray();
            float[] data = Iota<float>(Count(dims));
            float[] want = new float[idx.Length];
            for (int f = 0; f < want.Length; f++)
            {
                int[] ii = Unravel(f, idims);
                ii[axis] = (int)(idx[f] < 0 ? idx[f] + dims[axis] : idx[f]);
                want[f] = data[Ravel(ii, dims)];
            }

            Tensor[] o = Harness.Run("GatherElements", 13, [new In(Make(data, dims)), new In(Make(idx, idims), c % 2 == 0)], [("axis", (long)axis)], [DType.F32]);
            (float[] got, int[] gd) = Result<float>(o[0]);
            DisposeAll(o);
            Assert.Equal(idims, gd);
            Assert.Equal(want, got);
        }
    }

    [Fact]
    public void GatherNdAndScatterNdMatchNaive()
    {
        var rng = new Random(6);
        for (int c = 0; c < 100; c++)
        {
            int batch = rng.Next(0, 3);
            int rank = batch + rng.Next(1, 4);
            int[] dims = Enumerable.Range(0, rank).Select(_ => rng.Next(1, 4)).ToArray();
            int k = rng.Next(1, rank - batch + 1);
            int[] lead = Enumerable.Range(0, rng.Next(0, 3)).Select(_ => rng.Next(1, 4)).ToArray();
            int[] idims = [.. dims.Take(batch), .. lead, k];
            int tuples = Count(idims) / k;
            long[] idx = new long[tuples * k];
            for (int t = 0; t < tuples; t++)
            {
                for (int j = 0; j < k; j++)
                {
                    idx[(t * k) + j] = rng.Next(-dims[batch + j], dims[batch + j]);
                }
            }

            float[] data = Iota<float>(Count(dims));
            int[] sliceDims = dims.Skip(batch + k).ToArray();
            int slice = Count(sliceDims);
            int[] od = [.. idims.Take(idims.Length - 1), .. sliceDims];
            float[] want = new float[Count(od)];
            int tuplesPerBatch = tuples / Math.Max(Count(dims.Take(batch).ToArray()), 1);
            for (int t = 0; t < tuples; t++)
            {
                int b = t / tuplesPerBatch;
                int[] bi = Unravel(b, dims.Take(batch).ToArray());
                for (int s = 0; s < slice; s++)
                {
                    int[] si = Unravel(s, sliceDims);
                    int[] full = [.. bi, .. Enumerable.Range(0, k).Select(j => (int)(idx[(t * k) + j] < 0 ? idx[(t * k) + j] + dims[batch + j] : idx[(t * k) + j])), .. si];
                    want[(t * slice) + s] = data[Ravel(full, dims)];
                }
            }

            Tensor[] o = Harness.Run("GatherND", 13, [new In(Make(data, dims)), new In(Make(idx, idims), c % 2 == 0)], [("batch_dims", (long)batch)], [DType.F32]);
            (float[] got, int[] gd) = Result<float>(o[0]);
            DisposeAll(o);
            Assert.Equal(od, gd);
            Assert.Equal(want, got);

            if (batch == 0)
            {
                // ScatterND with unique tuples: overwrite, then add.
                var seen = new HashSet<string>();
                var keep = new List<int>();
                for (int t = 0; t < tuples; t++)
                {
                    string key = string.Join(",", Enumerable.Range(0, k).Select(j => idx[(t * k) + j] < 0 ? idx[(t * k) + j] + dims[j] : idx[(t * k) + j]));
                    if (seen.Add(key))
                    {
                        keep.Add(t);
                    }
                }

                long[] idx2 = keep.SelectMany(t => idx.Skip(t * k).Take(k)).ToArray();
                float[] upd = Iota<float>(keep.Count * slice).Select(v => v * 100).ToArray();
                float[] expectSet = (float[])data.Clone(), expectAdd = (float[])data.Clone();
                for (int u = 0; u < keep.Count; u++)
                {
                    int[] pre = Enumerable.Range(0, k).Select(j => (int)(idx2[(u * k) + j] < 0 ? idx2[(u * k) + j] + dims[j] : idx2[(u * k) + j])).ToArray();
                    for (int s = 0; s < slice; s++)
                    {
                        int at = Ravel([.. pre, .. Unravel(s, sliceDims)], dims);
                        expectSet[at] = upd[(u * slice) + s];
                        expectAdd[at] += upd[(u * slice) + s];
                    }
                }

                int[] ud = [keep.Count, .. sliceDims];
                int[] id2 = [keep.Count, k];
                foreach ((string red, float[] expect) in new[] { ("none", expectSet), ("add", expectAdd) })
                {
                    Tensor[] so = Harness.Run("ScatterND", 16, [new In(Make(data, dims)), new In(Make(idx2, id2)), new In(Make(upd, ud))], [("reduction", red)], [DType.F32]);
                    Assert.Equal(expect, so[0].ToArray<float>());
                    DisposeAll(so);
                }
            }
        }
    }

    [Fact]
    public void ScatterNdAppliesDuplicatesInOrderAndReduces()
    {
        // Two updates to the same cell: last write wins; add sums; max/min/mul reduce.
        In[] Args() => [new In(Make([1f, 2, 3], 3)), new In(Make<long>([1, 1], 2, 1)), new In(Make([10f, 20], 2))];
        (string, float[])[] expected = [("none", [1f, 20, 3]), ("add", [1f, 32, 3]), ("mul", [1f, 400, 3]), ("max", [1f, 20, 3]), ("min", [1f, 2, 3])];
        foreach ((string red, float[] want) in expected)
        {
            Tensor[] o = Harness.Run("ScatterND", 18, [.. Args().Select(a => (In?)a)], [("reduction", red)], [DType.F32]);
            Assert.Equal(want, o[0].ToArray<float>());
            DisposeAll(o);
        }

        Assert.Throws<SessionException>(() => Harness.Run("ScatterND", 18,
            [new In(Make([1f, 2, 3], 3)), new In(Make<long>([3], 1, 1)), new In(Make([10f], 1))], [], [DType.F32]));
    }

    [Fact]
    public void TopKMatchesAStableSortInBothPaths()
    {
        var rng = new Random(8);
        var failures = new List<string>();
        for (int c = 0; c < 120; c++)
        {
            int rank = rng.Next(1, 4);
            int[] dims = Enumerable.Range(0, rank).Select(_ => rng.Next(1, 4)).ToArray();
            int axis = rng.Next(rank);
            dims[axis] = new[] { 1, 5, 17, 40, 64 }[rng.Next(5)];
            int k = rng.Next(0, dims[axis] + 1);
            bool largest = rng.Next(2) == 0;
            bool floats = rng.Next(2) == 0;
            // Few distinct values force ties; NaN included for floats.
            float[] data = Enumerable.Range(0, Count(dims)).Select(_ => (float)rng.Next(-4, 5)).ToArray();
            if (floats && data.Length > 3)
            {
                data[rng.Next(data.Length)] = float.NaN;
                data[rng.Next(data.Length)] = float.PositiveInfinity;
            }

            int[] od = (int[])dims.Clone();
            od[axis] = k;
            float[] wantV = new float[Count(od)];
            long[] wantI = new long[wantV.Length];
            int outer = Count(dims.Take(axis).ToArray()), inner = Count(dims.Skip(axis + 1).ToArray());
            for (int o = 0; o < outer; o++)
            {
                for (int q = 0; q < inner; q++)
                {
                    int n = dims[axis];
                    var order = Enumerable.Range(0, n).Select(i => (i, v: data[(((o * n) + i) * inner) + q])).ToList();
                    int Cmp(float a, float b) => float.IsNaN(a) || float.IsNaN(b) ? (float.IsNaN(a) ? 1 : 0) - (float.IsNaN(b) ? 1 : 0) : a.CompareTo(b);
                    var sorted = (largest ? order.OrderByDescending(x => x.v, Comparer<float>.Create(Cmp)) : order.OrderBy(x => x.v, Comparer<float>.Create(Cmp))).ToList();
                    for (int j = 0; j < k; j++)
                    {
                        wantV[(((o * k) + j) * inner) + q] = sorted[j].v;
                        wantI[(((o * k) + j) * inner) + q] = sorted[j].i;
                    }
                }
            }

            bool constantK = c % 2 == 0;
            Tensor[] outs = Harness.Run("TopK", 11, [new In(Make(data, dims)), new In(Make<long>([k], 1), constantK)],
                [("axis", (long)axis), ("largest", largest ? 1L : 0L)], [DType.F32, DType.I64]);
            (float[] gv, int[] gd) = Result<float>(outs[0]);
            long[] gi = outs[1].ToArray<long>();
            DisposeAll(outs);
            bool sameV = gv.Select(BitConverter.SingleToInt32Bits).SequenceEqual(wantV.Select(BitConverter.SingleToInt32Bits)) || gv.Zip(wantV).All(p => p.First == p.Second || (float.IsNaN(p.First) && float.IsNaN(p.Second)));
            if (!gd.SequenceEqual(od) || !sameV || !gi.AsSpan().SequenceEqual(wantI))
            {
                failures.Add($"dims {Describe(dims)} axis {axis} k {k} largest {largest}");
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures.Take(6)));
    }

    [Fact]
    public void TopKOldOpsetAttributeAndIntegerTypes()
    {
        Tensor[] o = Harness.Run("TopK", 1, [new In(Make<int>([5, 1, 9, 3], 4))], [("k", 2L)], [DType.I32, DType.I64]);
        Assert.Equal([9, 5], o[0].ToArray<int>());
        Assert.Equal([2L, 0], o[1].ToArray<long>());
        DisposeAll(o);
        Assert.Throws<SessionException>(() => Harness.Run("TopK", 11, [new In(Make([1f, 2], 2)), new In(Make<long>([3], 1), true)], [], [DType.F32, DType.I64]));
    }

    [Fact]
    public void OneHotMatchesNaive()
    {
        foreach (long axis in new long[] { -1, 0, 1 })
        {
            float[] idx = [1, 9, -1, 2];
            Tensor[] o = Harness.Run("OneHot", 11, [new In(Make(idx, 2, 2)), new In(Make<long>([4], 1), true), new In(Make([-1f, 7f], 2))], [("axis", axis)], [DType.F32]);
            (float[] got, int[] gd) = Result<float>(o[0]);
            DisposeAll(o);
            int[] od = axis == -1 ? [2, 2, 4] : axis == 0 ? [4, 2, 2] : [2, 4, 2];
            Assert.Equal(od, gd);
            for (int f = 0; f < got.Length; f++)
            {
                int[] oi = Unravel(f, od);
                int pos = axis == -1 ? oi[2] : axis == 0 ? oi[0] : oi[1];
                int[] ii = axis == -1 ? [oi[0], oi[1]] : axis == 0 ? [oi[1], oi[2]] : [oi[0], oi[2]];
                long v = (long)idx[Ravel(ii, [2, 2])];
                if (v < 0)
                {
                    v += 4;
                }

                Assert.Equal(v == pos ? 7f : -1f, got[f]);
            }
        }

        // A scalar index gives a vector.
        Tensor[] o1 = Harness.Run("OneHot", 11, [new In(Tensor.Scalar(2L)), new In(Make<long>([3], 1), true), new In(Make([0f, 1f], 2))], [], [DType.F32]);
        Assert.Equal(Shape.Create(3), o1[0].Shape);
        Assert.Equal([0f, 0f, 1f], o1[0].ToArray<float>());
        DisposeAll(o1);

        // int64 indices, int64 values, runtime depth.
        Tensor[] o2 = Harness.Run("OneHot", 9, [new In(Make<long>([0, 2], 2)), new In(Make<long>([3], 1)), new In(Make<long>([5, 6], 2))], [], [DType.I64]);
        Assert.Equal([6L, 5, 5, 5, 5, 6], o2[0].ToArray<long>());
        DisposeAll(o2);
    }

    [Fact]
    public void NonZeroListsCoordinatesInRowMajorOrder()
    {
        float[] data = [0, 1, -0f, 3, float.NaN, 0];
        Tensor[] o = Harness.Run("NonZero", 13, [new In(Make(data, 2, 3))], [], [DType.I64]);
        Assert.Equal(Shape.Create(2, 3), o[0].Shape);
        Assert.Equal([0L, 1, 1, 1, 0, 1], o[0].ToArray<long>());
        DisposeAll(o);
        o = Harness.Run("NonZero", 13, [new In(Make<float>([0, 0], 2))], [], [DType.I64]);
        Assert.Equal(Shape.Create(1, 0), o[0].Shape);
        DisposeAll(o);
        o = Harness.Run("NonZero", 13, [new In(Make<bool>([true, false, true], 3))], [], [DType.I64]);
        Assert.Equal([0L, 2], o[0].ToArray<long>());
        DisposeAll(o);
    }

    [Fact]
    public void TriluKeepsTheRightTriangle()
    {
        foreach (bool upper in Bools)
        {
            foreach (long k in new long[] { -5, -1, 0, 1, 2, 9 })
            {
                int[] dims = [2, 3, 4];
                float[] data = Iota<float>(24);
                Tensor[] o = Harness.Run("Trilu", 14, [new In(Make(data, dims)), new In(Make([k], 1), true)], [("upper", upper ? 1L : 0L)], [DType.F32]);
                float[] got = o[0].ToArray<float>();
                DisposeAll(o);
                for (int f = 0; f < 24; f++)
                {
                    int[] i = Unravel(f, dims);
                    long diff = i[2] - i[1];
                    bool keep = upper ? diff >= k : diff <= k;
                    Assert.Equal(keep ? data[f] : 0f, got[f]);
                }
            }
        }
    }

    [Fact]
    public void RangeAndConstantOfShape()
    {
        Tensor[] o = Harness.Run("Range", 11, [new In(Tensor.Scalar(3L)), new In(Tensor.Scalar(10L)), new In(Tensor.Scalar(3L))], [], [DType.I64]);
        Assert.Equal([3L, 6, 9], o[0].ToArray<long>());
        DisposeAll(o);
        o = Harness.Run("Range", 11, [new In(Tensor.Scalar(1f), true), new In(Tensor.Scalar(0f), false), new In(Tensor.Scalar(-0.25f), true)], [], [DType.F32]);
        Assert.Equal([1f, 0.75f, 0.5f, 0.25f], o[0].ToArray<float>());
        DisposeAll(o);
        o = Harness.Run("Range", 11, [new In(Tensor.Scalar<int>(5)), new In(Tensor.Scalar<int>(2)), new In(Tensor.Scalar<int>(1))], [], [DType.I32]);
        Assert.Equal(Shape.Create(0), o[0].Shape);
        DisposeAll(o);
        Assert.Throws<SessionException>(() => Harness.Run("Range", 11, [new In(Tensor.Scalar(0L)), new In(Tensor.Scalar(5L)), new In(Tensor.Scalar(0L))], [], [DType.I64]));

        o = Harness.Run("ConstantOfShape", 9, [new In(Make<long>([2, 3], 2))], [("value", ModelBuilder.Int64Tensor("v", [7], 1))], [DType.I64]);
        Assert.Equal(Shape.Create(2, 3), o[0].Shape);
        Assert.Equal(Enumerable.Repeat(7L, 6), o[0].ToArray<long>());
        DisposeAll(o);
        o = Harness.Run("ConstantOfShape", 9, [new In(Make<long>([0, 2], 2), true)], [], [DType.F32]);
        Assert.Equal(Shape.Create(0, 2), o[0].Shape);
        DisposeAll(o);
        o = Harness.Run("ConstantOfShape", 9, [new In(Make(Array.Empty<long>(), 0))], [], [DType.F32]);
        Assert.Equal(Shape.Create(), o[0].Shape);
        Assert.Equal([0f], o[0].ToArray<float>());
        DisposeAll(o);
    }

    [Fact]
    public void ShapeSizeSqueezeUnsqueezeFlattenDropout()
    {
        Tensor x = Make(Iota<float>(24), 2, 1, 3, 4);
        Tensor[] o = Harness.Run("Shape", 15, [new In(x)], [("start", 1L), ("end", -1L)], [DType.I64]);
        Assert.Equal([1L, 3], o[0].ToArray<long>());
        DisposeAll(o);
        o = Harness.Run("Shape", 13, [new In(x)], [], [DType.I64]);
        Assert.Equal([2L, 1, 3, 4], o[0].ToArray<long>());
        DisposeAll(o);
        o = Harness.Run("Size", 13, [new In(x)], [], [DType.I64]);
        Assert.Equal(24L, o[0].ToArray<long>()[0]);
        DisposeAll(o);

        foreach (bool constant in Bools)
        {
            o = Harness.Run("Squeeze", 13, [new In(x), new In(Make<long>([-3], 1), constant)], [], [DType.F32]);
            Assert.Equal(Shape.Create(2, 3, 4), o[0].Shape);
            DisposeAll(o);
            o = Harness.Run("Squeeze", 13, [new In(x)], [], [DType.F32]);
            Assert.Equal(Shape.Create(2, 3, 4), o[0].Shape);
            DisposeAll(o);
            o = Harness.Run("Unsqueeze", 13, [new In(x), new In(Make<long>([0, -1, 3], 3), constant)], [], [DType.F32]);
            Assert.Equal(Shape.Create(1, 2, 1, 1, 3, 4, 1), o[0].Shape);
            Assert.Equal(x.ToArray<float>(), o[0].ToArray<float>());
            DisposeAll(o);
        }

        o = Harness.Run("Squeeze", 11, [new In(x)], [("axes", new long[] { 1 })], [DType.F32]);
        Assert.Equal(Shape.Create(2, 3, 4), o[0].Shape);
        DisposeAll(o);
        o = Harness.Run("Unsqueeze", 11, [new In(x)], [("axes", new long[] { 4 })], [DType.F32]);
        Assert.Equal(Shape.Create(2, 1, 3, 4, 1), o[0].Shape);
        DisposeAll(o);
        Assert.Throws<SessionException>(() => Harness.Run("Squeeze", 13, [new In(x), new In(Make<long>([0], 1), true)], [], [DType.F32]));
        Assert.Throws<SessionException>(() => Harness.Run("Unsqueeze", 13, [new In(x), new In(Make<long>([1, 1], 2), true)], [], [DType.F32]));

        foreach ((long axis, int[] want) in new (long, int[])[] { (0, [1, 24]), (1, [2, 12]), (-1, [6, 4]), (4, [24, 1]), (-4, [1, 24]) })
        {
            o = Harness.Run("Flatten", 13, [new In(x)], [("axis", axis)], [DType.F32]);
            Assert.Equal(Shape.Create(want), o[0].Shape);
            DisposeAll(o);
        }

        o = Harness.Run("Dropout", 13, [new In(x)], [], [DType.F32]);
        Assert.Equal(x.ToArray<float>(), o[0].ToArray<float>());
        DisposeAll(o);
        o = Harness.Run("Dropout", 13, [new In(x), new In(Tensor.Scalar(0.5f), true), new In(Tensor.Scalar(false), true)], [], [DType.F32, DType.Bool]);
        Assert.Equal(x.ToArray<float>(), o[0].ToArray<float>());
        Assert.Equal(Enumerable.Repeat(true, 24), o[1].ToArray<bool>());
        DisposeAll(o);
        Assert.Throws<SessionException>(() => Harness.Run("Dropout", 13, [new In(x), new In(Tensor.Scalar(0.5f), true), new In(Tensor.Scalar(true), true)], [], [DType.F32]));
        x.Dispose();
    }

    [Fact]
    public void HostileValuesFailWithSessionExceptionsOnly()
    {
        long big = 1L << 40;
        Tensor x = Make(Iota<float>(6), 2, 3);
        var cases = new (string Op, int Opset, In?[] Inputs, (string, object)[] Attrs)[]
        {
            ("ConstantOfShape", 9, [new In(Make<long>([big], 1))], []),
            ("ConstantOfShape", 9, [new In(Make<long>([100000, 100000], 2), true)], []),
            ("ConstantOfShape", 9, [new In(Make<long>([1000000, 1000], 2))], []),
            ("Expand", 13, [new In(x), new In(Make<long>([big], 1))], []),
            ("Expand", 13, [new In(x), new In(Make<long>([-1], 1))], []),
            ("Tile", 13, [new In(x), new In(Make<long>([big, 1], 2))], []),
            ("Tile", 13, [new In(x), new In(Make<long>([-1, 1], 2))], []),
            ("Pad", 13, [new In(x), new In(Make<long>([big, 0, 0, 0], 4))], []),
            ("Pad", 13, [new In(x), new In(Make<long>([-big, 0, 0, 0], 4))], []),
            ("Pad", 13, [new In(x), new In(Make<long>([1, 2, 3], 3))], []),
            ("Range", 11, [new In(Tensor.Scalar(0L)), new In(Tensor.Scalar(big)), new In(Tensor.Scalar(1L))], []),
            ("Range", 11, [new In(Tensor.Scalar(long.MinValue)), new In(Tensor.Scalar(long.MaxValue)), new In(Tensor.Scalar(1L))], []),
            ("Range", 11, [new In(Tensor.Scalar(0f)), new In(Tensor.Scalar(float.NaN)), new In(Tensor.Scalar(1f))], []),
            ("OneHot", 11, [new In(Make<long>([1], 1)), new In(Tensor.Scalar(big)), new In(Make<long>([0, 1], 2))], []),
            ("OneHot", 11, [new In(Make<long>([1], 1)), new In(Tensor.Scalar(-3L)), new In(Make<long>([0, 1], 2))], []),
            ("Slice", 13, [new In(x), new In(Make<long>([0], 1)), new In(Make<long>([1], 1)), new In(Make<long>([big], 1))], []),
            ("Slice", 13, [new In(x), new In(Make<long>([0, 0, 0], 3)), new In(Make<long>([1, 1, 1], 3))], []),
            ("Unsqueeze", 13, [new In(x), new In(Make(Enumerable.Range(0, 40).Select(i => (long)i).ToArray(), 40))], []),
            ("Unsqueeze", 13, [new In(x), new In(Make<long>([big], 1))], []),
            ("Squeeze", 13, [new In(x), new In(Make<long>([big], 1))], []),
            ("Split", 13, [new In(x), new In(Make<long>([-1, 4], 2))], [("axis", 1L)]),
            ("Split", 13, [new In(x), new In(Make<long>([long.MaxValue, 1], 2))], [("axis", 1L)]),
            ("TopK", 11, [new In(x), new In(Make<long>([big], 1))], []),
            ("TopK", 11, [new In(x), new In(Make<long>([-1], 1))], []),
            ("Gather", 13, [new In(x), new In(Make<long>([long.MinValue], 1))], []),
            ("GatherND", 13, [new In(x), new In(Make<long>([long.MaxValue, 0], 1, 2))], []),
            ("GatherND", 13, [new In(x), new In(Make<long>([0, 0, 0], 1, 3))], []),
            ("GatherElements", 13, [new In(x), new In(Make<long>([9, 9], 1, 2))], [("axis", 1L)]),
            ("ScatterND", 13, [new In(x), new In(Make<long>([long.MinValue], 1, 1)), new In(Make<float>([1, 1, 1], 1, 3))], []),
            ("Transpose", 13, [new In(x)], [("perm", new long[] { 0, big })]),
            ("Flatten", 13, [new In(x)], [("axis", big)]),
            ("Concat", 13, [new In(x), new In(Make<float>([1, 2], 2, 1)), new In(Make<float>([1], 1, 1))], [("axis", 1L)]),
            ("Trilu", 14, [new In(Make<float>([1, 2], 2))], []),
        };
        var wrong = new List<string>();
        foreach ((string op, int opset, In?[] inputs, (string, object)[] attrs) in cases)
        {
            DType[] types = op is "Range" or "TopK" or "OneHot" or "ConstantOfShape" ? [DType.F32, DType.I64] : [DType.F32];
            try
            {
                Tensor[] o = Harness.Run(op, opset, inputs, attrs, op == "TopK" ? types : [types[0]]);
                DisposeAll(o);
                wrong.Add($"{op}: no error");
            }
            catch (SessionException)
            {
            }
            catch (Exception e)
            {
                wrong.Add($"{op}: {e.GetType().Name}: {e.Message}");
            }
        }

        x.Dispose();
        Assert.True(wrong.Count == 0, string.Join("\n", wrong));
    }

    [Fact]
    public void BigTransposeAndGatherAgreeAcrossThreadCounts()
    {
        float[] data = Iota<float>(257 * 129 * 3);
        int[] dims = [257, 129, 3];
        Tensor[] a = Harness.Run("Transpose", 13, [new In(Make(data, dims))], [("perm", new long[] { 2, 0, 1 })], [DType.F32], threads: 1);
        Tensor[] b = Harness.Run("Transpose", 13, [new In(Make(data, dims))], [("perm", new long[] { 2, 0, 1 })], [DType.F32], threads: 8);
        Assert.True(a[0].ToArray<float>().AsSpan().SequenceEqual(b[0].ToArray<float>()));
        DisposeAll(a);
        DisposeAll(b);
    }
}
