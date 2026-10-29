using RecursiveAI.OneModel.Tensors;
using static RecursiveAI.OneModel.Tests.Graph.Operators.ShapeIndexing.Harness;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.ShapeIndexing;

/// <summary>Transpose, Concat, Split, Slice, Expand, Tile and Pad against naive per-element references, for every element size.</summary>
public class MovementTests
{
    private static readonly int[] SliceDims = [0, 1, 2, 5, 9];
    private static readonly int[] Forms = [0, 1, 2];
    private static readonly bool[] Bools = [true, false];
    private static readonly int[] Extents = [1, 2, 3, 5, 31, 32, 33, 40];

    private static T[] RefTranspose<T>(T[] data, int[] dims, int[] perm)
    {
        int[] od = perm.Select(p => dims[p]).ToArray();
        T[] o = new T[data.Length];
        for (int f = 0; f < o.Length; f++)
        {
            int[] oi = Unravel(f, od), ii = new int[dims.Length];
            for (int k = 0; k < perm.Length; k++)
            {
                ii[perm[k]] = oi[k];
            }

            o[f] = data[Ravel(ii, dims)];
        }

        return o;
    }

    private static string? TransposeCase<T>(Random rng, int caseNo) where T : unmanaged, System.Numerics.INumber<T>
    {
        int rank = rng.Next(1, 6);
        int[] dims;
        do
        {
            dims = Enumerable.Range(0, rank).Select(_ => Extents[rng.Next(Extents.Length)]).ToArray();
        }
        while (Count(dims) > 60000);

        int[] perm = Enumerable.Range(0, rank).OrderBy(_ => rng.Next()).ToArray();
        bool useDefault = rng.Next(5) == 0;
        if (useDefault)
        {
            perm = Enumerable.Range(0, rank).Reverse().ToArray();
        }

        T[] data = Iota<T>(Count(dims));
        var attrs = useDefault ? [] : new (string, object)[] { ("perm", perm.Select(p => (long)p).ToArray()) };
        Tensor[] outs = Harness.Run("Transpose", 13, [new In(Make(data, dims), caseNo % 2 == 0)], attrs, [DTypes.Of<T>()]);
        try
        {
            (T[] got, int[] gd) = Result<T>(outs[0]);
            int[] wd = perm.Select(p => dims[p]).ToArray();
            T[] want = RefTranspose(data, dims, perm);
            return gd.SequenceEqual(wd) && got.AsSpan().SequenceEqual(want) ? null : $"dims {Describe(dims)} perm {Describe(perm)}";
        }
        finally
        {
            DisposeAll(outs);
        }
    }

    [Fact]
    public void TransposeMatchesNaiveForEveryElementSize()
    {
        var rng = new Random(1234);
        var failures = new List<string>();
        for (int i = 0; i < 160; i++)
        {
            string? f = (i % 4) switch
            {
                0 => TransposeCase<float>(rng, i),
                1 => TransposeCase<long>(rng, i),
                2 => TransposeCase<short>(rng, i),
                _ => TransposeCase<byte>(rng, i),
            };
            if (f is not null)
            {
                failures.Add(f);
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures.Take(6)));
    }

    [Fact]
    public void TransposeOfEmptyAndScalarAndOnesWorks()
    {
        Tensor[] o = Harness.Run("Transpose", 13, [new In(Make(Array.Empty<float>(), 0, 3))], [], [DType.F32]);
        Assert.Equal(Shape.Create(3, 0), o[0].Shape);
        DisposeAll(o);
        o = Harness.Run("Transpose", 13, [new In(Make([7f]))], [], [DType.F32]);
        Assert.Equal([7f], o[0].ToArray<float>());
        DisposeAll(o);
        o = Harness.Run("Transpose", 13, [new In(Make([1f, 2, 3], 1, 3, 1))], [("perm", new long[] { 2, 0, 1 })], [DType.F32]);
        Assert.Equal(Shape.Create(1, 1, 3), o[0].Shape);
        Assert.Equal([1f, 2, 3], o[0].ToArray<float>());
        DisposeAll(o);
        Assert.Throws<RecursiveAI.OneModel.Graph.SessionException>(() =>
            Harness.Run("Transpose", 13, [new In(Make([1f, 2], 2))], [("perm", new long[] { 1 })], [DType.F32]));
        Assert.Throws<RecursiveAI.OneModel.Graph.SessionException>(() =>
            Harness.Run("Transpose", 13, [new In(Make([1f, 2, 3, 4], 2, 2))], [("perm", new long[] { 0, 0 })], [DType.F32]));
    }

    // ---- slice ----------------------------------------------------------------------------------------------------------------

    // The positions an ONNX slice selects along one axis of extent n, written as the spec's text reads.
    private static List<int> SlicePositions(long start, long end, long step, int n)
    {
        var pos = new List<int>();
        if (n == 0)
        {
            return pos;
        }

        if (step < 0 && (end == int.MaxValue || end == long.MaxValue))
        {
            end = long.MinValue; // ONNX Runtime reads these ends as "to the beginning" for a negative step
        }

        long s = start < 0 ? start + n : start;
        long e = end < 0 ? end + n : end;
        if (step > 0)
        {
            s = Math.Clamp(s, 0, n);
            e = Math.Clamp(e, 0, n);
            for (long p = s; p < e; p += Math.Min(step, n + 1L))
            {
                pos.Add((int)p);
            }
        }
        else
        {
            s = Math.Clamp(s, 0, n - 1);
            e = Math.Clamp(e, -1, n - 1);
            for (long p = s; p > e; p += Math.Max(step, -n - 1L))
            {
                pos.Add((int)p);
            }
        }

        return pos;
    }

    private static readonly long[] Bounds = [long.MinValue, -100, -7, -3, -2, -1, 0, 1, 2, 3, 5, 9, 100, int.MaxValue, long.MaxValue];
    private static readonly long[] Steps = [1, 1, 1, 2, 3, 7, -1, -1, -2, -4, long.MaxValue, long.MinValue + 1];

    [Fact]
    public void SliceMatchesTheSpecForEveryOpsetForm()
    {
        var rng = new Random(77);
        var failures = new List<string>();
        for (int c = 0; c < 400; c++)
        {
            int rank = rng.Next(1, 5);
            int[] dims = Enumerable.Range(0, rank).Select(_ => SliceDims[rng.Next(5)]).ToArray();
            if (rng.Next(4) != 0)
            {
                dims = dims.Select(d => Math.Max(d, 1)).ToArray();
            }

            int n = rng.Next(1, rank + 1);
            int[] axes = Enumerable.Range(0, rank).OrderBy(_ => rng.Next()).Take(n).ToArray();
            long[] starts = axes.Select(_ => Bounds[rng.Next(Bounds.Length)]).ToArray();
            long[] ends = axes.Select(_ => Bounds[rng.Next(Bounds.Length)]).ToArray();
            bool withSteps = rng.Next(3) != 0;
            long[] steps = axes.Select(_ => withSteps ? Steps[rng.Next(Steps.Length)] : 1).ToArray();
            bool negativeAxes = rng.Next(2) == 0;
            long[] axesArg = axes.Select(a => negativeAxes ? a - rank : a).Select(a => (long)a).ToArray();

            var positions = new List<int>[rank];
            for (int d = 0; d < rank; d++)
            {
                positions[d] = Enumerable.Range(0, dims[d]).ToList();
            }

            for (int i = 0; i < n; i++)
            {
                positions[axes[i]] = SlicePositions(starts[i], ends[i], steps[i], dims[axes[i]]);
            }

            int[] od = positions.Select(p => p.Count).ToArray();
            float[] data = Iota<float>(Count(dims));
            float[] want = new float[Count(od)];
            for (int f = 0; f < want.Length; f++)
            {
                int[] oi = Unravel(f, od);
                int[] ii = oi.Select((v, d) => positions[d][v]).ToArray();
                want[f] = data[Ravel(ii, dims)];
            }

            string label = $"dims {Describe(dims)} axes {string.Join(",", axesArg)} starts {string.Join(",", starts)} ends {string.Join(",", ends)} steps {string.Join(",", steps)}";
            foreach (int form in Forms)
            {
                try
                {
                    Tensor[] outs;
                    if (form == 0 && withSteps == false)
                    {
                        // Opset 1: attributes (positive steps of 1 only).
                        outs = Harness.Run("Slice", 1, [new In(Make(data, dims))],
                            [("starts", starts.Select(s => Math.Clamp(s, int.MinValue, int.MaxValue)).ToArray()),
                             ("ends", ends.Select(s => Math.Clamp(s, int.MinValue, int.MaxValue)).ToArray()), ("axes", axesArg)], [DType.F32]);
                    }
                    else if (form == 0)
                    {
                        continue;
                    }
                    else
                    {
                        bool constant = form == 1;
                        outs = Harness.Run("Slice", 13,
                            [new In(Make(data, dims)), new In(Make(starts, n), constant), new In(Make(ends, n), constant),
                             new In(Make(axesArg, n), constant), withSteps ? new In(Make(steps, n), constant) : null], [], [DType.F32]);
                    }

                    (float[] got, int[] gd) = Result<float>(outs[0]);
                    DisposeAll(outs);
                    if (!gd.SequenceEqual(od) || !got.AsSpan().SequenceEqual(want))
                    {
                        failures.Add($"form {form}: {label}: got {Describe(gd)} want {Describe(od)}");
                    }
                }
                catch (Exception e)
                {
                    failures.Add($"form {form}: {label}: {e.GetType().Name}: {e.Message}");
                }
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures.Take(6)));
    }

    [Fact]
    public void SliceRejectsStepZeroAndBadAxes()
    {
        In?[] Args(long axis, long step) =>
            [new In(Make([1f, 2, 3, 4], 4)), new In(Make<long>([0], 1), true), new In(Make<long>([2], 1), true), new In(Make([axis], 1), true), new In(Make([step], 1), true)];
        Assert.Throws<RecursiveAI.OneModel.Graph.SessionException>(() => Harness.Run("Slice", 13, Args(0, 0), [], [DType.F32]));
        Assert.Throws<RecursiveAI.OneModel.Graph.SessionException>(() => Harness.Run("Slice", 13, Args(1, 1), [], [DType.F32]));
        Assert.Throws<RecursiveAI.OneModel.Graph.SessionException>(() => Harness.Run("Slice", 13, Args(-2, 1), [], [DType.F32]));
    }

    // ---- concat and split -----------------------------------------------------------------------------------------------------

    [Fact]
    public void ConcatAndSplitRoundTripAlongEveryAxis()
    {
        var rng = new Random(5);
        var failures = new List<string>();
        for (int c = 0; c < 120; c++)
        {
            int rank = rng.Next(1, 5);
            int axis = rng.Next(rank);
            int[] baseDims = Enumerable.Range(0, rank).Select(_ => rng.Next(1, 5)).ToArray();
            int parts = rng.Next(1, 5);
            int[] sizes = Enumerable.Range(0, parts).Select(_ => rng.Next(0, 5)).ToArray();
            if (sizes.Sum() == 0)
            {
                sizes[0] = 1;
            }

            var pieces = new List<float[]>();
            var inputs = new List<In?>();
            var dimsOf = new List<int[]>();
            for (int p = 0; p < parts; p++)
            {
                int[] d = (int[])baseDims.Clone();
                d[axis] = sizes[p];
                float[] v = Iota<float>(Count(d)).Select(x => x + (p * 1000)).ToArray();
                pieces.Add(v);
                dimsOf.Add(d);
                inputs.Add(new In(Make(v, d), p % 2 == 0));
            }

            int[] total = (int[])baseDims.Clone();
            total[axis] = sizes.Sum();
            float[] want = new float[Count(total)];
            for (int f = 0; f < want.Length; f++)
            {
                int[] idx = Unravel(f, total);
                int a = idx[axis], p = 0;
                while (a >= sizes[p])
                {
                    a -= sizes[p++];
                }

                int[] local = (int[])idx.Clone();
                local[axis] = a;
                want[f] = pieces[p][Ravel(local, dimsOf[p])];
            }

            bool neg = rng.Next(2) == 0;
            Tensor[] o = Harness.Run("Concat", 13, [.. inputs], [("axis", (long)(neg ? axis - rank : axis))], [DType.F32]);
            (float[] got, int[] gd) = Result<float>(o[0]);
            DisposeAll(o);
            if (!gd.SequenceEqual(total) || !got.AsSpan().SequenceEqual(want))
            {
                failures.Add($"concat {Describe(baseDims)} axis {axis} sizes {string.Join(",", sizes)}");
                continue;
            }

            // Split it back: opset 13 (sizes input, constant and not), opset 11 (attribute).
            foreach (int form in Forms)
            {
                Tensor whole = Make(want, total);
                long[] sz = sizes.Select(s => (long)s).ToArray();
                Tensor[] s = form == 2
                    ? Harness.Run("Split", 11, [new In(whole)], [("axis", (long)axis), ("split", sz)], Enumerable.Repeat(DType.F32, parts).ToArray())
                    : Harness.Run("Split", 13, [new In(whole), new In(Make(sz, parts), form == 0)], [("axis", (long)axis)],
                        Enumerable.Repeat(DType.F32, parts).ToArray());
                for (int p = 0; p < parts; p++)
                {
                    (float[] g, int[] gdm) = Result<float>(s[p]);
                    if (!gdm.SequenceEqual(dimsOf[p]) || !g.AsSpan().SequenceEqual(pieces[p]))
                    {
                        failures.Add($"split form {form} part {p}: {Describe(baseDims)} axis {axis} sizes {string.Join(",", sizes)}");
                    }
                }

                DisposeAll(s);
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures.Take(6)));
    }

    [Fact]
    public void SplitWithoutSizesIsEqualBeforeOpset18AndCeilingChunksAfter()
    {
        float[] data = Iota<float>(10);
        Tensor[] o = Harness.Run("Split", 13, [new In(Make(data, 10))], [], [DType.F32, DType.F32]);
        Assert.Equal(Shape.Create(5), o[0].Shape);
        Assert.Equal([6f, 7, 8, 9, 10], o[1].ToArray<float>());
        DisposeAll(o);
        Assert.Throws<RecursiveAI.OneModel.Graph.SessionException>(() =>
            Harness.Run("Split", 13, [new In(Make(Iota<float>(7), 7))], [], [DType.F32, DType.F32]));

        o = Harness.Run("Split", 18, [new In(Make(Iota<float>(7), 7))], [("num_outputs", 3L)], [DType.F32, DType.F32, DType.F32]);
        Assert.Equal([3, 3, 1], o.Select(t => t.Shape[0]).ToArray());
        Assert.Equal([7f], o[2].ToArray<float>());
        DisposeAll(o);
    }

    // ---- expand and tile ------------------------------------------------------------------------------------------------------

    [Fact]
    public void ExpandMatchesBroadcasting()
    {
        var cases = new (int[] In, long[] To, int[] Out)[]
        {
            ([3, 1], [2, 1, 4], [2, 3, 4]),
            ([1], [5], [5]),
            ([4], [3, 4], [3, 4]),
            ([2, 3], [1, 3], [2, 3]),
            ([1, 3], [2, 1], [2, 3]),
            ([2, 1, 3], [2, 4, 3], [2, 4, 3]),
            ([], [3], [3]),
            ([0, 1], [0, 4], [0, 4]),
        };
        foreach (bool constant in Bools)
        {
            foreach ((int[] ind, long[] to, int[] outd) in cases)
            {
                float[] data = Iota<float>(Count(ind));
                Tensor[] o = Harness.Run("Expand", 13, [new In(Make(data, ind)), new In(Make(to, to.Length), constant)], [], [DType.F32]);
                (float[] got, int[] gd) = Result<float>(o[0]);
                DisposeAll(o);
                Assert.Equal(outd, gd);
                float[] want = new float[Count(outd)];
                for (int f = 0; f < want.Length; f++)
                {
                    int[] oi = Unravel(f, outd);
                    int[] ii = new int[ind.Length];
                    for (int d = 0; d < ind.Length; d++)
                    {
                        int od = d + (outd.Length - ind.Length);
                        ii[d] = ind[d] == 1 ? 0 : oi[od];
                    }

                    want[f] = data[Ravel(ii, ind)];
                }

                Assert.Equal(want, got);
            }
        }
    }

    [Fact]
    public void TileMatchesNaive()
    {
        var rng = new Random(9);
        for (int c = 0; c < 60; c++)
        {
            int rank = rng.Next(1, 4);
            int[] dims = Enumerable.Range(0, rank).Select(_ => rng.Next(0, 4)).ToArray();
            long[] reps = Enumerable.Range(0, rank).Select(_ => (long)rng.Next(0, 4)).ToArray();
            float[] data = Iota<float>(Count(dims));
            Tensor[] o = Harness.Run("Tile", 13, [new In(Make(data, dims)), new In(Make(reps, rank), c % 2 == 0)], [], [DType.F32]);
            (float[] got, int[] gd) = Result<float>(o[0]);
            DisposeAll(o);
            int[] od = dims.Select((d, i) => d * (int)reps[i]).ToArray();
            Assert.Equal(od, gd);
            float[] want = new float[Count(od)];
            for (int f = 0; f < want.Length; f++)
            {
                int[] oi = Unravel(f, od);
                want[f] = data[Ravel(oi.Select((v, d) => v % dims[d]).ToArray(), dims)];
            }

            Assert.Equal(want, got);
        }
    }

    // ---- pad ------------------------------------------------------------------------------------------------------------------

    private static int PadSource(int j, long before, int n, string mode)
    {
        long p = j - before;
        if (p >= 0 && p < n)
        {
            return (int)p;
        }

        switch (mode)
        {
            case "edge":
                return p < 0 ? 0 : n - 1;
            case "wrap":
                while (p < 0)
                {
                    p += n;
                }

                return (int)(p % n);
            default: // reflect
                if (n == 1)
                {
                    return 0;
                }

                while (p < 0 || p >= n)
                {
                    p = p < 0 ? -p : (2L * (n - 1)) - p;
                }

                return (int)p;
        }
    }

    [Fact]
    public void PadMatchesNaiveInEveryMode()
    {
        var rng = new Random(31);
        var failures = new List<string>();
        for (int c = 0; c < 200; c++)
        {
            int rank = rng.Next(1, 4);
            int[] dims = Enumerable.Range(0, rank).Select(_ => rng.Next(1, 6)).ToArray();
            string mode = new[] { "constant", "constant", "reflect", "edge", "wrap" }[rng.Next(5)];
            long[] before = dims.Select(d => (long)rng.Next(mode == "constant" ? -d : 0, 5)).ToArray();
            long[] after = dims.Select(d => (long)rng.Next(mode == "constant" ? -d : 0, 5)).ToArray();
            int[] od = dims.Select((d, i) => (int)(d + before[i] + after[i])).ToArray();
            if (od.Any(d => d < 0))
            {
                continue;
            }

            float[] data = Iota<float>(Count(dims));
            float fill = 9.5f;
            float[] want = new float[Count(od)];
            for (int f = 0; f < want.Length; f++)
            {
                int[] oi = Unravel(f, od);
                int[] ii = new int[rank];
                bool outside = false;
                for (int d = 0; d < rank; d++)
                {
                    if (mode == "constant")
                    {
                        long p = oi[d] - before[d];
                        outside |= p < 0 || p >= dims[d];
                        ii[d] = (int)Math.Clamp(p, 0, dims[d] - 1);
                    }
                    else
                    {
                        ii[d] = PadSource(oi[d], before[d], dims[d], mode);
                    }
                }

                want[f] = outside ? fill : data[Ravel(ii, dims)];
            }

            long[] pads = [.. before, .. after];
            bool constant = c % 2 == 0;
            try
            {
                Tensor[] o = Harness.Run("Pad", 13,
                    [new In(Make(data, dims)), new In(Make(pads, pads.Length), constant), mode == "constant" ? new In(Make([fill], 1), constant) : null],
                    [("mode", mode)], [DType.F32]);
                (float[] got, int[] gd) = Result<float>(o[0]);
                DisposeAll(o);
                if (!gd.SequenceEqual(od) || !got.AsSpan().SequenceEqual(want))
                {
                    failures.Add($"{mode} dims {Describe(dims)} pads {string.Join(",", pads)}");
                }
            }
            catch (Exception e)
            {
                failures.Add($"{mode} dims {Describe(dims)} pads {string.Join(",", pads)}: {e.Message}");
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures.Take(6)));
    }

    [Fact]
    public void PadOldOpsetsAndAxes()
    {
        // Opset 2: attributes; value defaults to 0.
        Tensor[] o = Harness.Run("Pad", 2, [new In(Make([1f, 2, 3, 4], 2, 2))], [("pads", new long[] { 1, 0, 0, 1 }), ("value", 5f)], [DType.F32]);
        Assert.Equal(Shape.Create(3, 3), o[0].Shape);
        Assert.Equal([5f, 5, 5, 1, 2, 5, 3, 4, 5], o[0].ToArray<float>());
        DisposeAll(o);

        // Opset 18: pads for the listed axes only.
        o = Harness.Run("Pad", 18, [new In(Make([1f, 2, 3, 4], 2, 2)), new In(Make<long>([1, 1], 2), true), null, new In(Make<long>([-1], 1), true)],
            [("mode", "edge")], [DType.F32]);
        Assert.Equal(Shape.Create(2, 4), o[0].Shape);
        Assert.Equal([1f, 1, 2, 2, 3, 3, 4, 4], o[0].ToArray<float>());
        DisposeAll(o);

        // Opset 19 wrap on int64.
        o = Harness.Run("Pad", 19, [new In(Make<long>([1, 2, 3], 3)), new In(Make<long>([2, 1], 2), true)], [("mode", "wrap")], [DType.I64]);
        Assert.Equal([2L, 3, 1, 2, 3, 1], o[0].ToArray<long>());
        DisposeAll(o);
    }
}
