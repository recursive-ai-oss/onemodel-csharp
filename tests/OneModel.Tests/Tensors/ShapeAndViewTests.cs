using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Tensors;

public class ShapeAndViewTests
{
    [Fact]
    public void ShapeRejectsBadAndOverflowingSizes()
    {
        Assert.Throws<TensorShapeException>(() => Shape.Create(-1));
        Assert.Throws<TensorShapeException>(() => Shape.Create(1, 1, 1, 1, 1, 1, 1, 1, 1));
        Assert.Throws<TensorShapeException>(() => Shape.Create(65536, 65536));
        Assert.Throws<TensorShapeException>(() => Shape.Create(int.MaxValue, 2));
        Assert.Throws<TensorShapeException>(() => Shape.Create(new long[] { 1L << 31 }));
        Assert.Throws<TensorShapeException>(() => Shape.Create(new long[] { -5 }));
        Assert.Equal(int.MaxValue, Shape.Create(int.MaxValue).Length);
        Assert.Equal(1, default(Shape).Length);
        Assert.Equal(0, Shape.Create(3, 0, 4).Length);
        Assert.False(Shape.TryBroadcast(Shape.Create(65536, 1), Shape.Create(1, 65536), out _));
        Assert.True(Shape.TryBroadcast(Shape.Create(3, 1), Shape.Create(4), out Shape r));
        Assert.Equal(Shape.Create(3, 4), r);
        Assert.False(Shape.TryBroadcast(Shape.Create(3), Shape.Create(4), out _));
        Assert.Equal(2, Shape.NormalizeAxis(-1, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => Shape.NormalizeAxis(3, 3));
    }

    [Fact]
    public void TensorSizeLimitsAndViewBoundsAreChecked()
    {
        // 2^30 floats is 4 GiB: over the 2^31-1 byte storage limit, rejected before anything is allocated.
        Assert.Throws<TensorShapeException>(() => Tensor.Create(DType.F32, Shape.Create(1 << 30)));
        using Tensor t = Tensor.Create(DType.F32, Shape.Create(4, 4));
        Assert.Throws<TensorShapeException>(() => t.AsStrided(Shape.Create(4, 4), Strides.Create(8, 1), 0));
        Assert.Throws<TensorShapeException>(() => t.AsStrided(Shape.Create(2), Strides.Create(1), 15));
        Assert.Throws<TensorShapeException>(() => t.AsStrided(Shape.Create(2), Strides.Create(-1), 0));
        Assert.Throws<TensorShapeException>(() => t.Slice(0, 2, 5));
        Assert.Throws<TensorShapeException>(() => t.Slice(0, 0, 4, 0));
        Assert.Throws<TensorShapeException>(() => t.Expand(Shape.Create(3, 4)));
        Assert.Throws<TensorShapeException>(() => t.Squeeze(0));
        Assert.Throws<TensorShapeException>(() => t.Reshape(Shape.Create(5, 3)));
        Assert.Throws<TensorShapeException>(() => t.Permute(0, 0));
        Assert.Throws<ArgumentException>(() => t.Data<int>());
        Assert.Throws<ArgumentException>(() => t.Get<float>(1));
        Assert.Throws<ArgumentException>(() => t.Get<float>(4, 0));
    }

    // A deliberately naive n-d array: every operation builds a new row-major array by mapping each output index to an input index.
    private sealed class Ref(int[] shape, int[] data)
    {
        public int[] Shape { get; } = shape;
        public int[] Data { get; } = data;

        public Ref Map(int[] newShape, Func<int[], int[]> toOld)
        {
            int n = newShape.Aggregate(1, (a, b) => a * b);
            var data = new int[n];
            var idx = new int[newShape.Length];
            for (int i = 0; i < n; i++)
            {
                int rem = i;
                for (int d = newShape.Length - 1; d >= 0; d--)
                {
                    idx[d] = rem % newShape[d];
                    rem /= newShape[d];
                }

                int[] o = toOld(idx);
                int flat = 0;
                for (int d = 0; d < Shape.Length; d++)
                {
                    flat = (flat * Shape[d]) + o[d];
                }

                data[i] = Data[flat];
            }

            return new Ref(newShape, data);
        }
    }

    private static Tensor Iota(int[] shape)
    {
        int n = shape.Aggregate(1, (a, b) => a * b);
        return Tensor.FromSpan<int>(Enumerable.Range(0, n).ToArray(), Shape.Create(shape));
    }

    [Fact]
    public void RandomViewChainsMatchNaiveModel()
    {
        var rng = new Random(2024);
        int zeroCases = 0, sizeOneCases = 0, noncontig = 0;
        for (int iter = 0; iter < 3000; iter++)
        {
            int rank = rng.Next(0, 5);
            int[] shape = new int[rank];
            for (int i = 0; i < rank; i++)
            {
                shape[i] = rng.Next(0, 6) == 0 ? (rng.Next(2) == 0 ? 0 : 1) : rng.Next(1, 6);
            }

            using Tensor baseT = Iota(shape);
            Tensor cur = baseT;
            Ref model = new(shape, Enumerable.Range(0, baseT.Length).ToArray());
            int ops = rng.Next(1, 6);
            for (int k = 0; k < ops; k++)
            {
                int r = model.Shape.Length;
                int kind = rng.Next(8);
                switch (kind)
                {
                    case 0 when r >= 2:
                    {
                        int[] perm = Enumerable.Range(0, r).OrderBy(_ => rng.Next()).ToArray();
                        cur = cur.Permute(perm);
                        model = model.Map(perm.Select(p => model.Shape[p]).ToArray(), idx =>
                        {
                            var o = new int[r];
                            for (int i = 0; i < r; i++)
                            {
                                o[perm[i]] = idx[i];
                            }

                            return o;
                        });
                        break;
                    }

                    case 1 when r >= 1:
                    {
                        int a = rng.Next(r), size = model.Shape[a];
                        int start = rng.Next(0, size + 1), end = rng.Next(start, size + 1), step = rng.Next(1, 4);
                        cur = cur.Slice(a, start, end, step);
                        int count = (end - start + step - 1) / step;
                        int[] ns = (int[])model.Shape.Clone();
                        ns[a] = count;
                        model = model.Map(ns, idx =>
                        {
                            int[] o = (int[])idx.Clone();
                            o[a] = start + (idx[a] * step);
                            return o;
                        });
                        break;
                    }

                    case 2 when r >= 1:
                    {
                        int a = rng.Next(r);
                        if (model.Shape[a] == 0)
                        {
                            break;
                        }

                        int pick = rng.Next(model.Shape[a]);
                        cur = cur.Select(a, pick);
                        model = model.Map(model.Shape.Where((_, i) => i != a).ToArray(), idx =>
                        {
                            var o = new List<int>(idx);
                            o.Insert(a, pick);
                            return o.ToArray();
                        });
                        break;
                    }

                    case 3 when r >= 1:
                    {
                        int a = rng.Next(r), size = model.Shape[a];
                        cur = cur.Flip(a);
                        model = model.Map(model.Shape, idx =>
                        {
                            int[] o = (int[])idx.Clone();
                            o[a] = size - 1 - idx[a];
                            return o;
                        });
                        break;
                    }

                    case 4 when r < Shape.MaxRank:
                    {
                        int a = rng.Next(r + 1);
                        cur = cur.Unsqueeze(a);
                        var ns = new List<int>(model.Shape);
                        ns.Insert(a, 1);
                        model = model.Map(ns.ToArray(), idx => idx.Where((_, i) => i != a).ToArray());
                        break;
                    }

                    case 5 when r >= 1:
                    {
                        int a = rng.Next(r);
                        if (model.Shape[a] != 1)
                        {
                            break;
                        }

                        cur = cur.Squeeze(a);
                        model = model.Map(model.Shape.Where((_, i) => i != a).ToArray(), idx =>
                        {
                            var o = new List<int>(idx);
                            o.Insert(a, 0);
                            return o.ToArray();
                        });
                        break;
                    }

                    case 6 when r < Shape.MaxRank - 1 && model.Data.Length > 0:
                    {
                        int lead = rng.Next(0, 2);
                        int[] target = new int[r + lead];
                        for (int i = 0; i < target.Length; i++)
                        {
                            int src = i - lead;
                            target[i] = src < 0 ? rng.Next(1, 4) : model.Shape[src] == 1 ? rng.Next(1, 4) : model.Shape[src];
                        }

                        cur = cur.Expand(Shape.Create(target));
                        model = model.Map(target, idx => Enumerable.Range(0, r).Select(i => model.Shape[i] == 1 ? 0 : idx[i + lead]).ToArray());
                        break;
                    }

                    case 7:
                    {
                        int[] ns = RandomFactorization(rng, model.Data.Length);
                        bool viewable = cur.TryReshape(Shape.Create(ns), out Tensor v);
                        Tensor re = cur.Reshape(Shape.Create(ns));
                        Assert.Equal(model.Data, re.ToArray<int>());
                        if (viewable)
                        {
                            Assert.Equal(model.Data, v.ToArray<int>());
                            Assert.Same(cur.Buffer, v.Buffer);
                        }

                        cur = re;
                        model = new Ref(ns, model.Data);
                        break;
                    }
                }

                Assert.Equal(model.Shape, cur.Shape.Dimensions.ToArray());
                Assert.Equal(model.Data, cur.ToArray<int>());
            }

            if (model.Data.Length == 0)
            {
                zeroCases++;
            }

            if (model.Shape.Contains(1))
            {
                sizeOneCases++;
            }

            if (!cur.IsContiguous)
            {
                noncontig++;
            }

            // Materialization agrees, and a contiguous copy is contiguous.
            using Tensor copy = cur.Clone();
            Assert.True(copy.IsContiguous);
            Assert.Equal(model.Data, copy.Data<int>().ToArray());
            if (!ReferenceEquals(cur.Buffer, baseT.Buffer))
            {
                cur.Dispose(); // a reshape had to copy: that storage is ours
            }
        }

        Assert.True(zeroCases > 20 && sizeOneCases > 200 && noncontig > 300, $"coverage {zeroCases} {sizeOneCases} {noncontig}");
    }

    private static int[] RandomFactorization(Random rng, int n)
    {
        if (n == 0)
        {
            return rng.Next(2) == 0 ? [0] : [2, 0];
        }

        var dims = new List<int>();
        int rem = n;
        while (rem > 1 && dims.Count < 4)
        {
            var divisors = Enumerable.Range(1, rem).Where(d => rem % d == 0).ToArray();
            int d = divisors[rng.Next(divisors.Length)];
            dims.Add(d);
            rem /= d;
        }

        if (rem > 1)
        {
            dims.Add(rem);
        }

        if (rng.Next(4) == 0)
        {
            dims.Insert(rng.Next(dims.Count + 1), 1);
        }

        return dims.ToArray();
    }

    [Fact]
    public void ReshapeOfTransposedNeedsACopyAndCompatibleOnesDoNot()
    {
        using Tensor t = Iota([2, 3, 4]);
        Tensor tr = t.Permute(2, 0, 1); // (4,2,3)
        Assert.False(tr.TryReshape(Shape.Create(24), out _));
        Assert.True(tr.TryReshape(Shape.Create(4, 6), out Tensor v)); // merging the last two (contiguous run) is fine
        Assert.Equal(tr.ToArray<int>(), v.ToArray<int>());
        Assert.True(t.Slice(2, 0, 2).TryReshape(Shape.Create(2, 3, 2, 1), out _));
        Assert.False(t.Slice(2, 0, 2).TryReshape(Shape.Create(12), out _));
        Assert.Throws<InvalidOperationException>(() => tr.View(Shape.Create(24)));
        Tensor flat = tr.Reshape(24);
        Assert.NotSame(tr.Buffer, flat.Buffer);
        Assert.Equal(tr.ToArray<int>(), flat.ToArray<int>());
        flat.Dispose();
        Tensor inferred = t.Reshape(-1, 4);
        Assert.Equal(Shape.Create(6, 4), inferred.Shape);
        Assert.Throws<TensorShapeException>(() => t.Reshape(-1, -1));
        Assert.Throws<TensorShapeException>(() => t.Reshape(-1, 5));
    }

    [Fact]
    public void ViewsShareStorageAndMutationIsVisible()
    {
        using Tensor t = Tensor.Zeros(DType.F32, Shape.Create(3, 4));
        Tensor col = t.Select(1, 2);
        long v0 = t.Version;
        col.Fill(7f);
        Assert.True(t.Version > v0);
        Assert.Equal(7f, t.Get<float>(1, 2));
        Assert.Equal(0f, t.Get<float>(1, 1));
        Tensor flipped = t.Flip(1);
        Assert.Equal(7f, flipped.Get<float>(2, 1));
        t.Expand(Shape.Create(2, 3, 4)); // broadcast view builds without copying
        Assert.Equal(DType.F32, t.DType);
    }
}
