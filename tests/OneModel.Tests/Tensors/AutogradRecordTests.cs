using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Tensors;

// A throwaway backward engine written against BackwardRecord only, to show the record holds what each operation's backward
// needs. It uses plain double loops for the reductions and the library's own MatMul on views for the matrix products.
internal static class MiniBackward
{
    public static void Run(Tensor root, Tensor seed)
    {
        var order = new List<AutogradMeta>();
        var seen = new HashSet<AutogradMeta>();
        Visit(root.Autograd!);
        var grads = new Dictionary<AutogradMeta, Tensor> { [root.Autograd!] = seed };
        using NoGradScope scope = GradMode.NoGrad();
        for (int i = order.Count - 1; i >= 0; i--)
        {
            AutogradMeta m = order[i];
            if (!grads.TryGetValue(m, out Tensor g))
            {
                continue;
            }

            if (m.GradFn is null)
            {
                m.Grad = g;
                continue;
            }

            Tensor?[] ins = Apply(m.GradFn, g);
            for (int k = 0; k < ins.Length; k++)
            {
                AutogradMeta? im = m.GradFn.InputMeta(k);
                if (im is null || ins[k] is not { } gi)
                {
                    continue;
                }

                grads[im] = grads.TryGetValue(im, out Tensor acc) ? Ops.Add(acc, gi) : gi;
            }
        }

        void Visit(AutogradMeta m)
        {
            if (!seen.Add(m))
            {
                return;
            }

            if (m.GradFn is { } f)
            {
                for (int k = 0; k < f.InputCount; k++)
                {
                    if (f.InputMeta(k) is { } im)
                    {
                        Visit(im);
                    }
                }
            }

            order.Add(m);
        }
    }

    private static Tensor?[] Apply(BackwardRecord r, Tensor g)
    {
        switch (r.Op)
        {
            case OpKind.Add:
                return [SumTo(g, r.InputShape(0)), SumTo(g, r.InputShape(1))];
            case OpKind.Mul:
            {
                Tensor? da = r.NeedsInputGrad(0) ? SumTo(Ops.Mul(g, r.Saved(1).Unpack()), r.InputShape(0)) : null;
                Tensor? db = r.NeedsInputGrad(1) ? SumTo(Ops.Mul(g, r.Saved(0).Unpack()), r.InputShape(1)) : null;
                return [da, db];
            }

            case OpKind.Relu:
            {
                Tensor y = r.Saved(0).Unpack();
                Tensor dx = Tensor.Create(DType.F32, g.Shape);
                float[] gv = g.ToArray<float>(), yv = y.ToArray<float>();
                Span<float> o = dx.MutableData<float>();
                for (int i = 0; i < o.Length; i++)
                {
                    o[i] = yv[i] > 0 ? gv[i] : 0;
                }

                return [dx];
            }

            case OpKind.MatMul:
            {
                // dA = g · Bᵀ and dB = Aᵀ · g, the transposes being zero-copy views of the saved operands.
                Tensor? da = r.NeedsInputGrad(0) ? Ops.MatMul(g, r.Saved(1).Unpack().T()) : null;
                Tensor? db = r.NeedsInputGrad(1) ? Ops.MatMul(r.Saved(0).Unpack().T(), g) : null;
                return [da, db];
            }

            case OpKind.MaxPool2d:
            {
                Tensor idx = r.Saved(0).Unpack();
                Tensor dx = Tensor.Zeros(DType.F32, r.InputShape(0));
                Span<float> o = dx.MutableData<float>();
                float[] gv = g.ToArray<float>();
                int[] iv = idx.ToArray<int>();
                for (int i = 0; i < iv.Length; i++)
                {
                    o[iv[i]] += gv[i];
                }

                return [dx];
            }

            case OpKind.Reshape:
                return [g.Reshape(r.InputShape(0))];
            case OpKind.Permute:
            {
                long[] p = r.Ints.ToArray();
                var inv = new int[p.Length];
                for (int i = 0; i < p.Length; i++)
                {
                    inv[p[i]] = i;
                }

                return [g.Permute(inv)];
            }

            case OpKind.Expand:
                return [SumTo(g, r.InputShape(0))];
            case OpKind.Select:
            {
                Tensor dx = Tensor.Zeros(DType.F32, r.InputShape(0));
                dx.Select((int)r.Ints[0], (int)r.Ints[1]).CopyFrom(g);
                return [dx];
            }

            default:
                throw new NotSupportedException(r.Op.ToString());
        }
    }

    // Sums g (broadcast result shape) down to `shape`, the gradient of an implicit broadcast.
    public static Tensor SumTo(Tensor g, Shape shape)
    {
        if (g.Shape == shape)
        {
            return g;
        }

        Tensor outT = Tensor.Zeros(DType.F32, shape);
        Span<float> o = outT.MutableData<float>();
        float[] gv = g.ToArray<float>();
        int lead = g.Rank - shape.Rank;
        var idx = new int[g.Rank];
        for (int i = 0; i < gv.Length; i++)
        {
            int rem = i;
            for (int d = g.Rank - 1; d >= 0; d--)
            {
                idx[d] = rem % g.Shape[d];
                rem /= g.Shape[d];
            }

            int flat = 0;
            for (int d = 0; d < shape.Rank; d++)
            {
                flat = (flat * shape[d]) + (shape[d] == 1 ? 0 : idx[d + lead]);
            }

            o[flat] += gv[i];
        }

        return outT;
    }
}

public class AutogradRecordTests
{
    [Fact]
    public void NothingIsRecordedWhenGradientsAreOff()
    {
        using Tensor a = Tensor.Zeros(DType.F32, Shape.Create(2, 3));
        using Tensor b = Tensor.Zeros(DType.F32, Shape.Create(3));
        Tensor c = Ops.Add(a, b);
        Assert.False(c.RequiresGrad);
        Assert.Null(c.Autograd);

        Tensor ta = a.WithRequiresGrad();
        using (GradMode.NoGrad())
        {
            Assert.False(Ops.Add(ta, b).RequiresGrad);
            Assert.False(GradMode.IsEnabled);
        }

        Assert.True(GradMode.IsEnabled);
        Tensor d = Ops.Add(ta, b);
        Assert.True(d.RequiresGrad);
        Assert.Equal(OpKind.Add, d.Autograd!.GradFn!.Op);
        Assert.False(d.Autograd.GradFn.NeedsInputGrad(1));
        Assert.Equal(Shape.Create(3), d.Autograd.GradFn.InputShape(1));
        Assert.Throws<InvalidOperationException>(() => Tensor.Zeros(DType.I32, Shape.Create(1)).WithRequiresGrad());
        c.Dispose();
        d.Dispose();
    }

    [Fact]
    public void MutatingASavedTensorIsDetected()
    {
        using Tensor a = Tensor.FromSpan<float>([1, 2, 3, 4], Shape.Create(2, 2));
        using Tensor b = Tensor.FromSpan<float>([5, 6, 7, 8], Shape.Create(2, 2));
        Tensor wa = a.WithRequiresGrad(), wb = b.WithRequiresGrad();
        Tensor c = Ops.Mul(wa, wb);
        BackwardRecord rec = c.Autograd!.GradFn!;
        Assert.Equal(a.ToArray<float>(), rec.Saved(0).Unpack().ToArray<float>());

        a.Slice(0, 1, 2).Fill(0f); // an in-place write through a view of the saved storage
        Assert.Throws<TensorVersionException>(() => rec.Saved(0).Unpack());
        Assert.Equal(b.ToArray<float>(), rec.Saved(1).Unpack().ToArray<float>()); // the other operand is untouched
        Assert.Throws<TensorVersionException>(() =>
        {
            using Tensor x = Tensor.FromSpan<float>([1, -2], Shape.Create(2));
            Tensor y = Ops.Relu(x.WithRequiresGrad());
            y.MutableData<float>()[0] = 9; // mutating the output the record saved
            y.Autograd!.GradFn!.Saved(0).Unpack();
        });
        c.Dispose();
    }

    [Fact]
    public void RecordsSufficeForGradientsThroughAChainAgainstFiniteDifferences()
    {
        var rng = new Random(99);
        float[] Rand(int n, double s = 1) => Enumerable.Range(0, n).Select(_ => (float)(((rng.NextDouble() * 2) - 1) * s)).ToArray();

        // x(1,2,4,4) * scale(1,2,1,1) + bias(2,1,1) -> relu -> maxpool 2x2 -> reshape (1,8) -> matmul with wT view (8,3)
        float[] xv = Rand(32), sv = Rand(2), bv = Rand(2), wv = Rand(24), cv = Rand(3);
        using Tensor x = Tensor.FromSpan<float>(xv, Shape.Create(1, 2, 4, 4));
        using Tensor scale = Tensor.FromSpan<float>(sv, Shape.Create(1, 2, 1, 1));
        using Tensor bias = Tensor.FromSpan<float>(bv, Shape.Create(2, 1, 1));
        using Tensor w = Tensor.FromSpan<float>(wv, Shape.Create(3, 8));
        Tensor s = scale.WithRequiresGrad(), b = bias.WithRequiresGrad(), wt = w.WithRequiresGrad();

        Tensor h = Ops.Add(Ops.Mul(x, s), b);
        Tensor r = Ops.Relu(h);
        Tensor p = Ops.MaxPool2d(r, new Pool2dParams(2, 2, 2, 2, 0, 0));
        Tensor flat = p.Reshape(1, 8);
        Tensor y = Ops.MatMul(flat, wt.T());
        Assert.Equal(Shape.Create(1, 3), y.Shape);

        using Tensor seed = Tensor.FromSpan<float>(cv, Shape.Create(1, 3));
        MiniBackward.Run(y, seed);

        // Double-precision reference forward and central differences.
        double Loss(double[] sc, double[] bi, double[] ww)
        {
            var pooled = new double[8];
            for (int c = 0; c < 2; c++)
            {
                for (int oy = 0; oy < 2; oy++)
                {
                    for (int ox = 0; ox < 2; ox++)
                    {
                        double best = double.NegativeInfinity;
                        for (int dy = 0; dy < 2; dy++)
                        {
                            for (int dx = 0; dx < 2; dx++)
                            {
                                double v = (xv[(c * 16) + (((oy * 2) + dy) * 4) + (ox * 2) + dx] * sc[c]) + bi[c];
                                best = Math.Max(best, Math.Max(v, 0));
                            }
                        }

                        pooled[(c * 4) + (oy * 2) + ox] = best;
                    }
                }
            }

            double loss = 0;
            for (int o = 0; o < 3; o++)
            {
                double acc = 0;
                for (int k = 0; k < 8; k++)
                {
                    acc += pooled[k] * ww[(o * 8) + k];
                }

                loss += acc * cv[o];
            }

            return loss;
        }

        double[] S() => sv.Select(v => (double)v).ToArray();
        double[] B() => bv.Select(v => (double)v).ToArray();
        double[] W() => wv.Select(v => (double)v).ToArray();
        void Check(float[] analytic, double[] p0, Func<double[], double[], double[], double> f, int which)
        {
            for (int i = 0; i < p0.Length; i++)
            {
                double[][] up = [S(), B(), W()], dn = [S(), B(), W()];
                up[which][i] += 1e-5;
                dn[which][i] -= 1e-5;
                double fd = (f(up[0], up[1], up[2]) - f(dn[0], dn[1], dn[2])) / 2e-5;
                Assert.True(Math.Abs(fd - analytic[i]) <= 2e-3 * Math.Max(1, Math.Abs(fd)), $"param {which}[{i}]: fd {fd} vs {analytic[i]}");
            }
        }

        Check(s.Autograd!.Grad.ToArray<float>(), S(), Loss, 0);
        Check(b.Autograd!.Grad.ToArray<float>(), B(), Loss, 1);
        Check(wt.Autograd!.Grad.ToArray<float>(), W(), Loss, 2);
        Assert.Equal(Shape.Create(2, 1, 1), b.Autograd.Grad.Shape);
        Assert.Equal(Shape.Create(3, 8), wt.Autograd.Grad.Shape);
    }

    [Fact]
    public void ViewRecordsCarryWhatTheInverseNeeds()
    {
        using Tensor x = Tensor.FromSpan<float>(Enumerable.Range(0, 24).Select(i => (float)i).ToArray(), Shape.Create(2, 3, 4));
        Tensor wx = x.WithRequiresGrad();
        Tensor p = wx.Permute(2, 0, 1);
        Assert.Equal(OpKind.Permute, p.Autograd!.GradFn!.Op);
        Assert.Equal(new long[] { 2, 0, 1 }, p.Autograd.GradFn.Ints.ToArray());
        Tensor sl = wx.Slice(2, 1, 4, 2);
        Assert.Equal(new long[] { 2, 1, 4, 2 }, sl.Autograd!.GradFn!.Ints.ToArray());
        Assert.Equal(wx.Shape, sl.Autograd.GradFn.InputShape(0));
        Tensor re = p.Reshape(24); // not viewable: copy then reshape, still one Reshape record on the permuted input
        Assert.Equal(OpKind.Reshape, re.Autograd!.GradFn!.Op);
        Assert.Equal(p.Shape, re.Autograd.GradFn.InputShape(0));
        Tensor e = wx.Select(0, 1).Unsqueeze(0).Expand(Shape.Create(5, 1, 3, 4));
        Assert.Equal(OpKind.Expand, e.Autograd!.GradFn!.Op);

        // Gradient through permute + reshape + expand by the mini engine: d(sum(c * expand)) is the broadcast-summed seed.
        using Tensor seed = Tensor.Zeros(DType.F32, e.Shape);
        seed.Fill(1f);
        MiniBackward.Run(e, seed);
        float[] g = wx.Autograd!.Grad.ToArray<float>();
        Assert.Equal(24, g.Length);
        Assert.Equal(0f, g[0]);   // the first outer slice was not selected
        Assert.Equal(5f, g[12]);  // each selected element is counted once per broadcast copy
        re.Dispose();
    }
}
