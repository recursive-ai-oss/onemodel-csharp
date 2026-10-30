using System;
namespace RecursiveAI.OneModel.Tensors;

/// <summary>
/// The backward function of every <see cref="OpKind"/>: reads the record and the incoming gradient (contiguous, the output's
/// shape) and emits one gradient per input that needs one. Every gradient the engine handles is a contiguous float32 tensor of its
/// input's shape. Sums over broadcast or reduced axes accumulate in double in increasing element order, so the result does not
/// depend on threads; matrix products and convolutions run on the blocked kernels. docs/autograd.md lists each formula.
/// </summary>
internal static partial class BackwardOps
{
    public static void Run(Engine e, BackwardRecord r, in Tensor g)
    {
        switch (r.Op)
        {
            case OpKind.Add:
                EmitSummed(e, r, g, negateSecond: false);
                break;
            case OpKind.Sub:
                EmitSummed(e, r, g, negateSecond: true);
                break;
            case OpKind.Mul:
                if (r.NeedsInputGrad(0))
                {
                    Tensor t = Ops.Mul(g, r.Saved(1).Unpack());
                    Engine.Emit(r, 0, SumTo(t, r.InputShape(0), out bool same), owned: true);
                    DisposeIf(t, !same);
                }

                if (r.NeedsInputGrad(1))
                {
                    Tensor t = Ops.Mul(g, r.Saved(0).Unpack());
                    Engine.Emit(r, 1, SumTo(t, r.InputShape(1), out bool same), owned: true);
                    DisposeIf(t, !same);
                }

                break;
            case OpKind.Div:
                if (r.NeedsInputGrad(0))
                {
                    Tensor t = Ops.Div(g, r.Saved(1).Unpack());
                    Engine.Emit(r, 0, SumTo(t, r.InputShape(0), out bool same), owned: true);
                    DisposeIf(t, !same);
                }

                if (r.NeedsInputGrad(1))
                {
                    // d(a/b)/db = -a/b^2
                    Tensor b = r.Saved(1).Unpack();
                    Tensor ga = Ops.Mul(g, r.Saved(0).Unpack());
                    Tensor bb = Ops.Mul(b, b);
                    Tensor q = Ops.Div(ga, bb);
                    Tensor t = Ops.Neg(q);
                    ga.Dispose();
                    bb.Dispose();
                    q.Dispose();
                    Engine.Emit(r, 1, SumTo(t, r.InputShape(1), out bool same), owned: true);
                    DisposeIf(t, !same);
                }

                break;
            case OpKind.Neg:
                Engine.Emit(r, 0, Ops.Neg(g), owned: true);
                break;
            case OpKind.Relu:
                Engine.Emit(r, 0, Map2(g, r.Saved(0).Unpack(), static (gv, y) => y > 0f ? gv : 0f), owned: true);
                break;
            case OpKind.MatMul:
                if (r.NeedsInputGrad(0))
                {
                    Tensor b = r.Saved(1).Unpack();
                    Engine.Emit(r, 0, Ops.MatMul(g, b.T()), owned: true);
                }

                if (r.NeedsInputGrad(1))
                {
                    Tensor a = r.Saved(0).Unpack();
                    Engine.Emit(r, 1, Ops.MatMul(a.T(), g), owned: true);
                }

                break;
            case OpKind.MaxPool2d:
                MaxPoolBackward(e, r, g);
                break;
            case OpKind.Cast:
                Engine.Emit(r, 0, g.Cast(r.InputDType(0)), owned: true);
                break;
            case OpKind.Slice:
            {
                ReadOnlySpan<long> p = r.Ints;
                Tensor dx = Tensor.Zeros(DType.F32, r.InputShape(0));
                dx.Slice((int)p[0], (int)p[1], (int)p[2], (int)p[3]).CopyFrom(g);
                Engine.Emit(r, 0, dx, owned: true);
                break;
            }

            case OpKind.Select:
            {
                ReadOnlySpan<long> p = r.Ints;
                Tensor dx = Tensor.Zeros(DType.F32, r.InputShape(0));
                dx.Select((int)p[0], (int)p[1]).CopyFrom(g);
                Engine.Emit(r, 0, dx, owned: true);
                break;
            }

            case OpKind.Permute:
            {
                ReadOnlySpan<long> p = r.Ints;
                Span<int> inv = stackalloc int[p.Length];
                for (int i = 0; i < p.Length; i++)
                {
                    inv[(int)p[i]] = i;
                }

                Engine.Emit(r, 0, g.Permute(inv).Clone(), owned: true);
                break;
            }

            case OpKind.Reshape:
                e.EmitAliased(r, 0, g.Reshape(r.InputShape(0)));
                break;
            case OpKind.Expand:
            {
                Tensor s = SumTo(g, r.InputShape(0), out bool same);
                if (same)
                {
                    e.EmitAliased(r, 0, g);
                }
                else
                {
                    Engine.Emit(r, 0, s, owned: true);
                }

                break;
            }

            case OpKind.Flip:
                Engine.Emit(r, 0, g.Flip((int)r.Ints[0]).Clone(), owned: true);
                break;
            default:
                RunNn(e, r, g);
                break;
        }
    }

    private static void DisposeIf(in Tensor t, bool condition)
    {
        if (condition)
        {
            t.Dispose();
        }
    }

    // Add and Sub: the output gradient summed down to each input's shape; Sub negates the second.
    private static void EmitSummed(Engine e, BackwardRecord r, in Tensor g, bool negateSecond)
    {
        bool need0 = r.NeedsInputGrad(0), need1 = r.NeedsInputGrad(1);
        if (need0)
        {
            Tensor s = SumTo(g, r.InputShape(0), out bool same);
            if (!same)
            {
                Engine.Emit(r, 0, s, owned: true);
            }
            else if (!need1)
            {
                e.EmitAliased(r, 0, g);
            }
            else
            {
                Engine.Emit(r, 0, g, owned: false);
            }
        }

        if (need1)
        {
            Tensor s = SumTo(g, r.InputShape(1), out bool same);
            if (negateSecond)
            {
                Tensor n = Ops.Neg(same ? g : s);
                DisposeIf(s, !same);
                Engine.Emit(r, 1, n, owned: true);
            }
            else if (!same)
            {
                Engine.Emit(r, 1, s, owned: true);
            }
            else
            {
                e.EmitAliased(r, 1, g);
            }
        }
    }

    /// <summary>
    /// Sums <paramref name="g"/> down to <paramref name="target"/> (undoing a broadcast: leading axes and axes where the target has
    /// extent 1). Returns <paramref name="g"/> itself with <paramref name="same"/> true when the shapes already match. Accumulates in
    /// double in increasing element order.
    /// </summary>
    public static Tensor SumTo(in Tensor g, Shape target, out bool same)
    {
        same = g.Shape == target;
        if (same)
        {
            return g;
        }

        int rank = g.Rank, lead = rank - target.Rank;
        if (lead < 0)
        {
            throw new TensorShapeException($"Cannot sum {g.Shape} down to {target}.");
        }

        Span<int> tstride = stackalloc int[Shape.MaxRank];
        int acc = 1;
        for (int ax = rank - 1; ax >= 0; ax--)
        {
            int t = ax - lead;
            if (t < 0 || target[t] == 1)
            {
                tstride[ax] = 0;
            }
            else
            {
                tstride[ax] = acc;
                acc *= target[t];
            }
        }

        Tensor scratch = Tensor.Create(DType.F64, target, zero: true);
        Span<double> sum = scratch.MutableData<double>();
        ReadOnlySpan<float> src = g.Data<float>();
        Span<int> idx = stackalloc int[Shape.MaxRank];
        int inner = rank == 0 ? 1 : g.Shape[rank - 1];
        int innerStride = rank == 0 ? 0 : tstride[rank - 1];
        int outer = inner == 0 ? 0 : src.Length / inner;
        int baseOff = 0;
        for (int o = 0; o < outer; o++)
        {
            for (int i = 0, t = baseOff; i < inner; i++, t += innerStride)
            {
                sum[t] += src[(o * inner) + i];
            }

            for (int ax = rank - 2; ax >= 0; ax--)
            {
                idx[ax]++;
                baseOff += tstride[ax];
                if (idx[ax] < g.Shape[ax])
                {
                    break;
                }

                baseOff -= tstride[ax] * g.Shape[ax];
                idx[ax] = 0;
            }
        }

        Tensor result = Tensor.Create(DType.F32, target);
        Span<float> dst = result.MutableData<float>();
        for (int i = 0; i < dst.Length; i++)
        {
            dst[i] = (float)sum[i];
        }

        scratch.Dispose();
        return result;
    }

    /// <summary>A new contiguous tensor <c>f(a[i], b[i])</c> over two same-shape contiguous tensors.</summary>
    public static Tensor Map2(in Tensor a, in Tensor b, Func<float, float, float> f)
    {
        Tensor r = Tensor.Create(DType.F32, a.Shape);
        Span<float> d = r.MutableData<float>();
        ReadOnlySpan<float> x = a.Data<float>(), y = b.Data<float>();
        for (int i = 0; i < d.Length; i++)
        {
            d[i] = f(x[i], y[i]);
        }

        return r;
    }

    private static void MaxPoolBackward(Engine e, BackwardRecord r, in Tensor g)
    {
        Tensor idx = r.Saved(0).Unpack();
        Tensor dx = Tensor.Zeros(DType.F32, r.InputShape(0));
        Span<float> d = dx.MutableData<float>();
        ReadOnlySpan<float> gv = g.Data<float>();
        ReadOnlySpan<int> iv = idx.Data<int>();
        for (int i = 0; i < gv.Length; i++)
        {
            d[iv[i]] += gv[i];
        }

        Engine.Emit(r, 0, dx, owned: true);
    }
}
