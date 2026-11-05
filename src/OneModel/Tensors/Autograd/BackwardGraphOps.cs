using System;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
namespace RecursiveAI.OneModel.Tensors;

// Backward functions of the operations in Ops.GraphOps.cs. Same conventions as BackwardOps: the incoming gradient is contiguous at
// the output's shape, every emitted gradient is contiguous at its input's shape, sums over broadcast axes accumulate in double.
internal static partial class BackwardOps
{
    private static void RunGraphOps(Engine e, BackwardRecord r, in Tensor g)
    {
        switch (r.Op)
        {
            case OpKind.BatchMatMul:
                if (r.NeedsInputGrad(0))
                {
                    Tensor b = r.Saved(1).Unpack();
                    Tensor t = Ops.MatMulBatched(g, b.Transpose(-1, -2));
                    Engine.Emit(r, 0, SumTo(t, r.InputShape(0), out bool same), owned: true);
                    DisposeIf(t, !same);
                }

                if (r.NeedsInputGrad(1))
                {
                    Tensor a = r.Saved(0).Unpack();
                    Tensor t = Ops.MatMulBatched(a.Transpose(-1, -2), g);
                    Engine.Emit(r, 1, SumTo(t, r.InputShape(1), out bool same), owned: true);
                    DisposeIf(t, !same);
                }

                break;
            case OpKind.Erf:
                Engine.Emit(r, 0, VectorMaps.Map2(new ErfGradBin(), g, r.Saved(0).Unpack()), owned: true);
                break;
            case OpKind.Exp:
                Engine.Emit(r, 0, VectorMaps.Map2(new MulBin(), g, r.Saved(0).Unpack()), owned: true);
                break;
            case OpKind.Sqrt:
                Engine.Emit(r, 0, VectorMaps.Map2(new SqrtGradBin(), g, r.Saved(0).Unpack()), owned: true);
                break;
            case OpKind.Pow:
            {
                float p = (float)r.Floats[0];
                Tensor x = r.Saved(0).Unpack();
                Tensor dx = Tensor.Create(DType.F32, x.Shape);
                ReadOnlySpan<float> gv = g.InternalData<float>(), xv = x.InternalData<float>();
                Span<float> d = dx.MutableData<float>();
                for (int i = 0; i < d.Length; i++)
                {
                    d[i] = gv[i] * PowDerivative(xv[i], p);
                }

                Engine.Emit(r, 0, dx, owned: true);
                break;
            }

            case OpKind.Where:
                WhereBackward(r, g);
                break;
            case OpKind.Gather:
                GatherBackward(r, g);
                break;
            default:
                throw new NotSupportedException($"No backward function for {r.Op}.");
        }
    }

    // d/dx x^p; the exact exponents use only IEEE operations.
    private static float PowDerivative(float x, float p) => p switch
    {
        1f => 1f,
        2f => x + x,
        0.5f => 1f / (MathF.Sqrt(x) + MathF.Sqrt(x)),
        -1f => -1f / (x * x),
        _ => p * MathF.Pow(x, p - 1f),
    };

    private static void WhereBackward(BackwardRecord r, in Tensor g)
    {
        ReadOnlySpan<bool> c = r.Saved(0).Unpack().InternalData<bool>();
        ReadOnlySpan<float> gv = g.InternalData<float>();
        for (int side = 0; side < 2; side++)
        {
            int input = side + 1;
            if (!r.NeedsInputGrad(input))
            {
                continue;
            }

            Tensor t = Tensor.Create(DType.F32, g.Shape);
            Span<float> d = t.MutableData<float>();
            bool wanted = side == 0;
            for (int i = 0; i < d.Length; i++)
            {
                d[i] = c[i] == wanted ? gv[i] : 0f;
            }

            Engine.Emit(r, input, SumTo(t, r.InputShape(input), out bool same), owned: true);
            DisposeIf(t, !same);
        }
    }

    private static void GatherBackward(BackwardRecord r, in Tensor g)
    {
        Tensor idx = r.Saved(0).Unpack();
        Shape ds = r.InputShape(0);
        int ax = (int)r.Ints[0];
        int outer = 1, inner = 1, n = ds[ax];
        for (int i = 0; i < ax; i++)
        {
            outer *= ds[i];
        }

        for (int i = ax + 1; i < ds.Rank; i++)
        {
            inner *= ds[i];
        }

        Tensor dx = Tensor.Zeros(DType.F32, ds);
        Span<float> d = dx.MutableData<float>();
        ReadOnlySpan<float> gv = g.InternalData<float>();
        int count = idx.Length;
        for (int o = 0; o < outer; o++)
        {
            for (int j = 0; j < count; j++)
            {
                int id = (int)Ops.GatherIndex(idx, j, n);
                ReadOnlySpan<float> src = gv.Slice(((o * count) + j) * inner, inner);
                Span<float> dst = d.Slice(((o * n) + id) * inner, inner);
                for (int k = 0; k < inner; k++)
                {
                    dst[k] += src[k];
                }
            }
        }

        Engine.Emit(r, 0, dx, owned: true);
    }
}

/// <summary>Erf backward: g · (2/√π) · e^(−x²), multiplied left to right (b is the saved input).</summary>
internal readonly struct ErfGradBin : IFloatBinary
{
    public TV Apply<TV, TO>(TV g, TV x) where TV : struct where TO : IVecOps<TV> =>
        TO.Mul(TO.Mul(g, TO.Splat(1.1283792f)), Transcendental.Exp<TV, TO>(TO.Sub(TO.Zero, TO.Mul(x, x))));
}

/// <summary>Square-root backward: g / (y + y) (b is the saved output).</summary>
internal readonly struct SqrtGradBin : IFloatBinary
{
    public TV Apply<TV, TO>(TV g, TV y) where TV : struct where TO : IVecOps<TV> => TO.Div(g, TO.Add(y, y));
}
