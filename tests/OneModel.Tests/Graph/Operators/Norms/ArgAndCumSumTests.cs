using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using static RecursiveAI.OneModel.Tests.Graph.Operators.Norms.NormHarness;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.Norms;

public class ArgAndCumSumTests
{
    private static object[] Dims(int[] d) => [.. d.Cast<object>()];

    // Scan from the first element; a later element wins when strictly better (or equal, when selecting the last).
    private static long[] ArgRef(bool max, bool last, float[] x, int[] shape, int axis)
    {
        int outer = shape.Take(axis).Aggregate(1, (a, b) => a * b), inner = shape.Skip(axis + 1).Aggregate(1, (a, b) => a * b), len = shape[axis];
        long[] y = new long[outer * inner];
        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                float At(int r) => x[(((o * len) + r) * inner) + i];
                int best = 0;
                for (int r = 1; r < len; r++)
                {
                    float v = At(r), b = At(best);
                    if (max ? (last ? v >= b : v > b) : (last ? v <= b : v < b))
                    {
                        best = r;
                    }
                }

                y[(o * inner) + i] = best;
            }
        }

        return y;
    }

    [Theory]
    [InlineData("ArgMax")]
    [InlineData("ArgMin")]
    public void ArgExtremeMatchesTheReference(string op)
    {
        var rng = new Random(op.Length);
        foreach (int[] shape in new[] { new[] { 3, 4, 5 }, new[] { 7 }, new[] { 2, 33 }, new[] { 1, 5, 1 } })
        {
            // Few distinct values, so ties are common; one NaN somewhere.
            float[] x = Enumerable.Range(0, Count(shape)).Select(_ => (float)rng.Next(4)).ToArray();
            x[rng.Next(x.Length)] = float.NaN;
            for (int axis = 0; axis < shape.Length; axis++)
            {
                foreach (bool last in new[] { false, true })
                {
                    foreach (bool keep in new[] { false, true })
                    {
                        int[] outDims = [.. shape.Select((d, i) => i == axis ? (keep ? 1 : -1) : d).Where(d => d >= 0)];
                        byte[] model = new ModelBuilder(13).Input("x", 1, Dims(shape)).Output("y", 7, Dims(outDims))
                            .Node(op, ["x"], ["y"], ("axis", (long)(axis - shape.Length)), ("keepdims", keep ? 1L : 0L), ("select_last_index", last ? 1L : 0L)).Build();
                        Out[] got = Run(model, ExecutionMode.Fast, null, 1, 1, (x, shape));
                        Assert.Equal(Shape.Create(outDims), got[0].Shape);
                        Assert.Equal(ArgRef(op == "ArgMax", last, x, shape, axis), got[0].L!);
                    }
                }
            }
        }
    }

    [Fact]
    public void ArgMaxIgnoresNaNLikeOnnxRuntime()
    {
        // Probed with ONNX Runtime: [1, nan, 3] -> 2, [nan, nan, nan] -> 0, [nan, nan, 5] -> 0.
        float n = float.NaN;
        byte[] model = new ModelBuilder(13).Input("x", 1, 3, 3).Output("y", 7, 3).Node("ArgMax", ["x"], ["y"], ("axis", 1L), ("keepdims", 0L)).Build();
        Assert.Equal([2L, 0L, 0L], Run(model, ExecutionMode.Fast, null, 1, 1, ([1, n, 3, n, n, n, n, n, 5], [3, 3]))[0].L!);
    }

    private static float[] CumRef(float[] x, int[] shape, int axis, bool exclusive, bool reverse)
    {
        int outer = shape.Take(axis).Aggregate(1, (a, b) => a * b), inner = shape.Skip(axis + 1).Aggregate(1, (a, b) => a * b), len = shape[axis];
        float[] y = new float[x.Length];
        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                float acc = 0;
                for (int step = 0; step < len; step++)
                {
                    int r = reverse ? len - 1 - step : step;
                    int at = (((o * len) + r) * inner) + i;
                    if (exclusive)
                    {
                        y[at] = acc;
                        acc += x[at];
                    }
                    else
                    {
                        acc += x[at];
                        y[at] = acc;
                    }
                }
            }
        }

        return y;
    }

    [Fact]
    public void CumSumMatchesTheReferenceInEveryDirection()
    {
        var rng = new Random(2);
        int[] shape = [3, 4, 5];
        float[] x = Random(rng, 60);
        for (int axis = -3; axis < 3; axis++)
        {
            foreach (bool exclusive in new[] { false, true })
            {
                foreach (bool reverse in new[] { false, true })
                {
                    byte[] model = new ModelBuilder(14).Input("x", 1, 3, 4, 5).Initializer("a", new long[] { axis }).Output("y", 1, 3, 4, 5)
                        .Node("CumSum", ["x", "a"], ["y"], ("exclusive", exclusive ? 1L : 0L), ("reverse", reverse ? 1L : 0L)).Build();
                    float[] got = Run(model, ExecutionMode.Exact, null, 1, 1, (x, shape))[0].F!;
                    AssertSameBits(got, CumRef(x, shape, axis < 0 ? axis + 3 : axis, exclusive, reverse), $"axis {axis} exclusive {exclusive} reverse {reverse}");
                }
            }
        }
    }
}
