using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using static RecursiveAI.OneModel.Tests.Graph.Operators.Norms.NormHarness;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.Norms;

public class SoftmaxTests
{
    private static object[] Dims(int[] d) => [.. d.Cast<object>()];

    // Reference along one axis of a row-major array (double).
    private static double[] Ref(string op, float[] x, int[] shape, int axis)
    {
        int outer = 1, inner = 1, len = shape[axis];
        for (int i = 0; i < shape.Length; i++)
        {
            if (i < axis)
            {
                outer *= shape[i];
            }
            else if (i > axis)
            {
                inner *= shape[i];
            }
        }

        double[] y = new double[x.Length];
        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                int At(int r) => (((o * len) + r) * inner) + i;
                double max = double.NegativeInfinity;
                for (int r = 0; r < len; r++)
                {
                    max = Math.Max(max, x[At(r)]);
                }

                double sum = 0;
                for (int r = 0; r < len; r++)
                {
                    sum += Math.Exp(x[At(r)] - max);
                }

                int best = 0;
                for (int r = 1; r < len; r++)
                {
                    if (x[At(r)] > x[At(best)])
                    {
                        best = r;
                    }
                }

                for (int r = 0; r < len; r++)
                {
                    y[At(r)] = op switch
                    {
                        "Softmax" => Math.Exp(x[At(r)] - max) / sum,
                        "LogSoftmax" => x[At(r)] - max - Math.Log(sum),
                        _ => r == best ? 1 : 0,
                    };
                }
            }
        }

        return y;
    }

    // Opset 12 and earlier flatten to [prod(shape[:axis]), prod(shape[axis:])].
    private static double[] RefCoerce(string op, float[] x, int[] shape, int axis)
    {
        int rows = shape.Take(axis).Aggregate(1, (a, b) => a * b);
        return Ref(op, x, [rows, x.Length / Math.Max(1, rows)], 1);
    }

    [Theory]
    [InlineData("Softmax")]
    [InlineData("LogSoftmax")]
    [InlineData("Hardmax")]
    public void NewAxisSemanticsAlongAnyAxis(string op)
    {
        var rng = new Random(op.Length);
        foreach (int[] shape in new[] { new[] { 3, 4, 5 }, new[] { 2, 37 }, new[] { 5 }, new[] { 2, 1, 6, 3 }, new[] { 4, 33, 2 } })
        {
            float[] x = Random(rng, Count(shape), -6, 6);
            for (int axis = -shape.Length; axis < shape.Length; axis++)
            {
                byte[] model = new ModelBuilder(13).Input("x", 1, Dims(shape)).Output("y", 1, Dims(shape)).Node(op, ["x"], ["y"], ("axis", (long)axis)).Build();
                double[] want = Ref(op, x, shape, axis < 0 ? axis + shape.Length : axis);
                Out[] fast = Run(model, ArithmeticMode.Fast, null, 1, 1, (x, shape));
                Assert.True(ScaledError(fast[0].F!, want) < 1e-5, $"{op} {string.Join('x', shape)} axis {axis}: {ScaledError(fast[0].F!, want)}");
                Out[] many = Run(model, ArithmeticMode.Fast, null, 8, 1, (x, shape));
                AssertSameBits(many[0].F!, fast[0].F!, "threads");
                if (op != "Hardmax")
                {
                    Out[] exact = Run(model, ArithmeticMode.Exact, null, 1, 1, (x, shape));
                    Assert.True(ScaledError(exact[0].F!, want) < 1e-5);
                    foreach (var path in Paths)
                    {
                        AssertSameBits(Run(model, ArithmeticMode.Exact, path, 3, 1, (x, shape))[0].F!, exact[0].F!, $"{op} exact {path}");
                    }
                }
            }
        }
    }

    [Theory]
    [InlineData("Softmax")]
    [InlineData("LogSoftmax")]
    [InlineData("Hardmax")]
    public void OldAxisSemanticsCoerceTo2D(string op)
    {
        var rng = new Random(7);
        int[] shape = [3, 4, 5];
        float[] x = Random(rng, 60, -6, 6);
        foreach (int opset in new[] { 1, 11 })
        {
            for (int axis = 0; axis < 3; axis++)
            {
                byte[] model = new ModelBuilder(opset).Input("x", 1, 3, 4, 5).Output("y", 1, 3, 4, 5).Node(op, ["x"], ["y"], ("axis", (long)axis)).Build();
                Out[] got = Run(model, ArithmeticMode.Fast, null, 1, 1, (x, shape));
                Assert.True(ScaledError(got[0].F!, RefCoerce(op, x, shape, axis)) < 1e-5, $"{op} opset {opset} axis {axis}");
            }
        }

        // The default axis is 1 up to opset 12.
        byte[] def = new ModelBuilder(11).Input("x", 1, 3, 4, 5).Output("y", 1, 3, 4, 5).Node(op, ["x"], ["y"]).Build();
        Assert.True(ScaledError(Run(def, ArithmeticMode.Fast, null, 1, 1, (x, shape))[0].F!, RefCoerce(op, x, shape, 1)) < 1e-5);
    }

    [Fact]
    public void SpecialRowsMatchOnnxRuntime()
    {
        float inf = float.PositiveInfinity, nan = float.NaN;
        float[] x = [inf, 1, -inf, -inf, nan, 1, -inf, 1];
        byte[] log = new ModelBuilder(13).Input("x", 1, 4, 2).Output("y", 1, 4, 2).Node("LogSoftmax", ["x"], ["y"]).Build();
        byte[] soft = new ModelBuilder(13).Input("x", 1, 4, 2).Output("y", 1, 4, 2).Node("Softmax", ["x"], ["y"]).Build();
        foreach (var path in Paths)
        {
            foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
            {
                float[] l = Run(log, mode, path, 1, 1, (x, [4, 2]))[0].F!;
                float[] s = Run(soft, mode, path, 1, 1, (x, [4, 2]))[0].F!;
                for (int i = 0; i < 6; i++)
                {
                    Assert.True(float.IsNaN(l[i]) && float.IsNaN(s[i]), $"{mode} {path} element {i}");
                }

                Assert.Equal(float.NegativeInfinity, l[6]);
                Assert.Equal(0f, l[7], 1e-6);
                Assert.Equal(0f, s[6]);
                Assert.Equal(1f, s[7], 1e-6);
            }
        }
    }

    [Fact]
    public void HardmaxTiesAndNaNFollowOnnxRuntime()
    {
        float[] x = [float.NaN, 1, 1, 2, 2, 1];
        byte[] model = new ModelBuilder(13).Input("x", 1, 2, 3).Output("y", 1, 2, 3).Node("Hardmax", ["x"], ["y"]).Build();
        float[] y = Run(model, ArithmeticMode.Fast, null, 1, 1, (x, [2, 3]))[0].F!;
        Assert.Equal([0f, 0f, 0f, 1f, 0f, 0f], y);
    }
}
