using System.Globalization;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Training;

namespace RecursiveAI.OneModel.Tests.Tensors;

// Central finite differences against the analytic gradients. The operations are float32, so the numerical derivative is taken in
// float32 with a step large enough to beat rounding (error ~ eps * |L| / h) and small enough for the curvature (error ~ h^2 f'''),
// and the objective L = sum(W * f(x)) with a random fixed W is accumulated in double. Inputs sit away from the kinks of
// piecewise functions by construction.
internal static class GradCheck
{
    public sealed record Outcome(string Name, double MaxAbs, double MaxScaled, int Elements);

    public static Func<Random, float> Uniform(float lo, float hi) => rng => lo + ((hi - lo) * (float)rng.NextDouble());

    // Magnitude in [lo, hi] with a random sign: keeps relu / max away from zero.
    public static Func<Random, float> AwayFromZero(float lo, float hi) => rng => (rng.Next(2) == 0 ? -1f : 1f) * (lo + ((hi - lo) * (float)rng.NextDouble()));

    public static Tensor Fill(Shape shape, Random rng, Func<Random, float> gen)
    {
        Tensor t = Tensor.Create(DType.F32, shape);
        Span<float> d = t.MutableData<float>();
        for (int i = 0; i < d.Length; i++)
        {
            d[i] = gen(rng);
        }

        return t;
    }

    /// <summary>Checks every input of <paramref name="f"/>; returns the worst error over inputs.</summary>
    public static Outcome Run(string name, Func<Tensor[], Tensor> f, Shape[] shapes, Func<Random, float>[]? gens = null,
        double h = 1e-2, double tolerance = 5e-3, int seed = 1234, bool[]? check = null)
    {
        var rng = new Random(seed);
        var leaves = new Tensor[shapes.Length];
        for (int i = 0; i < shapes.Length; i++)
        {
            leaves[i] = Fill(shapes[i], rng, gens?[i] ?? Uniform(-1f, 1f)).WithRequiresGrad();
        }

        GradientCheckResult[] results = GradientCheck.Run(f, leaves, h, seed + 1);
        double maxAbs = 0, maxScaled = 0;
        int count = 0;
        for (int i = 0; i < leaves.Length; i++)
        {
            if (check is not null && !check[i])
            {
                continue;
            }

            Assert.Equal(shapes[i], leaves[i].Grad.Shape);
            GradientCheckResult r = results[i];
            maxAbs = Math.Max(maxAbs, r.MaxAbsError);
            maxScaled = Math.Max(maxScaled, r.ScaledError);
            count += r.Elements;
            Assert.True(r.ScaledError <= tolerance, $"{name}: input {i}: max abs error {r.MaxAbsError:G3}, scaled {r.ScaledError:G3} (tolerance {tolerance:G3})");
        }

        foreach (Tensor l in leaves)
        {
            Backprop.ClearGrad(l);
            l.Dispose();
        }

        var outcome = new Outcome(name, maxAbs, maxScaled, count);
        Log(outcome);
        return outcome;
    }

    private static readonly object s_lock = new();

    private static void Log(Outcome o)
    {
        string? path = Environment.GetEnvironmentVariable("ONEMODEL_GRADCHECK_LOG");
        if (path is null)
        {
            return;
        }

        lock (s_lock)
        {
            File.AppendAllText(path, string.Create(CultureInfo.InvariantCulture, $"| {o.Name} | {o.Elements} | {o.MaxAbs:E2} | {o.MaxScaled:E2} |\n"));
        }
    }
}
