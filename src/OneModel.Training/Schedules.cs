using System;
using System.Collections.Generic;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Training;

/// <summary>Gradient clipping. Both functions work in place on <see cref="Tensor.Grad"/> and skip parameters without a gradient.</summary>
public static class GradClip
{
    /// <summary>
    /// Scales all gradients by <c>min(1, maxNorm/(norm + 1e-6))</c>, where norm is the global norm over every element of every gradient:
    /// the L2 norm (<c>√Σg²</c>, squares summed in double in parameter and element order) or, for infinity, the largest <c>|g|</c>.
    /// </summary>
    /// <param name="parameters">Parameters whose gradients are clipped together.</param>
    /// <param name="maxNorm">The largest allowed norm, positive.</param>
    /// <param name="infinityNorm">Use the max-abs norm instead of the L2 norm (default false).</param>
    /// <returns>The norm before clipping (float32 of the double result); NaN or infinity if a gradient has them (nothing is scaled then).</returns>
    public static float ClipNorm(IEnumerable<Tensor> parameters, float maxNorm, bool infinityNorm = false)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        if (!(maxNorm > 0f))
        {
            throw new ArgumentOutOfRangeException(nameof(maxNorm), maxNorm, "The maximum norm must be positive.");
        }

        double acc = 0;
        foreach (Tensor p in parameters)
        {
            Tensor g = p.Grad;
            if (g.IsEmpty)
            {
                continue;
            }

            ReadOnlySpan<float> s = g.Data<float>();
            if (infinityNorm)
            {
                foreach (float v in s)
                {
                    double a = Math.Abs((double)v);
                    if (a > acc || double.IsNaN(a))
                    {
                        acc = a;
                    }
                }
            }
            else
            {
                foreach (float v in s)
                {
                    acc += (double)v * v;
                }
            }
        }

        double norm = infinityNorm ? acc : Math.Sqrt(acc);
        float total = (float)norm;
        if (!double.IsFinite(norm))
        {
            return total;
        }

        float coef = (float)((double)maxNorm / (norm + 1e-6));
        if (coef < 1f)
        {
            foreach (Tensor p in parameters)
            {
                Tensor g = p.Grad;
                if (!g.IsEmpty)
                {
                    Span<float> s = g.MutableData<float>();
                    for (int i = 0; i < s.Length; i++)
                    {
                        s[i] *= coef;
                    }
                }
            }
        }

        return total;
    }

    /// <summary>Clamps every gradient element to [−<paramref name="clipValue"/>, +<paramref name="clipValue"/>]; NaN stays NaN.</summary>
    /// <param name="parameters">Parameters whose gradients are clamped.</param>
    /// <param name="clipValue">The bound, not negative.</param>
    public static void ClipValue(IEnumerable<Tensor> parameters, float clipValue)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        if (!(clipValue >= 0f))
        {
            throw new ArgumentOutOfRangeException(nameof(clipValue), clipValue, "The bound must not be negative.");
        }

        foreach (Tensor p in parameters)
        {
            Tensor g = p.Grad;
            if (g.IsEmpty)
            {
                continue;
            }

            Span<float> s = g.MutableData<float>();
            for (int i = 0; i < s.Length; i++)
            {
                float v = s[i];
                s[i] = v > clipValue ? clipValue : v < -clipValue ? -clipValue : v;
            }
        }
    }
}

/// <summary>
/// A learning-rate schedule: a pure function from the number of optimizer steps already taken to a learning rate. It keeps no state,
/// so resuming from a checkpoint needs only the optimizer's <see cref="Optimizer.StepCount"/>. All arithmetic is IEEE double
/// (<c>+ − × ÷</c>, <c>sqrt</c> only; the cosine is a fixed series, <see cref="Cos"/>), so rates are identical on every platform.
/// </summary>
public abstract class LrSchedule
{
    /// <summary>The learning rate for the optimizer step with index <paramref name="stepNumber"/> (0 for the first step).</summary>
    /// <param name="stepNumber">Steps already taken, not negative.</param>
    public abstract float LrAt(long stepNumber);

    /// <summary>Sets <see cref="Optimizer.LearningRate"/> to the rate for the optimizer's current <see cref="Optimizer.StepCount"/>.</summary>
    /// <param name="optimizer">The optimizer to update.</param>
    public void Apply(Optimizer optimizer)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        optimizer.LearningRate = LrAt(optimizer.StepCount);
    }

    /// <summary>
    /// <c>cos(π·t)</c> for t in [0, 1], from IEEE double operations only: by symmetry about t = 1/2 the argument is reduced to
    /// [0, π/2] and the Maclaurin series is summed to the term of degree 26 (truncation error below 1e-21) by Horner's rule. The
    /// result is within 1e-15 of the true cosine and the same on every platform; <c>Math.Cos</c> is not, so it is never used here.
    /// </summary>
    /// <param name="t">A value in [0, 1] (clamped).</param>
    public static double Cos(double t)
    {
        if (!(t > 0.0))
        {
            return 1.0;
        }

        if (t >= 1.0)
        {
            return -1.0;
        }

        double sign = 1.0;
        double y = t;
        if (y > 0.5)
        {
            sign = -1.0;
            y = 1.0 - y;
        }

        double a = 3.14159265358979323846 * y;
        double a2 = a * a;
        double r = 1.0;
        for (int k = 13; k >= 1; k--)
        {
            // r = 1 - a2/((2k-1)(2k)) * r
            r = 1.0 - a2 / ((2.0 * k - 1.0) * (2.0 * k)) * r;
        }

        return sign * r;
    }

    /// <summary>The input value, with a negative step treated as 0.</summary>
    protected static long Clamp(long stepNumber) => stepNumber < 0 ? 0 : stepNumber;
}

/// <summary>A constant learning rate.</summary>
public sealed class ConstantLr : LrSchedule
{
    private readonly float _lr;

    /// <summary>Creates the schedule.</summary>
    /// <param name="learningRate">The rate.</param>
    public ConstantLr(float learningRate) => _lr = learningRate;

    /// <inheritdoc/>
    public override float LrAt(long stepNumber) => _lr;
}

/// <summary>Multiplies the rate by <c>gamma</c> every <c>stepSize</c> steps: <c>base·gamma^⌊step/stepSize⌋</c>.</summary>
public sealed class StepLr : LrSchedule
{
    private readonly double _base;
    private readonly long _stepSize;
    private readonly double _gamma;

    /// <summary>Creates the schedule.</summary>
    /// <param name="baseLr">The initial rate.</param>
    /// <param name="stepSize">Steps between decays, at least 1.</param>
    /// <param name="gamma">The decay factor (default 0.1).</param>
    public StepLr(float baseLr, long stepSize, float gamma = 0.1f)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(stepSize, 1);
        _base = baseLr;
        _stepSize = stepSize;
        _gamma = gamma;
    }

    /// <inheritdoc/>
    public override float LrAt(long stepNumber)
    {
        long n = Clamp(stepNumber) / _stepSize;
        double r = _base;
        double b = _gamma;
        while (n > 0)
        {
            if ((n & 1) != 0)
            {
                r *= b;
            }

            b *= b;
            n >>= 1;
        }

        return (float)r;
    }
}

/// <summary>
/// Cosine annealing (Loshchilov and Hutter 2017, one cycle, no restarts): <c>min + (base − min)·(1 + cos(π·step/total))/2</c>,
/// held at <c>min</c> after <c>total</c> steps.
/// </summary>
public sealed class CosineLr : LrSchedule
{
    private readonly double _base;
    private readonly double _min;
    private readonly long _total;

    /// <summary>Creates the schedule.</summary>
    /// <param name="baseLr">The initial rate.</param>
    /// <param name="totalSteps">Steps over which the rate falls from <paramref name="baseLr"/> to <paramref name="minLr"/>, at least 1.</param>
    /// <param name="minLr">The final rate (default 0).</param>
    public CosineLr(float baseLr, long totalSteps, float minLr = 0f)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(totalSteps, 1);
        _base = baseLr;
        _min = minLr;
        _total = totalSteps;
    }

    /// <inheritdoc/>
    public override float LrAt(long stepNumber)
    {
        long s = Clamp(stepNumber);
        double t = s >= _total ? 1.0 : (double)s / _total;
        return (float)(_min + (_base - _min) * (1.0 + Cos(t)) / 2.0);
    }
}

/// <summary>
/// Linear warm-up into another schedule: for the first <c>warmupSteps</c> steps the rate rises linearly from
/// <c>startFactor·r₀</c> to <c>r₀</c>, where r₀ is the inner schedule's rate at its step 0; afterwards the inner schedule runs with
/// its own step counted from the end of the warm-up.
/// </summary>
public sealed class WarmupLr : LrSchedule
{
    private readonly LrSchedule _after;
    private readonly long _warmup;
    private readonly double _start;

    /// <summary>Creates the schedule.</summary>
    /// <param name="after">The schedule that takes over.</param>
    /// <param name="warmupSteps">Length of the warm-up, at least 1.</param>
    /// <param name="startFactor">The warm-up's first rate as a fraction of the inner schedule's first rate, in [0, 1] (default 0).</param>
    public WarmupLr(LrSchedule after, long warmupSteps, float startFactor = 0f)
    {
        ArgumentNullException.ThrowIfNull(after);
        ArgumentOutOfRangeException.ThrowIfLessThan(warmupSteps, 1);
        _after = after;
        _warmup = warmupSteps;
        _start = startFactor;
    }

    /// <inheritdoc/>
    public override float LrAt(long stepNumber)
    {
        long s = Clamp(stepNumber);
        if (s >= _warmup)
        {
            return _after.LrAt(s - _warmup);
        }

        double r0 = _after.LrAt(0);
        return (float)(r0 * (_start + (1.0 - _start) * ((double)s / _warmup)));
    }
}

/// <summary>
/// Linear warm-up then a floored half cosine, as a pure function of the step: for step s &lt; <c>warmupSteps</c> the rate is
/// <c>base·(s+1)/warmupSteps</c>; afterwards, with <c>p = (s − warmupSteps)/max(1, totalSteps − warmupSteps)</c>, it is
/// <c>base·(floor + (1 − floor)·(1 + cos(π·p))/2)</c>. Unlike <see cref="WarmupLr"/> the first step already has a nonzero rate and the
/// decay is measured over the steps after the warm-up. Steps past <c>totalSteps</c> hold the floor.
/// </summary>
public sealed class WarmupCosineLr : LrSchedule
{
    private readonly double _base;
    private readonly long _warmup;
    private readonly long _total;
    private readonly double _floor;

    /// <summary>Creates the schedule.</summary>
    /// <param name="baseLr">The peak rate.</param>
    /// <param name="warmupSteps">Warm-up length, at least 1.</param>
    /// <param name="totalSteps">Step at which the decay ends, greater than <paramref name="warmupSteps"/>.</param>
    /// <param name="floorFraction">Final rate as a fraction of <paramref name="baseLr"/>, in [0, 1] (default 0).</param>
    public WarmupCosineLr(float baseLr, long warmupSteps, long totalSteps, float floorFraction = 0f)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(warmupSteps, 1);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(totalSteps, warmupSteps);
        ArgumentOutOfRangeException.ThrowIfNegative(floorFraction);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(floorFraction, 1f);
        _base = baseLr;
        _warmup = warmupSteps;
        _total = totalSteps;
        _floor = floorFraction;
    }

    /// <inheritdoc/>
    public override float LrAt(long stepNumber)
    {
        long s = Clamp(stepNumber);
        if (s < _warmup)
        {
            return (float)(_base * (s + 1) / _warmup);
        }

        double p = Math.Min(1.0, (double)(s - _warmup) / Math.Max(1, _total - _warmup));
        return (float)(_base * (_floor + ((1.0 - _floor) * 0.5 * (1.0 + Cos(p)))));
    }
}

/// <summary>
/// The one-cycle policy (Smith and Topin 2018): the rate rises from <c>max/divFactor</c> to <c>max</c> over the first
/// <c>pctStart</c> of the steps, then falls to <c>max/divFactor/finalDivFactor</c>, both phases along a half cosine. Momentum is not cycled.
/// </summary>
public sealed class OneCycleLr : LrSchedule
{
    private readonly double _max;
    private readonly double _initial;
    private readonly double _min;
    private readonly double _upEnd;
    private readonly double _downEnd;

    /// <summary>Creates the schedule.</summary>
    /// <param name="maxLr">The peak rate.</param>
    /// <param name="totalSteps">Number of optimizer steps in the cycle, at least 2.</param>
    /// <param name="pctStart">Fraction of the cycle spent rising, in (0, 1) (default 0.3).</param>
    /// <param name="divFactor">The first rate is <c>maxLr/divFactor</c> (default 25).</param>
    /// <param name="finalDivFactor">The last rate is the first divided by this (default 1e4).</param>
    public OneCycleLr(float maxLr, long totalSteps, float pctStart = 0.3f, float divFactor = 25f, float finalDivFactor = 1e4f)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(totalSteps, 2);
        if (!(pctStart > 0f && pctStart < 1f))
        {
            throw new ArgumentOutOfRangeException(nameof(pctStart), pctStart, "pctStart must be in (0, 1).");
        }

        _max = maxLr;
        _initial = (double)maxLr / divFactor;
        _min = _initial / finalDivFactor;
        _upEnd = (double)pctStart * totalSteps - 1.0;
        _downEnd = totalSteps - 1.0;
    }

    private static double Anneal(double start, double end, double pct) => end + (start - end) / 2.0 * (Cos(pct) + 1.0);

    /// <inheritdoc/>
    public override float LrAt(long stepNumber)
    {
        double s = Clamp(stepNumber);
        if (s > _downEnd)
        {
            s = _downEnd;
        }

        // Cosine annealing from a to b over a phase is Anneal(a, b, pct); Cos(pct) here is cos(pi*pct).
        return s <= _upEnd
            ? (float)Anneal(_initial, _max, s / _upEnd)
            : (float)Anneal(_max, _min, (s - _upEnd) / (_downEnd - _upEnd));
    }
}
