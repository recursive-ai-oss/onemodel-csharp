using System;
using System.Collections.Generic;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Training;

/// <summary>The agreement between analytic and finite-difference gradients for one input of a checked function.</summary>
/// <param name="Elements">Number of gradient elements compared.</param>
/// <param name="MaxAbsError">Largest <c>|analytic − numeric|</c>.</param>
/// <param name="ScaledError"><see cref="MaxAbsError"/> divided by the largest numeric gradient magnitude (at least 1e-3).</param>
public readonly record struct GradientCheckResult(int Elements, double MaxAbsError, double ScaledError);

/// <summary>
/// Checks the backward pass of a function against central finite differences of its forward pass. The objective is
/// <c>L = Σ W·f(x)</c> with a fixed pseudo-random weight tensor W, summed in double; the numeric derivative is
/// <c>(L(x+h) − L(x−h))/(2h)</c> in float32 arithmetic, so the achievable accuracy is about <c>ε₃₂·|L|/h</c> plus the curvature term
/// <c>h²·f‴</c>: pick <c>h</c> near 1e-2 for smooth functions of order-one inputs and keep inputs away from kinks (ReLU at 0, ties in
/// max pooling).
/// </summary>
public static class GradientCheck
{
    /// <summary>Compares the gradient of every input in <paramref name="inputs"/> that requires one.</summary>
    /// <param name="function">The forward function over the inputs; it must be deterministic and must not modify the inputs. Every call may allocate freely: it runs inside an <see cref="ActivationScope"/>.</param>
    /// <param name="inputs">Contiguous float32 leaf tensors (made with <see cref="Tensor.WithRequiresGrad"/>). Their data is perturbed during the check and restored; their <see cref="Tensor.Grad"/> holds the analytic gradient afterwards.</param>
    /// <param name="step">The finite-difference step <c>h</c>, default 1e-2.</param>
    /// <param name="seed">Seed of the weight tensor W, default 1.</param>
    /// <returns>One result per input; inputs that do not require gradients get an all-zero result.</returns>
    /// <exception cref="InvalidOperationException">The function's output does not depend on a required input differentiably (no gradient arrives).</exception>
    public static GradientCheckResult[] Run(Func<Tensor[], Tensor> function, Tensor[] inputs, double step = 1e-2, int seed = 1)
    {
        ArgumentNullException.ThrowIfNull(function);
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(step, 0.0);
        float[] weights;
        Shape outShape;
        using (new ActivationScope())
        using (GradMode.NoGrad())
        {
            Tensor o = function(inputs);
            outShape = o.Shape;
            weights = new float[outShape.Length];
            // A fixed xorshift stream, so the weights depend only on the seed.
            ulong state = ((ulong)(uint)seed * 0x9E3779B97F4A7C15UL) | 1UL;
            for (int i = 0; i < weights.Length; i++)
            {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                weights[i] = (float)(((state >> 40) / (double)(1 << 24) * 2) - 1);
            }
        }

        using (new ActivationScope())
        {
            Tensor y = function(inputs);
            using Tensor w = Tensor.FromSpan<float>(weights, outShape);
            y.Backward(w);
        }

        var results = new GradientCheckResult[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            if (!inputs[i].RequiresGrad)
            {
                continue;
            }

            Tensor grad = inputs[i].Grad;
            if (grad.IsEmpty)
            {
                throw new InvalidOperationException($"Input {i} requires a gradient but none arrived.");
            }

            float[] analytic = grad.ToArray<float>();
            Span<float> data = inputs[i].MutableData<float>();
            double maxAbs = 0, maxNumeric = 0;
            var numeric = new double[analytic.Length];
            for (int e = 0; e < data.Length; e++)
            {
                float orig = data[e];
                float up = orig + (float)step, down = orig - (float)step;
                data[e] = up;
                double lp = Objective(function, inputs, weights);
                data[e] = down;
                double lm = Objective(function, inputs, weights);
                data[e] = orig;
                numeric[e] = (lp - lm) / ((double)up - down);
                maxNumeric = Math.Max(maxNumeric, Math.Abs(numeric[e]));
            }

            for (int e = 0; e < analytic.Length; e++)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs(analytic[e] - numeric[e]));
            }

            results[i] = new GradientCheckResult(analytic.Length, maxAbs, maxAbs / Math.Max(maxNumeric, 1e-3));
        }

        return results;
    }

    private static double Objective(Func<Tensor[], Tensor> function, Tensor[] inputs, float[] weights)
    {
        using var scope = new ActivationScope();
        using var noGrad = GradMode.NoGrad();
        float[] v = function(inputs).Clone().ToArray<float>();
        double sum = 0;
        for (int i = 0; i < v.Length; i++)
        {
            sum += (double)v[i] * weights[i];
        }

        return sum;
    }
}
