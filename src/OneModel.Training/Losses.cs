using System;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Training;

/// <summary>
/// Loss functions as compositions of differentiable tensor ops (so their gradients are the ops' gradients). Every loss returns a
/// rank-0 float32 tensor. Constants a loss creates are handed to the open <see cref="ActivationScope"/>; open one per step.
/// </summary>
public static class Losses
{
    /// <summary>
    /// Cross-entropy of class logits: <c>−log softmax(logits)[n, target[n]]</c>, with optional label smoothing
    /// (<c>(1−ε)·NLL + ε·mean over classes of −log p</c>, the smoothing term taken over the same rows) and an ignored target value.
    /// </summary>
    /// <param name="logits">Float32 [N, C].</param>
    /// <param name="targets">Class indices [N], Int32 or Int64.</param>
    /// <param name="labelSmoothing">ε in [0, 1], default 0.</param>
    /// <param name="ignoreIndex">Target value whose rows are skipped (default -100).</param>
    /// <param name="mean">Average over the counted rows (default) or sum. With no counted row the mean is NaN.</param>
    public static Tensor CrossEntropy(in Tensor logits, in Tensor targets, float labelSmoothing = 0f, int ignoreIndex = -100, bool mean = true)
    {
        if (!(labelSmoothing >= 0f && labelSmoothing <= 1f))
        {
            throw new ArgumentOutOfRangeException(nameof(labelSmoothing), labelSmoothing, "Label smoothing must be in [0, 1].");
        }

        Tensor logp = Ops.LogSoftmax(logits);
        Tensor nll = Ops.NllLoss(logp, targets, mean, ignoreIndex);
        if (labelSmoothing == 0f)
        {
            return nll;
        }

        // Smoothing term: per row -mean_c(logp), masked to the counted rows.
        int rows = logits.Shape[0];
        Tensor mask = Tensor.Create(DType.F32, Shape.Create(rows));
        Span<float> m = mask.MutableData<float>();
        int counted = 0;
        for (int i = 0; i < rows; i++)
        {
            long t = targets.DType == DType.I64 ? targets.Get<long>(i) : targets.Get<int>(i);
            bool keep = t != ignoreIndex;
            m[i] = keep ? 1f : 0f;
            counted += keep ? 1 : 0;
        }

        ActivationScope.Track(mask);
        Tensor perRow = Ops.Neg(Ops.Mean(logp, 1));
        Tensor smooth = Ops.Sum(Ops.Mul(perRow, mask));
        if (mean)
        {
            smooth = Ops.Mul(smooth, Init.Scalar(counted == 0 ? float.NaN : 1f / counted));
        }

        return Ops.Add(Ops.Mul(nll, Init.Scalar(1f - labelSmoothing)), Ops.Mul(smooth, Init.Scalar(labelSmoothing)));
    }

    /// <summary>Negative log-likelihood of class indices under log-probabilities (see <see cref="Ops.NllLoss"/>).</summary>
    /// <param name="logProbs">Float32 [N, C].</param>
    /// <param name="targets">Class indices [N].</param>
    /// <param name="ignoreIndex">Target value whose rows are skipped (default -100).</param>
    /// <param name="mean">Average (default) or sum.</param>
    public static Tensor NllLoss(in Tensor logProbs, in Tensor targets, int ignoreIndex = -100, bool mean = true) =>
        Ops.NllLoss(logProbs, targets, mean, ignoreIndex);

    /// <summary>
    /// Connectionist temporal classification loss (Graves et al. 2006) over log-probabilities [T, B, C]; see <see cref="Ops.CtcLoss"/> for
    /// the reductions and for how the gradient relates to PyTorch's (the two agree after a log-softmax).
    /// </summary>
    /// <param name="logProbs">Float32 [T, B, C], normally a log-softmax output.</param>
    /// <param name="targets">All target labels concatenated, none equal to the blank.</param>
    /// <param name="inputLengths">Valid steps per sequence.</param>
    /// <param name="targetLengths">Target length per sequence.</param>
    /// <param name="blank">The blank class (default 0).</param>
    /// <param name="reduction">Default: each loss divided by its target length, then averaged over the batch.</param>
    /// <param name="zeroInfinity">Replace infinite losses and their gradients by zero (default false).</param>
    public static Tensor Ctc(in Tensor logProbs, ReadOnlySpan<int> targets, ReadOnlySpan<int> inputLengths, ReadOnlySpan<int> targetLengths,
        int blank = 0, CtcReduction reduction = CtcReduction.Mean, bool zeroInfinity = false) =>
        Ops.CtcLoss(logProbs, targets, inputLengths, targetLengths, blank, reduction, zeroInfinity);

    /// <summary>Mean (or sum) of squared differences <c>(prediction − target)²</c>.</summary>
    /// <param name="prediction">Float32 tensor.</param>
    /// <param name="target">Float32 tensor, broadcastable with <paramref name="prediction"/>. It may require gradients.</param>
    /// <param name="mean">Average over all elements (default) or sum.</param>
    public static Tensor Mse(in Tensor prediction, in Tensor target, bool mean = true)
    {
        Tensor d = Ops.Sub(prediction, target);
        Tensor sq = Ops.Mul(d, d);
        return mean ? Ops.Mean(sq) : Ops.Sum(sq);
    }

    /// <summary>
    /// Mean (or sum) of absolute differences. <c>|d|</c> is built as <c>relu(d) + relu(−d)</c>, whose gradient at <c>d = 0</c> is 0.
    /// </summary>
    /// <param name="prediction">Float32 tensor.</param>
    /// <param name="target">Float32 tensor, broadcastable with <paramref name="prediction"/>.</param>
    /// <param name="mean">Average over all elements (default) or sum.</param>
    public static Tensor L1(in Tensor prediction, in Tensor target, bool mean = true)
    {
        Tensor d = Ops.Sub(prediction, target);
        Tensor abs = Ops.Add(Ops.Relu(d), Ops.Relu(Ops.Neg(d)));
        return mean ? Ops.Mean(abs) : Ops.Sum(abs);
    }

    /// <summary>
    /// Binary cross-entropy on logits: <c>softplus(x) − x·t</c> per element, with softplus computed stably as the negated
    /// log-softmax of <c>[x, 0]</c> at the zero entry (so large |x| neither overflows nor loses the small term).
    /// </summary>
    /// <param name="logits">Float32 tensor of any shape.</param>
    /// <param name="targets">Float32 tensor of the same shape with values in [0, 1].</param>
    /// <param name="mean">Average over all elements (default) or sum.</param>
    public static Tensor BceWithLogits(in Tensor logits, in Tensor targets, bool mean = true)
    {
        if (!logits.Shape.Equals(targets.Shape))
        {
            throw new TensorShapeException($"BceWithLogits needs equal shapes, not {logits.Shape} and {targets.Shape}.");
        }

        int count = logits.Length;
        Tensor x = logits.Reshape(count, 1);
        Tensor zeros = Tensor.Zeros(DType.F32, Shape.Create(count, 1));
        ActivationScope.Track(zeros);
        Tensor logSoftmax = Ops.LogSoftmax(Ops.Concat([x, zeros], 1));
        Tensor softplus = Ops.Neg(logSoftmax.Select(1, 1));
        Tensor xt = Ops.Mul(logits.Reshape(count), targets.Reshape(count));
        Tensor perElement = Ops.Sub(softplus, xt);
        return mean ? Ops.Mean(perElement) : Ops.Sum(perElement);
    }
}
