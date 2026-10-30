using System;
namespace RecursiveAI.OneModel.Tensors;

/// <summary>How <see cref="Ops.CtcLoss"/> combines the per-sequence losses.</summary>
public enum CtcReduction
{
    /// <summary>One loss per sequence, shape <c>[B]</c>.</summary>
    None,
    /// <summary>Each loss divided by its target length (at least 1), then the mean over the batch (PyTorch's default).</summary>
    Mean,
    /// <summary>The sum of the losses.</summary>
    Sum,
}

public static partial class Ops
{
    /// <summary>
    /// Connectionist temporal classification loss (Graves, Fernández, Gomez and Schmidhuber 2006): the negative log of the total
    /// probability of all alignments of the target label sequence (with blanks inserted and repeats collapsed) under per-step class
    /// log-probabilities. Computed in log space with the forward–backward recursions in double precision, using our own fixed
    /// <c>exp</c> and <c>log</c> (docs/ctc.md), so the result is the same on every machine and in both compute modes.
    /// Record: slot 0 = the log-probabilities; Ints = [blank, zeroInfinity, reduction, B, inputLengths…, targetLengths…, targets…]. Backward
    /// recomputes the recursions and returns the true partial derivative with respect to each log-probability,
    /// <c>−exp(log Σ α·β − lp + loss)</c> on the frames used, 0 elsewhere. (PyTorch's own gradient additionally contains an
    /// <c>exp(lp)</c> term that vanishes after a log-softmax; the gradients with respect to the logits are equal.)
    /// A sequence with no valid alignment (input too short for the target) has loss +∞ — or 0 with
    /// <paramref name="zeroInfinity"/> — and gets a zero gradient either way.
    /// </summary>
    /// <param name="logProbs">Float32 log-probabilities <c>[T, B, C]</c>, usually the output of <see cref="LogSoftmax"/>.</param>
    /// <param name="targets">All target labels concatenated, sequence after sequence. Each in <c>[0, C)</c> and not the blank.</param>
    /// <param name="inputLengths">Valid steps per sequence, each in <c>[0, T]</c>; length B.</param>
    /// <param name="targetLengths">Target length per sequence, at least 0; length B; the sum must equal <c>targets.Length</c>.</param>
    /// <param name="blank">The blank class index, default 0.</param>
    /// <param name="reduction">How to combine the per-sequence losses; default <see cref="CtcReduction.Mean"/>.</param>
    /// <param name="zeroInfinity">Replace infinite losses (and their gradients) by zero. Default false.</param>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    /// <returns>A scalar for <c>Mean</c> and <c>Sum</c>, or <c>[B]</c> for <c>None</c>.</returns>
    /// <exception cref="ArgumentException">A length, label or the blank index is out of range, or the lengths do not match the targets.</exception>
    /// <exception cref="TensorShapeException"><paramref name="logProbs"/> is not rank 3.</exception>
    public static Tensor CtcLoss(in Tensor logProbs, ReadOnlySpan<int> targets, ReadOnlySpan<int> inputLengths, ReadOnlySpan<int> targetLengths,
        int blank = 0, CtcReduction reduction = CtcReduction.Mean, bool zeroInfinity = false, BufferPool? pool = null)
    {
        RequireFloat32(logProbs);
        if (logProbs.Rank != 3)
        {
            throw new TensorShapeException($"CtcLoss needs log-probabilities [T,B,C], not {logProbs.Shape}.");
        }

        int steps = logProbs.Shape[0], batch = logProbs.Shape[1], classes = logProbs.Shape[2];
        if (inputLengths.Length != batch || targetLengths.Length != batch)
        {
            throw new ArgumentException($"Expected {batch} input and target lengths, got {inputLengths.Length} and {targetLengths.Length}.");
        }

        if ((uint)blank >= (uint)classes)
        {
            throw new ArgumentException($"The blank index {blank} is outside [0, {classes}).", nameof(blank));
        }

        long total = 0;
        for (int b = 0; b < batch; b++)
        {
            if ((uint)inputLengths[b] > (uint)steps)
            {
                throw new ArgumentException($"Input length {inputLengths[b]} of sequence {b} is outside [0, {steps}].", nameof(inputLengths));
            }

            if (targetLengths[b] < 0)
            {
                throw new ArgumentException($"Target length {targetLengths[b]} of sequence {b} is negative.", nameof(targetLengths));
            }

            total += targetLengths[b];
        }

        if (total != targets.Length)
        {
            throw new ArgumentException($"The target lengths sum to {total} but {targets.Length} labels were given.", nameof(targets));
        }

        foreach (int label in targets)
        {
            if ((uint)label >= (uint)classes || label == blank)
            {
                throw new ArgumentException($"Target label {label} must be in [0, {classes}) and differ from the blank {blank}.", nameof(targets));
            }
        }

        Tensor lpc = Contig(logProbs, out bool copied);
        int maxS = 1;
        for (int b = 0; b < batch; b++)
        {
            maxS = Math.Max(maxS, (2 * targetLengths[b]) + 1);
        }

        double[] alpha = new double[Math.Max(1, steps) * maxS];
        var per = new double[batch];
        int offset = 0;
        for (int b = 0; b < batch; b++)
        {
            ReadOnlySpan<int> tgt = targets.Slice(offset, targetLengths[b]);
            offset += targetLengths[b];
            double nll = CtcLossMath.Alpha(lpc.Data<float>(), steps, batch, classes, b, inputLengths[b], tgt, blank, alpha);
            per[b] = double.IsPositiveInfinity(nll) && zeroInfinity ? 0.0 : nll;
        }

        Tensor result = NewOut(reduction == CtcReduction.None ? Shape.Create(batch) : Shape.Create(), pool);
        Span<float> rv = result.MutableData<float>();
        if (reduction == CtcReduction.None)
        {
            for (int b = 0; b < batch; b++)
            {
                rv[b] = (float)per[b];
            }
        }
        else
        {
            double sum = 0;
            for (int b = 0; b < batch; b++)
            {
                sum += reduction == CtcReduction.Mean ? per[b] / Math.Max(1, targetLengths[b]) : per[b];
            }

            rv[0] = (float)(reduction == CtcReduction.Mean ? sum / batch : sum);
        }

        if (!Recording(logProbs))
        {
            Free(lpc, copied);
            return result;
        }

        BackwardRecord br = BackwardRecord.Create(OpKind.CtcLoss, 1, logProbs);
        SaveContig(br, 0, lpc, copied);
        long[] ints = new long[4 + (2 * batch) + targets.Length];
        ints[0] = blank;
        ints[1] = zeroInfinity ? 1 : 0;
        ints[2] = (int)reduction;
        ints[3] = batch;
        for (int b = 0; b < batch; b++)
        {
            ints[4 + b] = inputLengths[b];
            ints[4 + batch + b] = targetLengths[b];
        }

        for (int i = 0; i < targets.Length; i++)
        {
            ints[4 + (2 * batch) + i] = targets[i];
        }

        br.SetInts(ints);
        return br.Attach(result);
    }
}

/// <summary>The forward–backward recursions of CTC in log space, double precision, written as in docs/ctc.md.</summary>
internal static class CtcLossMath
{
    private const double NegInf = double.NegativeInfinity;

    // The label at extended position s: blank at even s, target[(s - 1) / 2] at odd s.
    private static int Label(ReadOnlySpan<int> tgt, int blank, int s) => (s & 1) == 0 ? blank : tgt[s >> 1];

    // Whether the path may skip from s - 2 to s: only onto a non-blank that differs from the label two back.
    private static bool CanSkip(ReadOnlySpan<int> tgt, int blank, int s) => s >= 2 && (s & 1) == 1 && tgt[s >> 1] != tgt[(s >> 1) - 1];

    private static double Emit(ReadOnlySpan<float> lp, int batch, int classes, int t, int b, int cls) => lp[(((t * batch) + b) * classes) + cls];

    /// <summary>Fills alpha[t * (2L+1) + s] for t &lt; len and returns the loss −log p (+∞ when no alignment exists).</summary>
    public static double Alpha(ReadOnlySpan<float> lp, int steps, int batch, int classes, int b, int len, ReadOnlySpan<int> tgt, int blank, double[] alpha)
    {
        int ext = (2 * tgt.Length) + 1;
        if (len == 0)
        {
            // Only the empty target aligns with no frames.
            return tgt.Length == 0 ? 0.0 : double.PositiveInfinity;
        }

        for (int t = 0; t < len; t++)
        {
            for (int s = 0; s < ext; s++)
            {
                double a;
                if (t == 0)
                {
                    a = s <= 1 ? 0.0 : NegInf;
                }
                else
                {
                    a = alpha[((t - 1) * ext) + s];
                    if (s >= 1)
                    {
                        a = CtcMath.LogAdd(a, alpha[((t - 1) * ext) + s - 1]);
                    }

                    if (CanSkip(tgt, blank, s))
                    {
                        a = CtcMath.LogAdd(a, alpha[((t - 1) * ext) + s - 2]);
                    }
                }

                alpha[(t * ext) + s] = double.IsNegativeInfinity(a) ? NegInf : a + Emit(lp, batch, classes, t, b, Label(tgt, blank, s));
            }
        }

        double p = alpha[((len - 1) * ext) + ext - 1];
        if (ext >= 2)
        {
            p = CtcMath.LogAdd(p, alpha[((len - 1) * ext) + ext - 2]);
        }

        return double.IsNegativeInfinity(p) ? double.PositiveInfinity : -p;
    }

    /// <summary>
    /// Adds <paramref name="scale"/> times the partial derivative of the loss with respect to every log-probability of sequence
    /// <paramref name="b"/> into <paramref name="grad"/> ([T,B,C]); scratch holds at least 2 * len * (2L+1) doubles.
    /// </summary>
    public static void Gradient(ReadOnlySpan<float> lp, Span<float> grad, int steps, int batch, int classes, int b, int len, ReadOnlySpan<int> tgt,
        int blank, double scale, double[] alpha, double[] beta, double[] lse)
    {
        int ext = (2 * tgt.Length) + 1;
        double nll = Alpha(lp, steps, batch, classes, b, len, tgt, blank, alpha);
        if (len == 0 || double.IsPositiveInfinity(nll))
        {
            return;
        }

        for (int t = len - 1; t >= 0; t--)
        {
            for (int s = ext - 1; s >= 0; s--)
            {
                double v;
                if (t == len - 1)
                {
                    v = s >= ext - 2 ? 0.0 : NegInf;
                }
                else
                {
                    v = beta[((t + 1) * ext) + s];
                    if (s + 1 < ext)
                    {
                        v = CtcMath.LogAdd(v, beta[((t + 1) * ext) + s + 1]);
                    }

                    if (s + 2 < ext && CanSkip(tgt, blank, s + 2))
                    {
                        v = CtcMath.LogAdd(v, beta[((t + 1) * ext) + s + 2]);
                    }
                }

                // beta[t, s] includes the emission at t, as alpha does, so alpha * beta double-counts it; it is divided out below.
                beta[(t * ext) + s] = double.IsNegativeInfinity(v) ? NegInf : v + Emit(lp, batch, classes, t, b, Label(tgt, blank, s));
            }

            Span<double> acc = lse.AsSpan(0, classes);
            acc.Fill(NegInf);
            for (int s = 0; s < ext; s++)
            {
                double ab = alpha[(t * ext) + s] + beta[(t * ext) + s];
                if (!double.IsNegativeInfinity(ab) && !double.IsNaN(ab))
                {
                    int cls = Label(tgt, blank, s);
                    acc[cls] = CtcMath.LogAdd(acc[cls], ab);
                }
            }

            for (int c = 0; c < classes; c++)
            {
                if (double.IsNegativeInfinity(acc[c]))
                {
                    continue;
                }

                // d(-log p)/d lp[t,c] = -(1/p) * (1/y) * sum_{s: l'(s)=c} alpha*beta, y = exp(lp[t,c]): alpha and beta both contain the
                // emission, which the 1/y per factor accounts for once more than the product formula needs, leaving one division.
                double l = Emit(lp, batch, classes, t, b, c);
                grad[(((t * batch) + b) * classes) + c] += (float)(-scale * CtcMath.Exp(acc[c] - l + nll));
            }
        }
    }
}
