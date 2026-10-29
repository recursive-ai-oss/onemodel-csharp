using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using RecursiveAI.OneModel.Kernels;

namespace RecursiveAI.OneModel.Tensors;

/// <summary>How the scores handed to <see cref="Ctc"/> are to be read.</summary>
public enum CtcScores
{
    /// <summary>Unnormalised class scores (network logits). A softmax over the classes of each step is taken where probabilities are needed.</summary>
    Logits = 0,

    /// <summary>Natural-log probabilities that already sum (in probability) to one over the classes of each step, such as a log-softmax output.</summary>
    LogProbabilities = 1,
}

/// <summary>One emitted label of a greedy decoding: the class, the steps its run covered, and the run's probability.</summary>
/// <param name="Label">The class index (never the blank).</param>
/// <param name="FirstStep">The first time step of the run (0-based).</param>
/// <param name="LastStep">The last time step of the run, at least <paramref name="FirstStep"/>.</param>
/// <param name="Confidence">The largest probability the label's class had at any step of the run, in [0, 1].</param>
public readonly record struct CtcRun(int Label, int FirstStep, int LastStep, float Confidence);

/// <summary>A greedy (best-path) decoding of one sequence.</summary>
public sealed class CtcGreedyResult
{
    internal CtcGreedyResult(CtcRun[] runs, float confidence)
    {
        Runs = runs;
        Confidence = confidence;
    }

    /// <summary>The emitted labels in order, each with its steps and confidence. Empty when the best path is all blanks.</summary>
    public IReadOnlyList<CtcRun> Runs { get; }

    /// <summary>The mean of the runs' confidences, in [0, 1]; 0 when there are no runs.</summary>
    public float Confidence { get; }

    /// <summary>The label sequence, a copy.</summary>
    public int[] Labels()
    {
        var labels = new int[Runs.Count];
        for (int i = 0; i < labels.Length; i++)
        {
            labels[i] = Runs[i].Label;
        }

        return labels;
    }
}

/// <summary>One hypothesis of a prefix beam search.</summary>
/// <param name="Labels">The label sequence (no blanks, no repeats from merged runs).</param>
/// <param name="LogProbability">Natural log of the probability of the sequence summed over all alignments the search saw; at most 0.</param>
/// <param name="Confidence">exp(LogProbability / Labels.Length), the per-label geometric mean, in [0, 1]; 0 for the empty sequence.</param>
public readonly record struct CtcHypothesis(int[] Labels, double LogProbability, float Confidence);

/// <summary>
/// Connectionist temporal classification decoding of per-step class scores: greedy best-path and prefix beam search (Graves et
/// al. 2006; Hannun et al. 2014). Scores are row-major <c>[steps, classes]</c>, a batch is <c>[batch, steps, classes]</c>.
/// The functions are pure, thread-safe and give the same result on every machine: probabilities come from the library's exact
/// softmax and exponential, and the beam search uses its own double-precision exp and log built from + − × ÷ only.
/// The definitions are in <c>docs/ctc.md</c>.
/// </summary>
public static class Ctc
{
    /// <summary>The most prefixes a beam search may create, which bounds its memory: steps × beam width × (classes − 1) + 1.</summary>
    public const long MaxBeamNodes = 1L << 24;

    /// <summary>The widest beam accepted.</summary>
    public const int MaxBeamWidth = 4096;

    /// <summary>
    /// Greedy decoding without allocating: the best class at every step (the lowest index on ties), repeats merged, blanks dropped.
    /// </summary>
    /// <param name="scores"><c>steps × classes</c> scores, row-major. A NaN score is never chosen over a number; a step of only NaNs picks class 0.</param>
    /// <param name="steps">Number of time steps, 0 or more.</param>
    /// <param name="classes">Number of classes including the blank, at least 1.</param>
    /// <param name="runs">Receives the emitted runs in order; needs room for <paramref name="steps"/> runs (the most there can be).</param>
    /// <param name="confidence">The mean of the runs' confidences; 0 when nothing is emitted.</param>
    /// <param name="blank">The blank class, in [0, classes). Default 0.</param>
    /// <param name="kind">Whether <paramref name="scores"/> are logits or log-probabilities.</param>
    /// <param name="stepProbabilities">Optional: receives, for every step, the probability of the step's best class (needs <paramref name="steps"/> entries when given).</param>
    /// <returns>The number of runs written.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="steps"/>, <paramref name="classes"/> or <paramref name="blank"/> out of range.</exception>
    /// <exception cref="ArgumentException"><paramref name="scores"/>, <paramref name="runs"/> or <paramref name="stepProbabilities"/> is too short.</exception>
    public static int Greedy(ReadOnlySpan<float> scores, int steps, int classes, Span<CtcRun> runs, out float confidence,
        int blank = 0, CtcScores kind = CtcScores.Logits, Span<float> stepProbabilities = default)
    {
        Check(scores.Length, steps, classes, blank);
        if (runs.Length < steps)
        {
            throw new ArgumentException("runs needs room for one run per step.", nameof(runs));
        }

        if (!stepProbabilities.IsEmpty && stepProbabilities.Length < steps)
        {
            throw new ArgumentException("stepProbabilities needs one entry per step.", nameof(stepProbabilities));
        }

        confidence = 0f;
        if (steps == 0)
        {
            return 0;
        }

        float[] rented = ArrayPool<float>.Shared.Rent(classes);
        try
        {
            Span<float> tmp = rented.AsSpan(0, classes);
            int count = 0;
            int last = blank;
            int first = 0;
            float runMax = 0f;
            bool wantAll = !stepProbabilities.IsEmpty;
            for (int t = 0; t < steps; t++)
            {
                ReadOnlySpan<float> frame = scores.Slice(t * classes, classes);
                int best = 0;
                float bestValue = frame[0];
                for (int j = 1; j < frame.Length; j++)
                {
                    // `>` makes ties go to the lower index and keeps NaN from winning over a number.
                    if (frame[j] > bestValue || float.IsNaN(bestValue))
                    {
                        bestValue = frame[j];
                        best = j;
                    }
                }

                float p = 0f;
                if (wantAll || best != blank)
                {
                    p = BestProbability(frame, best, kind, tmp);
                    if (wantAll)
                    {
                        stepProbabilities[t] = p;
                    }
                }

                if (best != last)
                {
                    if (last != blank)
                    {
                        runs[count++] = new CtcRun(last, first, t - 1, runMax);
                    }

                    if (best != blank)
                    {
                        first = t;
                        runMax = p;
                    }
                }
                else if (best != blank && p > runMax)
                {
                    runMax = p;
                }

                last = best;
            }

            if (last != blank)
            {
                runs[count++] = new CtcRun(last, first, steps - 1, runMax);
            }

            float sum = 0f;
            for (int i = 0; i < count; i++)
            {
                sum += runs[i].Confidence;
            }

            confidence = count == 0 ? 0f : sum / count;
            return count;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }
    }

    /// <summary>Greedy decoding of one sequence, returning a new result. See <see cref="Greedy(ReadOnlySpan{float}, int, int, Span{CtcRun}, out float, int, CtcScores, Span{float})"/>.</summary>
    /// <param name="scores"><c>steps × classes</c> scores, row-major.</param>
    /// <param name="steps">Number of time steps, 0 or more.</param>
    /// <param name="classes">Number of classes including the blank, at least 1.</param>
    /// <param name="blank">The blank class. Default 0.</param>
    /// <param name="kind">Whether <paramref name="scores"/> are logits or log-probabilities.</param>
    /// <exception cref="ArgumentOutOfRangeException">A size or the blank index is out of range.</exception>
    /// <exception cref="ArgumentException"><paramref name="scores"/> is shorter than <c>steps × classes</c>.</exception>
    public static CtcGreedyResult Greedy(ReadOnlySpan<float> scores, int steps, int classes, int blank = 0, CtcScores kind = CtcScores.Logits)
    {
        Check(scores.Length, steps, classes, blank);
        var runs = new CtcRun[steps];
        int n = Greedy(scores, steps, classes, runs, out float confidence, blank, kind);
        return new CtcGreedyResult(n == runs.Length ? runs : runs.AsSpan(0, n).ToArray(), confidence);
    }

    /// <summary>Greedy decoding of a batch <c>[batch, steps, classes]</c>.</summary>
    /// <param name="scores">The scores, row-major.</param>
    /// <param name="batch">Number of sequences, 0 or more.</param>
    /// <param name="steps">Padded number of steps per sequence.</param>
    /// <param name="classes">Number of classes including the blank.</param>
    /// <param name="lengths">Optional valid steps per sequence (each in [0, steps]); steps beyond are ignored. Empty means all.</param>
    /// <param name="blank">The blank class. Default 0.</param>
    /// <param name="kind">Whether <paramref name="scores"/> are logits or log-probabilities.</param>
    /// <exception cref="ArgumentOutOfRangeException">A size, a length or the blank index is out of range.</exception>
    /// <exception cref="ArgumentException"><paramref name="scores"/> is too short, or <paramref name="lengths"/> is neither empty nor <paramref name="batch"/> long.</exception>
    public static CtcGreedyResult[] GreedyBatch(ReadOnlySpan<float> scores, int batch, int steps, int classes,
        ReadOnlySpan<int> lengths = default, int blank = 0, CtcScores kind = CtcScores.Logits)
    {
        if (batch < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batch), batch, "The batch size must not be negative.");
        }

        if (steps < 0 || classes < 1)
        {
            Check(0, steps, classes, blank);
        }

        long per = (long)steps * classes;
        if (scores.Length < per * batch)
        {
            throw new ArgumentException("scores is shorter than batch × steps × classes.", nameof(scores));
        }

        if (!lengths.IsEmpty && lengths.Length != batch)
        {
            throw new ArgumentException("lengths must have one entry per sequence.", nameof(lengths));
        }

        var result = new CtcGreedyResult[batch];
        for (int b = 0; b < batch; b++)
        {
            int len = lengths.IsEmpty ? steps : lengths[b];
            if (len < 0 || len > steps)
            {
                throw new ArgumentOutOfRangeException(nameof(lengths), len, $"The length of sequence {b} is outside [0, {steps}].");
            }

            result[b] = Greedy(scores.Slice((int)(b * per), (int)(len * (long)classes)), len, classes, blank, kind);
        }

        return result;
    }

    /// <summary>
    /// Prefix beam search without a language model: at each step the best <paramref name="beamWidth"/> prefixes are kept, and
    /// alignments that give the same text are added together (Hannun et al. 2014, Algorithm 1). With a beam at least as wide as
    /// the number of distinct prefixes it is the exact most probable text.
    /// </summary>
    /// <param name="scores"><c>steps × classes</c> scores, row-major. Must not contain NaN.</param>
    /// <param name="steps">Number of time steps, 0 or more.</param>
    /// <param name="classes">Number of classes including the blank, at least 1.</param>
    /// <param name="beamWidth">Prefixes kept per step, in [1, <see cref="MaxBeamWidth"/>]. Default 8.</param>
    /// <param name="topN">How many hypotheses to return at most, at least 1; fewer come back if the beam holds fewer. Default 1.</param>
    /// <param name="blank">The blank class. Default 0.</param>
    /// <param name="kind">Whether <paramref name="scores"/> are logits or log-probabilities.</param>
    /// <returns>Hypotheses best first (ties: the prefix created earlier); at least one.</returns>
    /// <exception cref="ArgumentOutOfRangeException">A size, the beam width or the blank index is out of range, or the search could create more than <see cref="MaxBeamNodes"/> prefixes.</exception>
    /// <exception cref="ArgumentException"><paramref name="scores"/> is too short or contains NaN.</exception>
    public static CtcHypothesis[] BeamSearch(ReadOnlySpan<float> scores, int steps, int classes, int beamWidth = 8, int topN = 1,
        int blank = 0, CtcScores kind = CtcScores.Logits)
    {
        Check(scores.Length, steps, classes, blank);
        if (beamWidth is < 1 or > MaxBeamWidth)
        {
            throw new ArgumentOutOfRangeException(nameof(beamWidth), beamWidth, $"The beam width must be in [1, {MaxBeamWidth}].");
        }

        if (topN < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(topN), topN, "topN must be at least 1.");
        }

        if (((long)steps * beamWidth * Math.Max(1, classes - 1)) + 1 > MaxBeamNodes)
        {
            throw new ArgumentOutOfRangeException(nameof(steps), steps, $"steps × beamWidth × (classes − 1) exceeds {MaxBeamNodes} prefixes.");
        }

        for (int i = 0; i < steps * classes; i++)
        {
            if (float.IsNaN(scores[i]))
            {
                throw new ArgumentException($"scores contains NaN at index {i}.", nameof(scores));
            }
        }

        // Prefix tree: node 0 is the empty prefix; a node is its parent plus one label.
        var parent = new List<int> { -1 };
        var label = new List<int> { -1 };
        var children = new Dictionary<long, int>();

        int[] beamNode = new int[beamWidth];
        double[] beamPb = new double[beamWidth], beamPnb = new double[beamWidth];
        int beamCount = 1;
        beamNode[0] = 0;
        beamPb[0] = 0.0; // log 1: the empty prefix has been "ended in blank" with certainty before the first step
        beamPnb[0] = double.NegativeInfinity;

        var slotOf = new Dictionary<int, int>();
        var nextNode = new List<int>();
        var nextPb = new List<double>();
        var nextPnb = new List<double>();
        double[] lp = new double[classes];
        const double NegInf = double.NegativeInfinity;

        for (int t = 0; t < steps; t++)
        {
            LogProbabilities(scores.Slice(t * classes, classes), kind, lp);
            slotOf.Clear();
            nextNode.Clear();
            nextPb.Clear();
            nextPnb.Clear();

            for (int i = 0; i < beamCount; i++)
            {
                int node = beamNode[i];
                double pb = beamPb[i], pnb = beamPnb[i];
                double total = CtcMath.LogAdd(pb, pnb);
                int c = label[node];

                // The prefix stays the same: a blank after anything, or its last label repeated without a blank in between.
                Add(node, total + lp[blank], NegInf);
                if (c >= 0 && pnb != NegInf)
                {
                    Add(node, NegInf, pnb + lp[c]);
                }

                for (int j = 0; j < classes; j++)
                {
                    if (j == blank)
                    {
                        continue;
                    }

                    // Extending by the last label needs a blank in between (pb); any other label may follow anything (total).
                    double v = (j == c ? pb : total) + lp[j];
                    if (v == NegInf)
                    {
                        continue;
                    }

                    long key = ((long)node << 32) | (uint)j;
                    if (!children.TryGetValue(key, out int ext))
                    {
                        ext = parent.Count;
                        parent.Add(node);
                        label.Add(j);
                        children[key] = ext;
                    }

                    Add(ext, NegInf, v);
                }
            }

            // Keep the best beamWidth prefixes; equal scores keep creation order, so the result does not depend on the sort.
            int m = nextNode.Count;
            int[] order = new int[m];
            double[] score = new double[m];
            for (int i = 0; i < m; i++)
            {
                order[i] = i;
                score[i] = CtcMath.LogAdd(nextPb[i], nextPnb[i]);
            }

            Array.Sort(order, (a, b) =>
            {
                int cmp = score[b].CompareTo(score[a]);
                return cmp != 0 ? cmp : a.CompareTo(b);
            });

            beamCount = Math.Min(m, beamWidth);
            for (int i = 0; i < beamCount; i++)
            {
                beamNode[i] = nextNode[order[i]];
                beamPb[i] = nextPb[order[i]];
                beamPnb[i] = nextPnb[order[i]];
            }
        }

        int outCount = Math.Min(beamCount, topN);
        var result = new CtcHypothesis[outCount];
        for (int i = 0; i < outCount; i++)
        {
            int node = beamNode[i];
            double logp = CtcMath.LogAdd(beamPb[i], beamPnb[i]);
            int depth = 0;
            for (int n = node; n > 0; n = parent[n])
            {
                depth++;
            }

            int[] labels = new int[depth];
            for (int n = node, k = depth - 1; n > 0; n = parent[n], k--)
            {
                labels[k] = label[n];
            }

            float conf = 0f;
            if (depth > 0)
            {
                double g = CtcMath.Exp(logp / depth);
                conf = (float)(g > 1.0 ? 1.0 : g);
            }

            result[i] = new CtcHypothesis(labels, logp, conf);
        }

        return result;

        void Add(int node, double addPb, double addPnb)
        {
            if (addPb == NegInf && addPnb == NegInf)
            {
                return;
            }

            if (!slotOf.TryGetValue(node, out int slot))
            {
                slot = nextNode.Count;
                slotOf[node] = slot;
                nextNode.Add(node);
                nextPb.Add(NegInf);
                nextPnb.Add(NegInf);
            }

            nextPb[slot] = CtcMath.LogAdd(nextPb[slot], addPb);
            nextPnb[slot] = CtcMath.LogAdd(nextPnb[slot], addPnb);
        }
    }

    private static void Check(int length, int steps, int classes, int blank)
    {
        if (steps < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(steps), steps, "The number of steps must not be negative.");
        }

        if (classes < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(classes), classes, "There must be at least one class.");
        }

        if ((uint)blank >= (uint)classes)
        {
            throw new ArgumentOutOfRangeException(nameof(blank), blank, $"The blank class must be in [0, {classes}).");
        }

        if ((long)steps * classes > length)
        {
            throw new ArgumentException("scores is shorter than steps × classes.");
        }
    }

    // The probability of class `best` at one step: for logits exp(z_best - max) / sum_j exp(z_j - max) from the exact softmax kernel
    // (so the same bits on every machine), for log-probabilities exp(value) from the exact exponential.
    private static float BestProbability(ReadOnlySpan<float> frame, int best, CtcScores kind, Span<float> tmp)
    {
        if (kind == CtcScores.LogProbabilities)
        {
            Span<float> one = tmp.Slice(0, 1);
            one[0] = frame[best];
            ElementwiseMath.Apply(ElementwiseFn.Exp, true, VecIsa.Auto, one, one);
            return Math.Min(1f, one[0]);
        }

        Softmax.Rows(true, VecIsa.Auto, frame, tmp, 1, frame.Length, frame.Length, frame.Length);
        return tmp[best];
    }

    // Natural-log probabilities of the classes of one step, in double: (z_j - max) - ln(sum_i exp(z_i - max)). The subtraction of the
    // float values are widened to double first, so the subtraction is exact; exp and log are the deterministic ones below.
    private static void LogProbabilities(ReadOnlySpan<float> frame, CtcScores kind, double[] lp)
    {
        if (kind == CtcScores.LogProbabilities)
        {
            for (int j = 0; j < frame.Length; j++)
            {
                lp[j] = frame[j];
            }

            return;
        }

        float max = frame[0];
        for (int j = 1; j < frame.Length; j++)
        {
            if (frame[j] > max)
            {
                max = frame[j];
            }
        }

        double sum = 0.0;
        for (int j = 0; j < frame.Length; j++)
        {
            sum += CtcMath.Exp((double)frame[j] - (double)max);
        }

        double ln = CtcMath.Log(sum);
        for (int j = 0; j < frame.Length; j++)
        {
            lp[j] = ((double)frame[j] - (double)max) - ln;
        }
    }
}

/// <summary>
/// Double-precision exp and ln from + − × ÷ and comparisons only, in the fixed order written in docs/ctc.md, so the beam search
/// returns the same bits on every platform (the runtime's own Math.Exp and Math.Log may differ between machines).
/// </summary>
internal static class CtcMath
{
    private const double Log2E = 1.4426950408889634;
    private const double Ln2Hi = 6.93147180369123816490e-01; // 32 significant bits, so k * Ln2Hi is exact for |k| < 2^20
    private const double Ln2Lo = 1.90821492927058770002e-10;
    private const double Ln2 = 0.6931471805599453;

    public static double Exp(double x)
    {
        if (!(x > -700.0))
        {
            return 0.0;
        }

        if (x > 700.0)
        {
            x = 700.0;
        }

        double k = Math.Floor((x * Log2E) + 0.5);
        double r = (x - (k * Ln2Hi)) - (k * Ln2Lo);

        // Taylor polynomial of exp of degree 13 on |r| <= 0.35, Horner's rule.
        double p = 1.0 / 6227020800.0;
        p = (p * r) + (1.0 / 479001600.0);
        p = (p * r) + (1.0 / 39916800.0);
        p = (p * r) + (1.0 / 3628800.0);
        p = (p * r) + (1.0 / 362880.0);
        p = (p * r) + (1.0 / 40320.0);
        p = (p * r) + (1.0 / 5040.0);
        p = (p * r) + (1.0 / 720.0);
        p = (p * r) + (1.0 / 120.0);
        p = (p * r) + (1.0 / 24.0);
        p = (p * r) + (1.0 / 6.0);
        p = (p * r) + 0.5;
        p = (p * r) + 1.0;
        p = (p * r) + 1.0;
        return p * BitConverter.Int64BitsToDouble((long)((int)k + 1023) << 52);
    }

    public static double Log(double x)
    {
        if (x == 0.0)
        {
            return double.NegativeInfinity;
        }

        if (!(x > 0.0) || double.IsPositiveInfinity(x))
        {
            return x;
        }

        long bits = BitConverter.DoubleToInt64Bits(x);
        int e = (int)((bits >> 52) & 0x7FF);
        if (e == 0)
        {
            // Subnormal: scale by 2^54 first.
            bits = BitConverter.DoubleToInt64Bits(x * 18014398509481984.0);
            e = (int)((bits >> 52) & 0x7FF) - 54;
        }

        e -= 1023;
        double m = BitConverter.Int64BitsToDouble((bits & 0x000FFFFFFFFFFFFFL) | 0x3FF0000000000000L);
        if (m > 1.4142135623730951)
        {
            m *= 0.5;
            e++;
        }

        // ln m = 2 atanh(s) with s = (m - 1) / (m + 1), |s| <= 0.1716: 2 s (1 + s^2/3 + s^4/5 + ... + s^18/19).
        double s = (m - 1.0) / (m + 1.0);
        double z = s * s;
        double q = 1.0 / 19.0;
        q = (q * z) + (1.0 / 17.0);
        q = (q * z) + (1.0 / 15.0);
        q = (q * z) + (1.0 / 13.0);
        q = (q * z) + (1.0 / 11.0);
        q = (q * z) + (1.0 / 9.0);
        q = (q * z) + (1.0 / 7.0);
        q = (q * z) + (1.0 / 5.0);
        q = (q * z) + (1.0 / 3.0);
        q = (q * z) + 1.0;
        return (e * Ln2) + (2.0 * s * q);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double LogAdd(double a, double b)
    {
        if (a < b)
        {
            (a, b) = (b, a);
        }

        if (double.IsNegativeInfinity(b))
        {
            return a;
        }

        double d = b - a;
        return d < -40.0 ? a : a + Log(1.0 + Exp(d));
    }
}
