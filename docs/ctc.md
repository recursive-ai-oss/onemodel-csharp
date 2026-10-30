# CTC decoding

A helper for consumers of recognisers and speech models (not an ONNX operator): `RecursiveAI.OneModel.Tensors.Ctc`. It turns per-step class
scores into label sequences. Scores are row-major `[steps, classes]` (a batch is `[batch, steps, classes]`), either logits or natural-log
probabilities that already normalise over the classes of each step. The blank class is configurable (default 0). Every function is pure and
gives the same result on every machine: probabilities come from the exact softmax and exponential (`exact-transcendentals.md`), and the beam
search uses the double-precision exp and log defined below.

## Greedy (best path)

Graves et al. 2006, section 3.1.

1. At each step choose the class with the largest score, the **lowest index on ties**. A NaN is never chosen over a number.
2. Merge consecutive equal classes into runs and drop the blank. Two equal labels separated by a blank are two labels.
3. The probability of a step's class `p(t)` is
   - logits: `exp(z_best - max) / sum_j exp(z_j - max)`, taken from the exact softmax kernel (`kernels-attention.md`); it equals `1 / sum_j exp(z_j - z_best)`;
   - log-probabilities: `exp(value)` with the exact exponential, capped at 1.

**Confidences.** A *run* is the steps of one emitted label; its confidence is the **largest `p(t)` of its class over the run**. The sequence confidence
is the **mean of the run confidences** in order (float32 sum, then divide by the run count); 0 when nothing is emitted. This is the usual
line-recogniser convention: the mean over emitted characters of each character's largest softmax probability. Per-step
confidence (`stepProbabilities`) is `p(t)` of the step's best class, blank included.

Each run also carries its first and last step, which consumers use for approximate alignment (a character occupies `[first, last]`).

The span entry point allocates nothing in steady state (it borrows a scratch row from the shared array pool). Output room: `steps` runs.

## Prefix beam search

Graves et al. 2006; Hannun et al. 2014, Algorithm 1. No language model.

State: a set of at most `beamWidth` prefixes, each with `pb` and `pnb`, the log probabilities of all alignments of the steps so far that give the prefix and
end in blank / in a non-blank. Start: the empty prefix with `pb = 0`, `pnb = -inf`.

For each step `t`, with `lp(j)` the log probability of class `j`, and for each prefix `p` with last label `c` (none for the empty prefix) and
`total = logadd(pb, pnb)`:

- `p` gets `pb += total + lp(blank)`;
- if `c` exists: `p` gets `pnb += pnb + lp(c)` (the run of `c` continues);
- for every class `j` other than the blank, the prefix `p·j` gets `pnb += (j == c ? pb : total) + lp(j)`; contributions equal to probability 0 are not added.

Contributions to one prefix are combined with `logadd`; new prefixes are listed in creation order. After each step the prefixes are ranked by
`logadd(pb, pnb)`, best first, equal scores in creation order, and the first `beamWidth` are kept. After the last step the best `topN` are returned,
each with `LogProbability = logadd(pb, pnb)` and `Confidence = exp(LogProbability / n)` for `n` labels (the per-label geometric mean; 0 for the empty
text). With a beam at least as wide as the number of distinct prefixes the search is exact: the best text over the sum of all alignments.

`lp(j)` for logits is `(z_j - max) - ln(S)` with `S = sum_i exp(z_i - max)`, accumulated in increasing `i` order in double precision (the float values are widened
first, so the subtraction is exact); for log-probabilities it is the given value.

Limits: `beamWidth` in `[1, 4096]`; `steps * beamWidth * (classes - 1) + 1` at most 2^24 (the most prefixes the search can create); NaN scores are refused.
The search allocates (a prefix tree and the result); it is a decoding step, not a per-sample kernel.

### Deterministic exp and ln (double)

Only `+ - * /`, comparisons and exact `floor`, in the order written; no fused multiply-add. Relative error below 1e-15 on the ranges used.

- `exp(x)`: `x <= -700` (or NaN) gives 0, `x > 700` is clamped to 700; `k = floor(x * 1.4426950408889634 + 0.5)`;
  `r = (x - k * 6.93147180369123816490e-01) - k * 1.90821492927058770002e-10`; the degree-13 Taylor polynomial of exp in Horner form with coefficients
  `1/n!` (`n = 13 … 0`), each step `p = p * r + c`; the result is `p * 2^k` built from the exponent bits.
- `ln(x)`: `x = m * 2^e` with `m` in `[1, sqrt 2)` (halve `m` and add 1 to `e` when it is above `sqrt 2`; subnormals are first scaled by 2^54);
  `s = (m - 1) / (m + 1)`, `z = s * s`, `q` the Horner form of `1 + z/3 + z²/5 + … + z⁹/19` (coefficients `1/(2j+1)`), result `e * 0.6931471805599453 + 2 * s * q`.
  `ln 0 = -inf`.
- `logadd(a, b)`: with `a >= b`, `a` if `b` is minus infinity or `b - a < -40`, else `a + ln(1 + exp(b - a))`.

## Loss (training)

`Ops.CtcLoss` is the negative log of the total probability of all alignments of the target (Graves et al. 2006). With extended labels
`l'` of length `2L+1` (blank at even positions, target label `s/2` at odd positions), `alpha[0][0] = lp[0][blank]`, `alpha[0][1] = lp[0][l'_1]`
and for `t > 0`: `alpha[t][s] = logadd(alpha[t-1][s], alpha[t-1][s-1], alpha[t-1][s-2]) + lp[t][l'_s]` (sums in log space, the `s-2` term only
onto a non-blank that differs from the label two positions back). The loss is `-logadd(alpha[T-1][2L], alpha[T-1][2L-1])` over the `T` valid
frames of the sequence; no alignment gives +infinity. All arithmetic is `double` with the `logadd`, `exp` and `ln` above in the written order.
`beta` is the same recursion from the end (including the emission at its own frame). The gradient of the loss with respect to `lp[t][c]` is
`-exp(logsum_{s: l'_s = c}(alpha[t][s] + beta[t][s]) - lp[t][c] + loss)` (0 if the sum is empty), 0 for frames at or beyond the input length.
`Mean` reduction divides each loss by `max(1, L)` and averages over the batch. See `autograd.md` section 8 for the PyTorch comparison.

## Checks

`tools/ctc_reference.py` is a straightforward double-precision Python implementation (numpy only): greedy, a dictionary-based prefix beam search, and for
small sizes the exact best text by enumerating every alignment. `testdata/ctc/cases.json` holds its output for 40 seeded random cases; the tests require the
same labels, run and step confidences within 2e-6, beam log probabilities within 1e-9, and the wide beam equal to the exhaustive result.
