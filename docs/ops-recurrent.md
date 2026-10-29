# Recurrent operators: RNN, GRU, LSTM

Semantics are those of `Operators.md` (opsets 7, 14, 22 are one implementation: the attributes and inputs did not change in a way
that needs a second one; `layout` arrived in opset 14). This note records what the specification leaves open, how each point was
settled, and the order of operations that makes exact mode bit-identical everywhere.

## Support

| | |
|---|---|
| Operators | `RNN`, `GRU`, `LSTM`, default domain, opset 7 and later |
| Element types | float32 (X, W, R, B, initial state, P, Y, Y_h, Y_c); `sequence_lens` is int32 |
| Directions | `forward`, `reverse`, `bidirectional` |
| `layout` | 0 (`[seq, batch, ...]`) and 1 (`[batch, seq, ...]`); state tensors `[dirs, batch, hidden]` or `[batch, dirs, hidden]` |
| Inputs | X, W, R, B, sequence_lens, initial_h; LSTM also initial_c and P (peepholes), in the specification's order; any optional one may be omitted |
| Outputs | Y, Y_h; LSTM also Y_c. Any may be unused |
| Attributes | `direction`, `hidden_size`, `layout`, `activations`, `activation_alpha`, `activation_beta`, `clip`; LSTM `input_forget`; GRU `linear_before_reset` |
| Not supported | float16 and double recurrent weights |
| Exact mode | supported for all three |

Gate order inside W, R and B follows the specification: LSTM `i, o, f, c`; GRU `z, r, h`; RNN one gate. `B` is `[dirs, 2 * gates * hidden]`:
the W biases then the R biases. `P` is `[dirs, 3 * hidden]` in the order `i, o, f`.

## Computation

For one direction, with `G` gates and `H` hidden units:

1. The input projections of **all** time steps are one matrix product, `XW = X · Wᵀ` of shape `[seq * batch, G * H]`, on the shared
   GEMM (accumulation order of `kernels-gemm.md`, one accumulator per output, increasing k from +0).
2. For step `s` the recurrent product `Rec = H · Rᵀ` (`[batch, G * H]`, also the shared GEMM) is formed for the whole batch. For GRU with
   `linear_before_reset = 0` only the `z` and `r` columns are formed at this point, and the candidate's product `(r ⊙ H) · Rhᵀ` is a second
   small product after `r` is known.
3. The gate arithmetic below runs per batch row.

Weights are transposed once per plan when they are constants; otherwise on each run into scratch.

**Pre-activation (all cells), in this order:** `pre = ((xw + rec) + Wb) + Rb`, each `+` one rounding, bias terms skipped when `B` is absent.
For LSTM the peephole term is added after that: `pre_i += (Pi * c)`, `pre_f += (Pf * c)` with the previous cell, and `pre_o += (Po * c)` with the
new cell. `clip`, when present and positive, then clamps the pre-activation to `[-clip, clip]` (a NaN passes through); it applies to every
gate's input, as the specification says ("applied to the input of activations").

**LSTM:** `i = f(pre_i)`, `fg = f(pre_f)` (or `1 - i` with `input_forget`), `ct = g(pre_c)`, then
`c = (fg * c_prev) + (i * ct)` as two products and one sum, `o = f(pre_o)`, `h = o * h_act(c)`.

**GRU:** `z = f(pre_z)`, `r = f(pre_r)`.
`linear_before_reset = 0`: `pre_h = ((xw_h + (r ⊙ H_prev) · Rhᵀ) + Wbh) + Rbh`.
`linear_before_reset = 1`: `pre_h = (xw_h + (r * (rec_h + Rbh))) + Wbh`.
`ht = g(pre_h)`, `H = ((1 - z) * ht) + (z * H_prev)`, again as separate roundings.

**RNN:** `H = f(pre)`.

No step fuses a multiply with an add. The only fused arithmetic in fast mode is inside the GEMM (`kernels-gemm.md`).

## Activations

Names are matched case-insensitively. `f`, `g`, `h` are the first, second and third entries of `activations` per direction (defaults:
LSTM `Sigmoid, Tanh, Tanh`; GRU `Sigmoid, Tanh`; RNN `Tanh`). For a bidirectional layer the list may name either one set (used for both
directions) or both sets in order forward, reverse.

Supported: `Sigmoid`, `Tanh`, `Relu`, `Affine`, `LeakyRelu`, `ThresholdedRelu`, `ScaledTanh`, `HardSigmoid`, `Elu`, `Softsign`, `Softplus`.
`Sigmoid`, `Tanh`, `Elu`, `ScaledTanh` and `Softplus` use the library's own polynomials (`exact-transcendentals.md`); the others are
compares, selects and IEEE operations. In exact mode every one is bit-identical on every machine.

`activation_alpha` and `activation_beta` are consumed in activation order, **one value per activation that takes the parameter**, with separate
cursors for alpha and beta (`Affine`, `ScaledTanh`, `HardSigmoid` take both; `LeakyRelu`, `ThresholdedRelu`, `Elu` take alpha). Missing values default
to the activation's own defaults (`LeakyRelu` 0.01, `ThresholdedRelu` 1, `Elu` 1, `HardSigmoid` 0.2 and 0.5, `Affine` and `ScaledTanh` 1 and 0).
Settled by black box: recorded ONNX Runtime outputs for a `HardSigmoid, LeakyRelu, ScaledTanh` LSTM with `activation_alpha = [0.3, 0.2, 1.2]` and
`activation_beta = [0.4, 0.7]` (separate cursors) match.

Definitions: `Affine(x) = alpha * x + beta`; `LeakyRelu(x) = x < 0 ? alpha * x : x`; `ThresholdedRelu(x) = x > alpha ? x : 0`;
`ScaledTanh(x) = alpha * tanh(beta * x)`; `HardSigmoid(x) = min(1, max(0, alpha * x + beta))`; `Elu(x) = x >= 0 ? x : alpha * (exp(x) - 1)`;
`Softsign(x) = x / (1 + |x|)`; `Softplus(x) = x > 20 ? x : log(1 + exp(x))`; `Relu` keeps a NaN.

## Choices the specification leaves open (settled by recorded ONNX Runtime output, tools/ops_recurrent_oracle.py)

- **`sequence_lens`.** Row `b` runs its first `len_b` steps only; in the reverse direction it starts at step `len_b - 1`. Y is zero at
  steps `t >= len_b`. `Y_h` and `Y_c` are the state after the row's last valid step. **A row of length 0 has `Y_h = Y_c = 0`** (not the initial state);
  the reference does this, and the specification is silent. A length outside `[0, seq]` is an `InvalidInput` error at run time.
- **`input_forget`.** Couples the gates as `f = 1 - i`; recorded outputs match.
- **`layout = 1`.** The recorded ONNX Runtime CPU kernels refuse `layout = 1`, so the layout-1 cases are the layout-0 case with every tensor transposed as
  `Operators.md` defines (X and the initial states batch-first; Y `[batch, seq, dirs, hidden]`; Y_h/Y_c `[batch, dirs, hidden]`).
- **Initial state when only `initial_c` or `P` is given.** Omitted inputs are zeros (no peepholes).
- **Zero-size.** `seq = 0` or `batch = 0` runs nothing; Y is empty and the final states are the initial states (or zeros).
- **`hidden_size`.** Optional; when present it must equal R's last dimension. Zero hidden size is an error.

## Exact mode

All three operators are exact for float32. The canonical order is the one above; the matrix products follow `kernels-gemm.md`; the activation
functions follow `exact-transcendentals.md`; every NaN in Y, Y_h and Y_c is written as `0x7FC00000`. The scalar gate loops are IEEE
operations in the written order, so they are identical on every path. Tests run each recorded case on every kernel path the machine supports and
with 1 and 3 threads and require equal hashes.

## Fast mode

Same order; the GEMM may fuse multiply-adds and the elementwise functions use the fast polynomials. Results do not depend on the thread count
(the GEMM's chunking is fixed).

## Limits

The per-direction scratch (`seq * batch * G * H` floats for the projections plus a few `batch * H` buffers and the transposed weights when not
constant) must stay below 2^30 floats; a larger layer is refused with `Unsupported` at plan time.
