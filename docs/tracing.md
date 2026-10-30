# Tracing: tensor-layer models into the graph layer

Normative for what `Tracer` records and refuses. Source: `src/OneModel/Tracing/`, the seam in `src/OneModel/Tensors/Tracing/` and
`Tensor.cs`, `Autograd/BackwardRecord.cs`. The graph written is ONNX (opset 20, IR version 9) as defined by `Operators.md`, so a
traced model is an ordinary graph: it runs through `Session` with the usual passes, saves with `ModelFormat`, and any ONNX consumer can run
`TracedModel.ToOnnx()`.

## 1. Method

Tracing is a recorded eager run. `Tracer.Begin()` activates a recording on the calling thread. The model runs once on example inputs
(`Tracer.Input`), computing real values as it always does. Every operator that already builds a `BackwardRecord` (the records carry the
operator, its input tensors, integer and float attributes) is lowered to ONNX nodes when at least one input is *traced*, that is, depends
on an `Input`. While a trace records, traced tensors report `RequiresGrad`, so the existing "record when an input requires gradients"
paths fire without edits to the operators; the result tensor gets a tracer value instead of a plain autograd identity.

- **Constants.** A tensor that does not depend on an input (a weight, a buffer, a scalar, a transposed copy of a weight) used by a traced
  operator becomes an initializer holding a contiguous copy of the logical elements at that moment, deduplicated by storage, offset,
  shape and strides. Operators whose inputs are all constants (a weight's `T()`, `Add` of two buffers) run eagerly and stay constants;
  no node is written. `Tracer.Parameter(name, tensor)` names a weight; a differently strided view of a named weight is named
  `<name>.view_<n>`. `NnModule.Trace` (training package) names every parameter and buffer by its state-dict path.
- **Only what the outputs need is kept.** Scratch work recorded inside composite operators, and values nothing used, are pruned when
  `Finish` runs. Output values are renamed to the requested names (an `Identity` node when an output is an input or repeated).
- **Static graphs only.** Python-style control flow of the model (loops over layers, `if` on sizes or flags) runs once at trace time and is
  unrolled. Anything decided from tensor *values* is refused (section 3).
- **Aliases.** A reshape to the same shape, a copy (`Contiguous`, `Clone`) and a full-range `Slice` write no node: the result shares the
  value of its input (a copy of a traced tensor keeps its identity). `Detach` of a traced tensor returns it unchanged.

## 2. Operator mapping

| Eager operation (record) | ONNX |
|---|---|
| `Add`, `Sub`, `Mul`, `Div` | same, numpy broadcasting |
| `Neg`, `Relu`, `Sigmoid`, `Tanh` | same |
| `Gelu` (erf or tanh) | `Gelu`, `approximate` none / tanh |
| `MatMul` | `MatMul` |
| `Softmax`, `LogSoftmax` (last axis) | same, `axis` = -1 (other axes arrive as `Permute` views around it) |
| `LayerNorm` | `LayerNormalization`, `axis` -1, `epsilon` |
| `BatchNorm2d`, evaluation mode | `BatchNormalization` with the running statistics as constants (training mode is refused) |
| `Conv2d`, `ConvTranspose2d` | `Conv`, `ConvTranspose` (strides, pads, dilations, group; output padding recovered from the shapes) |
| `MaxPool2d`, `AvgPool2d` | `MaxPool`, `AveragePool` (`count_include_pad`) |
| `Resize2d` | `Resize` with explicit `sizes`; nearest = `asymmetric` + `floor`, bilinear = `half_pixel` or `align_corners` |
| `Concat` | `Concat` |
| `Embedding` | `Gather` axis 0 (the padding index only matters for gradients) |
| `Attention` | `Transpose`, `MatMul`, `Mul` by the scale, `Add` of the mask (a `[B,L,L]` mask gets an `Unsqueeze`), `Softmax`, `MatMul`: the pattern the attention fusion pass recognises |
| `LstmLayer`, `GruLayer` | `LSTM`, `GRU` (one direction, `linear_before_reset` = 1 for GRU); PyTorch's gate order is converted to ONNX's by `Gather` on the weight rows (constants fold), initial states by `Unsqueeze`, `Y` by `Squeeze`. `Lstm`/`Gru` stacks are compositions |
| `Sum`, `Mean` (all or one axis) | `ReduceSum`, `ReduceMean` with `keepdims` |
| `Permute`, `Transpose` | `Transpose` |
| `Reshape`, `Squeeze`, `Unsqueeze`, copying reshape | `Reshape` (section 4) |
| `Expand` | `Expand` |
| `Slice`, `Select`, `Flip` | `Slice`; `Gather` with a scalar index; `Slice` with step -1 |
| `Cast` | `Cast` |
| `Linear`, `GlobalAvgPool2d`, `CrossEntropy`, multi-layer RNNs | compositions of the above |

**Not traceable** (`TraceException`, kind `UnsupportedOperation`, naming the operator): `CtcLoss` and `NllLoss` (and so `CrossEntropy`):
losses have no ONNX equivalent in the opset used and tracing is for inference graphs; `BatchNorm2d` in training mode; `Dropout` in training
mode (a random mask would be frozen); blocked-layout tensors as constants; constants over 2 GiB. Operators added to `Ops` later fail the
same way until they get a mapping.

## 3. What fails loudly, and when

All failures are `TraceException` with a `TraceErrorKind`, raised at the offending call while tracing, naming the read or operation.

- `DataDependent`: `ToArray`, `Get`, `Data`, `Storage`, or `CopyFrom` out of a traced tensor. The library's own kernels read through
  internal accessors, so only user code is stopped. (Reading a traced tensor *after* `Finish` is allowed: the values are the example run's.)
- `InPlaceMutation`: `Fill`, `Set`, `MutableData`, `MutableStorage`, `CopyFrom` into a traced tensor; and, at `Finish`, a captured constant
  whose storage version changed during the trace (the graph holds the earlier values).
- `NoGradInTrace`: `GradMode.NoGrad()` while tracing. Results computed inside would not be recorded and would be frozen as constants.
- `SymbolicShape`: a symbolic axis reaches something that cannot express it (a partial slice of a symbolic axis, a reshape that mixes
  symbolic sizes ambiguously, a symbolic spatial axis into `Resize2d`).
- `InvalidUsage`: nested tracers on a thread, a name used twice, an output that does not depend on an input, a symbolic axis whose
  example size is below 2 or equals another symbol's.

## 4. Symbolic axes

`Input(name, example, axisNames)` may name axes whose size can differ at run time (batch, sequence, width). The graph input then has
`dim_param` names, and the session plans per concrete input shape as it does for any model with symbolic dimensions.

Eager code takes sizes from shapes (`x.Shape[0]`) as plain integers, so the tracer keeps, for every traced value, which axes are
symbolic as `coefficient × product of atoms`, where an atom is a declared symbol or an extent derived by an operation (a convolution's
output width, a concatenation). It is propagated through elementwise operators (broadcasting), matrix products, norms, reductions,
permutes, selects, concatenations, convolutions and pools (batch carries over, a symbolic spatial extent becomes a new atom held by the
output), embeddings and recurrent layers. A `Reshape` splits both shapes into groups of equal element count and gives each output
dimension the atoms whose example sizes multiply to it (exact products first, then divisors); this is why symbolic example sizes must be at
least 2 and mutually distinct, and why a static dimension equal to a symbol's size can be mistaken for it (trace with sizes that avoid the
model's fixed sizes). The shape operand of a `Reshape` or `Expand` is then:

- all constants, when nothing is symbolic;
- the constant with one `-1`, when exactly one extent is a single symbol;
- otherwise computed from `Shape(source, start, end)` nodes for the atoms, multiplied, and concatenated with the constants.

Plan-time constant evaluation (`graph-runtime.md`) folds those chains per input-shape signature, so they cost nothing per run.

## 5. Exactness

The traced graph runs the graph layer's operators, so exact mode is the graph layer's: a session in `ExecutionMode.Exact` fails at
creation naming any operator without an exact kernel, and otherwise is bit-identical across machines. Traced and eager results are
equal to the last bits in fast mode (checked to 1e-5 scaled; measured ~1e-7) and in exact mode where both layers share the kernel
(convolution, matrix product, pooling, activations); they differ in the last bits where the layers sum differently (eager norms and
softmax-type rows accumulate in double, the graph's norms and softmax use the kernels of `docs/ops-*.md`). The graph passes' bit rules are
those of `graph-passes.md`. The attention written as separate operators is fused back by the attention pass in fast mode only.

## 6. Output formats

`TracedModel.Save` writes the library format v1 (`format.md`) from the recorded graph before the session's passes, like any imported model;
`Model.Load`/`Session.Load` read it back and give outputs bit-identical to the in-memory session. `SaveOnnx` writes the ModelProto the
writer in `src/OneModel/Onnx/OnnxWriter.cs` produces (fields in field-number order, so equal graphs give equal bytes); ONNX Runtime
1.31.1 loads these files and agrees with the eager outputs at the traced and at other shapes (`tools/trace_oracle.py`).
