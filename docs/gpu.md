# Running on the GPU (`RecursiveAI.OneModel.Gpu`)

`ModelOptions.Device = ComputeDevice.Gpu` offers a session's operators to a graphics processor, through the `RecursiveAI.OneModel.Gpu`
package (Vulkan, Metal or browser WebGPU, from managed code, on `RecursiveAI.OneGpu` 1.0.0 and its backend packages). The core package knows nothing about it: the GPU package registers
itself when asked.

```csharp
OneModelGpu.Register();                                    // once per process
using var session = Session.Load("model.onnx", new ModelOptions { Device = ComputeDevice.Gpu });
Console.WriteLine(session.GetDiagnostics(inputs).NodesOnDevice);
```

## Which operators run where

- **A kernel table decides.** Each GPU operator is one entry (operator type, whether a deterministic kernel exists, a check of
  attributes and opset, and a binder that sees element types and shapes). Float32, float16, int32, int64 and boolean tensors, uint8
  activations of the quantized product, uint8 and int8 values that a `QuantizeLinear` makes and a `DequantizeLinear` reads, and uint8 tables and values that a lookup and a `DequantizeLinear` handle, run on the device; other element types (16-bit integers, uint8 elsewhere,
  float64, strings) stay on the CPU. A boolean tensor is one byte per element, as on the host.

  | Family | Operators |
  |---|---|
  | Matrix product | `MatMul` (numpy batch broadcasting when B has no batch, both batches are equal, or A has none), `Gemm` (`transA`, `transB`, `alpha`, `beta`, a `C` that broadcasts) |
  | Attention | the fused attention node (below), one kernel for `softmax(scale * Q K^T + mask) V` |
  | Elementwise, with ONNX broadcasting | `Add` `Sub` `Mul` `Div` `Pow` (float only) `Min` `Max` (two inputs; `Add` `Sub` `Mul` `Div` `Min` `Max` `Neg` `Abs` also on int32 and int64), `Sqrt` `Reciprocal` `Exp` `Log` `Tanh` `Erf` `Sigmoid` `Relu` `LeakyRelu` `Clip` `Gelu` |
  | Normalisation and reductions | `LayerNormalization` (opset 17; the `Mean` and `InvStdDev` outputs must be unused), `Softmax` (opset 13), `ReduceSum` `ReduceMean` `ReduceMax` `ReduceMin` `ReduceProd` `ReduceL1` `ReduceL2` `ReduceSumSquare` `ReduceLogSum` (axes constant) |
  | Integers, booleans, masks | `Mod` (int32, int64; both `fmod` forms) `Equal` (float, int, bool) `Less` `Greater` `LessOrEqual` `GreaterOrEqual` (float, int) `And` `Or` `Xor` `Not` `Where` (float, int values) `Expand` (a constant shape) `Range` and `ConstantOfShape` (constant inputs), all bit-exact against the CPU |
  | Lookup | `Gather` (float32, float16, uint8, int32, int64 or boolean data, int32 or int64 indices; float16 rows need an even length) |
  | Layout | `Transpose`, `Cast` between float32, float16, int32, int64 and bool, and the views `Reshape` `Flatten` `Squeeze` `Unsqueeze` `Identity` `Dropout` (of any of those types) |
  | Vision | `Conv` (one or two spatial axes; groups, depthwise, dilation, `auto_pad`, bias; the fused forms below; float16 models' convolutions run in float16, with float32 accumulation), `ConvTranspose` (explicit pads and `output_padding`, groups), `MaxPool` (without the indices output), `AveragePool` (`ceil_mode`, `count_include_pad`), `GlobalAveragePool`, `GlobalMaxPool`, `Resize` (nearest and linear; every coordinate mode but `tf_crop_and_resize`; constant scales or sizes on the two spatial axes), `Concat` (one dispatch per four inputs, however large the batch), `HardSwish`, `HardSigmoid`; float32 or float16 tensors in NCHW, weights OIHW. A float16 model's convolutions are not fused with the activation and residual that follow (the graph rewrites leave the float16 form as written), so each is its own pass |
  | 8-bit | `DynamicQuantizeLinear` (float32 input), and the dequantizing product `MatMulInteger, Cast, Mul [, Add]` that the graph rewrite makes (uint8 A, constant uint8 or int8 B, optional zero points, a scalar or per-column scale, an optional constant bias), and `QLinearMatMul` (uint8 A, constant uint8 or int8 B, constant scales and output zero point, zero points per tensor or per row/column, onemodel's optional int32 bias; uint8 result), and `DequantizeLinear` (uint8, int8 or int32 input; per tensor or per axis, optional zero point of the input's type, float32 scale and result, no `block_size`), and `QuantizeLinear` (float32 input and scale, uint8 or int8 result with a zero point of that type, per tensor or per axis, no `block_size`; 16-bit results stay on the CPU), and `ConvInteger` (one or two spatial axes; uint8 or int8 input and weights in any mix; a scalar input zero point and a scalar or per-channel weight zero point; groups, dilation, strides, padding, `auto_pad`; int32 result), bit-exact against the CPU. `ConvInteger` unfolds the windows into a column matrix kept in the arena for the length of the step (at most 64 MiB, the batch runs in chunks of images beyond that) and multiplies with the 8-bit product |
  | Recurrent | `LSTM`, `GRU`, `RNN` (float32; forward, reverse, bidirectional; layouts 0 and 1; bias, initial states, peepholes, `clip`, `input_forget`, `linear_before_reset`, `sequence_lens`, every activation ONNX names with its `activation_alpha` and `activation_beta`); a bidirectional LSTM recognizer is one region with its convolutions |

- **Vision models.** ResNet-50, MobileNetV3 and the detector of the reference set run as one region with all of their operators on the
  device (`RequireGpu` passes). The graph rewrites the convolution nodes take are the device's too: batch-norm folding (in both modes), the
  activation (`Relu`, `Clip`, `HardSwish`, `HardSigmoid`, `Sigmoid`, `Gelu`, `Tanh`, `LeakyRelu`) and the residual `Add` fused into the
  convolution's last pass (also in exact mode, where the CPU would not fuse the residual: the device's own kernel takes it in either mode).
  A residual that broadcasts, a convolution of three spatial axes, `Resize` with cubic interpolation, `antialias`, `exclude_outside` or
  run-time scales, and `MaxPool` with its indices output stay on the CPU, each with its reason in the diagnostics. Float16 convolutions
  `QLinearConv` is not mapped; `ConvInteger` is (below). The quantize-dequantize form of the 8-bit vision models (ResNet-50 and
  MobileNetV3 with a `QuantizeLinear`/`DequantizeLinear` pair around each float operator) runs as one region: each pair is computed on the device with the CPU's
  arithmetic, and the convolutions between them are the float32 ones, so the result is the quantized model's, not the integer kernels' (their rounding
  differs in the last bit, which a quantization step can turn into a one-step difference that spreads). Against the CPU's exact mode the
  model-level error is of the size of the CPU's own fast-versus-exact difference on the same input (ResNet-50 1.7e-2, MobileNetV3 up to 1.3e-1).
  A concatenation is one copy per input and image (the device's concatenation refuses an input in the output's buffer, and an arena slab
  holds both), so a batch of more than a few hundred copies stays on the CPU.
- **Everything else stays on the CPU**, per operator, unless `RequireGpu` is set. The reason for each operator that stayed is in
  `GetDiagnostics`: no kernel, a type or attribute the kernel does not take, a size above a device limit, a shape that is only known at
  run time, or exact mode with no deterministic kernel. The attention-mask chain of MiniLM and BERT (`Cast`, `Gather` of the mask,
  `And`, `Expand`, `Where`, and at larger sizes the `Range`, `ConstantOfShape` and comparison that build the causal triangle) runs on the
  device, so a float32 or float16 encoder is one region and passes `RequireGpu`. A dynamically quantized (int8) encoder is one region too: its
  embedding tables' `Gather` (uint8 data) and the `DequantizeLinear` after it run on the device, and the rows are dequantized with the
  CPU's own arithmetic (`(float)(x - zero_point) * scale`, one rounding), so the values match the CPU's bit for bit.
- **Recurrent layers.** `LSTM`, `GRU` and `RNN` use the layer's attributes as the CPU operator resolved them, and the weights as stored (no repacking). The
  input projection of every step is one batched matrix product, then one dispatch per step (a dispatch cannot wait for every other
  workgroup of itself, so the steps are separate passes); each recurrent dot product is summed in an order fixed by the shapes, so fast and
  exact mode run the same kernel and a run is bit-identical to the next. `sequence_lens` is checked on the host with the CPU's message and
  range (at planning for a constant, on every run for a value from the host); lengths made on the device stay on the CPU. A layer's two
  scratch buffers (projection and states) are shared by all layers of a session, sized at planning, counted in `ArenaBytes`. Float16 layers
  and an empty sequence or batch stay on the CPU.
- **Two operators that are built from others.** `Expand` is a selection (`Where` for numbers, `And` for booleans) over a scratch of "true"
  values of the target shape, because the device's own expand needs separate input and output buffers and an arena slab is one buffer. A
  `Gather` of booleans widens the data to int32 scratch, gathers, and narrows it back. Both are exact for 0 and 1.
- **Views cost nothing.** `Reshape` and its kin give their result the memory of their input: the arena hands the result the same slot,
  and the step records no commands. A view whose result a CPU operator reads (or the graph returns) stays on the CPU, because the device
  holds the only copy of its source; the source is then read back instead.
- **Regions.** Operators on the GPU form a region. The plan orders the schedule so that operators the GPU did not take run as early as their
  inputs allow, which puts the GPU's operators next to each other (unrelated CPU work no longer splits a region). A region is one upload
  of the tensors it reads from the CPU, one command submission, and one readback of the tensors the CPU reads afterwards (or the graph
  returns). Nothing is copied inside a region, so a model whose operators all run on the GPU moves its inputs and outputs only. Weights are
  uploaded once per session; float16 weights with an odd element count are padded by one element, because the kernels read half-precision
  data in pairs.
- **Graph rewrites.** The device chooses which of the session's rewrites run. It takes the convolution rewrites (above), the fused attention (`softmax` over the scaled, masked
  scores between two products, with the head-split and head-merge transposes absorbed: the kernel reads and writes the strided layouts),
  the one-pass `Gelu` made from the five-operator form, and the dequantizing 8-bit product, in both modes (the device kernels are
  deterministic). It declines the CPU's packed-weight rewrites (`LinearOp`, convolution packing), whose weights are laid out for the CPU
  kernels. A model with float16 values of its own keeps its graph as written, so it runs the product, softmax, norm and elementwise operators
  separately. `GpuOptions.FuseOperators = false` turns the two device fusions off.
- **Bias and activation in the product.** `MatMul` followed by an `Add` of a constant bias vector (one value per output column) and then
  `Relu` or `Gelu` becomes one dispatch when nothing else reads the intermediate results: the product's kernel adds the float32 bias and applies
  the function to each element as it writes it. The float32 result has the bits it has with the operators run separately (the epilogue is
  the elementwise kernels' own add and function; the tests compare hashes), in fast and exact mode, so exact mode stays run-to-run
  identical. A float16 product folds in fast mode only, where skipping the intermediate rounding to half precision stays within the
  float16 tolerance; exact mode keeps the operators separate. A product that starts from a `Gemm` `C` is not folded. A BERT or MiniLM
  layer saves one dispatch per projection and two per feed-forward block. `GpuOptions.FuseEpilogues = false` turns it off.
- **Attention details.** The mask is the additive float mask of the node, broadcast over batch, heads and rows; a mask with one value per
  row (last dimension 1) is spread across the keys once into scratch. A query row with no visible key gives NaN, as the CPU's softmax does.
  The head dimension is at most 256. A product of more work than the device's submission budget (Metal) goes in one submission per batch entry.
- **Gather and indices.** An index outside its axis fails the run with `SessionErrorKind.InvalidInput`, "Gather index 40 is out of range
  for an extent of 40.", the CPU's error and message. Indices from the host are checked in safe code before the upload; indices the device
  made are recorded by the kernel in a status word read after the submission (the message then gives the position instead of the value).
  A constant index that is out of range keeps the operator on the CPU, which raises the error.

- **Weights larger than one buffer binding.** A vocabulary table or logits projection (51,865 by 768 float32 is 159 MB) is above the
  128 MiB some devices bind in one storage buffer. Such a constant is uploaded as several buffers cut along whole rows (a lookup, or a
  product with `transB`) or whole columns (a product, repacked dense), each within the limit. A `MatMul` or `Gemm` with such a weight runs
  one product per piece, each writing its columns of the result, so every output is summed in the order the single-buffer kernel uses and
  the bits equal the unsplit run's. A `Gather` copies each requested row from the piece that holds it, so it needs indices from the host or
  constants (an out-of-range index is the CPU's error, as above) and is limited to 65,536 rows per run; a product with a bias, indices
  made on the device and a result larger than one binding stay on the CPU with that reason. Float32 products and float32 or float16
  lookups only.
- **Long regions are cut into submissions.** Where the platform names no budget for one submission, a region's recording goes to the
  device after about 34 GFLOP of kernel work and continues in a new command buffer, so a slow or shared device (the integrated GPU of a
  desktop processor takes seconds for a speech encoder) never holds the queue for long enough for the driver to reset it. The cut falls
  between operators, keeps their order, and changes no bit of the result; on a discrete card 34 GFLOP is a few milliseconds.

## Speech models (encoder and decoder)

The reference speech model is two graphs, as exported: an encoder (log-mel input `[N, 80, 3000]`, two one-axis convolutions with Gelu,
a position add, twelve transformer blocks) and a decoder that takes the whole token sequence and the encoder states at every step and
returns the logits of every position (there is no key/value cache in this export, so each step recomputes the attention keys and values of
the whole prefix and the cross-attention projections of the 1500 encoder positions). Both graphs run completely on the device, one region
each, and pass `RequireGpu`. The sequence length of the decoder changes at each step, so each step builds a plan for its length (the
second run of a length reuses it, 16 plans are kept) and uploads the few constants the plan folds (about 13 buffers); the weights and the
arena stay. Per step the device receives the token ids and the encoder states and returns the logits.

Against the CPU's exact mode (scaled error `max|gpu - cpu| / max|cpu|` of every output, a synthetic log-mel of normal values with deviation
0.5): the decoder's logits are within 3.2e-6 at every step of a 64-step greedy run on every device, the greedy tokens equal the CPU's, and
the encoder's states are within 7.1e-5 to 9.2e-5 at batch 1 (fast and exact, RTX 4070, an AMD integrated GPU, an Apple M2, a software
rasteriser). At batch 4 the encoder's error is 1.6e-4 to 3.2e-4 (fast and exact, RTX 4070, an Apple M2, a software rasteriser; 1.1e-4 on both with the chain instead of the fused
attention). That is the model's float32 noise, not a fault of one kernel. `tools/GpuReport` with `--attribute` compares every executor with a float64
evaluation of the same graph (`Whisper64.cs`, tools only): per item the CPU's exact mode is 2.9e-5 to 1.4e-4 away from the true value (median 5.5e-5, 16 inputs) and the GPU
is 4e-5 to 2.8e-4 (mean 8.7e-5 fast, 9.4e-5 exact, against 6.4e-5 for the CPU). An item's error does not depend on the batch it sits in (item 0 has the same error at
batch 1 and batch 4), and fusing attention, Gelu and the bias epilogue moves the mean by less than the spread between items (7.9e-5 unfused, 8.7e-5 fused). So two
independent float32 evaluations differ by about 1e-4 per item, and the largest of four items sits above or below 1e-4 by the draw.

The size of the error comes from the sums. The CPU and the device add the terms of every product and convolution in one chain (increasing k, fused multiply-add) and
produce the same stage errors to the last digits (a stage from the same rounded input: 0.7e-6 to 1.3e-6 scaled, CPU and device alike). The graph amplifies early
rounding by a factor of 20 to 30: the two stem convolutions alone, with one chain each and every other operator exact, give 1.6e-5 to 7.4e-5 at the output (mean 3.8e-5). With every sum
evaluated in float32 and the rest in float64, per item over eight inputs: one chain 7e-5 on average (2.6e-5 to 1.8e-4), eight interleaved chains added pairwise 1.0e-5
(5.5e-6 to 1.6e-5), compensated sums 4.4e-6. The 1e-4 gate against the CPU's exact mode cannot hold for every input of this model while the CPU keeps one chain;
the device would be closer to the true value than the CPU with eight lanes in its matrix product and convolution kernels. The greedy tokens of a 16-step decode with the device's encoder and decoder equal the CPU's
at batch 1 and 4 on the RTX 4070 and the M2, and at batch 1 on the software rasteriser.

With float32 sums taken in blocks (32 terms, then groups of 128, joined in increasing k; the matrix kernels' f16 and bf16 paths keep one chain), the same 16 inputs at batch 4 give a mean
error against float64 of 1.02e-4 (fast) and 1.08e-4 (exact) on the RTX 4070 and 1.15e-4 and 1.09e-4 on the M2, against 8.9e-5 for the CPU's exact mode: 1.14 and 1.21 times the CPU's on the RTX 4070
(before the blocks 1.45 and 1.57), 1.28 and 1.22 on the M2 (1.47 and 1.60). At batch 1 (16 inputs) the means are 7.1e-5 fast and 6.9e-5 exact on the RTX 4070 (CPU exact 7.0e-5), 8.7e-5 and 8.0e-5 on the M2.
Per item the device is below the CPU's exact mode on 6 to 8 of 16 inputs at batch 4 on the RTX 4070 and on 4 of 16 on the M2, and on 4 to 7 of 16 at batch 1. The 1e-4 gate against the CPU's exact mode still
fails on that model for the inputs where the CPU itself is far from the float64 value (up to 3.9e-4 at batch 1 on the RTX 4070).

Exact mode is bit-identical from run to run on each device.

The dynamically quantized speech encoder runs as one region too, its two stem convolutions (`ConvInteger`, int8 weights, per-channel zero points) on the device. The first stem's output
is bit-identical to the CPU's, as every integer sum is (the tests compare `ConvInteger` bit for bit on one and two spatial axes, groups, dilation, strides, padding, per-channel zero
points and every mix of uint8 and int8). Further in, the float operators between the quantization steps differ in the last bits (the CPU's exact mode uses its own polynomials for
`Erf` and `Exp`), and a quantization step turns such a difference into a one-step difference that spreads: after the second stem the largest difference is 2e-3 on values up to 6.1
(rms 1.4e-5), and at the output the scaled error against the CPU's exact mode is 0.34 to 1.0 on the devices tried, against 0.45 between the CPU's own fast and exact modes on the same input.
Two runs on one device are bit-identical in both modes.

## Fused or chained attention

The fused attention kernel is the default everywhere, browsers included. `GpuOptions.FusedAttention = false` runs the chain it replaces instead (two
products, the mask addition and a softmax; both forms are deterministic in exact mode). Apple M2, fast mode, fused against chain: the speech
encoder (1500 keys) 355 ms against 428 ms at batch 1 and 1.32 s against 1.62 s at batch 4; on the RTX 4070 the speech encoder at batch 4 is 210 ms against 261 ms.
The error against the CPU's exact mode is about 1e-5 for either form on the transformer encoders; the speech encoder at batch 4 shows 1.8e-4 (fused) and
1.1e-4 (chain) against a floor of 5e-5 to 7e-5 that the CPU's own exact mode shows when its input moves by one unit in the last place.

## Speed on Apple silicon

Indicative, Apple M2 (4 performance and 4 efficiency cores), fast mode, median of 20 runs, wall time of the device run against the CPU's fast mode (4 threads, median
of 10), in milliseconds. Batch 1: MiniLM 5.5 against 24.1, BERT-base 23.4 against 69.8, ResNet-50 17.7 against 38.4 (best CPU run 25.6), MobileNetV3 5.3 against 13.8,
DBNet 5.2 against 7.8, the recognizer 2.1 against 29.4 (best CPU run 3.8), the speech encoder 352 against 1395 (8-bit: 705 against 898), BERT-base 8-bit 60 against 70.
Batch 8: MiniLM 23.3 against 78.8, BERT-base 140 against 552, ResNet-50 97 against 198, MobileNetV3 20.2 against 22.9.

Consecutive passes of a run share one Metal compute encoder, so a small model's cost is its kernels plus a couple of microseconds per dependent pass. Metal samples its
counters only at encoder boundaries, so timing a pass splits the encoder around it and costs about 20 microseconds. With `CollectTimings` on, the device time of a run is
therefore taken as one span, from the start of the first operators' passes to the end of the last ones, and the passes between them run untimed: the timing no longer
slows the run it measures (same build, timing as one span against timing every pass, batch 1: MiniLM 6.4 against 7.5 ms, MobileNetV3 5.4 against 7.3, the recognizer 2.1
against 3.6, ResNet-50 17.6 against 20.0). `GpuSessionInfo.Timings.Dispatches` and `ComputePasses` are counted by the command encoder as it records, so they are the same on every backend with or without
timing; `GpuReport --profile` times each step on its own and prints GPU time and dispatches by operator (BERT-base 8-bit: 328 passes, the speech encoder 8-bit: 357).

## Host cost of a run

With `CollectTimings` a run's host time is split into recording (binding operands and encoding the passes), the submit calls, and the rest (waiting for
the device and copying the outputs back): `GpuTimings.RecordMilliseconds`, `SubmitMilliseconds`, and `SubmitAndReadbackMilliseconds` minus both.
Recording is small (RTX 4070 Vulkan, batch 1: BERT-base 0.4 ms for 146 dispatches, MiniLM 0.2 ms for 80, the speech decoder 0.6 ms for 256) and the
bind groups are all cached; the cost that showed up in wall time was that the device sat idle until the whole region was recorded and handed over.
A region of more than a few operators is therefore handed to the device in pieces while it is still being recorded: the first piece after 6
dispatches, each later piece twice as large. Barriers between pieces are tracked per buffer, so bits are unchanged (a test compares them, and the
counts). Wall time against the summed kernel time, same device: BERT-base 9.80 -> 9.31 ms (kernels 8.85 / 8.61), MiniLM 2.83 -> 2.65 (2.20 / 2.21),
speech decoder 19.94 -> 19.55 (18.15 / 18.11); in Chrome on the same GPU 26.0 -> 24.1, 12.0 -> 10.0, 42.9 -> 41.8 ms. On Metal it loses: every commit
costs enough that the 24 ms BERT run becomes 27 ms, so Metal hands a region over whole; there the run is already within 0.5 to 1.3 ms of its kernel time
(BERT 24.65 against 23.90, MiniLM 5.64 against 5.16, speech decoder 87.25 against 85.92). What remains on Vulkan, 0.5 to 1.5 ms, is about 5 microseconds
per dispatch, the size of the gap the device leaves between dependent passes; recording is hidden behind the kernels now.

## The contracts later extensions rely on

- **Op table** (`GpuOpTable`). One `GpuOpEntry` per ONNX operator type: `Check` runs at session creation with attributes only, `Bind` runs
  per plan with element types, shapes and constants known, and returns a `GpuBinding` or the reason in words. Adding a kernel family is
  adding entries; nothing else changes. A binding holds everything its kernel call needs (shapes, constants on the device, the kernel
  mode) and does no allocation when it encodes.
- **Partition.** The core offers every statically shaped step to `ClaimStep` in schedule order, then forms regions from the accepted
  steps (see above). `Compile` gets the regions once, after the schedule is final.
- **Arena.** `GpuProgram.Create` turns each region into lifetimes: a value lives from the step that makes it (or the start of the region,
  for a region input or an output a kernel starts from) to its last reader (or the end, for a region output). A view's lifetime extends its
  source's. A binding may also ask for float32 scratch for the length of its step (the float16 product accumulates in float32 and
  rounds once; so do the convolutions, the recurrent layers and the requantizing product). Offsets are chosen by first fit per element type (float32, float16, int32, int64, uint8); slabs only grow.
- **Folding.** After the bindings of a region exist, `GpuProgram` looks for a product whose sole reader is a bias `Add` and then an
  activation, hands them to the product's binding and replaces the absorbed steps with ones that record nothing and read and write no value.
  The product then writes the last result's value, and lifetimes are planned from the changed inputs and outputs.

## `RequireGpu`

Creating the session fails with `SessionErrorKind.DeviceUnsupported`, `SessionException.Operators` and the message naming every
operator that cannot run on the device and why. An operator whose inputs are all known before the run (weights, shape arithmetic) might be
evaluated once while the plan is made and never run; it is not named at creation, and the plan checks it again, with the folding done, for
the shapes it is built for. So `RequireGpu` can also fail the first run or `GetDiagnostics` of a shape, naming what is left. A reason that
depends on the shapes of the first run (an operand larger than the device's binding limit) fails that run the same way. `RequireGpu` needs
`Device = Gpu`. If no device exists at all, creation fails with `DeviceUnsupported` whatever `RequireGpu` says.

## Fast and exact mode

- **Fast** uses the fastest kernels, including matrix hardware where the device has it. The result is within the fast-mode tolerance
  of the CPU (scaled error `max|gpu - cpu| / max|cpu|`, about 1e-6 to 1e-5 for the float32 encoders; float16 models round to 2e-3).
- **Exact** uses kernels with a fixed summation order, no atomics and no scheduling dependence: two runs on the same device and
  driver give the same bits. Products, softmax, normalisation and reductions have deterministic variants. Elementwise operators and
  layout operators have no reduction order, so one kernel serves both modes and is bit-reproducible on a device. That includes the
  transcendental functions (`Erf`, `Tanh`, `Exp`, ...): they are reproducible on one GPU and driver, but they are not the CPU exact
  mode's fixed polynomials, so a result matches the CPU's exact mode only within the float32 tolerance (1e-5 per operator,
  1e-4 for a whole model), and another GPU, driver or adapter may differ in the last bits. An operator with no deterministic kernel stays on
  the CPU.
- **Float16** values stay float16 on the device. Products accumulate in float32 and round once; the elementwise, normalisation and
  softmax kernels compute in float32 and round once per operator, like the CPU's float16 path. A float16 softmax or layer
  normalisation needs an even row length (the kernels write rows as 32-bit words), otherwise it stays on the CPU.

## Device memory

The scratch a kernel needs while it runs lives in the arena for the length of its step, so it costs memory only then and is reused by
the steps after it: the column matrix and accumulators of the convolutions that unfold their input (none on Apple silicon in float32, where the
convolution gathers windows while it multiplies), the transposed convolution's columns, the recurrent layers' projection and state, the sums
of a requantizing 8-bit product. A convolution gets the whole batch's worth up to 64 MiB, else as many whole images as fit (at least
one); a smaller buffer runs the batch in chunks with the same bits.

One arena per session holds the intermediate tensors of every plan. Offsets are planned from tensor lifetimes (a tensor's memory is
reused once the last operator that reads it has run), per element type, in slabs no bigger than one buffer binding
on the device. The arena only grows, and only when a plan needs more than any before it, so the second run of a shape, and a run of any
smaller one, allocates nothing on the device. `OneModelGpu.GetInfo` reports the arena's size, the weights' size and the number of device
allocations made so far.

Every kernel family (matrix products, elementwise, norm, softmax, reduce, transpose, integer, gather, attention, the 8-bit product, dynamic
quantization and dequantization) keeps its bind groups (the device's descriptor sets) by the buffers they name, up to 4096 sets, which is more than the
weights of the largest encoders touch, so a warm run builds none. When the arena grows and replaces its buffers the families drop their
cached groups with `ReleaseBindings`, so the old buffers are released; the families themselves, with their pipelines, stay, and the next run
builds each set once more. The dequantizing 8-bit product applies its scale and biases as it stores each sum, so it is one dispatch and keeps nothing between passes; the requantizing one takes its int32 sums from the arena.

## In a browser (WebGPU)

The package runs on the `browser-wasm` runtime through the WebGPU backend of `RecursiveAI.OneGpu`. A browser has one thread and cannot wait
for the GPU, so two calls differ from the desktop ones:

```csharp
OneModelGpu.Register();
await OneModelGpu.PrepareDeviceAsync();                    // before each GPU session: the browser makes its device asynchronously
using var session = Session.Load(modelBytes, new ModelOptions { Device = ComputeDevice.Gpu });
using SessionOutputs outputs = await OneModelGpu.RunAsync(session, inputs);   // same results as Run, without blocking
```

`OneModelGpu.RunAsync` works on every platform and gives the bits of `Session.Run`. In a browser a blocking `Run` that reaches a device region
throws a `SessionException` (`DeviceFailure`) saying the browser cannot wait for the GPU; a model that runs entirely on the CPU can still use `Run`.
Awaited runs of one session take turns. Creating a GPU session without `PrepareDeviceAsync` first throws `DeviceUnsupported`. A session owns the
device made for it, so prepare one per session; sessions can live side by side in one page, each on its own device (a model's encoder and decoder, or a
fallback session made after `RequireGpu` refused the first).

What the browser changes, all invisible in the results:

- **Buffers.** A browser refuses one buffer that a dispatch binds both for reading and for writing, even at disjoint ranges. The arena
  therefore keeps an operator's inputs, outputs and scratch in different slabs (more slabs and more memory: 9.0 MiB against 5.0 MiB for MiniLM at batch 1), and the boolean
  gather uses a second scratch for its gathered words. The same layout can be tried on any device with `ONEMODEL_GPU_SEPARATE_SLABS=1`; it
  changes no bits.
- **Attention.** `GpuOptions.FusedAttention` null means the fused kernel in a browser too. In Chrome on an RTX 4070 (warm run, fused against the chain): the speech
  encoder 68 against 81 ms, BERT-base at batch 1 19 against 21 ms and at batch 8 39 against 41 ms, MiniLM at batch 1 8.2 against 8.5 ms; on SwiftShader (a software adapter)
  MiniLM 576 against 888 ms and BERT-base 3.8 against 4.1 s. Set it to false to run the model's own operators; results are within the same tolerance.
- **Float16** models run without the `shader-f16` feature: the kernels read and write 16-bit values packed in 32-bit words.
- **Subgroups and matrix hardware** are not used; the portable variants run.
- **Memory.** Model bytes and the prepared weights both live in the runtime's linear memory (4 GiB at most). The float32 speech decoder
  (774 MB file) loads and runs; a model of 1.1 GB or more has not been tried. Weights above one buffer binding are split as on the desktop.
- **Operators that stay on the CPU** run in WebAssembly, which is much slower than the desktop CPU path. The reference set has none left: the speech encoder's 8-bit
  convolutions (`ConvInteger`) run on the device since they were mapped (1.1 s a run in Chrome before, 45 ms after).

`tools/gpu_browser.sh` runs reference-set models in headless Chrome and prints parity against the desktop CPU exact mode, placement,
`RequireGpu` outcome, repeatability and timing next to the desktop GPU run and onnxruntime-web's WebGPU provider.

## Errors

Device failures are `SessionException`s that name what was being done: `DeviceFailure` for out of memory (with the size asked for),
a lost device and a missing capability; `Unsupported` for a shape a kernel rejects. After a lost device, create a new session.

## Concurrency

Concurrent runs of one session overlap on the CPU but take turns on the GPU (one device per session, guarded by a lock).
