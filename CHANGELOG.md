# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [1.2.0] - 2026-11-08

Everything since 1.0.0, the previous nuget.org release: the 1.1.0 speed work, trainable imported graphs and the other training additions of 1.2.0-preview.1, and the loading fixes of 1.2.0-preview.2 (details in those sections below). No breaking change; inference and training results are unchanged bit for bit except where 1.1.0 documents fast-mode convolution differences within tolerance.

### Added

- Training: `OnnxModule` trains an imported ONNX (or OneModel) graph as an `NnModule` and writes the weights back to a `Session`; per-parameter freezing (`NnModule.Freeze`, `Tensor.SetRequiresGrad`); row-sparse embedding gradients with `SparseAdam` and sparse `Sgd`; `Ops.Conv1d` and the `Conv1d` module; `Ops.MatMulBatched`, `Erf`, `Exp`, `Sqrt`, `Pow`, `Where` and `Gather` with backward.
- Loading from caller-owned memory without a copy: `ReadOnlyMemory<byte>` overloads of `Model.Load`, `Session.Load`, `OnnxImporter.Load` and `ModelFormat.Load`, and in-place loading of an `UnmanagedMemoryStream` (an embedded resource) through the stream overloads.

### Fixed

- Checkpoints above 2 GiB: `Checkpoint.Save` and `Load` stream with a bounded buffer instead of building the file in memory. Loading needs a seekable stream; the file format is unchanged.
- Models above 2 GiB load from ONNX with external data and from OneModel files on the memory-mapped path; a single-file ONNX above 2 GiB (a protobuf limit) and unmapped loads above 2 GiB fail with a typed error naming the supported path.
- `If` allocates nothing per call once warm, with caller-owned inputs and outputs.

### Changed

- Speed: recurrent layers threaded in fixed hidden-unit chunks, Winograd F(2x2, 3x3) for prepacked 3x3 convolutions in fast mode, batched convolution waves, and lower per-call overhead on small models (1.1.0); Adam and AdamW update in fixed-size chunks on the thread pool, and gradient clipping threads its scaling pass.

## [1.2.0-preview.2] - 2026-11-08

Loading and training fixes for large files, in-place loading, an allocation-free `If`, and threaded Adam/AdamW; no breaking change, inference and training results unchanged bit for bit.

### Added

- `Model.Load`, `Session.Load`, `OnnxImporter.Load` and `ModelFormat.Load` overloads for caller-owned `ReadOnlyMemory<byte>`, importing raw initializers as views without a serialized-file copy. Pointer-exposing `UnmanagedMemoryStream` inputs (including embedded assembly resources) also load in place through the stream overloads. Caller storage must remain valid and unchanged while the model or session lives. Session weight preparation is unchanged.

### Fixed

- `If` no longer allocates argument arrays or an output-disposal enumerator on each call. Warm runs with caller-owned contiguous tensor inputs/outputs allocate no managed memory when the branches are allocation-free. Concurrent calls lease independent arrays; `Loop`, `Scan` and `SequenceMap` also reuse body argument arrays, though their other iteration bookkeeping still allocates.
- Mapped OneModel containers can exceed 2 GiB; ONNX external-data offsets remain 64-bit. Unsupported whole-stream and unmapped paths fail with typed errors suggesting the supported alternatives. A single ONNX protobuf still must be smaller than 2 GiB.
- Checkpoint loading no longer buffers the whole file. Saving and loading use bounded payload buffers and 64-bit file offsets, including checkpoints above 2 GiB. Loading requires a seekable stream; non-seekable inputs receive a typed error. The checkpoint bytes are unchanged.

### Changed

- Dense Adam/AdamW and `GradClip.ClipNorm`'s scaling pass use fixed-size chunks on the shared thread pool, limited by `TensorCompute.Threads`. The global norm's running double sum stays serial to preserve rounding. Parameters, gradients, moments and recorded training hashes are unchanged bit for bit at every thread count.

## [1.2.0-preview.1] - 2026-11-05

Training additions; no breaking change, inference and existing training results unchanged bit for bit.

### Added

- `Ops.Conv1d(x, weight, bias, Conv1dParams)` and the `Conv1d` module: 1-D convolution on [N, C, L] input with stride, left/right padding, dilation and groups (depthwise included), backward for input, weight and bias, `padding="same"`/`"valid"` on the module, and PyTorch's default initialisation. It runs on the 2-D convolution kernels, so fast results do not depend on the thread count and exact results are bit-identical on every ISA path. A traced `Conv1d` becomes an ONNX `Conv` with 1-D attributes.
- Training package: `OnnxModule` loads an ONNX file (or a OneModel file) as a trainable `NnModule`. Its float32 initializers are parameters named as in the file, in file order, so state dicts, checkpoints, optimizers and freezing apply; its `Forward` runs the graph on the eager tensor operators with gradients (attention, rotary embeddings, local-window masks, GeGLU, bias-free layer norm and the other operators of BERT-class exports), while shape and mask arithmetic runs on the graph runtime's own kernels. `CreateSession`, `Save` and `ToModel` write the trained weights back so the model is served by the ordinary `Session`. A graph that needs an operator without a gradient fails at load, naming all of them (`docs/training.md` section 11, `docs/guide/training.md`).
- Tensor layer: `Ops.MatMulBatched` (batched product with NumPy broadcasting), `Ops.Erf`, `Ops.Exp`, `Ops.Sqrt`, `Ops.Pow` (constant exponent), `Ops.Where` and `Ops.Gather`, each with a backward record (`docs/autograd.md`).
- `tools/export_finetune_models.py` and `tools/finetune_reference.py` export xlm-roberta-base and mmBERT-base (pinned revisions in `tools/models.json`) and produce PyTorch reference gradients and loss curves; `tools/finetune_oracle.py` generates the small committed graphs in `testdata/finetune`.
- Freezing parameters (training): `NnModule.Freeze` / `Unfreeze` (all, by parameter name or module path, or by predicate), `IsFrozen`, `TrainableParameters`, `NamedTrainableParameters`, and `Tensor.SetRequiresGrad` (PyTorch's `requires_grad_`). A frozen parameter gets no gradient, keeps no optimizer state when the optimizer is built while it is frozen, is skipped by every optimizer and by `GradClip`, and stays in state dicts and checkpoints. Optimizers now accept frozen parameters in their list. Checkpoints adopt the state shapes of the file, so they load whichever way the target optimizer was built; existing checkpoints load unchanged.
- Row-sparse embedding gradients: `Embedding(count, width, generator, paddingIndex, sparse)` and `Ops.Embedding(table, ids, paddingIndex, sparse)` leave a `RowSparseGradient` (`Tensor.SparseGrad`: touched rows ascending, duplicates summed in position order, bit-equal to the dense gradient's rows, independent of the thread count) instead of a dense table gradient. `Sgd` (without momentum and weight decay) updates only those rows; the new `SparseAdam` matches `torch.optim.SparseAdam` (lazy moments). `GradClip` and `ZeroGrad` handle sparse gradients; `Adam` and `AdamW` reject them with a message naming the fix. There is no sparse AdamW (see `docs/training.md` section 3.1).
- `bench/Micro` command `sparseembed`: gradient bytes, optimizer state, step time and peak memory of a 250 002 x 768 embedding fine-tuned dense, sparse and frozen.

## [1.1.0] - 2026-11-04

Speed only: no API change, fast-mode results within the documented tolerance, exact mode unchanged bit for bit. Re-measured rows are in `docs/performance.md`.

### Changed

- Speed of recurrent layers (RNN, GRU, LSTM): each time step is split into fixed chunks of 32 hidden units that run on the thread pool, and bidirectional layers step both directions in one pass. Chunk boundaries depend only on the hidden size, so results do not depend on the thread count. A CRNN text recogniser at batch 8 on 32 x64 threads now takes 1.4 times ONNX Runtime's time, from 4.5 times in 1.0.
- Speed of convolution networks (fast mode): 3x3 stride-1 convolutions with 32 to 256 input and output channels (up to 65 536 channel pairs) (ResNet-50's 64-, 128- and 256-channel stages, the CRNN text recogniser's front end) use Winograd's F(2x2, 3x3) transform when the weights are prepacked at load: 16 instead of 36 multiplications per output pair, on the same register-tile microkernels. Their results differ from the direct convolution's in the last bits (within the documented tolerance) and do not depend on the thread count; exact mode is untouched. A convolution that takes several row tiles and runs more than one image now gathers and multiplies the images in waves, so a batch of 8 keeps all cores busy in two parallel phases per layer instead of two per image.
- Speed of small models (DBNet-class detectors, batch 1): nearest-neighbour `Resize` runs in one threaded pass (about 6x faster on a 2x upsample); the thread pool wakes only the helpers a call uses; depthwise convolution builds its padded copy without clearing the interior, shares it between the output channels of one input channel, fuses bias and Relu in one pass, and computes wide rows' leftover vectors four rows at a time; the Conv epilogue (bias, Relu) runs in one pass per row. Fast-mode results are unchanged bit for bit; exact mode is untouched.

## [1.0.0] - 2026-10-31

The first stable release. The public API is reshaped once, below, and follows Semantic Versioning from here. Speed and peak memory against ONNX Runtime on x64 and arm64, with the rows where it is still slower, are in `docs/performance.md`. Of the 1905 `onnx/backend/test` cases, 1580 pass in fast mode and 1382 in exact mode, none fail, and the rest are listed gaps (`docs/conformance.md`). Exact-mode output hashes of the reference models and every node case are equal across x64 and arm64 JIT (default, one thread, narrower instruction sets, hardware intrinsics off), NativeAOT and browser WebAssembly (interpreted and compiled).

### Breaking changes

The public API is reshaped for 1.0. Every change below lists its replacement.

- **Tensor ownership (`Tensor`).** `Dispose` is idempotent: disposing a tensor again, on any copy, on a default tensor, or after a method that disposed it, does nothing instead of throwing `StaleTensorException`, and a stale handle can never release memory that was re-rented to someone else. Reading data through a released handle still throws `StaleTensorException`, which now derives from `ObjectDisposedException` (was `InvalidOperationException`). New: `Tensor.IsReleased`. `Session.Run` treats an already released output tensor as an empty slot. The ownership rule is in `docs/guide/tensors.md`.
- **Named runs and results.** New `Session.Run(IReadOnlyDictionary<string, Tensor> inputs, IReadOnlyList<string>? outputNames = null)` returns `SessionOutputs`, a disposable container that owns the output tensors (by position or name). The span overloads stay as the allocation-free path.
- **One exception base.** `SessionException`, `OnnxImportException`, `ModelFormatException`, `TraceException` and (training package) `CheckpointException` derive from the new `OneModelException` (`Category`, `InputName`, `Offset`, `FieldPath`). `TraceException` no longer derives from `InvalidOperationException`. Long field paths are shortened in the message; the property keeps the full path.
- **File-system errors are the BCL's.** The `Io` kinds are gone (`OnnxErrorKind.Io`, `ModelFormatErrorKind.Io`): a missing or unreadable file or a failing stream throws `FileNotFoundException`, `DirectoryNotFoundException`, `UnauthorizedAccessException` or `IOException` unwrapped, from `Model.Load`, `Session.Load`, `OnnxImporter.Load`, `ModelFormat.Load/Save/Convert` alike (as `Checkpoint.Load` already did). Replace `catch (OnnxImportException e) when (e.Kind == OnnxErrorKind.Io)` with `catch (IOException)`.
- **Use after dispose** throws `ObjectDisposedException` from `Session`, `Model` and `TracedModel`. `SessionErrorKind.Disposed` is removed.
- **Load options reach every entry point.** `OnnxLoadOptions` (namespace `Onnx`) is now `ModelLoadOptions` (namespace `RecursiveAI.OneModel`) and gains `MaxModelBytes` (default and maximum 2^31 − 1). `Model.Load`, `Session.Load`, `ModelFormat.Load` and `Convert` take it (`Model.Load(path, options, loadOptions, cancellationToken)`; a positional `CancellationToken` third argument must become the fourth). `MaxNesting` is validated when set (8 to 250). A stream is read only up to the limit: an endless or huge stream fails with an `Unsupported` import or format error, never `OutOfMemoryException`.
- **Names that collide with ONNX Runtime.** `SessionOptions` is now `ModelOptions` (namespace `RecursiveAI.OneModel.Graph`) and `ExecutionMode` is now `ArithmeticMode` (namespace `RecursiveAI.OneModel`); the values are unchanged (`Fast`, `Exact`).
- **Smaller public surface.** Internal now: `BackwardRecord`, `AutogradMeta`, `SavedTensor`, `OpKind`, `Tensor.Autograd`, `GradMode.ShouldRecord`, `TensorBuffer.Address`, `BufferPool.ClassSize`, `OneModelInfo`, and the kernel plumbing `ITensorKernels`, `KernelSet`, `DenseKernels`, `ReferenceKernels`, `UnaryOp`, `BinaryOp` (the `kernels` argument of `Ops.Add/Sub/Mul/Div/Neg/Relu/MatMul/MaxPool2d` is gone; the `Ops` functions take the default kernels). `PackedValue` and `PackedKind` are marked `[Experimental("ONEMODEL0001")]`: sequences and optionals as packed values may change.
- **`ModelFormat.Convert`** refuses an output that is its input (`ArgumentException`) and now also accepts a file already in the OneModel format (it is read and rewritten). Parameter names are `inputPath` and `outputPath`.
- Eager `Ops` on an empty tensor throw `ArgumentException` saying so, instead of the float32-only message.

### Added

- A usage guide, `docs/guide`: getting started, tensors and ownership, errors, the format, training and checkpoints, tracing, and migration notes from Microsoft.ML.OnnxRuntime and TorchSharp. Its code is built and run in CI (`tests/GuideSamples`, `tools/sync_guide.py`).
- Operators on `Tensor` (`+ - * /`, unary `-`, with `float` scalars) routed to the differentiable `Ops`; `Tensor.Item<T>()`; `Tensor.Zeros(dtype, dims)` and `Tensor.FromSpan(data, dims)` with the dimensions as `params`.
- `ActivationScope.Keep(tensor)` moves a result out of a scope. `TensorCompute.Use(mode, threads)` scopes the eager compute settings to a flow of execution.
- `ModelFormat.Detect(path | bytes)` returns `ModelFileKind` (`Onnx`, `OneModel`, `Unknown`).
- `Model.Run(input, shape, out Shape outputShape)` reports the output's shape.
- `TensorInfo.ToString()` prints `TensorInfo { x: F32[N, 16] }`.

### Changed

- Faster fp32 convolution on AVX-512: 16x16 and 8x32 tiles chosen by the output channel count, 1x1 unit-stride convolutions read their input in place without a gather phase, stride-2 patches are gathered with a shuffle, and depthwise layers on narrow maps run as one flat row (`docs/kernels-conv.md`, `docs/kernels-depthwise.md`). Exact mode is unchanged.
- `docs/performance.md`: the 1.0 measurements against ONNX Runtime.
- Fast-mode erf `Gelu` (the operator, the fused erf chain and the matrix-product epilogue) uses a degree 5 over degree 5 rational function instead of the accurate exponential form: 3.5 to 5 times faster, absolute error below 1e-6 (`docs/exact-transcendentals.md`). Exact mode and training are unchanged. int8 and fp32 transformers gain 5 to 20 percent end to end.

## [0.5.0-preview.1] - 2026-10-31

Training arrives as a second package, `RecursiveAI.OneModel.Training`: reverse-mode automatic differentiation on the tensor layer, modules, losses, SGD/Adam/AdamW with schedules, deterministic data loading, bit-exact checkpoints, and tracing a model into a graph that runs and saves like an imported one. Inference got faster across convolution, int8 and transformer graphs, especially at higher thread counts (`docs/kernels-conv.md`, `docs/int8.md`, `docs/kernels-gemm.md`). The README and package readme describe what works today.

### Added

- `Ops.Lstm`, `Ops.Gru` and `Recurrent.Forward` take per-row sequence lengths (packed-sequence semantics: the reverse direction starts at each row's last valid step, padded outputs are zero), and `WarmupCosineLr` (linear warm-up, then a cosine to a floor). `tools/CrnnFinetune` fine-tunes a CRNN line recognizer, scores it and writes the weights as a model container; `tools/crnn_ref` runs and compares a PyTorch reference (docs/training.md section 10),
- int8 speed: vector packing of the B operand and of convolution patches (`Int8Pack`), patches packed on demand by the thread that multiplies them when several threads work, a one-pass epilogue for symmetric weights, a quantized addition and a byte max pooling that the QDQ pass uses in place of `DequantizeLinear, Add, QuantizeLinear` and `DequantizeLinear, MaxPool, QuantizeLinear`, and faster int8 depthwise rows (`docs/int8.md`).
- `bench/Micro` command `mixed`: step time, peak memory, activation memory and bfloat16 conversion cost of three training steps; the findings that reduced-precision training does not pay on the current kernels, and the one case that would for memory, are in `docs/training.md` section 9.
- Reverse-mode automatic differentiation on the tensor layer (`docs/autograd.md`): `Tensor.Backward`, leaf gradients, `Backprop.ZeroGrad`, `retainGraph`, saved-tensor version checks and `ActivationScope`, which frees a step's intermediates in one call.
- Differentiable operators in `Ops`: `Gelu`, `Sigmoid`, `Tanh`, `Sum`, `Mean`, `Softmax`, `LogSoftmax`, `LayerNorm`, `BatchNorm2d`, `Conv2d` (groups, stride, padding, dilation, depthwise), `AvgPool2d`, `GlobalAvgPool2d`, `Concat`, `Embedding`, `NllLoss`, `CrossEntropy`, `Linear` and `Attention`, with backward for these and for the existing arithmetic, `MatMul`, `MaxPool2d` and view operators.
- `TensorCompute` selects fast or exact arithmetic and the thread count for the differentiable operators. Exact-mode gradients are bit-identical across ISA paths and thread counts.
- The `RecursiveAI.OneModel.Training` package, starting with `GradientCheck` (analytic gradients against finite differences).
- `tools/autograd_oracle.py` and `testdata/autograd/`: PyTorch reference gradients for a conv net and a two-layer transformer encoder.
- Training API in `RecursiveAI.OneModel.Training` (`docs/training.md`): `NnModule` with parameters, buffers, train/eval mode and a state dict; `Linear`, `Conv2d`, `BatchNorm2d`, `LayerNorm`, `Embedding`, a seeded `Dropout`, `Sequential`, `MultiHeadAttention`, `TransformerEncoderLayer` and activation/pooling adapters; losses (`Losses.CrossEntropy` with label smoothing and ignore index, `Mse`, `L1`, `BceWithLogits`); `Sgd` (momentum, Nesterov, weight decay), `Adam`, `AdamW`; `GradClip.ClipNorm` / `ClipValue`; learning-rate schedules (`StepLr`, `CosineLr`, `WarmupLr`, `OneCycleLr`).
- `Checkpoint.Save` / `Load`: model and optimizer state in a documented 64-byte-aligned file (`ONEMCKPT`). Saving then loading restores every bit, and a run resumed from a checkpoint equals the uninterrupted run bit for bit. Bad files fail with `CheckpointException`.
- `tools/training_oracle.py` and `testdata/training/`: PyTorch reference training runs (MLP with SGD+momentum, conv net with AdamW + cosine + clipping, two-layer transformer classifier with Adam + warm-up), optimizer step, clipping and schedule references. Losses agree with PyTorch's to 4e-6 relative over 50 to 100 steps.
- `ActivationScope.Track` is public, so modules can hand masks and constants to a step's scope.
- The Training package is built and packed by CI and the release workflow beside the runtime; fuzz target `checkpoint`.
- Differentiable recurrent layers (`Ops.LstmLayer`, `GruLayer`, and multi-layer, bidirectional, batch-first `Ops.Lstm` / `Ops.Gru` over `RnnLayer` weights in PyTorch's gate layout), `Ops.CtcLoss` (log-space forward-backward, reductions none/mean/sum, zero-infinity), `Ops.ConvTranspose2d` and `Ops.Resize2d` (nearest, bilinear, align corners), with backward through time (`docs/autograd.md` section 8).
- `tools/rnn_oracle.py` and `testdata/autograd/{crnn,gru,decoder}.*`: PyTorch gradients for a CRNN with CTC loss, a bidirectional GRU and a transposed-convolution/resize decoder, matched to 1e-6 scaled.
- Data loading in `RecursiveAI.OneModel.Training.Data` (`docs/data-loading.md`): `IDataset<T>` with list, concatenated, mapped and fixed-record file datasets; sequential, seeded-shuffle, weighted-mixing and width-bucketed samplers; `ImageBatchCollator` (padding to a width multiple, widths, CTC-style concatenated targets, seeded in-place augmentation into pooled tensors); `ImageTensors` for 8-bit gray and RGB pixels; and a prefetching `DataLoader` whose batch at step g depends only on (seed, g), whatever the worker count or prefetch depth, and resumes exactly at any step.
- Tracing (`docs/tracing.md`): `Tracer` records a model written on the eager `Ops` (or, in the training package, an `NnModule` through `ModuleTracing.Trace`) into the graph layer in one run on example inputs, with optional symbolic batch/sequence axes. The `TracedModel` runs through `Session` with the usual passes, saves in the library format (and loads back bit-identically) and writes ONNX, which ONNX Runtime loads and agrees with. Reading or changing a traced tensor, `GradMode.NoGrad()` and operations without a graph equivalent fail with `TraceException`.
- `tools/trace_oracle.py`: runs traced models' ONNX files in ONNX Runtime against the eager outputs.

### Changed

- The eager element-wise operators and the gradients of Relu, Sigmoid, Tanh and GELU run on fixed-width vector lanes (about 2x faster encoder training steps in a first look); results equal the scalar loops bit for bit, except that fast-mode sigmoid and tanh now use the fast polynomials instead of the BCL (`docs/autograd.md` section 7).
- `Ops.MatMul` (default kernels, `DenseKernels`) runs on the blocked GEMM instead of the double-accumulating reference loop; pass `ReferenceKernels.Instance` for the old behaviour.
- Faster convolution in fast mode: regular and grouped convolutions gather their input patches straight into the packed layout (no im2col matrix, no separate packing phase), their weights are packed once per session (`conv-pack` pass), tiles are balanced, and small layers use more threads (`docs/kernels-conv.md`). A single-row matrix product uses a matrix-vector kernel; large element-wise outputs and float `MaxPool` run in parallel chunks. Fast-mode results are independent of the thread count as before; exact mode is unchanged.
- A convolution followed by an `Add` of a skip connection adds it on the finished tile (graph pass `conv-residual`, fast mode); ResNet-50 at 8 threads about 10% faster.

## [0.3.0-preview.1] - 2026-10-30

Our own model file, ONNX backend-test conformance with control flow, and exact-mode results checked bit for bit across platforms. Of the 1905 `onnx/backend/test` cases, 1580 pass in fast mode and 1382 in exact mode, none fail, and the rest are listed gaps (`docs/conformance.md`). Exact-mode output hashes are equal on Windows, Linux and macOS (x64 and arm64, JIT and NativeAOT) and on browser WebAssembly (`docs/golden-hashes.md`).

### Added

- Exact-mode golden hashes: `tools/GoldenModels`, `tools/golden*.sh`, and a CI check on every platform against `testdata/graphs/golden-exact.txt`.
- The OneModel model file (`ModelFormat`, `docs/format.md`; format version 1, unstable until 1.0): a 64-byte header, a JSON manifest of the graph and 64-byte-aligned raw weights that load as memory-mapped views. `ModelFormat.Convert` turns an ONNX file into one, `ModelFormat.Save` writes an imported model, and `Model.Load` / `Session.Load` open either kind by its first eight bytes. A converted model gives bit-identical outputs. Bad files fail with `ModelFormatException` naming the field.
- `OnnxSweep conformance` and `tools/conformance.sh`: one runner over `onnx/backend/test` (node, simple, pytorch-converted, pytorch-operator) in fast and exact mode, writing per-case status and `docs/conformance.md`.
- About 70 operators for backend-test coverage (activations and trigonometry, bitwise, scatter and set operators, loss functions, LRN, Einsum, DFT/STFT and windows, GridSample, RoiAlign, NonMaxSuppression, Attention opset 23 and others; `docs/ops-coverage.md`), ConvTranspose and Conv with three spatial axes, Resize `antialias`, MelWeightMatrix and IsNaN.
- Control flow and collections: `If`, `Loop`, `Scan`, the sequence operators, `SequenceMap` and the optional operators run as ONNX defines them. Sessions carry sequences and optionals as packed `uint8` tensors (`PackedValue`, `TensorInfo.Kind`); bodies run as nested sessions (`docs/graph-runtime.md` section 8).

### Changed

- int8 packing narrows through value-preserving types, so results agree on WebAssembly, where vector narrowing saturates in compiled code and truncates in the interpreter.
- The int8 model-level metric adds a noise floor on the first quantized tensor of token-input encoders (`docs/int8.md`).
- RoiAlign without `coordinate_transformation_mode` defaults to `half_pixel` from opset 16; Squeeze and Unsqueeze accept a scalar `axes`.

## [0.2.0-preview.1] - 2026-10-30

Graph optimizations, int8 and half-precision storage. All reference models still match ONNX Runtime within 1e-4 scaled error in float32 with outputs independent of thread count; exact-mode results are unchanged by the new passes.

### Added

- Graph optimizations, chosen by `SessionOptions.OptimizationLevel` (`None`, `Basic`, `All`; `All` is the default): constant folding, matrix-product weights packed once per session with bias and activation applied per tile, convolution with a following activation, Conv+BatchNormalization folding, Gelu and attention fusion. Passes that change rounding are skipped in exact mode (`docs/graph-passes.md`).
- int8 inference: `QuantizeLinear`, `DequantizeLinear`, `DynamicQuantizeLinear`, `MatMulInteger`, `ConvInteger`, `QLinearMatMul`, `QLinearConv`; exact int32 product kernels for AVX2, AVX-512 and Arm dot product with a scalar fallback; the `qdq` and `matmul-integer-fusion` passes. See `docs/int8.md`.
- Half-precision storage: models whose weights or values are float16 or bfloat16 run (each operator widens its inputs to float32 and rounds its results back; `docs/fp16.md`). `SessionOptions.WeightStorage` keeps large MatMul, Gemm and Conv weights of float32 models as float16 or bfloat16 to halve their memory.
- `tools/export_models.py --fp16` makes float16 variants of MiniLM-L6 and ResNet-50; `Bar` takes `--weights original|fp16|bf16`.
- A direct depthwise convolution (phase planes for strides, 16/8/4/1-lane paths, four rows at a time for narrow maps) replaces the first one: 3-8x faster on MobileNetV3's layers (`docs/kernels-depthwise.md`).

### Changed

- Matrix products, convolutions and the normalisation operators use no more threads than their work can use (about 80 microseconds of work per thread), so small shapes no longer run slower on many threads; the parallel-for allocates nothing per call.

## [0.1.0-preview.1] - 2026-10-29

First preview: ONNX models load and run on the CPU through `Model.Load(...).Run(...)` or a `Session`, with no native dependencies. The reference set (ResNet-50, MobileNetV3, BERT-base, MiniLM-L6, Whisper-small encoder and decoder, and two text-recognition models) matches ONNX Runtime within 1e-4 scaled error, with outputs independent of thread count. No graph optimizations, int8 or fp16 compute yet; speed is not tuned.

### Added

- Shape and indexing operators: Shape, Size, Squeeze, Unsqueeze, Flatten, Dropout (inference), Transpose, Concat, Split, Slice, Expand, Tile, Gather, GatherElements, GatherND, ScatterND, Pad, ConstantOfShape, Range, OneHot, Trilu, NonZero and TopK. All move or compare elements only, so they are exact in both modes.
- Recurrent operators RNN, GRU and LSTM (all directions, both layouts, sequence lengths, initial state, peepholes, clipping, activation lists; fast and exact) and `Ctc`, greedy and prefix-beam CTC decoding for consumers of recognisers and speech models.
- Element-wise operators: Sub, Div, Mod, Pow, Max, Min, Sum, Mean, Exp, Log, Tanh, Sigmoid, Erf, Gelu, HardSigmoid, HardSwish, LeakyRelu, Neg, Abs, Sign, Reciprocal, Sqrt, Floor, Ceil, Round, Clip, the comparisons, And, Or, Xor, Not, Where, Cast and CastLike; all but Pow have exact kernels.
- The graph runtime: operator contract and static operator table, shape inference and constant evaluation per set of input shapes, execution plans with liveness-based buffer reuse, thread-safe sessions with fast and exact modes, and the `Model` happy path. Operators so far: Add, Mul, Relu, MatMul, Gemm, Reshape, Identity, Constant.
- `Bar check` and `tools/check_reference.sh` compare the library with stored ONNX Runtime outputs and hash outputs across thread counts; `tools/op_inventory.py` lists the operators a set of models uses. The happy path fills in `attention_mask` and `token_type_ids` for text encoders, so `m.Run(tokens)` works on them.
- Convolution and pooling operators: Conv (1-D and 2-D, groups, depthwise, dilation, `auto_pad`, bias), ConvTranspose, MaxPool (with Indices), AveragePool, GlobalAveragePool, GlobalMaxPool, BatchNormalization (inference), Resize and Upsample. `Conv` kernel jobs no longer allocate per call.
- The bar runner can time this library (`--side ours --mode fast|exact`).
- Operators: LayerNormalization, RMSNormalization, InstanceNormalization, GroupNormalization, Softmax, LogSoftmax, Hardmax, the Reduce* family, ArgMax, ArgMin and CumSum, all with exact kernels (`docs/ops-norms.md`).
- The solution skeleton: library, tests, NativeAOT and browser-wasm smoke apps, micro-benchmarks and a fuzz harness.
- Tensor layer: data types with fp16/bf16 conversion, strided views, NCHW/NHWC/blocked layouts, a pooled allocator and arena, and gradient records (public, unstable until the first preview).
- Fast and exact fp32 matrix-product, convolution, attention and softmax kernels, exact transcendental functions, and a fixed-chunk thread pool (internal for now).
- Tooling: reference-model export and pinning, reference outputs from ONNX Runtime, cross-machine output hashing, and a benchmark runner against ONNX Runtime.
- ONNX import: a protobuf reader and ONNX model reader of our own (IR versions 3 and later, any opset, external data confined to the model's directory, memory-mapped weights), and the graph representation it builds. Bad input produces `OnnxImportException`.
