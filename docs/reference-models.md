# Reference models

The seven reference models run through `Model.Load(path).Run(...)` or `Session.Run(...)` and are compared with ONNX Runtime's outputs stored by `tools/make_reference.py` (opset-17 exports, files and pins in `tools/models.json`, method in [oracles.md](oracles.md)).

## Method

* `tools/check_reference.sh` builds `bench/Bar` and runs `Bar check` per model and per thread count (1 and every core). It is a correctness run, not a timing run.
* **Scaled error** per output: `max|ours − ONNX Runtime| / max|ONNX Runtime|`. The gate is 1e-4. **Element-wise error** is `|ours − ref| / max(|ref|, 1e-6)`; it is reported, not gated, because ONNX Runtime itself differs from its other optimization levels by more than that on tiny elements.
* **Hash**: SHA-256 (first 16 hex digits) of each output's bytes in fast mode, from two runs at each thread count. All must be equal: fast mode does not depend on the thread count or the run.
* `tools/op_inventory.py` lists the operators, attributes and node counts of a set of models and marks those missing from the operator table.

## Operators used (opset 17 exports)

Add, And, Cast, Concat, Constant, ConstantOfShape, Conv, Div, Equal, Erf, Expand, Flatten, Gather, Gemm, GlobalAveragePool, GreaterOrEqual, HardSigmoid, HardSwish, LSTM, LayerNormalization, LessOrEqual, MatMul, MaxPool, Mul, Range, ReduceMean, Relu, Reshape, Resize, Shape, Slice, Softmax, Squeeze, Transpose, Unsqueeze, Where (35 operators; the default domain only).

## Results

Fast mode against ONNX Runtime 1.31.1 (`ORT_ENABLE_ALL`) on the stored reference inputs. Scaled error is the gate (1e-4). "Hash equal" means the output hashes were equal at 1 thread and at every core, and from run to run.

| Model (batch) | Outputs | Max scaled error x64 (AVX-512, 32 cores) | Max scaled error arm64 (NEON, 8 cores) | Max element-wise | Hash equal | Status |
|---|---|---|---|---|---|---|
| ResNet-50 (1) | logits | 1.5e-6 | 1.6e-6 | 1.1e-4 | yes (x64 and arm64 hashes also agree) | pass |
| MobileNetV3-large (1) | logits | 3.7e-6 | 3.0e-6 | 1.6e-3 | yes | pass |
| BERT-base (1, 32) | last_hidden_state | 1.3e-6, 7.9e-6 | not run | 1.3 (tiny elements) | yes | pass |
| MiniLM-L6 (1, 32) | last_hidden_state | 7.6e-7, 1.1e-6 | 7.6e-7, 1.1e-6 | 1.0 (tiny elements) | yes (x64 and arm64 hashes also agree) | pass |
| Whisper-small encoder (1) | encoder_hidden_states | 3.4e-5 | not run | 2.0 (tiny elements) | yes | pass |
| Whisper-small decoder (1, 16 tokens) | logits | 3.5e-6 | not run | 0.64 (tiny elements) | yes | pass |
| CRNN (1, 8) | logits | 6.9e-7, 8.5e-7 | not run | 3.9e-4 | yes | pass |
| DBNet (1, two inputs) | prob, layout, orient | 3.7e-7 | not run | 4.1e-4 | yes | pass |

The Whisper encoder is the largest scaled error (3.4e-5, a third of the gate) and is deterministic; the cause (accumulation order over its 12 attention blocks versus ONNX Runtime's) was not investigated. Element-wise errors above 1e-3 come from outputs near zero, as described in [oracles.md](oracles.md).

### Allocation

With one thread, a steady-state `Session.Run` of MiniLM (reused output tensor) allocates 0 bytes (checked by a test when `ONEMODEL_MINILM` names the file). The one-line `Model.Run(tokens)` allocates the output array and, for models with a mask and segment inputs, small input tensors per call.

### Indicative single-thread latency (not a published measurement)

One arm64 laptop-class machine (8 cores, other work running), batch 1, 1 thread, both engines in the same harness, median of 2 repetitions; no weight prepacking, fusion or folding yet.

| Model | Ours (ms) | ONNX Runtime (ms) | Ratio |
|---|---|---|---|
| MiniLM-L6 | 48.8 | 32.1 | 1.52 |
| ResNet-50 | 129.6 | 89.8 | 1.44 |
| MobileNetV3-large | 16.7 | 11.0 | 1.52 |
| BERT-base | 330 | 228 | 1.45 |
| Whisper-small encoder | 4898 | 3684 | 1.33 |

## The one-line happy path on a text encoder

`m.Run(tokens)` works on a model whose inputs are the token ids plus `attention_mask` and `token_type_ids`: the mask is filled with ones (nothing is padding) and the segment ids with zeros, both in the ids' shape; token ids declared `[batch, sequence]` with no shape given are taken as one sequence. Anything else with several inputs goes through `Model.Session.Run` with named tensors.
