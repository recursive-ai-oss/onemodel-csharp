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

Filled in as the models run.

## The one-line happy path on a text encoder

`m.Run(tokens)` works on a model whose inputs are the token ids plus `attention_mask` and `token_type_ids`: the mask is filled with ones (nothing is padding) and the segment ids with zeros, both in the ids' shape; token ids declared `[batch, sequence]` with no shape given are taken as one sequence. Anything else with several inputs goes through `Model.Session.Run` with named tensors.
