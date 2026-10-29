# Shape and indexing operators

Notes on the operators that move, select or describe elements without computing on them: what each one does where the ONNX text leaves room, and how each choice was settled. Semantics come from ONNX's `Operators.md` and `Changelog.md`; open points were settled by running ONNX Runtime 1.31.1 as a black box on crafted inputs (`tools/ops_shape_indexing.py` generates the cases and records its outputs).

## 1. Common rules

- **Types.** Operators that only move elements accept every stored element type (a gather moves 4-byte elements the same way for float32, int32 and uint32). Operators that compare or count (TopK, NonZero, ScatterND reductions, Range, OneHot) list their types below.
- **Exact by construction.** None of these operators does arithmetic on floating-point values, so each is bit-identical in fast and exact mode, on every instruction set and thread count (the kernels are scalar span loops and `memmove`; none uses vector intrinsics). All are flagged as having an exact kernel. NaN payloads and signed zeros pass through untouched. Exceptions in a *value* sense: ScatterND `add` and `mul` do one float add or multiply per element (correctly rounded, so also identical everywhere).
- **Views.** Squeeze, Unsqueeze, Flatten, Reshape, Identity and Dropout (without a mask output) are views: the session builds them and no byte moves. Everything else that returns existing elements copies them.
- **Plan-time values.** Shape and Size are constants of the plan. A node whose shape depends on an input value (Slice's starts, Squeeze's axes, Expand's shape, Tile's repeats, Pad's pads, Range, OneHot's depth, TopK's k, ConstantOfShape's shape, Split's sizes) reads it at plan time when it is a constant, which is what turns `Shape → Gather → Unsqueeze → Concat → Reshape` chains into a literal target shape. When it is not a constant the node asks for run-time values and is shaped again in every run. Only NonZero is data dependent in the contract's sense (its output extent is a count of values).
- **Errors.** Shape and attribute problems are `SessionException(InvalidModel)` naming the node when the session builds the plan. An index outside its extent at run time is `SessionException(InvalidInput)`. Nothing indexes outside a buffer on bad input.
- **Threads.** The kernels are single-threaded: they are memory-bound copies and the result cannot depend on the thread count.

## 2. Operators

| Operator | Opsets | Kind | Notes |
|---|---|---|---|
| Shape | 1+ | plan constant | `start`/`end` (15+): negative values count from the end, both clamp to [0, rank]; start past end gives an empty vector. |
| Size | 1+ | plan constant | element count as an int64 scalar. |
| Squeeze | 1+ | view | axes: attribute before 13, optional input from 13, none = every extent-1 axis. An axis whose extent is not 1, or listed twice, is an error. |
| Unsqueeze | 1+ | view | axes index the *output*, negative allowed; duplicates are an error. Output rank limit 8. |
| Flatten | 1+ | view | `axis` in [-rank, rank]; axis 0 gives `[1, size]`. |
| Dropout | 1+ | view, or copy when a mask output is used | Inference only: output = input, mask (if used) all true. `ratio` is ignored. A `training_mode` that is true is `Unsupported`; one fed at run time is read at run time. |
| Transpose | 1+ | copy | `perm` default reverses the axes. Axes that stay adjacent in both layouts are merged; a real transpose runs in 32x32 tiles. |
| Concat | 1+ | copy | `axis` required from opset 4 (default 1 before). Empty inputs are allowed. |
| Split | 1+ | copy | Sizes from the `split` attribute (before 13) or input (13+). With no sizes: equal parts (the extent must divide) before opset 18; from 18, parts of `ceil(extent / n)` with the last part taking the rest (`num_outputs`, default the output count). |
| Slice | 1+ | copy | See 3. |
| Expand | 8+ | copy | Output = bidirectional broadcast of the input and the shape input (so an input extent may exceed a shape entry of 1). |
| Tile | 6+ | copy | A repeat of 0 gives an empty axis. |
| Gather | 1+ | copy | Indices int32 or int64, negative allowed (counted from the end). |
| GatherElements | 11+ | copy | Indices int32 or int64; the indices' extents may be smaller than the data's off the axis. |
| GatherND | 11+ | copy | `batch_dims` supported; indices int64. |
| ScatterND | 11+ | copy | Output = data, then the updates applied in index order (duplicates: the last write wins, or the reduction folds them in order). `reduction` none/add/mul/max/min on float32, float64, int32, int64. For max and min the result is `max(u, o)` / `min(u, o)` written as `u < o ? o : u` and `o < u ? o : u`, so a NaN in either operand makes the update win; this is what ONNX Runtime does. |
| Pad | 1+ | copy | See 4. |
| ConstantOfShape | 9+ | fill | Value attribute of one element of any stored type, default float32 0. A negative extent is an error. |
| Range | 11+ | generate | float32, float64, int16, int32, int64, and float16 / bfloat16 (computed in float32, rounded to nearest even). Count = max(ceil((limit - start) / delta), 0), with the quotient taken **in double precision** from the (float) values (settled by black box: `Range(0, 0.3, 0.1)` in float32 has 4 elements); element i = start + i*delta in the element type. Delta 0 and a non-finite count are errors. |
| OneHot | 9+ | generate | Indices of any numeric type (a NaN index selects nothing), depth any numeric scalar, values any stored type. An index outside [-depth, depth) gives an all-off row. Rank-0 indices work (ONNX Runtime refuses them). |
| Trilu | 14+ | copy | `k` is optional. Batched over leading axes. |
| NonZero | 9+ | data dependent | Output `[max(rank,1), count]`; -0.0 is zero, NaN is not. A scalar input counts as a 1-element vector (ONNX Runtime agrees: output `[1, count]`). |
| TopK | 1+ | copy | See 5. |

## 3. Slice

Starts, ends, axes and steps come from attributes (opset 1-9) or inputs (10+). Per sliced axis of extent n:

- A negative start or end adds n first. For a positive step both then clamp to [0, n]; for a negative step the start clamps to [0, n-1] and the end to [-1, n-1] (the end -1 means "through element 0"). The step is clamped to the int32 range first (a larger step selects at most one element anyway), so INT64_MAX steps work. Step 0 is an error. An axis of extent 0 gives an empty result.
- Count = `ceil((end - start) / step)` for a positive step, `ceil((start - end) / -step)` for a negative one, at least 0.
- **ONNX Runtime's reading of the end sentinels (black box).** With a negative step an end of exactly INT32_MAX or INT64_MAX means "to the beginning", the same as INT_MIN, not "clamped to n-1" (`x[INT64_MAX : INT32_MAX : -1]` of five elements gives all five, reversed; an end of 100 gives nothing). The spec's clamping alone would give an empty slice; models rely on the sentinel, so we follow ONNX Runtime. 400 random start/end/step combinations plus the probes agree.
- Axes may be negative (11+) and must not repeat. Unlisted axes are taken whole.

## 4. Pad

`pads = [begin_0 .. begin_{k-1}, end_0 .. end_{k-1}]` for the listed axes (all axes unless an `axes` input is given, opset 18+); other axes get 0. Attributes before opset 11 (`paddings` in opset 1, `pads` in 2 to 10, plus `value`); `pads`, an optional `constant_value` and the optional `axes` are inputs from 11. A negative pad crops.

- **constant:** the output is filled with the value (default 0, any type) and the overlap with the input is copied.
- **edge:** an index outside the input reads the nearest edge element.
- **reflect:** mirrors without repeating the edge element. **Beyond one reflection we continue periodically** (period `2*(n-1)`; an extent of 1 repeats its only element). ONNX Runtime refuses a reflect pad of `n` or more on an axis, and refuses an axis of extent 1; we accept both. The accepted range agrees with ONNX Runtime wherever it runs.
- **wrap** (opset 19): periodic with period n.
- Non-constant modes on an empty axis that would need elements are an error.

## 5. TopK

`k` is an attribute before opset 10 and an input from then on (a constant k gives plan-time shapes; a computed k is read at run time). `largest` (default 1) picks the order; `sorted` is accepted and ignored (the output is always sorted, which is a valid reading of both values).

- **Ties** keep input order: of equal values the lower index comes first, in both directions. This matches ONNX Runtime on every tie case tried, for k up to 64 and axis extents up to 64.
- **NaN** counts as greater than every number (first when `largest`, last when not). The spec does not define it, and ONNX Runtime's result for NaN inputs differs between its own code paths (the same input gives different placements for k = 2 and k = extent), so the NaN cases of the oracle comparison are skipped. This is a deliberate, documented choice, not agreement.
- Selection per row: insertion into a sorted list of k for k <= 16, otherwise a stable merge sort of the row's indices. Either way the result is the stable order, so the paths agree.

## 6. Not covered here

Scatter and ScatterElements, the sequence operators (SplitToSequence and friends), Compress, Reshape before opset 5 (the shape was an attribute), and `Identity` on sequence types.
