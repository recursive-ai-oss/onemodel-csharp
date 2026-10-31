# Tensors and memory

Normative description of the tensor layer: element types, shapes and strides, views, layouts, buffer ownership, version counters,
the backward record, the kernel seam and the numeric conversions. Source: `src/OneModel/Tensors/`. A port to another language
must reproduce the rules in sections 2, 3, 4, 8 and 10 bit for bit; the rest is behaviour a port should match.

## 1. Model in one paragraph

A **buffer** is a 64-byte-aligned block of memory with a length in bytes, a generation number and a version counter. A **tensor** is
a value (no heap object) holding a buffer reference, the buffer generation it was created in, an element type, a logical shape,
strides, and an offset. Every view operation makes a new tensor value over the same buffer. Nothing is copied unless asked.

## 2. Element types

The numeric value of `DType` is the ONNX `TensorProto.DataType` code, so no mapping table exists.

| DType | Code | Bytes | Meaning |
|---|---|---|---|
| F32 | 1 | 4 | IEEE 754 binary32 |
| U8 | 2 | 1 | unsigned |
| I8 | 3 | 1 | signed, two's complement |
| U16 | 4 | 2 | unsigned |
| I16 | 5 | 2 | signed |
| I32 | 6 | 4 | signed |
| I64 | 7 | 8 | signed |
| Bool | 9 | 1 | 0 or 1 |
| F16 | 10 | 2 | IEEE 754 binary16 |
| F64 | 11 | 8 | IEEE 754 binary64 |
| U32 | 12 | 4 | unsigned |
| U64 | 13 | 8 | unsigned |
| BF16 | 16 | 2 | bfloat16 |

All multi-byte values are stored little-endian. Other ONNX codes (STRING 8, COMPLEX64/128 14–15, FLOAT8 variants 17–20, 4-bit and
2-bit integers 21–22 and 25–26, FLOAT4E2M1 23, FLOAT8E8M0 24, FLOAT6 27–28) are named by `DTypes.OnnxName` and rejected by
`DTypes.FromOnnx` with a message that names the type. Only floating-point tensors can require gradients.

## 3. Shapes, strides, limits

- Rank 0 to 8. Rank 0 is a scalar with one element.
- Each extent is a non-negative 32-bit integer. The element count (the product of extents) is at most 2^31−1, and each **prefix**
  product (left to right) must also fit, so `[100000, 100000, 0]` is rejected.
- A buffer holds at most 2^31−1 bytes, so `count × element size` must fit. Violations throw `TensorShapeException` before anything is
  allocated.
- Strides are signed 32-bit integers counting **elements** (not bytes). Zero means broadcast; negative means reversed.
- Element `(i0, …, ik)` lives at element offset `offset + Σ i_d · stride_d`. A view is valid only if, for a non-empty tensor, the
  lowest reachable offset (`offset + Σ over negative strides of (extent−1)·stride`) is ≥ 0 and the highest is below the buffer's
  element count. The check runs when the view is made.
- **Contiguous** means: the stride of every dimension with extent ≠ 1 equals the product of the extents to its right (empty tensors
  are contiguous). Strides of extent-1 dimensions are ignored. `Contiguous()` returns the tensor itself when contiguous, else a copy
  in row-major order.
- Broadcasting (`Shape.Broadcast`) is numpy's multidirectional rule: right-align, each pair of extents equal or one of them 1. The
  result must also satisfy the size limit.

## 4. Views (no data moves)

| Operation | Result |
|---|---|
| `Permute(p)` | dimension i of the result is dimension p[i] of the source; shape and strides permuted |
| `Transpose(a, b)`, `T()` | swap two dimensions |
| `Slice(axis, start, end, step)` | `0 ≤ start ≤ end ≤ size`, `step ≥ 1`; extent `ceil((end−start)/step)`; offset += `start·stride` (only if the result is non-empty); stride ×= step |
| `Select(axis, i)` | removes the axis; offset += `i·stride` |
| `Flip(axis)` | stride negated; offset += `(extent−1)·stride` (if extent > 0) |
| `Unsqueeze(a)` | inserts an extent-1 dimension (stride = `extent·stride` of the dimension it precedes, or 1 at the end) |
| `Squeeze(a)` / `Squeeze()` | removes one extent-1 dimension / all of them |
| `Expand(shape)` | right-aligned; extent 1 or a missing leading dimension gets stride 0 and the target's extent |
| `TryReshape(shape)` | see below |

**Reshape viewability.** Drop source dimensions of extent 1. Merge neighbouring source dimensions into a *run* while
`stride[i] == stride[i+1] · extent[i+1]`. Each run has a length (product of its extents) and the stride of its innermost
dimension. The new shape is viewable iff its dimensions, taken left to right, partition into groups whose extent products equal the
run lengths in order, followed only by extent-1 dimensions. Inside a group, strides are contiguous starting from the run's inner
stride. Empty tensors are always viewable with contiguous strides. `Reshape` falls back to `Contiguous()` + view when
`TryReshape` fails; row-major element order is the same either way. `Reshape(-1, …)` infers one extent.

## 5. Layouts

The logical shape of an image tensor is always `(N, C, H, W)`. Layout says where the elements live.

| Kind | Storage | How it is expressed |
|---|---|---|
| Nchw | row-major N, C, H, W | contiguous strides |
| Nhwc | row-major N, H, W, C | strides `(H·W·C, 1, W·C, C)` over the NCHW logical shape |
| NchwC(B) | `[N][ceil(C/B)][H][W][B]`, channel `c` at block `c/B`, lane `c%B`; lanes past `C` in the last block are **zero** | not strided: an opaque tensor flagged with block size B (1…256) |

`Layout` of a plain rank-4 tensor is derived from its strides (extent-1 dimensions ignored; when both patterns match, Nchw wins).
Other ranks and other strides report Strided. `ToLayout` converts between the three; it is a pure data movement (values and logical
shape are unchanged). NCHW and NHWC conversions are differentiable (backward is the identity on the logical tensor); blocked tensors
are kernel-internal: only `ToLayout`, `Data<T>()` (the physical span of `ceil(C/B)·N·H·W·B` elements) and metadata work on them, and
they cannot require gradients. Further blocked families (for convolution weights) are added the same way, as a flagged kind.

## 6. Buffers and the pool

- Every buffer starts on a 64-byte boundary.
- **Pool** (`BufferPool`). A request of `n` bytes (0 ≤ n ≤ 2^31−1) is served from a size class. Class size for `n ≤ 256`:
  `max(64, n rounded up to a multiple of 64)`. For `n > 256`: let `b` be the largest power of two below `n` (`b = 2^floor(log2(n−1))`),
  `step = b/4`, class = `b + ceil((n − b)/step)·step`. So classes are `b·{1.25, 1.5, 1.75, 2}`; waste is below 25 %. Released
  buffers wait in a per-class free list; the next request of the class takes one. Idle memory is capped (`MaxIdleBytes`, default
  2 GiB); beyond the cap a released block is freed. Contents of a rented buffer are unspecified unless zeroing was requested.
  Nothing in the pool depends on thread count or timing; it is safe to use from many threads.
- **Arena** (`Arena`). Bump allocation, 64-byte aligned, for the state of one run. Chunks (default 1 MiB, or the request if larger)
  come from a pool. `Reset()` reclaims everything; every tensor allocated since is then stale. Not thread-safe: one arena per
  concurrent run. After a warm-up run, a run with the same allocation sequence allocates no managed or native memory.
- **Lifetime.** `Tensor.Dispose()` returns the buffer for the tensor and all its views and copies of the handle. A buffer
  carries a generation; it increases when the buffer is returned or its arena reset. Every access compares the tensor's remembered
  generation to the buffer's and throws `StaleTensorException` on mismatch, so a tensor can never read memory re-rented to someone
  else. Releasing is idempotent: releasing again through any copy of the handle does nothing, and a stale handle never releases the memory
  re-rented to someone else; only reading or writing data through it throws (`StaleTensorException`, an `ObjectDisposedException`). A leaked (never disposed) pooled buffer is freed by the GC's finalizer.
- **`CHECKED` builds.** New and returned memory is filled with 0xFF (a NaN for floats, −1 for integers), so reads of
  unwritten or released memory stand out. 64 canary bytes (0xA5) follow every buffer's length and are verified when it is released
  or its arena reset; a kernel that wrote past the end makes release throw.

## 7. Version counters

Every buffer has a 64-bit version, 0 at rent. All views share it. It increases by one on: `MutableData`, `MutableStorage`, `Set`,
`Fill`, `CopyFrom`. (Kernels that write an output through a writable span obtain it through these accessors.) A write through a span
obtained earlier is not tracked. `SavedTensor` remembers the version at save time and `Unpack()` throws `TensorVersionException`
if it differs, so an in-place update to anything the backward pass needs is caught at backward time rather than yielding a wrong
gradient.

## 8. Gradient tracking and the backward record

- `WithRequiresGrad()` returns a handle that is a gradient leaf. Because a tensor is a value, the flag is on the returned handle.
- An operation records only if `GradMode.IsEnabled` (per thread; `GradMode.NoGrad()` scopes it off) and at least one input requires
  gradients. Otherwise it does no extra work and allocates nothing; for example, `MaxPool2d` then does not compute argmax.
- A `BackwardRecord` holds: the op (`OpKind`), per input the autograd node (null if that input needs no gradient), shape and type;
  positional `Saved` slots (tensors that share storage with the original, with the version at save time; a slot is empty when the
  operation did not need it); `Ints` and `Floats` attribute vectors. The result tensor points to the record (`GradFn`).
  Tensors the record created itself (index tensors, intermediates) are *owned* and freed by `Release()`.
- In-place `CopyFrom` on a tensor that requires gradients (while recording is on) throws.

| Op | Saved slots | Ints | Backward uses |
|---|---|---|---|
| Add, Sub | none | | gradient summed to each input shape (Sub negates b's) |
| Mul | 0 = a if b needs grad; 1 = b if a needs grad | | da = g·b, db = g·a, each summed to its input shape |
| Div | 0 = a if b needs grad; 1 = b | | da = g/b; db = −g·a/b² |
| Neg | none | | −g |
| Relu | 0 = output | | g where output > 0 |
| MatMul | 0 = a if b needs grad; 1 = b if a needs grad | | dA = g·bᵀ, dB = aᵀ·g, the transposes being strided views of the saved tensors |
| MaxPool2d | 0 = Int32 flat indices of the winners (logical row-major NCHW position in the input) | kernelH, kernelW, strideH, strideW, padH, padW | each output gradient is added at its index; input shape from InputShape(0) |
| Cast | none | | cast g back to InputDType(0) |
| Slice | none | axis, start, end, step | zeros of the input shape with g written into the slice |
| Permute | none | the permutation | inverse permutation |
| Reshape (also Squeeze, Unsqueeze, Contiguous, NHWC conversion, and Reshape-with-copy) | none | | reshape g to InputShape(0) |
| Expand | none | | sum g down to InputShape(0) |
| Flip | none | axis | flip g |
| Select | none | axis, index | zeros of the input shape with g written at the index |

The operators added with the engine (GELU, sigmoid, tanh, sum, mean, softmax, log-softmax, layer and batch norm, Conv2d, AvgPool2d,
Concat, Embedding, NllLoss, Attention) and their records are in `autograd.md` section 5. Conventions as first fixed: *normalization* saves the normalized rows and
the inverse standard deviation as owned slots plus the scale input; *GELU* saves the input and the tanh of its inner term;
*softmax* saves its output and keeps the scale in `Floats`; *convolution* saves the input and the weights (or the unfolded input
as an owned slot) and keeps kernel, stride, padding and padded K in `Ints`; *embedding lookup* saves the index tensor and keeps the
skip id in `Ints`, the table shape coming from InputShape(0); fused ReLU+pooling saves Int32 indices with −1 meaning "no positive
element, no gradient".

## 9. The kernel seam

`ITensorKernels` is the one place the original eager operations call kernels: `Binary`, `Unary`, `MatMul`, `MaxPool2d`. The default set,
`DenseKernels`, is `ReferenceKernels` except that `MatMul` runs on the blocked GEMM (fast or exact per `TensorCompute.Mode`); the
operators added with the autograd engine call the kernel layer directly (`autograd.md` section 4). Inputs are float32
tensors with arbitrary strides (the caller expands broadcasts to stride 0 first, so all operands of `Binary` have one shape); the
output is allocated by the caller. An implementation is selected once, at startup or session creation (`KernelSet.SetDefault`, or
the `kernels` argument), from the machine's instruction sets and the fast-or-exact choice. `ReferenceKernels` defines the meaning:
scalar loops, `double` accumulation in `MatMul`, no threads. Pooling semantics: the window starts at `y = oy·stride − pad` and
includes only in-bounds positions (padding never wins); the first element wins ties; a NaN in the window wins, the first NaN is
reported; output extent `floor((size + 2·pad − kernel)/stride) + 1`, with `pad < kernel`.

## 10. Numeric conversions

**float → binary16** (round to nearest, ties to even, IEEE 754-2008 §4.3.1): with `a` the bits of |x|:
`a > 0x7F800000` (NaN): sign, `0x7E00`, top nine payload bits (`(a >> 13) & 0x1FF`); `a = 0x7F800000` or `a ≥ 0x477FF000`
(|x| ≥ 65520): signed infinity; `a ≥ 0x38800000` (normal): `m = a − (112 << 23)`, result bits `(m + 0xFFF + ((m >> 13) & 1)) >> 13`
(a carry into the exponent is correct); `a ≤ 0x33000000` (|x| ≤ 2^−25): signed zero; otherwise subnormal: with `e = a >> 23`,
`mant = (a & 0x7FFFFF) | 0x800000`, `shift = 126 − e`, `r = mant >> shift`, round up when the remainder exceeds half, or equals
half and `r` is odd.
**binary16 → float** is exact: subnormals as `man · 2^−24`, NaN keeps its payload shifted left 13 with the quiet bit forced.
**float → bfloat16**: NaN → `(bits >> 16) | 0x40`; otherwise `(bits + 0x7FFF + ((bits >> 16) & 1)) >> 16`. **bfloat16 → float**:
`bits << 16`, quiet bit forced for NaN. Both are verified over all 65536 half and bfloat16 patterns and 2 million random and
tie-constructed floats against an independent nearest-neighbour reference.

**Cast** (reference rules): float → float by the conversions above (float64 → 16-bit goes through float32 and so rounds twice);
integer or bool → float rounds to nearest even; float → integer truncates toward zero, saturates to the target range, NaN → 0;
integer → integer wraps modulo 2^bits; any → bool is "≠ 0" (NaN is true); bool → number is 0 or 1. Where ONNX leaves
out-of-range behaviour open, these rules are provisional and are checked against ONNX Runtime's output by the operator work.
