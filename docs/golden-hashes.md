# Golden hashes

Exact mode (see [exact-mode.md](exact-mode.md)) promises the same bits on every OS, CPU, instruction set, thread count and runtime. This file records the check of that promise: the SHA-256 of every output of the reference models, run in `ExecutionMode.Exact`, and how the rows compared. Only hashes appear here.

## Method

`tools/GoldenModels` loads a model, binds the stored inputs (`.npy` files; the input hash is printed too, so a row that was fed different bytes shows up), runs it once in exact mode and prints `<model>/<output> <sha256>`. The hash covers the element type, the rank and dimensions as little-endian 64-bit integers, then the elements row-major. `tools/golden_hash.py` runs the same command under each row's environment and compares line by line. `tools/golden.sh` does it for the whole set on one machine; `tools/golden_wasm.sh` runs the same code on the browser-wasm runtime under node; `--merge` joins saved results from several machines into one table.

The ONNX backend node cases (`--cases`) are hashed the same way, one hash per case (output tensors in order). A case the library cannot run in exact mode prints a fixed marker hash instead, so "skipped" is compared across rows like any other result.

Rows (each must equal every other):

| Row | How |
|---|---|
| x64 JIT, default threads and `ONEMODEL_THREADS=1` | AVX-512 kernel path |
| x64 JIT, `DOTNET_EnableAVX512=0` | AVX2 kernels |
| x64 JIT, `DOTNET_EnableAVX2=0` (and AVX-512 off with it) | 4-lane SSE kernels |
| x64 JIT, `DOTNET_EnableHWIntrinsic=0` | scalar kernels |
| x64 NativeAOT (built for `x86-64-v3`), default threads and one thread | |
| arm64 JIT (macOS), default threads, one thread, `DOTNET_EnableArm64Dp=0`, `DOTNET_EnableHWIntrinsic=0` | NEON, with and without dot product, scalar |
| arm64 NativeAOT (macOS), default threads and one thread | |
| browser-wasm, interpreter | node, single thread, `PackedSimd` |
| browser-wasm, ahead-of-time compiled | node, single thread |

The wasm rows hold 11 of the 14 files: the fp32 BERT-base and the fp32 Whisper-small encoder and decoder are left out because the runtime has a 32-bit address space and the files are copied into its in-memory file system. All 1765 node cases ran on wasm.

Windows x64 and Linux arm64 run on GitHub-hosted runners: `.github/workflows/ci.yml` runs `tools/golden_ci.sh` on every platform in its matrix (JIT, one thread, hardware intrinsics off, NativeAOT, and wasm), comparing five committed test graphs with the expected hashes in `testdata/graphs/golden-exact.txt`. A mismatch on any runner fails the job.

## Result

All rows equal, for every model and for the node-case set. The reference models and the int8 variants all run in exact mode; no operator they use is flagged "not exact". (Fusion passes that change bits do not run in exact mode, see [graph-passes.md](graph-passes.md); the QDQ rewrite is one of them, so the QDQ models run as dequantize, float convolution, quantize.)

Batch 1 inputs from the reference set (`reference-models.md`). The hash of the first output:

| Model | Output | SHA-256 |
|---|---|---|
| resnet50 fp32 | logits | `89f74a216bdf17bd450d0a74ed64e751b8c36ea314fc2a4f66d59576a327eaad` |
| mobilenet_v3_large fp32 | logits | `dbb86f523fb1323dd4d88f9b2b9d782f231bcc06a4774113e23cb710f8bedc37` |
| bert_base fp32 | last_hidden_state | `a2ec1187598fcdeffbb5802fc762765f7ade1ad0a2568a3ea16da27f10c3e962` |
| minilm_l6 fp32 | last_hidden_state | `ef19e17c1851b2124d844cbfa7255aeaad08c97d2f6d53e4c1995ff996eb1c53` |
| whisper_small encoder fp32 | encoder_hidden_states | `285cb006ab4f9840d5ae851841a9d173b7ba20a4cee847f462563815f2c62835` |
| whisper_small decoder fp32 | logits | `ab12ed9d247c8e4a6c93a89f0c40d3f15db891102bb5a0814d21d11b53ec5cd1` |
| crnn fp32 | logits | `d2a360cf1b37c911a7560b83562135dfbfca5fd8514d23bf330acbc10685425d` |
| dbnet fp32 | layout | `1302d16bf52f8c52c48179dbdd11ddb50b9c7a2a1ad79df325f465f8f5f3b338` |
| dbnet fp32 | orient | `9022e8cfc054257e6eecff03c01a4b05e22248e05d3c9cfc27933a3573d6fec1` |
| dbnet fp32 | prob | `1d5d519c988e5d91f7197fce32baf56eee80ebb10447c613ee616fe43391d508` |
| resnet50 int8 QDQ | logits | `6eb104959d4e5177bb587ab429cc078c775dd64917fb9f13ab859e1ea8fdc797` |
| mobilenet_v3_large int8 QDQ | logits | `0b54098f8cdfcb2ca03db41f4b0c375cc8dcfa54f676f5e20123371bf3b15ab3` |
| bert_base int8 dynamic | last_hidden_state | `a172050ebfbdde931fb3df08e833792f555740cda8be9fc4d7b7a3ba995a8790` |
| minilm_l6 int8 dynamic | last_hidden_state | `4cd90837755ebea1af067bd03ef631f8dd2779ef660b19f97b5bcf60f935615b` |
| whisper_small encoder int8 dynamic | encoder_hidden_states | `d2cb6f442450770ee80e2420331e2c24ea949d207efda3fbc07e125dc0757549` |
| whisper_small decoder int8 dynamic | logits | `745f0d6bfc63693a7815f4aeb7779402fda70a35f7bdda8befda7a13cd942764` |

The model files are the fetch-and-export results pinned in `tools/models.json`. A re-export that is not byte-identical (ResNet-50 and MobileNetV3 are not reproducible across machines) changes these hashes; the macOS host used copies of the exported files.

Node cases (ONNX 1.22.0 frozen set, 1765 cases): 1005 run in exact mode; 760 are skipped, all because the library does not implement an operator the case uses or stores a type it lacks (int4, float8), except 16 `Pow` cases: **`Pow` has no exact kernel and is the only implemented operator flagged "not exact"**. The SHA-256 over the sorted `case/<name> <hash>` lines is `87bfc594948556efe83691dc65e5da0cd89ed0d39aff391ead6844ac9d34a90a`.

## A bug the wasm row found

The first wasm rows differed on the two QDQ models. Cause: narrowing sixteen quantized values from `ushort` to `byte` (`Vector128.Narrow`) is truncating in the interpreter and in .NET on x64 and arm64, but saturating once the browser-wasm runtime has compiled the code (after about 2200 calls); a negative int8 value then became 0. `Int8Quantize.Pack16` now narrows through types that hold every value (int to short, then short to sbyte or byte), so truncating and saturating narrowing agree. A narrowing of `uint` to `ushort` (used by the fp16 conversions) showed no such difference in a 200000-call probe.

## Exact versus fast, indicative

Best of 5 warm runs, batch 1, one run per row on a loaded shared host, so read the ratios to about ±15%: exact time divided by fast time. Not a bar measurement.

| Model | x64 (AVX-512) 1 thread | x64 8 threads | arm64 (NEON) 1 thread | arm64 6 threads |
|---|---|---|---|---|
| resnet50 fp32 | 1.18 | 1.14 | 1.57 | 1.44 |
| mobilenet_v3_large fp32 | 1.59 | 1.25 | 0.75 | 1.43 |
| bert_base fp32 | 1.49 | 1.87 | 1.76 | 2.31 |
| minilm_l6 fp32 | 1.50 | 2.41 | 1.72 | 2.70 |
| whisper_small encoder fp32 | 1.60 | 2.35 | 1.53 | 1.97 |
| whisper_small decoder fp32 | 1.44 | 1.89 | 1.83 | 2.17 |
| crnn fp32 | 1.29 | 1.13 | 1.29 | 1.23 |
| dbnet fp32 | 1.51 | 1.57 | 1.36 | 1.27 |
| resnet50 int8 QDQ | 1.81 | 1.20 | 3.61 | 2.14 |
| mobilenet_v3_large int8 QDQ | 0.51 | 1.06 | 0.55 | 0.47 |
| bert_base int8 dynamic | 1.09 | 1.51 | 1.13 | 1.49 |
| minilm_l6 int8 dynamic | 1.31 | 1.48 | 1.21 | 1.60 |
| whisper_small encoder int8 dynamic | 1.50 | 1.81 | 1.14 | 1.49 |
| whisper_small decoder int8 dynamic | 1.29 | 1.84 | 1.10 | 1.76 |

Ratios below 1 are models whose fast run is slower than the exact one (the fast QDQ path is the integer kernels, the exact one is float). Exact is slower mainly because the fused attention, linear/GELU and QDQ rewrites do not run in exact mode, and because exact GEMM cannot fuse the multiply and add.

## Reproducing

```
tools/golden.sh --aot            # this machine's rows, needs the fetched models (tools/prepare.sh)
tools/golden_wasm.sh [--aot]     # browser-wasm rows under node (set NODE), needs the wasm-tools workload
tools/golden_ci.sh jit|aot|wasm  # the committed-hash check CI runs
tools/golden_cost.sh "1 8"       # exact-versus-fast timings
```
