# Performance against ONNX Runtime

Latency and peak memory of this library against ONNX Runtime 1.31.1 on CPU, measured for the 1.0 release (the full table) and re-measured for 1.1 on the fp32 rows that release changed, with `bench/Bar` (method: [bench-method.md](bench-method.md)). Each cell is ours / ONNX Runtime: below 1 means we are faster or smaller. ONNX Runtime runs at `ORT_ENABLE_ALL` with the same thread count as ours. Fast mode, optimization level All, median of 3 repetitions, one process per row and engine.

Machines:

- **x64:** AMD Ryzen 9 7950X3D (Zen 4; AVX2, AVX-512, AVX-512 VNNI), 32 logical processors, Linux. "All" is 32 threads.
- **arm64:** Apple M2 (4 performance and 4 efficiency cores), macOS. "All" is 8 threads.

Batch N is 32 for the text encoders and 8 for the convolutional models. The Whisper encoder and decoder and the DBNet text detector were measured at batch 1 only, and the CRNN and DBNet OCR models in fp32 only.

## Latency, ours / ONNX Runtime

| Model | Type | Batch | x64 1 thread | x64 all | arm64 1 thread | arm64 all |
|---|---|---|---|---|---|---|
| MiniLM-L6 | fp32 | 1 | 1.18 | 0.76 | 1.04 | 0.86 |
| MiniLM-L6 | fp32 | 32 | 1.07 | 0.68 | 1.10 | 0.68 |
| BERT-base | fp32 | 1 | 1.09 | 0.77 | 1.07 | 1.02 |
| BERT-base | fp32 | 32 | 1.18 | 0.66 | 1.12 | 1.00 |
| ResNet-50 | fp32 | 1 | 1.42 | 1.16 | 1.15 | 0.76 |
| ResNet-50 | fp32 | 8 | 1.36 | 1.90 | 1.12 | 0.79 |
| MobileNetV3-Large | fp32 | 1 | 1.49 | 0.74 | 0.82 | 0.70 |
| MobileNetV3-Large | fp32 | 8 | 1.49 | 0.87 | 0.82 | 0.82 |
| Whisper-small encoder | fp32 | 1 | 1.23 | 0.52 | 1.41 | 1.05 |
| Whisper-small decoder | fp32 | 1 | 1.12 | 0.54 | 1.07 | 0.82 |
| CRNN (text lines) | fp32 | 1 | 1.53 | 1.40 | 1.26 | 1.25 |
| CRNN (text lines) | fp32 | 8 | 1.69 | 4.48 | 1.49 | 1.79 |
| DBNet (text detection) | fp32 | 1 | 2.38 | 1.73 | 1.52 | 1.67 |
| MiniLM-L6 | int8 dynamic | 1 | 2.33 | 0.77 | 1.36 | 1.36 |
| MiniLM-L6 | int8 dynamic | 32 | 2.06 | 0.96 | 1.33 | 1.12 |
| BERT-base | int8 dynamic | 1 | 2.44 | 1.20 | 1.24 | 1.06 |
| BERT-base | int8 dynamic | 32 | 2.14 | 1.26 | 1.18 | 0.95 |
| ResNet-50 | int8 QDQ | 1 | 1.38 | 0.36 | 1.46 | 0.96 |
| ResNet-50 | int8 QDQ | 8 | 1.28 | 0.34 | 1.39 | 1.20 |
| MobileNetV3-Large | int8 QDQ | 1 | 1.23 | 0.40 | 1.49 | 0.90 |
| MobileNetV3-Large | int8 QDQ | 8 | 1.19 | 0.27 | 1.43 | 0.81 |
| Whisper-small encoder | int8 dynamic | 1 | 1.90 | 0.68 | 1.78 | 1.28 |
| Whisper-small decoder | int8 dynamic | 1 | 2.84 | 1.12 | 1.50 | 1.08 |

### 1.1 re-measurement (fp32, fast mode)

Release 1.1 changed the recurrent layers, small-model per-call costs and 3x3 convolutions (Winograd), so these fp32 rows were measured again on the same machines; every other row is unchanged from the table above.

| Model | Batch | x64 1 thread | x64 all | arm64 1 thread | arm64 all |
|---|---|---|---|---|---|
| ResNet-50 | 1 | 1.18 | 1.02 | 0.98 | 0.70 |
| ResNet-50 | 8 | 1.24 | 0.97 | 0.98 | 0.55 |
| MobileNetV3-Large | 1 | 1.46 | 0.80 | 0.81 | 0.60 |
| MobileNetV3-Large | 8 | 1.47 | 1.23 | 0.79 | 0.54 |
| MiniLM-L6 | 1 | 1.17 | 0.71 | 1.07 | 1.01 |
| BERT-base | 1 | 1.11 | 0.60 | 1.04 | 0.99 |
| CRNN (text lines) | 1 | 1.04 | 1.22 | 1.01 | 1.28 |
| CRNN (text lines) | 8 | 1.01 | 1.40 | 0.93 | 0.78 |
| DBNet (text detection) | 1 | 1.58 | 0.84 | 1.08 | 1.23 |

## Where it is slower, and why

- **int8 matrix products on x64 at one thread** (MiniLM, BERT, Whisper int8, about 2 to 2.8 times). .NET 10 exposes the x86 integer dot-product instructions only in their VEX forms (AVX-VNNI), which the test CPU does not have, so the int8 kernels use `vpmaddwd` on 16-bit operands, half the multiply-adds per instruction of a VNNI kernel (`docs/int8.md`). On CPUs with AVX-VNNI, and on arm64 with its dot-product instructions, the gap is smaller.
- **DBNet and other very small models on x64 at one thread** (about 1.5 ms per call; 1.58 times in 1.1, 2.38 at 1.0): fixed per-call costs weigh more.
- Since 1.1 the recurrent layers run on the thread pool in fixed hidden-unit chunks, and ResNet-50 at batch 8 on 32 x64 threads uses Winograd convolutions; both rows are now within 1.5 times.

## Peak memory, ours / ONNX Runtime

Peak resident memory is at most 1.40 times ONNX Runtime's on every row on x64 (largest: Whisper decoder fp32 1.40, Whisper decoder int8 1.31, BERT-base fp32 1.29) and below 1 on most int8 and batch-N rows. On arm64 the Whisper decoder fp32 is 1.53 times at one thread (1714 MB against 1122 MB) and 1.45 times at 8; every other arm64 row is at most 1.36 times. In the 1.1 re-measurement the largest ratio is ResNet-50 fp32 batch 1 on arm64 at 1.44 (313 MB against 217 MB); on x64 the re-measured rows are at most 1.29 (BERT-base). Steady-state runs allocate no managed memory per call.

## Load time

Session creation (import, graph passes, weight packing) is reported per row in the raw results and is not gated.

Loading from an `UnmanagedMemoryStream` (an embedded resource), which 1.2 views in place instead of buffering, is about twice as fast as in 1.0 and 1.1 (indicative: x64, 4 threads, fresh processes, 2-3 recorded repetitions on a shared machine; XLM-RoBERTa-base 1.1 GB 0.85-0.94 s against 1.84-1.97 s with peak memory 2.19 against 3.25 GB, MiniLM-L6 124-152 ms against 218-229 ms); loading from a path or another stream, and the first run, are unchanged.
