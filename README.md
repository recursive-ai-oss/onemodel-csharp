# OneModel

A neural-network runtime for .NET in pure managed C#, with no native binaries and no dependency beyond the shared framework. It loads ONNX models and its own format, optimizes the graph and runs it on SIMD CPU kernels, and it trains models on the same kernels.

**Status:** preview. What works:

- **Inference:** ONNX import (our own protobuf reader, any opset from 7, external data), graph passes (constant folding, weight prepacking, conv/GEMM/attention fusions), memory planning, thread-safe sessions and the one-line `Model.Load(...).Run(...)`. Coverage of the ONNX backend test suite is published in [docs/conformance.md](docs/conformance.md), gaps included; control flow (`If`, `Loop`, `Scan`), sequences and optionals run.
- **Data types:** fp32, fp16/bf16 models and weight storage, int8 (dynamic and static quantization, per-channel weights, QDQ graphs).
- **Kernels:** AVX2, AVX-512, NEON and portable SIMD for matrix products, convolution, depthwise convolution, attention, recurrent layers, norms and reductions; a fixed-chunk thread pool, so results do not depend on the thread count.
- **Exact mode:** opt-in, bit-identical outputs across operating systems, CPUs, instruction sets, thread counts and WebAssembly ([docs/exact-mode.md](docs/exact-mode.md), [docs/golden-hashes.md](docs/golden-hashes.md)).
- **Own format:** save and load an imported model in a simple memory-mappable format ([docs/format.md](docs/format.md)).
- **Training** (package `RecursiveAI.OneModel.Training`): reverse-mode autodiff on the tensor layer, modules, losses, SGD/Adam/AdamW, learning-rate schedules, deterministic data loading, bit-exact checkpoints, and tracing a module into a graph that runs and saves like an imported one ([docs/training.md](docs/training.md), [docs/tracing.md](docs/tracing.md)).

The API may still change between previews; see [CHANGELOG.md](CHANGELOG.md).

## Layout

| Path | Holds |
|---|---|
| `src/OneModel` | the library, package `RecursiveAI.OneModel` |
| `src/OneModel.Training` | training, package `RecursiveAI.OneModel.Training` |
| `tests/OneModel.Tests` | unit tests (xUnit v3) |
| `tests/AotSmoke`, `tests/WasmSmoke` | NativeAOT and browser-wasm smoke apps |
| `tests/GuideSamples` | the examples of the usage guide, built and run in CI |
| `tests/OnnxSweep` | runs the ONNX backend test cases and writes the conformance table |
| `bench/Micro` | BenchmarkDotNet micro-benchmarks and kernel timing |
| `bench/Bar` | side-by-side timing against ONNX Runtime (the only place it is referenced) |
| `tools/` | scripts that fetch and export reference models, produce reference outputs and compare output hashes across machines |
| `fuzz/` | SharpFuzz targets |
| `docs/` | normative notes; see [docs/README.md](docs/README.md) |

## Running a model

    using var m = Model.Load("model.onnx");
    float[] v = m.Run(tokens);          // float[], long[] or int[] in; names and shapes come from the model

The usage guide, with examples that CI builds and runs, is in [docs/guide](docs/guide/README.md): getting started, tensors and who owns their memory, errors, the file format, training and checkpoints, tracing, and migration notes from ONNX Runtime and TorchSharp.

For several inputs or outputs, tensors, exact mode (`new ModelOptions { Mode = ArithmeticMode.Exact }`, `using RecursiveAI.OneModel.Graph;`), a thread limit, an optimization level or compact fp16/bf16 weight storage, use `m.Session` or `Session.Load` directly. A model with operators that are not supported fails to load with an exception that lists every missing one; supported operators and opsets are in [docs/conformance.md](docs/conformance.md). How graphs are planned and run: [docs/graph-runtime.md](docs/graph-runtime.md). `ModelFormat.Convert` saves an ONNX model in our own format, and `Model.Load` reads either.

### NativeAOT users: set `IlcInstructionSet`

A NativeAOT binary targets the x86-64 baseline unless told otherwise, and the library then silently runs its 4-lane kernels instead of the AVX2 or AVX-512 ones. Results are the same, but slower. Publish with the instruction set you target, for example:

    dotnet publish -c Release -p:IlcInstructionSet=x86-64-v3     # AVX2 and FMA machines (2013 and later)
    dotnet publish -c Release -p:IlcInstructionSet=native         # the machine you publish on

`Session.KernelPath` reports which kernels a session chose (`Avx2`, `Avx512`, `Neon`, `Portable128`, `Scalar`). `tests/AotSmoke` publishes with `x86-64-v3` and fails if an x64 machine does not pick AVX2 or AVX-512. The ordinary JIT build needs nothing: it checks the hardware at start-up.

## Building

The .NET SDK in `global.json` is required. Build output goes to `$ONEMODEL_ARTIFACTS` (default: a folder in the temp directory), never into the checkout.

    dotnet build -c Release
    dotnet test --solution OneModel.slnx -c Release

`-c Checked` builds optimized code with the `CHECKED` symbol, which enables bounds assertions at kernel entry points and poisons freed buffers. Tests and fuzzing run in `Checked`.

## Licence

`MIT OR Apache-2.0`, see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE). Third-party material is listed in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
