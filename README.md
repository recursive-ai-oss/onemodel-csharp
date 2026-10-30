# OneModel

A neural-network runtime for .NET in pure managed C#, with no native binaries and no dependency beyond the shared framework. It will load ONNX models and our own format and run them on SIMD CPU kernels, with training to follow on the same kernels.

**Status:** work in progress; models cannot be run yet. In place: the tensor layer (strided views over a pooled allocator, fp16/bf16, the record each operation keeps for gradients), fast and bit-exact fp32 kernels for matrix products, convolution, attention and softmax on AVX2, AVX-512, NEON and portable SIMD, our own exp/log/tanh/erf, and a fixed-chunk thread pool. Next: ONNX import, sessions and `Model.Load(...).Run(...)`.

## Layout

| Path | Holds |
|---|---|
| `src/OneModel` | the library, package `RecursiveAI.OneModel` |
| `tests/OneModel.Tests` | unit tests (xUnit v3) |
| `tests/AotSmoke`, `tests/WasmSmoke` | NativeAOT and browser-wasm smoke apps |
| `bench/Micro` | BenchmarkDotNet micro-benchmarks and kernel timing |
| `bench/Bar` | side-by-side timing against ONNX Runtime (the only place it is referenced) |
| `tools/` | scripts that fetch and export reference models, produce reference outputs and compare output hashes across machines |
| `fuzz/` | SharpFuzz targets |
| `docs/` | normative notes; see [docs/README.md](docs/README.md) |

## Running a model

    using var m = Model.Load("model.onnx");
    float[] v = m.Run(tokens);          // float[], long[] or int[] in; names and shapes come from the model

For several inputs or outputs, tensors, exact mode (`new SessionOptions { Mode = ExecutionMode.Exact }`) or a thread limit, use `m.Session` or `Session.Load` directly. Only a handful of operators exist so far (Add, Mul, Relu, MatMul, Gemm, Reshape, Identity, Constant); a model with others fails to load with an exception that lists every missing one. How graphs are planned and run: [docs/graph-runtime.md](docs/graph-runtime.md).

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
