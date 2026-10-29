# OneModel

A neural-network runtime for .NET in pure managed C#, with no native binaries and no dependency beyond the shared framework. It will load ONNX models and our own format and run them on SIMD CPU kernels, with training to follow on the same kernels. The name is provisional.

**Status:** work in progress. The repository holds the buildable skeleton; there is nothing to use yet.

## Layout

| Path | Holds |
|---|---|
| `src/OneModel` | the library, package `RecursiveAI.OneModel` |
| `tests/OneModel.Tests` | unit tests (xUnit v3) |
| `tests/AotSmoke`, `tests/WasmSmoke` | NativeAOT and browser-wasm smoke apps |
| `bench/Micro` | BenchmarkDotNet micro-benchmarks |
| `fuzz/` | SharpFuzz targets |
| `docs/` | normative notes; see [docs/README.md](docs/README.md) |

## Building

The .NET SDK in `global.json` is required. Build output goes to `$ONEMODEL_ARTIFACTS` (default: a folder in the temp directory), never into the checkout.

    dotnet build -c Release
    dotnet test --solution OneModel.slnx -c Release

`-c Checked` builds optimized code with the `CHECKED` symbol, which enables bounds assertions at kernel entry points and poisons freed buffers. Tests and fuzzing run in `Checked`.

## Licence

`MIT OR Apache-2.0`, see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE). Third-party material is listed in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
