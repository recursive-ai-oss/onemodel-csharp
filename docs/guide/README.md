# OneModel guide

How to use the library, with examples that are built and run in CI (`tests/GuideSamples`). The notes one level up in [docs](../README.md) are the normative specifications; this guide is for people who want to run a model, or train one, today.

| If you want to | Read |
|---|---|
| Run an ONNX model in a few lines, then with options, named tensors and limits | [getting-started.md](getting-started.md) |
| Work with tensors: creation, views, arithmetic, who owns the memory, scopes | [tensors.md](tensors.md) |
| Know what can go wrong and what to catch | [errors.md](errors.md) |
| Save a model in the OneModel format, or tell what a file is | [format.md](format.md) |
| Train a model, checkpoint it and resume | [training.md](training.md) |
| Turn a model written in C# into a graph that runs and saves like an imported one | [tracing.md](tracing.md) |
| Move code over from Microsoft.ML.OnnxRuntime | [migrating-from-onnxruntime.md](migrating-from-onnxruntime.md) |
| Move code over from TorchSharp | [migrating-from-torchsharp.md](migrating-from-torchsharp.md) |

## The three layers

You use the lowest one that does the job.

| Layer | Types | Namespace |
|---|---|---|
| Happy path: load a file, pass arrays, get an array | `Model` | `RecursiveAI.OneModel` |
| Graph layer: sessions, options, named tensors, the imported graph | `Session`, `ModelOptions`, `ModelLoadOptions`, `SessionOutputs`, `IrModel`, `OnnxImporter`, `ModelFormat`, `Tracer` | `RecursiveAI.OneModel.Graph`, `.Onnx`, `.Format`, `.Tracing` |
| Tensor layer: tensors, differentiable operations, scopes (and, in `RecursiveAI.OneModel.Training`, modules, optimizers, losses, checkpoints) | `Tensor`, `Ops`, `ActivationScope`, `TensorCompute` | `RecursiveAI.OneModel.Tensors` |

`ArithmeticMode`, `ModelLoadOptions` and the base exception `OneModelException` live in the root namespace `RecursiveAI.OneModel`, which is also the one `Model` is in. The examples assume `using` lines for the namespaces they use.

## Install

    dotnet add package RecursiveAI.OneModel
    dotnet add package RecursiveAI.OneModel.Training      # only to train

Both packages reference nothing but the .NET shared framework (net10.0). They run unchanged on Linux, macOS and Windows, x64 and arm64, and in a browser (WebAssembly). Under NativeAOT see the note in the [README](../../README.md).
