# OneModel

A neural-network runtime for .NET in pure managed C#, with no native binaries and no dependency beyond the shared framework.

- **RecursiveAI.OneModel:** ONNX import, graph optimization and SIMD CPU kernels (AVX2, AVX-512, NEON, portable SIMD) for inference in fp32, fp16/bf16 and int8; thread-safe sessions; an opt-in exact mode with bit-identical results on every platform; its own memory-mappable model format. NativeAOT-, trim- and WebAssembly-compatible.
- **RecursiveAI.OneModel.Training:** reverse-mode autodiff, modules, losses, SGD/Adam/AdamW, data loading, checkpoints, and tracing a trained module into a graph.

```csharp
using RecursiveAI.OneModel;

using var m = Model.Load("model.onnx");
float[] v = m.Run(tokens);   // names and shapes come from the model
```

This is a preview; the API may change between previews. Documentation, supported operators, source and issues: <https://github.com/recursive-ai-oss/onemodel-csharp>.

Licensed `MIT OR Apache-2.0`.
