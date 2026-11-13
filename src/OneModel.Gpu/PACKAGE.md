# RecursiveAI.OneModel.Gpu

Runs [RecursiveAI.OneModel](https://www.nuget.org/packages/RecursiveAI.OneModel) models on the GPU: Vulkan on Windows and Linux, Metal on
Apple silicon, WebGPU in the browser, all from managed code through the platform's own graphics API. No native binaries ship in the
package and no CUDA is needed.

```csharp
using RecursiveAI.OneModel.Gpu;
using RecursiveAI.OneModel.Graph;

OneModelGpu.Register();                                    // once per process
using var session = Session.Load("model.onnx", new ModelOptions { Device = ComputeDevice.Gpu });
var outputs = session.Run(inputs);
var placement = session.GetDiagnostics(inputs);            // which operators ran where, and why not
```

Operators the GPU has a kernel for run there; the rest stay on the CPU, and `ModelOptions.RequireGpu` makes that an error that names every
operator. Exact mode is reproducible on one device and driver, not across machines. Supported operators, placement rules and speed against ONNX Runtime's GPU providers: [docs/gpu.md](https://github.com/recursive-ai-oss/onemodel-csharp/blob/main/docs/gpu.md) and [docs/bench.md](https://github.com/recursive-ai-oss/onemodel-csharp/blob/main/docs/bench.md). Source and issues: <https://github.com/recursive-ai-oss/onemodel-csharp>.

Licensed `MIT OR Apache-2.0`.
