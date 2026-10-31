# Coming from Microsoft.ML.OnnxRuntime

OneModel runs the same ONNX files, in managed code, with no native library. This page maps the ONNX Runtime C# API onto OneModel's. The ONNX Runtime snippets are shown as plain text for comparison (they are not built here); the OneModel snippets are built and run in CI.

## The same job in both

ONNX Runtime:

```text
using var options = new SessionOptions { IntraOpNumThreads = 4 };
using var session = new InferenceSession("model.onnx", options);

var data = new float[32];
using var x = OrtValue.CreateTensorValueFromMemory(data, [2, 16]);
var inputs = new Dictionary<string, OrtValue> { ["x"] = x };
using var runOptions = new RunOptions();
using IDisposableReadOnlyCollection<OrtValue> results = session.Run(runOptions, inputs, session.OutputNames);

float[] y = results[0].GetTensorDataAsSpan<float>().ToArray();
```

OneModel:

<!-- sample: migrate-ort -->
```csharp
using var session = Session.Load(modelPath, new ModelOptions { IntraOpThreads = 4 });

using Tensor x = Tensor.FromSpan<float>(data, 2, 16);
using SessionOutputs results = session.Run(new Dictionary<string, Tensor> { ["x"] = x });

float[] y = results["y"].ToArray<float>();
```

For the common case of one array in and one array out, `Model` is shorter still:

```text
using var model = Model.Load("model.onnx");
float[] y = model.Run(data);
```

## Mapping

| ONNX Runtime | OneModel | Notes |
|---|---|---|
| `new InferenceSession(path \| bytes, options)` | `Session.Load(path \| stream, options)` | OneModel also reads its own format; the file kind is detected. |
| `SessionOptions` | `ModelOptions` | Renamed so both libraries can be used in one file without aliases. Immutable once built (`init` properties). |
| `SessionOptions.IntraOpNumThreads` | `ModelOptions.IntraOpThreads` | 0 means one per core. There is no inter-op thread setting: one `Run` is one graph walk, and many threads can call `Run` at once. |
| `GraphOptimizationLevel` | `ModelOptions.OptimizationLevel` | `None`, `Basic`, `All`. |
| `ExecutionMode` (`ORT_SEQUENTIAL`/`ORT_PARALLEL`) | not needed | OneModel's `ArithmeticMode` (`Fast`/`Exact`) is a different thing: exact gives bit-identical results everywhere. |
| execution providers | none | CPU only. |
| `OrtValue` / `DenseTensor<T>` / `NamedOnnxValue` | `Tensor` | One tensor type for inputs, outputs and eager computation. |
| `OrtValue.CreateTensorValueFromMemory(data, shape)` | `Tensor.FromSpan<T>(data, dims)` | Copies the data. For no copy at all use the span overload of `Run`. |
| `session.Run(runOptions, inputs, outputNames)` returning `IDisposableReadOnlyCollection<OrtValue>` | `session.Run(inputs, outputNames)` returning `SessionOutputs` | Dispose the result. `outputNames` may be omitted to get every output. |
| parallel arrays of names and values | `session.Run(inputNames, inputs, outputNames, outputs)` | The allocation-free form. |
| `RunOptions` (tag, termination) | `CancellationToken` on load | A run is short and not cancellable midway. |
| `session.InputMetadata` / `OutputMetadata` | `session.Inputs` / `session.Outputs` | `TensorInfo`: name, element type, declared shape (`Dims`, symbolic extents have a name). |
| `OnnxRuntimeException` | `OneModelException` and subclasses | The kind is a property (`Kind`), not text in the message. File errors are the BCL's own. |
| `GetTensorDataAsSpan<T>()` | `Data<T>()` | Contiguous tensors only; `ToArray<T>()` copies. |
| `session.Dispose()` then `Run` crashes the process | `ObjectDisposedException` | |
| `OrtEnv`, logging | none | No global environment. |
| custom operators | not supported yet | A model with an unknown operator fails at load, naming every one. |
| `SessionOptions.OptimizedModelFilePath` | `ModelFormat.Convert` | Saves the imported graph in the OneModel format. |

Names to remember when two libraries meet: ONNX Runtime's `SessionOptions` and `ExecutionMode` do not exist here (they are `ModelOptions` and `ArithmeticMode`), so `using Microsoft.ML.OnnxRuntime;` and `using RecursiveAI.OneModel.Graph;` can sit in the same file.

## Looking at the model

<!-- sample: migrate-ort-names -->
```csharp
foreach (TensorInfo input in session.Inputs)
{
    Console.WriteLine($"{input.Name} {input.DType} rank {input.Rank}");   // names, element types and declared shapes
}
```

## Things that behave differently

- **Ownership.** A `Tensor` you get from `SessionOutputs` is owned by the container. Disposing a tensor is always safe and idempotent; see [tensors.md](tensors.md).
- **Shapes.** Declared shapes with symbolic extents are checked, and a model can be run at any size that fits them. Each distinct set of input shapes builds an execution plan once and caches it.
- **Determinism.** The default `Fast` mode gives the same output on one machine whatever the thread count; `Exact` gives the same bits everywhere.
- **Errors for missing inputs** are raised before anything runs, with the input names the model has.
