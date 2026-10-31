# Tracing

Tracing turns a model written in C# (an `NnModule`, or any function on `Ops`) into an ordinary graph: one run on an example input records every operation, and the result runs in a `Session` with the usual optimizations, saves in the [OneModel format](format.md), and can be written as ONNX.

<!-- sample: tracing -->
```csharp
using Tensor example = Tensor.Zeros(DType.F32, 4, 2);               // an example input fixes the shapes
using TracedModel traced = model.Trace(example, "batch", null);     // "batch" stays symbolic (example size at least 2, and not equal to another axis); the second axis is fixed

traced.Save(path);                                                  // OneModel format; traced.SaveOnnx(path) writes ONNX
using var session = traced.CreateSession();                         // an ordinary Session over the recorded graph

using Tensor x = Tensor.FromSpan<float>([0.5f, -0.5f, 1, 2, 3, 4], 3, 2);
using SessionOutputs outputs = session.Run(new Dictionary<string, Tensor> { ["input"] = x });
Tensor scores = outputs["output"];                                  // 3 x 2, equal to model.Forward(x)
```

Details:

- The recorded graph has the inputs and outputs you name (`input` and `output` for `NnModule.Trace`; use `Tracer` directly for other names or several inputs).
- Axes you name stay symbolic (a batch or sequence length); others are fixed to the example's size. Symbolic example sizes must be at least 2 and different from each other so the tracer can tell them apart.
- Tracing records a static graph. Reading a traced tensor's data, changing it in place, `GradMode.NoGrad()` and operations without a graph equivalent throw `TraceException` naming the operation, because anything decided from the example's values (a branch, a loop count) would be frozen into the graph. Losses, dropout in training mode and `CtcLoss` cannot be traced.
- `NnModule.Trace` puts the module in evaluation mode while it records and restores the mode afterwards.
- The traced session typically runs faster than the eager code, since the graph passes apply to it.

The operator mapping and the exact rules are in [tracing.md](../tracing.md).
