# Getting started

The examples below use a small model with one float input `x` of shape `[N, 16]` and one output `y` of shape `[N, 2, 4]`; `modelPath` is its file.

## Run a model

<!-- sample: happy-path -->
```csharp
using var model = Model.Load(modelPath);
float[] one = model.Run(new float[16]);     // one row of 16 values; the batch size comes from the length
float[] two = model.Run(new float[32]);     // two rows
```

`Model.Load` reads an ONNX file or a file in the [OneModel format](format.md) (it looks at the first eight bytes). Names, layouts and the batch size come from the model, one thread per core is used inside operators, and results never depend on the thread count. `Run` is safe to call from many threads at once.

`Run` takes `float[]`, `long[]` or `int[]` (token ids) in row-major order and returns `float[]`. It also accepts an explicit shape, and can report the output's shape:

<!-- sample: happy-path-shape -->
```csharp
float[] values = model.Run<float, float>(new float[32], default, out Shape outputShape);
Console.WriteLine(outputShape);             // [2,2,4]
```

A text encoder with the usual three inputs (`input_ids`, `attention_mask`, `token_type_ids`) works the same way: your values feed the first input, an `attention_mask` input is filled with ones and a `token_type_ids` input with zeros. Any other extra input needs the session API below.

## Options

<!-- sample: options -->
```csharp
var options = new ModelOptions
{
    Mode = ArithmeticMode.Exact,            // bit-identical results on every machine (fails at load if an operator has no exact kernel)
    IntraOpThreads = 1,                     // 0 (the default) means one thread per core
};
using var exact = Model.Load(modelPath, options);
```

`ModelOptions` is fixed when the model is loaded. The properties:

| Property | Default | Meaning |
|---|---|---|
| `Mode` | `Fast` | `ArithmeticMode.Exact` gives bit-identical results on every OS, CPU and thread count, at some cost in speed ([exact mode](../exact-mode.md)). Loading fails and names the operators if one has no exact kernel. |
| `IntraOpThreads` | 0 (one per core) | The most threads one run uses inside an operator. 1 runs on the calling thread. Never changes a result. |
| `OptimizationLevel` | `All` | `None`, `Basic` (constant folding) or `All` (weight packing, fusion). |
| `WeightStorage` | `Original` | `Binary16` or `BFloat16` keeps large weights in half precision: less memory, widened for each use. |
| `MaxCachedPlans` | 16 | How many input-shape plans a session keeps. |

## Several inputs and outputs: sessions and tensors

<!-- sample: session-tensors -->
```csharp
using var session = Session.Load(modelPath);
Console.WriteLine(session.Inputs[0]);       // x: F32[N, 16]

using Tensor x = Tensor.FromSpan<float>(new float[32], 2, 16);
using SessionOutputs outputs = session.Run(new Dictionary<string, Tensor> { ["x"] = x });

Tensor y = outputs["y"];                    // valid until `outputs` is disposed
float[] values = y.ToArray<float>();        // copy out what you want to keep
```

`Session.Run` takes the inputs by name and returns a `SessionOutputs` that owns the output tensors: one `using` releases them all. Ask for particular outputs, in a particular order, with `session.Run(inputs, ["y"])`. Input names, element types and declared shapes are in `session.Inputs` and `session.Outputs`. A wrong name, type or shape throws `SessionException` (kind `InvalidInput`) and the message lists the valid names. Sessions are immutable once created and `Run` can be called from many threads at once.

The tensors in the result are valid until the container is disposed; copy values out with `ToArray<T>()`, or `Clone()` a tensor to keep it. [tensors.md](tensors.md) has the ownership rule in full.

## Without allocating

For a model with one input, `Run` can write straight into your buffers. After the first call with a shape, it allocates nothing:

<!-- sample: allocation-free -->
```csharp
float[] input = new float[32];
float[] result = new float[16];
Shape shape = session.Run("x", input, [2, 16], "y", result);   // copies the output into `result`; allocates nothing once warm
```

Tensors work the same way: pass an array of default (empty) tensors as the outputs and the session fills them; pass tensors of the right type and shape and it overwrites them in place:

<!-- sample: reuse-outputs -->
```csharp
using Tensor x = Tensor.FromSpan<float>(input, 2, 16);
var outs = new Tensor[1];                   // an empty (default) slot is filled with a new tensor you own
session.Run(["x"], [x], ["y"], outs);
session.Run(["x"], [x], ["y"], outs);       // a slot of the right type and shape is overwritten in place
outs[0].Dispose();
```

## Files you do not trust

<!-- sample: load-limits -->
```csharp
using var stream = File.OpenRead(modelPath);
var limits = new ModelLoadOptions
{
    MaxModelBytes = 64L << 20,              // reading stops with an error past this size, even on an endless stream
    MaxNesting = 32,                        // the deepest nesting of messages accepted
};
using var model = Model.Load(stream, loadOptions: limits);
```

`ModelLoadOptions` also sets `MemoryMap` (default on: weights stay in the page cache until used) and `ExternalDataDirectory` (where ONNX external-data files are looked for; for a path it is the model's directory, for a stream there is none unless you set one). Input is checked before it is used: lengths against what remains, nesting, shape arithmetic, and external-data paths, which must stay inside the model's directory. Bad input is a typed error, never a crash. `Session.Load`, `OnnxImporter.Load` and `ModelFormat.Load` take the same options.

## Errors

<!-- sample: errors -->
```csharp
try
{
    using var model = Model.Load(badModelPath);
}
catch (FileNotFoundException)
{
    // A file the operating system cannot open is the BCL's own exception, never wrapped.
}
catch (OneModelException e)
{
    // Everything the library rejects about the data it was given: a corrupt model, an unsupported operator,
    // a wrong input shape. e.Category says which part; the specific Kind is on the derived type.
    report = $"{e.Category}: {e.Message}";
}
```

The full rule is in [errors.md](errors.md).

## Where next

Speed and memory notes are in the [graph runtime](../graph-runtime.md) notes; the operators supported, opset by opset, are in the [conformance table](../conformance.md). A model that uses an operator this library does not have fails at load with an exception that lists every missing operator.
