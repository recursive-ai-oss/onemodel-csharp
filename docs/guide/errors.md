# Errors

The rule: **the library reports the data it was given with `OneModelException`; everything else is the .NET exception that fits.**

| What happened | What is thrown |
|---|---|
| A model file, stream or checkpoint is corrupt, unsupported or inconsistent; a model uses an operator that is missing; inputs do not fit the model; a module cannot be traced | a type derived from `OneModelException`: `OnnxImportException`, `ModelFormatException`, `SessionException`, `CheckpointException` (in the training package), `TraceException`. Each has a `Kind` enum for the cause. |
| A file is missing, a directory does not exist, access is denied, the disk is full, a stream fails | the BCL's own `FileNotFoundException`, `DirectoryNotFoundException`, `UnauthorizedAccessException`, `IOException`, never wrapped, from every method that opens a file or reads a stream |
| An argument is wrong (a negative dimension, a null path, a shape that does not fill the values, an option out of range) | `ArgumentException` and its subclasses (`TensorShapeException` is one) with the parameter name |
| A session, model or traced model is used after `Dispose`; a tensor's data is read after its storage was released | `ObjectDisposedException` (`StaleTensorException` for tensors) |
| An operation does not apply to the object's state (an empty tensor's `Item`, the gradient of an integer tensor) | `InvalidOperationException` |
| Cancellation through a `CancellationToken` | `OperationCanceledException` |

So loading a model needs at most:

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

`OneModelException` carries `Category` (which part of the library), and, where the fault is in a file, `InputName`, `Offset` (a byte offset, or -1) and `FieldPath` (for example `ModelProto.graph.node[3].attribute[1].t`); the message contains them too, with a very long path shortened. `SessionException.Operators` lists every unsupported operator at once.

Hostile input is survived: truncated or overlong data, bad lengths, absurd shapes, excessive nesting, external-data paths that leave the model's directory, and endless streams all end in one of the types above, never in `IndexOutOfRangeException`, `NullReferenceException`, `OverflowException` or `OutOfMemoryException`. Set tighter limits with `ModelLoadOptions` ([getting-started.md](getting-started.md)). The method and the fuzz targets are in the repository's `fuzz` folder.
