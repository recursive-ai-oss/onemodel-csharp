# The OneModel format

A model can be saved in a simple file: a 64-byte header, a JSON manifest of the graph and the weights as raw little-endian bytes aligned to 64 bytes, so a large file is memory-mapped and each weight used in place. The same model always writes the same bytes. The layout is in [format.md](../format.md); the version is 1 and may still change before 1.0.

<!-- sample: format -->
```csharp
ModelFormat.Convert(onnxPath, oneModelPath);           // import once, save in the OneModel format

ModelFileKind kind = ModelFormat.Detect(oneModelPath); // Onnx, OneModel or Unknown, from the first bytes
using var model = Model.Load(oneModelPath);            // Model.Load and Session.Load read either format
```

What is saved is the imported graph as it was, before optimization, so a saved file optimizes and runs exactly like the original and gives the same outputs bit for bit. Weights that were in ONNX external-data files are copied into the one file. `ModelFormat.Convert` refuses to write over its own input, and reading and rewriting a file that is already in the format is allowed (it normalizes it).

To save a graph you built or imported yourself:

<!-- sample: format-save -->
```csharp
using IrModel imported = Onnx.OnnxImporter.Load(onnxPath);      // the graph layer's model
ModelFormat.Save(imported, Path.Combine(workDir, "saved.onemodel"));
```

`ModelFormat.Detect` looks only at the first bytes: a file detected as `Onnx` can still fail to import. `ModelFormat.Load` and `OnnxImporter.Load` give the graph layer's `IrModel` for `Session.Create`; `Model.Load` and `Session.Load` do both steps.
