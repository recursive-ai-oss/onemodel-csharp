using RecursiveAI.OneModel.Format;
using RecursiveAI.OneModel.Graph;

namespace RecursiveAI.OneModel.Guide;

internal static class FormatGuide
{
    public static void Convert(string onnxPath, string workDir)
    {
        string oneModelPath = Path.Combine(workDir, "model.onemodel");

        #region format
        ModelFormat.Convert(onnxPath, oneModelPath);           // import once, save in the OneModel format

        ModelFileKind kind = ModelFormat.Detect(oneModelPath); // Onnx, OneModel or Unknown, from the first bytes
        using var model = Model.Load(oneModelPath);            // Model.Load and Session.Load read either format
        #endregion

        Check.That(kind == ModelFileKind.OneModel, "detect");
        Check.That(model.Run(new float[16]).Length == 8, "format load");
        Check.That(ModelFormat.Detect(onnxPath) == ModelFileKind.Onnx, "detect onnx");

        // Check the file really is the same model: outputs are bit-identical.
        using var original = Model.Load(onnxPath);
        float[] x = Enumerable.Range(0, 16).Select(i => i * 0.1f).ToArray();
        Check.That(original.Run(x).AsSpan().SequenceEqual(model.Run(x)), "format outputs equal");

        #region format-save
        using IrModel imported = Onnx.OnnxImporter.Load(onnxPath);      // the graph layer's model
        ModelFormat.Save(imported, Path.Combine(workDir, "saved.onemodel"));
        #endregion
    }
}
