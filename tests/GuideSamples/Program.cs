using RecursiveAI.OneModel.Guide;

// Runs every example from docs/guide. Usage: GuideSamples <directory holding mlp.onnx>
string graphs = args.Length > 0 ? args[0] : Path.Combine("testdata", "graphs");
string model = Path.GetFullPath(Path.Combine(graphs, "mlp.onnx"));
string work = Path.Combine(Path.GetTempPath(), "onemodel-guide-" + Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(work);
try
{
    string bad = Path.Combine(work, "bad.onnx");
    File.WriteAllBytes(bad, [0x0a, 0xff, 0xff, 0xff, 0xff, 0x0f, 1, 2, 3]);

    GettingStarted.HappyPath(model);
    GettingStarted.Options(model);
    GettingStarted.NamedTensors(model);
    GettingStarted.ReuseBuffers(model);
    GettingStarted.Limits(model);
    GettingStarted.Errors(bad, Path.Combine(work, "missing.onnx"));
    TensorGuide.Basics();
    TensorGuide.Ownership();
    TensorGuide.Arithmetic();
    TensorGuide.Scopes();
    FormatGuide.Convert(model, work);
    TrainingGuide.TrainAndResume(work);
    TrainingGuide.Trace(work);
    MigrationGuide.FromOnnxRuntime(model);
    MigrationGuide.FromTorchSharp();
    Console.WriteLine("guide samples ok");
}
finally
{
    Directory.Delete(work, recursive: true);
}

namespace RecursiveAI.OneModel.Guide
{
    internal static class Check
    {
        public static void That(bool condition, string what)
        {
            if (!condition)
            {
                throw new InvalidOperationException("Guide sample failed: " + what);
            }
        }

        public static void Throws<T>(Action action, string what) where T : Exception
        {
            try
            {
                action();
            }
            catch (T)
            {
                return;
            }

            throw new InvalidOperationException("Guide sample failed (no " + typeof(T).Name + "): " + what);
        }
    }
}
