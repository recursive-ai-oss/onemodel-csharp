using RecursiveAI.OneModel;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Guide;

// Each #region below is copied into docs/guide by tools/sync_guide.py. `modelPath` is a small ONNX model with one float input
// "x" of shape [N, 16] and one output "y" of shape [N, 2, 4].
internal static class GettingStarted
{
    public static void HappyPath(string modelPath)
    {
        #region happy-path
        using var model = Model.Load(modelPath);
        float[] one = model.Run(new float[16]);     // one row of 16 values; the batch size comes from the length
        float[] two = model.Run(new float[32]);     // two rows
        #endregion

        Check.That(one.Length == 8 && two.Length == 16, "happy path output sizes");

        #region happy-path-shape
        float[] values = model.Run<float, float>(new float[32], default, out Shape outputShape);
        Console.WriteLine(outputShape);             // [2,2,4]
        #endregion

        Check.That(values.Length == outputShape.Length, "output shape");
    }

    public static void Options(string modelPath)
    {
        #region options
        var options = new ModelOptions
        {
            Mode = ArithmeticMode.Exact,            // bit-identical results on every machine (fails at load if an operator has no exact kernel)
            IntraOpThreads = 1,                     // 0 (the default) means one thread per core
        };
        using var exact = Model.Load(modelPath, options);
        #endregion

        Check.That(exact.Run(new float[16]).Length == 8, "exact model runs");
    }

    public static void NamedTensors(string modelPath)
    {
        #region session-tensors
        using var session = Session.Load(modelPath);
        Console.WriteLine(session.Inputs[0]);       // x: F32[N, 16]

        using Tensor x = Tensor.FromSpan<float>(new float[32], 2, 16);
        using SessionOutputs outputs = session.Run(new Dictionary<string, Tensor> { ["x"] = x });

        Tensor y = outputs["y"];                    // valid until `outputs` is disposed
        float[] values = y.ToArray<float>();        // copy out what you want to keep
        #endregion

        Check.That(values.Length == 16 && y.Shape == Shape.Create(2, 2, 4), "named run");
    }

    public static void ReuseBuffers(string modelPath)
    {
        using var session = Session.Load(modelPath);

        #region allocation-free
        float[] input = new float[32];
        float[] result = new float[16];
        Shape shape = session.Run("x", input, [2, 16], "y", result);   // copies the output into `result`; allocates nothing once warm
        #endregion

        Check.That(shape == Shape.Create(2, 2, 4), "span run");

        #region reuse-outputs
        using Tensor x = Tensor.FromSpan<float>(input, 2, 16);
        var outs = new Tensor[1];                   // an empty (default) slot is filled with a new tensor you own
        session.Run(["x"], [x], ["y"], outs);
        session.Run(["x"], [x], ["y"], outs);       // a slot of the right type and shape is overwritten in place
        outs[0].Dispose();
        #endregion
    }

    public static void Limits(string modelPath)
    {
        #region load-limits
        using var stream = File.OpenRead(modelPath);
        var limits = new ModelLoadOptions
        {
            MaxModelBytes = 64L << 20,              // reading stops with an error past this size, even on an endless stream
            MaxNesting = 32,                        // the deepest nesting of messages accepted
        };
        using var model = Model.Load(stream, loadOptions: limits);
        #endregion

        Check.That(model.Run(new float[16]).Length == 8, "limited load");
    }

    public static void Errors(string badModelPath, string missingPath)
    {
        string? report = null;
        bool missing = false;

        #region errors
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
        #endregion

        try
        {
            using var model = Model.Load(missingPath);
        }
        catch (FileNotFoundException)
        {
            missing = true;
        }

        Check.That(report is not null && missing, "errors: " + report);
    }
}
