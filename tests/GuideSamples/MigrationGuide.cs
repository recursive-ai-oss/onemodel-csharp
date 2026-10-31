using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Guide;

// The OneModel side of the migration pages. The ONNX Runtime and TorchSharp sides are quoted in the pages as text only.
internal static class MigrationGuide
{
    public static void FromOnnxRuntime(string modelPath)
    {
        float[] data = new float[32];

        #region migrate-ort
        using var session = Session.Load(modelPath, new ModelOptions { IntraOpThreads = 4 });

        using Tensor x = Tensor.FromSpan<float>(data, 2, 16);
        using SessionOutputs results = session.Run(new Dictionary<string, Tensor> { ["x"] = x });

        float[] y = results["y"].ToArray<float>();
        #endregion

        Check.That(y.Length == 16, "migrate ort");

        #region migrate-ort-names
        foreach (TensorInfo input in session.Inputs)
        {
            Console.WriteLine($"{input.Name} {input.DType} rank {input.Rank}");   // names, element types and declared shapes
        }
        #endregion
    }

    public static void FromTorchSharp()
    {
        #region migrate-torchsharp
        using Tensor x = Tensor.FromSpan<float>([1, 2, 3, 4], 2, 2);

        float total;
        using (new ActivationScope())               // plays the role of torch.NewDisposeScope()
        {
            Tensor y = Ops.Relu(x * 2f - 3f);       // Ops for functions, operators for arithmetic
            total = Ops.Sum(y).Item<float>();       // .item<float>() is .Item<float>()
        }
        #endregion

        Check.That(total == 1f + 3f + 5f, $"migrate torchsharp ({total})");
    }
}
