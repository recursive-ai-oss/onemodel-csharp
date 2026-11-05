using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Tracing;
using RecursiveAI.OneModel.Training;

namespace RecursiveAI.OneModel.Guide;

#region training-model
internal sealed class Classifier : NnModule
{
    private readonly Linear _hidden;
    private readonly Linear _output;

    public Classifier(Generator generator)
    {
        // Children are registered by name, so parameters get stable names and checkpoints know where each one goes.
        _hidden = RegisterModule("hidden", new Linear(2, 16, generator));
        _output = RegisterModule("output", new Linear(16, 2, generator));
    }

    public override Tensor Forward(Tensor input) => _output.Forward(Ops.Relu(_hidden.Forward(input)));
}
#endregion

internal static class TrainingGuide
{
    // Points (x0, x1); the class is whether x0 > x1.
    private static (Tensor Inputs, Tensor Labels) Data()
    {
        var g = new Generator(7);
        var xs = new float[64 * 2];
        var ys = new long[64];
        for (int i = 0; i < 64; i++)
        {
            xs[2 * i] = g.Uniform(-1, 1);
            xs[2 * i + 1] = g.Uniform(-1, 1);
            ys[i] = xs[2 * i] > xs[2 * i + 1] ? 1 : 0;
        }

        return (Tensor.FromSpan<float>(xs, 64, 2), Tensor.FromSpan<long>(ys, 64));
    }

    public static void TrainAndResume(string workDir)
    {
        (Tensor inputs, Tensor labels) = Data();
        string path = Path.Combine(workDir, "run.ckpt");

        #region training-loop
        var generator = new Generator(1234);                    // seeded: the same seed gives the same weights
        using var model = new Classifier(generator);
        using var optimizer = new AdamW(model.Parameters(), learningRate: 1e-2f);
        var schedule = new CosineLr(baseLr: 1e-2f, totalSteps: 200);

        float loss = 0;
        for (int step = 0; step < 100; step++)
        {
            using var scope = new ActivationScope();            // returns this step's intermediate tensors to the pool
            schedule.Apply(optimizer);                          // the rate is a pure function of the step count
            optimizer.ZeroGrad();
            Tensor logits = model.Forward(inputs);
            Tensor lossTensor = Losses.CrossEntropy(logits, labels);
            lossTensor.Backward();
            GradClip.ClipNorm(model.Parameters(), maxNorm: 1f);
            optimizer.Step();
            loss = lossTensor.Item<float>();
        }
        #endregion

        Check.That(loss < 0.5f, $"training converges (loss {loss})");

        #region checkpoint-save
        Checkpoint.Save(path, model, optimizer, new Dictionary<string, long> { ["step"] = optimizer.StepCount });
        #endregion

        // Continue the run in memory ...
        float continued = Steps(model, optimizer, schedule, inputs, labels, 50);

        #region checkpoint-resume
        using var resumedModel = new Classifier(new Generator(999));        // any weights: the checkpoint overwrites them
        using var resumedOptimizer = new AdamW(resumedModel.Parameters(), learningRate: 1e-2f);
        CheckpointInfo info = Checkpoint.Load(path, resumedModel, resumedOptimizer);
        long resumedAt = info.Extra["step"];                                // 100; the optimizer's step count is restored too
        #endregion

        // ... and from the checkpoint: the same bits.
        float resumed = Steps(resumedModel, resumedOptimizer, schedule, inputs, labels, 50);
        Check.That(resumedAt == 100 && resumed == continued, $"resume equals uninterrupted ({resumed} vs {continued})");

        inputs.Dispose();
        labels.Dispose();
    }

    private static float Steps(NnModule model, Optimizer optimizer, LrSchedule schedule, Tensor inputs, Tensor labels, int count)
    {
        float loss = 0;
        for (int i = 0; i < count; i++)
        {
            using var scope = new ActivationScope();
            schedule.Apply(optimizer);
            optimizer.ZeroGrad();
            Tensor lossTensor = Losses.CrossEntropy(model.Forward(inputs), labels);
            lossTensor.Backward();
            optimizer.Step();
            loss = lossTensor.Item<float>();
        }

        return loss;
    }

    public static void Trace(string workDir)
    {
        using var model = new Classifier(new Generator(3));
        string path = Path.Combine(workDir, "classifier.onemodel");

        #region tracing
        using Tensor example = Tensor.Zeros(DType.F32, 4, 2);               // an example input fixes the shapes
        using TracedModel traced = model.Trace(example, "batch", null);     // "batch" stays symbolic (example size at least 2, and not equal to another axis); the second axis is fixed

        traced.Save(path);                                                  // OneModel format; traced.SaveOnnx(path) writes ONNX
        using var session = traced.CreateSession();                         // an ordinary Session over the recorded graph

        using Tensor x = Tensor.FromSpan<float>([0.5f, -0.5f, 1, 2, 3, 4], 3, 2);
        using SessionOutputs outputs = session.Run(new Dictionary<string, Tensor> { ["input"] = x });
        Tensor scores = outputs["output"];                                  // 3 x 2, equal to model.Forward(x)
        #endregion

        using var reloaded = Model.Load(path);                              // runs in any program, with no training package
        Check.That(scores.Shape == Shape.Create(3, 2), "traced output shape");
        float[] viaFile = reloaded.Run(x.ToArray<float>());
        Check.That(viaFile.AsSpan().SequenceEqual(scores.ToArray<float>()), "traced model from file equals session");
    }

    public static void FineTuneImported(string modelPath, string workDir)
    {
        #region finetune-load
        // An ONNX encoder with Int64 inputs input_ids and attention_mask, outputs hidden and pooled. Its float32 weights become
        // parameters named as in the file; operators outside the supported set fail here, all of them named.
        using OnnxModule model = OnnxModule.Load(modelPath);
        string first = model.NamedParameters()[0].Name;
        #endregion

        #region finetune-loop
        using var optimizer = new AdamW(model.Parameters(), learningRate: 1e-3f);
        using Tensor ids = Tensor.FromSpan<long>([3, 9, 4, 17, 2, 0, 0, 5, 11, 6, 8, 23, 1, 7], 2, 7);
        using Tensor mask = Tensor.FromSpan<long>([1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1], 2, 7);
        var inputs = new Dictionary<string, Tensor> { ["input_ids"] = ids, ["attention_mask"] = mask };
        using Tensor target = Tensor.Zeros(DType.F32, 2, 16);

        float firstLoss = 0, lastLoss = 0;
        for (int step = 0; step < 5; step++)
        {
            using var scope = new ActivationScope();
            optimizer.ZeroGrad();
            Tensor pooled = model.Forward(inputs)[1];            // outputs are in model.OutputNames order
            Tensor loss = Losses.Mse(pooled, target);
            loss.Backward();
            optimizer.Step();
            lastLoss = loss.Item<float>();
            firstLoss = step == 0 ? lastLoss : firstLoss;
        }
        #endregion

        #region finetune-serve
        using Session session = model.CreateSession();         // the same graph with the trained weights
        model.Save(Path.Combine(workDir, "tuned.onemodel"));    // or keep it as a file: Model.Load reads it
        #endregion

        Check.That(lastLoss < firstLoss, $"fine-tuning lowers the loss ({firstLoss} to {lastLoss})");
        Check.That(first.Length > 0 && session.Outputs.Length == 2, "the trained model runs in a session");
    }
}
