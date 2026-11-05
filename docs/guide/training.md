# Training

Training lives in the package `RecursiveAI.OneModel.Training` (namespace `RecursiveAI.OneModel.Training`): modules, losses, optimizers, learning-rate schedules, gradient clipping, data loading and checkpoints, on the [tensor layer](tensors.md). The reference is [training.md](../training.md).

## A module

A model is a subclass of `NnModule`. Children are registered by name, which gives parameters stable dotted names and tells checkpoints where each one goes; nothing is found by reflection, so it works under NativeAOT and trimming.

<!-- sample: training-model -->
```csharp
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
```

## A loop

<!-- sample: training-loop -->
```csharp
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
```

The pieces:

- `Generator` is a seeded random source; the same seed gives the same weights, dropout masks and batch order on every machine.
- Optimizers are `Sgd`, `Adam` and `AdamW`. Their arithmetic is plain IEEE float32 with no fused multiply-add, so training steps are bit-identical in fast and exact mode.
- A learning-rate schedule (`ConstantLr`, `StepLr`, `CosineLr`, `WarmupLr`, `WarmupCosineLr`) is a pure function of the step count. `schedule.Apply(optimizer)` sets the rate for the optimizer's current step, so resuming needs no schedule state.
- `Losses` has `CrossEntropy` (with label smoothing and an ignore index), `NllLoss`, `Mse`, `L1`, `BceWithLogits` and `Ctc`.
- The `ActivationScope` returns the step's intermediate tensors to the pool, so a steady-state step allocates no tensor memory.
- `TensorCompute.Use(...)` ([tensors.md](tensors.md)) picks exact arithmetic or a thread limit for a run.

Data loading (`Data` namespace: datasets, deterministic samplers and a prefetching `DataLoader` whose batch at step *g* depends only on the seed and *g*) is described in [data-loading.md](../data-loading.md).

## Checkpoints

A checkpoint is one file holding the model's parameters and buffers, the optimizer's state and any integers you add (the epoch, the step). Saving then loading restores every bit, and a run resumed from a checkpoint matches the uninterrupted run exactly.

<!-- sample: checkpoint-save -->
```csharp
Checkpoint.Save(path, model, optimizer, new Dictionary<string, long> { ["step"] = optimizer.StepCount });
```

<!-- sample: checkpoint-resume -->
```csharp
using var resumedModel = new Classifier(new Generator(999));        // any weights: the checkpoint overwrites them
using var resumedOptimizer = new AdamW(resumedModel.Parameters(), learningRate: 1e-2f);
CheckpointInfo info = Checkpoint.Load(path, resumedModel, resumedOptimizer);
long resumedAt = info.Extra["step"];                                // 100; the optimizer's step count is restored too
```

Loading checks the whole file against the model before it changes anything: a file that does not fit (a missing parameter, a different shape or optimizer) throws `CheckpointException` with a precise `Kind` and message, and leaves the model and optimizer untouched. A missing file is the BCL's `FileNotFoundException`.

## Fine-tuning an ONNX model

A model that exists as an ONNX file does not have to be rewritten as modules to be fine-tuned. `OnnxModule` loads it with its float32 weights as parameters, named as in the file, so everything above applies: optimizers, clipping, schedules, checkpoints.

<!-- sample: finetune-load -->
```csharp
// An ONNX encoder with Int64 inputs input_ids and attention_mask, outputs hidden and pooled. Its float32 weights become
// parameters named as in the file; operators outside the supported set fail here, all of them named.
using OnnxModule model = OnnxModule.Load(modelPath);
string first = model.NamedParameters()[0].Name;
```

The loop is the one above, with the module's `Forward` taking the graph's inputs by name:

<!-- sample: finetune-loop -->
```csharp
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
```

The trained weights go back into the graph, which then runs in an ordinary session or as a file:

<!-- sample: finetune-serve -->
```csharp
using Session session = model.CreateSession();         // the same graph with the trained weights
model.Save(Path.Combine(workDir, "tuned.onemodel"));    // or keep it as a file: Model.Load reads it
```

The operators that can sit between a weight and the loss are listed in [training.md](../training.md) (section 11); a graph that needs another one fails at `Load`, naming all of them. Shape arithmetic, masks and position tables run on the same kernels as `Session` and carry no gradient. Dropout nodes in an exported graph run as the identity. Constants that feed `Sin` or `Cos` (rotary-embedding frequencies) stay frozen; `OnnxModuleOptions.IsTrainable` freezes more by name.
