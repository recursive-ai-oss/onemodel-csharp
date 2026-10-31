# Coming from TorchSharp

OneModel's tensor layer plays the role of `torch.Tensor` for CPU training and inference of the usual layers, in managed code. This page maps the TorchSharp API onto OneModel's. The TorchSharp snippets are plain text for comparison (not built here); the OneModel snippets are built and run in CI.

## The same expression in both

TorchSharp:

```text
float total;
using (torch.NewDisposeScope())
{
    var x = torch.tensor(new float[] { 1, 2, 3, 4 }, new long[] { 2, 2 });
    var y = (x * 2 - 3).relu();
    total = y.sum().item<float>();
}
```

OneModel:

<!-- sample: migrate-torchsharp -->
```csharp
using Tensor x = Tensor.FromSpan<float>([1, 2, 3, 4], 2, 2);

float total;
using (new ActivationScope())               // plays the role of torch.NewDisposeScope()
{
    Tensor y = Ops.Relu(x * 2f - 3f);       // Ops for functions, operators for arithmetic
    total = Ops.Sum(y).Item<float>();       // .item<float>() is .Item<float>()
}
```

## Mapping

| TorchSharp | OneModel | Notes |
|---|---|---|
| `torch.tensor(data, shape)` | `Tensor.FromSpan<T>(data, dims)` | Dimensions are `int`, as `params`. |
| `torch.zeros(shape)` | `Tensor.Zeros(DType.F32, dims)` | |
| `x + y`, `x * 2`, `-x` | the same operators | Differentiable. `*` is element-wise. |
| `torch.matmul`, `x.relu()`, `x.softmax(-1)`, `x.sum()` | `Ops.MatMul`, `Ops.Relu`, `Ops.Softmax`, `Ops.Sum` | Functions on `Ops` rather than methods. |
| `x.item<float>()` | `x.Item<float>()` | |
| `x.data<float>().ToArray()` | `x.ToArray<float>()` | `Data<T>()` gives a span. |
| `x.requires_grad_(true)` | `x.WithRequiresGrad()` | A tensor is a handle, so the flag lives on the handle you get back; keep that one. |
| `loss.backward()`, `x.grad` | `loss.Backward()`, `x.Grad` | |
| `x.detach()` | `x.Detach()` | |
| `torch.no_grad()` | `GradMode.NoGrad()` | |
| `torch.NewDisposeScope()` | `new ActivationScope()` | Collects the tensors the `Ops` allocate. `ActivationScope.Keep(t)` is `MoveToOuterDisposeScope()`. |
| `x.Dispose()` twice | allowed | `Dispose` is idempotent and affects every copy of the handle. |
| `nn.Module<Tensor, Tensor>`, `RegisterComponents()` | `NnModule`, `RegisterModule("name", child)` | Registration is explicit (no reflection), which keeps NativeAOT and trimming working. |
| `nn.Linear`, `nn.Conv2d`, `nn.LayerNorm`, `nn.Embedding`, `nn.BatchNorm2d`, `nn.Dropout` | `Linear`, `Conv2d`, `LayerNorm`, `Embedding`, `BatchNorm2d`, `Dropout` | In `RecursiveAI.OneModel.Training`. |
| `optim.SGD`, `optim.Adam`, `optim.AdamW` | `Sgd`, `Adam`, `AdamW` | `optimizer.Step()`, `ZeroGrad()`. |
| `optim.lr_scheduler.StepLR`, `CosineAnnealingLR`, `LinearLR` | `StepLr`, `CosineLr`, `WarmupLr` | Pure functions of the step: `schedule.Apply(optimizer)`. No scheduler state to save or fast-forward when resuming. |
| `nn.functional.cross_entropy`, `mse_loss` | `Losses.CrossEntropy`, `Losses.Mse` | Also `NllLoss`, `L1`, `BceWithLogits`, `Ctc`. |
| `torch.nn.utils.clip_grad_norm_` | `GradClip.ClipNorm` | |
| `module.save(path)` / `load(path)`, optimizer files, step counters kept by hand | `Checkpoint.Save(path, model, optimizer, extra)` / `Checkpoint.Load(path, model, optimizer)` | One file with model, optimizer state and integer extras; a resumed run is bit-identical to an uninterrupted one. |
| `torch.manual_seed` | `new Generator(seed)` passed to the layers and to data loading | Explicit generators, same results on every machine. |
| (no equivalent) | `ModuleTracing.Trace`, `Tracer` | Turns a module into a graph that runs and saves like an imported ONNX model ([tracing.md](tracing.md)). |
| shape errors that abort the process | `TensorShapeException` | |

## What is not there

- Eager operations take float32 tensors. Integer tensors are for indices and labels; other element types are for model inputs and outputs.
- CPU only. Autograd covers the operators of the layers above (convolutions, normalization, attention, recurrent layers, CTC, resize), not every `torch` function; `docs/autograd.md` lists them.
- No dynamic control flow in traced graphs (tracing records a static graph and says so when it cannot).
- Higher-order derivatives are not recorded.
