# Tensors

The tensor layer is `Tensor` (a strided view over pooled memory), `Ops` (the differentiable operations), and `ActivationScope` (frees the temporaries of a computation). The sessions of the [graph layer](getting-started.md) use the same `Tensor` for inputs and outputs. The normative description is [tensors.md](../tensors.md).

## Creating and viewing

<!-- sample: tensor-basics -->
```csharp
using Tensor a = Tensor.FromSpan<float>([1, 2, 3, 4, 5, 6], 2, 3);   // copies the values; shape 2 x 3
using Tensor zeros = Tensor.Zeros(DType.F32, 2, 3);
using Tensor labels = Tensor.FromSpan<long>([0L, 1L], 2);

Tensor transposed = a.T();                  // a view: no copy, shares a's storage
Tensor firstRow = a.Select(0, 0);           // also a view
float corner = a.Get<float>(1, 2);          // 6
ReadOnlySpan<float> flat = a.Data<float>(); // row-major span over a contiguous tensor
using Tensor packed = transposed.Contiguous();   // a copy in row-major order
```

A `Tensor` has an element type (`DType`), a shape, strides and an offset. Slices, transposes, selections and broadcasts are views: they share storage and cost nothing. `Contiguous()` makes a row-major copy when you need one; `Data<T>()` needs a contiguous tensor and returns a span over it. The element types are the ONNX ones (`F32`, `F16`, `BF16`, `I8`, `U8`, `I32`, `I64`, ...). The eager `Ops` take float32 tensors; other types are for inputs, outputs and data movement.

## Who owns the memory

One rule:

> **A `Tensor` is a handle. Every copy of it, and every view or slice made from it, refers to the same storage. The storage is released when `Dispose` is called on any one of them, or when the arena or `ActivationScope` that owns it ends. `Dispose` can be called any number of times, on any handle. After release, reading or writing data through any handle throws `StaleTensorException`, an `ObjectDisposedException`.**

<!-- sample: ownership -->
```csharp
Tensor t = Tensor.Zeros(DType.F32, 4);
Tensor alias = t;                           // another handle to the same storage
Tensor view = t.Slice(0, 0, 2);             // a view: the same storage again

t.Dispose();                                // releases the storage for every handle
alias.Dispose();                            // harmless: Dispose can be called any number of times
bool stale = view.IsReleased;               // true; view.Data<float>() would now throw StaleTensorException
```

So the habits of ordinary C# are safe: you can `using` a tensor you were handed, pass it to a method that disposes it, or dispose it again yourself.

<!-- sample: ownership-using -->
```csharp
static float Sum(Tensor x)
{
    using (x)                               // a method may dispose what it was given; the caller's own Dispose is still fine
    {
        float s = 0;
        foreach (float v in x.Data<float>())
        {
            s += v;
        }

        return s;
    }
}

Tensor data = Tensor.FromSpan<float>([1, 2, 3], 3);
float total = Sum(data);                    // 6
data.Dispose();
```

What the rule does not do is keep memory alive for a handle after someone else disposed it: use after release is the one thing that throws, and `IsReleased` lets you ask first. Memory is never reused behind a stale handle's back: a handle remembers which use of the storage it belongs to, so a stale handle can neither read nor release the storage's next owner.

Who disposes what:

| You get a tensor from | It is owned by | You dispose it? |
|---|---|---|
| `Tensor.Create`, `Zeros`, `FromSpan`, `Clone`, `Contiguous` (when it copies) | you | yes, or let the GC reclaim it (the pool block is freed when the object is finalized) |
| `Session.Run` returning `SessionOutputs` | the `SessionOutputs` | dispose the container |
| `Session.Run` filling your `Tensor[]` | you | yes, each non-empty slot |
| an `Ops` function or an operator | you, or the open `ActivationScope` | see below |
| a view (`Slice`, `T()`, `Reshape` that does not copy, `Detach`) | its source | no |
| `module.Parameters()` | the module | no |

Forgetting to dispose is a slow leak, not a crash: an undisposed tensor's block is freed when the GC finalizes it. A steady-state loop should dispose or use a scope so it allocates nothing.

## Arithmetic and gradients

<!-- sample: operators -->
```csharp
using Tensor w = Tensor.FromSpan<float>([1, 2, 3, 4], 2, 2).WithRequiresGrad();   // track gradients for w

using (new ActivationScope())               // frees every intermediate tensor when the scope ends
{
    Tensor loss = Ops.Sum(w * w * 0.5f - w);    // operators are the differentiable Ops functions
    loss.Backward();
    Console.WriteLine(loss.Item<float>());      // Item reads a one-element tensor
}

float[] grad = w.Grad.ToArray<float>();     // d/dw (w^2/2 - w) = w - 1: 0, 1, 2, 3
```

`+ - * /` and unary `-` between tensors, and between a tensor and a `float`, are the differentiable `Ops.Add`, `Sub`, `Mul`, `Div` and `Neg`, with broadcasting. `*` is element-wise; the matrix product is `Ops.MatMul`. Everything else is a function on `Ops`: `Linear`, `Conv2d`, `Softmax`, `LayerNorm`, `Gelu`, `Sum`, `Mean`, the recurrent layers, `CtcLoss`, and more. `WithRequiresGrad()` marks a leaf; `loss.Backward()` fills `Grad` on every leaf. Gradients, what is saved for backward, and which operations are exact are in [autograd.md](../autograd.md). The eager operations take float32 only; integer tensors are for indices and labels.

## Scopes

An expression such as `Ops.Relu(x * 2f - 3f)` allocates several tensors you cannot name. An `ActivationScope` collects every tensor the operations allocate on the calling thread and releases them when it is disposed, so a training step or an evaluation allocates nothing in steady state. Scopes nest, and the innermost collects.

To keep one result past the end of a scope:

<!-- sample: keep -->
```csharp
Tensor kept;
using (new ActivationScope())
{
    Tensor scratch = Ops.Add(a, a);         // collected, released at the end of the scope
    kept = ActivationScope.Keep(Ops.Relu(scratch));   // moved out: yours to dispose
}

float[] result = kept.ToArray<float>();     // 2, 0
kept.Dispose();
```

`Keep` hands the tensor to the enclosing scope (or to you, when there is none). `Clone()` and `ToArray()` also make storage the scope does not track. `Detach()` is a view, so it ends with its source.

## Compute settings

`TensorCompute.Mode` (fast or exact arithmetic) and `TensorCompute.Threads` are process-wide defaults for the eager operations and their gradients. To give one job its own settings, open a scope:

<!-- sample: compute-scope -->
```csharp
using (TensorCompute.Use(ArithmeticMode.Exact, threads: 1))
{
    // Everything started from here, including awaited work, computes exactly and on one thread.
    using Tensor y = Ops.Mul(a, a);
}
// Other code in the process keeps its own setting; TensorCompute.Mode sets the process-wide default.
```

The scope follows `await` and threads started from it, and is invisible to other code, so two training jobs in one process can differ.
