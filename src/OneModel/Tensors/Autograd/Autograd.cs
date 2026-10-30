using System;
using System.Collections.Generic;
using System.Threading;
namespace RecursiveAI.OneModel.Tensors;

/// <summary>The operation a <see cref="BackwardRecord"/> belongs to. A backward engine dispatches on this value through a static table.</summary>
public enum OpKind
{
    /// <summary>No operation (a default record).</summary>
    None = 0,
    /// <summary>Elementwise <c>a + b</c> with broadcasting.</summary>
    Add,
    /// <summary>Elementwise <c>a - b</c> with broadcasting.</summary>
    Sub,
    /// <summary>Elementwise <c>a * b</c> with broadcasting.</summary>
    Mul,
    /// <summary>Elementwise <c>a / b</c> with broadcasting.</summary>
    Div,
    /// <summary>Elementwise negation.</summary>
    Neg,
    /// <summary>Elementwise <c>max(x, 0)</c>.</summary>
    Relu,
    /// <summary>2-D matrix product <c>a[m,k] · b[k,n]</c>.</summary>
    MatMul,
    /// <summary>2-D max pooling over NCHW input, with argmax indices.</summary>
    MaxPool2d,
    /// <summary>Element type conversion.</summary>
    Cast,
    /// <summary>View: slice along one axis. Ints = [axis, start, end, step].</summary>
    Slice,
    /// <summary>View: axis permutation. Ints = the permutation.</summary>
    Permute,
    /// <summary>View: reshape (also squeeze and unsqueeze). The input shape is in <see cref="BackwardRecord.InputShape"/>.</summary>
    Reshape,
    /// <summary>View: broadcast expand. The gradient is the output gradient summed down to the input shape.</summary>
    Expand,
    /// <summary>View: reversal along one axis. Ints = [axis].</summary>
    Flip,
    /// <summary>View: one index along one axis, removing it. Ints = [axis, index].</summary>
    Select,
    /// <summary>GELU. Ints = [0 erf form, 1 tanh form]. Saves the input.</summary>
    Gelu,
    /// <summary>Logistic sigmoid. Saves the output.</summary>
    Sigmoid,
    /// <summary>Hyperbolic tangent. Saves the output.</summary>
    Tanh,
    /// <summary>Sum over all elements (Ints[0] = -1) or one axis (Ints = [axis, keepDim]).</summary>
    Sum,
    /// <summary>Mean over all elements (Ints[0] = -1) or one axis (Ints = [axis, keepDim]).</summary>
    Mean,
    /// <summary>Softmax over the last axis. Saves the output.</summary>
    Softmax,
    /// <summary>Log-softmax over the last axis. Saves the output.</summary>
    LogSoftmax,
    /// <summary>Layer normalization over the last axis. Inputs x, gamma, beta. Saves x, gamma, mean, 1/std. Floats = [epsilon].</summary>
    LayerNorm,
    /// <summary>Batch normalization of an NCHW tensor. Inputs x, gamma, beta. Saves x, gamma, mean, 1/std. Ints = [training].</summary>
    BatchNorm,
    /// <summary>2-D convolution. Inputs x, weight, bias. Saves x and weight.</summary>
    Conv2d,
    /// <summary>2-D average pooling. Ints = [kernelH, kernelW, strideH, strideW, padH, padW, countIncludePad].</summary>
    AvgPool2d,
    /// <summary>Concatenation along Ints[0].</summary>
    Concat,
    /// <summary>Row gather from a table. Saves the index tensor. Ints = [paddingIndex or -1].</summary>
    Embedding,
    /// <summary>Negative log-likelihood of class indices under log-probabilities. Ints = [mean, ignoreIndex].</summary>
    NllLoss,
    /// <summary>Scaled dot-product attention over [batch, heads, length, headDim]. Saves q, k, v and the mask (slots 0-3; the mask slot is empty without one). Floats = [scale].</summary>
    Attention,
}

/// <summary>
/// Autograd state attached to a tensor that takes part in differentiation: whether it needs a gradient, the record of the operation
/// that produced it (null for a leaf), and its accumulated gradient. Tensors that do not require gradients have none.
/// </summary>
public sealed class AutogradMeta
{
    internal AutogradMeta(bool requiresGrad, BackwardRecord? gradFn, int outputIndex)
    {
        RequiresGrad = requiresGrad;
        GradFn = gradFn;
        OutputIndex = outputIndex;
    }

    /// <summary>True if gradients flow to this tensor.</summary>
    public bool RequiresGrad { get; }

    /// <summary>The record of the operation that produced this tensor; null for a leaf (a tensor made by the user).</summary>
    public BackwardRecord? GradFn { get; }

    /// <summary>Which output of <see cref="GradFn"/> this tensor is (0 for single-output operations).</summary>
    public int OutputIndex { get; }

    /// <summary>True when <see cref="GradFn"/> is null.</summary>
    public bool IsLeaf => GradFn is null;

    /// <summary>The accumulated gradient of a leaf; empty until a backward pass writes it. The engine owns the accumulation.</summary>
    public Tensor Grad { get; set; }
}

/// <summary>
/// Switches recording of backward information. Recording is on by default and happens only when an operation has an input that
/// requires gradients; <see cref="NoGrad"/> turns it off for the current thread, for inference or for optimizer updates. The setting
/// is per thread (not per async flow).
/// </summary>
public static class GradMode
{
    [ThreadStatic]
    private static bool s_disabled;

    /// <summary>True when operations on this thread may record backward information.</summary>
    public static bool IsEnabled => !s_disabled;

    /// <summary>Disables recording on this thread until the returned scope is disposed.</summary>
    public static NoGradScope NoGrad()
    {
        bool prev = s_disabled;
        s_disabled = true;
        return new NoGradScope(prev);
    }

    internal static void Restore(bool wasDisabled) => s_disabled = wasDisabled;

    /// <summary>The cheap test every eager operation makes: record only when enabled and some input requires gradients.</summary>
    public static bool ShouldRecord(in Tensor a) => !s_disabled && a.RequiresGrad;

    /// <summary>As <see cref="ShouldRecord(in Tensor)"/> for two inputs.</summary>
    public static bool ShouldRecord(in Tensor a, in Tensor b) => !s_disabled && (a.RequiresGrad || b.RequiresGrad);
}

/// <summary>Restores the previous <see cref="GradMode"/> setting when disposed.</summary>
public readonly struct NoGradScope : IDisposable
{
    private readonly bool _previouslyDisabled;

    internal NoGradScope(bool previouslyDisabled) => _previouslyDisabled = previouslyDisabled;

    /// <inheritdoc/>
    public void Dispose() => GradMode.Restore(_previouslyDisabled);
}

/// <summary>A saved tensor was modified in place after the forward pass saved it, so backward would use the wrong values.</summary>
public sealed class TensorVersionException : InvalidOperationException
{
    /// <summary>Creates the exception.</summary>
    public TensorVersionException(string message) : base(message) { }
}

/// <summary>
/// A tensor kept for the backward pass together with the version its storage had when it was saved. <see cref="Unpack"/> fails if
/// any in-place operation touched the storage since.
/// </summary>
public readonly struct SavedTensor
{
    internal SavedTensor(in Tensor value, bool owned)
    {
        Value = value;
        Version = value.Version;
        Owned = owned;
    }

    /// <summary>The saved tensor (possibly a view of a larger one). No version check; prefer <see cref="Unpack"/>.</summary>
    public Tensor Value { get; }

    /// <summary>The storage version at the time of saving.</summary>
    public long Version { get; }

    /// <summary>True when the record created the tensor (an index tensor or intermediate) and frees it in <see cref="BackwardRecord.Release"/>.</summary>
    public bool Owned { get; }

    /// <summary>True when nothing was saved in this slot (the operation did not need it).</summary>
    public bool IsEmpty => Value.IsEmpty;

    /// <summary>Returns the saved tensor after checking that its storage was not modified in place since it was saved.</summary>
    /// <exception cref="InvalidOperationException">Nothing was saved in this slot.</exception>
    /// <exception cref="TensorVersionException">The storage version changed.</exception>
    public Tensor Unpack()
    {
        if (Value.IsEmpty)
        {
            throw new InvalidOperationException("Nothing was saved in this slot.");
        }

        long now = Value.Version;
        if (now != Version)
        {
            throw new TensorVersionException(
                $"A tensor saved for the backward pass was modified in place (saved at version {Version}, now {now}).");
        }

        return Value;
    }
}
