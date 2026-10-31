using System;

namespace RecursiveAI.OneModel.Tensors;

public readonly partial struct Tensor
{
    // ---- arithmetic operators ---------------------------------------------------------------------------------------------
    // They are the differentiable Ops functions with the default pool and kernels, so they record for backward exactly as Ops.Add
    // does. The result is a new tensor the caller owns (or the open ActivationScope collects). A scalar operand becomes a rank-0
    // float32 tensor that an open ActivationScope also collects; without a scope it is returned to the pool when the GC finalizes it.

    /// <summary><c>a + b</c> with broadcasting; see <see cref="Ops.Add"/>.</summary>
    public static Tensor operator +(in Tensor a, in Tensor b) => Ops.Add(a, b);

    /// <summary><c>a + s</c> for a scalar <paramref name="s"/>.</summary>
    public static Tensor operator +(in Tensor a, float s) => Ops.Add(a, ScalarOperand(s));

    /// <summary><c>s + a</c> for a scalar <paramref name="s"/>.</summary>
    public static Tensor operator +(float s, in Tensor a) => Ops.Add(ScalarOperand(s), a);

    /// <summary><c>a - b</c> with broadcasting; see <see cref="Ops.Sub"/>.</summary>
    public static Tensor operator -(in Tensor a, in Tensor b) => Ops.Sub(a, b);

    /// <summary><c>a - s</c> for a scalar <paramref name="s"/>.</summary>
    public static Tensor operator -(in Tensor a, float s) => Ops.Sub(a, ScalarOperand(s));

    /// <summary><c>s - a</c> for a scalar <paramref name="s"/>.</summary>
    public static Tensor operator -(float s, in Tensor a) => Ops.Sub(ScalarOperand(s), a);

    /// <summary><c>-a</c>; see <see cref="Ops.Neg"/>.</summary>
    public static Tensor operator -(in Tensor a) => Ops.Neg(a);

    /// <summary>Element-wise <c>a * b</c> with broadcasting (not a matrix product; that is <see cref="Ops.MatMul"/>); see <see cref="Ops.Mul"/>.</summary>
    public static Tensor operator *(in Tensor a, in Tensor b) => Ops.Mul(a, b);

    /// <summary><c>a * s</c> for a scalar <paramref name="s"/>.</summary>
    public static Tensor operator *(in Tensor a, float s) => Ops.Mul(a, ScalarOperand(s));

    /// <summary><c>s * a</c> for a scalar <paramref name="s"/>.</summary>
    public static Tensor operator *(float s, in Tensor a) => Ops.Mul(ScalarOperand(s), a);

    /// <summary><c>a / b</c> with broadcasting; see <see cref="Ops.Div"/>.</summary>
    public static Tensor operator /(in Tensor a, in Tensor b) => Ops.Div(a, b);

    /// <summary><c>a / s</c> for a scalar <paramref name="s"/>.</summary>
    public static Tensor operator /(in Tensor a, float s) => Ops.Div(a, ScalarOperand(s));

    /// <summary><c>s / a</c> for a scalar <paramref name="s"/>.</summary>
    public static Tensor operator /(float s, in Tensor a) => Ops.Div(ScalarOperand(s), a);

    /// <summary>Named form of <c>+</c> for languages without operators.</summary>
    public static Tensor Add(in Tensor left, in Tensor right) => Ops.Add(left, right);

    /// <summary>Named form of <c>-</c> for languages without operators.</summary>
    public static Tensor Subtract(in Tensor left, in Tensor right) => Ops.Sub(left, right);

    /// <summary>Named form of <c>*</c> for languages without operators.</summary>
    public static Tensor Multiply(in Tensor left, in Tensor right) => Ops.Mul(left, right);

    /// <summary>Named form of <c>/</c> for languages without operators.</summary>
    public static Tensor Divide(in Tensor dividend, in Tensor divisor) => Ops.Div(dividend, divisor);

    /// <summary>Named form of unary <c>-</c> for languages without operators.</summary>
    public static Tensor Negate(in Tensor value) => Ops.Neg(value);

    private static Tensor ScalarOperand(float s)
    {
        Tensor t = Scalar(s);
        ActivationScope.Track(t);
        return t;
    }

    // ---- conveniences -----------------------------------------------------------------------------------------------------

    /// <summary>Allocates a zero-filled float or integer tensor of the given dimensions: <c>Tensor.Zeros(DType.F32, 2, 3)</c>.</summary>
    /// <param name="dtype">The element type.</param>
    /// <param name="dims">The extent of each dimension (none for a scalar).</param>
    /// <exception cref="TensorShapeException">A dimension is negative or the tensor would exceed the size limits.</exception>
    public static Tensor Zeros(DType dtype, params ReadOnlySpan<int> dims) => Zeros(dtype, Shape.Create(dims));

    /// <summary>Allocates a tensor of the given dimensions holding a copy of <paramref name="data"/> (row-major): <c>Tensor.FromSpan&lt;float&gt;([1, 2, 3, 4], 2, 2)</c>.</summary>
    /// <param name="data">The values.</param>
    /// <param name="dims">The extent of each dimension; their product must equal the number of values.</param>
    /// <exception cref="ArgumentException">The values do not fill the dimensions.</exception>
    public static Tensor FromSpan<T>(ReadOnlySpan<T> data, params ReadOnlySpan<int> dims) where T : unmanaged => FromSpan(data, Shape.Create(dims));

    /// <summary>The value of a tensor with exactly one element (a scalar, or any shape of size 1), for example a loss.</summary>
    /// <typeparam name="T">Must match <see cref="DType"/>.</typeparam>
    /// <exception cref="InvalidOperationException">The tensor does not have exactly one element, or is empty.</exception>
    /// <exception cref="ArgumentException"><typeparamref name="T"/> does not match <see cref="DType"/>.</exception>
    /// <exception cref="StaleTensorException">The storage was released.</exception>
    public T Item<T>() where T : unmanaged
    {
        ThrowIfEmpty();
        if (Length != 1)
        {
            throw new InvalidOperationException($"Item needs a tensor with one element, not {_shape} ({Length} elements).");
        }

        Span<int> zeros = stackalloc int[Shape.MaxRank];
        return Get<T>(zeros[..Rank]);
    }
}
