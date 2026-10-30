using System;
using System.Runtime.CompilerServices;
using System.Threading;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Tensors;

/// <summary>
/// How the differentiable operations (forward and backward) compute. The setting is process-wide and read when an operation
/// starts, so change it between steps, not during one.
/// </summary>
/// <remarks>
/// <see cref="ExecutionMode.Fast"/> (the default) uses fused multiply-add where the kernels do and the BCL's <c>Math</c> functions;
/// <see cref="ExecutionMode.Exact"/> uses the exact kernels and our own transcendental polynomials, and gives the same bits on every
/// machine and thread count (docs/autograd.md says which operations are exact). The thread count changes speed only.
/// </remarks>
public static class TensorCompute
{
    private static int s_mode;
    private static int s_threads = ParallelFor.DefaultThreads;

    /// <summary>Fast or exact arithmetic. Default <see cref="ExecutionMode.Fast"/>.</summary>
    public static ExecutionMode Mode
    {
        get => (ExecutionMode)Volatile.Read(ref s_mode);
        set => Volatile.Write(ref s_mode, (int)value);
    }

    /// <summary>Upper bound on threads used inside one operation (at least 1). Default: the machine's core count, capped. Never changes a result.</summary>
    public static int Threads
    {
        get => Volatile.Read(ref s_threads);
        set
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(value, 1);
            Volatile.Write(ref s_threads, value);
        }
    }

    internal static bool Exact => Mode == ExecutionMode.Exact;

    internal static KernelMode KMode => Exact ? KernelMode.Exact : KernelMode.Fast;

    internal static readonly KernelPath Path = KernelPaths.Best();

    internal static VecIsa Isa => RowIsa.Of(Path);
}

/// <summary>Scalar math for the differentiable operations: the BCL in fast mode, our fixed polynomials in exact mode.</summary>
internal static class Fn
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Exp(bool exact, float x) => exact ? Transcendental.Exp<float, ExactVecScalar>(x) : MathF.Exp(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Log(bool exact, float x) => exact ? Transcendental.Log<float, ExactVecScalar>(x) : MathF.Log(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Tanh(bool exact, float x) => exact ? Transcendental.Tanh<float, ExactVecScalar>(x) : MathF.Tanh(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sigmoid(bool exact, float x) =>
        exact ? Transcendental.Sigmoid<float, ExactVecScalar>(x) : 1f / (1f + MathF.Exp(-x));

    // The BCL has no erf, so both modes use our polynomial.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Erf(float x) => Transcendental.Erf<float, ExactVecScalar>(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float GeluErf(float x) => Transcendental.GeluErf<float, ExactVecScalar>(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float GeluTanh(float x) => Transcendental.GeluTanh<float, ExactVecScalar>(x);
}
