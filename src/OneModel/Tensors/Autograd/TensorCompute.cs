using System;
using System.Runtime.CompilerServices;
using System.Threading;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Tensors;

/// <summary>
/// How the differentiable operations (forward and backward) compute. The process-wide setting is read when an operation starts, so
/// change it between steps, not during one. To give one job its own setting without touching the others, open a
/// <see cref="Use"/> scope: it applies to the calling code and everything it awaits or starts, and restores the earlier setting when
/// disposed.
/// </summary>
/// <remarks>
/// <see cref="ArithmeticMode.Fast"/> (the default) uses fused multiply-add where the kernels do and the BCL's <c>Math</c> functions;
/// <see cref="ArithmeticMode.Exact"/> uses the exact kernels and our own transcendental polynomials, and gives the same bits on every
/// machine and thread count (docs/autograd.md says which operations are exact). The thread count changes speed only.
/// </remarks>
public static class TensorCompute
{
    private static int s_mode;
    private static int s_threads = ParallelFor.DefaultThreads;
    private static readonly AsyncLocal<Override?> s_override = new();

    private sealed record Override(ArithmeticMode? Mode, int? Threads);

    /// <summary>
    /// Overrides the mode and/or the thread limit for the current logical flow of execution (it follows <c>await</c> and threads
    /// started from here, and is invisible to other flows) until the returned scope is disposed. Scopes nest. Work that other
    /// threads do for an operation (its parallel chunks) uses the setting the operation read when it started.
    /// </summary>
    /// <param name="mode">The arithmetic mode to use; null keeps the current one.</param>
    /// <param name="threads">The thread limit to use (at least 1); null keeps the current one.</param>
    /// <returns>Dispose it to restore the previous setting.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="threads"/> is below 1.</exception>
    public static ComputeScope Use(ArithmeticMode? mode = null, int? threads = null)
    {
        if (threads is < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(threads), threads, "At least one thread.");
        }

        Override? previous = s_override.Value;
        s_override.Value = new Override(mode ?? previous?.Mode, threads ?? previous?.Threads);
        return new ComputeScope(previous);
    }

    /// <summary>Restores the setting that was current before <see cref="Use"/>. Dispose it exactly where the scope should end.</summary>
    public readonly struct ComputeScope : IDisposable
    {
        private readonly Override? _previous;

        internal ComputeScope(object? previous) => _previous = (Override?)previous;

        /// <summary>Ends the scope.</summary>
        public void Dispose() => s_override.Value = _previous;
    }

    /// <summary>Fast or exact arithmetic. Default <see cref="ArithmeticMode.Fast"/>. Reading gives the innermost <see cref="Use"/> scope's value if one is open; setting changes the process-wide value.</summary>
    public static ArithmeticMode Mode
    {
        get => s_override.Value?.Mode ?? (ArithmeticMode)Volatile.Read(ref s_mode);
        set => Volatile.Write(ref s_mode, (int)value);
    }

    /// <summary>Upper bound on threads used inside one operation (at least 1). Default: the machine's core count, capped. Never changes a result.</summary>
    public static int Threads
    {
        get => s_override.Value?.Threads ?? Volatile.Read(ref s_threads);
        set
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(value, 1);
            Volatile.Write(ref s_threads, value);
        }
    }

    internal static bool Exact => Mode == ArithmeticMode.Exact;

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
