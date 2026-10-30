using System;

namespace RecursiveAI.OneModel.Tensors;

/// <summary>Why a trace failed.</summary>
public enum TraceErrorKind
{
    /// <summary>The model read the value of a traced tensor (<c>ToArray</c>, <c>Get</c>, <c>Data</c>, a copy into another tensor, ...). A branch or loop count taken from it would be baked into the graph.</summary>
    DataDependent,

    /// <summary>A traced tensor was changed in place, or a constant captured by the trace changed during it.</summary>
    InPlaceMutation,

    /// <summary><see cref="GradMode.NoGrad"/> was entered while tracing.</summary>
    NoGradInTrace,

    /// <summary>The operation has no equivalent in the graph layer, or this combination of arguments has none. See <c>docs/tracing.md</c> for the list.</summary>
    UnsupportedOperation,

    /// <summary>A symbolic dimension reached an operation that cannot express it.</summary>
    SymbolicShape,

    /// <summary>The tracer was used wrongly: nested or concurrent traces on one thread, an output that was not computed from the inputs, a duplicate name.</summary>
    InvalidUsage,
}

/// <summary>A model could not be traced. <see cref="Kind"/> says why; the message names the operation or read.</summary>
public sealed class TraceException : InvalidOperationException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="kind">Why tracing failed.</param>
    /// <param name="message">What was wrong and where.</param>
    public TraceException(TraceErrorKind kind, string message) : base(message)
    {
        Kind = kind;
    }

    /// <summary>Why tracing failed.</summary>
    public TraceErrorKind Kind { get; }
}

/// <summary>
/// The seam between the eager operations and the tracer (which lives in <c>RecursiveAI.OneModel.Tracing</c>): operations already
/// build a <see cref="BackwardRecord"/> for each result that depends on a tensor requiring gradients, and while a trace records, every
/// traced tensor requires gradients, so the same records tell the tracer what ran.
/// </summary>
internal abstract class TraceSink
{
    /// <summary>True if <paramref name="trace"/> is a value of this recording.</summary>
    public abstract bool Owns(object trace);

    /// <summary>Gives <paramref name="output"/> its traced identity after the record's operation ran; returns the tensor to hand back to the caller.</summary>
    public abstract Tensor Attach(BackwardRecord record, in Tensor output, int outputIndex);

    /// <summary>Called for reads and writes of a traced tensor's data while recording. Always throws.</summary>
    public abstract void Reject(TraceErrorKind kind, in Tensor tensor, string what);
}

internal static class TraceContext
{
    [ThreadStatic]
    private static TraceSink? s_current;

    /// <summary>The recording active on this thread, or null.</summary>
    public static TraceSink? Current => s_current;

    public static void Enter(TraceSink sink)
    {
        if (s_current is not null)
        {
            throw new TraceException(TraceErrorKind.InvalidUsage, "A trace is already recording on this thread; finish or dispose it first.");
        }

        s_current = sink;
    }

    public static void Exit(TraceSink sink)
    {
        if (ReferenceEquals(s_current, sink))
        {
            s_current = null;
        }
    }
}
