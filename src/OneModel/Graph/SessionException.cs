using System;
using System.Collections.Immutable;

namespace RecursiveAI.OneModel.Graph;

/// <summary>What went wrong when creating a session or running it.</summary>
public enum SessionErrorKind
{
    /// <summary>The model uses an operator (or an opset of it) this library has no kernel for. <see cref="SessionException.Operators"/> lists them all.</summary>
    UnsupportedOperator,

    /// <summary>Exact mode was requested but the model uses operators without an exact kernel. <see cref="SessionException.Operators"/> lists them all.</summary>
    ExactModeUnavailable,

    /// <summary>The model itself is inconsistent: a wrong number of inputs, a type or shape clash between operators, a bad attribute.</summary>
    InvalidModel,

    /// <summary>A feature the model needs is not implemented yet (for example an element type, or a sparse initializer).</summary>
    Unsupported,

    /// <summary>The caller's inputs or outputs do not fit the model: unknown name, wrong element type, wrong shape, wrong buffer size.</summary>
    InvalidInput,

    /// <summary>The session was disposed.</summary>
    Disposed,
}

/// <summary>Raised for every failure of the graph layer that is the model's or the caller's doing rather than a bug.</summary>
public sealed class SessionException : Exception
{
    /// <summary>Creates the exception.</summary>
    /// <param name="kind">The category.</param>
    /// <param name="message">What was wrong and where, in plain words.</param>
    /// <param name="nodeName">The graph node involved, if one is.</param>
    /// <param name="operators">The operator types involved, if the failure is about operators.</param>
    /// <param name="inner">The exception that caused this one, if any.</param>
    public SessionException(SessionErrorKind kind, string message, string? nodeName = null,
        ImmutableArray<string> operators = default, Exception? inner = null)
        : base(message, inner)
    {
        Kind = kind;
        NodeName = nodeName;
        Operators = operators.IsDefault ? [] : operators;
    }

    /// <summary>The category of the failure.</summary>
    public SessionErrorKind Kind { get; }

    /// <summary>The graph node involved, or null.</summary>
    public string? NodeName { get; }

    /// <summary>Operator types (<c>domain.OpType</c>, opset) the failure is about; empty when it is not about operators.</summary>
    public ImmutableArray<string> Operators { get; }
}
