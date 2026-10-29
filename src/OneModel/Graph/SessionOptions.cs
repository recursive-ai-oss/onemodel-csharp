using System;

namespace RecursiveAI.OneModel.Graph;

/// <summary>How exactly a session computes. See docs/exact-mode.md.</summary>
public enum ExecutionMode
{
    /// <summary>
    /// The default. Kernels may use fused multiply-add and any blocking. Results depend only on the inputs, the options and the
    /// machine's kernel path, never on the thread count or scheduling.
    /// </summary>
    Fast = 0,

    /// <summary>
    /// Opt-in. Every result is bit-identical across operating systems, CPUs, instruction sets and thread counts. Creating a session
    /// fails if the model uses an operator that has no exact kernel.
    /// </summary>
    Exact = 1,
}

/// <summary>How much the graph is rewritten before it is planned.</summary>
public enum OptimizationLevel
{
    /// <summary>The graph runs as imported (apart from evaluating constant subgraphs when a plan is made).</summary>
    None = 0,

    /// <summary>Rewrites that never change a result: constant subgraphs are evaluated once when the session is created.</summary>
    Basic = 1,

    /// <summary>
    /// The default. Everything in <see cref="Basic"/>, plus constant weights packed once for the matrix kernels and operator fusion
    /// (bias and activation applied to each finished tile, Gelu and attention patterns, convolution with its batch normalization).
    /// In <see cref="ExecutionMode.Exact"/> only the passes that keep every bit run; see docs/graph-passes.md.
    /// </summary>
    All = 2,
}

/// <summary>Settings fixed when a <see cref="Session"/> is created.</summary>
public sealed class SessionOptions
{
    /// <summary>Fast or exact arithmetic. Default <see cref="ExecutionMode.Fast"/>.</summary>
    public ExecutionMode Mode { get; init; } = ExecutionMode.Fast;

    /// <summary>
    /// The most threads one <c>Run</c> may use inside an operator. 0 (the default) means one per logical core, up to 256. 1 runs
    /// everything on the calling thread. Changes speed only: results never depend on it.
    /// </summary>
    public int IntraOpThreads { get; init; }

    /// <summary>Graph rewriting level. Default <see cref="OptimizationLevel.All"/>.</summary>
    public OptimizationLevel OptimizationLevel { get; init; } = OptimizationLevel.All;

    /// <summary>
    /// How many execution plans (one per distinct set of input shapes) a session keeps. When more distinct shapes arrive, the least
    /// recently created plan is dropped. Default 16, at least 1.
    /// </summary>
    public int MaxCachedPlans { get; init; } = 16;

    internal void Validate()
    {
        if (!Enum.IsDefined(Mode))
        {
            throw new ArgumentOutOfRangeException(nameof(Mode), Mode, "Unknown execution mode.");
        }

        if (IntraOpThreads is < 0 or > 256)
        {
            throw new ArgumentOutOfRangeException(nameof(IntraOpThreads), IntraOpThreads, "Threads must be between 0 and 256.");
        }

        if (!Enum.IsDefined(OptimizationLevel))
        {
            throw new ArgumentOutOfRangeException(nameof(OptimizationLevel), OptimizationLevel, "Unknown optimization level.");
        }

        if (MaxCachedPlans < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxCachedPlans), MaxCachedPlans, "At least one plan must be cached.");
        }
    }
}
