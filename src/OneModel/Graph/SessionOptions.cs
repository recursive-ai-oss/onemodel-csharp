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
    /// <summary>The graph runs as imported (apart from evaluating constant subgraphs when a plan is made). The only level so far.</summary>
    None = 0,
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

    /// <summary>Graph rewriting level. Default and only value: <see cref="OptimizationLevel.None"/>.</summary>
    public OptimizationLevel OptimizationLevel { get; init; } = OptimizationLevel.None;

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

        if (OptimizationLevel != OptimizationLevel.None)
        {
            throw new ArgumentOutOfRangeException(nameof(OptimizationLevel), OptimizationLevel, "Only OptimizationLevel.None exists so far.");
        }

        if (MaxCachedPlans < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxCachedPlans), MaxCachedPlans, "At least one plan must be cached.");
        }
    }
}
