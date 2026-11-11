using System;
using System.Collections.Generic;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Runtime;

/// <summary>
/// An execution backend runs some of a plan's steps somewhere other than the built-in CPU kernels. The plan offers each shaped step
/// to the backend, groups the accepted steps that follow each other into <see cref="BackendRegion"/>s, and hands the regions back
/// to be compiled once per plan. The plan moves steps the backend did not take ahead of ones it did, as far as their inputs allow, so that
/// accepted steps are adjacent and form few regions. A region is the unit of transfer: its inputs are made available to the backend before it runs and
/// its outputs are back in host memory when it returns, so nothing crosses between host and backend inside a region.
/// </summary>
/// <remarks>
/// Implementations must be safe for the concurrent runs of one plan (the plan is shared by any number of runs), either by locking
/// or by keeping per-run state of their own.
/// </remarks>
internal interface IExecutionBackend : IDisposable
{
    /// <summary>A short name for diagnostics, for example <c>accelerator</c>.</summary>
    string Name { get; }

    /// <summary>
    /// Which graph passes run for a session that offers its steps to this backend. Asked once per pass when the session is created, with the
    /// pass's name and the lowest optimization level that runs it: null keeps the default (the level and arithmetic mode decide), false skips the
    /// pass, true runs it also in exact mode. A pass above the session's level never runs. Operators a pass introduces reach
    /// <see cref="CheckNode"/> and <see cref="ClaimStep"/> like any other; an operator the backend does not take stays on the host.
    /// </summary>
    bool? ChoosePass(string passName, OptimizationLevel minLevel) => null;

    /// <summary>
    /// Asked once per node when a session is created, before any shape is known: can this backend ever run the node, given its operator,
    /// attributes and the session's arithmetic mode? Null means it may; otherwise the reason it cannot, in plain words.
    /// </summary>
    string? CheckNode(CompiledNode node, ArithmeticMode mode);

    /// <summary>
    /// Whether the backend wants to be offered view steps (operators that only reinterpret their input's storage, such as Reshape) as well.
    /// A claimed view must give its output the same storage as its input. A claimed view whose result is read outside its region stays on
    /// the host instead, because the backend's storage has no host copy of the source to view.
    /// </summary>
    bool ClaimsViews => false;

    /// <summary>
    /// Asked once per statically shaped step when a plan is built (view steps only when <see cref="ClaimsViews"/>), in schedule order, with
    /// the plan's value types, shapes and constants available. Null accepts the step; otherwise the reason it stays on the host.
    /// </summary>
    string? ClaimStep(ExecutionPlan plan, PlanStep step);

    /// <summary>Called once after every step has been offered, with the regions formed from the accepted ones (at least one).</summary>
    IBackendProgram Compile(ExecutionPlan plan, IReadOnlyList<BackendRegion> regions);
}

/// <summary>The compiled regions of one plan.</summary>
internal interface IBackendProgram : IDisposable
{
    /// <summary>
    /// Runs one region. <paramref name="values"/> is the run's value table indexed by value id: the region's inputs are readable in it, and
    /// each of the region's outputs is a host tensor with its planned type and shape that the call must have filled when it returns.
    /// Values the region keeps to itself are not touched.
    /// </summary>
    void Run(int region, Tensor[] values);

    /// <summary>Bytes copied from the host into the backend by one run, summed over regions (constants are copied once and not counted).</summary>
    long BytesToBackendPerRun { get; }

    /// <summary>Bytes copied from the backend to the host by one run, summed over regions.</summary>
    long BytesFromBackendPerRun { get; }
}

/// <summary>A maximal run of consecutive scheduled steps that a backend accepted, and the values that cross its boundary.</summary>
internal sealed class BackendRegion
{
    internal BackendRegion(int index, int firstStep, int lastStep, PlanStep[] steps, int[] inputs, int[] constants, int[] outputs)
    {
        Index = index;
        FirstStep = firstStep;
        LastStep = lastStep;
        Steps = steps;
        Inputs = inputs;
        Constants = constants;
        Outputs = outputs;
    }

    /// <summary>Position in <see cref="ExecutionPlan.Regions"/>.</summary>
    public int Index { get; }

    /// <summary>Index in <see cref="ExecutionPlan.Steps"/> of the first and last step of the region.</summary>
    public int FirstStep { get; }

    public int LastStep { get; }

    public PlanStep[] Steps { get; }

    /// <summary>Value ids read by the region and made outside it: graph inputs and results of host steps. Each appears once.</summary>
    public int[] Inputs { get; }

    /// <summary>Value ids of constants (weights and folded values) the region reads. Each appears once.</summary>
    public int[] Constants { get; }

    /// <summary>Value ids made in the region and read outside it or returned by the graph. Each appears once.</summary>
    public int[] Outputs { get; }
}

/// <summary>Where one scheduled node ran, for diagnostics.</summary>
/// <param name="NodeName">The graph node's name, possibly empty.</param>
/// <param name="OpType">The operator type.</param>
/// <param name="Backend">The backend's name, or <c>cpu</c> for the built-in kernels.</param>
/// <param name="Region">The region index when a backend ran the node, else -1.</param>
/// <param name="Reason">Why a node stayed on the host although a backend was present; null otherwise.</param>
internal readonly record struct NodePlacement(string NodeName, string OpType, string Backend, int Region, string? Reason);

/// <summary>A node a backend can never run, found when the session is created.</summary>
/// <param name="NodeName">The graph node's name, possibly empty.</param>
/// <param name="OpType">The operator type.</param>
/// <param name="Reason">Why, in plain words.</param>
internal readonly record struct NodeDecline(string NodeName, string OpType, string Reason);
