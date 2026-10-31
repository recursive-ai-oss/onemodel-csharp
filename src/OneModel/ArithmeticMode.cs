namespace RecursiveAI.OneModel;

/// <summary>How exactly a session computes. See docs/exact-mode.md.</summary>
public enum ArithmeticMode
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
