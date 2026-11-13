namespace RecursiveAI.OneModel.Gpu;

/// <summary>How <see cref="OneModelGpu.Register(GpuOptions?)"/> sets up the device every GPU session of the process creates.</summary>
/// <remarks>Immutable once registered. Each session makes its own device.</remarks>
public sealed class GpuOptions
{
    /// <summary>
    /// Picks the adapter whose name contains this text (case-insensitive), or one of the kinds <c>discrete</c>, <c>integrated</c> or
    /// <c>software</c>. Null (the default) takes the platform's first choice. The environment variable <c>ONEGPU_ADAPTER</c> overrides it,
    /// and <c>ONEGPU_BACKEND</c> picks the graphics API.
    /// </summary>
    public string? Adapter { get; init; }

    /// <summary>
    /// Turns on the graphics API's validation layer. The environment variable <c>ONEGPU_VALIDATION=1</c> also turns it on. Slow; for tests
    /// and for chasing a problem. Default false.
    /// </summary>
    public bool Validation { get; init; }

    /// <summary>
    /// Measures where a run spends its time (upload, GPU execution, readback), read back with <see cref="OneModelGpu.GetInfo(RecursiveAI.OneModel.Graph.Session)"/>. GPU
    /// execution time comes from timestamp queries and is zero where the device has none. Costs a few timer reads per region. On Metal, which samples its counters only at compute-encoder boundaries, the GPU time is one span from the start of the first timed operator to the end of the last, so timing does not split the run's encoder; on Vulkan it is the sum of the passes' own times.
    /// Default false.
    /// </summary>
    public bool CollectTimings { get; init; }

    /// <summary>
    /// Lets the graph rewrites that have a device kernel run: attention as one kernel, and Gelu as one pass. Turn it off to run the operators
    /// the model was written with, for example to see what the fusion saves. Default true.
    /// </summary>
    public bool FuseOperators { get; init; } = true;

    /// <summary>
    /// Lets a matrix product add the bias and apply the activation (ReLU or GELU) that follow it in the same dispatch, when nothing else
    /// reads the intermediate results. The float32 result has the same bits as running them separately; a float16 product skips the
    /// rounding of its intermediate to half precision (float16 products take this in fast mode only). Also off when
    /// <see cref="FuseOperators"/> is off. Default true.
    /// </summary>
    public bool FuseEpilogues { get; init; } = true;

    // For tools that attribute an error to one fusion: leaves that one rewrite out while FuseOperators stays on.
    internal bool KeepAttentionUnfused { get; init; }

    internal bool KeepGeluUnfused { get; init; }

    // For tools: dispatches in the first piece of a region handed to the device early (0 = whole region at once); null takes the platform's choice.
    internal int? PipelineFirstDispatches { get; init; }

    // For tools: time every step by itself (needs CollectTimings and a device with timestamps).
    internal bool ProfileSteps { get; init; }

    /// <summary>
    /// Whether attention runs as the one fused kernel (true) or as the chain of two products, the mask addition and a softmax that the
    /// fusion replaces (false). Null (the default) takes the fused kernel everywhere: on Vulkan and Metal, and in a browser, where it measured
    /// as fast as the chain or faster on hardware adapters (a speech encoder's 1500 keys: 68 ms against 81 ms; BERT-base at batch 1: 19 ms against
    /// 21 ms; MiniLM at batch 1: 8.2 ms against 8.5 ms, RTX 4070 in Chrome) and on a software adapter (MiniLM 576 ms against 888 ms). On desktop it is faster at short sequences on every device and at
    /// long ones on discrete cards and Apple silicon. Both run in fast and exact mode and are deterministic, within the same tolerance. Only
    /// has an effect while <see cref="FuseOperators"/> is true.
    /// </summary>
    public bool? FusedAttention { get; init; }
}
