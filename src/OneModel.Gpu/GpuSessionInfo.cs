using System.Collections.Immutable;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>A snapshot of a GPU session: the device, what it holds, and what runs cost. Made by <see cref="OneModelGpu.GetInfo(RecursiveAI.OneModel.Graph.Session)"/>.</summary>
public sealed class GpuSessionInfo
{
    internal GpuSessionInfo(string adapter, string api, ImmutableArray<string> kernels, long arenaBytes, int arenaAllocations, long constantBytes,
        int constantBuffers, long deviceAllocations, GpuTimings timings, GpuPlanInfo? plan)
    {
        Adapter = adapter;
        Api = api;
        KernelVariants = kernels;
        ArenaBytes = arenaBytes;
        ArenaAllocations = arenaAllocations;
        ConstantBytes = constantBytes;
        ConstantBuffers = constantBuffers;
        DeviceAllocations = deviceAllocations;
        Timings = timings;
        Plan = plan;
    }

    /// <summary>The adapter's name as the driver reports it.</summary>
    public string Adapter { get; }

    /// <summary>The graphics API in use: <c>Vulkan</c>, <c>Metal</c> or <c>WebGpu</c>.</summary>
    public string Api { get; }

    /// <summary>The kernel variant chosen for each kernel the session has created, for example <c>f32 deterministic: tiled-64x64x16</c>.</summary>
    public ImmutableArray<string> KernelVariants { get; }

    /// <summary>Bytes of device memory held by the activation arena: the intermediate tensors of every plan, shared and reused across runs. Grows only when a plan needs more than any before it.</summary>
    public long ArenaBytes { get; }

    /// <summary>How many times the arena has allocated a device buffer (including growth). A run of a shape that has run before does not change it.</summary>
    public int ArenaAllocations { get; }

    /// <summary>Bytes of device memory holding constant weights, uploaded once.</summary>
    public long ConstantBytes { get; }

    /// <summary>Number of device buffers holding constant weights.</summary>
    public int ConstantBuffers { get; }

    /// <summary>Device buffers allocated by this session so far, the arena's and the weights'. A second run of the same shape leaves it unchanged.</summary>
    public long DeviceAllocations { get; }

    /// <summary>Time spent so far; all zero unless <see cref="GpuOptions.CollectTimings"/> was set.</summary>
    public GpuTimings Timings { get; }

    /// <summary>The cost of one run of the plan asked about, or null when <see cref="OneModelGpu.GetInfo(RecursiveAI.OneModel.Graph.Session)"/> was used.</summary>
    public GpuPlanInfo? Plan { get; }
}

/// <summary>What one run of one plan does on the device.</summary>
/// <param name="Regions">Runs of consecutive device operators; each is one submission.</param>
/// <param name="Flops">Floating-point operations on the device per run, counting a multiply-add as two.</param>
/// <param name="BytesToDevice">Bytes uploaded per run (weights excluded).</param>
/// <param name="BytesFromDevice">Bytes read back per run.</param>
/// <param name="ArenaBytes">Device memory the plan's intermediate tensors need.</param>
public sealed record GpuPlanInfo(int Regions, double Flops, long BytesToDevice, long BytesFromDevice, long ArenaBytes);

/// <summary>Time a GPU session has spent on its device regions, summed over runs.</summary>
/// <param name="Regions">Regions run.</param>
/// <param name="UploadMilliseconds">Host time copying inputs to the device (queueing the copies).</param>
/// <param name="GpuMilliseconds">Device time from timestamp queries, the sum of the kernels' start-to-end times; zero where the device has no timestamps.</param>
/// <param name="SubmitAndReadbackMilliseconds">Host time from recording the commands to having the outputs in host memory: kernels, waiting and copying back.</param>
/// <param name="TotalMilliseconds">Host time inside regions, all of the above.</param>
/// <param name="LongestRegionMilliseconds">The longest single region, host time.</param>
/// <param name="Dispatches">Dispatches recorded (empty grids included), counted by the command encoder as it records, so the number is the same on every backend with or without span timing.</param>
/// <param name="RecordMilliseconds">Host time recording the commands: binding operands and encoding every pass, before anything is handed to the device.</param>
/// <param name="SubmitMilliseconds">Host time inside the device's submit calls. What is left of <c>SubmitAndReadbackMilliseconds</c> after both is waiting for the device and copying the outputs back.</param>
/// <param name="ComputePasses">Compute passes recorded.</param>
public readonly record struct GpuTimings(long Regions, double UploadMilliseconds, double GpuMilliseconds, double SubmitAndReadbackMilliseconds, double TotalMilliseconds,
    double LongestRegionMilliseconds = 0, long Dispatches = 0, double RecordMilliseconds = 0, double SubmitMilliseconds = 0, long ComputePasses = 0);
