using System.Collections.Immutable;

namespace RecursiveAI.OneModel.Graph;

/// <summary>Where a session's operators run for one set of input shapes. Made by <see cref="Session.GetDiagnostics"/>.</summary>
public sealed class SessionDiagnostics
{
    internal SessionDiagnostics(string device, ImmutableArray<NodeDiagnostic> nodes, int deviceRegions, long bytesToDevice, long bytesFromDevice)
    {
        Device = device;
        Nodes = nodes;
        DeviceRegions = deviceRegions;
        BytesToDevicePerRun = bytesToDevice;
        BytesFromDevicePerRun = bytesFromDevice;
    }

    /// <summary>The device the session was asked to use: <c>cpu</c>, or the name of the accelerator (<c>gpu</c>).</summary>
    public string Device { get; }

    /// <summary>The operators in execution order, each with its placement.</summary>
    public ImmutableArray<NodeDiagnostic> Nodes { get; }

    /// <summary>Number of runs of consecutive operators that execute on the device; tensors cross to and from the device once per run of operators.</summary>
    public int DeviceRegions { get; }

    /// <summary>Bytes one model run copies from the host to the device (constant weights are copied once and not counted).</summary>
    public long BytesToDevicePerRun { get; }

    /// <summary>Bytes one model run copies from the device back to the host.</summary>
    public long BytesFromDevicePerRun { get; }

    /// <summary>Number of operators that run on the device.</summary>
    public int NodesOnDevice
    {
        get
        {
            int n = 0;
            foreach (NodeDiagnostic d in Nodes)
            {
                if (d.Region >= 0)
                {
                    n++;
                }
            }

            return n;
        }
    }

    /// <summary>Number of operators that run on the CPU.</summary>
    public int NodesOnHost => Nodes.Length - NodesOnDevice;
}

/// <summary>The placement of one operator.</summary>
/// <param name="Name">The graph node's name; may be empty.</param>
/// <param name="OpType">The ONNX operator type.</param>
/// <param name="Placement">The device's name, or <c>cpu</c>.</param>
/// <param name="Region">Index of the run of device operators this one belongs to, or -1 on the CPU.</param>
/// <param name="FallbackReason">When the session has a device and this operator stayed on the CPU, why; otherwise null.</param>
public readonly record struct NodeDiagnostic(string Name, string OpType, string Placement, int Region, string? FallbackReason);
