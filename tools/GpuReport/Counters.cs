using System;
using System.Linq;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Gpu;

namespace GpuReport;

internal static class Counters
{
    // The kernel families count the bind groups they have built; by family.
    public static int BindGroups(GpuBackend backend) => PerFamily(backend).Sum(f => f.Count);

    /// <summary>Bind groups built so far, by kernel family.</summary>
    public static (string Family, int Count)[] PerFamily(GpuBackend backend)
    {
        return
        [
            ("elementwise", backend.Elementwise.BindGroupsCreated),
            ("norm", backend.Norm.BindGroupsCreated),
            ("softmax", backend.Softmax.BindGroupsCreated),
            ("reduce", backend.Reduce.BindGroupsCreated),
            ("transpose", backend.Transpose.BindGroupsCreated),
            ("integer", backend.Integer.BindGroupsCreated),
            ("attention", backend.Attention.BindGroupsCreated),
            ("gather", backend.Gather.BindGroupsCreated),
            ("conv", backend.Conv.BindGroupsCreated),
            ("pool", backend.Pool.BindGroupsCreated),
            ("resize", backend.Resize.BindGroupsCreated),
            ("int8 gemm", backend.Int8.BindGroupsCreated),
            ("dynamic quantize", backend.DynamicQuantize.BindGroupsCreated),
            ("dequantize", backend.Dequantize.BindGroupsCreated),
            ("recurrent", backend.Recurrent.BindGroupsCreated),
            ("gemm32", backend.GetGemm(DeviceElement.F32).BindGroupsCreated),
            ("gemm16", backend.GetGemm(DeviceElement.F16).BindGroupsCreated),
        ];
    }
}
