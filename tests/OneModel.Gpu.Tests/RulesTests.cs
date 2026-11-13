using System.Collections.Frozen;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tests.Graph;

namespace RecursiveAI.OneModel.Gpu.Tests;

/// <summary>The rules around the operator table (exact mode, attributes, reasons) and the mapping of device failures to typed errors.</summary>
public class RulesTests
{
    // An operator table with a stand-in "Relu" entry: it never binds, and says whether it has a deterministic kernel.
    private static FrozenDictionary<string, GpuOpEntry> Table(bool deterministic) => new[]
    {
        MatMulOps.MatMul,
        new GpuOpEntry("Relu", deterministic, _ => null, (GpuClaimContext _, out GpuBinding? b) => { b = null; return "stand-in kernel never runs"; }),
    }.ToFrozenDictionary(e => e.OpType, StringComparer.Ordinal);

    private static byte[] Model() => new ModelBuilder().Input("x", 1, 4, 4).Output("y", 1, 4, 4)
        .Initializer("w", GpuTestHost.Random(16, 1), 4, 4)
        .Node("MatMul", ["x", "w"], ["m"]).Node("Relu", ["m"], ["y"]).Build();

    private static Session Create(ModelOptions options, bool deterministic)
    {
        using IrModel ir = OnnxImporter.Load(new MemoryStream(Model()));
        var backend = GpuBackend.Create(options, new GpuOptions { Validation = true }, Table(deterministic));
        return Session.Create(ir, options, OpRegistry.Default, null, backend, default);
    }

    [Fact]
    public void ExactModeKeepsAnOperatorWithoutADeterministicKernelOnTheCpu()
    {
        GpuTestHost.Require();
        using Session exact = Create(new ModelOptions { Mode = ArithmeticMode.Exact, Device = ComputeDevice.Gpu, OptimizationLevel = OptimizationLevel.Basic }, deterministic: false);
        NodeDecline d = Assert.Single(exact.BackendDeclines);
        Assert.Equal("Relu", d.OpType);
        Assert.Equal("no deterministic GPU kernel for Relu, so exact mode keeps it on the CPU", d.Reason);

        // The same operator in fast mode, or with a deterministic kernel, is the device's to run.
        using Session fast = Create(new ModelOptions { Mode = ArithmeticMode.Fast, Device = ComputeDevice.Gpu, OptimizationLevel = OptimizationLevel.Basic }, deterministic: false);
        Assert.Empty(fast.BackendDeclines);
        using Session det = Create(new ModelOptions { Mode = ArithmeticMode.Exact, Device = ComputeDevice.Gpu, OptimizationLevel = OptimizationLevel.Basic }, deterministic: true);
        Assert.Empty(det.BackendDeclines);
    }

    [Fact]
    public void RequireGpuInExactModeNamesTheOperatorWithoutADeterministicKernel()
    {
        GpuTestHost.Require();
        var options = new ModelOptions { Mode = ArithmeticMode.Exact, Device = ComputeDevice.Gpu, RequireGpu = true, OptimizationLevel = OptimizationLevel.Basic };
        SessionException e = Assert.Throws<SessionException>(() => Create(options, deterministic: false));
        Assert.Equal(SessionErrorKind.DeviceUnsupported, e.Kind);
        Assert.Equal(["Relu"], e.Operators);
        Assert.Contains("no deterministic GPU kernel for Relu", e.Message);
    }

    [Fact]
    public void ADeclinedStepFallsBackWithItsReasonAndTheModelStillRuns()
    {
        GpuTestHost.Require();
        using Session s = Create(new ModelOptions { Device = ComputeDevice.Gpu, OptimizationLevel = OptimizationLevel.Basic }, deterministic: true);
        using var x = Tensors.Tensor.FromSpan<float>(GpuTestHost.Random(16, 2), Tensors.Shape.Create(4, 4));
        var feed = new Dictionary<string, Tensors.Tensor> { ["x"] = x };
        SessionDiagnostics d = s.GetDiagnostics(feed);
        Assert.Equal("stand-in kernel never runs", d.Nodes[1].FallbackReason);
        using SessionOutputs o = s.Run(feed);
        Assert.All(o["y"].Data<float>().ToArray(), v => Assert.True(v >= 0));
    }

    [Fact]
    public void DeviceFailuresBecomeTypedSessionErrorsThatNameWhatFailed()
    {
        SessionException oom = GpuErrors.Wrap("running 3 operator(s)", new GpuOutOfMemoryException("no memory", 123456789UL));
        Assert.Equal(SessionErrorKind.DeviceFailure, oom.Kind);
        Assert.Contains("ran out of memory", oom.Message);
        Assert.Contains("123456789", oom.Message);

        SessionException lost = GpuErrors.Wrap("running", new GpuDeviceLostException("the device was removed", "vkQueueSubmit"));
        Assert.Equal(SessionErrorKind.DeviceFailure, lost.Kind);
        Assert.Contains("lost", lost.Message);
        Assert.Contains("vkQueueSubmit", lost.Message);

        SessionException cap = GpuErrors.Wrap("preparing", new GpuUnsupportedException("no fp16", "ShaderFloat16", "supported", "absent"));
        Assert.Equal(SessionErrorKind.DeviceFailure, cap.Kind);
        Assert.Contains("ShaderFloat16", cap.Message);

        SessionException shape = GpuErrors.Wrap("running", new GpuKernelShapeException("Gemm", "StrideC", "overlap"));
        Assert.Equal(SessionErrorKind.Unsupported, shape.Kind);
        Assert.Contains("StrideC", shape.Message);
        Assert.IsType<GpuKernelShapeException>(shape.InnerException);
    }
}
