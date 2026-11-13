using System.Security.Cryptography;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tests.Graph;
using RecursiveAI.OneModel.Tensors;

// Several devices created at once in one process crash the NVIDIA driver with validation on, and a test run needs one device at a time.
[assembly: CollectionBehavior(CollectionBehavior.CollectionPerAssembly)]
[assembly: AssemblyFixture(typeof(RecursiveAI.OneModel.Gpu.Tests.DeviceLeakCheck))]

namespace RecursiveAI.OneModel.Gpu.Tests;

/// <summary>Registers the GPU package with validation and timings on, and skips a test when the machine has no usable device.</summary>
internal static class GpuTestHost
{
    private static readonly Lazy<string?> s_unavailable = new(Probe);

    private static readonly GpuOptions s_default = new() { Validation = true, CollectTimings = true };

    /// <summary>Registers the package with other options until the returned scope is disposed (tests run one at a time, so nothing else is creating sessions).</summary>
    public static IDisposable With(GpuOptions options)
    {
        OneModelGpu.Register(options);
        return new Restore();
    }

    private sealed class Restore : IDisposable
    {
        public void Dispose() => OneModelGpu.Register(s_default);
    }

    private static string? Probe()
    {
        OneModelGpu.Register(s_default);
        try
        {
            byte[] model = new ModelBuilder().Input("x", 1, 2, 2).Output("y", 1, 2, 2).Node("MatMul", ["x", "x"], ["y"]).Build();
            using Session s = Session.Load(new MemoryStream(model), new ModelOptions { Device = ComputeDevice.Gpu });
            return null;
        }
        catch (SessionException e) when (e.Kind == SessionErrorKind.DeviceUnsupported)
        {
            return e.Message;
        }
    }

    /// <summary>Skips the calling test, printing why, when there is no GPU device.</summary>
    public static void Require()
    {
        // A machine that is meant to have a device (ONEMODEL_REQUIRE_DEVICE=1) fails instead of skipping everything.
        Assert.False(s_unavailable.Value is not null && Environment.GetEnvironmentVariable("ONEMODEL_REQUIRE_DEVICE") == "1",
            "no GPU device, but one is required: " + s_unavailable.Value);
        Assert.SkipWhen(s_unavailable.Value is not null, s_unavailable.Value ?? "");
    }

    public static ModelOptions Gpu(ArithmeticMode mode = ArithmeticMode.Fast, bool require = false) =>
        new() { Mode = mode, Device = ComputeDevice.Gpu, RequireGpu = require, IntraOpThreads = 2 };

    public static ModelOptions Cpu(ArithmeticMode mode = ArithmeticMode.Fast) => new() { Mode = mode, IntraOpThreads = 2 };

    public static Session Load(byte[] model, ModelOptions options)
    {
        Session s = Session.Load(new MemoryStream(model), options);
        if (s.Backend is GpuBackend backend)
        {
            DeviceLeakCheck.Track(backend);
        }

        return s;
    }

    /// <summary>The validation layer's messages since the last call; a test fails on any.</summary>
    public static void AssertValidationClean(Session session)
    {
        var backend = (GpuBackend)session.Backend!;
        string[] messages = backend.Device.TakeValidationMessages();
        Assert.True(messages.Length == 0, "validation messages: " + string.Join(" | ", messages));
    }

    /// <summary>max|a - b| / max|b|, the library's scaled error.</summary>
    public static double ScaledError(ReadOnlySpan<float> ours, ReadOnlySpan<float> reference)
    {
        Assert.Equal(reference.Length, ours.Length);
        double max = 0, diff = 0;
        for (int i = 0; i < ours.Length; i++)
        {
            max = Math.Max(max, Math.Abs(reference[i]));
            diff = Math.Max(diff, Math.Abs((double)ours[i] - reference[i]));
        }

        return max == 0 ? diff : diff / max;
    }

    public static string Hash(ReadOnlySpan<float> data) => Convert.ToHexString(SHA256.HashData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(data)))[..16];

    public static float[] Random(int n, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        return [.. Enumerable.Range(0, n).Select(_ => ((float)rng.NextDouble() - 0.5f) * scale)];
    }

    /// <summary>Runs a one-input model and returns the first output.</summary>
    public static float[] Run(Session s, string input, float[] x, int[] shape, string output, int outLength)
    {
        float[] y = new float[outLength];
        s.Run(input, x.AsSpan(), shape, output, y.AsSpan());
        return y;
    }
}

/// <summary>
/// Remembers the backend of every GPU session the tests create. When the run ends it checks that each was disposed and that the device
/// reported nothing after everything was released: a leaked buffer or pipeline shows up there and nowhere earlier.
/// </summary>
internal sealed class DeviceLeakCheck : IDisposable
{
    private static readonly System.Collections.Concurrent.ConcurrentBag<GpuBackend> s_backends = [];

    public static void Track(GpuBackend backend) => s_backends.Add(backend);

    public void Dispose()
    {
        var problems = new List<string>();
        foreach (GpuBackend b in s_backends)
        {
            if (!b.IsDisposed)
            {
                problems.Add("a session's device was never released");
            }
            else if (b.MessagesAfterDispose.Length > 0)
            {
                problems.Add("after release: " + string.Join(" | ", b.MessagesAfterDispose));
            }
        }

        Assert.True(problems.Count == 0, $"{problems.Count} device problem(s): {string.Join("; ", problems.Distinct())}");
    }
}
