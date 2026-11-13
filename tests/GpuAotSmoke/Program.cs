using System;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using RecursiveAI.OneModel;
using RecursiveAI.OneModel.Gpu;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tests.Graph;
using RecursiveAI.OneModel.Tensors;

// Runs a small model on the GPU from a NativeAOT binary, in fast and exact mode, against the ONNX Runtime output stored with the model,
// and prints which operators ran on the device. Exits 77 (skipped) when the machine has no GPU device.
OneModelGpu.Register();
Assembly asm = typeof(Program).Assembly;
using JsonDocument doc = JsonDocument.Parse(asm.GetManifestResourceStream("reference.json")!);
JsonElement c = doc.RootElement.GetProperty("cases").GetProperty("mlp-5");
float[] input = [.. c.GetProperty("input").EnumerateArray().Select(e => (float)e.GetDouble())];
float[] expected = [.. c.GetProperty("output").EnumerateArray().Select(e => (float)e.GetDouble())];
int[] shape = [.. c.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32())];

foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
{
    Session session;
    try
    {
        session = Session.Load(asm.GetManifestResourceStream("mlp.onnx")!, new ModelOptions { Mode = mode, Device = ComputeDevice.Gpu });
    }
    catch (SessionException e) when (e.Kind == SessionErrorKind.DeviceUnsupported)
    {
        Console.WriteLine("skipped: " + e.Message);
        return 77;
    }

    using (session)
    {
        using Tensor x = Tensor.FromSpan<float>(input, Shape.Create(shape));
        string name = session.Inputs[0].Name;
        var feed = new System.Collections.Generic.Dictionary<string, Tensor> { [name] = x };
        using SessionOutputs o = session.Run(feed);
        ReadOnlySpan<float> y = o[session.Outputs[0].Name].Data<float>();
        double max = 0, diff = 0;
        for (int i = 0; i < y.Length; i++)
        {
            max = Math.Max(max, Math.Abs(expected[i]));
            diff = Math.Max(diff, Math.Abs(y[i] - expected[i]));
        }

        // The awaited run gives the blocking run's values.
        using SessionOutputs awaited = OneModelGpu.RunAsync(session, feed).AsTask().GetAwaiter().GetResult();
        if (!awaited[session.Outputs[0].Name].Data<float>().SequenceEqual(y))
        {
            Console.WriteLine("the awaited run differs from the blocking run");
            return 1;
        }

        SessionDiagnostics d = session.GetDiagnostics(feed);
        GpuSessionInfo info = OneModelGpu.GetInfo(session);
        Console.WriteLine($"{mode}: {info.Adapter} ({info.Api}), {d.NodesOnDevice} of {d.Nodes.Length} operators on the device, scaled error {diff / max:E2}");
        if (diff / max > 1e-4 || d.NodesOnDevice == 0)
        {
            return 1;
        }
    }
}

// A convolution path: Conv + Relu, MaxPool, a residual Conv + Add, GlobalAveragePool, Flatten and Gemm, against the CPU in the same mode.
{
    var rng = new Random(7);
    float[] Rand(int n, float scale) => [.. Enumerable.Range(0, n).Select(_ => ((float)rng.NextDouble() - 0.5f) * scale)];
    byte[] conv = new ModelBuilder().Input("x", 1, 1, 3, 16, 16).Output("y", 1)
        .Initializer("w1", Rand(8 * 3 * 9, 0.5f), 8, 3, 3, 3).Initializer("b1", Rand(8, 1f), 8)
        .Initializer("w2", Rand(8 * 8 * 9, 0.3f), 8, 8, 3, 3).Initializer("fw", Rand(8 * 5, 0.5f), 5, 8).Initializer("fb", Rand(5, 1f), 5)
        .Node("Conv", ["x", "w1", "b1"], ["c1"], ("pads", new long[] { 1, 1, 1, 1 })).Node("Relu", ["c1"], ["r1"])
        .Node("MaxPool", ["r1"], ["p"], ("kernel_shape", new long[] { 2, 2 }), ("strides", new long[] { 2, 2 }))
        .Node("Conv", ["p", "w2"], ["c2"], ("pads", new long[] { 1, 1, 1, 1 })).Node("Add", ["c2", "p"], ["a"])
        .Node("GlobalAveragePool", ["a"], ["g"]).Node("Flatten", ["g"], ["f"]).Node("Gemm", ["f", "fw", "fb"], ["y"], ("transB", 1L)).Build();
    float[] image = Rand(3 * 256, 2f);
    foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
    {
        using Tensor x = Tensor.FromSpan<float>(image, Shape.Create(1, 3, 16, 16));
        var feed = new System.Collections.Generic.Dictionary<string, Tensor> { ["x"] = x };
        float[] cpu;
        using (Session c2 = Session.Load(new System.IO.MemoryStream(conv), new ModelOptions { Mode = mode }))
        using (SessionOutputs o = c2.Run(feed))
        {
            cpu = o["y"].Data<float>().ToArray();
        }

        using Session g = Session.Load(new System.IO.MemoryStream(conv), new ModelOptions { Mode = mode, Device = ComputeDevice.Gpu, RequireGpu = true });
        using SessionOutputs go = g.Run(feed);
        ReadOnlySpan<float> y = go["y"].Data<float>();
        double max = 0, diff = 0;
        for (int i = 0; i < y.Length; i++)
        {
            max = Math.Max(max, Math.Abs(cpu[i]));
            diff = Math.Max(diff, Math.Abs(y[i] - cpu[i]));
        }

        SessionDiagnostics d = g.GetDiagnostics(feed);
        Console.WriteLine($"conv {mode}: {d.NodesOnDevice} of {d.Nodes.Length} operators on the device in {d.DeviceRegions} region(s), scaled error {diff / max:E2}");
        if (diff / max > 1e-4 || d.NodesOnDevice != d.Nodes.Length)
        {
            return 1;
        }
    }
}

// A recurrent path: a bidirectional LSTM with sequence lengths from the host, and an int8 quantize-dequantize pair, against the CPU in the same mode.
{
    var rng = new Random(9);
    float[] Rand(int n, float scale) => [.. Enumerable.Range(0, n).Select(_ => ((float)rng.NextDouble() - 0.5f) * scale)];
    byte[] lstm = new ModelBuilder(14).Input("x", 1, 6, 2, 5).Input("lens", 6, 2).Output("y", 1).Output("yh", 1).Output("q", 1)
        .Initializer("w", Rand(2 * 4 * 7 * 5, 0.6f), 2, 28, 5).Initializer("r", Rand(2 * 4 * 7 * 7, 0.6f), 2, 28, 7).Initializer("b", Rand(2 * 56, 0.5f), 2, 56)
        .Node("LSTM", ["x", "w", "r", "b", "lens"], ["y", "yh"], ("hidden_size", 7L), ("direction", "bidirectional"))
        .RawInitializer("s", 1, [], BitConverter.GetBytes(0.05f)).RawInitializer("z", 3, [], [unchecked((byte)-3)])
        .Node("QuantizeLinear", ["y", "s", "z"], ["yq"]).Node("DequantizeLinear", ["yq", "s", "z"], ["q"]).Build();
    float[] seq = Rand(6 * 2 * 5, 2f);
    foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
    {
        using Tensor x = Tensor.FromSpan<float>(seq, Shape.Create(6, 2, 5));
        using Tensor lens = Tensor.FromSpan<int>([6, 4], Shape.Create(2));
        var feed = new System.Collections.Generic.Dictionary<string, Tensor> { ["x"] = x, ["lens"] = lens };
        float[] cpu, cpuQ;
        using (Session c2 = Session.Load(new System.IO.MemoryStream(lstm), new ModelOptions { Mode = mode }))
        using (SessionOutputs o = c2.Run(feed))
        {
            cpu = o["y"].Data<float>().ToArray();
            cpuQ = o["q"].Data<float>().ToArray();
        }

        using Session g = Session.Load(new System.IO.MemoryStream(lstm), new ModelOptions { Mode = mode, Device = ComputeDevice.Gpu, RequireGpu = true });
        using SessionOutputs go = g.Run(feed);
        ReadOnlySpan<float> y = go["y"].Data<float>(), q = go["q"].Data<float>();
        double max = 0, diff = 0, qdiff = 0;
        for (int i = 0; i < y.Length; i++)
        {
            max = Math.Max(max, Math.Abs(cpu[i]));
            diff = Math.Max(diff, Math.Abs(y[i] - cpu[i]));
            qdiff = Math.Max(qdiff, Math.Abs(q[i] - cpuQ[i]));
        }

        SessionDiagnostics d = g.GetDiagnostics(feed);
        Console.WriteLine($"lstm {mode}: {d.NodesOnDevice} of {d.Nodes.Length} operators on the device in {d.DeviceRegions} region(s), scaled error {diff / max:E2}, quantized values differ by at most {qdiff:E1}");
        // A last-bit difference in a layer's output may land on the other side of a quantization step (0.05); no more than one step.
        if (diff / max > 1e-4 || qdiff > 0.0501 || d.NodesOnDevice != d.Nodes.Length)
        {
            return 1;
        }
    }
}

// An integer convolution path: DynamicQuantizeLinear, ConvInteger (one spatial axis, per-channel weight zero points), Cast and Mul, against the CPU in the same
// mode. The sums are integers, so the outputs must be equal bit for bit.
{
    var rng = new Random(11);
    float[] Rand(int n, float scale) => [.. Enumerable.Range(0, n).Select(_ => ((float)rng.NextDouble() - 0.5f) * scale)];
    byte[] wq = [.. Enumerable.Range(0, 6 * 4 * 3).Select(_ => (byte)rng.Next(256))], wz = [.. Enumerable.Range(0, 6).Select(_ => (byte)rng.Next(256))];
    byte[] qconv = new ModelBuilder().Input("x", 1, 2, 4, 40).Output("y", 1)
        .RawInitializer("w", 2, [6, 4, 3], wq).RawInitializer("wz", 2, [6], wz).RawInitializer("ws", 1, [], BitConverter.GetBytes(0.01f))
        .Node("DynamicQuantizeLinear", ["x"], ["xq", "xs", "xz"])
        .Node("ConvInteger", ["xq", "w", "xz", "wz"], ["acc"], ("pads", new long[] { 1, 1 }))
        .Node("Cast", ["acc"], ["accf"], ("to", 1L)).Node("Mul", ["xs", "ws"], ["sc"]).Node("Mul", ["accf", "sc"], ["y"]).Build();
    float[] data = Rand(2 * 4 * 40, 3f);
    foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
    {
        using Tensor x = Tensor.FromSpan<float>(data, Shape.Create(2, 4, 40));
        var feed = new System.Collections.Generic.Dictionary<string, Tensor> { ["x"] = x };
        float[] cpu;
        using (Session c2 = Session.Load(new System.IO.MemoryStream(qconv), new ModelOptions { Mode = mode }))
        using (SessionOutputs o = c2.Run(feed))
        {
            cpu = o["y"].Data<float>().ToArray();
        }

        using Session g = Session.Load(new System.IO.MemoryStream(qconv), new ModelOptions { Mode = mode, Device = ComputeDevice.Gpu, RequireGpu = true });
        using SessionOutputs go = g.Run(feed);
        bool same = go["y"].Data<float>().SequenceEqual(cpu);
        SessionDiagnostics d = g.GetDiagnostics(feed);
        Console.WriteLine($"convinteger {mode}: {d.NodesOnDevice} of {d.Nodes.Length} operators on the device in {d.DeviceRegions} region(s), outputs {(same ? "equal" : "DIFFER")}");
        if (!same || d.NodesOnDevice != d.Nodes.Length)
        {
            return 1;
        }
    }
}

Console.WriteLine("ok");
return 0;
