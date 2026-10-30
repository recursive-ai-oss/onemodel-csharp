using System.Diagnostics;
using System.Runtime.InteropServices;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Tests.Training;
using RecursiveAI.OneModel.Training;
using RecursiveAI.OneModel.Tracing;

namespace RecursiveAI.OneModel.Tests.Tracing;

// Indicative timing of traced models in a session against the same modules run eagerly, and the ONNX files for tools/trace_timing.py
// (ONNX Runtime on the same file). Runs only when ONEMODEL_TRACE_TIMING names an output directory; appends one line per row.
[Collection("autograd")]
public class TraceTiming
{
    private static double MedianMs(Action run, int warm = 5, int reps = 25)
    {
        for (int i = 0; i < warm; i++)
        {
            run();
        }

        var t = new double[reps];
        for (int i = 0; i < reps; i++)
        {
            long a = Stopwatch.GetTimestamp();
            run();
            t[i] = Stopwatch.GetElapsedTime(a).TotalMilliseconds;
        }

        Array.Sort(t);
        return t[reps / 2];
    }

    private static void Row(string dir, string name, NnModule net, Tensor example, string?[] axes)
    {
        net.Eval();
        using TracedModel tm = net.Trace(example, axes);
        tm.SaveOnnx(Path.Combine(dir, name + ".onnx"));
        foreach (int threads in new[] { 1, 8 })
        {
            TensorCompute.Threads = threads;
            double eager = MedianMs(() =>
            {
                using var scope = new ActivationScope();
                net.Forward(example);
            });
            using Session s = tm.CreateSession(new SessionOptions { IntraOpThreads = threads });
            Tensor[] outs = new Tensor[1];
            double traced = MedianMs(() => s.Run(["input"], [example], ["output"], outs));
            File.AppendAllText(Path.Combine(dir, "timing.txt"), $"{name} threads={threads} eager={eager:F2}ms session={traced:F2}ms ratio={eager / traced:F2}\n");
        }

        TensorCompute.Threads = 1;
    }

    [Fact]
    public void MeasureTracedAgainstEager()
    {
        string? dir = Environment.GetEnvironmentVariable("ONEMODEL_TRACE_TIMING");
        if (string.IsNullOrEmpty(dir))
        {
            return;
        }

        Directory.CreateDirectory(dir);
        var enc = new EncoderClassifier(new Generator(3), 1000, 384, 12, 1536, 6, 8);
        Tensor ids = TraceModels.Ids(1, 1000, 8, 64);
        Row(dir, "encoder", enc, ids, ["b", "n"]);
        File.WriteAllBytes(Path.Combine(dir, "encoder.ids.bin"), MemoryMarshal.AsBytes<long>(ids.ToArray<long>()).ToArray());

        var g = new Generator(5);
        var cnn = new Sequential(
            new Conv2d(3, 32, 3, g, padding: 1), new BatchNorm2d(32), Lambda.Relu(), Lambda.MaxPool2d(2),
            new Conv2d(32, 64, 3, g, padding: 1), new BatchNorm2d(64), Lambda.Relu(), Lambda.MaxPool2d(2),
            new Conv2d(64, 128, 3, g, padding: 1), new BatchNorm2d(128), Lambda.Relu(), Lambda.GlobalAvgPool2d(), Lambda.Flatten(), new Linear(128, 10, g));
        Tensor img = TraceModels.Rand(2, 1f, 8, 3, 32, 32);
        Row(dir, "convnet", cnn, img, ["n", null, null, null]);
        File.WriteAllBytes(Path.Combine(dir, "convnet.x.bin"), MemoryMarshal.AsBytes<float>(img.ToArray<float>()).ToArray());
    }
}
