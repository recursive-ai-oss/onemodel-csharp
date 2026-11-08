using System;
using System.Diagnostics;
using System.Globalization;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Training;

namespace RecursiveAI.OneModel.Benchmarks;

/// <summary>Indicative caller-buffer allocations and large-parameter clipping/AdamW timings, without a forward/backward pass.</summary>
internal static class TrainingStep
{
    public static int Run(string[] args)
    {
        if (args.Length > 0)
        {
            using var session = Session.Load(args[0], new ModelOptions { IntraOpThreads = 1 });
            using Tensor condition = Tensor.Scalar(true);
            using Tensor x = Tensor.FromSpan<float>([1f], Shape.Create(1));
            using Tensor y = Tensor.Zeros(DType.F32, Shape.Create(1));
            string[] names = ["condition", "x"], wanted = ["y"];
            Tensor[] inputs = [condition, x], outputs = [y];
            for (int i = 0; i < 10000; i++) session.Run(names, inputs, wanted, outputs);
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 10000; i++) session.Run(names, inputs, wanted, outputs);
            long bytes = GC.GetAllocatedBytesForCurrentThread() - before;
            Console.WriteLine($"If bytes/call: {bytes / 10000.0:R}; output {y.Data<float>()[0]:R}");
        }

        if (args.Length > 1 && args[1] == "alloc-only") return 0;

        const int count = 110_000_000;
        using Tensor p = Tensor.Zeros(DType.F32, Shape.Create(count)).WithRequiresGrad();
        using Tensor gradient = Tensor.Zeros(DType.F32, p.Shape);
        p.Autograd!.Grad = gradient;
        using (GradMode.NoGrad()) p.MutableData<float>().Fill(0.25f);
        Tensor[] parameters = [p];
        using var adam = new AdamW(parameters);
        foreach (int threads in new[] { 1, 8 })
        {
            TensorCompute.Threads = threads;
            gradient.MutableData<float>().Fill(0.125f);
            GradClip.ClipNorm(parameters, 1f);
            adam.Step();
            double clip = 0, update = 0;
            for (int i = 0; i < 3; i++)
            {
                gradient.MutableData<float>().Fill(0.125f);
                long start = Stopwatch.GetTimestamp();
                GradClip.ClipNorm(parameters, 1f);
                clip += Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                start = Stopwatch.GetTimestamp();
                adam.Step();
                update += Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            }

            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"indicative params={count} threads={threads} clip_ms={clip / 3:F3} adamw_ms={update / 3:F3}"));
        }

        p.Autograd.Grad = default;
        return 0;
    }
}
