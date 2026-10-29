using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace RecursiveAI.OneModel.Benchmarks;

/// <summary>A placeholder benchmark that shows the harness and the allocation diagnoser work.</summary>
[MemoryDiagnoser]
public class SelfCheckBenchmarks
{
    private readonly byte[] _data = new byte[4096];

    /// <summary>Hashes 4 KiB.</summary>
    [Benchmark]
    public ulong Fnv1a4K() => OneModelInfo.Fnv1a(_data);
}

internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length > 0 && args[0] == "attention-spike")
        {
            return AttentionSpike.Run(args[1..]);
        }

        BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
        return 0;
    }
}
