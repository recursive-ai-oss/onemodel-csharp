using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace RecursiveAI.OneModel.Graph.Runtime;

// LOCAL PROFILING PATCH, not committed.
internal static class NodeProfile
{
    public static readonly bool On = Environment.GetEnvironmentVariable("ONEMODEL_PROFILE") == "1";
    private static readonly Dictionary<string, (long ticks, int count)> Acc = new();
    private static bool _hooked;
    public static long Start() => Stopwatch.GetTimestamp();
    public static void Stop(long t0, string key)
    {
        long d = Stopwatch.GetTimestamp() - t0;
        lock (Acc)
        {
            if (!_hooked) { _hooked = true; AppDomain.CurrentDomain.ProcessExit += (_, _) => Dump(); }
            Acc.TryGetValue(key, out var v);
            Acc[key] = (v.ticks + d, v.count + 1);
        }
    }
    private static void Dump()
    {
        var sb = new StringBuilder();
        foreach (var kv in Acc.OrderByDescending(k => k.Value.ticks))
            sb.AppendLine($"PROFILE\t{kv.Key}\t{kv.Value.count}\t{kv.Value.ticks * 1e3 / Stopwatch.Frequency:F3}");
        Console.Error.Write(sb.ToString());
    }
}
