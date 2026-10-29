using System;
using SharpFuzz;

namespace RecursiveAI.OneModel.Fuzz;

/// <summary>
/// Entry point of the fuzz harness. SharpFuzz instruments the library and libFuzzer drives <see cref="Fuzzer.LibFuzzer.Run(ReadOnlySpanAction)"/>.
/// There are no targets yet: the readers add theirs as they land.
/// </summary>
internal static class Program
{
    private static void Main()
    {
        Fuzzer.LibFuzzer.Run(static input => Target(input));
    }

    private static void Target(ReadOnlySpan<byte> input)
    {
        _ = OneModelInfo.Fnv1a(input);
    }
}
