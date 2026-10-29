using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.Recurrent;

/// <summary>
/// RNN, GRU and LSTM through the whole stack against ONNX Runtime's outputs, recorded by tools/ops_recurrent_oracle.py into
/// testdata/ops-recurrent (random sizes, directions, layouts, sequence lengths, initial state, peepholes, clip, activations).
/// </summary>
public class RecurrentTests
{
    private static string CasesDir([CallerFilePath] string file = "") =>
        Path.GetFullPath(Path.Combine(Path.GetDirectoryName(file)!, "..", "..", "..", "..", "..", "testdata", "ops-recurrent"));

    private static IEnumerable<string> CaseDirs() =>
        Directory.GetDirectories(CasesDir()).Order(StringComparer.Ordinal);

    private sealed class Case : IDisposable
    {
        public required IrModel Model { get; init; }

        public required Tensor[] Inputs { get; init; }

        public required Tensor[] Expected { get; init; }

        public void Dispose()
        {
            Model.Dispose();
            foreach (Tensor t in Inputs)
            {
                t.Dispose();
            }

            foreach (Tensor t in Expected)
            {
                t.Dispose();
            }
        }
    }

    private static Case Load(string dir)
    {
        IrModel model = OnnxImporter.Load(Path.Combine(dir, "model.onnx"));
        int nIn = Directory.GetFiles(dir, "input_*.pb").Length, nOut = Directory.GetFiles(dir, "output_*.pb").Length;
        var inputs = new Tensor[nIn];
        var expected = new Tensor[nOut];
        for (int i = 0; i < nIn; i++)
        {
            inputs[i] = OnnxImporter.LoadTensor(File.ReadAllBytes(Path.Combine(dir, $"input_{i}.pb"))).ToTensor();
        }

        for (int i = 0; i < nOut; i++)
        {
            expected[i] = OnnxImporter.LoadTensor(File.ReadAllBytes(Path.Combine(dir, $"output_{i}.pb"))).ToTensor();
        }

        return new Case
        {
            Model = model,
            Inputs = inputs,
            Expected = expected,
        };
    }

    private static Tensor[] RunCase(Case c, ExecutionMode mode, KernelPath? path, int threads)
    {
        using var s = Session.Create(c.Model, new SessionOptions { Mode = mode, IntraOpThreads = threads }, OpRegistry.Default, path, default);
        var outs = new Tensor[s.Outputs.Length];
        s.Run(Names(s.Inputs), c.Inputs, Names(s.Outputs), outs);
        return outs;
    }

    private static string[] Names(IEnumerable<TensorInfo> infos) => [.. infos.Select(i => i.Name)];

    private static double ScaledError(Tensor ours, Tensor reference)
    {
        Assert.Equal(reference.Shape, ours.Shape);
        ReadOnlySpan<float> a = ours.Data<float>(), b = reference.Data<float>();
        double max = 0, diff = 0;
        for (int i = 0; i < a.Length; i++)
        {
            max = Math.Max(max, Math.Abs(b[i]));
            diff = Math.Max(diff, Math.Abs((double)a[i] - b[i]));
        }

        return max == 0 ? diff : diff / max;
    }

    private static string Hash(Tensor[] outs)
    {
        using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (Tensor t in outs)
        {
            h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(t.Data<float>()));
        }

        return Convert.ToHexString(h.GetHashAndReset());
    }

    private static void DisposeAll(Tensor[] ts)
    {
        foreach (Tensor t in ts)
        {
            t.Dispose();
        }
    }

    [Theory]
    [InlineData(ExecutionMode.Fast)]
    [InlineData(ExecutionMode.Exact)]
    public void MatchesOnnxRuntimeOnRecordedCases(ExecutionMode mode)
    {
        int n = 0;
        var failures = new List<string>();
        foreach (string dir in CaseDirs())
        {
            using Case c = Load(dir);
            Tensor[] outs = RunCase(c, mode, null, 1);
            for (int i = 0; i < outs.Length; i++)
            {
                double e = ScaledError(outs[i], c.Expected[i]);
                if (!(e <= 2e-5))
                {
                    failures.Add($"{Path.GetFileName(dir)} output {i}: scaled error {e:g3}");
                }
            }

            DisposeAll(outs);
            n++;
        }

        Assert.True(n >= 60, $"only {n} cases found");
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    [Fact]
    public void ExactModeIsBitIdenticalOnEveryKernelPathAndThreadCount()
    {
        var failures = new List<string>();
        foreach (string dir in CaseDirs())
        {
            using Case c = Load(dir);
            string? first = null;
            foreach (KernelPath path in KernelPaths.All.Where(KernelPaths.IsSupported))
            {
                foreach (int threads in new[] { 1, 3 })
                {
                    Tensor[] outs = RunCase(c, ExecutionMode.Exact, path, threads);
                    string h = Hash(outs);
                    DisposeAll(outs);
                    first ??= h;
                    if (h != first)
                    {
                        failures.Add($"{Path.GetFileName(dir)} differs on {path} with {threads} threads");
                    }
                }
            }
        }

        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    [Fact]
    public void FastModeDoesNotDependOnThreadCountOrTheRun()
    {
        var failures = new List<string>();
        foreach (string dir in CaseDirs())
        {
            using Case c = Load(dir);
            Tensor[] a = RunCase(c, ExecutionMode.Fast, null, 1);
            Tensor[] b = RunCase(c, ExecutionMode.Fast, null, 4);
            Tensor[] again = RunCase(c, ExecutionMode.Fast, null, 4);
            if (Hash(a) != Hash(b) || Hash(b) != Hash(again))
            {
                failures.Add(Path.GetFileName(dir));
            }

            DisposeAll(a);
            DisposeAll(b);
            DisposeAll(again);
        }

        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    [Theory]
    [InlineData(ExecutionMode.Fast)]
    [InlineData(ExecutionMode.Exact)]
    public void SteadyStateRunAllocatesNothing(ExecutionMode mode)
    {
        foreach (string dir in CaseDirs().Where(d => d.Contains("_lstm_bidirectional_", StringComparison.Ordinal) ||
                                                      d.Contains("_gru_forward_", StringComparison.Ordinal)).Take(4))
        {
            using Case c = Load(dir);
            using var s = Session.Create(c.Model, new SessionOptions { Mode = mode, IntraOpThreads = 1 }, OpRegistry.Default, null, default);
            string[] inNames = Names(s.Inputs), outNames = Names(s.Outputs);
            var outs = new Tensor[outNames.Length];
            for (int i = 0; i < outs.Length; i++)
            {
                outs[i] = Tensor.Create(DType.F32, c.Expected[i].Shape);
            }

            for (int i = 0; i < 20; i++)
            {
                s.Run(inNames, c.Inputs, outNames, outs);
            }

            long best = long.MaxValue;
            for (int attempt = 0; attempt < 3 && best != 0; attempt++)
            {
                long before = GC.GetAllocatedBytesForCurrentThread();
                for (int i = 0; i < 100; i++)
                {
                    s.Run(inNames, c.Inputs, outNames, outs);
                }

                best = Math.Min(best, GC.GetAllocatedBytesForCurrentThread() - before);
            }

            DisposeAll(outs);
            Assert.True(best == 0, $"{Path.GetFileName(dir)} allocated {best} bytes");
        }
    }

    [Fact]
    public void StepProductEqualsTheExactGemmBitForBitOnEveryLaneWidth()
    {
        var rng = new Random(11);
        foreach ((int m, int n, int k) in new[] { (1, 768, 192), (3, 37, 5), (8, 130, 64), (2, 1, 7), (5, 16, 0), (1, 4, 1), (4, 100, 33) })
        {
            float[] a = new float[m * k], b = new float[k * n];
            for (int i = 0; i < a.Length; i++) { a[i] = (float)((rng.NextDouble() * 4) - 2); }
            for (int i = 0; i < b.Length; i++) { b[i] = (float)((rng.NextDouble() * 4) - 2); }
            if (a.Length > 3) { a[2] = float.NaN; }
            float[] expect = new float[m * n];
            Gemm.Run(KernelMode.Exact, KernelPath.Scalar, m, n, k, a, k, b, n, expect, n, 1);
            foreach (VecIsa isa in new[] { VecIsa.Scalar, VecIsa.V128, VecIsa.V256, VecIsa.V512 })
            {
                if (isa == VecIsa.V256 && !System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated) { continue; }
                if (isa == VecIsa.V512 && !System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated) { continue; }
                float[] got = new float[m * n];
                RecurrentGemv.Run(true, isa, m, n, k, a, k, b, n, got, n);
                for (int i = 0; i < got.Length; i++)
                {
                    uint x = BitConverter.SingleToUInt32Bits(got[i]), y = BitConverter.SingleToUInt32Bits(expect[i]);
                    if (float.IsNaN(got[i]) && float.IsNaN(expect[i])) { continue; }
                    Assert.True(x == y, $"m{m} n{n} k{k} {isa} element {i}");
                }
            }
        }
    }
}
