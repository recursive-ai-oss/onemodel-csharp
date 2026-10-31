using System.Security.Cryptography;
using System.Text;
using RecursiveAI.OneModel.Graph;
using static RecursiveAI.OneModel.Tests.Graph.Operators.Norms.NormHarness;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.Norms;

/// <summary>
/// One hash over the exact-mode outputs of every operator in this family on seeded inputs. The same value must come out on every
/// operating system, CPU, instruction set (run it with DOTNET_EnableAVX512=0, DOTNET_EnableAVX2=0, DOTNET_EnableHWIntrinsic=0) and
/// thread count. Float outputs are hashed as raw bits, so a changed rounding anywhere changes the hash.
/// </summary>
public class ExactGoldenTests
{
    // Update only for an intended change to a kernel's arithmetic or order, and say why in the change.
    private const string Golden = "8095cc17d5130bb134301fca02fd17395e91e7c2eeceb4ffaefef7c4ba2106dd";

    private static object[] Dims(int[] d) => [.. d.Cast<object>()];

    private static void Add(IncrementalHash h, string name, Out[] outs)
    {
        h.AppendData(Encoding.UTF8.GetBytes(name));
        foreach (Out o in outs)
        {
            if (o.F is not null)
            {
                float[] f = o.F.Select(v => float.IsNaN(v) ? BitConverter.UInt32BitsToSingle(0x7FC00000u) : v).ToArray();
                h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes<float>(f));
            }
            else
            {
                h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes<long>(o.L!));
            }
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(6)]
    public void ExactOutputsHaveOneHashEverywhere(int threads)
    {
        var rng = new Random(2026);
        using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        ArithmeticMode m = ArithmeticMode.Exact;

        foreach (int cols in new[] { 1, 5, 16, 17, 33, 100, 384 })
        {
            int[] shape = [3, 4, cols];
            float[] x = Random(rng, 12 * cols, -4, 4), s = Random(rng, cols, 0.5, 1.5), b = Random(rng, cols);
            byte[] ln = new ModelBuilder(17).Input("x", 1, 3, 4, cols).Initializer("s", s, cols).Initializer("b", b, cols)
                .Output("y", 1, 3, 4, cols).Output("m", 1, 3, 4, 1).Output("i", 1, 3, 4, 1)
                .Node("LayerNormalization", ["x", "s", "b"], ["y", "m", "i"]).Build();
            Add(h, $"ln{cols}", Run(ln, m, null, threads, 3, (x, shape)));
            byte[] rms = new ModelBuilder(23).Input("x", 1, 3, 4, cols).Initializer("s", s, cols).Output("y", 1, 3, 4, cols)
                .Node("RMSNormalization", ["x", "s"], ["y"]).Build();
            Add(h, $"rms{cols}", Run(rms, m, null, threads, 1, (x, shape)));
            foreach (string op in new[] { "Softmax", "LogSoftmax", "Hardmax" })
            {
                foreach (int axis in new[] { -1, 1, 0 })
                {
                    byte[] sm = new ModelBuilder(13).Input("x", 1, 3, 4, cols).Output("y", 1, 3, 4, cols).Node(op, ["x"], ["y"], ("axis", (long)axis)).Build();
                    Add(h, $"{op}{cols}/{axis}", Run(sm, m, null, threads, 1, (x, shape)));
                }
            }
        }

        {
            int[] shape = [2, 6, 21];
            float[] x = Random(rng, 252, -3, 3), s = Random(rng, 6, 0.5, 1.5), b = Random(rng, 6);
            byte[] inorm = new ModelBuilder(6).Input("x", 1, 2, 6, 21).Initializer("s", s, 6).Initializer("b", b, 6).Output("y", 1, 2, 6, 21)
                .Node("InstanceNormalization", ["x", "s", "b"], ["y"]).Build();
            Add(h, "instance", Run(inorm, m, null, threads, 1, (x, shape)));
            byte[] g21 = new ModelBuilder(21).Input("x", 1, 2, 6, 21).Initializer("s", s, 6).Initializer("b", b, 6).Output("y", 1, 2, 6, 21)
                .Node("GroupNormalization", ["x", "s", "b"], ["y"], ("num_groups", 3L)).Build();
            Add(h, "group21", Run(g21, m, null, threads, 1, (x, shape)));
            byte[] g18 = new ModelBuilder(18).Input("x", 1, 2, 6, 21).Initializer("s", s[..3], 3).Initializer("b", b[..3], 3).Output("y", 1, 2, 6, 21)
                .Node("GroupNormalization", ["x", "s", "b"], ["y"], ("num_groups", 3L)).Build();
            Add(h, "group18", Run(g18, m, null, threads, 1, (x, shape)));
        }

        string[] ops = ["ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2", "ReduceSumSquare", "ReduceLogSum", "ReduceLogSumExp"];
        int[] rshape = [5, 7, 19, 3];
        float[] rx = Random(rng, Count(rshape), 0.25, 1.75);
        foreach (string op in ops)
        {
            for (int mask = 1; mask < 16; mask++)
            {
                int[] axes = [.. Enumerable.Range(0, 4).Where(i => (mask >> i & 1) != 0)];
                int[] outDims = [.. rshape.Where((_, i) => (mask >> i & 1) == 0)];
                var mb = new ModelBuilder(18).Input("x", 1, Dims(rshape)).Initializer("axes", axes.Select(a => (long)a).ToArray(), axes.Length)
                    .Output("y", 1, Dims(outDims));
                byte[] model = mb.Node(op, ["x", "axes"], ["y"], ("keepdims", 0L)).Build();
                Add(h, $"{op}/{mask}", Run(model, m, null, threads, 1, (rx, rshape)));
            }
        }

        // A reduction long enough to spread over several chunks, in both layouts.
        int[] big = [70, 2000, 3];
        float[] bx = Random(rng, Count(big), -1, 1);
        foreach (long[] axes in new[] { new long[] { 0 }, new long[] { 1 }, new long[] { 2 }, new long[] { 0, 2 } })
        {
            byte[] model = new ModelBuilder(18).Input("x", 1, Dims(big)).Initializer("axes", axes, axes.Length).Output("y", 1)
                .Node("ReduceSum", ["x", "axes"], ["y"], ("keepdims", 0L)).Build();
            Add(h, "big" + string.Join('_', axes), Run(model, m, null, threads, 1, (bx, big)));
        }

        float[] ax = Enumerable.Range(0, 3 * 5 * 7).Select(_ => (float)rng.Next(4)).ToArray();
        foreach (string op in new[] { "ArgMax", "ArgMin" })
        {
            byte[] model = new ModelBuilder(13).Input("x", 1, 3, 5, 7).Output("y", 7, 3, 1, 7).Node(op, ["x"], ["y"], ("axis", 1L), ("select_last_index", 1L)).Build();
            Add(h, op, Run(model, m, null, threads, 1, (ax, [3, 5, 7])));
        }

        float[] cx = Random(rng, 3 * 5 * 7);
        byte[] cum = new ModelBuilder(14).Input("x", 1, 3, 5, 7).Initializer("a", new long[] { 1 }).Output("y", 1, 3, 5, 7)
            .Node("CumSum", ["x", "a"], ["y"], ("reverse", 1L), ("exclusive", 1L)).Build();
        Add(h, "cumsum", Run(cum, m, null, threads, 1, (cx, [3, 5, 7])));

        string hex = Convert.ToHexString(h.GetHashAndReset()).ToLowerInvariant();
        Assert.True(hex == Golden, $"golden hash is {hex}");
    }
}
