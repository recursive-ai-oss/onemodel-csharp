using System.Runtime.InteropServices;
using RecursiveAI.OneModel.Kernels;

namespace RecursiveAI.OneModel.Tests.Kernels;

public class Int8PoolTests
{
    [Fact]
    public void MatchesTheGenericLoopForBothTypes()
    {
        (int H, int W, int KH, int KW, int SH, int SW, int PH, int PW, int OH, int OW)[] cases =
        [
            (112, 112, 3, 3, 2, 2, 1, 1, 56, 56),
            (13, 17, 3, 3, 1, 1, 1, 1, 13, 17),
            (9, 31, 2, 2, 2, 2, 0, 0, 4, 15),
            (9, 31, 2, 2, 2, 2, 0, 0, 5, 16),
            (7, 7, 5, 5, 1, 1, 2, 2, 7, 7),
            (10, 33, 3, 1, 1, 3, 1, 0, 10, 11),
            (5, 5, 3, 3, 3, 3, 2, 2, 3, 3),
            (1, 40, 1, 7, 1, 2, 0, 3, 1, 20),
            (6, 6, 4, 4, 1, 1, 3, 3, 9, 9),
        ];
        var rng = new Random(5);
        foreach (var c in cases)
        {
            var g = new PoolGeom(new Dim3(1, c.H, c.W), new Dim3(1, c.KH, c.KW), new Dim3(1, c.SH, c.SW), new Dim3(1, 1, 1),
                new Dim3(0, c.PH, c.PW), new Dim3(0, c.PH, c.PW), new Dim3(1, c.OH, c.OW));
            const int planes = 7;
            byte[] x = new byte[planes * c.H * c.W];
            rng.NextBytes(x);
            foreach (bool signed in new[] { false, true })
            {
                byte[] want = new byte[planes * c.OH * c.OW];
                byte[] got = new byte[want.Length];
                if (signed)
                {
                    Pooling.Max<sbyte>(g, planes, MemoryMarshal.Cast<byte, sbyte>(x.AsSpan()), MemoryMarshal.Cast<byte, sbyte>(want.AsSpan()), default, false);
                }
                else
                {
                    Pooling.Max<byte>(g, planes, x, want, default, false);
                }

                foreach (int threads in new[] { 1, 4 })
                {
                    Array.Clear(got);
                    Int8Pool.Run(g, planes, x, got, signed, threads);
                    Assert.True(want.AsSpan().SequenceEqual(got), $"{c} signed={signed} threads={threads}");
                }
            }
        }
    }
}
