using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tests.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu.Tests;

/// <summary>A long region is handed to the device in several submissions; results and synchronization must not change.</summary>
public class SubmissionTests
{
    private const int F = 64;

    // Six dependent products: each reads what the one before wrote, so a missing barrier at a cut would show.
    private static byte[] Chain()
    {
        var b = new ModelBuilder().Input("x", 1, "N", F).Output("y", 1, "N", F);
        string prev = "x";
        for (int i = 0; i < 6; i++)
        {
            string next = i == 5 ? "y" : $"t{i}";
            b.Initializer($"w{i}", GpuTestHost.Random(F * F, 10 + i, 0.3f), F, F).Node("MatMul", [prev, $"w{i}"], [$"m{i}"]).Node("Tanh", [$"m{i}"], [next]);
            prev = next;
        }

        return b.Build();
    }

    [Theory]
    [InlineData(ArithmeticMode.Fast)]
    [InlineData(ArithmeticMode.Exact)]
    public void ARegionCutIntoSeveralSubmissionsGivesTheSameBitsAndNoMessages(ArithmeticMode mode)
    {
        GpuTestHost.Require();
        using Tensor x = Tensor.FromSpan<float>(GpuTestHost.Random(40 * F, 1), Shape.Create(40, F));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        byte[] model = Chain();
        string whole, cut;
        using (Session s = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true)))
        using (SessionOutputs o = s.Run(feed))
        {
            whole = GpuTestHost.Hash(o["y"].Data<float>());
        }

        using Session gpu = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true));
        ((GpuBackend)gpu.Backend!).SplitWorkBudget = 1;   // a cut after every product
        using (SessionOutputs o = gpu.Run(feed))
        {
            cut = GpuTestHost.Hash(o["y"].Data<float>());
        }

        Assert.Equal(whole, cut);
        Assert.True(OneModelGpu.GetInfo(gpu).Timings.GpuMilliseconds >= 0);
        GpuTestHost.AssertValidationClean(gpu);
    }

    // A region handed over in growing pieces while it is still being recorded must give the same bits, the same counts and no messages.
    [Theory]
    [InlineData(ArithmeticMode.Fast, 1)]
    [InlineData(ArithmeticMode.Fast, 3)]
    [InlineData(ArithmeticMode.Exact, 2)]
    public void ARegionHandedOverWhileRecordingGivesTheSameBitsAndCounts(ArithmeticMode mode, int firstPiece)
    {
        GpuTestHost.Require();
        using Tensor x = Tensor.FromSpan<float>(GpuTestHost.Random(40 * F, 1), Shape.Create(40, F));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        byte[] model = Chain();
        (string Hash, GpuTimings Timings) Run(int pipe)
        {
            using IDisposable scope = GpuTestHost.With(new GpuOptions { CollectTimings = true, PipelineFirstDispatches = pipe });
            using Session s = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true));
            using (s.Run(feed)) { }
            OneModelGpu.ResetTimings(s);
            string hash;
            using (SessionOutputs o = s.Run(feed))
            {
                hash = GpuTestHost.Hash(o["y"].Data<float>());
            }

            GpuTestHost.AssertValidationClean(s);
            return (hash, OneModelGpu.GetInfo(s).Timings);
        }

        (string whole, GpuTimings wholeTimings) = Run(0);
        (string piecewise, GpuTimings pieceTimings) = Run(firstPiece);
        Assert.Equal(whole, piecewise);
        Assert.True(wholeTimings.Dispatches > 0);
        Assert.Equal(wholeTimings.Dispatches, pieceTimings.Dispatches);
        Assert.Equal(wholeTimings.ComputePasses, pieceTimings.ComputePasses);
        Assert.True(pieceTimings.RecordMilliseconds >= 0 && pieceTimings.SubmitMilliseconds >= 0);
    }
}
