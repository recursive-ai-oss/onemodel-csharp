using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Tests.Tensors;
using RecursiveAI.OneModel.Training;

namespace RecursiveAI.OneModel.Tests.Training;

[Collection("autograd")]
public class CheckpointTests
{
    private static OracleFile Enc => OracleFile.Load("encoder", "training");
    private static OracleFile CnnData => OracleFile.Load("cnn", "training");

    private static byte[] SaveBytes(NnModule m, Optimizer? o, Dictionary<string, long>? extra = null)
    {
        using var ms = new MemoryStream();
        Checkpoint.Save(ms, m, o, extra);
        return ms.ToArray();
    }

    [Fact]
    public void RoundTripIsBitExactAndSavingIsDeterministic()
    {
        TrainCase a = TrainCase.Encoder(Enc, dropout: 0.1f);
        for (int i = 0; i < 6; i++)
        {
            a.Step(i);
        }

        byte[] bytes = SaveBytes(a.Model, a.Opt, new() { ["step"] = 6, ["epoch"] = 0 });
        Assert.Equal(bytes, SaveBytes(a.Model, a.Opt, new() { ["epoch"] = 0, ["step"] = 6 }));   // extras are written in key order

        // A fresh case has different weights, state and step count; loading must overwrite all of it.
        TrainCase b = TrainCase.Encoder(Enc, dropout: 0.1f, () => new Generator(999));
        foreach ((_, Tensor t) in b.Model.NamedParameters())
        {
            t.MutableData<float>().Fill(0.25f);
        }

        CheckpointInfo info = Checkpoint.Load(new MemoryStream(bytes), b.Model, b.Opt);
        Assert.True(info.HadOptimizer);
        Assert.Equal(6, info.Extra["step"]);
        Assert.Equal(TrainCase.StateHash(a.Model, a.Opt), TrainCase.StateHash(b.Model, b.Opt));
        Assert.Equal(bytes, SaveBytes(b.Model, b.Opt, new() { ["step"] = 6, ["epoch"] = 0 }));
        Assert.Equal(6, b.Opt.StepCount);
    }

    private static void ResumeEqualsUninterrupted(Func<TrainCase> make, int total, int cut)
    {
        TrainCase whole = make();
        var lossesWhole = new List<float>();
        for (int i = 0; i < total; i++)
        {
            lossesWhole.Add(whole.Step(i));
        }

        TrainCase first = make();
        var lossesSplit = new List<float>();
        for (int i = 0; i < cut; i++)
        {
            lossesSplit.Add(first.Step(i));
        }

        string path = Path.Combine(Path.GetTempPath(), "onemodel-ckpt-" + Guid.NewGuid().ToString("N") + ".ckpt");
        try
        {
            Checkpoint.Save(path, first.Model, first.Opt, new Dictionary<string, long> { ["step"] = cut });
            TrainCase second = make();
            CheckpointInfo info = Checkpoint.Load(path, second.Model, second.Opt);
            for (int i = (int)info.Extra["step"]; i < total; i++)
            {
                lossesSplit.Add(second.Step(i));
            }

            Assert.Equal(lossesWhole, lossesSplit);
            Assert.Equal(TrainCase.StateHash(whole.Model, whole.Opt), TrainCase.StateHash(second.Model, second.Opt));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ResumedTransformerWithDropoutEqualsUninterruptedRun() =>
        ResumeEqualsUninterrupted(() => TrainCase.Encoder(Enc, dropout: 0.1f), total: 20, cut: 9);

    [Fact]
    public void ResumedConvNetWithBatchNormClipAndCosineEqualsUninterruptedRun() =>
        ResumeEqualsUninterrupted(() => TrainCase.Cnn(CnnData), total: 24, cut: 11);

    [Fact]
    public void ResumedSgdMomentumEqualsUninterruptedRun() =>
        ResumeEqualsUninterrupted(() => TrainCase.Mlp(OracleFile.Load("mlp", "training")), total: 30, cut: 13);

    [Fact]
    public void ModelOnlyCheckpointLoadsWithoutOptimizerAndOptimizerLoadNeedsOne()
    {
        TrainCase a = TrainCase.Mlp(OracleFile.Load("mlp", "training"));
        a.Step(0);
        byte[] modelOnly = SaveBytes(a.Model, null);
        TrainCase b = TrainCase.Mlp(OracleFile.Load("mlp", "training"));
        Assert.False(Checkpoint.Load(new MemoryStream(modelOnly), b.Model).HadOptimizer);
        Assert.Equal(TrainCase.StateHash(a.Model, null), TrainCase.StateHash(b.Model, null));
        CheckpointException e = Assert.Throws<CheckpointException>(() => Checkpoint.Load(new MemoryStream(modelOnly), b.Model, b.Opt));
        Assert.Equal(CheckpointErrorKind.Mismatch, e.Kind);

        // A file with optimizer state loads fine into a model alone.
        byte[] full = SaveBytes(a.Model, a.Opt);
        Assert.True(Checkpoint.Load(new MemoryStream(full), b.Model).HadOptimizer);
    }

    [Fact]
    public void MismatchesAreReportedAndNothingIsModified()
    {
        TrainCase a = TrainCase.Mlp(OracleFile.Load("mlp", "training"));
        a.Step(0);
        byte[] bytes = SaveBytes(a.Model, a.Opt);

        var g = new Generator(5);
        using var other = new Sequential(new Linear(20, 32, g), Lambda.Relu(), new Linear(32, 5, g));   // different output width
        string before = TrainCase.StateHash(other, null);
        CheckpointException e = Assert.Throws<CheckpointException>(() => Checkpoint.Load(new MemoryStream(bytes), other));
        Assert.Equal(CheckpointErrorKind.Mismatch, e.Kind);
        Assert.Equal(before, TrainCase.StateHash(other, null));

        TrainCase c = TrainCase.Cnn(CnnData);
        Assert.Equal(CheckpointErrorKind.Mismatch, Assert.Throws<CheckpointException>(() => Checkpoint.Load(new MemoryStream(bytes), c.Model)).Kind);

        // Optimizer of another kind.
        TrainCase d = TrainCase.Mlp(OracleFile.Load("mlp", "training"));
        using var adam = new Adam(d.Model.Parameters());
        Assert.Equal(CheckpointErrorKind.Mismatch, Assert.Throws<CheckpointException>(() => Checkpoint.Load(new MemoryStream(bytes), d.Model, adam)).Kind);
    }

    [Fact]
    public void HostileFilesFailWithTypedErrors()
    {
        var g = new Generator(1);
        using var model = new Sequential(new Linear(3, 2, g), new BatchNorm2d(2));
        using var opt = new Adam(model.Parameters());
        byte[] good = SaveBytes(model, opt, new() { ["step"] = 1 });

        Assert.Equal(CheckpointErrorKind.BadHeader, Assert.Throws<CheckpointException>(() => Checkpoint.Load(new MemoryStream(good[..40]), model, opt)).Kind);
        Assert.Equal(CheckpointErrorKind.BadHeader, Assert.Throws<CheckpointException>(() => Checkpoint.Load(new MemoryStream(good[..^1]), model, opt)).Kind);
        Assert.Equal(CheckpointErrorKind.BadHeader, Assert.Throws<CheckpointException>(() => Checkpoint.Load(new MemoryStream([.. good, 0]), model, opt)).Kind);
        Assert.Equal(CheckpointErrorKind.BadHeader, Assert.Throws<CheckpointException>(() => Checkpoint.Load(new MemoryStream([]), model, opt)).Kind);

        // Every single-byte corruption anywhere in the file either loads (a changed weight) or fails with CheckpointException,
        // never with another exception type.
        for (int i = 0; i < good.Length; i++)
        {
            byte[] bad = (byte[])good.Clone();
            bad[i] ^= 0xFF;
            try
            {
                Checkpoint.Load(new MemoryStream(bad), model, opt);
            }
            catch (CheckpointException)
            {
            }
        }

        // Hand-made header lies: manifest length beyond the file, data offset not a multiple of 64.
        byte[] lie = (byte[])good.Clone();
        System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(lie.AsSpan(24), ulong.MaxValue);
        Assert.Equal(CheckpointErrorKind.BadHeader, Assert.Throws<CheckpointException>(() => Checkpoint.Load(new MemoryStream(lie), model, opt)).Kind);
        lie = (byte[])good.Clone();
        System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(lie.AsSpan(32), 65);
        Assert.Equal(CheckpointErrorKind.BadHeader, Assert.Throws<CheckpointException>(() => Checkpoint.Load(new MemoryStream(lie), model, opt)).Kind);
    }
}
