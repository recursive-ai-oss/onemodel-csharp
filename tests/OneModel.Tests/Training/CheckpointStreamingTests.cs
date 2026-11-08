using System.Buffers.Binary;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Training;

namespace RecursiveAI.OneModel.Tests.Training;

[Collection("autograd")]
public class CheckpointStreamingTests
{
    private sealed class Weights : NnModule
    {
        public Weights(int count, int length)
        {
            for (int i = 0; i < count; i++)
            {
                RegisterParameter("p" + i, Tensor.Zeros(DType.F32, Shape.Create(length)));
            }
        }

        public override Tensor Forward(Tensor input) => input;
    }

    [Fact]
    public void WriterMatchesOriginalLayoutAndStreamPositionIsRelative()
    {
        using var model = new Weights(1, 3);
        model.Parameters()[0].MutableData<float>()[0] = -0f;
        model.Parameters()[0].MutableData<float>()[1] = 1.25f;
        model.Parameters()[0].MutableData<float>()[2] = BitConverter.Int32BitsToSingle(0x7fc12345);
        using var stream = new MemoryStream();
        stream.Write(new byte[17]);
        Checkpoint.Save(stream, model);
        byte[] actual = stream.ToArray()[17..];
        // Independently assembled version-1 bytes, including the original writer's padding and field order.
        byte[] manifest = System.Text.Encoding.UTF8.GetBytes("""{"format":"onemodel-checkpoint","version":1,"model":[{"name":"p0","dtype":"f32","shape":[3],"offset":0,"length":12}],"extra":[]}""");
        int data = (64 + manifest.Length + 63) & ~63;
        byte[] expected = new byte[data + 64];
        "ONEMCKPT"u8.CopyTo(expected);
        BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan(8), 1);
        BinaryPrimitives.WriteUInt64LittleEndian(expected.AsSpan(16), 64);
        BinaryPrimitives.WriteUInt64LittleEndian(expected.AsSpan(24), (ulong)manifest.Length);
        BinaryPrimitives.WriteUInt64LittleEndian(expected.AsSpan(32), (ulong)data);
        BinaryPrimitives.WriteUInt64LittleEndian(expected.AsSpan(40), 64);
        BinaryPrimitives.WriteUInt64LittleEndian(expected.AsSpan(48), (ulong)expected.Length);
        manifest.CopyTo(expected, 64);
        BinaryPrimitives.WriteInt32LittleEndian(expected.AsSpan(data), unchecked((int)0x80000000));
        BinaryPrimitives.WriteSingleLittleEndian(expected.AsSpan(data + 4), 1.25f);
        BinaryPrimitives.WriteInt32LittleEndian(expected.AsSpan(data + 8), 0x7fc12345);
        Assert.Equal(expected, actual);
        stream.Position = 17;
        Checkpoint.Load(stream, model);
        Assert.Equal(stream.Length, stream.Position);
        using var roundTrip = new MemoryStream();
        Checkpoint.Save(roundTrip, model);
        Assert.Equal(expected, roundTrip.ToArray());
    }

    [Fact]
    public void NonSeekableInputIsTypedAndNonSeekableOutputWorks()
    {
        using var model = new Weights(1, 3);
        using var backing = new MemoryStream();
        using var stream = new ForwardOnly(backing);
        Checkpoint.Save(stream, model);
        Assert.True(backing.Length > 64);
        CheckpointException e = Assert.Throws<CheckpointException>(() => Checkpoint.Load(stream, model));
        Assert.Equal(CheckpointErrorKind.UnsupportedStream, e.Kind);
    }

    private sealed class ForwardOnly(Stream inner) : Stream
    {
        public override bool CanRead => true;
        public override bool CanWrite => true;
        public override bool CanSeek => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() => inner.Flush();
        public override int Read(byte[] buffer, int offset, int count) => inner.Read(buffer, offset, count);
        public override void Write(byte[] buffer, int offset, int count) => inner.Write(buffer, offset, count);
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
    }

    [Fact]
    public void MultiGigabyteFileRoundTripsWithOptimizer()
    {
        if (Environment.GetEnvironmentVariable("ONEMODEL_LARGE_FILES") != "1")
        {
            return;
        }

        // 2.4 GB of parameters and 2.4 GB of momentum, with no tensor near the per-tensor limit.
        using var model = new Weights(12, 50_000_000);
        using var opt = new Sgd(model.Parameters(), 0.125f, momentum: 0.9f);
        string path = Path.Combine(Path.GetTempPath(), "checkpoint-large-" + Guid.NewGuid().ToString("N"));
        var tensors = model.StateDict().Concat(opt.StateTensors).ToArray();
        void Fill(bool clear)
        {
            for (int j = 0; j < tensors.Length; j++)
            {
                if (tensors[j].Tensor.DType == DType.I32)
                {
                    tensors[j].Tensor.MutableData<int>().Fill(clear ? 0 : 123);
                    continue;
                }

                Span<float> values = tensors[j].Tensor.MutableData<float>();
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = clear ? 0 : (i % 8191 - 4096) * (j + 1f);
                }
            }
        }

        void Verify()
        {
            for (int j = 0; j < tensors.Length; j++)
            {
                if (tensors[j].Tensor.DType == DType.I32)
                {
                    Assert.Equal(123, tensors[j].Tensor.Data<int>()[0]);
                    continue;
                }

                ReadOnlySpan<float> values = tensors[j].Tensor.Data<float>();
                for (int i = 0; i < values.Length; i++)
                {
                    if (BitConverter.SingleToInt32Bits(values[i]) != BitConverter.SingleToInt32Bits((i % 8191 - 4096) * (j + 1f)))
                    {
                        Assert.Fail($"Tensor {j}, element {i} differs.");
                    }
                }
            }
        }

        void Memory(string phase)
        {
            string? log = Environment.GetEnvironmentVariable("ONEMODEL_MEMORY_LOG");
            if (log is not null)
            {
                using var process = System.Diagnostics.Process.GetCurrentProcess();
                File.AppendAllText(log, $"{phase}: RSS={process.WorkingSet64}, peak={process.PeakWorkingSet64}, parameters=2400000000, momentum=2400000000\n");
            }
        }

        try
        {
            Fill(false);
            Memory("before-save");
            Checkpoint.Save(path, model, opt);
            Memory("after-path-save");
            Assert.True(new FileInfo(path).Length > int.MaxValue);
            Fill(true);
            Checkpoint.Load(path, model, opt);
            Memory("after-path-load");
            Verify();
            using (FileStream output = File.Create(path))
            {
                Checkpoint.Save(output, model, opt);
            }

            Memory("after-stream-save");
            Fill(true);
            using (FileStream input = File.OpenRead(path))
            {
                Checkpoint.Load(input, model, opt);
            }

            Memory("after-stream-load");
            Verify();
        }
        finally
        {
            File.Delete(path);
        }
    }
}
