using System.Buffers.Binary;
using RecursiveAI.OneModel.Format;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;

namespace RecursiveAI.OneModel.Tests.Onnx;

public class InPlaceLoadTests
{
    private static readonly ModelOptions Options = new() { IntraOpThreads = 1, OptimizationLevel = OptimizationLevel.None };

    private static byte[] Container(byte[] onnx)
    {
        using IrModel graph = OnnxImporter.Load(onnx);
        using var stream = new MemoryStream();
        ModelFormat.Save(graph, stream);
        return stream.ToArray();
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public unsafe void CallerMemoryAndUnmanagedStreamEqualPath(bool format)
    {
        byte[] bytes = Build.AddModel();
        if (format)
        {
            bytes = Container(bytes);
        }

        string path = Path.Combine(Path.GetTempPath(), "model-memory-" + Guid.NewGuid().ToString("N"));
        try
        {
            File.WriteAllBytes(path, bytes);
            using var baseline = Model.Load(path, Options);
            float[] expected = baseline.Run(new float[] { 10, 20, 30, 40 });
            byte[] padded = new byte[bytes.Length + 29];
            bytes.CopyTo(padded, 13);
            ReadOnlyMemory<byte> memory = padded.AsMemory(13, bytes.Length);
            using var model = Model.Load(memory, Options);
            Assert.Equal(expected, model.Run(new float[] { 10, 20, 30, 40 }));
            using var session = Session.Load(memory, Options);
            Assert.Single(session.Inputs);
            fixed (byte* pointer = padded)
            {
                // A nonzero stream position selects the remaining model bytes without copying.
                using var exact = new UnmanagedMemoryStream(pointer, 13 + bytes.Length);
                exact.Position = 13;
                using var unmanaged = Model.Load(exact, Options);
                Assert.Equal(exact.Length, exact.Position);
                Assert.Equal(expected, unmanaged.Run(new float[] { 10, 20, 30, 40 }));
                Assert.True(exact.CanRead); // The model never owns the caller's stream.
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public unsafe void RawInitializerIsAViewNotACopy(bool format)
    {
        byte[] bytes = Build.AddModel();
        if (format)
        {
            bytes = Container(bytes);
        }

        using IrModel graph = format ? ModelFormat.Load(bytes.AsMemory()) : OnnxImporter.Load(bytes.AsMemory());
        ReadOnlySpan<byte> raw = graph.Graph.Initializers[0].RawData;
        int index = bytes.AsSpan().IndexOf(raw);
        Assert.True(index >= 0);
        // Deliberate mutation solely to verify aliasing; production callers must keep storage unchanged.
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(index), 19f);
        Assert.Equal(19f, BinaryPrimitives.ReadSingleLittleEndian(graph.Graph.Initializers[0].RawData));
        fixed (byte* pointer = bytes)
        {
            using var stream = new UnmanagedMemoryStream(pointer, bytes.Length);
            using IrModel fromStream = format ? ModelFormat.Load(stream) : OnnxImporter.Load(stream);
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(index), 23f);
            Assert.Equal(23f, BinaryPrimitives.ReadSingleLittleEndian(fromStream.Graph.Initializers[0].RawData));
        }
    }

    [Fact]
    public unsafe void UnmanagedStreamAtOrBeyondEndFailsWithAnImportError()
    {
        byte[] bytes = new byte[64];
        fixed (byte* pointer = bytes)
        {
            using var stream = new UnmanagedMemoryStream(pointer, bytes.Length);
            stream.Position = bytes.Length;
            Assert.Throws<OnnxImportException>(() => Session.Load(stream));
            stream.Position = bytes.Length + 16;
            Assert.Throws<OnnxImportException>(() => Session.Load(stream));
        }
    }

    private sealed class OversizedStream(byte[] bytes) : MemoryStream(bytes)
    {
        public override long Length => 1L << 31;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RefusedBufferedStreamRetainsTheFormatErrorKind(bool format)
    {
        byte[] bytes = format ? Container(Build.AddModel()) : Build.AddModel();
        using var stream = new OversizedStream(bytes);
        if (format)
        {
            var e = Assert.Throws<ModelFormatException>(() => Session.Load(stream));
            Assert.Equal(ModelFormatErrorKind.Unsupported, e.Kind);
        }
        else
        {
            var e = Assert.Throws<OnnxImportException>(() => Session.Load(stream));
            Assert.Equal(OnnxErrorKind.Unsupported, e.Kind);
        }
    }

    [Fact]
    public void EmbeddedResourceStreamLoadsInPlace()
    {
        using Stream stream = typeof(InPlaceLoadTests).Assembly.GetManifestResourceStream("test-model.onnx")!;
        Assert.IsAssignableFrom<UnmanagedMemoryStream>(stream);
        using var model = Model.Load(stream, new ModelOptions { IntraOpThreads = 1 });
        Assert.NotNull(model);
    }

    [Fact]
    public unsafe void LargeInPlaceLoadAllocatesMetadataNotSerializedWeights()
    {
        int megabytes = Environment.GetEnvironmentVariable("ONEMODEL_LARGE_FILES") == "1" ? 1024 : 4;
        int count = megabytes * 1024 * 1024 / 4;
        // Assemble protobuf framing without a second weight-sized managed array.
        byte[] header = new Pb().PackedInts(1, count).Int(2, 1).Str(8, "w").Tag(9, 2).Varint((ulong)count * 4).ToArray();
        byte[] graphHead = new Pb().Str(2, "memory").Tag(5, 2).Varint((ulong)(header.Length + (long)count * 4)).ToArray();
        byte[] modelHead = new Pb().Int(1, 8).Msg(8, new Pb().Str(1, "").Int(2, 13)).Tag(7, 2)
            .Varint((ulong)(graphHead.Length + header.Length + (long)count * 4)).ToArray();
        byte[] bytes = new byte[modelHead.Length + graphHead.Length + header.Length + count * 4];
        modelHead.CopyTo(bytes, 0);
        graphHead.CopyTo(bytes, modelHead.Length);
        header.CopyTo(bytes, modelHead.Length + graphHead.Length);
        bytes.AsSpan(modelHead.Length + graphHead.Length + header.Length).Fill(0x3f);
        OnnxImporter.Load(Build.AddModel()).Dispose(); // Warm metadata paths.
        long before = GC.GetAllocatedBytesForCurrentThread();
        using IrModel memory = OnnxImporter.Load(bytes.AsMemory());
        long managed = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(managed < 1_000_000, $"Import allocated {managed} bytes for {bytes.Length} input bytes.");
        fixed (byte* pointer = bytes)
        {
            using var stream = new UnmanagedMemoryStream(pointer, bytes.Length);
            before = GC.GetAllocatedBytesForCurrentThread();
            using IrModel unmanaged = OnnxImporter.Load(stream);
            long unmanagedAllocated = GC.GetAllocatedBytesForCurrentThread() - before;
            Assert.True(unmanagedAllocated < 1_000_000);
            string? log = Environment.GetEnvironmentVariable("ONEMODEL_MEMORY_LOG");
            if (log is not null)
            {
                using var process = System.Diagnostics.Process.GetCurrentProcess();
                File.AppendAllText(log, $"in-place serialized={bytes.Length}, ROM managed={managed}, unmanaged stream managed={unmanagedAllocated}, RSS={process.WorkingSet64}, peak={process.PeakWorkingSet64}\n");
            }
        }
    }
}
