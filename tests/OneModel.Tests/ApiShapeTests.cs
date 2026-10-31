using System.Runtime.CompilerServices;
using RecursiveAI.OneModel.Format;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests;

/// <summary>The public API shape: ownership, result containers, errors, limits, operators, scopes, format detection.</summary>
public class ApiShapeTests
{
    private static string GraphPath(string name, [CallerFilePath] string file = "") =>
        Path.Combine(Path.GetDirectoryName(file)!, "..", "..", "testdata", "graphs", name);

    private sealed class EndlessStream : Stream
    {
        public long Served;
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count)
        {
            buffer.AsSpan(offset, count).Fill(0x08);
            Served += count;
            return count;
        }

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    // ---- C1 ownership ---------------------------------------------------------------------------------------------------

    [Fact]
    public void OrdinaryOwnershipPatternsDoNotThrow()
    {
        using var session = Session.Load(GraphPath("mlp.onnx"));
        using Tensor x = Tensor.FromSpan<float>(new float[16], 1, 16);
        var outs = new Tensor[1];
        for (int i = 0; i < 3; i++)
        {
            session.Run(["x"], [x], ["y"], outs);
            using (Tensor y = outs[0]) // take it out of the array and dispose it, as any C# code would
            {
                Assert.Equal(8, y.Length);
            }
        }

        Tensor d = Tensor.Zeros(DType.F32, 4);
        Tensor alias = d;
        static void Consume(Tensor t)
        {
            using (t)
            {
                Assert.Equal(4, t.Length);
            }
        }

        Consume(d);
        d.Dispose();
        alias.Dispose();
        d.Dispose();
        Assert.True(alias.IsReleased);
        Assert.Throws<StaleTensorException>(() => alias.Data<float>());
        Assert.IsAssignableFrom<ObjectDisposedException>(Assert.Throws<StaleTensorException>(() => d.Data<float>()));
        new Tensor().Dispose();
    }

    // ---- C3 named run and result container --------------------------------------------------------------------------------

    [Fact]
    public void DictionaryRunReturnsOutputsThatOneUsingReleases()
    {
        using var session = Session.Load(GraphPath("mlp.onnx"));
        using Tensor x = Tensor.FromSpan<float>(new float[32], 2, 16);
        Tensor y;
        using (SessionOutputs outs = session.Run(new Dictionary<string, Tensor> { ["x"] = x }))
        {
            Assert.Single(outs);
            y = outs["y"];
            Assert.Equal(Shape.Create(2, 2, 4), y.Shape);
            Assert.Equal("y", outs.Names[0]);
            Assert.Throws<KeyNotFoundException>(() => outs["nope"]);
            Assert.False(outs.TryGetValue("nope", out _));
            Assert.Equal(y.Length, outs[0].Length);
        }

        Assert.True(y.IsReleased);
        var e = Assert.Throws<SessionException>(() => session.Run(new Dictionary<string, Tensor> { ["z"] = x }));
        Assert.Equal(SessionErrorKind.InvalidInput, e.Kind);
    }

    // ---- C4 one base type, BCL file errors --------------------------------------------------------------------------------

    [Fact]
    public void LibraryErrorsShareABaseAndFileErrorsAreTheBclsOwn()
    {
        var onnx = Assert.ThrowsAny<OneModelException>(() => OnnxImporter.Load(new byte[] { 0xFF, 0xFF, 0xFF }));
        Assert.Equal(OneModelErrorCategory.OnnxImport, onnx.Category);
        var fmt = Assert.ThrowsAny<OneModelException>(() => ModelFormat.Load("ONEMODEL"u8.ToArray()));
        Assert.Equal(OneModelErrorCategory.ModelFormat, fmt.Category);
        var session = Assert.ThrowsAny<OneModelException>(() => Session.Load(GraphPath("mlp.onnx")).Run(new Dictionary<string, Tensor>()));
        Assert.Equal(OneModelErrorCategory.Session, session.Category);

        string missing = Path.Combine(Path.GetTempPath(), "onemodel-missing-" + Guid.NewGuid().ToString("N") + ".onnx");
        Assert.Throws<FileNotFoundException>(() => Model.Load(missing));
        Assert.Throws<FileNotFoundException>(() => Session.Load(missing));
        Assert.Throws<FileNotFoundException>(() => ModelFormat.Load(missing));
        Assert.Throws<DirectoryNotFoundException>(() => ModelFormat.Convert(GraphPath("mlp.onnx"), Path.Combine(missing, "x", "out.onnx")));
    }

    [Fact]
    public void LongFieldPathsAreShortenedInTheMessageButKeptInTheProperty()
    {
        string path = string.Join('.', Enumerable.Repeat("node[3]", 60));
        var e = new OnnxImportException(OnnxErrorKind.Malformed, "bad", "f.onnx", 7, path);
        Assert.Equal(path, e.FieldPath);
        Assert.True(e.Message.Length < 400);
        Assert.Contains("at byte 7", e.Message);
    }

    // ---- C5 limits --------------------------------------------------------------------------------------------------------

    [Fact]
    public void AnEndlessStreamStopsWithATypedError()
    {
        var limits = new ModelLoadOptions { MaxModelBytes = 1 << 20 };
        var endless = new EndlessStream();
        var e = Assert.Throws<OnnxImportException>(() => Model.Load(endless, null, limits));
        Assert.Equal(OnnxErrorKind.Unsupported, e.Kind);
        Assert.True(endless.Served < (4 << 20), "reading must stop soon after the limit");

        var endless2 = new EndlessStream();
        Assert.Throws<OnnxImportException>(() => Session.Load(endless2, null, limits));
        Assert.Throws<OnnxImportException>(() => OnnxImporter.Load(new EndlessStream(), limits));

        byte[] file = File.ReadAllBytes(GraphPath("mlp.onnx"));
        Assert.True(file.Length > 100);
        Assert.Throws<OnnxImportException>(() => OnnxImporter.Load(new MemoryStream(file), new ModelLoadOptions { MaxModelBytes = 100 }));
        Assert.Throws<OnnxImportException>(() => Model.Load(GraphPath("mlp.onnx"), null, new ModelLoadOptions { MaxModelBytes = 100 }));
        using var ok = Model.Load(new MemoryStream(file), null, new ModelLoadOptions { MaxModelBytes = file.Length });
    }

    [Fact]
    public void LoadOptionsAreValidatedWhenSet()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ModelLoadOptions { MaxNesting = 1000 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new ModelLoadOptions { MaxNesting = 3 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new ModelLoadOptions { MaxModelBytes = 0 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new ModelLoadOptions { MaxModelBytes = long.MaxValue });
    }

    // ---- C6 operators -----------------------------------------------------------------------------------------------------

    [Fact]
    public void OperatorsAreTheDifferentiableOps()
    {
        using var scope = new ActivationScope();
        Tensor w = Tensor.FromSpan<float>([1f, 2f, 3f, 4f], 2, 2).WithRequiresGrad();
        Tensor loss = Ops.Sum(w * 2f + w - 1f / (w + 1f));
        Tensor same = Ops.Sum(Ops.Sub(Ops.Add(Ops.Mul(w, Tensor.Scalar(2f)), w), Ops.Div(Tensor.Scalar(1f), Ops.Add(w, Tensor.Scalar(1f)))));
        Assert.Equal(same.Item<float>(), loss.Item<float>());
        loss.Backward();
        float[] g = w.Grad.ToArray<float>();
        // d/dw (3w - 1/(w+1)) = 3 + 1/(w+1)^2
        Assert.Equal(3f + 1f / 4f, g[0], 5);
        Assert.Equal(3f + 1f / 25f, g[3], 5);
        Assert.Equal(-1f, (-Tensor.Scalar(1f)).Item<float>());
        Assert.Throws<InvalidOperationException>(() => w.Item<float>());
        Assert.Throws<ArgumentException>(() => Ops.Add(default, w));
    }

    [Fact]
    public void ShapeConveniencesMatchShapeCreate()
    {
        using Tensor a = Tensor.Zeros(DType.I32, 2, 3);
        using Tensor b = Tensor.Zeros(DType.I32, Shape.Create(2, 3));
        Assert.Equal(b.Shape, a.Shape);
        using Tensor c = Tensor.FromSpan<float>([1, 2, 3, 4, 5, 6], 2, 3);
        Assert.Equal(6f, c.Get<float>(1, 2));
        using Tensor s = Tensor.Zeros(DType.F32);
        Assert.Equal(0, s.Rank);
    }

    // ---- C10, C15 scopes --------------------------------------------------------------------------------------------------

    [Fact]
    public void KeepMovesAResultOutOfTheScope()
    {
        using Tensor a = Tensor.FromSpan<float>([1f, -2f], 2);
        Tensor kept, dropped;
        using (new ActivationScope())
        {
            dropped = Ops.Relu(Ops.Add(a, a));
            kept = ActivationScope.Keep(Ops.Relu(Ops.Add(a, a)));
        }

        Assert.True(dropped.IsReleased);
        Assert.False(kept.IsReleased);
        Assert.True(kept.ToArray<float>().AsSpan().SequenceEqual([2f, 0f]));
        kept.Dispose();

        Tensor inner;
        using (new ActivationScope())
        {
            using (new ActivationScope())
            {
                inner = ActivationScope.Keep(Ops.Add(a, a));
            }

            Assert.False(inner.IsReleased); // belongs to the outer scope now
        }

        Assert.True(inner.IsReleased);
    }

    [Fact]
    public async Task ComputeSettingsCanBeScoped()
    {
        ArithmeticMode before = TensorCompute.Mode;
        using (TensorCompute.Use(ArithmeticMode.Exact, threads: 2))
        {
            Assert.Equal(ArithmeticMode.Exact, TensorCompute.Mode);
            Assert.Equal(2, TensorCompute.Threads);
            using (TensorCompute.Use(threads: 1))
            {
                Assert.Equal(ArithmeticMode.Exact, TensorCompute.Mode);
                Assert.Equal(1, TensorCompute.Threads);
            }

            Assert.Equal(2, TensorCompute.Threads);
            await Task.Yield();
            Assert.Equal(ArithmeticMode.Exact, TensorCompute.Mode);
            Assert.Equal(ArithmeticMode.Exact, await Task.Run(() => TensorCompute.Mode)); // follows the flow of execution
        }

        Assert.Equal(before, TensorCompute.Mode);
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorCompute.Use(threads: 0));
    }

    // ---- C9, C12, C13 -----------------------------------------------------------------------------------------------------

    [Fact]
    public void UseAfterDisposeIsAnObjectDisposedException()
    {
        var m = Model.Load(GraphPath("mlp.onnx"));
        m.Dispose();
        m.Dispose();
        Assert.Throws<ObjectDisposedException>(() => m.Run(new float[16]));
    }

    [Fact]
    public void FormatDetectionAndConvertGuards()
    {
        string dir = Path.Combine(Path.GetTempPath(), "onemodel-api-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string onnx = Path.Combine(dir, "m.onnx"), ours = Path.Combine(dir, "m.onemodel"), again = Path.Combine(dir, "again.onemodel");
            File.Copy(GraphPath("mlp.onnx"), onnx);
            Assert.Equal(ModelFileKind.Onnx, ModelFormat.Detect(onnx));
            ModelFormat.Convert(onnx, ours);
            Assert.Equal(ModelFileKind.OneModel, ModelFormat.Detect(ours));
            Assert.Equal(ModelFileKind.Unknown, ModelFormat.Detect([]));
            Assert.Equal(ModelFileKind.Unknown, ModelFormat.Detect("not a model"u8));

            Assert.Throws<ArgumentException>(() => ModelFormat.Convert(onnx, onnx));
            Assert.Equal(ModelFileKind.Onnx, ModelFormat.Detect(onnx));
            ModelFormat.Convert(ours, again); // already converted: read and rewritten
            Assert.Equal(File.ReadAllBytes(ours), File.ReadAllBytes(again));
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public void TensorInfoPrintsItsShapeAndTheHappyPathReportsOutputShape()
    {
        using var m = Model.Load(GraphPath("mlp.onnx"));
        Assert.Contains("F32[", m.Session.Inputs[0].ToString());
        Assert.DoesNotContain("Immutable", m.Session.Inputs[0].ToString());
        float[] y = m.Run<float, float>(new float[32], default, out Shape shape);
        Assert.Equal(y.Length, shape.Length);
        Assert.Equal(2, shape[0]);
    }
}
