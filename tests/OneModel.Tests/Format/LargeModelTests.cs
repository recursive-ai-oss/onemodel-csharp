using System.Buffers.Binary;
using RecursiveAI.OneModel.Format;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tests.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Format;

public class LargeModelTests
{
    [Fact]
    public void OversizedProtobufAndUnmappedFormatHaveTypedErrors()
    {
        if (Environment.GetEnvironmentVariable("ONEMODEL_LARGE_FILES") != "1")
        {
            return;
        }

        string path = Path.Combine(Path.GetTempPath(), "model-limit-" + Guid.NewGuid().ToString("N"));
        try
        {
            using (FileStream f = File.Create(path))
            {
                f.SetLength(1L << 31);
            }

            var onnx = Assert.Throws<OnnxImportException>(() => OnnxImporter.Load(path));
            Assert.Equal(OnnxErrorKind.Unsupported, onnx.Kind);
            Assert.Contains("external data", onnx.Message, StringComparison.Ordinal);
            var format = Assert.Throws<ModelFormatException>(() => ModelFormat.Load(path, new ModelLoadOptions { MemoryMap = false }));
            Assert.Equal(ModelFormatErrorKind.Unsupported, format.Kind);
            Assert.Contains("MemoryMap", format.Message, StringComparison.Ordinal);
            using FileStream input = File.OpenRead(path);
            Assert.Equal(ModelFormatErrorKind.Unsupported, Assert.Throws<ModelFormatException>(() => ModelFormat.Load(input)).Kind);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ExternalDataAndModelFormatAboveTwoGiBRun()
    {
        if (Environment.GetEnvironmentVariable("ONEMODEL_LARGE_FILES") != "1")
        {
            return;
        }

        string dir = Path.Combine(Path.GetTempPath(), "model-large-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        const int size = 8192, count = 10;
        const long tensorBytes = (long)size * size * 4;
        string onnx = Path.Combine(dir, "large.onnx");
        string format = Path.Combine(dir, "large.onemodel");
        try
        {
            var tensors = new Pb[count];
            var nodes = new Pb[count];
            var outputs = new Pb[count];
            using (FileStream data = File.Create(Path.Combine(dir, "weights.bin")))
            {
                data.SetLength(count * tensorBytes);
                byte[] value = new byte[4];
                for (int j = 0; j < count; j++)
                {
                    data.Position = j * tensorBytes;
                    BinaryPrimitives.WriteSingleLittleEndian(value, j + 1);
                    data.Write(value);
                    tensors[j] = new Pb().PackedInts(1, size, size).Int(2, 1).Str(8, "w" + j).Int(14, 1)
                        .Msg(13, new Pb().Str(1, "location").Str(2, "weights.bin"))
                        .Msg(13, new Pb().Str(1, "offset").Str(2, (j * tensorBytes).ToString(System.Globalization.CultureInfo.InvariantCulture)))
                        .Msg(13, new Pb().Str(1, "length").Str(2, tensorBytes.ToString(System.Globalization.CultureInfo.InvariantCulture)));
                    nodes[j] = Build.Node("MatMul", ["x", "w" + j], ["y" + j]);
                    outputs[j] = Build.ValueInfo("y" + j, 1, 1, size);
                }
            }

            File.WriteAllBytes(onnx, Build.Model(Build.Graph("large", nodes, [Build.ValueInfo("x", 1, 1, size)], outputs, tensors)));
            using (IrModel model = OnnxImporter.Load(onnx))
            {
                Assert.Equal(10, model.Graph.Initializers.Length);
                Assert.Equal(10f, BinaryPrimitives.ReadSingleLittleEndian(model.Graph.Initializers[9].RawData));
                ModelFormat.Save(model, format);
            }

            Assert.True(new FileInfo(format).Length > int.MaxValue);
            foreach (string path in new[] { onnx, format })
            {
                using Session session = Session.Load(path, new ModelOptions { IntraOpThreads = 1, OptimizationLevel = OptimizationLevel.None });
                float[] x = new float[size];
                x[0] = 2;
                using Tensor input = Tensor.FromSpan<float>(x, Shape.Create(1, size));
                using SessionOutputs result = session.Run(new Dictionary<string, Tensor> { ["x"] = input });
                for (int j = 0; j < count; j++)
                {
                    ReadOnlySpan<float> y = result[j].Data<float>();
                    Assert.Equal(2 * (j + 1f), y[0]);
                    Assert.True(y[1..].IndexOfAnyExcept(0f) < 0);
                }
            }

            // A separate oracle process can consume these before deletion when requested.
            string? oracle = Environment.GetEnvironmentVariable("ONEMODEL_LARGE_ORACLE");
            if (oracle is not null)
            {
                var start = new System.Diagnostics.ProcessStartInfo(oracle) { UseShellExecute = false };
                start.ArgumentList.Add("-c");
                start.ArgumentList.Add("import sys,numpy as np,onnxruntime as ort; assert ort.__version__=='1.31.1'; o=ort.SessionOptions(); o.intra_op_num_threads=1; s=ort.InferenceSession(sys.argv[1],sess_options=o,providers=['CPUExecutionProvider']); x=np.zeros((1,8192),dtype=np.float32); x[0,0]=2; ys=s.run(None,{'x':x}); assert all(y[0,0]==2*(j+1) and np.count_nonzero(y)==1 for j,y in enumerate(ys)); print('ORT 1.31.1: 10/10 outputs exact, scaled error 0')");
                start.ArgumentList.Add(onnx);
                using var process = System.Diagnostics.Process.Start(start)!;
                process.WaitForExit();
                Assert.Equal(0, process.ExitCode);
            }
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
