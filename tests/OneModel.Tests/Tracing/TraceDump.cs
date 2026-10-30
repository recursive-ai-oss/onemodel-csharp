using System.Runtime.InteropServices;
using System.Text;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Tracing;

namespace RecursiveAI.OneModel.Tests.Tracing;

// Writes the traced models as ONNX with example inputs and the eager outputs, for tools/trace_oracle.py (ONNX Runtime as a black box).
// Runs only when ONEMODEL_TRACE_DUMP names a directory.
[Collection("autograd")]
public class TraceDump
{
    private static void Write(string dir, string name, TracedModel tm, string[] inputNames, Tensor[] inputs, Tensor expected, string tag)
    {
        var meta = new StringBuilder();
        using var bin = new MemoryStream();
        meta.Append("{\"inputs\":[");
        for (int i = 0; i < inputs.Length; i++)
        {
            bool isLong = inputs[i].DType == DType.I64;
            byte[] raw = isLong
                ? MemoryMarshal.AsBytes<long>(inputs[i].ToArray<long>()).ToArray()
                : MemoryMarshal.AsBytes<float>(inputs[i].ToArray<float>()).ToArray();
            meta.Append(i > 0 ? "," : "")
                .Append($"{{\"name\":\"{inputNames[i]}\",\"dtype\":\"{(isLong ? "int64" : "float32")}\",\"shape\":[{string.Join(',', inputs[i].Shape.Dimensions.ToArray())}],\"offset\":{bin.Length},\"bytes\":{raw.Length}}}");
            bin.Write(raw);
        }

        meta.Append($"],\"output\":\"y\",\"outputShape\":[{string.Join(',', expected.Shape.Dimensions.ToArray())}]}}");
        File.WriteAllText(Path.Combine(dir, $"{name}.{tag}.json"), meta.ToString());
        File.WriteAllBytes(Path.Combine(dir, $"{name}.{tag}.in.bin"), bin.ToArray());
        File.WriteAllBytes(Path.Combine(dir, $"{name}.{tag}.out.bin"), MemoryMarshal.AsBytes<float>(expected.ToArray<float>()).ToArray());
        tm.SaveOnnx(Path.Combine(dir, name + ".onnx"));
    }

    [Fact]
    public void WriteModels()
    {
        string? dir = Environment.GetEnvironmentVariable("ONEMODEL_TRACE_DUMP");
        if (string.IsNullOrEmpty(dir))
        {
            return;
        }

        Directory.CreateDirectory(dir);
        var cnn = new TraceModels.Cnn();
        using (var tracer = Tracer.Begin())
        {
            Tensor x = tracer.Input("x", TraceModels.Cnn.Input(4), "n", null, null, null);
            Tensor y = cnn.Forward(x);
            using TracedModel tm = tracer.Finish("y", y);
            Write(dir, "cnn", tm, ["x"], [TraceModels.Cnn.Input(4)], y, "a");
            Tensor x2 = TraceModels.Cnn.Input(7);
            Write(dir, "cnn", tm, ["x"], [x2], cnn.Forward(x2), "b");
        }

        var enc = new TraceModels.Encoder();
        using (var tracer = Tracer.Begin())
        {
            Tensor ids = tracer.Input("ids", TraceModels.Ids(7, 50, 3, 5), "b", "n");
            Tensor mask = tracer.Input("mask", TraceModels.Encoder.Mask(3, 5, 8), "b", "n", "n");
            Tensor y = enc.Forward(ids, mask);
            using TracedModel tm = tracer.Finish("y", y);
            Write(dir, "encoder", tm, ["ids", "mask"], [TraceModels.Ids(7, 50, 3, 5), TraceModels.Encoder.Mask(3, 5, 8)], y, "a");
            Tensor i2 = TraceModels.Ids(9, 50, 4, 9), m2 = TraceModels.Encoder.Mask(4, 9, 3);
            Write(dir, "encoder", tm, ["ids", "mask"], [i2, m2], enc.Forward(i2, m2), "b");
        }

        var crnn = new TraceModels.Crnn();
        using (var tracer = Tracer.Begin())
        {
            Tensor x = tracer.Input("x", TraceModels.Crnn.Input(3, 20), "b", null, null, "w");
            Tensor y = crnn.Forward(x);
            using TracedModel tm = tracer.Finish("y", y);
            Write(dir, "crnn", tm, ["x"], [TraceModels.Crnn.Input(3, 20)], y, "a");
            Tensor x2 = TraceModels.Crnn.Input(5, 32);
            Write(dir, "crnn", tm, ["x"], [x2], crnn.Forward(x2), "b");
        }
    }
}
