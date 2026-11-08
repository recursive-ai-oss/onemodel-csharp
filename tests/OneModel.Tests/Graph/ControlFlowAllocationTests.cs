using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Tests.Onnx;

namespace RecursiveAI.OneModel.Tests.Graph;

[Collection("autograd")]
public class ControlFlowAllocationTests
{
    private static byte[] Conditional()
    {
        Pb Branch(string name, string op) => Build.Graph(name,
            [Build.Node(op, ["x", "x"], [name])], null, [Build.ValueInfo(name, Build.Float, 1)]);
        return Build.Model(Build.Graph("conditional",
            [Build.Node("If", ["condition"], ["y"], "choose",
                Build.AttrGraph("then_branch", Branch("yes", "Add")), Build.AttrGraph("else_branch", Branch("no", "Mul")))],
            [Build.ValueInfo("condition", Build.Bool), Build.ValueInfo("x", Build.Float, 1)],
            [Build.ValueInfo("y", Build.Float, 1)]), 17, 8);
    }

    [Theory]
    [InlineData(ArithmeticMode.Fast)]
    [InlineData(ArithmeticMode.Exact)]
    public void ConditionalCallerBuffersAllocateNothing(ArithmeticMode mode)
    {
        using IrModel model = OnnxImporter.Load(Conditional());
        using Session session = Session.Create(model, new ModelOptions { Mode = mode, IntraOpThreads = 1 });
        Measure(session);
    }

    [Fact]
    public async Task ConcurrentConditionalCallsHaveIndependentArgumentsAndResults()
    {
        using IrModel model = OnnxImporter.Load(Conditional());
        using Session session = Session.Create(model, new ModelOptions { IntraOpThreads = 1 });
        await Task.WhenAll(Enumerable.Range(1, 8).Select(value => Task.Run(() =>
        {
            using Tensor condition = Tensor.Scalar(true);
            using Tensor x = Tensor.FromSpan<float>([(float)value], Shape.Create(1));
            using Tensor y = Tensor.Zeros(DType.F32, Shape.Create(1));
            string[] names = ["condition", "x"], wanted = ["y"];
            Tensor[] inputs = [condition, x], outputs = [y];
            for (int i = 0; i < 100; i++)
            {
                condition.MutableData<bool>()[0] = (i & 1) == 0;
                session.Run(names, inputs, wanted, outputs);
                Assert.Equal((i & 1) == 0 ? 2f * value : value * value, y.Data<float>()[0]);
            }
        })));
    }

    private static void Measure(Session session)
    {
        using Tensor condition = Tensor.Scalar(true);
        using Tensor x = Tensor.FromSpan<float>([3f], Shape.Create(1));
        using Tensor y = Tensor.Zeros(DType.F32, Shape.Create(1));
        string[] names = ["condition", "x"], wanted = ["y"];
        Tensor[] inputs = [condition, x], outputs = [y];
        for (int i = 0; i < 100; i++)
        {
            condition.MutableData<bool>()[0] = (i & 1) == 0;
            session.Run(names.AsSpan(), inputs.AsSpan(), wanted.AsSpan(), outputs.AsSpan());
            Assert.Equal((i & 1) == 0 ? 6f : 9f, y.Data<float>()[0]);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10000; i++)
        {
            condition.MutableData<bool>()[0] = (i & 1) == 0;
            session.Run(names.AsSpan(), inputs.AsSpan(), wanted.AsSpan(), outputs.AsSpan());
        }

        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, allocated);
    }
}
