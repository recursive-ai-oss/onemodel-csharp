using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.ShapeIndexing;

/// <summary>An operator input: a tensor, either fed at run time or baked into the model as an initializer (so shapes and values are known at plan time).</summary>
internal readonly record struct In(Tensor T, bool Const = false);

/// <summary>Runs one node through the whole stack (model bytes, import, session, plan, kernel) and has naive index helpers for references.</summary>
internal static class Harness
{
    public static Tensor[] Run(string op, int opset, In?[] inputs, (string Name, object Value)[] attrs, DType[] outTypes,
        ArithmeticMode mode = ArithmeticMode.Fast, int threads = 1)
    {
        var mb = new ModelBuilder(opset);
        var names = new string[inputs.Length];
        var feedNames = new List<string>();
        var feed = new List<Tensor>();
        for (int i = 0; i < inputs.Length; i++)
        {
            if (inputs[i] is not { } input)
            {
                names[i] = "";
                continue;
            }

            names[i] = "i" + i;
            Tensor t = input.T;
            long[] dims = t.Shape.Dimensions.ToArray().Select(d => (long)d).ToArray();
            if (input.Const)
            {
                using Tensor c = t.Clone();
                mb.RawInitializer(names[i], (int)t.DType, dims, c.RawMutableBytes().ToArray());
            }
            else
            {
                mb.Input(names[i], (int)t.DType, dims.Select(d => (object)d).ToArray());
                feedNames.Add(names[i]);
                feed.Add(t);
            }
        }

        var outNames = new string[outTypes.Length];
        for (int j = 0; j < outNames.Length; j++)
        {
            outNames[j] = "o" + j;
            mb.Output(outNames[j], (int)outTypes[j]);
        }

        mb.Node(op, names, outNames, attrs);
        using IrModel ir = OnnxImporter.Load(mb.Build());
        using var session = Session.Create(ir, new ModelOptions { Mode = mode, IntraOpThreads = threads });
        var outs = new Tensor[outNames.Length];
        session.Run(feedNames.ToArray(), feed.ToArray(), outNames, outs);
        return outs;
    }

    public static void DisposeAll(Tensor[] t)
    {
        foreach (Tensor x in t)
        {
            if (!x.IsEmpty)
            {
                x.Dispose();
            }
        }
    }

    public static Tensor Make<T>(T[] values, params int[] dims) where T : unmanaged => Tensor.FromSpan<T>(values, Shape.Create(dims));

    public static (T[] Data, int[] Dims) Result<T>(Tensor t) where T : unmanaged => (t.ToArray<T>(), t.Shape.Dimensions.ToArray());

    public static int Count(int[] dims) => dims.Aggregate(1, (a, b) => a * b);

    public static int[] Unravel(int flat, int[] dims)
    {
        int[] idx = new int[dims.Length];
        for (int d = dims.Length - 1; d >= 0; d--)
        {
            idx[d] = dims[d] == 0 ? 0 : flat % dims[d];
            flat = dims[d] == 0 ? 0 : flat / dims[d];
        }

        return idx;
    }

    public static int Ravel(int[] idx, int[] dims)
    {
        int flat = 0;
        for (int d = 0; d < dims.Length; d++)
        {
            flat = (flat * dims[d]) + idx[d];
        }

        return flat;
    }

    /// <summary>Data whose element k is k (so a wrongly placed element is visible), in the element type T.</summary>
    public static T[] Iota<T>(int n) where T : unmanaged, System.Numerics.INumber<T>
    {
        T[] v = new T[n];
        for (int i = 0; i < n; i++)
        {
            v[i] = T.CreateTruncating(i + 1);
        }

        return v;
    }

    public static string Describe(int[] dims) => "[" + string.Join(",", dims) + "]";
}
