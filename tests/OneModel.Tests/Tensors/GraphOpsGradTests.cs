using RecursiveAI.OneModel.Tensors;
using static RecursiveAI.OneModel.Tests.Tensors.GradCheck;

namespace RecursiveAI.OneModel.Tests.Tensors;

[Collection("autograd")]
public class GraphOpsGradTests
{
    private static Shape S(params int[] d) => Shape.Create(d);

    [Fact]
    public void BatchedMatMulGradients()
    {
        Run("3D x 3D", t => Ops.MatMulBatched(t[0], t[1]), [S(2, 3, 4), S(2, 4, 5)]);
        Run("4D x 4D", t => Ops.MatMulBatched(t[0], t[1]), [S(2, 2, 3, 4), S(2, 2, 4, 3)]);
        Run("batch broadcast (1 against 3)", t => Ops.MatMulBatched(t[0], t[1]), [S(1, 2, 3), S(3, 3, 4)]);
        Run("2D against 3D", t => Ops.MatMulBatched(t[0], t[1]), [S(2, 3), S(4, 3, 2)]);
        Run("4D against 3D (rank mismatch)", t => Ops.MatMulBatched(t[0], t[1]), [S(2, 3, 2, 3), S(3, 3, 2)]);
        Run("3D against 2D", t => Ops.MatMulBatched(t[0], t[1]), [S(2, 3, 4), S(4, 5)]);
        Run("transposed view operand", t => Ops.MatMulBatched(t[0], t[1].Transpose(-1, -2)), [S(2, 3, 4), S(2, 5, 4)]);
    }

    [Fact]
    public void BatchedMatMulMatchesPerBatchProducts()
    {
        var rng = new Random(5);
        using Tensor a = Fill(S(2, 3, 4), rng, Uniform(-1, 1)), b = Fill(S(1, 4, 2), rng, Uniform(-1, 1));
        using Tensor r = Ops.MatMulBatched(a, b);
        Assert.Equal(S(2, 3, 2), r.Shape);
        for (int i = 0; i < 2; i++)
        {
            using Tensor ref2 = Ops.MatMul(a.Select(0, i), b.Select(0, 0));
            Assert.True(r.Select(0, i).Contiguous().Data<float>().SequenceEqual(ref2.Data<float>()));
        }
    }

    [Fact]
    public void ElementwiseGradients()
    {
        Run("Erf", t => Ops.Erf(t[0]), [S(3, 5)], [Uniform(-2f, 2f)]);
        Run("Exp", t => Ops.Exp(t[0]), [S(3, 5)], [Uniform(-1.5f, 1.5f)]);
        Run("Sqrt", t => Ops.Sqrt(t[0]), [S(3, 5)], [Uniform(0.5f, 3f)]);
        Run("Pow 2", t => Ops.Pow(t[0], 2f), [S(3, 5)]);
        Run("Pow 3", t => Ops.Pow(t[0], 3f), [S(3, 5)]);
        Run("Pow 0.5", t => Ops.Pow(t[0], 0.5f), [S(3, 5)], [Uniform(0.5f, 3f)]);
        Run("Pow -1", t => Ops.Pow(t[0], -1f), [S(3, 5)], [AwayFromZero(0.6f, 2f)]);
    }

    [Fact]
    public void WhereAndGatherGradients()
    {
        using Tensor cond = Tensor.FromSpan<bool>([true, false, true, true, false, false], S(2, 3));
        Run("Where", t => Ops.Where(cond, t[0], t[1]), [S(2, 3), S(2, 3)]);
        Run("Where broadcast", t => Ops.Where(cond, t[0], t[1]), [S(3), S(2, 1)]);
        using Tensor ids = Tensor.FromSpan<long>([2, 0, 2, -1], S(2, 2));
        Run("Gather axis 0 (repeated and negative ids)", t => Ops.Gather(t[0], ids, 0), [S(3, 4)]);
        Run("Gather axis 1", t => Ops.Gather(t[0], ids, 1), [S(3, 3, 2)]);
        using Tensor scalar = Tensor.Scalar<long>(1);
        Run("Gather scalar index", t => Ops.Gather(t[0], scalar, 1), [S(2, 3, 4)]);
    }

    [Fact]
    public void GatherRejectsOutOfRangeIndices()
    {
        using Tensor data = Tensor.Zeros(DType.F32, S(3, 2));
        using Tensor ids = Tensor.FromSpan<long>([0, 3], S(2));
        Assert.Throws<ArgumentOutOfRangeException>(() => Ops.Gather(data, ids, 0));
    }

    [Fact]
    public void ExactModeRejectsInexactPowers()
    {
        using Tensor x = Tensor.FromSpan<float>([1f, 2f], S(2));
        using (TensorCompute.Use(ArithmeticMode.Exact))
        {
            Assert.Throws<NotSupportedException>(() => Ops.Pow(x, 3f));
            using Tensor y = Ops.Pow(x, 2f);
            Assert.True(y.Data<float>().SequenceEqual([1f, 4f]));
        }
    }

    // Exact mode: the new operators and an imported graph's gradients hash the same under every instruction-set setting and thread
    // count. The test checks 1 against 8 threads; the instruction-set rows come from running it under DOTNET_EnableAVX512=0 and
    // DOTNET_EnableHWIntrinsic=0 with ONEMODEL_HASH_OUT set and comparing the files.
    [Fact]
    public void ExactHashesAreEqualAtOneAndEightThreads()
    {
        string[] lines = new string[2];
        int slot = 0;
        foreach (int threads in new[] { 1, 8 })
        {
            using var compute = TensorCompute.Use(ArithmeticMode.Exact, threads);
            using var scope = new ActivationScope();
            var rng = new Random(11);
            using var h = System.Security.Cryptography.IncrementalHash.CreateHash(System.Security.Cryptography.HashAlgorithmName.SHA256);
            void Add(Tensor t)
            {
                foreach (float v in t.Detach().Contiguous().Data<float>())
                {
                    h.AppendData(BitConverter.GetBytes(v));
                }
            }

            Tensor a = Fill(S(2, 3, 5, 4), rng, Uniform(-1, 1)).WithRequiresGrad(), b = Fill(S(3, 4, 6), rng, Uniform(-1, 1)).WithRequiresGrad();
            Tensor y = Ops.MatMulBatched(a, b);
            Tensor z = Ops.Add(Ops.Erf(y), Ops.Exp(y * 0.3f));
            Tensor w = Ops.Add(z, Ops.Sqrt((y * y) + 1f));
            using Tensor cond = Tensor.FromSpan<bool>(Enumerable.Range(0, 6).Select(i => i % 3 != 0).ToArray(), S(6));
            Tensor sel = Ops.Where(cond, w, Ops.Pow(w, 2f));
            using Tensor ids = Tensor.FromSpan<long>([1, 0, 2, 1], S(2, 2));
            Tensor picked = Ops.Gather(sel, ids, 1);
            Ops.Sum(picked).Backward();
            Add(picked);
            Add(a.Grad);
            Add(b.Grad);
            lines[slot++] = Convert.ToHexString(h.GetHashAndReset());
        }

        Assert.Equal(lines[0], lines[1]);
        if (Environment.GetEnvironmentVariable("ONEMODEL_HASH_OUT") is { Length: > 0 } outPath)
        {
            File.AppendAllText(outPath, "graph-ops " + lines[0] + "\n");
        }
    }
}
