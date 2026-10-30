using RecursiveAI.OneModel.Tensors;
using static RecursiveAI.OneModel.Tests.Tensors.GradCheck;

namespace RecursiveAI.OneModel.Tests.Tensors;

[Collection("autograd")]
public class GradCheckTests
{
    private static Shape S(params int[] d) => Shape.Create(d);

    [Fact]
    public void BinaryBroadcasting()
    {
        Run("Add [3,4]+[4]", t => Ops.Add(t[0], t[1]), [S(3, 4), S(4)]);
        Run("Add [3,1]+[1,4]", t => Ops.Add(t[0], t[1]), [S(3, 1), S(1, 4)]);
        Run("Sub [2,3,4]-[3,1]", t => Ops.Sub(t[0], t[1]), [S(2, 3, 4), S(3, 1)]);
        Run("Mul [3,4]*[4]", t => Ops.Mul(t[0], t[1]), [S(3, 4), S(4)]);
        Run("Mul [2,3,4]*scalar", t => Ops.Mul(t[0], t[1]), [S(2, 3, 4), S()]);
        Run("Div [3,4]/[1,4]", t => Ops.Div(t[0], t[1]), [S(3, 4), S(1, 4)], [Uniform(-1, 1), Uniform(0.7f, 1.6f)]);
        Run("Div [3,1]/[3,4]", t => Ops.Div(t[0], t[1]), [S(3, 1), S(3, 4)], [Uniform(-1, 1), AwayFromZero(0.7f, 1.6f)]);
        Run("Mul x*x (shared input)", t => Ops.Mul(t[0], t[0]), [S(3, 4)]);
        Run("Neg", t => Ops.Neg(t[0]), [S(3, 4)]);
    }

    [Fact]
    public void Activations()
    {
        Run("Relu", t => Ops.Relu(t[0]), [S(4, 5)], [AwayFromZero(0.1f, 1f)], h: 5e-2);
        Run("Gelu erf", t => Ops.Gelu(t[0]), [S(4, 5)], [Uniform(-3, 3)]);
        Run("Gelu tanh", t => Ops.Gelu(t[0], approximateTanh: true), [S(4, 5)], [Uniform(-3, 3)]);
        Run("Sigmoid", t => Ops.Sigmoid(t[0]), [S(4, 5)], [Uniform(-3, 3)]);
        Run("Tanh", t => Ops.Tanh(t[0]), [S(4, 5)], [Uniform(-2, 2)]);
    }

    [Fact]
    public void MatMulViews()
    {
        Run("MatMul [3,5]x[5,4]", t => Ops.MatMul(t[0], t[1]), [S(3, 5), S(5, 4)]);
        Run("MatMul a^T b (transposed view)", t => Ops.MatMul(t[0].T(), t[1]), [S(5, 3), S(5, 4)]);
        Run("MatMul a b^T (transposed view)", t => Ops.MatMul(t[0], t[1].T()), [S(3, 5), S(4, 5)]);
        Run("MatMul sliced operands", t => Ops.MatMul(t[0].Slice(1, 1, 6, 2), t[1].Slice(0, 0, 3)), [S(4, 6), S(5, 4)], check: null);
        Run("Linear [2,3,5]", t => Ops.Linear(t[0], t[1], t[2]), [S(2, 3, 5), S(4, 5), S(4)]);
    }

    [Fact]
    public void ReductionsAndSoftmax()
    {
        Run("Sum all", t => Ops.Sum(t[0]), [S(3, 4)]);
        Run("Sum axis 1", t => Ops.Sum(t[0], 1), [S(2, 3, 4)]);
        Run("Sum axis -1 keepdim", t => Ops.Sum(t[0], -1, keepDim: true), [S(2, 3, 4)]);
        Run("Mean all", t => Ops.Mean(t[0]), [S(3, 4)]);
        Run("Mean axis 0", t => Ops.Mean(t[0], 0), [S(5, 4)]);
        Run("Softmax last", t => Ops.Softmax(t[0]), [S(3, 6)]);
        Run("Softmax axis 1 of rank 3", t => Ops.Softmax(t[0], 1), [S(2, 4, 3)]);
        Run("LogSoftmax last", t => Ops.LogSoftmax(t[0]), [S(3, 6)]);
        Run("LogSoftmax axis 1 of rank 3", t => Ops.LogSoftmax(t[0], 1), [S(2, 4, 3)]);
    }

    [Fact]
    public void Norms()
    {
        Run("LayerNorm [3,5,8]", t => Ops.LayerNorm(t[0], t[1], t[2]), [S(3, 5, 8), S(8), S(8)], [Uniform(-2, 2), Uniform(0.5f, 1.5f), Uniform(-1, 1)]);
        Run("BatchNorm2d training", t => BatchNorm(t, true), [S(3, 4, 3, 3), S(4), S(4)], [Uniform(-2, 2), Uniform(0.5f, 1.5f), Uniform(-1, 1)]);
        Run("BatchNorm2d eval", t => BatchNorm(t, false), [S(3, 4, 3, 3), S(4), S(4)], [Uniform(-2, 2), Uniform(0.5f, 1.5f), Uniform(-1, 1)]);
    }

    private static Tensor BatchNorm(Tensor[] t, bool training)
    {
        Tensor rm = Tensor.Zeros(DType.F32, S(4)), rv = Tensor.Create(DType.F32, S(4));
        rv.Fill(1f);
        return Ops.BatchNorm2d(t[0], t[1], t[2], rm, rv, training);
    }

    [Fact]
    public void ConvolutionAndPooling()
    {
        Run("Conv 3x3 pad1 bias", t => Ops.Conv2d(t[0], t[1], t[2], new Conv2dParams(PadH: 1, PadW: 1)), [S(2, 3, 5, 5), S(4, 3, 3, 3), S(4)]);
        Run("Conv 3x3 stride2 pad1", t => Ops.Conv2d(t[0], t[1], default, new Conv2dParams(2, 2, 1, 1)), [S(2, 2, 6, 7), S(3, 2, 3, 3)]);
        Run("Conv 3x3 dilation2 pad2", t => Ops.Conv2d(t[0], t[1], default, new Conv2dParams(PadH: 2, PadW: 2, DilH: 2, DilW: 2)), [S(1, 2, 7, 7), S(3, 2, 3, 3)]);
        Run("Conv groups=2", t => Ops.Conv2d(t[0], t[1], t[2], new Conv2dParams(PadH: 1, PadW: 1, Groups: 2)), [S(2, 4, 5, 5), S(6, 2, 3, 3), S(6)]);
        Run("Conv depthwise 3x3", t => Ops.Conv2d(t[0], t[1], t[2], new Conv2dParams(1, 1, 1, 1, Groups: 4)), [S(2, 4, 5, 5), S(4, 1, 3, 3), S(4)]);
        Run("Conv depthwise multiplier 2, stride 2", t => Ops.Conv2d(t[0], t[1], default, new Conv2dParams(2, 2, 1, 1, Groups: 3)), [S(1, 3, 6, 6), S(6, 1, 3, 3)]);
        Run("Conv 1x1", t => Ops.Conv2d(t[0], t[1], t[2]), [S(2, 3, 4, 4), S(5, 3, 1, 1), S(5)]);
        Run("Conv 3x1 kernel", t => Ops.Conv2d(t[0], t[1], default, new Conv2dParams(PadH: 1)), [S(1, 2, 5, 4), S(2, 2, 3, 1)]);
        Run("MaxPool 2x2", t => Ops.MaxPool2d(t[0], new Pool2dParams(2, 2, 2, 2, 0, 0)), [S(2, 2, 6, 6)], [Distinct(0.1f)], h: 2e-2);
        Run("MaxPool 3x3 s2 pad1", t => Ops.MaxPool2d(t[0], new Pool2dParams(3, 3, 2, 2, 1, 1)), [S(1, 2, 7, 7)], [Distinct(0.1f)], h: 2e-2);
        Run("AvgPool 2x2", t => Ops.AvgPool2d(t[0], new Pool2dParams(2, 2, 2, 2, 0, 0)), [S(2, 2, 6, 6)]);
        Run("AvgPool 3x3 s2 pad1 count pad", t => Ops.AvgPool2d(t[0], new Pool2dParams(3, 3, 2, 2, 1, 1), true), [S(1, 2, 7, 7)]);
        Run("AvgPool 3x3 s2 pad1 no pad", t => Ops.AvgPool2d(t[0], new Pool2dParams(3, 3, 2, 2, 1, 1), false), [S(1, 2, 7, 7)]);
        Run("GlobalAvgPool", t => Ops.GlobalAvgPool2d(t[0]), [S(2, 3, 4, 5)]);
    }

    // Values k * gap for a random permutation of k, so no two window entries are within `gap` of each other.
    private static Func<Random, float> Distinct(float gap)
    {
        int n = 0;
        int[]? perm = null;
        return rng =>
        {
            if (perm is null || n >= perm.Length)
            {
                perm = Enumerable.Range(0, 4096).OrderBy(_ => rng.Next()).ToArray();
                n = 0;
            }

            return perm[n++] * gap;
        };
    }

    [Fact]
    public void ViewsAndDataMovement()
    {
        Run("Reshape", t => Ops.Mul(t[0].Reshape(6, 2), t[1]), [S(3, 4), S(6, 2)]);
        Run("Transpose then Reshape (copy)", t => t[0].Transpose(0, 2).Reshape(-1, 3), [S(3, 4, 5)]);
        Run("Permute + Contiguous", t => t[0].Permute(2, 0, 1).Contiguous(), [S(3, 4, 5)]);
        Run("Slice step 2", t => t[0].Slice(1, 1, 7, 2), [S(3, 8)]);
        Run("Select", t => t[0].Select(1, 2), [S(3, 4, 2)]);
        Run("Flip", t => Ops.Mul(t[0].Flip(0), t[1]), [S(4, 3), S(4, 3)]);
        Run("Expand", t => Ops.Mul(t[0].Expand(S(2, 3, 4)), t[1]), [S(3, 1), S(2, 3, 4)]);
        Run("Unsqueeze/Squeeze", t => t[0].Unsqueeze(1).Squeeze(1), [S(3, 4)]);
        Run("Concat axis 1", t => Ops.Concat([t[0], t[1], t[2]], 1), [S(2, 3, 2), S(2, 1, 2), S(2, 4, 2)]);
        Run("Concat axis 0 of views", t => Ops.Concat([t[0].T(), t[1]], 0), [S(2, 3), S(4, 2)]);
    }

    [Fact]
    public void EmbeddingAndLosses()
    {
        using Tensor ids = Tensor.FromSpan<int>([1, 3, 3, 0, 2, 1], S(2, 3));
        Run("Embedding", t => Ops.Embedding(t[0], ids), [S(5, 4)]);
        using Tensor ids64 = Tensor.FromSpan<long>([1, 3, 3, 0], S(4));
        // padding_idx: rows of the ignored id get no gradient, the others as without it.
        Tensor table = Fill(S(5, 4), new Random(3), Uniform(-1, 1)).WithRequiresGrad();
        using (var scope = new ActivationScope())
        {
            Ops.Embedding(table, ids64, paddingIndex: 3).Backward(Fill(S(4, 4), new Random(4), Uniform(-1, 1)));
        }

        float[] g = table.Grad.ToArray<float>();
        Assert.All(g.AsSpan(3 * 4, 4).ToArray(), v => Assert.Equal(0f, v));
        Assert.NotEqual(0f, g[1 * 4]);
        Assert.NotEqual(0f, g[0]);
        using Tensor tg = Tensor.FromSpan<int>([2, 0, 4, 1], S(4));
        Run("NllLoss mean", t => Ops.NllLoss(Ops.LogSoftmax(t[0]), tg), [S(4, 5)]);
        Run("NllLoss sum ignore=0", t => Ops.NllLoss(Ops.LogSoftmax(t[0]), tg, mean: false, ignoreIndex: 0), [S(4, 5)]);
        Run("CrossEntropy", t => Ops.CrossEntropy(t[0], tg), [S(4, 5)]);
    }

    [Fact]
    public void Attention()
    {
        Run("Attention B2 H2 Lq5 Lk7 D4", t => Ops.Attention(t[0], t[1], t[2]), [S(2, 2, 5, 4), S(2, 2, 7, 4), S(2, 2, 7, 4)]);
        using Tensor mask = Tensor.Create(DType.F32, S(2, 5, 7));
        var rng = new Random(5);
        Span<float> m = mask.MutableData<float>();
        for (int i = 0; i < m.Length; i++)
        {
            m[i] = rng.Next(4) == 0 ? -3f : 0f;
        }

        Run("Attention masked [B,Lq,Lk]", t => Ops.Attention(t[0], t[1], t[2], mask), [S(2, 2, 5, 4), S(2, 2, 7, 4), S(2, 2, 7, 4)]);
        Run("Attention scale 0.3", t => Ops.Attention(t[0], t[1], t[2], default, 0.3f), [S(1, 2, 4, 6), S(1, 2, 4, 6), S(1, 2, 4, 6)]);
        Run("Attention from permuted views", t => Ops.Attention(t[0].Permute(0, 2, 1, 3), t[1].Permute(0, 2, 1, 3), t[2].Permute(0, 2, 1, 3)),
            [S(2, 5, 2, 4), S(2, 7, 2, 4), S(2, 7, 2, 4)]);
    }

    [Fact]
    public void CompositeConvGeluPoolAttention()
    {
        // The function of the NativeAOT smoke app.
        Run("Composite conv+gelu+avgpool+attention+logsoftmax", t =>
        {
            Tensor h = Ops.Gelu(Ops.Conv2d(t[0], t[1], default, new Conv2dParams(PadH: 1, PadW: 1)));
            Tensor a = Ops.AvgPool2d(h, new Pool2dParams(2, 2, 2, 2, 0, 0)).Reshape(1, 1, 3, 4);
            Tensor att = Ops.Attention(a, t[2].Slice(1, 0, 1), a);
            return Ops.LogSoftmax(Ops.Add(att.Reshape(3, 4), t[3].Reshape(3, 1)));
        }, [S(1, 2, 4, 4), S(3, 2, 3, 3), S(1, 2, 3, 4), S(3)]);
    }
}
