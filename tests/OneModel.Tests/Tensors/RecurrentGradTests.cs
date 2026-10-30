using RecursiveAI.OneModel.Tensors;
using static RecursiveAI.OneModel.Tests.Tensors.GradCheck;

namespace RecursiveAI.OneModel.Tests.Tensors;

[Collection("autograd")]
public class RecurrentGradTests
{
    private static Shape S(params int[] d) => Shape.Create(d);

    private static readonly Func<Random, float> Small = Uniform(-0.6f, 0.6f);

    // Gradients of every input of one layer in one direction, with and without biases and initial states.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LstmLayerAllInputs(bool reverse)
    {
        Run($"LstmLayer reverse={reverse}", t => Ops.LstmLayer(t[0], t[1], t[2], t[3], t[4], t[5], t[6], reverse),
            [S(4, 3, 5), S(8, 5), S(8, 2), S(8), S(8), S(3, 2), S(3, 2)], [Small, Small, Small, Small, Small, Small, Small], h: 2e-2, tolerance: 5e-3);
    }

    [Fact]
    public void LstmLayerWithoutBiasesOrStates()
    {
        Run("LstmLayer bare", t => Ops.LstmLayer(t[0], t[1], t[2]), [S(5, 2, 3), S(12, 3), S(12, 3)], [Small, Small, Small], h: 2e-2);
        Run("LstmLayer one step, batch 1", t => Ops.LstmLayer(t[0], t[1], t[2], t[3]), [S(1, 1, 3), S(4, 3), S(4, 1), S(4)], [Small, Small, Small, Small], h: 2e-2);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GruLayerAllInputs(bool reverse)
    {
        Run($"GruLayer reverse={reverse}", t => Ops.GruLayer(t[0], t[1], t[2], t[3], t[4], t[5], reverse),
            [S(4, 3, 5), S(6, 5), S(6, 2), S(6), S(6), S(3, 2)], [Small, Small, Small, Small, Small, Small], h: 2e-2);
    }

    [Fact]
    public void GruLayerWithoutBiasesOrStates()
    {
        Run("GruLayer bare", t => Ops.GruLayer(t[0], t[1], t[2]), [S(5, 2, 3), S(9, 3), S(9, 3)], [Small, Small, Small], h: 2e-2);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void LstmStack(bool bidirectional, bool batchFirst)
    {
        int dirs = bidirectional ? 2 : 1;
        // Two layers; layer 1 reads dirs*H features.
        Shape[] shapes = new Shape[1 + (2 * dirs * 4)];
        var gens = new Func<Random, float>[shapes.Length];
        shapes[0] = batchFirst ? S(3, 4, 3) : S(4, 3, 3);
        int k = 1;
        for (int l = 0; l < 2; l++)
        {
            for (int d = 0; d < dirs; d++)
            {
                int inSize = l == 0 ? 3 : 2 * dirs;
                shapes[k++] = S(8, inSize);
                shapes[k++] = S(8, 2);
                shapes[k++] = S(8);
                shapes[k++] = S(8);
            }
        }

        Array.Fill(gens, Small);
        Run($"Lstm stack bidir={bidirectional} batchFirst={batchFirst}", t => Ops.Lstm(t[0], Layers(t, dirs), bidirectional, batchFirst), shapes, gens, h: 2e-2);
    }

    [Fact]
    public void GruBidirectionalStack()
    {
        Shape[] shapes = new Shape[1 + (4 * 4)];
        var gens = new Func<Random, float>[shapes.Length];
        shapes[0] = S(4, 2, 3);
        int k = 1;
        for (int l = 0; l < 2; l++)
        {
            for (int d = 0; d < 2; d++)
            {
                shapes[k++] = S(6, l == 0 ? 3 : 4);
                shapes[k++] = S(6, 2);
                shapes[k++] = S(6);
                shapes[k++] = S(6);
            }
        }

        Array.Fill(gens, Small);
        Run("Gru stack bidir", t => Ops.Gru(t[0], Layers(t, 2), true, false), shapes, gens, h: 2e-2);
    }

    private static RnnLayer[] Layers(Tensor[] t, int dirs)
    {
        var layers = new RnnLayer[(t.Length - 1) / 4];
        for (int i = 0; i < layers.Length; i++)
        {
            layers[i] = new RnnLayer(t[1 + (i * 4)], t[2 + (i * 4)], t[3 + (i * 4)], t[4 + (i * 4)]);
        }

        return layers;
    }
}
