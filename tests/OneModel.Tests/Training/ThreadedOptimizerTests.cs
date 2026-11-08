using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Training;

namespace RecursiveAI.OneModel.Tests.Training;

[Collection("autograd")]
public class ThreadedOptimizerTests
{
    private static int[] Run(ArithmeticMode mode, int threads, bool amsgrad, bool infinity)
    {
        using var compute = TensorCompute.Use(mode, threads);
        using Tensor p = Tensor.Zeros(DType.F32, Shape.Create(100_003)).WithRequiresGrad();
        using Tensor q = Tensor.Zeros(DType.F32, Shape.Create(32_769)).WithRequiresGrad();
        using Tensor pg = Tensor.Zeros(DType.F32, p.Shape);
        using Tensor qg = Tensor.Zeros(DType.F32, q.Shape);
        p.Autograd!.Grad = pg;
        q.Autograd!.Grad = qg;
        using (GradMode.NoGrad())
        {
            p.MutableData<float>().Fill(0.25f);
            q.MutableData<float>().Fill(-0.5f);
        }

        Tensor[] parameters = [p, q];
        using var opt = new AdamW(parameters, 0.003f, weightDecay: 0.03f, amsgrad: amsgrad);
        var bits = new List<int>();
        for (int step = 0; step < 4; step++)
        {
            Span<float> ps = pg.MutableData<float>(), qs = qg.MutableData<float>();
            for (int i = 0; i < pg.Length; i++) ps[i] = ((i * 17 + step) % 101 - 50) * 0.013f;
            for (int i = 0; i < qg.Length; i++) qs[i] = ((i * 13 + step) % 71 - 35) * 0.017f;
            double sum = 0;
            foreach (Tensor g in new[] { pg, qg })
            {
                foreach (float value in g.Data<float>())
                {
                    if (infinity) sum = Math.Max(sum, Math.Abs((double)value));
                    else sum += (double)value * value;
                }
            }

            float expected = (float)(infinity ? sum : Math.Sqrt(sum));
            float norm = GradClip.ClipNorm(parameters, 0.1f, infinity);
            Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(norm));
            bits.Add(BitConverter.SingleToInt32Bits(norm));
            opt.Step();
        }

        foreach (Tensor t in new[] { p, q, pg, qg }.Concat(opt.StateTensors.Select(s => s.Tensor)))
        {
            foreach (float value in t.Data<float>()) bits.Add(BitConverter.SingleToInt32Bits(value));
        }

        p.Autograd.Grad = q.Autograd.Grad = default;
        return bits.ToArray();
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public void ClipAndAdamWKeepEveryBitAtEveryThreadCount(bool amsgrad, bool infinity)
    {
        int[] baseline = Run(ArithmeticMode.Exact, 1, amsgrad, infinity);
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            foreach (int threads in new[] { 1, 3, 8 }) Assert.Equal(baseline, Run(mode, threads, amsgrad, infinity));
        }
    }
}
