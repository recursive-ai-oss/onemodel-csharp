using RecursiveAI.OneModel.Training;

namespace RecursiveAI.OneModel.Tests.Training;

public class WarmupCosineLrTests
{
    [Fact]
    public void MatchesTheClosedForm()
    {
        const float lr = 6e-5f;
        var sched = new WarmupCosineLr(lr, 200, 2000, 0.02f);
        for (long s = 0; s <= 2100; s++)
        {
            double want = s < 200
                ? lr * (s + 1) / 200.0
                : lr * (0.02 + (0.98 * 0.5 * (1 + Math.Cos(Math.PI * Math.Min(1.0, (s - 200) / 1800.0)))));
            Assert.Equal(want, sched.LrAt(s), want * 1e-6);
        }
    }
}
