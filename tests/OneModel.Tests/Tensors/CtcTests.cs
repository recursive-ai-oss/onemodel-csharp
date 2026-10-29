using System.Runtime.CompilerServices;
using System.Text.Json;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Tensors;

/// <summary>CTC decoding against the double-precision references in testdata/ctc (tools/ctc_reference.py).</summary>
public class CtcTests
{
    private static string DataFile([CallerFilePath] string file = "") =>
        Path.GetFullPath(Path.Combine(Path.GetDirectoryName(file)!, "..", "..", "..", "testdata", "ctc", "cases.json"));

    private static List<JsonElement> Cases()
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(DataFile()));
        return [.. doc.RootElement.GetProperty("cases").EnumerateArray().Select(e => e.Clone())];
    }

    private static float[] Floats(JsonElement e) => [.. e.EnumerateArray().Select(x => (float)x.GetDouble())];

    private static int[] Ints(JsonElement e) => [.. e.EnumerateArray().Select(x => x.GetInt32())];

    private static CtcScores Kind(JsonElement c) => c.GetProperty("kind").GetString() == "logits" ? CtcScores.Logits : CtcScores.LogProbabilities;

    [Fact]
    public void GreedyMatchesTheReference()
    {
        foreach (JsonElement c in Cases())
        {
            int steps = c.GetProperty("steps").GetInt32(), classes = c.GetProperty("classes").GetInt32(), blank = c.GetProperty("blank").GetInt32();
            float[] scores = Floats(c.GetProperty("scores"));
            JsonElement g = c.GetProperty("greedy");
            var runs = new CtcRun[steps];
            var stepP = new float[steps];
            int n = Ctc.Greedy(scores, steps, classes, runs, out float conf, blank, Kind(c), stepP);
            Assert.Equal(Ints(g.GetProperty("labels")), runs.Take(n).Select(r => r.Label));
            Assert.Equal(Ints(g.GetProperty("first")), runs.Take(n).Select(r => r.FirstStep));
            Assert.Equal(Ints(g.GetProperty("last")), runs.Take(n).Select(r => r.LastStep));
            float[] refRun = Floats(g.GetProperty("runConfidence"));
            for (int i = 0; i < n; i++)
            {
                Assert.InRange(Math.Abs(runs[i].Confidence - refRun[i]), 0, 2e-6);
            }

            Assert.InRange(Math.Abs(conf - (float)g.GetProperty("confidence").GetDouble()), 0, 2e-6);
            float[] refStep = Floats(g.GetProperty("stepProbability"));
            for (int t = 0; t < steps; t++)
            {
                Assert.InRange(Math.Abs(stepP[t] - refStep[t]), 0, 2e-6);
            }

            CtcGreedyResult r = Ctc.Greedy(scores, steps, classes, blank, Kind(c));
            Assert.Equal(Ints(g.GetProperty("labels")), r.Labels());
        }
    }

    [Fact]
    public void BeamSearchMatchesThePrefixBeamReferenceAndTheExactBest()
    {
        foreach (JsonElement c in Cases())
        {
            int steps = c.GetProperty("steps").GetInt32(), classes = c.GetProperty("classes").GetInt32(), blank = c.GetProperty("blank").GetInt32();
            float[] scores = Floats(c.GetProperty("scores"));
            foreach (JsonElement b in c.GetProperty("beams").EnumerateArray())
            {
                CtcHypothesis h = Ctc.BeamSearch(scores, steps, classes, b.GetProperty("width").GetInt32(), 1, blank, Kind(c))[0];
                Assert.Equal(Ints(b.GetProperty("labels")), h.Labels);
                Assert.InRange(Math.Abs(h.LogProbability - b.GetProperty("logp").GetDouble()), 0, 1e-9);
            }

            if (c.TryGetProperty("exact", out JsonElement exact))
            {
                // A beam as wide as the number of distinct prefixes loses nothing.
                CtcHypothesis h = Ctc.BeamSearch(scores, steps, classes, 4096, 1, blank, Kind(c))[0];
                Assert.Equal(Ints(exact.GetProperty("labels")), h.Labels);
                Assert.InRange(Math.Abs(h.LogProbability - exact.GetProperty("logp").GetDouble()), 0, 1e-9);
            }
        }
    }

    [Fact]
    public void BlankIsConfigurableAndRepeatsNeedABlank()
    {
        // Path 1 1 2 2 0 2 with blank 0 reads "1 2 2" (the blank separates the two 2s); with blank 3 the 0 is a label.
        float[] s = new float[6 * 4];
        int[] path = [1, 1, 2, 2, 0, 2];
        for (int t = 0; t < 6; t++)
        {
            s[(t * 4) + path[t]] = 5f;
        }

        Assert.Equal([1, 2, 2], Ctc.Greedy(s, 6, 4).Labels());
        Assert.Equal([1, 2, 0, 2], Ctc.Greedy(s, 6, 4, blank: 3).Labels());
        Assert.Equal([1, 2, 2], Ctc.BeamSearch(s, 6, 4)[0].Labels);
    }

    [Fact]
    public void EmptyAndAllBlankInputsGiveEmptyText()
    {
        Assert.Empty(Ctc.Greedy([], 0, 5).Runs);
        Assert.Equal(0f, Ctc.Greedy([], 0, 5).Confidence);
        CtcHypothesis h = Ctc.BeamSearch([], 0, 5)[0];
        Assert.Empty(h.Labels);
        Assert.Equal(0.0, h.LogProbability);
        float[] blanks = new float[3 * 4];
        for (int t = 0; t < 3; t++)
        {
            blanks[t * 4] = 9f;
        }

        Assert.Empty(Ctc.Greedy(blanks, 3, 4).Runs);
        Assert.Empty(Ctc.BeamSearch(blanks, 3, 4)[0].Labels);
    }

    [Fact]
    public void BatchHonoursLengthsAndRejectsBadArguments()
    {
        float[] s = new float[2 * 3 * 3];
        // sequence 0: label 1 then label 2; sequence 1 is the same scores but only the first step is valid.
        for (int b = 0; b < 2; b++)
        {
            s[(b * 9) + 1] = 4f;
            s[(b * 9) + 3 + 0] = 4f;
            s[(b * 9) + 6 + 2] = 4f;
        }

        CtcGreedyResult[] r = Ctc.GreedyBatch(s, 2, 3, 3, [3, 1]);
        Assert.Equal([1, 2], r[0].Labels());
        Assert.Equal([1], r[1].Labels());
        Assert.Throws<ArgumentOutOfRangeException>(() => Ctc.GreedyBatch(s, 2, 3, 3, [3, 4]));
        Assert.Throws<ArgumentException>(() => Ctc.Greedy(new float[5], 2, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => Ctc.Greedy(new float[6], 2, 3, blank: 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => Ctc.BeamSearch(new float[6], 2, 3, beamWidth: 0));
        Assert.Throws<ArgumentException>(() => Ctc.BeamSearch([float.NaN, 0f, 0f], 1, 3));
    }

    [Fact]
    public void GreedyAllocatesNothingInSteadyState()
    {
        var rng = new Random(3);
        float[] s = new float[80 * 97];
        for (int i = 0; i < s.Length; i++)
        {
            s[i] = (float)((rng.NextDouble() * 8) - 4);
        }

        var runs = new CtcRun[80];
        for (int i = 0; i < 20; i++)
        {
            Ctc.Greedy(s, 80, 97, runs, out _);
        }

        long best = long.MaxValue;
        for (int attempt = 0; attempt < 3 && best != 0; attempt++)
        {
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 200; i++)
            {
                Ctc.Greedy(s, 80, 97, runs, out _);
            }

            best = Math.Min(best, GC.GetAllocatedBytesForCurrentThread() - before);
        }

        Assert.Equal(0, best);
    }
}
