using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Tests.Tensors;
using RecursiveAI.OneModel.Training;

namespace RecursiveAI.OneModel.Tests.Training;

// A classifier on an embedding: mean of the looked-up vectors, then a linear head.
internal sealed class EmbeddingClassifier : NnModule
{
    public Embedding Emb { get; }
    public Linear Head { get; }

    public EmbeddingClassifier(Generator g, int vocab, int width, int classes, bool sparse)
    {
        Emb = RegisterModule("emb", new Embedding(vocab, width, g, paddingIndex: 0, sparse: sparse));
        Head = RegisterModule("head", new Linear(width, classes, g));
    }

    public override Tensor Forward(Tensor ids) => Head.Forward(Ops.Mean(Emb.Forward(ids), 1));
}

[Collection("autograd")]
public class SparseAndFrozenTests
{
    private static void Log(string line)
    {
        string? path = Environment.GetEnvironmentVariable("ONEMODEL_TRAINING_LOG");
        if (path is not null)
        {
            File.AppendAllText(path, line + "\n");
        }
    }

    private static double Scaled(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double max = 0, mag = 1e-12;
        for (int i = 0; i < a.Length; i++)
        {
            max = Math.Max(max, Math.Abs((double)a[i] - b[i]));
            mag = Math.Max(mag, Math.Abs(b[i]));
        }

        return max / mag;
    }

    // ------------------------------------------------------------------------------------------------ frozen parameters

    [Fact]
    public void FrozenEncoderMatchesTorchWithRequiresGradFalse()
    {
        OracleFile enc = OracleFile.Load("encoder", "training"), want = OracleFile.Load("frozen", "training");
        TrainCase tc = TrainCase.Encoder(enc, beforeOptimizer: m =>
        {
            m.Freeze("emb");
            m.Freeze("layer0");
        });
        var initial = new Dictionary<string, float[]>();
        foreach ((string n, Tensor t) in tc.Model.NamedParameters())
        {
            initial[n] = t.Data<float>().ToArray();
        }

        float[] expected = want.Values("out.loss");
        double maxRel = 0, maxRelEarly = 0;
        for (int i = 0; i < tc.Steps; i++)
        {
            float l = tc.Step(i);
            double rel = Math.Abs(l - expected[i]) / Math.Abs(expected[i]);
            maxRel = Math.Max(maxRel, rel);
            if (i < 20)
            {
                maxRelEarly = Math.Max(maxRelEarly, rel);
            }
        }

        double worst = 0, noise = 0;
        string worstName = "";
        int frozenCount = 0;
        foreach ((string p, string k) in tc.Map)
        {
            Tensor t = tc.Param(p);
            double e = Scaled(t.Data<float>(), want.Values("out." + k));
            bool isFrozen = p.StartsWith("emb", StringComparison.Ordinal) || p.StartsWith("layer0", StringComparison.Ordinal);
            if (isFrozen)
            {
                frozenCount++;
                Assert.False(t.RequiresGrad);
                Assert.True(t.Grad.IsEmpty, $"{p} must not receive a gradient");
                Assert.True(t.Data<float>().SequenceEqual(initial[p]), $"{p} must not move");
                Assert.Equal(0d, e);   // and torch's frozen value is the initial value
            }
            else if (p.EndsWith("attn.k.bias", StringComparison.Ordinal))
            {
                noise = Math.Max(noise, e);   // true gradient zero (softmax ignores a key bias); Adam amplifies rounding noise
            }
            else if (e > worst)
            {
                worst = e;
                worstName = p;
            }
        }

        Assert.True(frozenCount > 10);
        Log($"frozen encoder (emb + layer0 requires_grad_(False)), Adam, {tc.Steps} steps: loss max relative diff {maxRel:E2} (first 20 steps {maxRelEarly:E2}), remaining parameters scaled error {worst:E2} (worst {worstName}), zero-gradient key bias {noise:E2}");
        // Adam amplifies float32 rounding differences between the two implementations, so the curve drifts apart slowly; the early steps
        // show the agreement of the arithmetic, the whole curve its stability.
        Assert.True(maxRelEarly <= 1e-6, $"loss relative difference over the first 20 steps {maxRelEarly:E2}");
        Assert.True(maxRel <= 1e-5, $"loss relative difference {maxRel:E2}");
        Assert.True(worst <= 2e-6, $"parameter scaled error {worst:E2} at {worstName}");

        // Frozen parameters cost no optimizer state; the others have full state.
        foreach ((string name, Tensor st) in tc.Opt.StateTensors)
        {
            int index = int.Parse(name[(name.LastIndexOf('.') + 1)..], System.Globalization.CultureInfo.InvariantCulture);
            Tensor p = tc.Opt.Parameters[index];
            Assert.Equal(p.RequiresGrad ? p.Length : 0, st.Length);
        }
    }

    [Fact]
    public void FreezeByNamePrefixPredicateAndUnfreeze()
    {
        var g = new Generator(1);
        using var model = new EmbeddingClassifier(g, 10, 4, 3, sparse: false);
        Assert.Equal(3, model.TrainableParameters().Length);
        Assert.Equal(1, model.Freeze("emb.weight"));
        Assert.True(model.IsFrozen("emb.weight"));
        Assert.Equal(2, model.TrainableParameters().Length);
        Assert.Equal(0, model.Freeze("emb"));                 // already frozen: nothing changes
        Assert.Equal(2, model.Freeze("head"));                // a module path covers its parameters
        Assert.Empty(model.NamedTrainableParameters());
        Assert.Equal(1, model.Unfreeze((n, _) => n.EndsWith("bias", StringComparison.Ordinal)));
        Assert.Equal([("head.bias")], model.NamedTrainableParameters().Select(p => p.Name).ToArray());
        Assert.Throws<ArgumentException>(() => model.Freeze("nope"));
        Assert.Throws<ArgumentException>(() => model.Freeze("em"));   // a prefix must end at a dot
        Assert.Equal(3, model.StateDict().Count);              // frozen parameters stay in the state dict
        Assert.Equal(2, model.Unfreeze());
    }

    [Fact]
    public void FrozenParametersGetNoGradientAndAreSkippedByOptimizerAndClipping()
    {
        var g = new Generator(2);
        using var model = new EmbeddingClassifier(g, 10, 4, 3, sparse: false);
        model.Freeze("emb");
        float[] before = model.Emb.Weight.Data<float>().ToArray();
        using var opt = new Adam(model.Parameters(), 0.1f);
        Tensor ids = Tensor.FromSpan<int>([1, 2, 3, 4, 5, 6], Shape.Create(2, 3));
        Tensor y = Tensor.FromSpan<long>([0, 2], Shape.Create(2));
        using (new ActivationScope())
        {
            Tensor loss = Losses.CrossEntropy(model.Forward(ids), y);
            loss.Backward();
            Assert.True(model.Emb.Weight.Grad.IsEmpty);
            float norm = GradClip.ClipNorm(model.Parameters(), 1e-3f);   // frozen ones neither counted nor scaled
            Assert.True(norm > 1e-3f);
            opt.Step();
        }

        Assert.True(model.Emb.Weight.Data<float>().SequenceEqual(before));
        Assert.False(model.Head.Weight.Grad.IsEmpty);

        // Freezing after a gradient exists frees it; unfreezing starts accumulating again and gives the state its first update.
        model.Freeze("head.weight");
        Assert.True(model.Head.Weight.Grad.IsEmpty);
        model.Unfreeze("emb");
        using (new ActivationScope())
        {
            opt.ZeroGrad();
            Losses.CrossEntropy(model.Forward(ids), y).Backward();
            Assert.False(model.Emb.Weight.Grad.IsEmpty);
            opt.Step();
        }

        Assert.False(model.Emb.Weight.Data<float>().SequenceEqual(before));
        Assert.Equal(model.Emb.Weight.Length, opt.StateTensors[0].Tensor.Length);   // exp_avg of the embedding was created at its first update
    }

    [Fact]
    public void FrozenSubgraphIsNotRecorded()
    {
        var g = new Generator(3);
        using var model = new EmbeddingClassifier(g, 10, 4, 3, sparse: false);
        model.Freeze();
        Tensor ids = Tensor.FromSpan<int>([1, 2, 3, 4, 5, 6], Shape.Create(2, 3));
        Tensor out1 = model.Forward(ids);
        Assert.False(out1.RequiresGrad);   // nothing was recorded, so a backward pass would have nothing to do
        out1.Dispose();
    }

    // ------------------------------------------------------------------------------------------------ sparse gradients

    private static (EmbeddingClassifier Model, float[] Ids, long[] Labels, int Steps) MakeSparseCase(OracleFile o, bool sparse)
    {
        var model = new EmbeddingClassifier(new Generator(5), 40, 8, 3, sparse);
        TrainCase.Set(model, "emb.weight", o.Values("in.emb"));
        TrainCase.Set(model, "head.weight", o.Values("in.hw"));
        TrainCase.Set(model, "head.bias", o.Values("in.hb"));
        return (model, o.Values("in.ids"), o.Values("in.y").Select(v => (long)v).ToArray(), o.ShapeOf("in.ids")[0]);
    }

    private static Tensor StepIds(float[] ids, int step, int rows, int seq)
    {
        var d = new int[rows * seq];
        for (int i = 0; i < d.Length; i++)
        {
            d[i] = (int)ids[step * rows * seq + i];
        }

        Tensor t = Tensor.FromSpan<int>(d, Shape.Create(rows, seq));
        ActivationScope.Track(t);
        return t;
    }

    private static float RunStep(EmbeddingClassifier model, float[] ids, long[] labels, int step, Optimizer[] opts)
    {
        int rows = 6, seq = 5;
        using var scope = new ActivationScope();
        Tensor yt = Tensor.FromSpan<long>(labels.AsSpan(step * rows, rows), Shape.Create(rows));
        ActivationScope.Track(yt);
        Tensor loss = Losses.CrossEntropy(model.Forward(StepIds(ids, step, rows, seq)), yt);
        float v = loss.Data<float>()[0];
        foreach (Optimizer o in opts)
        {
            o.ZeroGrad();
        }

        loss.Backward();
        foreach (Optimizer o in opts)
        {
            o.Step();
        }

        return v;
    }

    private static void CompareRun(string name, Optimizer[] opts, EmbeddingClassifier model, OracleFile o, float[] ids, long[] labels, int steps, double tol)
    {
        float[] want = o.Values($"out.{name}.loss");
        double maxRel = 0;
        for (int s = 0; s < steps; s++)
        {
            float l = RunStep(model, ids, labels, s, opts);
            maxRel = Math.Max(maxRel, Math.Abs(l - want[s]) / Math.Abs(want[s]));
        }

        double e1 = Scaled(model.Emb.Weight.Data<float>(), o.Values($"out.{name}.emb"));
        double e2 = Scaled(model.Head.Weight.Data<float>(), o.Values($"out.{name}.hw"));
        double e3 = Scaled(model.Head.Bias.Data<float>(), o.Values($"out.{name}.hb"));
        Log($"{name}: loss max relative diff {maxRel:E2}, embedding scaled error {e1:E2}, head weight {e2:E2}, head bias {e3:E2}");
        Assert.True(maxRel <= tol && e1 <= tol && e2 <= tol && e3 <= tol, $"{name}: loss {maxRel:E2}, emb {e1:E2}, hw {e2:E2}, hb {e3:E2} (tolerance {tol:E2})");
    }

    [Fact]
    public void SparseGradientIsTheDenseGradientsTouchedRowsAndMatchesTorch()
    {
        OracleFile o = OracleFile.Load("sparse", "training");
        var (sparseModel, ids, labels, _) = MakeSparseCase(o, true);
        var (denseModel, _, _, _) = MakeSparseCase(o, false);
        using (sparseModel)
        using (denseModel)
        {
            foreach (EmbeddingClassifier m in new[] { sparseModel, denseModel })
            {
                using var scope = new ActivationScope();
                Tensor yt = Tensor.FromSpan<long>(labels.AsSpan(0, 6), Shape.Create(6));
                ActivationScope.Track(yt);
                Losses.CrossEntropy(m.Forward(StepIds(ids, 0, 6, 5)), yt).Backward();
            }

            RowSparseGradient sg = sparseModel.Emb.Weight.SparseGrad!;
            Assert.NotNull(sg);
            Assert.True(sparseModel.Emb.Weight.Grad.IsEmpty);
            Assert.True(sg.RowCount < 40 && sg.RowCount > 3);
            for (int k = 1; k < sg.RowCount; k++)
            {
                Assert.True(sg.Rows[k] > sg.Rows[k - 1]);   // strictly ascending
            }

            Assert.DoesNotContain(0, sg.Rows.ToArray());     // the padding row is absent
            using Tensor dense = sg.ToDense();
            ReadOnlySpan<float> dd = denseModel.Emb.Weight.Grad.Data<float>();
            Assert.True(dense.Data<float>().SequenceEqual(dd), "the sparse gradient scatters to the dense gradient bit for bit");
            double e = Scaled(dense.Data<float>(), o.Values("out.sgd.grad0"));
            Log($"sparse gradient vs torch's sparse gradient: scaled error {e:E2}");
            Assert.True(e <= 1e-6);
        }
    }

    [Fact]
    public void SparseSgdIsBitEqualToDenseSgdAndMatchesTorch()
    {
        OracleFile o = OracleFile.Load("sparse", "training");
        var (sparseModel, ids, labels, steps) = MakeSparseCase(o, true);
        var (denseModel, _, _, _) = MakeSparseCase(o, false);
        using (sparseModel)
        using (denseModel)
        {
            using var s1 = new Sgd([sparseModel.Emb.Weight], 0.5f);
            using var s2 = new Sgd([sparseModel.Head.Weight, sparseModel.Head.Bias], 0.5f);
            using var d = new Sgd(denseModel.Parameters(), 0.5f);
            var losses = new List<float>();
            for (int s = 0; s < steps; s++)
            {
                float a = RunStep(sparseModel, ids, labels, s, [s1, s2]);
                float b = RunStep(denseModel, ids, labels, s, [d]);
                Assert.Equal(BitConverter.SingleToInt32Bits(b), BitConverter.SingleToInt32Bits(a));
                losses.Add(a);
            }

            Assert.True(sparseModel.Emb.Weight.Data<float>().SequenceEqual(denseModel.Emb.Weight.Data<float>()));
            Assert.True(sparseModel.Head.Weight.Data<float>().SequenceEqual(denseModel.Head.Weight.Data<float>()));
            float[] want = o.Values("out.sgd.loss");
            double rel = 0;
            for (int s = 0; s < steps; s++)
            {
                rel = Math.Max(rel, Math.Abs(losses[s] - want[s]) / Math.Abs(want[s]));
            }

            double e = Scaled(sparseModel.Emb.Weight.Data<float>(), o.Values("out.sgd.emb"));
            Log($"sparse sgd vs torch: loss max relative diff {rel:E2}, embedding scaled error {e:E2}");
            Assert.True(rel <= 1e-6 && e <= 1e-6, $"loss {rel:E2}, embedding {e:E2}");
        }
    }

    // The optimizer alone: feed the gradient torch computed (through lookups of distinct ids, so the gradient rows are exactly torch's)
    // and compare the table after one step. Only the optimizer's own arithmetic is in the error.
    [Fact]
    public void SparseSgdStepAloneMatchesTorchWithin1e7()
    {
        OracleFile o = OracleFile.Load("sparse", "training");
        float[] grad = o.Values("out.sgd.grad0");
        var rows = new List<int>();
        for (int r = 0; r < 40; r++)
        {
            if (grad.AsSpan(r * 8, 8).ToArray().Any(v => v != 0f))
            {
                rows.Add(r);
            }
        }

        using var table = Tensor.FromSpan<float>(o.Values("in.emb"), Shape.Create(40, 8)).WithRequiresGrad();
        var upstream = new float[rows.Count * 8];
        for (int k = 0; k < rows.Count; k++)
        {
            grad.AsSpan(rows[k] * 8, 8).CopyTo(upstream.AsSpan(k * 8, 8));
        }

        using var ids = Tensor.FromSpan<int>(rows.ToArray(), Shape.Create(rows.Count));
        using Tensor gradient = Tensor.FromSpan<float>(upstream, Shape.Create(rows.Count, 8));
        Ops.Embedding(table, ids, 0, true).Backward(gradient);
        Assert.Equal(rows.Count, table.SparseGrad!.RowCount);
        using var sgd = new Sgd([table], 0.5f);
        sgd.Step();
        double e = Scaled(table.Data<float>(), o.Values("out.sgd.emb1"));
        Log($"sparse sgd single step vs torch (identical gradient): scaled error {e:E2}");
        Assert.True(e <= 1e-7, $"{e:E2}");
    }

    [Fact]
    public void SparseSgdRejectsMomentumAndDecay()
    {
        OracleFile o = OracleFile.Load("sparse", "training");
        var (m, ids, labels, _) = MakeSparseCase(o, true);
        using (m)
        {
            using var mom = new Sgd([m.Emb.Weight], 0.1f, momentum: 0.9f);
            Assert.Throws<NotSupportedException>(() => RunStep(m, ids, labels, 0, [mom]));
            using var adam = new Adam([m.Emb.Weight]);
            Assert.Throws<InvalidOperationException>(() => RunStep(m, ids, labels, 0, [adam]));
        }
    }

    [Fact]
    public void SparseAdamMatchesTorchWithRepeatedAndPaddingIdsAndMixedOptimizers()
    {
        OracleFile o = OracleFile.Load("sparse", "training");
        {
            var (m, ids, labels, steps) = MakeSparseCase(o, true);
            using (m)
            {
                using var sa = new SparseAdam([m.Emb.Weight], 0.05f, epsilon: 1e-3f);
                using var aw = new AdamW(m.Head.Parameters(), 0.01f, weightDecay: 0.01f);
                CompareRun("sparseadam", [sa, aw], m, o, ids, labels, steps, 1e-6);
            }
        }

        {
            var (m, ids, labels, steps) = MakeSparseCase(o, true);
            using (m)
            {
                using var sa = new SparseAdam([m.Emb.Weight], 0.02f, beta1: 0.8f, beta2: 0.95f);
                using var ad = new Adam(m.Head.Parameters(), 0.01f);
                CompareRun("sparseadam2", [sa, ad], m, o, ids, labels, steps, 1e-6);
            }
        }
    }

    [Fact]
    public void SparseAdamLeavesUntouchedRowsAndTheirMomentsAlone()
    {
        OracleFile o = OracleFile.Load("sparse", "training");
        var (m, ids, labels, _) = MakeSparseCase(o, true);
        using (m)
        {
            using var sa = new SparseAdam([m.Emb.Weight], 0.05f);
            using var aw = new AdamW(m.Head.Parameters(), 0.01f);
            float[] before = m.Emb.Weight.Data<float>().ToArray();
            RunStep(m, ids, labels, 0, [sa, aw]);
            var touched = new HashSet<int>(m.Emb.Weight.SparseGrad!.Rows.ToArray());
            ReadOnlySpan<float> after = m.Emb.Weight.Data<float>();
            ReadOnlySpan<float> mom = sa.StateTensors[0].Tensor.Data<float>();
            for (int r = 0; r < 40; r++)
            {
                bool changed = !after.Slice(r * 8, 8).SequenceEqual(before.AsSpan(r * 8, 8));
                bool hasMoment = mom.Slice(r * 8, 8).ToArray().Any(v => v != 0f);
                Assert.Equal(touched.Contains(r), changed);
                Assert.Equal(touched.Contains(r), hasMoment);
            }
        }
    }

    [Fact]
    public void SparseGradientsAreEqualAtOneAndEightThreadsAndInBothModes()
    {
        OracleFile o = OracleFile.Load("sparse", "training");
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            string[] hashes = new string[2];
            int k = 0;
            foreach (int threads in new[] { 1, 8 })
            {
                using ComputeScopeHolder h = new(mode, threads);
                var (m, ids, labels, _) = MakeSparseCase(o, true);
                using (m)
                {
                    using var scope = new ActivationScope();
                    Tensor yt = Tensor.FromSpan<long>(labels.AsSpan(0, 6), Shape.Create(6));
                    ActivationScope.Track(yt);
                    Losses.CrossEntropy(m.Forward(StepIds(ids, 0, 6, 5)), yt).Backward();
                    RowSparseGradient sg = m.Emb.Weight.SparseGrad!;
                    hashes[k++] = Hash(sg);
                }
            }

            Assert.Equal(hashes[0], hashes[1]);
            Log($"sparse gradient hash ({mode}), 1 and 8 threads equal: {hashes[0][..16]}");
        }
    }

    private readonly struct ComputeScopeHolder : IDisposable
    {
        private readonly ArithmeticMode _mode;
        private readonly int _threads;

        public ComputeScopeHolder(ArithmeticMode mode, int threads)
        {
            _mode = TensorCompute.Mode;
            _threads = TensorCompute.Threads;
            TensorCompute.Mode = mode;
            TensorCompute.Threads = threads;
        }

        public void Dispose()
        {
            TensorCompute.Mode = _mode;
            TensorCompute.Threads = _threads;
        }
    }

    private static string Hash(RowSparseGradient g)
    {
        using var h = System.Security.Cryptography.IncrementalHash.CreateHash(System.Security.Cryptography.HashAlgorithmName.SHA256);
        foreach (int r in g.Rows)
        {
            h.AppendData(BitConverter.GetBytes(r));
        }

        foreach (float v in g.Values.Data<float>())
        {
            h.AppendData(BitConverter.GetBytes(BitConverter.SingleToInt32Bits(v)));
        }

        return Convert.ToHexString(h.GetHashAndReset());
    }

    // The exact-mode state after 6 steps of sparse training (SparseAdam on the table, AdamW on the head): the same on every ISA path,
    // thread count and OS (run with DOTNET_EnableAVX512=0, DOTNET_EnableAVX2=0 and DOTNET_EnableHWIntrinsic=0).
    private const string ExactSparse6 = "9BEDB8BFFDA223E8CFBA247CC490F60303E22635A7D071A1518A0D8ECBDA7882:3f8653d1";

    [Fact]
    public void ExactSparseTrainingHashIsPinnedAndIndependentOfThreads()
    {
        OracleFile o = OracleFile.Load("sparse", "training");
        string[] hashes = new string[2];
        int k = 0;
        foreach (int threads in new[] { 1, 8 })
        {
            using ComputeScopeHolder h = new(ArithmeticMode.Exact, threads);
            var (m, ids, labels, steps) = MakeSparseCase(o, true);
            using (m)
            {
                using var sa = new SparseAdam([m.Emb.Weight], 0.05f, epsilon: 1e-3f);
                using var aw = new AdamW(m.Head.Parameters(), 0.01f, weightDecay: 0.01f);
                float last = 0;
                for (int s = 0; s < steps; s++)
                {
                    last = RunStep(m, ids, labels, s, [sa, aw]);
                }

                hashes[k++] = TrainCase.StateHash(m, sa) + ":" + BitConverter.SingleToInt32Bits(last).ToString("x8", System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        Assert.Equal(hashes[0], hashes[1]);
        Log($"exact sparse training hash: {hashes[0]}");
        if (ExactSparse6.Length > 0)
        {
            Assert.Equal(ExactSparse6, hashes[0]);
        }
    }

    // ------------------------------------------------------------------------------------------------ checkpoints

    private static byte[] Save(NnModule m, params Optimizer[] opts)
    {
        using var ms = new MemoryStream();
        Checkpoint.Save(ms, m, opts.Length > 0 ? opts[0] : null);
        return ms.ToArray();
    }

    [Fact]
    public void SparseAdamCheckpointResumesBitExactly()
    {
        OracleFile o = OracleFile.Load("sparse", "training");

        string Run(int cut)
        {
            var (m, ids, labels, steps) = MakeSparseCase(o, true);
            using (m)
            {
                var sa = new SparseAdam([m.Emb.Weight], 0.05f, epsilon: 1e-3f);
                var aw = new AdamW(m.Head.Parameters(), 0.01f, weightDecay: 0.01f);
                int s = 0;
                for (; s < cut; s++)
                {
                    RunStep(m, ids, labels, s, [sa, aw]);
                }

                if (cut > 0)
                {
                    // Save both optimizers' state in two files, rebuild everything, load, continue.
                    byte[] ckSa, ckAw;
                    using (var ms = new MemoryStream())
                    {
                        Checkpoint.Save(ms, m, sa);
                        ckSa = ms.ToArray();
                    }

                    using (var ms = new MemoryStream())
                    {
                        Checkpoint.Save(ms, m, aw);
                        ckAw = ms.ToArray();
                    }

                    var (m2, _, _, _) = MakeSparseCase(o, true);
                    foreach ((string n, Tensor t) in m2.NamedParameters())
                    {
                        t.MutableData<float>().Fill(0.5f);
                    }

                    var sa2 = new SparseAdam([m2.Emb.Weight], 0.05f, epsilon: 1e-3f);
                    var aw2 = new AdamW(m2.Head.Parameters(), 0.01f, weightDecay: 0.01f);
                    Checkpoint.Load(new MemoryStream(ckSa), m2, sa2);
                    Checkpoint.Load(new MemoryStream(ckAw), m2, aw2);
                    for (; s < steps; s++)
                    {
                        RunStep(m2, ids, labels, s, [sa2, aw2]);
                    }

                    string h = TrainCase.StateHash(m2, sa2) + TrainCase.StateHash(m2, aw2);
                    m2.Dispose();
                    sa2.Dispose();
                    aw2.Dispose();
                    sa.Dispose();
                    aw.Dispose();
                    return h;
                }

                for (; s < steps; s++)
                {
                    RunStep(m, ids, labels, s, [sa, aw]);
                }

                string whole = TrainCase.StateHash(m, sa) + TrainCase.StateHash(m, aw);
                sa.Dispose();
                aw.Dispose();
                return whole;
            }
        }

        Assert.Equal(Run(0), Run(3));
    }

    [Fact]
    public void CheckpointWithFrozenParametersResumesAndLoadsAcrossFreezeStates()
    {
        OracleFile enc = OracleFile.Load("encoder", "training");

        TrainCase Make(bool freeze) => TrainCase.Encoder(enc, beforeOptimizer: freeze ? (m => m.Freeze("layer0")) : null);

        // Uninterrupted frozen run, 12 steps.
        TrainCase whole = Make(true);
        for (int i = 0; i < 12; i++)
        {
            whole.Step(i);
        }

        // Interrupted at step 5, resumed into a case built the same way.
        TrainCase a = Make(true);
        for (int i = 0; i < 5; i++)
        {
            a.Step(i);
        }

        byte[] bytes = Save(a.Model, a.Opt);
        TrainCase b = Make(true);
        Checkpoint.Load(new MemoryStream(bytes), b.Model, b.Opt);
        for (int i = 5; i < 12; i++)
        {
            b.Step(i);
        }

        Assert.Equal(TrainCase.StateHash(whole.Model, whole.Opt), TrainCase.StateHash(b.Model, b.Opt));

        // The same file loads into an optimizer that was built with nothing frozen (state shapes are adopted), and the file written
        // by an unfrozen optimizer loads into a frozen one.
        TrainCase c = Make(false);
        Checkpoint.Load(new MemoryStream(bytes), c.Model, c.Opt);
        Assert.Equal(bytes, Save(c.Model, c.Opt));
        TrainCase d = Make(false);
        for (int i = 0; i < 3; i++)
        {
            d.Step(i);
        }

        TrainCase e = Make(true);
        Checkpoint.Load(new MemoryStream(Save(d.Model, d.Opt)), e.Model, e.Opt);
        Assert.Equal(TrainCase.StateHash(d.Model, d.Opt), TrainCase.StateHash(e.Model, e.Opt));
    }
}
