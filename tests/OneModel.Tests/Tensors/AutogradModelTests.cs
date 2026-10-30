using System.Globalization;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Tensors;

[Collection("autograd")]
public class AutogradModelTests
{
    private static void Log(string line)
    {
        string? path = Environment.GetEnvironmentVariable("ONEMODEL_GRADCHECK_LOG");
        if (path is not null)
        {
            File.AppendAllText(path, line + "\n");
        }
    }

    // Scaled error: max |a - b| / max(max |b|, floor), as in docs/oracles.md.
    private static double Scaled(ReadOnlySpan<float> a, ReadOnlySpan<float> b, double floor = 1e-12)
    {
        double max = 0, mag = floor;
        for (int i = 0; i < a.Length; i++)
        {
            max = Math.Max(max, Math.Abs((double)a[i] - b[i]));
            mag = Math.Max(mag, Math.Abs(b[i]));
        }

        return max / mag;
    }

    private static void Compare(string model, OracleFile o, Dictionary<string, Tensor> grads, double tolerance)
    {
        double largest = 0;
        foreach (string key in grads.Keys)
        {
            largest = Math.Max(largest, o.Values("grad." + key).Max(v => Math.Abs(v)));
        }

        foreach (KeyValuePair<string, Tensor> kv in grads)
        {
            string key = "grad." + kv.Key;
            float[] expected = o.Values(key);
            Assert.Equal(o.ShapeOf(key), kv.Value.Shape);
            if (expected.Max(v => Math.Abs(v)) < 1e-5 * largest)
            {
                // Zero in theory (the bias of a convolution that feeds a batch norm; the key bias of attention, which the softmax
                // ignores): both sides are rounding noise, so require only that ours is noise too.
                double noise = kv.Value.Data<float>().ToArray().Max(v => Math.Abs(v)) / largest;
                Log(string.Create(CultureInfo.InvariantCulture, $"| {model} {kv.Key} {kv.Value.Shape} | zero in theory; ours {noise:E2} of the largest gradient |"));
                Assert.True(noise < 1e-5, $"{model} {kv.Key}: expected ~0, got {noise:E2} of the largest gradient");
                continue;
            }

            double err = Scaled(kv.Value.Data<float>(), expected);
            Log(string.Create(CultureInfo.InvariantCulture, $"| {model} {kv.Key} {kv.Value.Shape} | {err:E2} |"));
            Assert.True(err <= tolerance, $"{model} {kv.Key}: scaled error {err:E2} exceeds {tolerance:E1}");
        }
    }

    [Fact]
    public void CnnMatchesPyTorch()
    {
        OracleFile o = OracleFile.Load("cnn");
        var m = new CnnModel(o);
        float loss = m.Step();
        Assert.Equal(o.Values("out.loss")[0], loss, 1e-5f);
        Compare("cnn", o, m.Grads(), 1e-4);
        Assert.True(Scaled(m.Rm1.Data<float>(), o.Values("out.rm1")) < 1e-5);
        Assert.True(Scaled(m.Rv2.Data<float>(), o.Values("out.rv2")) < 1e-5);
    }

    [Fact]
    public void EncoderMatchesPyTorch()
    {
        OracleFile o = OracleFile.Load("encoder");
        var m = new EncoderModel(o);
        float loss = m.Step();
        Assert.Equal(o.Values("out.loss")[0], loss, o.Values("out.loss")[0] * 1e-4f);
        Compare("encoder", o, m.Grads(), 1e-4);
    }

    [Fact]
    public void GradientsAccumulateAcrossSteps()
    {
        OracleFile o = OracleFile.Load("cnn");
        var m = new CnnModel(o);
        m.Step();
        float[] once = m.Params["fcw"].Grad.ToArray<float>();
        using (var scope = new ActivationScope())
        {
            m.Forward(out _).Backward(); // no ZeroGrad: adds a second time
        }

        float[] twice = m.Params["fcw"].Grad.ToArray<float>();
        // The second forward sees updated running stats only; training-mode outputs are unchanged, so the gradient doubles.
        for (int i = 0; i < once.Length; i++)
        {
            Assert.Equal(2 * once[i], twice[i], Math.Abs(once[i]) * 1e-5f + 1e-7f);
        }
    }

    private static string RunHash(Func<(float Loss, Dictionary<string, Tensor> Grads)> run)
    {
        (float loss, Dictionary<string, Tensor> grads) = run();
        return GradHash.Of(grads, loss);
    }

    private static string CnnHash() => RunHash(() =>
    {
        var m = new CnnModel(OracleFile.Load("cnn"));
        float l = m.Step();
        return (l, m.Grads());
    });

    private static string EncoderHash() => RunHash(() =>
    {
        var m = new EncoderModel(OracleFile.Load("encoder"));
        float l = m.Step();
        return (l, m.Grads());
    });

    [Fact]
    public void FastModeIsIndependentOfThreadCount()
    {
        ExecutionMode mode = TensorCompute.Mode;
        int threads = TensorCompute.Threads;
        try
        {
            TensorCompute.Mode = ExecutionMode.Fast;
            TensorCompute.Threads = 1;
            string c1 = CnnHash(), e1 = EncoderHash();
            TensorCompute.Threads = 8;
            Assert.Equal(c1, CnnHash());
            Assert.Equal(e1, EncoderHash());
            TensorCompute.Threads = 3;
            Assert.Equal(c1, CnnHash());
            Assert.Equal(e1, EncoderHash());
            Log($"fast cnn {c1}\nfast encoder {e1}");
        }
        finally
        {
            TensorCompute.Mode = mode;
            TensorCompute.Threads = threads;
        }
    }

    // The exact-mode hashes below are pinned: the same bits on every ISA path (run the suite with DOTNET_EnableAVX512=0,
    // DOTNET_EnableAVX2=0 and DOTNET_EnableHWIntrinsic=0), every thread count and every OS.
    [Fact]
    public void ExactModeHashesArePinned()
    {
        ExecutionMode mode = TensorCompute.Mode;
        int threads = TensorCompute.Threads;
        try
        {
            TensorCompute.Mode = ExecutionMode.Exact;
            string c = "", e = "";
            foreach (int t in new[] { 1, 8 })
            {
                TensorCompute.Threads = t;
                c = CnnHash();
                e = EncoderHash();
                Log($"exact threads={t} cnn {c}\nexact threads={t} encoder {e}");
                Assert.Equal(ExactCnnHash, c);
                Assert.Equal(ExactEncoderHash, e);
            }
        }
        finally
        {
            TensorCompute.Mode = mode;
            TensorCompute.Threads = threads;
        }
    }

    private const string ExactCnnHash = "634FFFDF55082783FA1180B2395AFA1696FC00660C4E4D9967296B323760C1E1";
    private const string ExactEncoderHash = "E9ED495122DBBD465EB4EA6A51583CC830DBEF1B0CAE3AC9C934233EC498A576";

    [Fact]
    public void SteadyStateStepAllocatesNoTensorMemory()
    {
        var cnn = new CnnModel(OracleFile.Load("cnn"));
        var enc = new EncoderModel(OracleFile.Load("encoder"));
        for (int i = 0; i < 4; i++)
        {
            cnn.Step();
            enc.Step();
        }

        PoolStats before = BufferPool.Shared.GetStats();
        long managedCnn = Measure(() => cnn.Step());
        long managedEnc = Measure(() => enc.Step());
        PoolStats after = BufferPool.Shared.GetStats();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < 20; i++)
        {
            cnn.Step();
        }

        double cnnMs = sw.Elapsed.TotalMilliseconds / 20;
        sw.Restart();
        for (int i = 0; i < 20; i++)
        {
            enc.Step();
        }

        double encMs = sw.Elapsed.TotalMilliseconds / 20;
        Log(string.Create(CultureInfo.InvariantCulture, $"indicative time per step (not a measurement): cnn {cnnMs:F2} ms, encoder {encMs:F2} ms"));
        Log($"steady-state managed bytes per step: cnn {managedCnn}, encoder {managedEnc}; native allocations {after.NativeAllocations - before.NativeAllocations}; outstanding {after.OutstandingBytes - before.OutstandingBytes}");
        Assert.Equal(before.NativeAllocations, after.NativeAllocations);
        Assert.Equal(before.OutstandingBytes, after.OutstandingBytes);
    }

    private static long Measure(Action step)
    {
        long a = GC.GetAllocatedBytesForCurrentThread();
        step();
        return GC.GetAllocatedBytesForCurrentThread() - a;
    }
}

[CollectionDefinition("autograd", DisableParallelization = true)]
public class AutogradSerial;
