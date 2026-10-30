using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using RecursiveAI.OneModel;
using RecursiveAI.OneModel.Format;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Training;
using RecursiveAI.OneModel.Training.Data;

// Calls the library from a NativeAOT binary: the tensor/kernel self-check, then a small model end to end (import, session, plan,
// fast and exact runs) against the ONNX Runtime reference output stored with the model. CI fails on any trim or AOT warning when
// publishing, and on x64 the session must have picked an AVX2 or AVX-512 path (the project publishes with a v3 instruction-set
// baseline; a default NativeAOT publish would silently run the 4-lane kernels).
ulong hash = OneModelInfo.SelfCheck();
Console.WriteLine("self-check " + hash.ToString("x16", CultureInfo.InvariantCulture));
if (hash != 0x4242dc5249c33625UL)
{
    return 1;
}

Assembly asm = typeof(Program).Assembly;
using JsonDocument doc = JsonDocument.Parse(Open("reference.json"));
JsonElement c = doc.RootElement.GetProperty("cases").GetProperty("mlp-5");
float[] input = [.. c.GetProperty("input").EnumerateArray().Select(e => (float)e.GetDouble())];
float[] expected = [.. c.GetProperty("output").EnumerateArray().Select(e => (float)e.GetDouble())];
int[] shape = [.. c.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32())];

foreach (ExecutionMode mode in new[] { ExecutionMode.Fast, ExecutionMode.Exact })
{
    using Model m = Model.Load(Open("mlp.onnx"), new SessionOptions { Mode = mode });
    float[] y = m.Run(input, shape);
    double max = 0, diff = 0;
    for (int i = 0; i < y.Length; i++)
    {
        max = Math.Max(max, Math.Abs(expected[i]));
        diff = Math.Max(diff, Math.Abs(y[i] - expected[i]));
    }

    string h = Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes<float>(y)))[..16];
    Console.WriteLine($"{mode}: kernel path {m.Session.KernelPath}, scaled error {diff / max:E2}, output hash {h}");
    if (diff / max > 1e-4)
    {
        return 2;
    }

    if (RuntimeInformation.ProcessArchitecture == Architecture.X64 && m.Session.KernelPath is not ("Avx2" or "Avx512"))
    {
        Console.WriteLine("expected the AVX2 or AVX-512 kernels on x64");
        return 3;
    }
}

// The README one-liner on a real encoder, when its file is supplied (ONEMODEL_MINILM); skipped otherwise.
string? minilm = Environment.GetEnvironmentVariable("ONEMODEL_MINILM");
if (!string.IsNullOrEmpty(minilm) && File.Exists(minilm))
{
    using Model enc = Model.Load(minilm);
    long[] tokens = [101, 7592, 2088, 102];
    float[] v = enc.Run(tokens);
    bool finite = v.Length == tokens.Length * 384 && v.All(float.IsFinite);
    string h = Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes<float>(v)))[..16];
    Console.WriteLine($"minilm: {v.Length} values, finite {finite}, hash {h}");
    if (!finite)
    {
        return 5;
    }
}

// Data-movement operators (transpose, slice, gather, concat, pad, top-k, tile) move bits, so they must equal ONNX Runtime's exactly.
using JsonDocument moveDoc = JsonDocument.Parse(Open("shape_ops.json"));
float[] moveInput = [.. moveDoc.RootElement.GetProperty("input").EnumerateArray().Select(e => (float)e.GetDouble())];
float[] moveExpected = [.. moveDoc.RootElement.GetProperty("output").EnumerateArray().Select(e => (float)e.GetDouble())];
int[] moveShape = [.. moveDoc.RootElement.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32())];
foreach (ExecutionMode mode in new[] { ExecutionMode.Fast, ExecutionMode.Exact })
{
    using Model m = Model.Load(Open("shape_ops.onnx"), new SessionOptions { Mode = mode });
    float[] y = m.Run(moveInput, moveShape);
    bool same = y.AsSpan().SequenceEqual(moveExpected);
    Console.WriteLine($"{mode}: shape and indexing operators {(same ? "equal" : "DIFFER")} to ONNX Runtime");
    if (!same)
    {
        return 4;
    }
}

// int8: a dynamically quantized MLP and a static QDQ convolution net (made with ONNX Runtime's quantization tool) against ONNX Runtime's
// output. Integer products are exact, so the only differences are float roundings and the odd 8-bit step; the bound is loose on purpose.
foreach ((string name, double bound) in new[] { ("int8_dyn", 2e-2), ("int8_qdq", 1e-1) })
{
    using JsonDocument qd = JsonDocument.Parse(Open(name + ".json"));
    float[] qIn = [.. qd.RootElement.GetProperty("input").EnumerateArray().Select(e => (float)e.GetDouble())];
    float[] qWant = [.. qd.RootElement.GetProperty("output").EnumerateArray().Select(e => (float)e.GetDouble())];
    int[] qShape = [.. qd.RootElement.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32())];
    using Model m = Model.Load(Open(name + ".onnx"));
    float[] y = m.Run(qIn, qShape);
    double max = 1e-30, diff = 0;
    for (int i = 0; i < y.Length; i++)
    {
        max = Math.Max(max, Math.Abs(qWant[i]));
        diff = Math.Max(diff, Math.Abs(y[i] - qWant[i]));
    }

    Console.WriteLine($"{name}: kernel path {m.Session.KernelPath}, scaled error {diff / max:E2}");
    if (!(diff / max <= bound))
    {
        return 6;
    }
}

// The OneModel format: import, save to a stream, load that through the same entry point (chosen by magic), and get the same bits.
foreach (string name in new[] { "mlp", "int8_qdq" })
{
    using JsonDocument fd = JsonDocument.Parse(Open(name == "mlp" ? "reference.json" : "int8_qdq.json"));
    JsonElement fc = name == "mlp" ? fd.RootElement.GetProperty("cases").GetProperty("mlp-5") : fd.RootElement;
    float[] fIn = [.. fc.GetProperty("input").EnumerateArray().Select(e => (float)e.GetDouble())];
    int[] fShape = [.. fc.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32())];
    using var onnxBytes = new MemoryStream();
    Open(name + ".onnx").CopyTo(onnxBytes);
    using var saved = new MemoryStream();
    using (IrModel ir = OnnxImporter.Load(onnxBytes.ToArray()))
    {
        ModelFormat.Save(ir, saved);
    }

    saved.Position = 0;
    using Model direct = Model.Load(new MemoryStream(onnxBytes.ToArray()));
    using Model viaFormat = Model.Load(saved);
    bool same = direct.Run(fIn, fShape).AsSpan().SequenceEqual(viaFormat.Run(fIn, fShape));
    Console.WriteLine($"{name}: saved and reloaded in the OneModel format ({saved.Length} bytes), outputs {(same ? "equal" : "DIFFER")}");
    if (!same)
    {
        return 7;
    }
}

// Reverse-mode autodiff: a small conv / norm / attention net, analytic gradients against central finite differences, in fast and
// exact arithmetic (float32 finite differences on a small net: noise of a few 1e-3, while a wrong gradient is off by order one).
foreach (ExecutionMode gm in new[] { ExecutionMode.Fast, ExecutionMode.Exact })
{
    TensorCompute.Mode = gm;
    var rng = new Random(7);
    Tensor Rand(params int[] dims)
    {
        Tensor t = Tensor.Create(DType.F32, Shape.Create(dims));
        Span<float> d = t.MutableData<float>();
        for (int i = 0; i < d.Length; i++)
        {
            d[i] = (float)((rng.NextDouble() * 2) - 1);
        }

        return t.WithRequiresGrad();
    }

    Tensor[] gin = [Rand(1, 2, 4, 4), Rand(3, 2, 3, 3), Rand(1, 2, 3, 4), Rand(3)];
    GradientCheckResult[] gr = GradientCheck.Run(t =>
    {
        Tensor h = Ops.Gelu(Ops.Conv2d(t[0], t[1], default, new Conv2dParams(PadH: 1, PadW: 1)));   // [1,3,4,4]
        Tensor a = Ops.AvgPool2d(h, new Pool2dParams(2, 2, 2, 2, 0, 0)).Reshape(1, 1, 3, 4);       // [1,1,3,4]
        Tensor att = Ops.Attention(a, t[2].Slice(1, 0, 1), a);                    // [1,1,3,4]
        return Ops.LogSoftmax(Ops.Add(att.Reshape(3, 4), t[3].Reshape(3, 1)));
    }, gin);
    double worst = gr.Max(r => r.ScaledError);
    Console.WriteLine($"autograd {gm}: finite-difference check, worst scaled error {worst:E2}");
    if (!(worst < 2e-2))
    {
        return 8;
    }
}

// Training: a small conv net and a transformer layer learn a toy task with AdamW, a cosine schedule, clipping and dropout, and a
// checkpoint saved mid-run reloads to the same bits, in fast and exact arithmetic.
foreach (ExecutionMode tm in new[] { ExecutionMode.Fast, ExecutionMode.Exact })
{
    TensorCompute.Mode = tm;
    var gen = new Generator(5);
    var net = new Sequential(new Conv2d(1, 4, 3, gen, padding: 1), new BatchNorm2d(4), Lambda.Relu(), Lambda.MaxPool2d(2),
        Lambda.Flatten(), new Dropout(0.1f, 3), new Linear(4 * 4 * 4, 3, gen));
    var enc = new TransformerEncoderLayer(8, 2, 16, gen, dropout: 0.1f, dropoutSeed: 9);
    using var holder = new Sequential(net, enc);   // owns both; its state is what the checkpoint saves
    var parameters = holder.Parameters();
    using var opt = new AdamW(parameters, 1e-2f);
    var sched = new CosineLr(1e-2f, 30);
    var data = new Generator(11);
    var xs = new float[12 * 64];
    var ys = new long[12];
    for (int i = 0; i < ys.Length; i++)
    {
        ys[i] = i % 3;
        for (int j = 0; j < 64; j++)
        {
            xs[(i * 64) + j] = data.Uniform(-1, 1) + ((j % 3) == ys[i] ? 1f : 0f);
        }
    }

    float first = 0, last = 0;
    for (int step = 0; step < 30; step++)
    {
        using var scope = new ActivationScope();
        Tensor x = Tensor.FromSpan<float>(xs, Shape.Create(12, 1, 8, 8));
        Tensor y = Tensor.FromSpan<long>(ys, Shape.Create(12));
        ActivationScope.Track(x);
        ActivationScope.Track(y);
        Tensor loss = Losses.CrossEntropy(net.Forward(x), y, labelSmoothing: 0.05f);
        Tensor seq = enc.Forward(Ops.Add(x.Reshape(12, 8, 8), Tensor.Zeros(DType.F32, Shape.Create(12, 8, 8))));
        loss = Ops.Add(loss, Ops.Mul(Ops.Mean(Ops.Mul(seq, seq)), Tensor.Scalar(0.01f)));
        float lv = loss.Data<float>()[0];
        first = step == 0 ? lv : first;
        last = lv;
        opt.ZeroGrad();
        loss.Backward();
        GradClip.ClipNorm(parameters, 1f);
        sched.Apply(opt);
        opt.Step();
    }

    using var ms = new MemoryStream();
    Checkpoint.Save(ms, holder, opt);
    ms.Position = 0;
    var gen2 = new Generator(77);
    using var holder2 = new Sequential(
        new Sequential(new Conv2d(1, 4, 3, gen2, padding: 1), new BatchNorm2d(4), Lambda.Relu(), Lambda.MaxPool2d(2),
            Lambda.Flatten(), new Dropout(0.1f, 3), new Linear(4 * 4 * 4, 3, gen2)),
        new TransformerEncoderLayer(8, 2, 16, gen2, dropout: 0.1f, dropoutSeed: 9));
    using var opt2 = new AdamW(holder2.Parameters(), 1e-2f);
    Checkpoint.Load(ms, holder2, opt2);
    bool same = holder.StateDict().Zip(holder2.StateDict()).All(p => p.First.Tensor.DType != DType.F32 || p.First.Tensor.Data<float>().SequenceEqual(p.Second.Tensor.Data<float>()));
    Console.WriteLine($"training {tm}: loss {first:F4} -> {last:F4}, checkpoint round trip {(same ? "equal" : "DIFFERS")}, step {opt2.StepCount}");
    if (!(last < first) || !same || opt2.StepCount != 30)
    {
        return 11;
    }
}

// Recurrent, CTC, transposed-convolution and resize gradients: a bidirectional LSTM and a GRU into a CTC loss, and a small decoder.
foreach (ExecutionMode gm in new[] { ExecutionMode.Fast, ExecutionMode.Exact })
{
    TensorCompute.Mode = gm;
    var rng = new Random(11);
    Tensor Rand(params int[] dims)
    {
        Tensor t = Tensor.Create(DType.F32, Shape.Create(dims));
        Span<float> d = t.MutableData<float>();
        for (int i = 0; i < d.Length; i++)
        {
            d[i] = (float)((rng.NextDouble() * 1.2) - 0.6);
        }

        return t.WithRequiresGrad();
    }

    // x [T=5, B=2, 3]; one bidirectional LSTM layer (H=2), a GRU layer (H=2), a linear head to 4 classes, CTC with blank 0.
    Tensor[] rin = [Rand(5, 2, 3), Rand(8, 3), Rand(8, 2), Rand(8), Rand(8, 3), Rand(8, 2), Rand(8), Rand(6, 4), Rand(6, 2), Rand(4, 2)];
    GradientCheckResult[] rr = GradientCheck.Run(t =>
    {
        RnnLayer[] layers = [new RnnLayer(t[1], t[2], t[3], default), new RnnLayer(t[4], t[5], t[6], default)];
        Tensor y = Ops.Lstm(t[0], layers, bidirectional: true);                         // [5,2,4]
        Tensor g = Ops.GruLayer(y, t[7], t[8]);                                         // [5,2,2]
        Tensor lp = Ops.LogSoftmax(Ops.Add(Ops.MatMul(g.Reshape(10, 2), t[9].T()), Ops.Sum(t[9], 1)).Reshape(5, 2, 4));
        return Ops.CtcLoss(lp, [1, 2, 1], [5, 4], [2, 1]);
    }, rin, step: 2e-2);
    double rworst = rr.Max(r => r.ScaledError);
    Console.WriteLine($"autograd {gm}: LSTM/GRU/CTC finite-difference check, worst scaled error {rworst:E2}");
    if (!(rworst < 2e-2))
    {
        return 9;
    }

    Tensor[] din = [Rand(1, 2, 3, 3), Rand(2, 2, 2, 2), Rand(2)];
    GradientCheckResult[] dr = GradientCheck.Run(t =>
        Ops.Resize2d(Ops.ConvTranspose2d(t[0], t[1], t[2], new ConvTranspose2dParams(StrideH: 2, StrideW: 2)), 7, 5, InterpolationMode.Bilinear), din);
    double dworst = dr.Max(r => r.ScaledError);
    Console.WriteLine($"autograd {gm}: ConvTranspose2d/Resize2d finite-difference check, worst scaled error {dworst:E2}");
    if (!(dworst < 2e-2))
    {
        return 10;
    }
}

{
    // Data loading: the same batches from the calling thread and from two workers, hashed.
    var images = new ImageExample[40];
    for (int i = 0; i < images.Length; i++)
    {
        int w = 6 + (i * 7) % 19;
        var px = new float[2 * 4 * w];
        for (int k = 0; k < px.Length; k++)
        {
            px[k] = ((i + k) % 13) / 13f;
        }

        images[i] = new ImageExample(px, 2, 4, w, new[] { 1 + (i % 5), 2 + (i % 3) }, i % 4);
    }

    string LoaderHash(int workers)
    {
        var loader = new DataLoader<ImageExample, ImageBatch>(new ListDataset<ImageExample>(images), EpochSampler.Shuffled(images.Length, 6, 3),
            new ImageBatchCollator(4, -1f, (Span<float> px, int c, int h, int w, ref SampleRng rng) => px[0] += rng.NextSingle()),
            new DataLoaderOptions { Seed = 9, Workers = workers, Prefetch = 3 });
        using IncrementalHash ih = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        using BatchSequence<ImageBatch> seq = loader.Start(2, 20);
        while (seq.TryNext(out ImageBatch? b))
        {
            ih.AppendData(MemoryMarshal.AsBytes(b.Images.Data<float>()));
            ih.AppendData(MemoryMarshal.AsBytes(b.Targets.Data<int>()));
            ih.AppendData(MemoryMarshal.AsBytes(b.Widths.Data<int>()));
            b.Dispose();
        }

        return Convert.ToHexString(ih.GetHashAndReset());
    }

    string lh0 = LoaderHash(0), lh2 = LoaderHash(2);
    Console.WriteLine($"data loader: 18 batches, hash {lh0[..16]}, workers 0 and 2 {(lh0 == lh2 ? "equal" : "DIFFER")}");
    if (lh0 != lh2)
    {
        return 11;
    }
}

TensorCompute.Mode = ExecutionMode.Fast;
return 0;

Stream Open(string name) =>
    asm.GetManifestResourceStream(name) ?? throw new FileNotFoundException("embedded resource " + name);
