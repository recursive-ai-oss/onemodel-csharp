using System.Security.Cryptography;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Tests.Tensors;
using RecursiveAI.OneModel.Training;

namespace RecursiveAI.OneModel.Tests.Training;

// A two-layer-or-more transformer encoder classifier: embedding, encoder layers, mean over the sequence, linear head.
internal sealed class EncoderClassifier : NnModule
{
    private readonly Embedding _emb;
    private readonly TransformerEncoderLayer[] _layers;
    private readonly Linear _cls;

    public EncoderClassifier(Generator g, int vocab, int width, int heads, int ff, int layers, int classes, float dropout = 0f, ulong seed = 0)
    {
        _emb = RegisterModule("emb", new Embedding(vocab, width, g));
        _layers = new TransformerEncoderLayer[layers];
        for (int i = 0; i < layers; i++)
        {
            _layers[i] = RegisterModule($"layer{i}", new TransformerEncoderLayer(width, heads, ff, g, dropout, seed + (ulong)(2 * i)));
        }

        _cls = RegisterModule("cls", new Linear(width, classes, g));
    }

    public override Tensor Forward(Tensor ids)
    {
        Tensor x = _emb.Forward(ids);
        foreach (TransformerEncoderLayer l in _layers)
        {
            x = l.Forward(x);
        }

        return _cls.Forward(Ops.Mean(x, 1));
    }
}

// One of the reference training runs: the model, optimizer and schedule built from an oracle file, and one training step by index.
internal sealed class TrainCase
{
    public required NnModule Model { get; init; }
    public required Optimizer Opt { get; init; }
    public LrSchedule? Schedule { get; init; }
    public required int Steps { get; init; }
    public Func<int, float> StepFn { get; set; } = _ => throw new InvalidOperationException();
    public required OracleFile Oracle { get; init; }
    public required (string Param, string Oracle)[] Map { get; init; }
    public List<float> GradNorms { get; } = new();

    public float Step(int i) => StepFn(i);

    public static void Set(NnModule m, string name, float[] values)
    {
        foreach ((string n, Tensor t) in m.NamedParameters().Concat(m.NamedBuffers()))
        {
            if (n == name)
            {
                values.CopyTo(t.MutableData<float>());
                return;
            }
        }

        throw new InvalidOperationException("no parameter " + name);
    }

    public Tensor Param(string name) => Model.StateDict().First(p => p.Name == name).Tensor;

    private static Tensor Batch(float[] all, int rowLen, ReadOnlySpan<float> order, Shape shape)
    {
        var data = new float[order.Length * rowLen];
        for (int r = 0; r < order.Length; r++)
        {
            Array.Copy(all, (int)order[r] * rowLen, data, r * rowLen, rowLen);
        }

        Tensor t = Tensor.FromSpan<float>(data, shape);
        ActivationScope.Track(t);
        return t;
    }

    private static Tensor Labels(float[] y, ReadOnlySpan<float> order)
    {
        var data = new long[order.Length];
        for (int r = 0; r < data.Length; r++)
        {
            data[r] = (long)y[(int)order[r]];
        }

        Tensor t = Tensor.FromSpan<long>(data, Shape.Create(data.Length));
        ActivationScope.Track(t);
        return t;
    }

    private static ReadOnlySpan<float> Row(OracleFile o, int step)
    {
        int b = o.ShapeOf("in.order")[1];
        return o.Values("in.order").AsSpan(step * b, b);
    }

    public static TrainCase Mlp(OracleFile o)
    {
        var g = new Generator(1);
        var model = new Sequential(new Linear(20, 32, g), Lambda.Relu(), new Linear(32, 4, g));
        (string, string)[] map = [("0.weight", "w1"), ("0.bias", "b1"), ("2.weight", "w2"), ("2.bias", "b2")];
        foreach ((string p, string k) in map)
        {
            Set(model, p, o.Values("in." + k));
        }

        var opt = new Sgd(model.Parameters(), 0.05f, momentum: 0.9f, weightDecay: 1e-4f);
        float[] x = o.Values("in.x"), y = o.Values("in.y");
        int b = o.ShapeOf("in.order")[1];
        return new TrainCase
        {
            Model = model, Opt = opt, Steps = o.ShapeOf("in.order")[0], Oracle = o, Map = map,
            StepFn = i =>
            {
                using var scope = new ActivationScope();
                ReadOnlySpan<float> idx = Row(o, i);
                Tensor loss = Losses.CrossEntropy(model.Forward(Batch(x, 20, idx, Shape.Create(b, 20))), Labels(y, idx));
                float v = loss.Data<float>()[0];
                opt.ZeroGrad();
                loss.Backward();
                opt.Step();
                return v;
            },
        };
    }

    public static TrainCase Cnn(OracleFile o)
    {
        var g = new Generator(2);
        var model = new Sequential(
            new Conv2d(1, 4, 3, g, padding: 1), new BatchNorm2d(4), Lambda.Relu(), Lambda.MaxPool2d(2),
            new Conv2d(4, 8, 3, g, padding: 1), new BatchNorm2d(8), Lambda.Relu(), Lambda.AvgPool2d(2),
            Lambda.Flatten(), new Linear(72, 5, g));
        (string, string)[] map =
        [
            ("0.weight", "c1w"), ("0.bias", "c1b"), ("1.weight", "bn1g"), ("1.bias", "bn1b"),
            ("4.weight", "c2w"), ("4.bias", "c2b"), ("5.weight", "bn2g"), ("5.bias", "bn2b"), ("9.weight", "fcw"), ("9.bias", "fcb"),
        ];
        foreach ((string p, string k) in map)
        {
            Set(model, p, o.Values("in." + k));
        }

        int steps = o.ShapeOf("in.order")[0], b = o.ShapeOf("in.order")[1];
        var opt = new AdamW(model.Parameters(), 3e-3f, weightDecay: 0.01f);
        var sched = new CosineLr(3e-3f, steps);
        float[] x = o.Values("in.x"), y = o.Values("in.y");
        var tc = new TrainCase { Model = model, Opt = opt, Schedule = sched, Steps = steps, Oracle = o, Map = map };
        tc.StepFn = i =>
        {
            using var scope = new ActivationScope();
            ReadOnlySpan<float> idx = Row(o, i);
            Tensor loss = Losses.CrossEntropy(model.Forward(Batch(x, 144, idx, Shape.Create(b, 1, 12, 12))), Labels(y, idx));
            float v = loss.Data<float>()[0];
            opt.ZeroGrad();
            loss.Backward();
            tc.GradNorms.Add(GradClip.ClipNorm(model.Parameters(), 0.5f));
            sched.Apply(opt);
            opt.Step();
            return v;
        };
        return tc;
    }

    public static TrainCase Encoder(OracleFile o, float dropout = 0f, Func<Generator>? gen = null)
    {
        var model = new EncoderClassifier(gen?.Invoke() ?? new Generator(3), 11, 16, 2, 32, 2, 2, dropout, seed: 77);
        var map = new List<(string, string)> { ("emb.weight", "emb") };
        for (int i = 0; i < 2; i++)
        {
            foreach (string q in new[] { "q", "k", "v", "o" })
            {
                map.Add(($"layer{i}.attn.{q}.weight", $"l{i}.w{q}"));
                map.Add(($"layer{i}.attn.{q}.bias", $"l{i}.b{q}"));
            }

            map.Add(($"layer{i}.norm1.weight", $"l{i}.ln1g"));
            map.Add(($"layer{i}.norm1.bias", $"l{i}.ln1b"));
            map.Add(($"layer{i}.ff1.weight", $"l{i}.w1"));
            map.Add(($"layer{i}.ff1.bias", $"l{i}.b1"));
            map.Add(($"layer{i}.ff2.weight", $"l{i}.w2"));
            map.Add(($"layer{i}.ff2.bias", $"l{i}.b2"));
            map.Add(($"layer{i}.norm2.weight", $"l{i}.ln2g"));
            map.Add(($"layer{i}.norm2.bias", $"l{i}.ln2b"));
        }

        map.Add(("cls.weight", "clsw"));
        map.Add(("cls.bias", "clsb"));
        foreach ((string p, string k) in map)
        {
            Set(model, p, o.Values("in." + k));
        }

        var opt = new Adam(model.Parameters(), 2e-3f);
        var sched = new WarmupLr(new ConstantLr(2e-3f), 10, 0.1f);
        float[] ids = o.Values("in.ids"), y = o.Values("in.y");
        int steps = o.ShapeOf("in.order")[0], b = o.ShapeOf("in.order")[1], seq = o.ShapeOf("in.ids")[1];
        return new TrainCase
        {
            Model = model, Opt = opt, Schedule = sched, Steps = steps, Oracle = o, Map = [.. map],
            StepFn = i =>
            {
                using var scope = new ActivationScope();
                ReadOnlySpan<float> idx = Row(o, i);
                var idData = new int[b * seq];
                for (int r = 0; r < b; r++)
                {
                    for (int j = 0; j < seq; j++)
                    {
                        idData[r * seq + j] = (int)ids[(int)idx[r] * seq + j];
                    }
                }

                Tensor idt = Tensor.FromSpan<int>(idData, Shape.Create(b, seq));
                ActivationScope.Track(idt);
                Tensor loss = Losses.CrossEntropy(model.Forward(idt), Labels(y, idx));
                float v = loss.Data<float>()[0];
                opt.ZeroGrad();
                loss.Backward();
                sched.Apply(opt);
                opt.Step();
                return v;
            },
        };
    }

    // SHA-256 over every state tensor of the model and optimizer (name, shape, raw bits) and the step count.
    public static string StateHash(NnModule model, Optimizer? opt)
    {
        using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        void Add(string name, Tensor t)
        {
            h.AppendData(System.Text.Encoding.UTF8.GetBytes(name));
            foreach (int d in t.Shape.Dimensions)
            {
                h.AppendData(BitConverter.GetBytes(d));
            }

            switch (t.DType)
            {
                case DType.F32:
                    h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(t.Data<float>()));
                    break;
                case DType.I32:
                    h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(t.Data<int>()));
                    break;
                case DType.I64:
                    h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(t.Data<long>()));
                    break;
            }
        }

        foreach ((string n, Tensor t) in model.StateDict())
        {
            Add(n, t);
        }

        if (opt is not null)
        {
            foreach ((string n, Tensor t) in opt.StateTensors)
            {
                Add("opt." + n, t);
            }

            h.AppendData(BitConverter.GetBytes(opt.StepCount));
            h.AppendData(BitConverter.GetBytes(opt.LearningRate));
        }

        return Convert.ToHexString(h.GetHashAndReset());
    }
}
