using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using CrnnFinetune;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Training;

// crnn-finetune train --weights DIR --sets DIR --batches DIR --out DIR [--steps N] [--lr 6e-5] [--warmup 200] [--wd 0.01]
//                     [--schedule-steps N] [--until N] [--threads N] [--exact] [--eval-every N] [--skip-eval] [--follow]
// crnn-finetune export --weights DIR --template MODEL.onm --out FILE [--format-version 0|1] [--provenance TEXT]
// crnn-finetune eval  --weights DIR --sets DIR [--threads N] [--exact]
//
// weights DIR: weights.json + weights.bin (see tools/crnn_ref/dump_inputs.py); sets DIR: set-<name>.bin; batches DIR: batch-<step>.bin.
// Recipe (a published fine-tune recipe): AdamW (0.9, 0.999, 1e-8), 200-step linear warm-up then a floored half cosine, gradient norm
// clip 5, CTC with blank 0, mean reduction and zero_infinity, batch statistics in training mode.
string cmd = args.Length > 0 ? args[0] : "";
var o = new Dictionary<string, string>();
for (int i = 1; i < args.Length; i++)
{
    if (!args[i].StartsWith("--", StringComparison.Ordinal)) throw new ArgumentException("unexpected " + args[i]);
    string k = args[i][2..];
    o[k] = i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal) ? args[++i] : "true";
}

string Need(string k) => o.TryGetValue(k, out string? v) ? v : throw new ArgumentException("missing --" + k);
int Int(string k, int d) => o.TryGetValue(k, out string? v) ? int.Parse(v, CultureInfo.InvariantCulture) : d;
float Flt(string k, float d) => o.TryGetValue(k, out string? v) ? float.Parse(v, CultureInfo.InvariantCulture) : d;

if (o.ContainsKey("exact")) TensorCompute.Mode = ExecutionMode.Exact;
if (o.ContainsKey("threads")) TensorCompute.Threads = Int("threads", 1);

var (config, configJson, tensors) = WeightsFile.Read(Need("weights"));
using var model = new Crnn(config);
model.LoadStateDict(tensors.Select(t => (t.Name, Tensor.FromSpan<float>(t.Data, Shape.Create(t.Shape)))));

void EvalAll(string setsDir, string label, TextWriter w)
{
    foreach (string f in Directory.GetFiles(setsDir, "set-*.bin").OrderBy(x => x, StringComparer.Ordinal))
    {
        string name = Path.GetFileNameWithoutExtension(f)[4..];
        var (cer, acc, n) = LineSet.Read(f).Evaluate(model);
        w.WriteLine(JsonSerializer.Serialize(new { label, set = name, cer, line_acc = acc, lines = n }));
        w.Flush();
    }
}

if (cmd == "export")
{
    Container.Export(Need("weights"), Need("template"), Need("out"), Int("format-version", 1), o.GetValueOrDefault("provenance", "fine-tuned with crnn-finetune"));
    return 0;
}

if (cmd == "eval")
{
    EvalAll(Need("sets"), "eval", Console.Out);
    return 0;
}

if (cmd != "train") { Console.Error.WriteLine("usage: crnn-finetune train|eval ..."); return 2; }

string batches = Need("batches"), outDir = Need("out");
int steps = Int("steps", 2000);
Directory.CreateDirectory(outDir);
model.Train();
var opt = new AdamW(model.Parameters(), Flt("lr", 6e-5f), weightDecay: Flt("wd", 0.01f));
var sched = new WarmupCosineLr(Flt("lr", 6e-5f), Int("warmup", 200), Int("schedule-steps", steps), 0.02f);
// With --follow the batch stream may still be written by a trainer running beside this one: step k is complete once the stream's
// steps.jsonl (appended after the step's gradient update) lists it.
static void WaitForBatch(string dir, int step)
{
    string steps = Path.Combine(dir, "steps.jsonl");
    for (int waited = 0; waited < 4 * 3600 * 2; waited++)
    {
        if (File.Exists(steps))
        {
            using var f = new FileStream(steps, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            using var r = new StreamReader(f);
            int n = 0;
            while (r.ReadLine() is not null) n++;
            if (n > step) return;
        }

        Thread.Sleep(500);
    }

    throw new TimeoutException($"batch {step} did not appear in {dir}");
}

string ckpt = Path.Combine(outDir, "state.onemckpt");
int first = 0;
if (File.Exists(ckpt)) first = (int)Checkpoint.Load(ckpt, model, opt).Extra["step"];
int until = Math.Min(steps, Int("until", steps));
using var log = new StreamWriter(Path.Combine(outDir, "ours-steps.jsonl"), append: true);
using var evlog = new StreamWriter(Path.Combine(outDir, "ours-eval.jsonl"), append: true);
if (first == 0 && !o.ContainsKey("skip-eval") && o.ContainsKey("sets")) EvalAll(Need("sets"), "step0", evlog);
var clock = Stopwatch.StartNew();
var times = new List<double>();
for (int step = first; step < until; step++)
{
    long t0 = Stopwatch.GetTimestamp();
    float loss, gnorm;
    double fwdMs;
    {
        using var scope = new ActivationScope();
        if (o.ContainsKey("follow")) WaitForBatch(batches, step);
        Batch b = Batch.Read(Path.Combine(batches, $"batch-{step:D5}.bin"));
        Tensor x = b.ToInput();
        model.Train();
        long f0 = Stopwatch.GetTimestamp();
        Tensor logits = model.Forward(x, b.Widths);
        Tensor lp = Ops.LogSoftmax(logits, -1).Transpose(0, 1);   // [T, B, C]
        var ilen = b.Widths.Select(w => config.OutLength(w)).ToArray();
        Tensor l = Losses.Ctc(lp, b.Targets, ilen, b.TargetLengths, 0, CtcReduction.Mean, zeroInfinity: true);
        loss = l.Data<float>()[0];
        opt.ZeroGrad();
        fwdMs = Stopwatch.GetElapsedTime(f0).TotalMilliseconds;
        l.Backward();
        gnorm = GradClip.ClipNorm(model.Parameters(), 5.0f);
        sched.Apply(opt);
        opt.Step();
    }

    double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
    times.Add(ms);
    log.WriteLine(JsonSerializer.Serialize(new { step, loss, grad_norm = gnorm, lr = sched.LrAt(step), ms, fwd_ms = fwdMs, conv_ms = model.ConvMs, lstm_ms = model.LstmMs }));
    if (step % 50 == 0) { log.Flush(); Console.WriteLine($"step {step} loss {loss:F4} {ms:F0} ms"); }
    int done = step + 1;
    if (o.ContainsKey("sets") && !o.ContainsKey("skip-eval") && Int("eval-every", 0) > 0 && done % Int("eval-every", 0) == 0 && done < steps)
        EvalAll(Need("sets"), $"step{done}", evlog);
}

log.Flush();
if (until < steps)
{
    Checkpoint.Save(ckpt, model, opt, new Dictionary<string, long> { ["step"] = until });
    return 0;
}

times.Sort();
var peak = Process.GetCurrentProcess().PeakWorkingSet64;
File.WriteAllText(Path.Combine(outDir, "ours-summary.json"), JsonSerializer.Serialize(new
{
    steps, seconds = clock.Elapsed.TotalSeconds, median_step_ms = times[times.Count / 2], peak_rss_mb = peak / 1048576.0,
    threads = TensorCompute.Threads, exact = o.ContainsKey("exact"),
}));
WeightsFile.Write(Path.Combine(outDir, "final"), configJson, model.StateDict());
if (o.ContainsKey("sets") && !o.ContainsKey("skip-eval")) EvalAll(Need("sets"), $"step{steps}", evlog);
return 0;
