using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using RecursiveAI.OneModel;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

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

return 0;

Stream Open(string name) =>
    asm.GetManifestResourceStream(name) ?? throw new FileNotFoundException("embedded resource " + name);
