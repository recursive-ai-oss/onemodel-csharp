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

return 0;

Stream Open(string name) =>
    asm.GetManifestResourceStream(name) ?? throw new FileNotFoundException("embedded resource " + name);
