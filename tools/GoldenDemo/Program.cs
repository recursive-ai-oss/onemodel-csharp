// Demo target for tools/golden_hash.py. Prints "<name> <sha256>" lines.
//   exact_dot        canonical order, plain multiply and add on four fixed lanes: equal everywhere.
//   chunked_sum      fixed 256-element chunks combined in chunk order: equal at any thread count.
//   fma_dot          uses fused multiply-add where the CPU has it: differs between ISA rows.
//   threads_sum      chunk size derived from the thread count: differs between thread rows
//                    (a deliberate mistake, to show the runner flagging it).
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;

const int N = 100_003;
var rng = new Random(12345); // fixed seed: this program is a demo, not library code
var a = new float[N];
var b = new float[N];
for (int i = 0; i < N; i++) { a[i] = (float)(rng.NextDouble() - 0.5); b[i] = (float)(rng.NextDouble() - 0.5); }

int threads = int.TryParse(Environment.GetEnvironmentVariable("ONEMODEL_THREADS"), out var t) && t > 0
    ? t : Environment.ProcessorCount;

Emit("exact_dot", ExactDot(a, b));
Emit("chunked_sum", ChunkedSum(a, 256, threads));
Emit("fma_dot", FmaDot(a, b));
Emit("threads_sum", ChunkedSum(a, (N + threads - 1) / threads, threads));

// Four fixed lanes; lane j adds the products at i % 4 == j; lanes combined ((0+1)+(2+3)).
static float ExactDot(float[] x, float[] y)
{
    float l0 = 0, l1 = 0, l2 = 0, l3 = 0;
    int i = 0;
    for (; i + 4 <= x.Length; i += 4)
    {
        l0 += x[i] * y[i]; l1 += x[i + 1] * y[i + 1]; l2 += x[i + 2] * y[i + 2]; l3 += x[i + 3] * y[i + 3];
    }
    for (int k = 0; i < x.Length; i++, k++)
    {
        if (k == 0) l0 += x[i] * y[i]; else if (k == 1) l1 += x[i] * y[i]; else l2 += x[i] * y[i];
    }
    return (l0 + l1) + (l2 + l3);
}

static float ChunkedSum(float[] x, int chunk, int threads)
{
    int chunks = (x.Length + chunk - 1) / chunk;
    var partial = new float[chunks];
    Parallel.For(0, chunks, new ParallelOptions { MaxDegreeOfParallelism = threads }, c =>
    {
        float s = 0;
        int end = Math.Min(x.Length, (c + 1) * chunk);
        for (int i = c * chunk; i < end; i++) s += x[i];
        partial[c] = s;
    });
    float total = 0;
    foreach (var p in partial) total += p; // chunk-index order
    return total;
}

static float FmaDot(float[] x, float[] y)
{
    if (!Fma.IsSupported)
    {
        float s = 0;
        for (int i = 0; i < x.Length; i++) s += x[i] * y[i];
        return s;
    }
    float acc = 0;
    for (int i = 0; i < x.Length; i++) acc = MathF.FusedMultiplyAdd(x[i], y[i], acc);
    return acc;
}

static void Emit(string name, float v)
{
    Span<byte> bytes = stackalloc byte[4];
    BitConverter.TryWriteBytes(bytes, v);
    Console.WriteLine($"{name} {Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant()}");
}
