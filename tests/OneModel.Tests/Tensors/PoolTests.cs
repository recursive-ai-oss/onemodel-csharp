using System.Runtime.InteropServices;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Tensors;

public class PoolTests
{
    [Fact]
    public void ClassSizesAreAlignedBoundedAndMonotonic()
    {
        long prev = 0;
        for (long n = 0; n < 5000; n++)
        {
            long c = BufferPool.ClassSize(n);
            Assert.True(c >= n && c >= 64 && c % 64 == 0, $"{n} -> {c}");
            Assert.True(c >= prev);
            if (n > 256)
            {
                Assert.True(c <= (n * 5 / 4) + 1, $"{n} -> {c}");
            }

            prev = c;
        }

        Assert.True(BufferPool.ClassSize(int.MaxValue) >= int.MaxValue);
        Assert.Throws<TensorShapeException>(() => BufferPool.ClassSize(int.MaxValue + 1L));
        Assert.Throws<TensorShapeException>(() => BufferPool.ClassSize(-1));
    }

    [Fact]
    public void RentedBuffersAreAlignedAndReused()
    {
        using var pool = new BufferPool();
        Tensor a = Tensor.Create(DType.F32, Shape.Create(1000), pool);
        Assert.Equal(0u, (uint)(a.Buffer!.Address % 64));
        TensorBuffer first = a.Buffer;
        a.Dispose();
        Tensor b = Tensor.Create(DType.F32, Shape.Create(1001), pool); // same size class
        Assert.Same(first, b.Buffer);
        Assert.Equal(1, pool.GetStats().NativeAllocations);
        Tensor z = Tensor.Zeros(DType.F32, Shape.Create(1000), pool);
        Assert.All(z.Data<float>().ToArray(), x => Assert.Equal(0f, x));
        b.Dispose();
        z.Dispose();
        Assert.Equal(0, pool.GetStats().OutstandingBytes);
    }

    [Fact]
    public void SteadyStateAllocatesNothing()
    {
        using var pool = new BufferPool();
        Shape s = Shape.Create(8, 64);
        for (int i = 0; i < 10; i++)
        {
            Tensor.Create(DType.F32, s, pool).Dispose();
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++)
        {
            Tensor t = Tensor.Create(DType.F32, s, pool);
            Tensor v = t.Permute(1, 0).Slice(0, 0, 32, 2);
            t.MutableData<float>()[0] = v.Shape[0];
            t.Dispose();
        }

        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    [Fact]
    public void EagerOpsAllocateNoManagedMemoryAfterWarmup()
    {
        using var pool = new BufferPool();
        using Tensor a = Tensor.Zeros(DType.F32, Shape.Create(16, 32), pool);
        using Tensor b = Tensor.Zeros(DType.F32, Shape.Create(1, 32), pool);
        using Tensor w = Tensor.Zeros(DType.F32, Shape.Create(32, 8), pool);
        for (int i = 0; i < 5; i++)
        {
            Run();
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 200; i++)
        {
            Run();
        }

        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);

        void Run()
        {
            Tensor s = Ops.Add(a, b, pool);
            Tensor r = Ops.Relu(s, pool);
            Tensor m = Ops.MatMul(r, w, pool);
            s.Dispose();
            r.Dispose();
            m.Dispose();
        }
    }

    [Fact]
    public void ArenaResetsReuseMemoryAndInvalidateTensors()
    {
        using var pool = new BufferPool();
        using var arena = new Arena(pool, 4096);
        Tensor stale = default;
        for (int round = 0; round < 3; round++)
        {
            arena.Reset();
            Tensor x = Tensor.Create(DType.F32, Shape.Create(100), arena);
            Tensor y = Tensor.Create(DType.I64, Shape.Create(33), arena, zero: true);
            Tensor big = Tensor.Create(DType.F32, Shape.Create(5000), arena);
            Assert.Equal(0u, (uint)(x.Buffer!.Address % 64));
            Assert.Equal(0u, (uint)(y.Buffer!.Address % 64));
            Assert.Equal(0L, y.Data<long>()[32]);
            x.MutableData<float>().Fill(1f);
            y.MutableData<long>().Fill(2);
            big.MutableData<float>().Fill(3f);
            Assert.All(x.Data<float>().ToArray(), v => Assert.Equal(1f, v)); // neighbours did not overwrite
            Assert.Equal(3f, big.Data<float>()[4999]);
            if (round == 0)
            {
                stale = x;
            }
        }

        Assert.Throws<StaleTensorException>(() => stale.Data<float>());

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++)
        {
            arena.Reset();
            Tensor.Create(DType.F32, Shape.Create(100), arena);
            Tensor.Create(DType.I64, Shape.Create(33), arena, zero: true);
            Tensor.Create(DType.F32, Shape.Create(5000), arena);
        }

        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    [Fact]
    public void UseAfterReturnIsDetected()
    {
        using var pool = new BufferPool();
        Tensor t = Tensor.Create(DType.F32, Shape.Create(16), pool);
        Tensor view = t.Slice(0, 4, 8);
        Span<float> raw = t.MutableData<float>();
        raw.Fill(1f);
        t.Dispose();
        Assert.Throws<StaleTensorException>(() => view.Data<float>());
        Assert.Throws<StaleTensorException>(() => t.Dispose());
        Tensor again = Tensor.Create(DType.F32, Shape.Create(16), pool);
        Assert.Throws<StaleTensorException>(() => t.Data<float>()); // the block is back in use, the old handle stays dead
#if CHECKED
        // Returned memory is overwritten with 0xFF (a NaN): a span kept across the return reads poison, not stale data.
        Assert.True(float.IsNaN(raw[0]));
        Assert.True(float.IsNaN(raw[15]));
#endif
        again.Dispose();
    }

#if CHECKED
    [Fact]
    public void WritingPastTheEndIsCaughtAtRelease()
    {
        using var pool = new BufferPool();
        Tensor t = Tensor.Create(DType.F32, Shape.Create(10), pool);
        Span<float> ok = t.MutableData<float>();
        Span<byte> past = MemoryMarshal.CreateSpan(ref MemoryMarshal.GetReference(MemoryMarshal.AsBytes(ok)), 44);
        past[41] = 0;
        Assert.Throws<InvalidOperationException>(() => t.Dispose());
    }

    [Fact]
    public void FreshAndRecycledMemoryIsPoisonedUntilWritten()
    {
        using var pool = new BufferPool();
        Tensor t = Tensor.Create(DType.F32, Shape.Create(10), pool);
        Assert.All(t.Data<float>().ToArray(), v => Assert.True(float.IsNaN(v)));
        t.Dispose();
    }
#endif

    [Fact]
    public void ConcurrentRentAndReturnBalances()
    {
        using var pool = new BufferPool();
        Parallel.For(0, 8, w =>
        {
            var rng = new Random(w);
            for (int i = 0; i < 2000; i++)
            {
                Tensor t = Tensor.Create(DType.U8, Shape.Create(rng.Next(1, 5000)), pool);
                t.MutableData<byte>().Fill((byte)w);
                Assert.Equal(w, t.Data<byte>()[0]);
                t.Dispose();
            }
        });
        Assert.Equal(0, pool.GetStats().OutstandingBytes);
        Assert.True(pool.GetStats().NativeAllocations <= 8 * 40);
        pool.Trim();
        Assert.Equal(0, pool.GetStats().IdleBytes);
    }
}
