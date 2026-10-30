# Data loading (training)

Normative description of `RecursiveAI.OneModel.Training.Data`. A port must reproduce the batch stream from this text alone.

## 1. The contract

A run is a sequence of **global steps** g = 0, 1, 2, … With `S = sampler.StepsPerEpoch`, step g is epoch `g div S`, step-in-epoch `g mod S`. The batch at g is a **pure function of (seed, g)**, the dataset, the sampler's constructor arguments, the transform and the collator. It does not depend on:

- the number of worker threads (0 = build on the caller's thread), the prefetch depth, or timing;
- which other steps were built before, or in what order;
- whether the run was interrupted and resumed: a loader started at step k yields exactly the batches an uninterrupted run yields from k on.

Consumers receive batches in step order. Workers build a window of `Prefetch` steps ahead in any order; a batch waits in its ring slot (`step mod Prefetch`) until the consumer reaches it. A step is claimed only if `step < next-to-take + Prefetch`, so slots never collide. An exception while building step g is delivered when the consumer reaches g, and stops the stream.

Threads are created in one file (`Data/DataLoader.cs`, `BatchSequence`). This is I/O-style prefetch, not compute partitioning: nothing computed depends on which thread ran it.

Where the platform cannot start threads (browser WebAssembly, WASI: `OperatingSystem.IsBrowser()` or `IsWasi()`), the loader runs as 0 workers whatever `Workers` says, with no error; the batches are the same by the contract above.

Datasets are read from several threads at once (`IDataset<T>.GetSample` must be thread-safe) and must return the same sample for the same index.

## 2. Random numbers

`SampleRng` is SplitMix64: `state += 0x9E3779B97F4A7C15; z = state; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9; z = (z ^ (z >> 27)) * 0x94D049BB133111EB; return z ^ (z >> 31)` (all arithmetic mod 2^64). `Mix(x)` is the same function applied to `x` as if it were the state before the increment.

- `NextSingle` = `(u >> 40) / 2^24`; `NextDouble` = `(u >> 11) / 2^53`.
- `NextInt(n)`: unbiased multiply-shift. Take `m = u * n` as a 128-bit product; `low = m mod 2^64`; if `low < n`, `t = (2^64 − n) mod n` and redraw while `low < t`; the result is `m >> 64`.
- `NextNormal` = (sum of twelve `NextSingle`, accumulated left to right in float32) − 6.

Seed derivation: `Derive(seed, epoch, step, slot, stream)`: `h = Mix(seed XOR 0xD1B54A32D192ED03)`; then for each of epoch, step, slot, stream in turn `h = Mix(h + value)`. Streams: 0 loader transform, 1 collator augmentation, 2.. free; the samplers use 101 (epoch order), 102 (mixing), 103/104 (bucketing). The slot is the position in the batch.

## 3. Samplers

All are pure functions of `(epoch, step-in-epoch)`.

- **Sequential**: position `p = step·B + i` is index `p`. **Shuffled**: index `perm_e(p)` where `perm_e` is the keyed permutation below with key `Derive(seed, e, 0, 0, 101)`. Every index appears once per epoch. Drop-last (default) gives `S = floor(n/B)` full batches; otherwise `S = ceil(n/B)` with a short last batch.
- **Keyed permutation** of `[0, n)`: `bits = ceil(log2 n)`, `half = ceil(bits/2)`, `mask = 2^half − 1`; start with `x = p`; repeat { `l = x >> half`, `r = x & mask`; for rounds 0..3: `f = Mix(r XOR key XOR round·0x632BE59BD9B4E019) & mask`, `(l, r) = (r, l XOR f)`; `x = (l << half) | r` } while `x >= n`. A Feistel network is bijective on `2^(2·half)` values and cycle walking restricts it to `[0, n)`. For `n <= 1` the result is 0. It needs no table, so there is nothing to build or cache per epoch.
- **Mixing**: slot j of step s draws `rng = Derive(seed, e, s, j, 102)`; `u = NextDouble`; source = the first source whose cumulative normalised share exceeds `u` (skipping zero-share and empty sources; the last source is the fallback); index = `start[source] + NextInt(size[source])`. Draws are independent and with replacement, so the shares hold in expectation per batch, not exactly. The binomial standard deviation of a source with share p over N draws is `sqrt(N·p·(1−p))`; the test requires counts within 4σ over 10,000 draws.
- **Bucketed**: for epoch e, shuffle all indices with `Derive(seed, e, 0, 0, 103)`, keep the first `S·B` (drop-last), cut into pools of `B·poolBatches`, sort each pool ascending by the key `(length << 32) | index` (unique keys, so the order is independent of the sort algorithm), cut into batches of B, then place batch t of the sorted order at position `perm(t)` with key `Derive(seed, e, 1, 0, 104)` over S. The plan is O(n log n) and cached for the current and previous epoch.

## 4. Collation (`ImageBatchCollator`)

Samples are planar float32 `[C, H, W_i]`. A batch shares C and H. `Wp` = max `W_i` rounded up to a multiple of `WidthMultiple`. Outputs:

- `Images` float32 `[B, C, H, Wp]`; each image at the left of its rows, columns `W_i..Wp` filled with `PadValue`;
- `Widths` int32 `[B]` (real widths); `Targets` int32 `[sum]` (every sample's target concatenated in batch order, the CTC layout); `TargetLengths` int32 `[B]`; `Labels` int32 `[B]`; `Indices` int32 `[B]` (dataset indices).

If an augmentation is set, each sample is first copied to a contiguous `[C, H, W_i]` scratch, the delegate edits it in place with `Derive(seed, e, s, slot, 1)`, and the result is copied to the batch. It cannot change the width. Augmentations that change shape belong in the loader's `SampleTransform` (stream 0), which may return a new sample.

## 5. Memory

Batch tensors come from a `BufferPool`; disposing an `ImageBatch` returns them and recycles the batch object. After warm-up (pool size classes and scratch buffers reached) loading the next batch allocates nothing on the calling thread beyond what the dataset and a user transform allocate. The loader keeps one sample array per worker and clears it after each batch.

## 6. Image helpers

`ImageTensors.Gray8ToPlanar` computes `p·(1/(255·std)) − mean/std` for each 8-bit pixel; `Rgb8ToPlanar` does the same per channel from interleaved RGB into planar `[3, H, W]`; both accept a row stride. They are plain float32 `*` and `−`, so results are identical everywhere (no FMA is used by the scalar loops; the JIT does not contract them). Decoding image files is out of scope.

## 7. Measured (indicative; shared host, load about 20, not publishable)

Synthetic line images: 2,048 images of height 32 and width 64–256, batch 128, width multiple 4, one cheap in-place augmentation per image (gain and offset), prefetch 8, x64 JIT Release. Batches per second: 0 workers 1,343 and 1,635; 1 worker 1,506 and 1,520; 4 workers 4,269 and 4,088 (two runs each). Allocation per batch: 0 bytes at 0 workers (`GC.GetAllocatedBytesForCurrentThread`), 0–2 bytes averaged over workers.
