# Benchmark and golden-hash method

## The bar runner (`bench/Bar`)

A row compares two engines on one model file with the same inputs in the same harness: ONNX Runtime (the `Microsoft.ML.OnnxRuntime` package pinned in `docs/oracles.md`) and this library (a placeholder until its session exists).

**What is timed.** One `Run` call with inputs already bound and outputs left in the engine's own buffers. Session creation is timed separately and reported as `load_ms` (no gate). The first call is reported as `first_run_ms`.

**Procedure.** One process per row and engine. Load, 1 cold call, 3 untimed warm-up calls, then `reps` repetitions (default 3) of as many back-to-back calls as fit in about one second (at most 1000). A repetition's value is its mean time per call; the row reports the median of the repetitions, their min and max, and `spread_pct = (max − min) / median`. Timing uses `Stopwatch`; `alloc_bytes_per_iter` is the managed allocation on the timing thread per call (native allocations inside ONNX Runtime are not counted, which is what that column means for that engine).

**Matching settings.** ONNX Runtime runs at `ORT_ENABLE_ALL`, sequential execution, intra-op threads = N, inter-op = 1. The other engine gets the same N. N is 1 or "all", where "all" is the number of logical processors the OS reports (including efficiency cores and SMT siblings). Batch sizes are 32 for the text encoders, 8 for the conv models, and 1 or as the brief says otherwise.

**Output check.** After timing, the engine's last outputs are compared with the reference outputs for the same inputs (`check_scaled_err`: max |a−b| / max |b|). A fast row that gives wrong answers is not a result.

**Peak RSS.** The process of one row is wrapped by `/usr/bin/time -v` (Linux, "Maximum resident set size") or `/usr/bin/time -l` (macOS, bytes); the number goes into `peak_rss_mb`. Each engine is therefore measured in its own process, harness included.

**Machine record.** Every row carries: machine name, CPU model, the ISA features the runtime reports (`sse4.2 avx2 fma avx512f avx512bw avx512dq avx512vbmi avxvnni avx512vnni`, or `neon dotprod`), logical core count, OS, SDK, runtime, engine versions, and the 1-minute load average before the run (`load_before`), at the end of timing (`load_during`) and right after the process exits (`load_after`). The one-minute average lags real load by tens of seconds, so a short row's three readings are close to each other even if something started halfway; the benchmark lock below is the real guard. The lock's label is stored in `lock_label`.

**Pass criteria** (for rows with both engines): time ≤ 1.5× ONNX Runtime's passes, ≤ 1.15× meets the target; peak RSS ≤ 1.5×; load time is reported. A row measured while another heavy job was running on the machine is not published.

**Driver.** `bench/Bar/run-bar.sh` builds the runner, then claims the benchmark lock file atomically from inside the same script (`noclobber`, label `onemodel bar <what> <start> <expected end>`) and removes it on exit; it refuses to start if the lock exists or the 1-minute load exceeds half the cores. It takes `--rows "model:variant:batch ..."` and `--threads "1 all"` and appends to `bar.csv` and `bar.md`. The lock path comes from `ONEMODEL_LOCK` (default `~/WORKER-BENCH.lock`). Model files and reference directories are resolved from `tools/models.json` by `tools/model_path.py`.

**Kernel micro-rows.** `Timing.Measure` in `bench/Bar/Timing.cs` is the same warm-up/repetition/median routine as a reusable function, so kernel spikes report numbers from the same method.

## The golden-hash runner (`tools/golden_hash.py`)

Runs one command under a set of configuration rows and compares the SHA-256 lines it prints (`<name> <64 hex>` per output):

| Row | Setting |
|---|---|
| `jit-default` | the runtime's default ISA selection, all cores |
| `jit-1thread` | `ONEMODEL_THREADS=1` (the program decides what that means) |
| `jit-no-avx512` / `jit-no-avx2` (x64) | `DOTNET_EnableAVX512F=0` / `DOTNET_EnableAVX2=0` |
| `jit-no-dotprod` (arm64) | `DOTNET_EnableArm64Dp=0` |
| `jit-no-hwintrinsic` | `DOTNET_EnableHWIntrinsic=0` (scalar and software fallbacks) |
| `aot-default`, `aot-1thread` | a NativeAOT build of the same program (`--aot-cmd`) |
| anything else | `--row NAME:ENV:COMMAND`, for example a browser-WASM host |

`--save` stores one machine's results as JSON and `--merge` combines several machines into one table with an `equal`/`UNEQUAL` verdict per output; the exit status is 1 if any output differs. `tools/GoldenDemo` is a tiny program with two outputs that must agree everywhere (a four-lane dot product without FMA, a fixed-chunk sum combined in chunk order) and two deliberate failures (a fused multiply-add dot product and a sum whose chunk size depends on the thread count), to show the runner catches both.

NativeAOT note: an AOT build targets the baseline instruction set unless told otherwise, so it takes the non-AVX2 path of the demo (`aot-default` matched `jit-no-avx2`, not `jit-default`). Anything that depends on the ISA path therefore differs between the JIT and AOT rows by construction; exact kernels must not.
