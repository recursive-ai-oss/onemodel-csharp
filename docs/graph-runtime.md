# Graph runtime: operator contract, plans and sessions

This note says how an imported model becomes something that runs: how operators are described, how a plan for one set of input shapes is made, how memory is reused, and what a session promises. The C# types are named for orientation; the rules are the contract.

## 1. Layers

```
Model.Load(path|stream)            happy path: arrays/spans in, float[] out, no names or shapes needed
  └─ Session                       graph layer: named inputs/outputs as tensors or spans, thread-safe Run
       └─ CompiledGraph            operators resolved, weights loaded; independent of input shapes
            └─ ExecutionPlan       per set of input element types and shapes: shapes, constants, schedule, memory
                 └─ RunState       per run: value table, slab, arena
```

Each layer uses only the one below. The happy path never exposes ONNX names, layouts or kernel choices.

## 2. Operator contract

One operator type (or one family pattern) is one class under `Graph/Operators/` plus one `OpDef` line in `OpTable.cs`. The registry is a static table; nothing is found by reflection.

| Piece | When | What it does |
|---|---|---|
| `OpDef(opType, minOpset, maxOpset, factory, domain)` | table | Names the operator and the opset range one class serves. Ranges of one operator must not overlap; a lookup picks the definition whose range contains the node's opset. The default domain is `""` (`ai.onnx` is the same). |
| factory `(NodeInfo) → OpKernel` | session creation, once per node | Parses attributes (`GetInt`, `GetFloat`, `GetString`, `GetInts`, `GetFloats`, `Find`) and builds the kernel. Malformed attributes raise `SessionException(InvalidModel)` naming the node. |
| `Infer(InferContext)` | plan creation, once per node and shape set | Reads input element types and shapes (and values, if constant) and states every output with `SetOutput(i, type, shape)`. |
| `Prepare(PrepareContext)` | plan creation, after `Infer`, for nodes that will run | Optional. Turns constant inputs into a faster form (a transposed weight) with `CreateOwned`; the returned object comes back as `OpRun.State` on every run. |
| `Run(in OpRun)` | every run | Reads `Inputs`, fills `Outputs` (already allocated, contiguous, for the shapes `Infer` stated). Allocates no managed memory. Uses `Scratch(type, shape)` for temporaries. |

A kernel object is shared by every plan and every concurrent run. It holds what its attributes fixed and nothing else.

Flags on a kernel:

- **`SupportsExact`.** False means "not exact". A session in `ExecutionMode.Exact` over a model that uses such an operator fails at creation with `SessionException(ExactModeUnavailable)`, whose message and `Operators` list name every offending operator type. A kernel that supports exact receives `OpRun.Exact` and must then follow `exact-mode.md`: no fused multiply-add, one canonical accumulation order, NaN results rewritten to `0x7FC00000`.
- **`InPlaceInput`.** Output 0 may reuse that input's storage when the input is not needed afterwards and has the same element type and no fewer bytes. The kernel must be correct with output and input being the same memory.
- **`ViewOfInput`.** Output 0 is the input reinterpreted with the shape `Infer` stated (Reshape, Identity, Squeeze, ...). `Run` is never called; the session builds the view without a copy, and the source's storage stays alive as long as the view does.
- **`OutputsDataDependent`.** The output shapes depend on input values (NonZero, TopK with a computed k). `Infer` is not called; `Run` allocates its outputs with `AllocateOutput`, and everything downstream is shaped at run time (section 4).

Special outputs of `Infer`:

- `SetConstantOutput(i, tensor, planOwns)` states that the output is a constant for these shapes (Constant, Shape, Size). The node is not scheduled. All outputs of a node are constant or none are.
- `RequireRunTimeValues()` says the node needs an input *value* that is not known at plan time (Reshape with a computed shape). The node is then shaped again, with real values, in every run.

### Adding an operator

Write `Graph/Operators/<Name>.cs`: a class deriving `OpKernel` (or a family base such as `BinaryArithmeticKernel<TOp>`) with a `public static readonly OpDef Def`; add `<Name>Op.Def` to `OpTable.All`. Patterns to copy: `Binary.cs` (broadcasting element-wise), `Unary.cs`, `MatMul.cs` (matrix products with `Prepare`), `Views.cs` (views), `Constant.cs`.

### Subgraph operators (If, Loop, Scan)

The contract already has what they need; none is implemented yet. A factory receives the `IrNode` (`NodeInfo.Ir`) and can read its graph attributes. It compiles each subgraph once with `CompiledGraph.Create` (outer-scope values it captures become extra inputs), and keeps the compiled graphs as kernel state. `Run` makes or reuses an `ExecutionPlan` for the subgraph's input shapes and runs it with a `RunState` taken from that plan. Because the subgraph's shapes may differ per iteration, the node is declared `OutputsDataDependent` unless its output shapes follow from its inputs. The plan cache and per-plan state pools are what make repeated iterations allocation-free.

## 3. Plans

A plan is made the first time a model runs with a given list of input element types and shapes, and cached (default 16 per session, oldest dropped first). Making one:

1. **Types and shapes.** Nodes are visited in graph order. Input types and shapes are the caller's. Each kernel's `Infer` states its outputs. Shapes are concrete: the plan is for exactly these input shapes. Symbolic dimensions in the model are not propagated through operators; they are metadata (`TensorInfo.Dims`), checked against the caller's shapes (rank and fixed extents) before a plan is made.
2. **Constant evaluation.** A node whose inputs are all constants is run once now and its outputs become constants, unless the result would be larger than both its inputs and 64 KiB. A view of a constant is a constant. This is what turns `Shape → Gather → Concat → Reshape` chains into a literal target shape. It is bit-preserving by construction: the same kernel produces the same bits, only earlier.
3. **Dead nodes** (no path to an output) are dropped when the session is created.
4. **Dynamic nodes.** A node with a run-time-shaped input, with `OutputsDataDependent`, or that required run-time values, is scheduled as dynamic: shaped again in each run from the actual tensors, with outputs taken from the run's arena.
5. **Memory.** Section 5.

A plan is immutable after creation and is shared by any number of concurrent runs. Plans are reference-counted: the cache holds one reference and each run another, so evicting a plan never frees memory a run is using.

## 4. Sessions

- **Creation** (`Session.Create(IrModel, options, ct)`, `Session.Load(path|stream, options, ct)`): resolves every operator and reports *all* missing ones together (`UnsupportedOperator`, with `Operators` listing `Type (opset N)`), checks the mode, copies weights out of the model (the `IrModel` can be disposed afterwards), picks the kernel path once from the hardware checks (`KernelPaths.Best`). The token is checked between operators and while importing.
- **Options.** `Mode` (`Fast` default, `Exact`), `IntraOpThreads` (0 = one per logical core, up to 256; changes speed only), `OptimizationLevel` (only `None`), `MaxCachedPlans`.
- **`Run`** has two forms. The tensor form takes names and tensors for every input and the outputs wanted; empty output tensors are allocated for the caller, others are checked and overwritten. The span form (single-input models) copies the input into the plan's staging area and the output out of the result and returns the output shape. Inputs are never modified or kept. Non-contiguous input tensors are copied first.
- **Errors** are `SessionException` with a `Kind`: `UnsupportedOperator`, `ExactModeUnavailable`, `InvalidModel`, `Unsupported`, `InvalidInput`, `Disposed`, and the node name where there is one. Input validation (unknown name, wrong element type, wrong rank or fixed extent, short output buffer) happens before anything runs.
- **Threads.** `Run` may be called from any number of threads. The plan cache is a copy-on-write array read without locks; building a plan takes a lock. Each run takes a `RunState` from its plan's idle list (at most 8 kept), so concurrent runs share weights and plan but nothing mutable. `Dispose` waits for nothing: runs in progress finish, and the weights are freed when the last one leaves.
- **Determinism.** Fast mode: results depend on inputs, options and kernel path only, never on thread count or on how runs interleave. Exact mode: bit-identical everywhere. Neither depends on whether a plan was just made or reused, or on which pooled state served the run.
- **Allocation.** After a plan exists and the pools are warm, a `Run` on one intra-op thread allocates no managed memory in either form. With more intra-op threads the parallel-for dispatch itself still allocates a small object per parallel kernel call (a known gap in `Threading/ParallelFor.cs`).

## 5. Memory

All intermediate values of a plan that have static shapes live in one **slab** per run state, 64-byte aligned, at offsets the plan chose. The slab also holds one staging area per input (where span inputs are copied). Dynamic values and kernel scratch come from the run state's **arena**, which is reset at the start of every run.

Offsets come from liveness:

1. `lastUse[v]` is the last step that reads `v` (or the step that makes it, if nobody reads it); graph outputs live to the end. A view extends its source's life to its own.
2. Steps are walked in order. A new value takes the smallest free block that fits (best fit), else the slab grows. Outputs are placed *before* the step's dying inputs are freed, so a kernel never writes over what it is still reading, except through an explicit in-place alias.
3. In-place: if the kernel allows it, the first input is static, dies at this step, is the only user of its block, and the block is large enough with the same element type, the output takes over the input's block.
4. A view shares its source's block (reference-counted). Free blocks are coalesced.

A plan whose slab would exceed 2 GiB is refused (`Unsupported`).

Checked builds poison each block as soon as its last value dies, so a stale read shows up as garbage in the tests; freed arena memory is poisoned too.

## 6. Bits

- Plans, scheduling, buffer reuse, constant evaluation and views are bit-preserving: they change where and when a result is computed, never how.
- Element-wise add and multiply are single correctly rounded operations, so fast and exact give the same bits for the same inputs except for NaN payloads, which exact mode rewrites to `0x7FC00000`. Relu is a compare and select; `-0` becomes `+0`; a NaN passes through.
- Gemm and MatMul follow `kernels-gemm.md`. Gemm's `alpha` and `beta` epilogue is separate multiplies and an add, never fused, in both modes.
- Exact-mode status of the operators so far: Add, Mul, Relu, MatMul, Gemm, Reshape, Identity, Constant are exact for float32; the integer and double paths of Add, Mul, Relu and MatMul are exact by construction (wrap-around integer arithmetic; doubles use the same single operations).
