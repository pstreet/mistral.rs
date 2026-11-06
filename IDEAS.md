# Feature Ideas: candle + mistral.rs

Forward-looking feature backlog. Pairs well with TODO.md (active work/constraints) and COMPLETED.md (historical evidence). Ideas are not started - capture here for reference once the backlog clears.

Fact-checked against both repos 2026-11-06: every path and premise below was verified to exist as described.

Legend: [ ] = not started. Effort: [S] small, [M] medium, [L] large. Impact: [H] high, [M] medium, [L] low.

---

## I. Serving / API

### 1. Engine save/checkpoint and load API [ ] [H] Effort: [M]

No way to save a loaded engine's KV cache state for later resumption. Chat-level session storage already exists (mistralrs-core/src/engine/agentic_session.rs caches message history with a 30-min TTL), but reloading always replays the full prefill.

Relevant code: mistralrs-core/src/engine/ (Engine::run lifecycle), mistralrs-core/src/kv_cache/, mistralrs-server-core/src/handler_core.rs

Serializing the KV cache, engine step counter, and runtime state to disk, served on /engine/save + /engine/load, would enable:
- Chat persistency: reload a session mid-conversation without re-prefill (agentic_session.rs is the message-level precursor)
- Prefill saving: checkpoint state after the first forward and skip it on shared requests
- Decoupled prefill+decode: run prefill on a fast lane, checkpoint the serialized input slice onto the decode lane

### 2. KV cache offloading to CPU RAM [ ] [H] Effort: [L]

The most obvious gap vs llama.cpp. Candle already carries the multi-backend foundation (candle-core/src/cpu_backend/, cuda_backend/, hip_backend/, metal_backend/, npu_backend/), and this fork has offload precedent: MoE experts already spill to CPU RAM via `expert_device = "cpu"`. KV cache pools stay device-resident, which caps context length at block-pool size. A "spill to CPU RAM" path wired into mistralrs-core/src/kv_cache/ at the pool level would decouple context length from VRAM (on Strix Halo specifically, GTT staging is already measured at ~240 GB/s read, so a host-resident cold tier is viable).

### 3. gRPC streaming interface [ ] [M] Effort: [M]

The server is axum-based with SSE streaming (mistralrs-server-core/src/chat_completion.rs). A gRPC interface with unary and streaming RPCs would appeal to the ML serving ecosystem (KServe, BentoML, etc.). Would wire in cleanly beside the existing OpenAI-compatible routes.

### 4. Shared weight loading across multiple engines [ ] [M] Effort: [M]

Same-model lanes duplicate loading entirely - the prod hip/cuda twins each hold their own copy of the 35B weights. A shared, reference-counted weight cache would be a memory win for multi-model fleets (biggest where lanes share a device or CPU-RAM tier). New module (e.g. mistralrs-core/src/weight_cache/), following the existing workspace-singleton pattern (CUDA_PARAM_CACHE in candle-core cuda_backend, MOE_DECODE_Q8_WORKSPACE in mistralrs-quant). Interacts with the unload/reload path (/v1/models/unload already trims the caching pool, so refcounts must compose with it).

### 5. Capability-aware model routing [ ] [M] Effort: [M]

Instead of the current flat --model / arch auto-detection approach (mistralrs-core/src/pipeline/auto.rs), the engine could parse config.json for model capabilities (tool-calling, vision, native function calling) and a small routing layer on top would pick the right pipeline and advertise capabilities per lane. Natural fit for the multi-lane TOML config: per-model capability tags the router can match requests against.

---

## II. Model architecture / inference

### 6. Mamba-2 selective scan + RWKV inference [ ] [H] Effort: [M]

The linear-attention hybrid class is already first-class here: qwen3_next's gated delta net (GDN) layers have dedicated kernels (gdn_decode_recurrence, prefill dispatch env vars, in-kernel clamps). What is still missing is the canonical recurrence family: Mamba-2's selective scan and RWKV. candle-examples has rwkv, mamba, mamba2, and mamba-minimal references; the main work is model-level pipeline code plus a GDN-style kernel pass for the recurrence.

### 7. KV cache compaction under pressure [ ] [M] Effort: [L]

The codebase has both prefix caches (mistralrs-core/src/prefix_cacher.rs) and PagedAttention (mistralrs-core/src/paged_attention/, block_pool.rs). The missing link is a compaction/eviction pass inside the block pool: when the pool runs out, merge or drop low-attention trailing blocks (attention-weight-guided, H2O-style) instead of failing allocation or evicting whole sequences. Payoff is sub-linear or constant memory per sequence rather than strictly linear, i.e. deeper contexts in the same block pool. Care point: must compose with prefix-cache hit semantics (block identity is load-bearing there).

### 8. Entropy-guided speculative decoding [ ] [M] Effort: [M]

MTP acceptance already varies sharply with content and depth (65-86% on real text, n=3 worse than n=2 on prose per the Oct 27 sweep) - the draft cost is fixed even when the model is confidently predictable or unpredictably branching. Let the sampler's own entropy signal modulate speculation depth per step (speculate deeper on low-entropy spans, skip on high-entropy ones). Wiring point: mistralrs-core/src/sampler.rs + mistralrs-core/src/speculative/.

---

## III. Candle

### 9. ONNX export [ ] [H] Effort: [L]

Candle has candle-onnx/ for *loading* ONNX models into Candle. The opposite direction - exporting Candle models to ONNX - would enable TensorRT / ONNX-Runtime / PyTorch-serving deployment. Honest premise: candle has no forward-op recording today (candle-nn/func.rs is closures-as-modules, not a tracker), so this needs a small op-capture layer first (a Tensor that records producers into a graph handle). That capture layer is reusable for item 10.

### 10. Program tracer / IR emitter [ ] [M] Effort: [M]

Existing tracing is event-level: tracing-chrome in candle-examples and ad-hoc env-gated probes in core (MRS_TRACE_BIG_HTOD). A tracer that emits the actual op graph (a debug string, llvm::print-style) from the capture layer in item 9 would let you verify fusion and launch structure directly instead of inferring from rocprof kernel names. Improves the dev loop for every kernel-side item in the perf backlog.

### 11. Eager-mode allocation/launch overlap [ ] [M] Effort: [M]

Real pending items from the perf review backlog (TODO.md, candle-core CUDA/HIP section): to_dtype does a duplicated alloc+launch per dtype arm (candle-core/src/cuda_backend/mod.rs) and re-matches a 15-arm DType in hot transfer paths; mistralrs-quant/src/gguf/cuda.rs alloc_zeros a full output buffer on every fused call instead of reusing a workspace (the MOE_DECODE_Q8_WORKSPACE pattern already proves the fix). All are memory-bound-path wins on the ROCm decode loop.

### 12. Host-device transfer motion [ ] [M] Effort: [M]

Pooled events + same-stream elision already landed in both cudarc forks (Oct 24, candle 487ff1df). Remaining pending: cudarc's record-on-drop still records per device_ptr even for same-stream ops (core.rs), and the backlog's overlap idea - streaming GTT-resident weight reads concurrently with KV reads - is the structural next step for the DDR5-bound decode regime on Strix Halo.

---

## Top 3 picks for implementation

If I had to prioritize:

1. **KV cache offloading** - huge practical impact for anyone running large contexts on consumer hardware; the expert-offload precedent and GTT staging path de-risk it.

2. **Mamba-2 + RWKV** - adds the recurrence families the GDN hybrid work doesn't cover. Candle has the reference implementations; this is a model pipeline + one kernel pass.

3. **gRPC streaming interface** - fills a niche no one else covers in the mistral.rs serving ecosystem, leverages existing SSE code as a reference implementation.
