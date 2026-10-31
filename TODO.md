# TODO: candle + mistral.rs forks (active work)

Closed work lives in COMPLETED.md (condensed, with commits). Full evidence
survives in git history (`git log -p TODO.md`). Audit snapshot 2026-09-25:
candle 56 ahead / 26 behind huggingface/candle:main, mistral.rs 111 ahead /
17 behind EricLBuehler/mistral.rs:master - re-check before relying.
Legend: [ ] open, [x] done. P0 = do first, P1 = do next, P2 = merge prep,
P3 = polish, Deferred = do not do on RDNA.

## Current status (2026-10-31, prod on Strix Halo gfx1151, dual cuda,rocm,flash-attn)

Benchmark (Oct 31): mistralrs (MTP n=2, Q8_0 KV, CUDA graphs) vs llama.cpp (Q8_0 KV, no MTP):
- 854 tokens: mistralrs 54.4 tok/s vs llama.cpp ~50 tok/s (mistralrs wins, +9%)
- 1814 tokens: mistralrs 32.1 tok/s vs llama.cpp 49.9 tok/s (llama.cpp wins, -36%)
- 3094 tokens: mistralrs 29.4 tok/s vs llama.cpp 48.1 tok/s (llama.cpp wins, -39%)
- MTP n=3 tested and found WORSE than n=2 (low draft acceptance rate)
- Q4_0 KV cache tested and found SLOWER than Q8_0 on ROCm (dequant overhead)
- Block size 64: not feasible without kernel changes (dispatcher only supports 8/16/32)
- Contiguous KV cache: rejected as overkill for multi-user workloads
- Fused Q8_0 dequant+matmul: already implemented in MMVQ kernel
- Remaining gap is memory-bandwidth-bound kernel optimization (not config-tunable)

## Current status (2026-10-26, prod on the f3b2d3576 tip, dual cuda,rocm,flash-attn)
13 lanes configured, ALL lazy (headless boot, demand-load, per-category
defaults: chat -> qwen3.6-35b-a3b): 35B A3B hip + cuda twins, 27B Q6_K
+ MTP n=2, 16B, 4B, gemma-4-26b, Llama-3.2-30b, Muse-30b, flash-next
alias (inert until qwen4exp lands), embedding GGUF Q8_0, reranker GGUF
Q8_0 (openwebui flipped; TEI-shape compat rides the next binary swap).
GPU_MAX_HW_QUEUES=1 stays on all three ROCm tenants until the NaN bug
closes. Branch synced to origin/pstreet-rocm.

## Roofline profiling (2026-10-30, gfx1151, Qwen3.6-35B-A3B, Q8_0 KV)

All major kernels are **memory-bound**, not compute-bound. Arithmetic
intensity (FLOPs/byte) is 0.12-0.50 vs roofline transition at ~6.4 F/B.
GPU compute units are idle waiting for data.

### GPU peak (measured with rocprof-compute --roof-only)
| Metric | Value |
|--------|-------|
| FP32 VALU | 17.6 TFLOPS |
| BF16 VALU | 112.7 TFLOPS |
| MALL BW | 1221 GB/s |
| L2 BW | 2540 GB/s |
| GTT read (measured, 4GiB stream, 2026-10-31) | 241.3 GB/s |
| GTT write (measured) | 218.3 GB/s |
| GTT copy r+w (measured) | 208.6 GB/s |

### Correct roofline comparator (2026-10-31)
MALL (1210 GB/s) and L2 (2540 GB/s) are NOT the right ceilings for the
decode kernels: 22GB of weights live in GTT and never fit the on-package
cache, so streaming weight reads are DDR5-bound. The binding decode roofline
is **GTT read: 241 GB/s** (94% of the 256 GB/s LPDDR5X theoretical). Kernel
GB/s figures below must be judged against 241 GB/s, not MALL/L2.

### Kernel breakdown (300-token decode, graphs ON, 45.2 tok/s)
| Kernel | GFLOP/s | % FP32 peak | GB/s | % GTT roofline | Regime |
|--------|---------|-------------|------|----------------|--------|
| mmvq_q8_0_plain (17.4k calls) | 60.6 | 0.3% | 242 | 100% | at GTT roofline |
| moe_gemv_gate_up (5.2k) | 124.5 | 0.7% | 125 | 52% | mem-bound |
| moe_gemv_down_aggr (4.9k) | 91.5 | 0.5% | 183 | 76% | mem-bound |
| attention_paged (1.5k) | 168.6 | 1.0% | 337 | >100% (L2/MALL hits) | KV-cache-resident |
| gdn_recurrence (12.9k) | 13.3 | 0.08% | 13 | 5% | launch/latency-bound |

### Full bench profile (2026-10-31, rocprof-compute launch mode, MTP n=2,
### prompt 1024 / depth 854 / gen 64, workloads/bench_roc2)
| Kernel (real name) | calls | tot ms | share | GB/s | % of 241 GTT read |
|--------|------|--------|-------|------|-------------------|
| mmvq_gguf_q8_0_bf16_plain_cuda3 (MTP verify, 3 tok) | 15681 | 963 | 28.1% | 195 | 81% |
| mmvq_gguf_q8_0_bf16_plain_cuda1 (batch 1) | 1619 | 589 | 17.2% | 212 | 88% |
| moe_gemv_fused_gate_up_q4k_q8_1 | 4779 | 583 | 17.0% | 191 | 79% |
| moe_gemv_down_aggregate_q5k_q8_1 | 4545 | 409 | 11.9% | 163 | 68% |
| vllm::paged_attention_v2_kernel (bf16) | 1356 | 359 | 10.5% | 273 | >100% (KV in L2/MALL) |
| gdn_decode_recurrence_kernel (bf16) | 12299 | 176 | 5.1% | 229 | 95% |
| mmvq_gguf_q8_0_bf16_fused_qkv_cuda3 | 1157 | 131 | 3.8% | 166 | 69% |
| mmvq_gguf_q8_0_bf16_fused_glu_cuda3 | 4457 | 88 | 2.6% | 175 | 73% |
| mmvq_gguf_quantize_q8_1_bf16 (act quant) | 23651 | 39 | 1.1% | 21 | n/a |
| other (18 more variants incl. qkv/glu cuda1-2, v1/reduce attn) | ~1100 | 97 | 2.8% | | |

Total captured kernel time 3434 ms / 72125 dispatches. Kernel names in the
older table above were shorthand; the real symbol names are these.

### Corrected implications (2026-10-31)
- Decode kernels already run at 68-95% of the GTT read roofline. The
  earlier "10-15% of peak" reading compared against MALL (1210 GB/s),
  which weights never touch. True remaining headroom if every kernel hit
  241 GB/s: ~14% of captured kernel time.
- Biggest single-kernel gaps: moe_gemv_down_aggregate_q5k (163 GB/s, 68%)
  and mmvq_gguf_q8_0_bf16_plain_cuda3 (195 GB/s, 81%, but 28% of all time
  so ~5% wall savings if fixed).
- paged_attention_v2 sustains 273 GB/s at d854 (KV blocks L2/MALL-resident).
  The large-context decode drop is NOT kernel-side: see depth sweep below.- gdn kernels are tiny (14.3 us avg, 12.3k launches): amortized inside
  graphs, but any future per-step host work compounds there.
- The 14% ceiling assumes DDR5-bound reads stay DDR5-bound. Kernels that
  can exploit L2/MALL residency (paged attention proves it is possible)
  can exceed the GTT roofline; that is the only path to >241 GB/s for
  weight streaming, and active-expert sets (~1.7 GB/token) are far too
  big for the on-package cache.

## Depth sweep: d854 vs d3094 (2026-10-31, same bench, MTP n=2)

### Kernel GB/s by depth
| Kernel | d854 | d3094 | verdict |
|--------|------|-------|---------|
| paged_attention_v2 | 273 | 442 | FASTER at depth (refutes cache-spill) |
| mmvq_q8_0 plain (dominant batch) | 195 | 243-248 | at/above GTT roofline at depth |
| moe_gemv_fused_gate_up_q4k | 191 | 182 | flat |
| moe_gemv_down_aggregate_q5k | 163 | 172 | flat |
| gdn_decode_recurrence | 229 | 210 | flat |

### Wall clock and runtime behavior
- bench d3094 (clean, no profiler): **43.1 tok/s / 23.2 ms TPOT**
- server d3208 (prod binary, same env): **31.5 tok/s / 31.7 ms TPOT**
- server d4067: 30.2 tok/s; earlier server d3094: 29.4-29.8 -> server path
  is ~8.5 ms/token (37%) slower than bench path at matched depth.
- Profiled d3094 decode kernel structure: batch-1 dominant (26.4k
  mmvq_plain_cuda1 vs 1.1k cuda3 calls) = one full forward per token,
  MTP verify NOT batching. At d854 the structure is batch-3 dominant
  (15.7k cuda3) = MTP active.
- `CUDA decode graph capture failed ... kind: SpeculativeDecode
  (batch 1, seq_len 3)` occurs at d3094 in bench (2x clean, 4x profiled),
  shape then retires and the speculative path runs EAGER. Zero
  occurrences at d854. Prod showed 0 failures at d4067 but still decodes
  at the batch-1-equivalent rate, so MTP is not paying at depth either way.

### Revised large-context diagnosis
The 54->29 tok/s drop with depth is NOT memory-access degradation:
1. Kernels are at/above the GTT roofline at depth; attention is faster
   per byte at d3094 than d854 (longer dispatches hide latency better).
2. MTP stops contributing at depth: draft acceptance collapse and/or
   SpeculativeDecode graph capture failure (bench) -> pure batch-1 decode
   at ~23 ms/forward.
3. The server adds ~8.5 ms/token over the bench at matched depth:
   per-step scheduler/host overhead or a slower captured-graph path.
   This is the single biggest open lever at deep context.
4. llama.cpp flat 48-50 tok/s at these depths beats both paths; closing
   items 2+3 is the route there, not kernel tuning.

### Next actions (priority order)
1. Instrument MTP acceptance rate vs depth (log accept length per step).
2. Root-cause SpeculativeDecode graph capture failure at depth: capture
   workspace sizing vs KV-pool memory pressure (90% VRAM), graph pool
   cap ("3 resident captures kept").
3. Profile the server decode loop host time at depth (LAYER_PROFILE env
   seen in prod logs, or perf) to find the 8.5 ms/token vs bench.

### Implications
- 39% graph speedup (32.6 → 45.2 tok/s) came from eliminating host
  overhead, not from better compute utilization. GPU was already
  memory-bound; graphs just removed the non-GPU bottleneck.
- Remaining TODO items (mutex narrowing, clone elimination, etc.)
  will not move the needle on memory-bound workloads.
- Next-highest-ROI targets: fused Q8_0 dequant+matmul kernels, MoE
  expert packing, KV cache layout optimization.
- Low-impact items below are deprioritized until memory-bound work
  is addressed.

## Recently closed (detail: COMPLETED.md)
Sizing saga (ISQ estimator, fit-ctx everywhere, runtime reserve, mapping
clamp + exact pricing, attach reservation, floors + physical fallback, zero
manual context knobs); unload retention + ContextSize estimator (both GPUs);
twin parameterization + hip gates; on-device eviction; kernel test-env
failures (19 gated, q4k fixed, 2 FA3 gated); arena estimator tests; 27B/4B
live verifications; docs sync.

Oct 23 session: fleet config simplification + user tuning (a3d7ab3e9:
hip:0 default, MoE 35B on cuda:0 with expert offload, MTP on n=2 with
per-model pins 27B->3/4B->6, no ISQ, no cpu device, no default_model_id,
no cuda/cpu aliases, qwen3-coder-next retired, models.ini deleted); /v1/models
no longer synthesizes a 'default' entry (99a5775bf, b544bdc4c); fp8
roundtrip test fixed and gated on rocm (a52bfcae5, 303/0); candle book
rocm README + cudarc-hip FFI marking + Cargo.toml indentation + dead
metal todo cleanup (a4651d9e, b6016fbd, d8024a20, a6075cafd). Prod
swapped to new build + config (Oct 23 05:25 UTC), all commits pushed.

Oct 23-24 session (3090 throughput lane): expert-split cold path
forward_indexed + no-op removal + stacked in-place hot-cache slot
storage + router-predicted prefetch admission + policy worker off the
serving thread; nsys profile of the split vs no-cache root-caused the
68% serving overhead to cudarc per-op event churn; blanket
disable_event_tracking tried and REVERTED (host-read sync is
load-bearing); pooled-event + same-stream elision landed in both cudarc
forks (candle 487ff1df; no-cache lane 31.7 -> ~33 tok/s); hot-cache
verdict: net-negative and structurally capped on this hardware (stays
disabled; revisit = GPU-side partitioning AND 48GB-class coverage).
Prod swapped to the pooled-event build Oct 24 00:06 UTC, hip lane
smoke-tested via 4B demand-load. Full evidence chain in the
expert-split section below and COMPLETED.md.

Oct 24-26 sessions: device-keyed FlashInfer gate + FA3/MLA/sinks
twins on the NVIDIA role in dual builds (build-shape gates retired,
paged-attention docs now state per-device fast-path selection); UQFF
vs GGUF shootout closed (container-neutral at equal quant; UQFF wins
the ISQ workflow; q4k > q8_0 ~30% decode on the 4B GDN hybrid);
split-GGUF auto-expansion landed (name any one -NNNNN-of-NNNNN.gguf
shard; loader gate for qwen4exp PASSED); DFlash splitkv kernel lane:
plumbing shipped inert, kernel hardware-blocked on gfx1151 (precise
record + 4 reopen options in the dflash entry); reranker lane
live-validated on prod and flipped in openwebui; TEI-shape compat for
/v1/rerank landed (texts alias + return_text echo; prod picks it up
at the next binary swap); docs audit closed: expert_device/
expert_hot_cache_mb documented, stale lazy + default_model_id rows
corrected, README reranking mention. Details: COMPLETED.md.

Oct 27 session: 27B perf re-baseline + 27B-specific MTP n re-sweep on
prod (release, Q6_K hip, K=Q8_0/V=Q4_0, graphs+single-queue). Re-baseline:
prefill 396 (1K tok) / 648 (2.7K) / 441 (15.5K) T/s; arithmetic decode
19.4-20.5 T/s (vs 14.1-14.3 Sep 30, +40%); essay 600tok 12.6-13.0 T/s
(13.89 Sep 24 A/B best was an n=2 arm; heat-sag noted); N=2/N=4 aggregate
14.4/31.2 T/s; bandwidth math ~134GB/s effective on the arithmetic arm =
~80% of realistic APU peak, hip lane is bandwidth-bound with no broken
gates. MTP n sweep (essay x3, same methodology, restart per arm, n=3
reproduced 12.88/12.62 after restore): n=3 avg 12.79 vs n=2 avg 13.08
(+2.3% essay) but arithmetic 19.95 vs 18.22 (-8.7% at n=2, len caps 3.0);
per-pos accept 76/52/35% explains it - the third draft pays only at high
acceptance (structured/codey content), costs on prose. VERDICT: keep n=3
(model card); 27B re-sweep closed, no further MTP-pin churn expected.
dev-tools/decode_bench.py added (recreates the lost /tmp/opencode/
mtp_bench.py for decode A/Bs: TTFT/ITL/server-tok, essay|arithmetic
prompts). No cuda-27B twin: user directive, ISQ-Q4K 3090 lane stays
unpursued (27.7 T/s historical, ~4K ctx ceiling, displaces the 35B lane).

## Perf review backlog (opened 2026-10-28, candle + mistral.rs code review)

Scope: four hot-path scans (candle CPU/quantized, candle CUDA/HIP, mistral.rs quant,
mistral.rs core). ~84 candidates. Items are ideas, none started. Priority tags:
[H] high expected impact, [M] medium, [L] low. Validate against the NaN-rate
harness before weakening any sync. HIP reuses candle's shared body via include,
so candle CUDA fixes apply to the HIP role automatically.

### Start here (first wave)
- [x] [H] cuda_graph.rs:3050-3083 - paged-attn V2 workspace mutex held across whole
      graph replay + device sync before/after launch. Narrow lock; use stream/event
      sync where ordering allows. Validate NaN rate.
      (8309fb55f: moved device sync outside mutex, replaced device sync with stream sync)
- [x] [H] normal.rs:2738 - clones whole Vec<KVCache> per forward step. Pass borrowed
      slice/Arc instead.
      (5479aefe7: changed CudaDecodeGraphForwardInput to &[(Tensor, Tensor)]
      eliminating redundant clone; both normal.rs and multimodal.rs)
- [x] [H] qwen3_next.rs:2114 (also 2832,2932) - gdn_replay_stash Mutex per layer/token.
      Thread-localize or move behind non-locking context.
      (1a2e1bd81: replaced Mutex<Option<GdnReplayStash>> with
      UnsafeCell + unsafe impl Sync; all 7 access sites converted)
- [x] [H] candle device.rs:587-631 - kernel function handles not cached (only modules);
      load_function + RwLock per op. Cache CudaFunction by (module,name).
      (candle 5c50f222: added ModuleCache per module with
      RwLock<HashMap<String, CudaFunction>>; double-checked locking
      for lazy population; load_function now only called on first use)
- [x] [H] candle mod.rs:448-499 - graph H2D cache hit does full D2D try_clone() copy.
      Share refcounted non-owning handle.
      (candle 57dd6934: store Arc<CudaSlice<T>> in cache, added share() to
      cudarc-hip and cudarc-nvidia; on hit constructs non-owning CudaSlice
      sharing the same allocation, graph_owned=true; eliminates D2D copy)
- [x] [H] gguf/cuda.rs:958-970 (+cuda_hip) - fresh CudaSlice Q8_1 buffer per quantize.
      Per-device workspace cache.
      (620b91154: quantize_input_q8_1 now uses MOE_DECODE_Q8_WORKSPACE
      cache instead of fresh alloc; returns raw ptr + guard; eliminates
      thousands of GPU allocations per forward for large models)
- [x] [H] k_quants.rs:2606-2639 - matmul_f16 fully serial, per-row Vec<f32> alloc.
      Parallelize over output rows with rayon par_chunks_mut.
      (candle e9326fc3: rayon par_chunks_mut parallelizes matmul over
      independent output rows; lhs conversion remains serial)
- [x] [H] gemv/mod.rs:102-131 - GEMV_DEVICE_INFO Mutex -> RwLock (read hot path).
      RwLock/read-cache or cache on device handle.

### candle-core: CPU / quantized backend
- [x] [H] cpu_backend/mod.rs:3050-3068 - BF16 matmul widens both operands via collect + narrows serially.
      (2d3924c0: use in-place convert_to/from_f32_slice; avoids iterator overhead)
- [ ] [H] neon.rs (11 sites: 538,686,859,1135,1818,1882,2603,2667,2908,2971,3113) - thread_local RefCell<Vec<u64>> scratch borrow_mut + lookup per matmul. (ARM-only)
- [ ] [H] k_quants.rs:2423-2436 - LHS_SCRATCH same thread_local pattern.
- [ ] [H] neon.rs:516-522,1797,2581,2887 - features::get().i8mm checked inside hot vec_dot dispatch. (ARM-only)
- [ ] [M] cpu_backend/mod.rs:1037,1265,1334 - conv inp_cont alloc + triple-nested copy; no contiguous fast path.
- [ ] [M] cpu_backend/mod.rs:137 - pred.iter().zip(...).map(...).collect() without capacity.
- [ ] [M] k_quants.rs:2462-2488 - outer row loop with per-row execute_chunked over quads; barrier overhead for small n.
- [ ] [M] k_quants.rs:217-227,391-402,494-508 - per-block f16->f32 scalar dequant; no SIMD.
- [ ] [M] cpu_backend/mod.rs:726-769 - index-select per-id slice copy in nested loops; not parallelized.
- [ ] [M] k_quants.rs:1460-1474 - Q4K vec_dot_4 only NEON; x86/avx + simd128 fall back to two vec_dot_2.
- [ ] [L] k_quants.rs:630-646,800-816 - Q8_0/Q8_1 to_float scalar loops.
- [ ] [L] neon.rs:1806-1860 - gemv pool.execute_chunked with small n_groups; tune serial threshold.
- [ ] [L] cpu_backend/mod.rs:558 - upsample identity returns src.to_vec() instead of borrow.

### candle-core: CUDA / HIP backend (hip reuses shared body via include)
- [x] [H] mod.rs:83-134 - CUDA_PARAM_CACHE only consulted under graph capture;
      normal path re-uploads params + settle sync.
      (cc158813: cache now always consulted, params moved as key, RwLock
      replaces Mutex, eliminates redundant H2D upload on every call)
- [x] [H] mod.rs:101,116,129 - params.clone() keys + global Mutex<HashMap> per unique layout.
      (cc158813: same commit — move params as key, RwLock replaces Mutex)
- [x] [H] mod.rs:484-488 - slice_ptr calls device_ptr twice per input.
      (cc158813: single device_ptr + arithmetic offset, same sync semantics)
- [x] [H] device.rs:587-631 - kernel function handles not cached (only modules);
      load_function + RwLock per op.
      (849c51ef: global RwLock<ModuleStore> replaced with
      HashMap<idx, OnceLock<ModuleCache>> — cache hits lock-free,
      eliminates module store contention on every kernel launch)
- [ ] [M] mod.rs:1710-1727 - DType re-matched (15-arm) in hot transfer paths.
- [x] [M] mod.rs:176-179,1808 - format! kernel names per call in to_dtype.
      (9643f9f7: cast_kernel_name() pre-computed lookup replaces format!,
      eliminates per-call string allocations)
- [ ] [M] mod.rs:1773-1912 - duplicated alloc+launch per dtype in to_dtype.
- [ ] [M] mod.rs:123,131 - params_from_vec unconditional sync on miss/untracked.
- [ ] [M] mod.rs:28-30,102-103 - global Mutex for param cache; shard by device.
- [ ] [M] device.rs:691-693,766,816 - curand mutex + seed RwLock per rand call.
- [ ] [M] cudarc core.rs:1805-1836 - record-on-drop records per device_ptr even same stream.
- [ ] [L] mod.rs:2634-2818 + device.rs:918-1093 - duplicated dtype branches in copy_strided/conversions.
- [ ] [L] mod.rs:1404-1436 - env-gated debug D2H; ensure fast path.
- [ ] [L] device.rs:448-499 - graph H2D cache RefCell borrow churn.
- [ ] [L] cudarc core.rs:933-936,1017-1020 - alloc_bytes atomic fetch_add; use Relaxed.

### mistral.rs: mistralrs-quant
- [ ] [H] gguf/cpu.rs:144-191 - qtensor_indexed_moe_forward always clones x/ids on shape normalize; try_grouped_gather_mm clones x again.
- [ ] [H] gguf/cpu.rs:61-88 - grouped_gather_mm copies every expert id list + per-expert index_add alloc.
- [x] [H] gguf/cpu.rs:61 - host ids conversion on every indexed MoE call; skip if already U32/contiguous.
      (9e321f06a attempted but reverted: raw pointer access not viable without data_ptr; copy overhead negligible for tiny ids)
- [ ] [H] gguf/cuda.rs:958-970 (+cuda_hip) - fresh CudaSlice Q8_1 buffer per quantize.
- [ ] [H] gguf/cuda.rs:309-333 - alloc_zeros full output each fused call.
- [x] [H] lora/dynamic/expert_cuda.rs:471-476 - normalize_topk avoids contiguous() when already contiguous.
- [ ] [H] gemv/mod.rs:102-131 - GEMV_DEVICE_INFO Mutex locked on every should_use_gemv.
- [x] [M] distributed/layers.rs:981-986,1050-1058,~1075 - contiguous() only when !is_contiguous().
- [ ] [M] afq/ops.rs:430-452 - metal afq_mm clones lhs/rhs indices even if U32; broadcasts into new tensors.
- [ ] [M] f8q8/mod.rs:298-320 - per-row Rust loop with narrow+force_contiguous.
- [ ] [M] fp8/mod.rs:102-120 - per-row dequant temporaries on metal.
- [ ] [M] gguf/mod.rs:590-607 - bias gather not fused; index_select+reshape+broadcast.
- [ ] [M] gguf/weight_source.rs:613 - copies full tensor bytes to_vec during load.
- [ ] [M] gguf/archive.rs:649 - per-tensor shard storage lock on load path.
- [x] [L] utils/ops.rs:66-70 - leftshift reuses preallocated buffer.
- [x] [L] pending_layer.rs:46-76 - RwLock + borrowed resolve (no per-call Arc clone).
- [ ] [L] isq_executor.rs:206-246 - separate workers/state locks in submit path.
- [x] [L] gguf/cpu.rs:178-190 - reshape skipped when input already 3D.
- [ ] [L] utils/ops.rs:3702-3819 - fused_glu allocates output+scales each call.
- [x] [L] rotary/mod.rs:1893-1894 - host copies in rotary per call.
- [ ] [L] gguf/cpu.rs:196 - dequantizes all experts every fallback call.

### mistral.rs: mistralrs-core
- [x] [H] pipeline/cuda_graph.rs:3050 - paged-attn V2 workspace mutex across whole replay.
      (8309fb55f: first-wave done)
- [x] [H] pipeline/cuda_graph.rs:3059,3083 - device sync before/after graph launch.
      (8309fb55f: first-wave done)
- [x] [H] pipeline/normal.rs:2738 - clones whole Vec<KVCache> per forward step.
      (5479aefe7: first-wave done)
- [x] [H] models/qwen3_next.rs:2114 (also 2832,2932) - gdn_replay_stash Mutex per layer/token.
      (1a2e1bd81: first-wave done)
- [ ] [H] paged_attention/block_pool.rs:912 - prefix block retention mutex per allocation.
- [ ] [M] kv_cache/hybrid_cache.rs:2244-2245 - set_state_indices host Vec + slot remap per batch.
      (production path uses set_state_indices_with_host with pre-provided host indices;
      the D2H fallback via to_vec1 is per-batch, not per-token, so low impact)
- [ ] [M] gdn/cache.rs:93-94 - checkout clones conv/recurrent state per GDN layer.
- [x] [M] mla/forward.rs:459,476 - slot/block tables moved to CPU + to_vec2 per decode
      (3817f9c5b+06abd8e1f: slot_mappings_cpu/block_tables_cpu fields added;
      MLA forward reads CPU tensors directly, eliminating H2D copies)
- [ ] [M] paged_attention/scheduler.rs:876 - per-sequence Arc<Mutex<Sequence>> lock churn.
- [ ] [M] kv_cache/hybrid_cache.rs:1414 - logical_slots.to_vec() per batch mapping.
- [ ] [M] kv_cache/hybrid_cache.rs:2437,2493 - host_indices cloned into new Tensor per install.
- [ ] [M] cuda/speculative_rejection.rs:3006 - host staging clones Vecs + rebuilds Tensors per case.
- [ ] [M] sampler.rs:1143,1171,1187 - sampler top1/topk cache Mutex per token.
- [ ] [M] paged_attention/cache_engine.rs:328 - blocking MutexGuard over whole KV cache vector.
- [ ] [M] prefix_cacher.rs:388,401 - cache insertion clones full KV cache + tokens.
- [ ] [L] paged_attention/windowed_pool.rs:1000-1008 - pad_row clone per added row.
- [ ] [L] speculative/dflash.rs:642-649 - repeated reshape/contiguous + codebook contiguous.
- [ ] [L] sequence.rs:1063-1088 - RwLock read per state query in hot loops.
- [ ] [L] paged_attention/scheduler.rs:337 - candidates.clone() VecDeque of Arcs.
- [ ] [L] models/qwen3_next.rs:3063,3407,3409 - draft_lm_head Mutex in speculative paths.
- [ ] [L] kv_cache/hybrid_cache.rs:1026-1040 - RecurrentStatePool full clone.
- [ ] [L] pipeline/cuda_graph.rs:281 - sync in capture/timing path.

### Notes
- Several [H] items are perf-correctness coupled (graph replay sync, paged-attn workspace lock).
  Validate against the NaN-rate harness before weakening syncs.
- Keep changes small and measurable; profile with dev-tools/decode_bench.py and
  prefill_bench.py/conc_bench.py before/after.
- Roofline analysis (2026-10-30): all major kernels are memory-bound (AI 0.12-0.50 F/B vs
  roofline transition ~6.4). GPU compute utilization is <1%. Remaining TODO items targeting
  host-side latency (mutexes, clones) will not improve decode throughput; focus shifts to
  kernel fusion, MoE expert packing, and KV cache layout.

# ==== OPERATING RULES (2026-10-14, user directive) ====
# NEVER modify other tools or applications to bypass an issue in this
# stack. The openwebui webui.db capability patch (builtin_tools=false)
# used as the NaN workaround was wrong and was fully reverted the same
# night - fixes belong in mistral.rs/candle or in OUR OWN deploy config
# (mistralrs.toml / wrappers), nowhere else. tcpdump captures are
# read-only observation and fine; touching a third-party app's database
# or settings to dodge a bug is not.
# =====================================================

## Open P0: all-NaN engine bug

- [ ] ~S/P1 Prefix-cache checkpoint poisoning on NaN states (found
      2026-10-14 via an openwebui outage; PRE-EXISTING - the
      02:22:03 occurrence predates every candle/mistralrs change
      of the hunt week). MECHANISM (evidence chain): the model
      occasionally produces all-NaN logits on a long cold prefill
      - a rare numerical event (observed ~1-in-a-dozen on a 25KB
      chat; 10/10 clean hammer after), likely bf16 GDN state
      blowup on long/repetitive contexts. When it strikes
      mid-conversation, the prefix cache CAPTURES the poisoned
      recurrent state as a checkpoint, and every later request
      sharing that prefix RESTORES it: instant all-NaN logits,
      the conversation fails permanently (user saw 2/2 identical
      failures; retries cannot heal because failed sequences
      don't commit clean blocks either). Other chats and raw
      completions keep working. Two observed error faces: the
      sampling-probability NaN check ("n=248320 nan=248320"),
      including a 311ms instant failure (pure restore, no
      prefill), and the fused-categorical "invalid batched CUDA
      categorical output". Mitigation delivered: restart clears
      the cache (user's conversation works again; cold-prefill
      the first send is slow). CONTAINMENT FIX LANDED 2026-10-15:
      on a seq-level NaN-class sampling error, sample_and_add_toks
      now evicts that sequence's prefix family before marking Error
      (evict_seq_prefix: strict token prefixes; paged: prefix-or-equal
      block keys, lease drop revokes KV retention; clean exact/longer
      entries kept so retries restore clean state). No restart needed;
      one failed message instead of a bricked conversation. Lib green
      (1401/0 default incl 2 new tests), check clean all four shapes.
      REMAINING: the numerics origin (bf16 GDN long-context
      accumulation) - possible follow-up: fp32 state or clamped
      state norm; needs a live NaN repro to validate end-to-end
      (unproven until the var-off probe lane reproduces one).

- [ ] ~P2/BACKLOG all-NaN logits on multi-chunk prefill WITH tools array
      (openwebui outage root-caused to the trigger, 2026-10-14; root
      cause PARKED 2026-10-16 - revisit only with a promising change:
      AMD fix in a future stable changelog, a reproducible fault with
      addresses, or approval for disruptive concurrency tests).
      LIVE MITIGATIONS (prod healthy on these): GPU_MAX_HW_QUEUES=1
      on all ROCm tenants (var-gated corruption never manifests
      single-queue) + prefix-family invalidation on NaN-class sample
      errors (build 12+, one failed message instead of bricked chats)
      + openwebui builtin_tools=false workaround + dmesg early
      warning (init_user_pages failures / restore_worker hogging).
      Evidence: full hunt journal in git history (2026-10-15/16:
      first-fault localization, 10+ trials, vmcore forensics,
      stream/allocator/Lt audits, CUDA comparison, CWSR/SDMA
      exclusions). Prime theory: wild-address reads with ASLR content
      lottery (one hip-700 gap fault), crashes = kernel HMM/userptr
      UAF under fresh-load churn (11 crashes, TTM teardown family).
      Standing rules: NO var-off loads (3 crashes/21h); probe lane
      var-ON only. TRIGGER SHAPE (evidence
      chain): openwebui 0.12.0 injects its 38 built-in tool schemas
      (~26KB tools JSON) into EVERY chat request unless the model's
      meta.capabilities.builtin_tools=false; such requests render to
      ~6.8k-token prompts (4 prefill chunks at 2048). In prod's
      load-into-empty-GTT regime those requests produce all-NaN
      first-sample logits 13/13 (user's 5 openwebui failures +
      verbatim replays). Requests without tools: 100% clean all
      night (raw 120KB completions, essays, sanity, bare chats).
      Bisect history (midnight-02:00): tool ENTRY COUNT and payload
      SIZE both matter (38 entries fail ~100%, <=32 pass, 33-42
      noisy) but identical bytes flip outcomes across cold runs -
      per-request randomness (uuid token-count shifts block
      alignment). A/B'd on the probe lane (pre-crash machine):
      v_cache_type Q4_0 + empty-GTT load + 6.8k-token tools
      payload => NaN 5/5; V=Q8_0 => clean 3/3 same payload. THE
      REVERT: all five split-KV entries (35B q5/q4 arms + all three
      27Bs) moved back to V=Q8_0 in models/mistralrs.toml (the
      27Bs were masked by loading on demand alongside the resident
      35B). CAVEAT: post-reboot (post-crash) prod on Q8_0 still
      NaNs on OTHER multi-chunk tools payloads (med38 29KB,
      big38 70KB synthetic) - so Q4_0 was A/B-proven to be A
      trigger for the user-shaped payload, not THE bug; the class
      persists. WORKAROUND SHIPPED: webui.db model rows patched
      capabilities.builtin_tools=false (all 11 chat models) +
      open-webui restarted - openwebui now sends clean requests;
      chats work. Re-enable per-model when the engine bug closes.
      NEXT: the deterministic repro recipe (fresh boot/empty GTT
      + tools-array prompt >2048 tokens) on the probe lane; bisect
      engine knobs (mtp off / prefix_cache_n=0 / graphs) and re-add
      the stripped per-layer NaN probes (11993a14f) to find the
      layer where NaN first appears.
      ROOT-CAUSE HUNT 2026-10-15/16 (27B+MTP probe, :1245, fast dual):
      reproduced 2/2 var-off with med38. Mechanism-A (no-op hook)
      DEAD for 35B/27B: both serve via MultimodalPipeline, mtp_prefill
      runs per chunk with complete coverage. First-fault localized to
      TARGET prefill (drafter downstream): trial1 chunk2 layer-4 GDN
      rec-y inf; trial2 chunk1 layer-52 input-proj mixed_qkv inf from
      finite inputs; trial3 (blocking) chunk1 layer-43 attn-y inf.
      KILLED: MTP-KV holes (coverage complete), async queue race
      (HIP_LAUNCH_BLOCKING=1 still NaNs), gradual bf16 overflow
      (blowup sudden at modest magnitudes), fused-gating sign bug
      (kernel matches CPU ref exactly, g<=0 by construction).
      STANDING: nondeterministic across runs (3 trials, 3 layers,
      2 kernel families) with bounded math + finite probed inputs
      => kernels read memory the probes don't cover (wrong-row/OOB
      reads into ASLR-varying garbage) or beta!=0 GEMM over fresh
      buffers. Activation growth 37->252 across chunk-1 layers is
      real but benign (clean layers sustain it). NEXT: var-ON rate
      series (safe regime, script scratch/nan-hunt + /tmp/opencode/
      varon-series.sh) to test whether queue count is causal; then
      legacy-chunked kernel cell and 3090/CUDA comparison.
      VAR-ON SERIES RESULT 2026-10-16 (~00:35 UTC): 5/5 CLEAN var-ON
      (fresh loads, med38, MTP n=2, chunk 2048, zero anomalies) vs
      3/3 NaN var-off (layers 4/52/43, GDN-proj/rec-y/attn-y). With
      the earlier manual var-ON clean, 0/6 vs 3/3 (p~0.012): queue
      count is causal, not lottery. HIP_LAUNCH_BLOCKING=1 still NaN'd,
      so it is NOT an async kernel-timing race (blocking serializes
      launches but NOT library/SDMA streams) - prime suspect is now a
      missing cross-stream dependency (BLAS-lib stream or H2D vs
      compute) or layout-dependent addressing, both masked by the
      single-queue FIFO. D1 blocking trial also rules out in-process
      stream races as previously audited. NEXT: audit GDN-prefill
      producer->consumer stream edges (hipBLASLt handle stream,
      f32-upcast/slot-H2D visibility) for the missing event.
      STREAM AUDIT 2026-10-16: single per-device stream everywhere
      (BLAS Lt, custom kernels, H2D all share it) - NO cross-stream
      edge exists in-process; missing-dependency theory DEAD.
      ILLEGAL-ACCESS TRIAL: var-off probe faulted with hip error 700
      on a CK-GEMM (grid 372736) at fault va in NO live ledger range
      (18MB unmapped gap) - a wild GEMM address, hardware-confirmed.
      32F-accum cell still NaN'd (F16-overflow DEAD), legacy-chunked
      cell still NaN'd (vmajor-specific DEAD). Score: var-off 4/4 NaN
      (layers 1/4/43/52, GDN-proj/rec-y/attn-y + one hard fault) vs
      var-ON 6/6 clean. NEXT: per-tensor device addresses in probes
      (dtype-dispatched helper, built) + ONE var-off trial -> victim
      ranges vs ledger overlap analysis to place the fix (allocator
      overlap vs wild dims vs corrupt weights).
      OVERLAP RESULT 2026-10-16: 45k ledger events, ZERO double-allocs,
      victim range inside its own live buffer - pool allocator SOUND,
      no use-after-free in tracked ranges. Per-tensor addresses now
      log (dtype-dispatched helper). Running Lt-shape-identified
      trial (CANDLE_LT_DEBUG + addrs + ledger): last LTMUL before a
      fault names the faulting (m,n,k).
      CRASH-11 2026-10-16 05:15 UTC (9 min after clean boot, during
      no-trim var-off probe LOAD - no request ever fired): SAME
      unpopulate+0x4b GPF, fault addr 0xfe00fe00fe00fe18 = FILL
      PATTERN (freed-memory poison, not model data) => kernel
      USE-AFTER-FREE in TTM teardown (entry freed+poisoned, then
      dereferenced). No-trim trial crashed anyway => pool trims NOT
      necessary for crashes. 3 crashes in 21h, ALL around fresh
      var-off loads; var-ON loads (prod x4+) never crash. FIRM RULE:
      NO MORE VAR-OFF LOADS, period. Mechanism: fresh-load GEM/HMM
      churn (mmap'd GGUF userptr registrations + managed-weight
      migration faults) racing across multi-queue faults; corruption
      manifests later at teardown. NaN side stays userspace-side per
      the no-distress journals (or sub-threshold kernel staleness -
      untestable now).
      CWSR LEAD 2026-10-16 (~09:30 UTC): community precedent found -
      Framework/Strix-Halo docs list amdgpu.cwsr_enable=0 as the fix
      for GPU hangs/crashes; CWSR (compute wave save/restore on
      preemption) corrupting wavefront state on restore fits EVERYTHING
      (silent wrong-values AND hangs, run variance via preemption
      timing, multi-queue-gated, device-specific, tenant-aggravated).
      Current: cwsr_enable=1, runtime read-only (0444) => needs GRUB
      cmdline + reboot to test. PROPOSAL: add amdgpu.cwsr_enable=0,
      reboot, verify, then var-off trial (moratorium lifted for the
      test - preemption is disabled). Clean => ROOT CAUSE CONFIRMED.
      Determinism check (prod 35B var-ON 3x full outputs): identical
      md5s - safe regime fully deterministic.
      REBOOT 2026-10-16 ~04:36 UTC: clean boot on 7.2.9 with
      amdgpu.cwsr_enable=0 ACTIVE (verified sysfs). Tenants
      auto-started. DECISIVE TEST NEXT: var-off probe trial with
      preemption disabled - clean => CWSR corruption CONFIRMED as
      root cause; NaN => back to wild-address hunt (CWSR exonerated
      as the NaN mechanism, crashes may still implicate it).
      CWSR EXONERATED (NaN) 2026-10-16 ~05:00 UTC: var-off + CWSR-off
      trial STILL NaN'd (layer-4, pure-NaN). Preemption-restore
      corruption is NOT the NaN mechanism. SDMA THEORY NEXT: var may
      gate copy-engine concurrency (H2D/D2H vs compute cross-engine
      waits); testing var-off + HSA_ENABLE_SDMA=0 (shader-copy
      fallback, fully ordered). KFD MQDs confirmed every process
      carries SDMA + compute queues even var-ON.
      SDMA RESULT: var-off + HSA_ENABLE_SDMA=0 STILL NaN'd (layer-2,
      inf). Copy-engine concurrency excluded (caveat: HSA var governs
      HSA-API copies; HIP memcpy path may bypass it - weakens but
      does not kill the exclusion). CWSR REVERTED 2026-10-16 ~05:05
      UTC (reboot to stock 7.2.9, cwsr=1 verified): unproven cmdline
      change removed to keep one variable stable; all baselines are
      cwsr=1.
      CRASH-10 2026-10-16 03:47 UTC: box rebooted during noaslr var-off
      probe series (vmcore /var/crash/202610160347, 26GB). SAME signature
      as crash-8/9: GPF non-canonical address in
      amdgpu_ttm_tt_unpopulate+0x4b. 10th crash, all TTM teardown
      family. Wild-write theory now has: 10 kernel GPFs + 1 userspace
      hip-700 (gap address) + NaN lottery, all consistent with device
      wild-address writes stomping neighbors (activations AND TTM page
      arrays). NO MORE VAR-OFF RUNS without explicit approval - each
      fresh var-off load risks another crash-loop. Noaslr series died
      with it (2/2 NAN before the crash).
      UNINIT-BLOCK MECHANISM 2026-10-16 ~03:30 UTC: the ENTIRE paged KV
      cache is Tensor::empty (uninit_block, cache_engine.rs:57) - fresh
      blocks hold virgin GPU garbage, and the codebase KNOWS it
      (MISTRALRS_POISON_NAN fills fresh blocks with NaN so
      read-before-write shows up instead of flaky garbage). Null
      block 0 is never written. Any unmasked read of unwritten blocks
      (null padding, stale tables, not-yet-written tails) yields
      garbage: inf/NaN patterns cascade, finite patterns pass
      silently. Fits the lottery (garbage content), fresh-vs-grown
      (virgin vs recycled), Q4/Q8 (different garbage patterns), and
      var-ON/OFF (heap layout). DECISIVE TEST RUNNING: poisoned
      tiny38 (var-ON safe) - instant failure PROVES unmasked
      unwritten reads; clean EXONERATES masking (back to wild
      addresses).
      POISON VERDICT 2026-10-16 ~04:00 UTC: poisoned tiny38 AND
      poisoned med38 (ALL fresh KV blocks NaN) BOTH GENERATE
      correctly - every block the model reads is written first or
      masked, single AND multi-chunk. Read-before-write EXCLUDED.
      Remaining: wild-address writes (hip-700 proved), nondeterministic
      compute, or hardware. Testing output determinism next (3x same
      prompt temp=0 on prod 35B): differing outputs PROVE ambient
      wildness even in clean mode and give a cheap every-run oracle.
      GATE CENSUS VERDICT 2026-10-16 ~03:00 UTC: two trials with the
      census active show g<=0 EVERYWHERE yet still NaN - the +187 was
      garbage content landing in g's buffer, not a systematic gating
      bug. Pure-NaN runs (no inf phase) need a 0/0 path (attention
      softmax over empty/unmapped blocks?) or NaN-bit reads.
      SPLIT VERDICT 2026-10-16 ~04:15 UTC (post crash-10 forensics):
      journald shows ZERO kernel distress (no GUP failures, no
      restore-worker hogging) during ALL silent-NaN probe trials -
      distress appeared ONLY in the crash boot shortly before the GPF.
      So: silent-NaN = userspace-side (kernel mappings healthy),
      crashes = kernel HMM/userptr-side under churn (GUP failures +
      spinning restore worker + TTM free-path GPF). Crash-10 dmesg
      also reframes crash-7: page-array entries that look like fp32
      small-floats may be TRUNCATED PFNs (small ints), i.e. a kernel
      32/64-bit handling bug, not model-data stomps. QUEUED (need
      user approval): (a) tenants-down var-off probe (tests fault-
      concurrency interaction), (b) userptr-reduction deploy test
      (NO_MMAP + no-managed, needs fresh loads), (c) 7.2.11 stable
      changelog check (released 2026-10-15) for HMM/TTM race fixes.
      dmesg WATCH: init_user_pages failures / restore_worker hogging
      are now the early-warning pattern for crashes.
      7.2.11 VERDICT 2026-10-16 (~08:00 UTC, stable-log changelog
      v7.2.9..v7.2.11): NO TTM/HMM/userptr/populate fixes. Only
      TTM-adjacent change is an SDMA-DCC corruption workaround for
      gfx12 (RDNA4, not our gfx1151). STAY ON 7.2.9 - no kernel
      rebuild/reboot for 7.2.11.
      CUDA COMPARISON 2026-10-16 (~09:00 UTC): same 38-tools med38
      payload (7.3k tokens, MTP n=2, prefix cache, multi-chunk) on
      prod qwen3.8-27b-cuda (3090, speculative ENABLED): 3/3 CLEAN.
      Same math on NVIDIA is clean => NOT a model-math bug. Bug is
      ROCm-side (hipcc kernels, hipBLASLt/rocBLAS library behavior
      on gfx1151, KFD queues, or allocator) or HIP+code interaction.
      (27B-cuda resident ~20GB on 3090 now; 4B demand-reloads.) NEXT:
      block-table validation probe (unmapped/empty ids at dispatch)
      + repeated hard-fault collection (clustered addrs = systematic
      offset, scattered = content lottery).
      STRUCTURAL IDENTITY 2026-10-16 ~06:00 UTC: var-ON-clean vs
      var-off-NaN logs run IDENTICAL code paths (same fwds, qlens,
      chunks, MTP n=2). Variance must come from memory content
      (ASLR) or device nondeterminism, not branching. ASLR test
      running: setarch -R (no-aslr) var-off trials - same fault
      layer twice PROVES layout-determinism (bug becomes
      reproducible at will); differing layers PROVE true
      nondeterminism (race/thermal/ulp direction).
      HEAP VERDICT: first-alloc bases scatter WIDELY across var-off
      runs (0x7347-0x7c3f range) - heap base is per-run ASLR, NOT
      var-determined. So var does NOT act via heap layout. With
      single-stream FIFO + identical code/inputs, remaining var
      pathway: LIBRARY-INTERNAL concurrency (rocBLAS/hipBLASLt
      worker streams, split-K atomics, workspace hazards) gated by
      per-process queue count - invisible to source audit,
      unaffected by HIP_LAUNCH_BLOCKING, shape-dependent. Testing
      via concurrency, not var: batched (2x parallel med38,
      max_seqs=2) var-ON probe - NaN under batching CONFIRMS
      concurrency-gating in the safe regime.
      BATCHED VAR-ON RESULT: 2x parallel med38 (max_seqs=2) both
      completed clean - concurrency alone does NOT trip it (weak
      negative: overlap unconfirmed). TRIM-RACE THEORY (new prime):
      background pool-trim maintenance (engine-tail/per-poll/failed-
      reload trims from the retention fix) freeing live blocks that
      get recycled while in flight - timing-dependent (trim cadence
      vs inference), heap-dependent (reuse victims), var-modulated
      (throughput shifts interleaving). TESTABLE VIA KNOB:
      var-off + MISTRALRS_NO_POOL_TRIM=1 - clean IMPLICATES trims
      (fixable in our code), NaN EXCLUDES them.
      LT POINTER SCREEN 2026-10-16 ~05:00 UTC: all 2016 Lt GEMM A/C
      footprints fit inside live ledger ranges (one false alarm from
      a swapped-m/n footprint formula, corrected to transpose-invariant
      numel check). Lt GEMM addressing EXONERATED. Wild address must
      come from non-Lt kernels (custom FA/GDN/speculative/sampling,
      rocBLAS non-Lt, or missed paths).
      BREAKTHROUGH 2026-10-16 ~02:45 UTC: cross-checking the faulting
      layer's launch pointers against probed tensor addresses - EVERY
      pointer matches (q/k/v/g/beta/state/out/slot all verified). But
      rec-gb absmax=187 with beta=sigmoid<=1 PROVES +187 lives in the
      decay gate g, which the reference formula (g=-exp(a_log)*
      softplus(a+dt), fused kernel verified identical) says is
      IMPOSSIBLE for finite inputs (g<=0 always). +187 => expf => inf
      => NaN cascade. Either a wild write into g's buffer between
      gating and recurrence, or gating miscompute. Gate-sign census
      probe built (counts g>0 + first indices) to get the pattern
      (scattered=miscompute, clustered=stomp) on the next trial.
      GROWN-PROCESS RE-NaN 2026-10-16: same probe process NaN'd twice
      in a row (req2 after containment eviction => cold miss in grown
      pools still NaNs, layer-1 again, pure partial-NaN no inf). Fresh
      is worse but not required; grown is NOT safe. Layer-1 now leads
      4 of 7 fault loci. Launch-side pointer logging built (slot table
      addr at install vs consumed pointers at launch) to test
      stale-table reads directly.
- [ ] ~P0 MACHINE STABILITY (found 2026-10-14 02:20+): TWO kernel
      crashes within 15 minutes during heavy GPU stress testing
      (prod + repeated probe model loads + NaN-poison batteries).
      Both: general protection fault on a NON-CANONICAL garbage
      pointer in amdgpu_ttm_tt_unpopulate <- amdgpu_gem_object_free
      (kdump vmcores /var/crash/202610140221 and 202610140232).
      No MCE/EDAC RAM errors logged; rasdaemon not running. Could
      be an amdgpu TTM teardown race under heavy GEM churn (both
      crashes during/around process teardown of probe instances)
      OR bad RAM (no evidence either way yet - the NaN "lottery"
      and the GPF are consistent with either). DO NOT hammer this
      box with repeated model-load/unload cycles until understood;
      consider rasdaemon + a memtest86+ run at next convenient
      downtime; if crashes recur outside GPU stress, escalate to
      hardware. Reboot also wiped /tmp/opencode evidence files
      (pcap + exact payloads) - future evidence goes in
      scratch/nan-hunt/ (real fs). UPDATE 03:17-03:24 UTC: full
      crash-LOOP - four boots in ten minutes, each dying ~107-110s
      in with the identical amdgpu_ttm_tt_unpopulate GPF (third
      vmcore /var/crash/202610140322). Stopping mistralrs-server
      broke the loop at once (box sailed past T+190s) - trigger is
      OUR GPU workload, not background activity. Standing rule: one
      model-load per boot cycle while the driver is suspect, no
      load/unload churn loops; if boots start dying ~2min in again,
      stop our server FIRST. (User rules out a RAM fault - treat as
      amdgpu TTM teardown race under GEM churn unless new evidence
      says otherwise.) UPDATE 03:31-03:39 UTC: serving ALONE trips
      it - the user's single openwebui request crashed the box
      (fourth vmcore /var/crash/202610140334, identical
      amdgpu_ttm_tt_unpopulate+0x5b GPF; two reboot-loop boots
      before this one stabilized). Prod is DOWN and staying down:
      stopped + watchdog userstop marker placed so the healthcheck
      stays quiet (note: a bare `systemctl disable` deleted the
      unit symlink itself - restored, unit is `linked`/inactive;
      the marker suppresses watchdog restarts). Do NOT start
      mistralrs-server again until the driver crash is understood -
      every start risks another crash-loop. The NaN hunt is parked:
      repro + sampler analysis journaled, engine fix lands only
      after the box is trustworthy. KERNEL-FIX LEAD 03:45 UTC
      (user-directed audit): full change inventory finds NO
      code/config cause - same binary md5 all week, no uncommitted
      changes, wrappers identical, openwebui reverted+verified,
      kernel constant (7.2.6) for a month. Upstream instead:
      CVE-2026-98166, TTM swapout bulk_move-cursor UAF (dangling
      range endpoint -> freed-memory deref on later free paths),
      fixed in 7.2.8 (released 2026-09-25); our 7.2.6 was built
      2026-09-14 and predates the fix, and third-party source
      checks confirm 7.2.6 still carries the buggy `if (!ret)`.
      Our six crashes (GPF on non-canonical pointer, all on TTM
      free/teardown paths) match the family; memory-pressure
      swapout is the trigger class and the night's dual-35B
      co-resident churn is exactly such pressure. Caveat: our RIP
      (unpopulate+0x5b) is adjacent to, not identical with, the
      CVE's listed signatures - proof is crashes stopping on
      7.2.8, not the match alone. FIX RECIPE: fetch mainline
      7.2.8+ debs (kernel.ubuntu.com/~kernel-ppa/mainline/) into
      /var/local/kernel/, rebuild index per its README, apt
      install, reboot, verify `uname -r`, then ONE prod start.
      7.1.5 is NOT the fallback (bug is v7.1+, likely vulnerable
      too). BUILD STARTED 03:5x UTC: upstream 7.2.9 source
      (cdn.kernel.org) in scratch/kernel-build/ - fix VERIFIED in
      source (`if (ret > 0)` in ttm_bo_swapout_cb), configured from
      /boot/config-7.2.6 (olddefconfig, amdgpu=m/HSA=y posture
      kept), `make -j32 bindeb-pkg` relaunched fully detached
      (setsid session leader, survives the agent session, resumes
      incrementally). RC ruled OUT:
      only v7.3-rc3 prebuilt exists and the fix landed in rc4.
      After build: stage debs to /var/local/kernel/, dpkg-scanpackages,
      apt install, reboot, verify, single prod start. STAGED
      04:38 UTC: linux-image/headers-7.2.9_7.2.9-5 built, in
      /var/local/kernel/, index rebuilt, apt installed (ii),
      DKMS rebuilt clean for 7.2.9 (nvidia 615.71 + xrt-amdxdna),
      grub entries present (7.2.9 first). GRUB_DEFAULT is still
      pinned to 7.2.6 - first boot MUST use one-shot `grub-reboot`
      into the 7.2.9 id so a bad kernel falls back to 7.2.6 by
      itself; pin 7.2.9 only after it proves stable. REBOOT 04:4x
      UTC (user-approved): one-shot grub-reboot into 7.2.9.
      On return: verify `uname -r` = 7.2.9, single prod start +
      sanity, then openwebui traffic is the real trial. If the
      box is back on 7.2.6, the one-shot failed - diagnose before
      retrying. Pre-reboot verified: entry cmdline, initrd,
      grubenv write path, fallback intact. BOOTED 7.2.9 04:42
      UTC, clean reboot, prod loaded first try, sanity CORRECT,
      Q8_0, box past T+6min with zero oops/BUG/GPF (old window
      was ~2min) - the TTM fix holds so far. Real trial is user
      openwebui traffic next. PINNED 04:56 UTC (user-approved):
      GRUB_DEFAULT=gnulinux-7.2.9-advanced-... via /etc/default/grub
      + update-grub (grub.cfg line 40 confirms); future boots land
      on 7.2.9, 7.2.6/7.1.5 kept as fallback entries. User big-tools
      request still NaNs on Q8_0/7.2.9 (engine bug open, kernel
      now survives it). CRASH-7 05:00 UTC: 7.2.9 panicked at T+17min
      (T10677 tokio-runtime-w, GPF non-canonical 0xc0560000be560018
      in amdgpu_ttm_tt_populate+0xe2 via ttm_tt_populate <-
      ttm_bo_populate <- handle_move_mem <- validate <-
      init_reserved <- amdgpu_bo_create <- ... <-
      kfd_ioctl_alloc_memory_of_gpu). Prod IDLE 5+min (last
      completion 04:54:52, 0 running) - no inference in flight.
      8x the old ~2min window, so the swapout fix helped, but the
      family is NOT closed: same GPF shape, new site (populate/alloc
      path, not unpopulate/free). Vmcore /var/crash/202610140500/
      (5.8GB) + dmesg saved. Boot gap 05:02-05:26 (kdump rebooted
      05:02:07, box back 05:26:47 - user power-cycled: post-kdump
      reboot hung with nothing logged, box sat dead until manual
      power-on).
      Prod parked DOWN again; NaN hunt re-parked on stability.
      NEXT: check upstream TTM/amdgpu fixes past 7.2.9 (populate-path
      UAF?), then decide: newer stable build vs mitigation.
      FORENSICS 05:3x UTC (crash-utility built from master - distro
      9.0.1 predates the 7.x struct-module change): crash-7's
      ttm->pages array (32k entries/128MB KFD BO, ZERO_ALLOC) holds
      fp32-pattern garbage in entries [0..28159] (high bytes
      c0/bf/40/3f/be/3e/bd - model-weight distribution) with only
      the tail [28160..32255] valid page pointers. The populate loop
      is the VICTIM (first deref of pages[0]), not the bug: ~220KB
      of the array was stomped by an fp32-data writer. No upstream
      fix exists (ttm_pool.c frozen since the 7.2 merge; 7.2.10 has
      no TTM/amdgpu fix - only an unrelated x86/mm lockdep commit).
      Prime suspect shifts from TTM-logic bug to wild writer: box
      runs TWO ROCm tenants on the same iGPU/TTM pool (mistralrs +
      ComfyUI ROCm service, both active in the crash boot). The
      ISOLATION TEST 05:44 UTC (user-approved): comfyui.service +
      chatterbox.service stopped (both inactive, no leftover procs),
      prod started solo on 7.2.9 - clock starts at load, watching
      MTBF past the 17min crash-7 mark with no other ROCm tenant.
      VERDICT 06:05 UTC: solo T+20min loaded, box T+38min, prod
      active, zero oops/BUG/GPF - past crash-7's mark. Co-tenant
      isolation holds so far; openwebui traffic is the ongoing
      trial. ComfyUI/Chatterbox stay down until MTBF is convincing.
      SOLO REPRO MATRIX 06:21 UTC: tiny38/med38/big38 (4.9/29/70KB
      tools payloads, the class that NaN'd with tenants up) all
      completed 200 solo, zero seq-level errors - big38 took 22s
      prefill (multi-chunk for certain). Second datapoint same
      direction as MTBF, still small-n, no proof claimed. Tension
      noted: if NaN needs tenants up to reproduce, the engine hunt
      later needs them back (with crash risk); if solo keeps
      passing, the trigger lives in shared-state (pool/GTT
      pressure, prefix-cache warmth), not in our engine alone.
      RE-PROBE 06:58 UTC: full matrix clean again solo (tiny/med/
      big38 all length, no errors), box T+1:31, prod active, zero
      oops. No public bug matches our signature (CVE-2024-56594 is
      a 2024 Renoir DMA issue, long fixed) - freedesktop drm/amd
      filing DROPPED per user 2026-10-15 (not worth the upstream
      cycle; the wild-writer forensics stay in the journal if ever
      needed). REUNION 07:03 UTC
      (user-directed A/B): comfyui + chatterbox restarted onto the
      warm box (both active) - watching whether NaN/crash returns
      with tenants back. JOINT-PEG 07:18 UTC: sustained iGPU 100%
      (clocks maxed 2900MHz, 40W) with ZERO requests anywhere and
      quiet logs - follows the tenants jointly: each tenant solo
      settles to 0% (chatterbox 60s, comfy 90s), both together peg
      100% within a minute and hold it; tenants down drops to 0%
      in <25s (twice). Mechanism unidentified. Caveat: solo windows
      were short (60-90s) vs joint peg observed over minutes -
      missing control is one tenant alone for 5+ min.
      COMFY-SOLO 07:24 UTC: flat 0% over 6min - exonerated.
      MATRIX-4 07:28 UTC (chatterbox resident): tiny/med/big38 all
      length, zero oops, box T+2:01 - fourth clean matrix.
      CHATTER-SOLO 07:31 UTC: flat 0% over 6min except one 96%
      blip at 07:28:18 = our own big38 prefill (matrix-4) - also
      exonerated. PEG VERDICT: no runaway anywhere; the joint pegs
      were overlapping tenant model-load churn (cold page cache
      first time, warm contention the second) - alloc storms + GTT
      migration while prod serves. Implication for crashes: tenant
      (re)starts are churn events in the crash-friendly class;
      stagger them, never together, never during heavy serving.
      PAIR-PEG 07:37 UTC: peg needs ComfyUI + co-tenant (3/3):
      comfy solo 6min = 0%, chatter solo 6min = 0%, pair = 100%
      sustained, comfy stop drops to 0% in <25s. No per-process
      engine counters exist on this kernel (fdinfo lacks
      drm-engine, KFD queue debugfs empty - nvtop can't split it
      either). Mechanism still unknown; next probe is radeontop
      block mix (GFX vs SDMA = compute vs migration churn).
      RADEONTOP 07:41 UTC: peg is pure command-processor (cr 100%,
      every shader/mem block at 0.00%, GTT 21GB steady) - signal/
      event spin, NOT compute and NOT migration. Burns power, not
      memory; likely benign for crashes (a spinning CP doesn't
      write fp32). Peg and stomps are separate leads now.
      ONLINE MATCH 07:45 UTC: ollama#17958 (open, Aug 2026) is our
      peg on the same gfx1151: >=2 resident HIP contexts pins
      gpu_busy 100% + max clocks + ~25W at zero load, deterministic
      both directions, solo = clean. Companion ROCm-side filing:
      ROCm/TheRock#7578. Their lead: multiple armed HIP contexts
      defeat KFD/MES power-gating (queue state, not polling) -
      matches our radeontop (pure CP, zero shader work). Known
      Strix Halo behavior, power/heat cost, almost surely benign
      for the crashes. Also: legacy-rocm#6371 (HSA busy-spin
      threads under ComfyUI) matches our hot kfd/futex threads.
      FIX STATUS 07:5x UTC: AMD acknowledged - TheRock#7578 assigned
      (zichguan-amd, triage/in-progress); kernel fix WIP (same as
      legacy-rocm#5107; amd-jherichuang on the driver fix). Workaround
      with data: GPU_MAX_HW_QUEUES=1 on EVERY resident ROCm client
      -> idle clean (0%/600MHz), no LLM throughput change; partial
      coverage does not work. Nuances: pins depend on runtime
      version (7.14 pins, 7.13 doesn't) and CWSR setting; separate
      CPU-spin issue (rocm-systems#12674) mimics it on meters.
      WORKAROUND LIVE 08:00 UTC: GPU_MAX_HW_QUEUES=1 on all three
      (wrapper + both tenant units, staggered restarts) -> busy 0%
      with everything resident. ~25W recovered. Remove all three
      copies when the kernel fix lands.
      QUEUES-OFF TEST 08:26 UTC: var pulled from wrapper, prod
      restarted, matrix run - med38 NaN'd FIRST TRY (all-NaN,
      status 500), tiny38 passed just before. BUT confounded: the
      test also fresh-loaded prod (empty GTT + cold prefix cache =
      the original repro recipe), and earlier var-off matrices on
      warm prod were clean. Var-mask vs fresh-regime undecided.
      CRASH-8 08:27 UTC (~30s later, T+18min boot like crash-7):
      GPF non-canonical 0x7fff7fff7fff8017 in unpopulate+0x4b via
      BO-free/dmabuf-release path, during big38. Vmcore
      /var/crash/202610140827/ (4.2GB). Box kdump'd, back 08:33.
      NEXT: discriminate - (1) prod start now (var still off,
      fresh boot) + med38; (2) var back ON + prod restart (fresh
      load WITH var) + med38. NaN in (2) = regime wins; clean =
      var masks a race.
      REBOOT 08:0x UTC (user-approved): full host reboot to verify
      clean boot on pinned 7.2.9 + all-three queue workaround. On
      VERDICT 08:38 UTC: run1 (var OFF, fresh load) med38 NaN;
      run2 (var ON, fresh load) med38 clean - same regime, only
      the var differs. Single-HW-queue MASKS the NaN => race/
      timing bug in our engine (queue overlap), with a reliable
      on/off switch (off=repro, on=serve). Crash-8 also fits:
      multi-queue exposure during big38. NEXT: bisect the race
      (mtp off / graphs off / prefix_cache 0, var off) - needs a
      lane decision with user (probe second-load churn vs prod
      restarts). Prod left var-ON serving.
      CRASH-9 08:40 UTC (T+7min, shortest 7.2.9): GPF 0x3f0640043f813fd0
      (fp32-pattern again) in unpopulate+0x4b via BO-free/dmabuf path -
      same path as crash-8. Hit ~1min after the bisect prod restart
      (var OFF), during 35B load, no requests logged. Vmcore
      /var/crash/202610140840/. Var-off regime = 2 crashes in ~15min
      (crash-8 during big38, crash-9 during load). BISECT ABORTED:
      var restored ON in wrapper, further var-off steps suspended -
      the discriminator already answered (run1 NaN vs run2 clean).
      BISECT AUTHORIZED 08:4x UTC: user cleared prod as the test
      lane (no real traffic, resolution first). Knobs, var-OFF
      throughout, one at a time with restore: (0) baseline re-confirm,
      (1) MTP off, (2) graphs off, (3) prefix_cache_n=0. tiny+med38
      RACE HUNT 08:55 UTC (user: hunt it, prod cleared as lane but
      var-off prod steps crash - hunting via code + 4B probe instead,
      prod stays var-ON serving). CODE SUSPECTS (subagent, all with
      file:line in session ses_ec662f52): #1 GDN checkout/commit
      scatter vs next-chunk gather across HW queues (no event between,
      copy_rows has no bounds check - indexed_copy.rs:162); #2 paged
      recurrent prefix partial-hit vs chunk plan (tools HIT where
      others MISS); #3 CPU-fallback logit D2H racing forward on the
      default stream; #4 MTP draft H2D vs target GDN commit; #5 stale
      index OOB (guard never fired, blind during capture). 4B PROBE:
      fast rebuild in flight (scratch/fast-rebuild.log), probe-4b.toml
      :1244 + 4b-{tiny,med,big}38 payloads ready - tests whether the
      race generalizes off the 35B at small churn.
      BISECT MATRIX (probe-35b :1246, cold, multi-queue): n=2+
      cache4=NaN (x3: Q4K draft, stock head, graphs-off); n=1=clean
      (graphs on+off); MTP off=clean; cache0+n=2=clean; warm HIT=
      clean; 4B/27B probes clean (no MTP). VERDICT: needs BOTH MTP
      n=2 AND prefix cache - race is in MTP-speculative vs cache-
      checkpoint capture on cold multi-chunk prefill. Graphs
      excluded. Mechanism hunt (subagent ses_ec63b6d7) in flight;
      prod stays down per user (frees memory), probe lane active.
      each step; big38 skipped (crash-8 trigger).
      GROWN-POOL CELL 10:4x UTC: same probe process (pools grown),
      cold-miss med38 = CLEAN. Fresh-process-cold-miss NaN-prone
      (3/3 probe + 2/2 prod fresh) vs grown-process-cold-miss clean
      isolates the transient to PROCESS FRESHNESS (load-time pool
      growth/graph-rez/autotune), not cache state. Single-stream
      FIFO in-lane (audited) rules out in-process stream races -
      remaining actors: cross-process device contention, host-side
      first-touch races (OnceLock/static caches), SDMA-vs-compute.
      POISON 0/2: computed-corruption lean, uninit-read disfavored.
      PIVOT: var-off live runs crash the box ~every other cycle
      (crash-8/9) - hunt continues via code + filing, NOT live
      var-off runs. Prod var-ON serving CORRECT throughout.
      return: uname -r, unit states, staggered starts as needed,
      sanity, idle check. Note: tenants auto-start together (cold
      churn) - watch, don't fight it unless it crashes.
      BOOTED 08:08 UTC clean on 7.2.9, tenants auto-started and

      The
      still-armed MISTRALRS_GATHER_GUARD shows our stack already
      suspects its own OOB writes; guard logs are routine (no fire
      seen). Unification lead (unproven): one wild writer could
      explain BOTH the NaN logits (stomped activations/KV) and the
      TTM stomps. NEXT: isolate tenants (stop ComfyUI, prod alone,
      watch MTBF past 17min) - needs user go-ahead (their service).
      Build notes:
      needed `--jobserver-style=pipe` (something eats /tmp fifos
      mid-build), cleared Ubuntu `debian/canonical-*.pem` cert
      paths from .config (SecureBoot off, inert), kept MODULE_SIG

      PHASE B0 (2026-10-26, var-off + de-risked load, 27B Q6_K):
      de-risk = NO_MMAP + no-managed (removes userptr + managed
      migration churn = the crash class). GPU_MAX_HW_QUEUES unset.
      RESULT: NaN REPRODUCES (all-NaN logits at sampling:
      "invalid CUDA top-1 output: max_logit=NaN argmax=NaN"),
      6/15 = 40% rate (lottery-like, matches hunt). ZERO CRASHES
      across all 15 requests. CONCLUSION: NaN mechanism is in the
      COMPUTE path, not the load path. De-risk disarmed crashes but
      not NaNs -> crash-free NaN lab now available. Discriminator
      cells B1-B4 (trim race, graphs, MTP, per-layer probes) can run
      safely. Note: trial runner's NaN detection only checks the
      HTTP response body; the actual NaN errors surface in the
      server log as "invalid CUDA top-1 output: max_logit=NaN".
      Probe tooling: scratch/nan-probe/ (payload, toml, launch
      scripts, trial runner). Probe-nan branch holds the cherry-pick
      attempt (aborted; diff at /tmp/opencode/nan-probe-commit.patch).
      ROCm 10.2.0 (user building from source) = parallel reopen
      condition (a) test; slots in as a comparison trial when ready.
      B1 (trim-race discriminator, 2026-10-26): var-off +
      de-risked + MISTRALRS_NO_POOL_TRIM=1. RESULT: NaN STILL
      REPRODUCES (3/12 = 25% vs 6/15 = 40% with trims on).
      Trims are NOT the sole mechanism (NaN persists with trims
      off). Rate reduction may be real (trims amplify) or noise
      (small samples). Trims EXCLUDED as root cause per the
      discriminator's own criterion (NaN != clean).

      B2 (graph discriminator, 2026-10-26): var-off + de-risked +
      MISTRALRS_CUDA_GRAPHS=0 (trims ON, same as B0). RESULT: ZERO
      NaNs in 12/12 requests (0% vs 40% with graphs on). Clean
      A/B on graphs with trims held constant. GRAPHS ARE THE
      MECHANISM (or the gate through which the NaN manifests).
      CUDA graph replay is the prime suspect. This explains:
      - var-gating: graph replay interacts with queue scheduling
      - lottery: captured-state sensitivity to ASLR
      - HIP_LAUNCH_BLOCKING no help: graph replay is independent
        of launch serialization
      - per-layer probes: garbage in g's buffer during replay

      DEPLOYED 2026-10-26: MISTRALRS_CUDA_GRAPHS=0 in prod wrapper
      (~/.config/mistralrs-server-wrapper.sh). GPU_MAX_HW_QUEUES
      removed from all launch scripts (user accepts power/heat on
      APU). Prod restarted in idle window (0 requests in 10 min),
      dry-run gate passed, 13 models registered, health OK. Env
      verified in process (MISTRALRS_CUDA_GRAPHS=0 present,
      GPU_MAX_HW_QUEUES absent).

      FIX ATTEMPT (2026-10-26): Added synchronization in cuda_graph.rs
      replay path to close the multi-queue race:
      1. After input_ids.set() - stream sync (H2D copy of input_ids)
      2. prepare_copy() - device-level sync before H2D metadata copy
         (was: skip sync for same-stream)
      3. order_before_graph() - stream sync for same-stream completions
         (was: skip sync for same-stream)
      4. Before graph.launch() - device-level sync (all streams)
      5. After graph.launch() - device-level sync (ensure graph complete)
      RESULT: NaN reduced from 40% to ~25% but NOT eliminated
      (4/16 NaN across multiple test runs). Syncs address the
      H2D->graph-launch race but NaNs persist, suggesting the race is
      deeper - possibly in graph capture's stream dependencies or in
      operations that reference shared buffers during replay.

      CONCLUSION: Graphs cannot be safely re-enabled on multi-queue
      systems with simple sync fixes. The proper fix requires either:
      (a) explicit event dependencies for every buffer in the graph,
      (b) capture-time stream serialization, or (c) accepting that
      graphs only work with GPU_MAX_HW_QUEUES=1.
      FINAL TEST (2026-10-26, graphs ON + all 5 sync fixes): ~10%
      NaN rate (2/20) down from 40% (6/15) without syncs. Partial
      improvement but not elimination. Remaining NaNs likely from
      graph capture's unrecorded stream dependencies or shared buffer
      races that syncs cannot close.

      Graphs remain DISABLED in prod (MISTRALRS_CUDA_GRAPHS=0).
      Sync fixes committed as partial improvement (no harm, reduces
      NaN rate, passes 4-shape cargo check). Graphs stay off until
      a deeper fix (explicit event deps or capture-time serialization)
      lands. B3 (MTP cross-check) not run - graphs are the confirmed
      mechanism; MTP is not implicated.

      Graph ptr census (MISTRALRS_GRAPH_PTR_CENSUS=1) is armed in
      prod and may catch moved VAs if re-enabled.

      SINGLE-QUEUE EXPERIMENT (2026-10-26): graphs ON +
      GPU_MAX_HW_QUEUES=1 (single hardware queue) + all sync fixes +
      workspace mutex lock during replay. RESULT: 0/10 NaN. Definitive:
      the NaN is QUEUE-DEPENDENT. Graphs are safe on single-queue
      scheduling and unsafe on multi-queue scheduling on this ROCm
      runtime (TheRock). Multi-queue exposes a race that syncs cannot
      close (likely the runtime distributing one stream's captured
      graph across multiple hardware queues, or a captured-DAG
      dependency that only manifests under concurrent queue
      execution).

      Also shipped: paged-attn V2 workspace mutex is now acquired
      during graph replay (lock_paged_attn_v2_workspace). This closes
      the race on the shared global workspace for longer-context
      requests (V2 path) where the graph captured the V2 kernel
      without its CPU-side mutex. Harmless on the V1 path (short
      contexts) and passes 4-shape cargo check.

      CONCLUSION: graphs and multi-queue are mutually exclusive on
      this runtime. Two viable prod configurations:
      (A) GPU_MAX_HW_QUEUES=1 + MISTRALRS_CUDA_GRAPHS=1 (graphs on,
          single-queue power/heat profile), or
      (B) GPU_MAX_HW_QUEUES unset + MISTRALRS_CUDA_GRAPHS=0 (graphs
          off, multi-queue power/heat profile).
      PROD NOW RUNS (A) (user choice 2026-10-26 07:42): wrapper sets
      GPU_MAX_HW_QUEUES=1 + MISTRALRS_CUDA_GRAPHS=1 on the new binary
      (sync fixes + V2 workspace replay lock, md5 134443ea). Health
      OK, 13 models, graphs captured at startup, zero NaN events.
      One lane hit a capture failure at 07:47 (forward failed, not
      overflow; graceful eager fallback, no NaN) - watch for recurrence.
      TheRock 10.2 (user building from source) may fix the runtime
      side; re-test graphs on multi-queue after it lands.

## Closed (2026-10-26): 07:47 capture failure - root cause and fix

      PERF AUDIT (2026-10-26): the 35B lane's wall time is decode-bound.
      Turn 1 cold prefill 201s of 255s; turns 2-3 replay the prefix
      (~105K tok/s effective, 2s) and spend 64-139s decoding. Prefix
      cache works; decode is the only lever, and it ran EAGER all day:
      the 07:47 capture failure disabled all 30 graphs lane-wide
      (resident_entries=0, 605 eager dispatches, 1 replay total).

      ROOT CAUSE: lazy capture of a new graph variant mid-serving hit a
      host->device upload >4KB inside the capture window. candle's H2D
      content cache caps at CUDA_GRAPH_HTOD_CACHE_MAX_BYTES=4096, and
      over-cap uploads can NEVER be cached (unbounded content-keyed
      growth), so the warmup forward cannot pre-populate them; the
      in-capture forward bails ("cannot upload uncached host data",
      device.rs). That Err propagated to try_cuda_decode_graph_forward
      -> disable_cuda_decode_graph -> state.disable()+clear(): ALL
      resident entries dropped, none re-armed for the process lifetime.
      The startup precapture path already had per-bucket semantics
      ("failed for one bucket, kept earlier captures"); the runtime
      lazy path was the outlier.

      FIX P1a (mistralrs-core): capture failures are now key-scoped.
      capture_cuda_decode_graph_step returns CudaDecodeGraphCaptureError
      {err, state_restored}; runtime callers keep resident entries and
      fall back to eager for the step when state_restored, with a
      per-key retry budget (2) before the shape retires
      (dispatch reason capture_exhausted). Unrestorable live state
      (rollback/restore failures) still hard-fails and replay failures
      still kill the lane - NaN-safety conservatism is unchanged.
      Mirrored in normal.rs and multimodal.rs twins.

      FIX P1b (candle): bails now carry byte_len; MRS_TRACE_BIG_HTOD=1
      eprintlns every over-cap host upload with a backtrace (eager side)
      and appends backtraces to capture-path bails, so the >4KB
      uploader can be identified from a probe run. Repro asset:
      scratch/nan-probe/payload-long.json (~125K tokens, mirrors the
      prod conversation), launch-probe-htod.sh (prod memory profile:
      managed weights + mmap + host drop, single-queue, graphs on).

      PROBE RESULTS (2026-10-26 23:27, 27B at 12K tokens, new binary
      md5 9dd116343): 31 captures (29 startup + 2 lazy MTP decode
      variants), 0 failures, 24 graph replays, resident_entries=4
      (rest evicted by spec-state budget, 27 evictions - expected).
      Lazy captures at 12K are CLEAN; decode runs on graphs. The only
      over-cap H2D traffic is the hybrid checkpoint RESTORE path
      (HybridCache::restore_recurrent_checkpoint_state ->
      Tensor::to_device: 3MB f32 + 80KB bf16 x ~3 slots per capture
      step, 96 events of each size in one trial). These are legal
      post-capture uploads (outside the capture window) - not the
      07:47 culprit, but they churn the pool on every capture step.

      The 07:47 in-window >4KB uploader did NOT reproduce at 12K on
      the 27B. Full 125K probe repro is infeasible on this box: the
      padded 32-slot capture buckets x full-KV V2 gathers explode
      (kernel OOM killed the probe at 44GB unit peak, 2026-10-26
      23:17 - probe with oom_score_adj=100 was the preferred victim).
      Working hypothesis for the 35B at 133K: managed-weights pool
      trim during lazy-capture arena prep evicts weight pages; the
      in-capture forward faults them back in-window (>4KB page
      upload -> bail). Fits the 36864B-consumed early death and the
      35B lane's big KV-driven trim pressure.

      CLOSURE PATH: prod now runs the new binary (user deployed
      22:25, idle window, Option A env preserved) and the wrapper
      arms MRS_TRACE_BIG_HTOD=1 (TEMP, remove once named). The next
      natural occurrence of the failure class on the user's real
      133K workload will: (a) stay key-scoped (lane keeps its
      graphs, warn names the bucket + byte_len + backtrace),
      (b) name the exact uploader in the journal. Probe lesson:
      run probe lanes as systemd user units (mistralrs-htod-probe);
      harness restarts kill nohup'd AND setsid'd children
      (cgroup cleanup), only systemd units survive.

      Expected impact: graphs back on the serving lane = +7-10% decode
      (A/B 13.89 vs 12.54 tok/s) where decode is 90%+ of wall time.
      Landing needs a rebuild + idle-window prod swap; verify via
      /metrics: resident_entries > 0, replay dispatch climbing,
      zero capture{outcome=failure} growth.

## Open: architecture extension
- [ ] Role-identity Phase 4: Vulkan third shell (untouched; target state is
      one shared backend body with thin role shells - detail: COMPLETED.md).

## P1. Do next — high impact, slightly more work (week)

### Execution order (agreed 2026-10-09; state: prefix n=4 + MTP + RAG
live in prod, both GPUs healthy). Phase A consolidates (each step's
numbers steer the next): 1. graph-capture audit (read-only, with
prefix-hit traffic), 2. KV pool right-sizing (with the 739MB
reservation live), 3. MTP n re-sweep (acceptance-with-hits may have
moved it), 4. prefix polish, 5. batch the one-line prod knobs into a
SINGLE restart. Phase B throughput bets: UQFF/quant benches first
(they can redirect), then expert-split MoE (35B on the 3090), MTP-on-4B
and 35B KV precision as side quests. Phase C NVIDIA role in dependency
order with batched 3090 windows: cuda-alone sanity, 27B parity,
FA2/graph proofs, cuda-MTP fit, cuda-alias demand-load under n=4.
Phase D/E deferred deliberately (split prefill/decode, multi-backend,
GGUF embeddings, merge prep, polish). Operating rules: measure before
cutting (every config change wants a probe number first); one prod
restart per batch, never one per item.
STATUS 2026-10-11: Phase A resolved - (1) audit DONE (major finding:
the 35B never graphed, see its close), (2) right-sizing landed via
the native cap, (3) MTP re-sweep DONE (n=2 kept, fe8e6ea1d), (5)
knobs landed (draft-head Q4K + n=2 fleet-wide, 740d92c5b). (4) prefix
polish remains open. Phase B startable; the 35B-graphs item (P0
parity queue) now outranks it on impact.

### Perf-opportunity backlog (opened 2026-10-08, ideas, none started)

- [ ] ~M Expert-split MoE placement (llama.cpp `--cpu-moe` style):
      attention/router/shared-experts on GPU, cold experts in RAM.
      Unlocks the 35B on the 3090's 24GB. Requires new work: split
      placement in the MoE experts backend + cross-device dispatch per
      token (placement today is whole-layer only, verified: no
      lru/offload/evict machinery in `mistralrs-core/src/moe/`).
      STATUS 2026-10-17: BUILT (commit 18800233e) but BLOCKED on CPU throughput (see below):
      ... (see above for full plan) ...
      BUILT 2026-10-17 (commit 18800233e, ~500 lines, 30 files):
      plumbing landed end to end; fast probe on the 35B (Q5_K_XL,
      cuda:0, expert_device=cpu) loaded at ~2GB VRAM with experts in
      RAM, paged attn stayed on, MTP attached (n_predict=2).
      BLOCKER: CPU expert matmul throughput. ROOT CAUSE FOUND:
      candle's QTensor::indexed_gemv (candle-core/src/quantized/
      mod.rs:1197) is `#[cfg(target_arch = "aarch64")]` only — on
      x86_64 it returns Ok(None) immediately. So qtensor_indexed_moe_
      forward (mistralrs-quant/src/gguf/cpu.rs:144) falls through to
      dequantize ALL 64 experts to F32 per layer per step (33.5GB Q5
      -> 134GB of F32 allocs at Q8 scale; ~800KB -> 3.2MB per
      layer), then try_grouped_gather_mm on the F32 weights. That
      dequant-per-step is the 300x slowdown (measured 0.15 tok/s =
      6.7s/token; 984 gemv calls/step). The aarch64 path (repacked
      per-expert kernels) is the fast one and is not ported to x86.
      Options:
        (a) port indexed_gemv to x86_64 (AVX2/AVX512 repacked
            per-expert gemv in candle's repack module + mod.rs) —
            the real fix; brings CPU path to the bandwidth ceiling
            (~60-100 tok/s at Q5).
        (b) cache dequantized F32 weights (dequantize once at load,
            keep in RAM) + rayon-parallelize the grouped gather across
            experts — avoids per-step dequant; trades 4x RAM (35GB ->
            140GB for experts, exceeds 122GB avail at Q5; would need
            Q4 or a subset). Net ~10-20 tok/s.
        (c) phase-2 hot-expert VRAM cache with router-predicted async
            prefetch (the differentiator; ~0.27GB/expert at Q5, 3090
            holds dozens; prefetch overlaps with attention/GDN).
        (d) accept 35B-on-3090 as a slow/capacity lane only (not
            viable at 0.15 tok/s).
      OPTION (A) LANDED 2026-10-17 (candle commit 87aff026):
        indexed_gemv ported onto the repack_x86 tile machinery —
        try_indexed_gemv decodes at one shared matmul unit per
        (pair, expert n-tile) at m=1, >=32 pairs bucket by expert
        into one tiled matmul per expert tile range (nt0 threaded
        through matmul_tiles); TileQ5K + pack/kernels (512/256-bit)
        added; all four tile packs now barrier-pool parallel (tens of
        GB pack would otherwise stall first touch); indexed_gemv in
        mod.rs drops the aarch64 gate. Tested vs dequant+q8k-lhs
        reference (Q4K/Q5K/Q6K/Q8_0 x decode/shared-lhs/bucket/bf16)
        at native AVX512VNNI + forced AVX2/AVXVNNI: 54/54 green.
        FINDING: the 35B Q5_K_XL experts are ALL Q8_0 (123 tensors,
        ~35GB — unsloth's UD-XL keeps MoE at high precision), so the
        pre-existing Q8_0 tiles (1.0x RAM) serve the 35B; the new Q5K
        tiles cover other quant arms/models. MEASURED (release, prod
        shape, probe :1244): decode 35.0 tok/s (fast build 33-36),
        MTP accept 62-90% len 2.2-2.8, prefill 189-263 tok/s, ~4.6GB
        VRAM, ~60GB process RSS (35GB tiles + resident mmap). That is
        220x over the broken path but 55% of the hip lane (63.6 with
        flash) and under the 48-51 match-or-beat bar. Effective expert
        read ~84GB/s/step vs ~200 DDR5 peak — remaining headroom is
        unit granularity / barrier overhead (768 tiny units per
        tensor per layer at decode), not kernel math.
      GUARD FIX 2026-10-21: the x86 try_indexed_gemv wrapper
      quantized the lhs BEFORE its select() guard, so k%256!=0
      (expert-bias widths) hit the Q8K assert - fixed via shared
      indexed_supported() before every quantize_lhs; root cause of
      the expert-bias/stacked-shard test failures (detail: the
      loud-fallbacks close note).
      DEDUP LANDED 2026-10-17 late (candle commit 14f1348b): a live
        census (CANDLE_MOE_CENSUS=1, env-gated, 512-call stderr
        windows; STRIPPED 2026-10-18 after it delivered its gate data
        — the env flag no longer exists in candle) measured the real decode pattern: ~23 pairs (3 MTP
        verify positions x 8), only 14-17 DISTINCT experts, and
        ~94 indexed calls per 64ms step (~32% of the step in gemv;
        ~68% is serving overhead). Decode now groups pairs by expert
        and reads shared experts' tiles once for up to MAX_MT rows
        via per-pair row/dst indirection (no gather/scratch, one
        barrier region); quantize_lhs/widen/narrow go serial for
        decode-sized batches and the output buffer is uninit. Bench
        at the real Q8_0 shapes: overlapping routing 2.3-2.5x faster
        (gate_up 0.890 -> 0.355 ms/call), distinct unchanged. Fast
        probe A/B: decode 36 -> 41 tok/s (+14%), census avg 230 ->
        200us/call. Release probe deferred until the candle push so
        target/release/mistralrs stays == the pushed tip prod runs
        (expect ~43 on release). NEXT LEVERS, in value order: (1)
        the ~68% serving overhead - per-tensor-per-layer CPU<->GPU
        crossings (~94 calls x 3 transfers), per-call Tensor
        alloc/bookkeeping, scheduler syncs - a mistralrs-core choke
        point batching task; (2) phase-2 hot-expert VRAM cache with
        router-predicted prefetch (the differentiator; cuts the RAM
        read entirely for hot experts); (3) further kernel work is
        low-yield (gemv already ~145-175 GB/s at bench scale).
      BUG FOUND+FIXED 2026-10-17 (mistral.rs commit fc718e21a): the
        CUDA-role pipeline chunks prompts INTERNALLY (4096 cap,
        block-aligned to 32 for hybrid caches so recurrent
        checkpoints land on block boundaries); the tail chunk gathers
        prior KV, but the engine preflight reserved ZERO gather
        workspace for a no-prefix first batch — every prompt with
        len % 32 != 0 (or > cap) failed 100% with "prompt KV gather
        requires N bytes, exceeding its preflight workspace limit of
        0 bytes". Latent in EVERY cuda-role lane incl. prod's current
        binary (hip lanes never see it: their engine chunks are block
        aligned; the unexplained 4B 1009-token pass on prod may be
        FA2-paged prefill serving the tail — unverified). The preflight
        now reserves the tail gather when the pipeline device is CUDA
        and the batch would split; the bail names query dims + lens.
        Validated on the release probe: 355/567/695/1217-token
        prompts pass with mtp on and off.
      PHASE-1 VERDICT: mechanism proven end-to-end (MTP + paged attn +
        indexed experts, 220x), but the lane LOSES to the hip iGPU as
        a primary (35 vs 63.6 tok/s) and costs ~60GB RAM + shared DDR5
        (the CONTENDS note: alternative lanes, not additive). Park as
        a lazy capacity alias only if ever needed; the path to
        beating hip is (c) phase-2 hot-expert VRAM cache with
        router-predicted prefetch (cuts the RAM read for hot experts
        entirely), optionally + unit-granularity tuning in
        try_indexed_gemv. OPERATIONAL: the split probe does NOT
        coexist with a loaded prod lane + a build (0GB avail -> OOM
        risk); probe kills are PID-file only (see COMPLETED outage
        note).
      PHASE-2 INFRA LANDED 2026-10-18 (mistral.rs commit fa996df07,
        34 files, +620 lines): HotExpertCache per MoE block - BF16
        VRAM slots (6.66MB/slot at 256 experts, inter=512, hidden=
        2048), LRU eviction, use-count-2 admission, async fill worker
        (dequant + H2D + splice), split forward (hot pairs on GPU via
        candle matmul, cold pairs through the CPU gather path, partial
        weighted sums merged). Plumbing: expert_hot_cache_mb (MB) per
        [[models]] entry following the expert_device chain. Disabled
        by default (mb=0 / None -> no cache, no regression).
      PHASE-2 MEASUREMENT: the cache fills VRAM correctly (~12GB at
        12000MB, 46 slots/layer) and output stays coherent, but the
        split is 3x SLOWER (12.2 vs 36 tok/s). Root cause: the cold
        path runs per (layer, tensor) with high fixed overhead
        (1.69ms/call x 123 calls = 208ms/step) - the per-call overhead
        is 6x the original path's (0.4ms) because each cold call
        processes 3-4 pairs but pays the same tensor-op + pool-sync
        overhead. The hot path (6ms) is swamped. The infrastructure
        is sound; the integration is the bottleneck.
      GATE DATA (census, storage-keyed, 256-expert scale): 123 tensors,
        top-44 coverage 0.76-0.98 (p50 0.83-1.00) - the popularity
        skew is real and the cache would pay IF the cold-path overhead
        were fixed. Phase-2b (prefetch prediction) and the crossing/
        alloc overhead work remain.
      RECOMMENDED NEXT (in value order):
        1. Reduce cold-path per-call overhead: fuse gate+up+down in
           mistralrs-quant's gather path (or a direct indexed_gemv
           call bypassing forward_gather's tensor ops). This is the
           same 68% serving-overhead lever that helps the non-split
           lane too.
           SUBTASKS (cold-path batching refactor):
           A. Reduce per-call overhead in candle's QTensor::indexed_gemv:
              cache ids conversion (avoid to_vec1 per call), reuse dst
              buffer (avoid alloc per call), minimize storage guard.
              Location: candle/candle-core/src/quantized/mod.rs:1222.
           B. Reduce per-call overhead in mistralrs-quant's
              qtensor_indexed_moe_forward: avoid redundant shape
              normalization, cache ids dtype check, minimize allocs.
              Location: mistralrs-quant/src/gguf/cpu.rs.
           C. Reduce per-call overhead in FastExpertsWeights::forward_gather:
              eliminate process_routed_stats (no-ops), skip broadcast_mul/
              sum/to_dtype for topk=1 (split path), reuse intermediate
              tensors. Location: mistralrs-core/src/moe/experts/backends.rs:2001.
           D. Amortize mistralrs-core per-layer orchestration: batch the
              3 gather_forward calls (gate, up, down) into a single
              indexed pass where possible; reduce scheduler syncs.
              Location: mistralrs-core/src/moe/experts/mod.rs:331.
              IMPLEMENTED 2026-10-23: FastExpertsWeights::forward_indexed
              (backends.rs) runs gate+up through the fused kernel and down
              through indexed_gemv directly in the flat (num_tokens, topk,
              .) form, skipping the gather_forward wrapper (4D shape
              normalization, per-call bookkeeping) for every projection.
              forward_gather's CPU branch tries it first; the old 4D
              pipeline remains as fallback. hot_cache cold path now uses
              the same forward_indexed + inline weighting (supersedes the
              forward_cold_indexed duplicate, deleted).
           E. Measurement + validation: benchmark before/after on the
              3090 split lane (target ~1.4x: 35->~49 tok/s); verify
              output coherence vs forward_gather reference.
              MEASURED 2026-10-23 (post A+B+C, release probe :1244,
              35B Q5_K_XL cuda:0 + expert_device=cpu, no hot cache):
              essay 256tok x3 via /v1/chat/completions = 31.3-33.5
              tok/s (thinking model, HTTP overhead included); metrics-
              counter window on a reasoning-heavy count prompt = 18.7
              tok/s. Baseline (TODO, different harness): 35.0 tok/s
              release. NOT directly comparable (methodology differs:
              thinking tokens, MTP acceptance variance, prompt shape).
              Verdict: A+B+C are correctness-preserving micro-opts
              (no-op removal, squeeze-vs-sum, ids-conversion skip);
              gain is within measurement noise for these alone. A true
              A/B needs a before-binary built from the pre-change tip
              with the same harness. The ~1.4x target still rides on
              subtask D (per-layer batching) + phase-2b prefetch.
              POST-D MEASURED 2026-10-23 (same harness, same prompt,
              post forward_indexed): 31.7-31.9 tok/s x3, zero errors.
              Flat vs post-A+B+C (31.3-33.5): the wrapper-level opts
              (A-D) are all correctness-preserving and safe (moe
              tests 25/0 dual, 4-shape compile clean) but the decode
              wall is NOT the gather_forward wrapper overhead. The
              68% serving overhead lives deeper: per-step scheduler
              syncs, MTP verify machinery, allocator behavior. Next
              lever for this lane is phase-2b prefetch + a profile
              of where the 68% actually goes (perf flamegraph on the
              probe decode loop) before more code changes.
        2. Phase-2b: router-predicted async prefetch (currently
           admission is count-based/reactive).
           LANDED 2026-10-23 (fe6ae612c): admission now fills an EMPTY
           slot on the router's first selection (fill overlaps the
           step's compute); evicting a warm LRU slot still needs the
           use-count-2 gate; duplicate ids no longer double-admit;
           empty slots preferred over eviction.
           MEASURED 2026-10-23 (same harness as the no-cache baseline,
           35B Q5_K_XL cuda:0 + cpu experts, essay 256tok, warmup + 3):
           4000MB (~15 slots/layer): 19.9-20.2 tok/s; 6000MB (~22
           slots/layer): 18.0-20.1 tok/s; 8000MB: CUDA OOM during
           decode (3090 ceiling between 6-8GB with weights+KV+graphs).
           VERDICT: hot cache is ~37% SLOWER than no-cache (31.7-31.9
           tok/s) and MORE SLOTS DO NOT HELP - the split's fixed
           overhead (per-layer hot/cold partitioning, per-slot GPU
           gathers, cold_xs index_select + index_add scatter,
           hot_partial accumulation) dominates, not coverage. With
           forward_indexed serving cold pairs cheaply, the bandwidth
           saved on hot pairs no longer pays for the orchestration.
           The path to a paying cache is an integration rewrite: batch
           the hot path across ALL slots into one matmul per layer
           (not per-slot gathers) and fold the scatter into it (~L,
           new item). Until then the cache stays disabled by default
           (expert_hot_cache_mb unset). OOM CAVEAT (user-observed
           2026-10-23): an in-request CUDA OOM does NOT free VRAM -
           the caching allocator retains pool blocks in-process, so an
           OOM'd lane holds its VRAM (serving nothing) until explicit
           /v1/models/unload (trims per 18d04d953) or process exit.
           STACKED REWRITE LANDED + MEASURED 2026-10-23 (flat again):
           slot storage is now ONE persistent stacked tensor per cache
           ([slots, 2i, h] gate_up + [slots, h, i] down, BF16); fills
           write in place via Tensor::slice_set (no per-step weight
           cat, no double VRAM, fills never re-copy the stack); the
           hot bmm runs over ALL slots (inactive/Filling rows carry
           zero weight); hot+cold fold into ONE index_add; the second
           per-step state lock is gone. Verified coherent output, 0
           fill failures, moe tests 25/0 dual, 4 shapes clean. YET
           4000MB measures 18.2-20.3 tok/s (warmup 19.1) - flat vs
           the cat design (19.9-20.2). TWO theories now falsified
           (wrapper overhead, per-step weight cat). PRIME SUSPECT for
           the remaining ~0.45ms/layer: the hot path's 2 BLOCKING
           D2H syncs per layer (topk_ids + topk_weights to_vec1,
           ~82 pipeline-draining syncs/step) which the no-cache path
           does not have (it moves ids to CPU via async to_device and
           reads CPU storage). NEXT STEP IS A PROFILE, not more
           theory: nsys/perf flamegraph of the probe decode loop to
           place the 68% serving overhead before any further code
           change on this lane.
           PROFILE DONE 2026-10-23 (nsys, 30s windows, one 256-tok
           generation each, hot=4000MB vs no-cache; reports discarded
           after extraction, numbers preserved here):
           - Per-op CUDA churn IS the serving overhead, both paths:
             ~368-394K kernel launches, 1.25-1.4M event records,
             2.2-2.4M stream-waits per window. Candle pays ~5 API
             calls per tensor op.
           - THE SPLIT'S DELTA: cuMemcpyDtoHAsync 23,355 calls / 4.20s
             API time (avg 180us) hot vs 16,093 / 0.45s (avg 28us)
             no-cache - a 9.4x TIME difference for only +7K calls.
             The hot path's per-layer to_vec1 host reads of topk_ids/
             topk_weights are PAGEABLE async copies, which the driver
             executes synchronously via staging - each one waits for
             the GPU queue to drain (mid-step = deep queue). ~2 drains
             x 41 layers/step ~= 10ms/step of pipeline lockstep: the
             measured split cost.
           - CONFOUND: the hot window contained the FIRST request
             (warmup), so its 10.6s GPU-side HtoD time (vs 0.10s
             no-cache) is mostly the FILL WORKER's pageable 6.3MB/
             expert uploads during cache warm, not steady-state.
           - CONCLUSION: the fix is GPU-SIDE PARTITIONING - build
             batch_idx/pair_w/slot mapping with GPU ops from topk_ids
             (argsort machinery already exists; asort kernels are in
             the trace) so serving never reads routing to the host;
             zero per-layer DtoH in the hot path. Fill uploads should
             use pinned staging (pageable 6.3MB copies run ~100x
             slower than the no-cache path's 1.3us avg HtoD). Both
             are the concrete next ~L on this lane; do partitioning
             first.
           STAGE 1 MEASURED FLAT 2026-10-23: routing reads switched to
             the cheap to_device(Cpu) mechanism (the no-cache lane's
             pattern) + admission policy (use counts, admit, fills)
             moved OFF the serving thread to a dedicated cache worker
             fed per-step ids via channel - 19.4-20.0 tok/s, flat.
             Read mechanism and policy placement are NOT the wall.
           EVENT-TRACKING EXPERIMENT (REVERTED, 2026-10-23): root-caused
             the per-op churn to cudarc's CudaSlice use-tracking (event
             create/record/wait/destroy per use; candle's single
             dedicated stream is created via new_stream(), which flips
             num_streams>0 = "multi-stream mode" and activates the
             tracking). Tried disable_event_tracking() at device init
             (candle 70de65bc, reverted same day): hot lane CRASHED to
             10.8-11.4 tok/s (output stayed coherent; a slow fallback
             path, not garbage). LESSON: the event machinery is
             load-bearing for more than cross-stream order - it is
             also cudarc's HOST-READ completion sync (the DtoH path
             records + waits an event before the host touches the
             buffer). Blanket disable is WRONG. The finer fix (future
             candle item): keep read-sync events but stop the per-op
             create/destroy churn (reuse pooled events), i.e. attack
             the 816K creates / 2.4M waits volume, not the sync itself.
             That is a candle-core/cudarc task, fleet-wide in scope,
             and the highest-value remaining lever for BOTH lanes (the
             no-cache 68% overhead lives in the same machinery).
           POOLED-EVENT LANDED 2026-10-23 (candle 487ff1df, both cudarc
             forks): same-stream uses skip the launch/DevicePtr tracking
             waits+records (capture keeps byte-identical graph
             structure; cross-stream users and the load-bearing
             SyncOnDrop host-read sync untouched); per-slice sync events
             recycle through a context freelist (cap 1024;
             capture-recorded events never pooled; never-recorded events
             skip the drop wait). MEASURED (same harness): no-cache
             31.7 -> 32.0-34.5 tok/s (median ~33, +3-5% - most of the
             removed API time overlapped GPU execution, only the
             non-overlapped part converts to wall-clock); hot-cache
             lane flat ~20 (its wall is the per-layer DtoH drain
             reads). moe 57/0, cuda_graph 31/0 dual. VERDICT: keep
             (strict improvement, millions fewer driver calls); the
             68% overhead is NOT primarily the event API time - it is
             GPU-side op granularity + launch/alloc costs that overlap
             execution. Lane status: no-cache ~33 = the 3090 serving
             mode; hot cache stays off by default (net-negative).
        3. Only then re-enable the cache and re-measure.

      PHASE-2 INFRA LANDED 2026-10-18 (mistral.rs commit fa996df07,
        34 files, +620 lines): HotExpertCache per MoE block - BF16
        VRAM slots (6.66MB/slot at 256 experts, inter=512, hidden=
        2048), LRU eviction, use-count-2 admission, async fill worker
        (dequant + H2D + splice), split forward (hot pairs on GPU via
        candle matmul, cold pairs through the CPU gather path, partial
        weighted sums merged). Plumbing: expert_hot_cache_mb (MB) per
        [[models]] entry following the expert_device chain. Disabled
        by default (mb=0 / None -> no cache, no regression).
      PHASE-2 MEASUREMENT: the cache fills VRAM correctly (~12GB at
        12000MB, 46 slots/layer) and output stays coherent, but the
        split is 3x SLOWER (12.2 vs 36 tok/s). Root cause: the cold
        path runs per (layer, tensor) with high fixed overhead
        (1.69ms/call x 123 calls = 208ms/step) - the per-call overhead
        is 6x the original path's (0.4ms) because each cold call
        processes 3-4 pairs but pays the same tensor-op + pool-sync
        overhead. The hot path (6ms) is swamped. The infrastructure
        is sound; the integration is the bottleneck.
      GATE DATA (census, storage-keyed, 256-expert scale): 123 tensors,
        top-44 coverage 0.76-0.98 (p50 0.83-1.00) - the popularity
        skew is real and the cache would pay IF the cold-path overhead
        were fixed. Phase-2b (prefetch prediction) and the crossing/
        alloc overhead work remain.
      RECOMMENDED NEXT (in value order):
        1. Reduce cold-path per-call overhead: fuse gate+up+down in
           mistralrs-quant's gather path (or a direct indexed_gemv
           call bypassing forward_gather's tensor ops). This is the
           same 68% serving-overhead lever that helps the non-split
           lane too.
           SUBTASKS (cold-path batching refactor):
           A. Reduce per-call overhead in candle's QTensor::indexed_gemv:
              cache ids conversion (avoid to_vec1 per call), reuse dst
              buffer (avoid alloc per call), minimize storage guard.
              Location: candle/candle-core/src/quantized/mod.rs:1222.
           B. Reduce per-call overhead in mistralrs-quant's
              qtensor_indexed_moe_forward: avoid redundant shape
              normalization, cache ids dtype check, minimize allocs.
              Location: mistralrs-quant/src/gguf/cpu.rs.
           C. Reduce per-call overhead in FastExpertsWeights::forward_gather:
              eliminate process_routed_stats (no-ops), skip broadcast_mul/
              sum/to_dtype for topk=1 (split path), reuse intermediate
              tensors. Location: mistralrs-core/src/moe/experts/backends.rs:2001.
           D. Amortize mistralrs-core per-layer orchestration: batch the
              3 gather_forward calls (gate, up, down) into a single
              indexed pass where possible; reduce scheduler syncs.
              Location: mistralrs-core/src/moe/experts/mod.rs:331.
              IMPLEMENTED 2026-10-23: FastExpertsWeights::forward_indexed
              (backends.rs) runs gate+up through the fused kernel and down
              through indexed_gemv directly in the flat (num_tokens, topk,
              .) form, skipping the gather_forward wrapper (4D shape
              normalization, per-call bookkeeping) for every projection.
              forward_gather's CPU branch tries it first; the old 4D
              pipeline remains as fallback. hot_cache cold path now uses
              the same forward_indexed + inline weighting (supersedes the
              forward_cold_indexed duplicate, deleted).
           E. Measurement + validation: benchmark before/after on the
              3090 split lane (target ~1.4x: 35->~49 tok/s); verify
              output coherence vs forward_gather reference.
              MEASURED 2026-10-23 (post A+B+C, release probe :1244,
              35B Q5_K_XL cuda:0 + expert_device=cpu, no hot cache):
              essay 256tok x3 via /v1/chat/completions = 31.3-33.5
              tok/s (thinking model, HTTP overhead included); metrics-
              counter window on a reasoning-heavy count prompt = 18.7
              tok/s. Baseline (TODO, different harness): 35.0 tok/s
              release. NOT directly comparable (methodology differs:
              thinking tokens, MTP acceptance variance, prompt shape).
              Verdict: A+B+C are correctness-preserving micro-opts
              (no-op removal, squeeze-vs-sum, ids-conversion skip);
              gain is within measurement noise for these alone. A true
              A/B needs a before-binary built from the pre-change tip
              with the same harness. The ~1.4x target still rides on
              subtask D (per-layer batching) + phase-2b prefetch.
              POST-D MEASURED 2026-10-23 (same harness, same prompt,
              post forward_indexed): 31.7-31.9 tok/s x3, zero errors.
              Flat vs post-A+B+C (31.3-33.5): the wrapper-level opts
              (A-D) are all correctness-preserving and safe (moe
              tests 25/0 dual, 4-shape compile clean) but the decode
              wall is NOT the gather_forward wrapper overhead. The
              68% serving overhead lives deeper: per-step scheduler
              syncs, MTP verify machinery, allocator behavior. Next
              lever for this lane is phase-2b prefetch + a profile
              of where the 68% actually goes (perf flamegraph on the
              probe decode loop) before more code changes.
        2. Phase-2b: router-predicted async prefetch (currently
           admission is count-based/reactive).
           LANDED 2026-10-23 (fe6ae612c): admission now fills an EMPTY
           slot on the router's first selection (fill overlaps the
           step's compute); evicting a warm LRU slot still needs the
           use-count-2 gate; duplicate ids no longer double-admit;
           empty slots preferred over eviction.
           MEASURED 2026-10-23 (same harness as the no-cache baseline,
           35B Q5_K_XL cuda:0 + cpu experts, essay 256tok, warmup + 3):
           4000MB (~15 slots/layer): 19.9-20.2 tok/s; 6000MB (~22
           slots/layer): 18.0-20.1 tok/s; 8000MB: CUDA OOM during
           decode (3090 ceiling between 6-8GB with weights+KV+graphs).
           VERDICT: hot cache is ~37% SLOWER than no-cache (31.7-31.9
           tok/s) and MORE SLOTS DO NOT HELP - the split's fixed
           overhead (per-layer hot/cold partitioning, per-slot GPU
           gathers, cold_xs index_select + index_add scatter,
           hot_partial accumulation) dominates, not coverage. With
           forward_indexed serving cold pairs cheaply, the bandwidth
           saved on hot pairs no longer pays for the orchestration.
           The path to a paying cache is an integration rewrite: batch
           the hot path across ALL slots into one matmul per layer
           (not per-slot gathers) and fold the scatter into it (~L,
           new item). Until then the cache stays disabled by default
           (expert_hot_cache_mb unset). OOM CAVEAT (user-observed
           2026-10-23): an in-request CUDA OOM does NOT free VRAM -
           the caching allocator retains pool blocks in-process, so an
           OOM'd lane holds its VRAM (serving nothing) until explicit
           /v1/models/unload (trims per 18d04d953) or process exit.
           STACKED REWRITE LANDED + MEASURED 2026-10-23 (flat again):
           slot storage is now ONE persistent stacked tensor per cache
           ([slots, 2i, h] gate_up + [slots, h, i] down, BF16); fills
           write in place via Tensor::slice_set (no per-step weight
           cat, no double VRAM, fills never re-copy the stack); the
           hot bmm runs over ALL slots (inactive/Filling rows carry
           zero weight); hot+cold fold into ONE index_add; the second
           per-step state lock is gone. Verified coherent output, 0
           fill failures, moe tests 25/0 dual, 4 shapes clean. YET
           4000MB measures 18.2-20.3 tok/s (warmup 19.1) - flat vs
           the cat design (19.9-20.2). TWO theories now falsified
           (wrapper overhead, per-step weight cat). PRIME SUSPECT for
           the remaining ~0.45ms/layer: the hot path's 2 BLOCKING
           D2H syncs per layer (topk_ids + topk_weights to_vec1,
           ~82 pipeline-draining syncs/step) which the no-cache path
           does not have (it moves ids to CPU via async to_device and
           reads CPU storage). NEXT STEP IS A PROFILE, not more
           theory: nsys/perf flamegraph of the probe decode loop to
           place the 68% serving overhead before any further code
           change on this lane.
           PROFILE DONE 2026-10-23 (nsys, 30s windows, one 256-tok
           generation each, hot=4000MB vs no-cache; reports discarded
           after extraction, numbers preserved here):
           - Per-op CUDA churn IS the serving overhead, both paths:
             ~368-394K kernel launches, 1.25-1.4M event records,
             2.2-2.4M stream-waits per window. Candle pays ~5 API
             calls per tensor op.
           - THE SPLIT'S DELTA: cuMemcpyDtoHAsync 23,355 calls / 4.20s
             API time (avg 180us) hot vs 16,093 / 0.45s (avg 28us)
             no-cache - a 9.4x TIME difference for only +7K calls.
             The hot path's per-layer to_vec1 host reads of topk_ids/
             topk_weights are PAGEABLE async copies, which the driver
             executes synchronously via staging - each one waits for
             the GPU queue to drain (mid-step = deep queue). ~2 drains
             x 41 layers/step ~= 10ms/step of pipeline lockstep: the
             measured split cost.
           - CONFOUND: the hot window contained the FIRST request
             (warmup), so its 10.6s GPU-side HtoD time (vs 0.10s
             no-cache) is mostly the FILL WORKER's pageable 6.3MB/
             expert uploads during cache warm, not steady-state.
           - CONCLUSION: the fix is GPU-SIDE PARTITIONING - build
             batch_idx/pair_w/slot mapping with GPU ops from topk_ids
             (argsort machinery already exists; asort kernels are in
             the trace) so serving never reads routing to the host;
             zero per-layer DtoH in the hot path. Fill uploads should
             use pinned staging (pageable 6.3MB copies run ~100x
             slower than the no-cache path's 1.3us avg HtoD). Both
             are the concrete next ~L on this lane; do partitioning
             first.
           STAGE 1 MEASURED FLAT 2026-10-23: routing reads switched to
             the cheap to_device(Cpu) mechanism (the no-cache lane's
             pattern) + admission policy (use counts, admit, fills)
             moved OFF the serving thread to a dedicated cache worker
             fed per-step ids via channel - 19.4-20.0 tok/s, flat.
             Read mechanism and policy placement are NOT the wall.
           EVENT-TRACKING EXPERIMENT (REVERTED, 2026-10-23): root-caused
             the per-op churn to cudarc's CudaSlice use-tracking (event
             create/record/wait/destroy per use; candle's single
             dedicated stream is created via new_stream(), which flips
             num_streams>0 = "multi-stream mode" and activates the
             tracking). Tried disable_event_tracking() at device init
             (candle 70de65bc, reverted same day): hot lane CRASHED to
             10.8-11.4 tok/s (output stayed coherent; a slow fallback
             path, not garbage). LESSON: the event machinery is
             load-bearing for more than cross-stream order - it is
             also cudarc's HOST-READ completion sync (the DtoH path
             records + waits an event before the host touches the
             buffer). Blanket disable is WRONG. The finer fix (future
             candle item): keep read-sync events but stop the per-op
             create/destroy churn (reuse pooled events), i.e. attack
             the 816K creates / 2.4M waits volume, not the sync itself.
             That is a candle-core/cudarc task, fleet-wide in scope,
             and the highest-value remaining lever for BOTH lanes (the
             no-cache 68% overhead lives in the same machinery).
           POOLED-EVENT LANDED 2026-10-23 (candle 487ff1df, both cudarc
             forks): same-stream uses skip the launch/DevicePtr tracking
             waits+records (capture keeps byte-identical graph
             structure; cross-stream users and the load-bearing
             SyncOnDrop host-read sync untouched); per-slice sync events
             recycle through a context freelist (cap 1024;
             capture-recorded events never pooled; never-recorded events
             skip the drop wait). MEASURED (same harness): no-cache
             31.7 -> 32.0-34.5 tok/s (median ~33, +3-5% - most of the
             removed API time overlapped GPU execution, only the
             non-overlapped part converts to wall-clock); hot-cache
             lane flat ~20 (its wall is the per-layer DtoH drain
             reads). moe 57/0, cuda_graph 31/0 dual. VERDICT: keep
             (strict improvement, millions fewer driver calls); the
             68% overhead is NOT primarily the event API time - it is
             GPU-side op granularity + launch/alloc costs that overlap
             execution. Lane status: no-cache ~33 = the 3090 serving
             mode; hot cache stays off by default (net-negative).
        3. Only then re-enable the cache and re-measure.

      UPDATE 2026-10-23 (later same day): cold-path fast path implemented
      (mistral.rs a9cc884a1) — forward_cold_indexed bypasses
      forward_gather's per-call tensor-op overhead by running gate+up+down
      directly through candle's indexed_gemv kernels (fused gate+up lhs
      quantization, activation, weighted sum), with fallback to
      forward_gather on unsupported layouts. Verified end-to-end on a 3090
      probe (expert_hot_cache_mb=4000): requests complete with no errors,
      no OOM, no dtype mismatch. Lane remains parked (mechanism proven,
      perf optimization pending phase-2b prefetch + cold-path batching).
      PARKED 2026-10-23 (user decision): the expert-split MoE lane is
      usable as-is (~35 tok/s on the 3090, ~4.6GB VRAM, ~60GB RSS)
      and provides optionality/flexibility for running MoE models on
      the CUDA path, but it loses to the hip iGPU lane (63.6 tok/s)
      and the remaining lever (cold-path per-call overhead, ~68% of
      step time) is a substantial mistralrs-core refactor of
      MoEExperts::forward_gather for ~1.4x gain (~35->~49 tok/s).
      The mechanism is proven end-to-end (MTP + paged attn + indexed
      experts on x86); the bottleneck is mistralrs-core orchestration:
      per-tensor-per-layer CPU<->GPU crossings (~94 calls x 3
      transfers), per-call Tensor alloc/bookkeeping, scheduler syncs.
      forward_gather makes 3 gather_forward calls per layer (gate, up,
      down); gate+up already fused via indexed_gemv_fused_gate_up;
      down has a different input so it cannot be fused into the same
      kernel. The real win is amortizing per-call bookkeeping across
      all experts in a layer (single indexed_gemv-style call or
      batching transfers/syncs in mistralrs-core). Revisit when there
      is bandwidth; the path to beating hip is phase-2 hot-expert VRAM
      cache with router-predicted prefetch. Current config: 35B MoE
      entries use cuda:0 + expert_device=cpu (expert_hot_cache_mb unset
      = disabled by default).

      The hip iGPU lane (48-51 tok/s) remains the production 35B
      
  UPDATE 2026-10-17 (CPU PA + single-vendor CUDA MTP):
    - CPU PagedAttention now works even in GPU builds: cpu module is
      always compiled (mistralrs-paged-attn/src/cpu/), functions
      dispatched at runtime via device.is_cpu() checks in
      write_kv_cache and gather_kv_cache_for_layout. run_decode
      already falls back to run_decode_gather_sdpa on CPU.
    - backend_caps(Device::Cpu): paged_attention=true, mtp=true
      in all builds (CPU backend now always compiled, functions
      dispatched at runtime via device.is_cpu() checks).
    - Single-vendor CUDA MTP: enabled (mtp=true in backend_caps).
      Speculative rejection uses standard CUDA FFI (super::ffi),
      GDN speculative paths use generic fallbacks in gdn.rs.
      No gdn_cuda/ffi_cuda needed for single-vendor builds.
      lane until (a) or (b) lands. Note: the 35B's MTP head is also
      MoE (blk.40 has ffn_*_exps) so draft steps hit the same CPU
      bottleneck; MTP cannot help until the CPU path is fixed.
        (a) optimize the CPU path: rayon-parallelize across the 8
            active experts per layer, or use try_grouped_gather_mm
            with dequantized experts (2x RAM but faster GEMM).
        (b) phase-2 hot-expert VRAM cache with router-predicted
            prefetch (the differentiator; ~0.27GB/expert at Q5, 3090
            holds dozens; prefetch overlaps with attention/GDN).
        (c) accept 35B-on-3090 as a slow/capacity lane only.
      The hip iGPU lane (48-51 tok/s) remains the production 35B
      lane until (a) or (b) lands.
      VERIFIED FEASIBLE, CHEAPER THAN THE ~M SUGGESTED: every hard
      piece already ships in-tree. The Fast experts backend has a
      CPU/no-fused-ops gather branch (`forward_gather` else-arm in
      moe/experts/backends.rs) that runs `cpu_indexed_moe_forward` ->
      `qtensor.indexed_moe_forward` -> `QTensor::indexed_gemv`
      (mistralrs-quant/src/gguf/{mod,cpu}.rs) in every build shape -
      it is the CPU-only-build MoE path. GGUF prequantized expert
      loading honors the VarBuilder device, so placement is "load the
      experts vb on CPU". Router/topk stays on GPU (caller-side,
      untouched). Backend selection keys on xs.device() so a
      CPU-resident xs automatically takes the CPU gather path with no
      backend changes.
      35B STRUCTURE (from the GGUF dump): ALL 41 blocks incl. the MTP
      block (blk.40) are MoE (ffn_*_exps + _shexp + gate_inp); 30 GDN
      + 10 full-attn stay on GPU; expert mass = 93.1% of the file,
      GPU-resident slice ~7% (attn/router/norms/embed 6.6% + shared
      0.4% + mtp extras ~0%).
      DESIGN: per-model toml knob `expert_device = "cpu"` -> (mirror
      the mtp chain exactly) GGUFSpecificConfig -> NormalPipeline ->
      config-JSON injection `_mistralrs_expert_device` ->
      qwen3_next::Config -> MoEExpertsConfig.expert_device. Placement
      in MoEExperts::new/new_direct: experts vb at
      cfg.expert_device.unwrap_or(layer_device); router gate, shared
      expert, shared gate stay on layer_device. Forward choke point in
      MoEExperts::forward: when expert device != xs device, move
      xs_flat + topk_weights + topk_ids over, run backend, move ys
      back. Graphs: supports_cuda_decode_graphs must return false for
      qwen3_next when the split is on (CPU matmul mid-step breaks
      capture; the hip lane's +69% graphs win does not transfer).
      MTP stays ON: the MTP block is also MoE and also splits; the
      verify step amortizes expert reads across n+1 positions.
      Guard: bail loudly if expert_device and ISQ are both set.
      BUDGET (Q5_K_XL, 26G file): 3090 weights ~1.8GB + huge KV/GDN
      pool headroom; RAM experts ~24.2GB (107GB avail). Decode:
      expert reads 3.0GB/step (8/64 of expert mass, amortized over
      MTP positions) at ~180-220GB/s effective DDR5 = 14-17ms + GPU
      hot ~5ms + 41x2 small device crossings ~2ms = 21-24ms/step; at
      accept ~2.2 that is ~90-105 tok/s theoretical. First-cut target
      only: match-or-beat the hip lane (48-51 tok/s); CPU GEMV
      throughput is the biggest unknown, measure early in the probe.
      Prefill is the ugly number: a 2k-token chunk reads ~all 64
      experts per layer on CPU (~2-4s/chunk one-shot, llama.cpp
      --cpu-moe behaves the same); accepted for phase 1. Q8_0
      experts (33.5GB RAM, 4.2GB/step) is the quality-upside data
      point. CONTENDS for DDR5 with the iGPU lanes: 35B-hip +
      35B-cuda running concurrently halve each other; they are
      alternative lanes, not additive capacity.
      FILES (~500 lines): cli config ModelEntry + GGUFSpecificConfig
      + the three gguf.rs builder sites; loaders/mod.rs inject
      helper + normal.rs injection; qwen3_next.rs (Config field,
      SparseMoeBlock::load, graphs gate, ISQ guard);
      moe/experts/config.rs (MoEExpertsConfig field); moe/experts/
      mod.rs (placement + choke point). mistralrs-quant: NOTHING.
      VALIDATION: shapes x4 + tests; fast probe :1244 35B cuda:0
      expert_device=cpu (load, prefill, decode x3, accept, CPU%
      during decode to confirm threadpool saturation); greedy sanity
      vs hip-35B output (approximate-determinism caveat: coherence +
      accept, NOT token-exact); release probe -> rate decision ->
      if >= hip lane, add a LAZY prod alias qwen3.6-35b-cuda (loads
      on demand, hip lane untouched, zero-risk rollout).
      PHASE 2 (only if numbers justify): hot-expert VRAM cache with
      router-predicted async prefetch (the real differentiator;
      ~0.27GB/expert at Q5, the 3090 could hold dozens; router names
      the 8 experts before the FFN runs so prefetch overlaps the
      attention/GDN work); prefill acceleration (stream the touched
      experts per layer over PCIe for prefill chunks); per-layer
      graph subgraphs.

      PHASE 3 (device-agnostic MoE expert offloading): CLOSED.
      Conclusion: sync overhead is fundamental on discrete GPUs and cannot
      be eliminated. Root cause: routing runs after attention, so no
      prefetch window exists within a layer. The dependency chain
      (attention -> MoE -> next attention) prevents pipelining. "Keep xs
      on GPU, move cold weights to GPU" trades one sync for another
      (~1ms/layer) with no net gain. Separate CUDA streams don't help
      (compute stream idle during cold gemv, but next layer's attention
      can't start until MoE output is ready).
      Decision: (a) accept sync as the cost of capacity (running models
      that don't fit in VRAM), (b) hip iGPU is the throughput path
      (unified memory, zero sync, 63.6 tok/s), (c) discrete GPU split
      lane targets capacity (~42 tok/s on 3090), not throughput.
      Cannot assume more VRAM (expert_hot_cache_mb is user-tunable).
      Phase-2 infra + gather fusion + execute_chunked threshold are
      shipped (disabled by default). No further phase-3 implementation.
      Remaining perf lever: 68% serving-overhead campaign on hip lane
      (PARKED 2026-10-18, see SERVING-CAMPAIGN RE-SCOPE below).
      MEASURED (2026-10-18, release build): cache OFF 42.2-42.5 tok/s,
      cache ON 26.5-27.9 tok/s (net loss). Gather fusion landed:
      indexed_gemv_fused_gate_up quantizes xs once for gate+up.
      COMMITS: candle cde56382 + b9e59943 + 1f2bb865; mistral.rs
      cb8ddfdf1 + da0baa7b8 + 6357f09fa + 5c975bcfc + cccace074.

      SYNC OVERHEAD ANALYSIS (2026-10-18): the "keep xs on GPU, move cold
      weights to GPU" approach does NOT eliminate the sync. Root cause:
      dependency chain prevents prefetch. Routing runs AFTER attention (MoE
      block starts with the gate, which needs attention output). So cold
      expert weights are identified during the MoE, not before it. No
      prefetch window exists within a layer. The D2H sync of xs (~1ms/layer)
      is replaced by a weight H2D sync (~1ms/layer) -- same cost, not lower.
      Separate CUDA streams don't help either: the CPU gemv for layer L
      blocks attention L+1 (MoE output feeds next layer's attention input).
      The ONLY way to eliminate sync is to have ALL experts on GPU (model
      fits in VRAM). For models that don't fit, sync overhead is
      fundamental. Therefore: (1) maximize hot cache coverage (more VRAM
      slots = fewer cold pairs = fewer syncs), (2) accept remaining sync
      cost as the price of capacity (running larger models on small GPUs),
      (3) hip iGPU remains the throughput champion (unified memory = zero
      sync, 63.6 tok/s). Phase 3 should focus on (1) and (3), not on
      eliminating sync. The "keep xs on GPU" rewrite is NOT worth the
      effort -- it trades one sync for another with no net gain.
      FINAL PHASE 3 DECISION (2026-10-18): sync overhead is fundamental
      and cannot be eliminated on discrete GPUs (no prefetch window
      within a layer; dependency chain prevents pipelining). Cannot
      assume more VRAM (expert_hot_cache_mb is user-tunable, not
      guaranteed). Therefore: (a) accept sync as the cost of capacity
      (running models that don't fit in VRAM), (b) hip iGPU is the
      throughput path (unified memory, zero sync, 63.6 tok/s), (c)
      discrete GPU split lane targets capacity, not throughput.
      Phase-2 infra + gather fusion + execute_chunked threshold are
      shipped (disabled by default). No further phase-3 implementation
      planned. The 68% serving-overhead campaign on the hip lane is the
      remaining perf lever (PARKED 2026-10-18, user call - see the
      SERVING-CAMPAIGN RE-SCOPE block below for the profile, the
      roof-bound conclusion, and the failed lm_head Q4_K experiment).
      SERVING-CAMPAIGN RE-SCOPE + PROFILE (2026-10-18, rocprofv3 on
      :1246, release binary): the "68% serving overhead" census figure
      was a CPU-split-lane result - the hip lane decode is KERNEL-TIME
      bound (~97% GPU busy; 19.2ms/token batch=1 non-MTP, 13.6ms/
      token-equiv MTP n=2). MTP kernel re-rank (4-request tail):
      experts fused 31.9% (gate_up 19.4 + down 12.6, ~208GB/s
      effective = measured DDR5 roof - no kernel headroom), lm_head
      18.9% (659ms/window; output.weight 540MB Q8_0 streaming at
      ~237GB/s = roof; ~3 launches/cycle = 2 draft proposals + 1
      verify scoring), dense-verify plain_cuda3 22.8% (qkv 17MB@81us
      = 220GB/s roof; ssm_out/gate/o_proj at 170-190GB/s), fused
      qkv+glu 5.1%, router gemv+topk 3.5%, gdn_recurrence 3.6%,
      paged_attention 3.1%, element/mask/copy chains ~10%. Byte
      accounting per cycle: ~4.5GB stream (dense 1.47GB/verify pass +
      experts 3x18.9MBx41 layers + head 2.3GB) = 128GB/s overall;
      biggest slices already at roof.
      GGUF MAP (parsed header): hidden 2048, vocab 248320, 41 MoE
      layers (30 GDN + 11 full-attn incl nextn), 256 experts
      inter=512 topk=8 + shared experts, output/token_embd 540MB Q8_0.
      LM-HEAD DTYPE EXPERIMENT (2026-10-18): llama.cpp C-tool
      requant (tensor-type-pinned dry-run verified 753 tensors, only
      output.weight Q8_0->Q4_K; 5s runtime). A/B on :1244 same-day
      (baseline reproduced: decode 71.1-71.3 / total 82.7-82.9
      steady): headq4 decode 68.2 / total 79.5 = -4.3% NET. Failure
      math: -19% head bytes saves ~6% of cycle; acceptance drops ~10%
      relative (~6% fewer tokens/cycle) -> net loss. CORRECTION OF
      PREMISE: for this model the GGUF draft reuses output.weight for
      proposals AND verify scoring AND final sampling (the
      draft_lm_head Option path is not populated for GGUF loads), so
      the surgery perturbs the verify distribution itself, not just
      proposal quality. VERDICT: lm_head stays Q8_0 (kernel is
      roof-bound; only structural change - fewer launches/cycle -
      could attack it, and n_predict=2 is already known-best). 27GB
      experiment artifact deleted; probe artifacts (qflags.txt,
      llama.cpp build, rocprof CSVs) kept in scratch/.
      REMAINING MEASURED LEVERS (ranked): (a) dense batch-3 mmvq
      micro-tune (gate/ssm_out/o_proj 170-190 -> 220GB/s: ~2-4%),
      (b) elementwise/mask fusion around GDN + MTP accept machinery
      (~2-3%), (c) expert dtype downtrade Q5K/Q6K -> lower (~4-5%,
      quality tradeoff, USER SIGN-OFF required), (d) router topk
      fusion (~1%). Campaign conclusion: hip MTP lane is at/near the
      streaming roof on all big slices; levers are marginal singles,
      not the imagined double-digit "serving overhead" windfall.
      PARKED 2026-10-18 (USER CALL, campaign closed): none of (a)-(d)
      is economically viable for the effort at ~2-5% ceiling each.
      Ship state carried: 35B prod lane = hip iGPU, Q5_K_XL, MTP
      n=2 greedy, 71.3 tok/s decode / 82.8 total (validated twice
      same-day); lm_head Q8_0 (Q4_K downtrade measured NET-NEGATIVE,
      see LM-HEAD DTYPE EXPERIMENT); phase-2 hot-expert cache shipped
      disabled (mb=0, default); expert split kept as capacity lane
      only. Baselines for any future resumption are in Appendix A
      territory: non-MTP hip probe 50.8 tok/s; MTP 71.3; rocprofv3
      traces + parse scripts + llama.cpp requant tooling preserved in
      LocalAI/scratch (rocprof-out/, qflags.txt, llama.cpp/ build).
      Prod restarts only in verified idle windows iff a release build
      should be picked up (no code changed in this campaign - none
      needed). PUSH STATE AT PARKING: candle fully pushed (0 ahead);
      mistral.rs pstreet-rocm ahead 2 = the two 2026-10-18 ledger
      commits (beea6a8d7 + this PARKED commit), push is a user
      action.
- [x] ~S Solo-rocm loud fallbacks: fused FP8 GLU quantize (unfused
      path today), gemv -> plain GEMM, HQQ CPU bitpack-at-load.
      Pure wins, no architecture risk.
      CLOSED 2026-10-21 (logging-only; all three fallbacks already
      WORKED silently): (1) blockwise FP8 forward/gather dequant
      fallbacks now warn once when serving a GPU-resident input
      ("no native matmul on this device role") - fires for solo-rocm
      hip, dual hip, and sm-less NVIDIA, all honest cases; (2) solo-
      rocm should_use_gemv warns once on the first GPU-role query
      before returning false (plain GEMM serves), and the stale gemv
      bail text ("needs S2" - S2 shipped 2026-09-28) now states the
      gate; (3) HQQ bitpack tails (8/4/2/1-bit) warn once when the
      tensor-op packing fallback serves a GPU device (Three-bit has
      no kernel anywhere - universal path, left silent). Pattern =
      the file's existing DEEPGEMM_*_LOG Once + tracing::warn. Zero
      behavior change. Verified: 4 shapes check clean, fmt/clippy
      clean, dual --lib bar 341 passed / 4 failed IDENTICAL with the
      change stashed (the 4 = the documented expert-bias/stacked-
      shard family + its hqq/mxfp4 mirrors - pre-existing, do not
      chase). NEW pre-existing breakage recorded while verifying:
      mistralrs-quant default-shape LIB TESTS do not compile (test
      code references test_gpu_device/cuda_backend ungated) and the
      dual shape cannot compile routed_lora_cuda_tests +
      routed_lora_bench (stale LoraExecutionArena methods) - both
      stash-verified on the clean tree; dual --lib is the runnable
      gate for this crate.
      ALL OF THEM FIXED SAME DAY (2026-10-21):
      (1) CANDLE REAL BUG from the 2026-10-17 indexed_gemv port:
      repack_x86 try_indexed_gemv quantized the lhs BEFORE the
      select() guard, so k not divisible by 256 (expert-bias sizes
      like 64) hit BlockQ8K::from_float's assert - debug builds
      panicked, release computes over ZERO blocks (silent wrong
      results). Fix: shared indexed_supported() checked before
      every quantize_lhs (wrapper, _q head, fused gate/up). This
      was the root cause of the whole expert-bias/stacked-shard
      failure family (dual --lib 4: gguf uqff_shards +
      rank_three_concat + mxfp4 + hqq-3bit; core trio in
      gpt_oss/isq_flow) - dual --lib now 345/0, core 1541/0.
      (2) hqq 3-bit on GPU: to_dtype(I32) has NO GPU cast kernel
      (candle rejects i16/i32 on GPU backends) - the Three-bit
      closure now packs on CPU and returns to the origin device
      (subsumes the old metal-only special case); the hqq 4th
      failure (test_quantize_hqq on hip) green.
      (3) Test-target compile repairs: rotary dual_cuda_capture
      tests gated all(test, cuda); the two gguf/cpu grouped tests
      gated any(cuda,rocm); routed_lora_cuda_tests +
      routed_lora_bench gated all(cuda, not(rocm)) (API is
      solo-cuda-cfg'd) - they RUN again, 11/11 on the 3090;
      grouped_mmq_packed_cuda_tests PORTED off the dead pre-split
      API to the current NVIDIA-role names (grouped_*_cuda tensors
      + moe_dispatch_build_cuda exported from cuda_role) - 1/1 on
      the 3090 in dual, compiles in solo-cuda too;
      indexed_moe_lora_cuda_tests re-homed to solo-rocm (its API
      IS the gguf::cuda hip lane; as_hip_device/Storage::Hip
      idiom) - 3/3 on the iGPU; candle examples/ptx_probe gated
      to cuda (was breaking candle-core --all-targets in every
      non-cuda shape).
      (4) supported-models golden: the 2026-10-19 hand-added GGUF
      embedding note broke model_metadata byte-parity - the
      GENERATOR now emits it (survives regeneration).
      (5) embedding.rs fmt nit (from the feature commit) fixed in
      isolation; the 5 known-unformatted pre-existing files left
      alone. Bars after: all 4 shapes compile ALL targets +
      examples; default quant 303/0 (fp8 roundtrip fixed 2026-10-23:
      gated on feature=rocm, the only shape where scalar FP8 kernels
      are built); candle quantized_tests 55/0, candle-core
      --all-targets 0 errors.
- [x] ~M Dual NVIDIA-role device sampling (LANDED 2026-10-16,
      builds 15-18): device_sampling_supported dual arm, categorical
      packed twin, single-seq f32_packed twin, sampler single-seq
      gates widened; safe host fallbacks where twins don't exist
      (logprob top-k only now). Live verified: all sampling modes
      clean, 17x23/19x21 correct through the twins, temp-0 path
      unchanged. One-token lookahead twin ALSO LANDED (GraphStream-
      typed tail, per-combo wait/release, engagement counter).
      CLOSED 2026-10-16 (build 19, commit 384894552): resident top-k
      sampling submit twin (ops_cuda_role workspace/pinned-slots/
      token-ring + sample_ranked_topk, dual dispatchers for submit/
      wait/release/complete/cancel, sampler + batch wrapper role
      enums, host-fallback gate removed) - live verified on the
      qwen3.5-4b-cuda lane: 220tok temp-0.7/topk-40 coherent through
      the twin, invalid-token guard silent, journal clean. ENGAGEMENT
      TRAFFIC FOUND: the lookahead gate is device.is_cuda() = NVIDIA
      role ONLY in dual builds, so 4B-cuda decode traffic is what
      moves the counter. BENEFIT MEASURED (2026-10-16 A/B, build 18
      rollback vs build 19, same lane/prompts, warm runs): temp-0.7/
      topk-40 on qwen3.5-4b-cuda 88.7-89.2 tok/s (host fallback) ->
      111.2-112.0 tok/s (twin) = +25%; greedy control unchanged
      (~112) so sampled decode now matches in-graph argmax.
      ENGAGEMENT MAP: greedy argmax is sampled INSIDE the graph
      replay (no submission, zero counter movement); temp runs the
      lease chain with on-device top-k per step - the counter only
      counts chain (re)starts (~1 per 27 steps; steady-state chained
      steps are uncounted); logprobs requests move no counters (plan
      gate excludes device submit upstream, by design, 76 tok/s);
      concurrent pair at max_seqs=2 hits 199 tok/s combined;
      streaming == non-streaming. REMAINING: logprob top-k packed
      path still host-side on NVIDIA (untwinned
      cuda_topk_logits_packed_batched family); optional chained-step
      counter for per-step visibility LANDED (build 20, commit
      7c1b083f2): tail-path outcomes chained/chain_end/no_submit/
      role_mismatch/tail_unsupported on the same metric; accounting is
      now exact (a 220-tok temp request = 8 engaged + 210 chained +
      1 chain_end = 219/219 steps; greedy contributes ZERO - argmax
      is sampled inside the graph replay, never entering the submit
      machinery). PAIR-PARITY BENCH ANSWERED (2026-10-16, probe lane
      on the parked 4B-cuda VRAM, no prod outage; identical toml,
      Standard paged attention + Q8_0 KV pinned via
      MISTRALRS_FLASHINFER_DECODE=0, 3 warm runs each): dual 106.7/
      106.7/106.6 tok/s vs solo-cuda 112.2/112.1/112.2 tok/s = dual
      dispatch costs ~4.9% on sampled decode (better than the ~7%
      hypothesis); counter outcomes byte-identical between shapes
      (32/840/4, role cuda vs single) so the cost is per-step
      overhead, not behavioral. NEW FINDING: solo-cuda auto-selects
      FlashInfer decode for this model (dual deliberately excludes
      it on the NVIDIA role per flashinfer/mod.rs; FlashInfer
      rejects Q8_0 KV) - see the FlashInfer item below. LOGPROB
      TOP-K PACKED TWIN ALSO LANDED (2026-10-17, build 22, commit
      bb6aec9f7): ops_cuda_role twin for cuda_topk_logits_packed_
      batched + device route in the f32 wrapper; kernels were
      already in the nvcc companion, only the bf16/f16 ffi_cuda
      decls were missing. Path serves multi-choice batches
      (n>1/best_of>1 without response logprobs - response-logprob
      requests bail at cuda_batch_sampling_plan; single-seq
      logprob keeps the per-seq twin). Measured A/B on the
      qwen3.5-4b-cuda lane: n=2 request 76.9 tok/s (host fallback)
      -> 104.5-105.0 tok/s (twin) = +36%; single-seq logprob
      unchanged (~75-76); temp/greedy regressions clean (112.0/
      112.3). Also un-breaks the dflash caller on CUDA-role
      tensors in dual (previously refused loudly).
- [x] ~L FlashInfer decode on the NVIDIA role in DUAL builds (RE-SCOPED
      2026-10-16 after bench + plumbing audit; was ~M "maybe near-free").
      LANDED 2026-10-24: the gate is now DEVICE-KEYED, not
      build-shape-keyed, and the whole FlashInfer + FA3 paged
      surface runs on the cuda role in dual builds.
      Runtime gate: FlashInferAttentionBackend::supports_layer lifts
      to `feature = "cuda"` (compiled availability only); the
      serving-device decision lives in the new DeviceGatedConfig
      wrapper (paged_attention/config.rs) applied at
      build_normal_pipeline/build_multimodal_engine_pipeline where
      the device is in scope - attention_backend_kind(_for_layer)
      and kv_cache_layout(_for_layer) downgrade to Standard on
      non-cuda devices. Layering: KvQuantGated(DeviceGated(base))
      so KV-dtype and device gates compose.
      Port surface (mistralrs-paged-attn): backend/flashinfer.rs
      (873 ln) and backend/fa3.rs launch layers twinned onto
      cuda_backend/ffi_cuda via scratch/gen_twin.py (checked in
      spirit: rules live in scratch, rerun after backend edits);
      cuda_role/{flashinfer,fa3}.rs; ffi_cuda.rs gains
      cuda_flashinfer_* and cuda_fa3_* entries (dlopen from the
      nvcc companion); fa3_types.rs hoists the pure FA3 planning
      types so solo/dual share one type identity; role_dispatch.rs
      gains flashinfer_decode_auto, {reshape_and_cache,
      gather_kv_cache_flashinfer}_auto and
      fa3_{fp8_decode,prepare_paged_metadata,
      prepare_decode_metadata}_auto (non-cuda storage bails).
      attention_backend::from_cache lifts to all(cuda, unix).
      FA3 kernels: fa3_decode_api.cu extern-C APIs now carry
      visibility("default") (the -fvisibility=hidden build would
      hide them from dlsym); build_cuda_companion builds the fa3
      archive + libmistralrsfa3paged_cuda.so ONLY for cap == 90
      and prints MISTRALRS_FA3_CUDA_PLUGIN + has_fa3_fp8_paged;
      fa3_cuda_plugin.rs resolves it at runtime (env then
      exe-relative, no bake-time path since non-SM90 boxes never
      build it). Compile-verified by force-printing the cfg on
      this SM86 box (no SM90 hardware to run it; first H100-class
      card lights it up automatically).
      mistralrs-core: 136 cfg sites lifted from
      all(cuda, not(rocm), unix) to all(cuda, unix) across
      flashinfer/{mod,metadata}.rs, plan.rs, layers/
      paged_attention.rs, cache_engine.rs, inputs_processor.rs,
      cuda_graph.rs (prepare_fa3_decode_schedules), attention/
      mod.rs (sliding_window_left); dual import arms alias the
      _auto dispatchers. cuda_supports_fp8 now runs the real
      cap>=8 check on the cuda role in dual (early-true is hip
      only); cuda_sm_count probes the NVIDIA role in dual.
      Tests: dual paged-attn suite green for the first time (was
      test-compile-broken: twin begin_capture arity rot, fixed);
      backend dense tests made role-aware via role_test_device()
      (new_hip in dual, new_cuda in solo). Twin flashinfer kernel
      tests pass ON the 3090 in the dual build (f16/bf16+fp8 HND
      write/gather/decode parity).
      PROBE (dual fast binary, one 4B per device, auto KV):
      cuda:0 -> "Paged attention backend: FlashInfer", hip:0 ->
      "Standard" in ONE process - the device-key A/B holds.
      Ported this slice: MLA paged (mla.rs 897 + context_attention_mla.rs
      435) and flash_attn_sinks (385) now twin on the NVIDIA role in
      dual builds (cuda_role + role_dispatch *_auto + ffi_cuda entries);
      mistralrs-core lifts the cfg gates and imports the _auto aliases
      in dual. DFlash/dflash stays not(rocm).
      BENCH context (solo-cuda, 4B): FlashInfer+BF16 KV 117.5
      vs Standard+Q8_0 112.1 = +4.9% decode; KV cost doubles per
      token vs Q8_0. Prod stays Standard by config
      ([paged_attn] cache_type = Q8_0 gates it off); flip a cuda
      lane to auto/BF16 KV to activate. Payoff is future 3090
      residents (27B+) and closing the solo-vs-dual gap.
- [x] ~S UQFF vs GGUF shootout: same weights through the CLI
      `quantize`/`uqff` path, bench load + decode on this box.
      Unmeasured; native format may win both.
      CLOSED 2026-10-25 (probe :1245, release binary, 4B Qwen3.5-4B
      Q8_0 GGUF source, no MTP, temp=0, 3 warm runs each, single
      listener verified per lane). VERDICT: container-neutral at
      equal quant (q8_0 GGUF vs UQFF: decode 43.6 vs 43.6 tok/s,
      load 6.0s both) - no reason to convert native-GGUF fleet
      lanes. UQFF WINS THE ISQ WORKFLOW decisively: pre-baked
      UQFF q4k loads 6.2s vs 10.0s for GGUF + runtime --isq q4k
      (conversion cost skipped per load; scales up with model
      size), decodes 56.7 vs 53.8 tok/s (+5.4%) and prefills
      1108 vs 1008 tok/s (+10%) - same quant plan (isq_flow, q6k
      sensitive-tensor exceptions in both), 3 tight runs each,
      mechanism unproven (candidates: serialize round-trip
      layout/packing, kernel-path selection); honest caveat:
      single boot per lane. SIDE FINDING: q4k beats q8_0 by ~30%
      decode on the 4B GDN-hybrid (56.7 vs 43.6) - OPPOSITE of
      the 27B dense Q4 finding; if a faster 4B lane is ever
      wanted, pre-baked UQFF q4k is the artifact, not GGUF+ISQ.
      Artifacts + logs + bench script preserved in scratch/
      uqff-shootout/ (7GB, delete freely).
- [x] ~S MTP on the 4B: `Qwen3.5-4B-MTP-GGUF` carries the head but is
      pinned `mtp=false` in prod. Small-model MTP is usually
      high-acceptance; bench before pinning.
      LANDED 2026-10-17 (commit `02bfd9c21`, build 23, md5
      b9a950e77bf96acc225266824ca1f657): built-in MTP for the
      text-only Qwen3.5 path. The multimodal wrapper's machinery
      moved onto Qwen3_5TextModel (mtp_n_predict/draft_lm_head/
      pending tails + drafter forward/prefill/propose/prefill/
      commit), full SpeculativeTargetMixin impl for the text model
      (wrapper delegates, keeps DFlash-only branches), TextConfig
      gains the injected _mistralrs_mtp flag, the text loader
      passes cfg.mtp through, GGUF config synthesis maps
      nextn_predict_layers -> mtp_num_hidden_layers, and the hybrid
      cache config appends the MTP layer (qwen3_next pattern; KV
      +32MB per 16k ctx = exactly one layer, ~500MB lane total
      with head + Q4K draft lm_head).
      MEASURED (release, same-binary A/B, essay 500 tok):
      3090 lane 105.4 -> 149.7-150.6 tok/s (+42%), accept 54-59%
      len ~2.1; hip iGPU 40.9 -> 50.7-54.9 (+24-34%), accept
      53-63%. Prod live after the flip: cuda lane 151.3 tok/s
      (vs 112 pre-flip, +34%), accept 56.7-64%, 3090 at 20.5GB
      (3.5GB headroom), journal clean, hip lane attaches and
      decodes healthy. n=2 pinned per-model (global is 3).
      CORRECTNESS NOTE for future validations: greedy decode in
      this stack is only approximately deterministic - no-MTP runs
      flip near-tie tokens run-to-run (verified: same binary,
      same payload, char-135 "Goal:" vs "Identify" flips both
      with and without MTP). Token-exact greedy equivalence is
      NOT a usable gate; use accept-rate + coherence + journal.
      ALSO: hip-device probes must source .rocm-env.sh in the
      probe env (hipblas dlopen), the cuda-only probes never
      needed it. Follow-up ANSWERED 2026-10-17: qwen3-16b
      (Qwen3-16B-A3B) has NO MTP head - arch qwen3moe, 48 uniform
      12-tensor blocks, zero nextn.* tensors and no nextn_predict_
      layers metadata (the Qwen3 MoE line predates MTP heads).
      Its mtp=false pin is correct as-is; a hypothetical head would
      also need a qwen3_moe.rs wiring port (that class takes the
      default mixin bail).
      ANSWERED 2026-10-13 (cuda-alone probe, solo-cuda binary):
      the pin is LOAD-BEARING, not a choice - the Qwen3.5 model
      class never overrides `SpeculativeTargetMixin::attach_speculative`
      (only Qwen3.6/3.8 classes do), and with the runtime
      `[runtime] mtp = true` global, an unpinned 4B FAILS ITS LOAD
      outright ("This model does not support speculative
      decoding"). Whether this particular Q8_0 file even carries
      the head tensors is moot until the class gets the wiring.
      Re-scoped to ~M: port the attach mixin + drafter wiring to
      the Qwen3.5 class (4B + 16B would both benefit), then bench.
      Baseline recorded while probing: 4B solo-cuda no-MTP essay
      125.21 tok/s (thinking-trace decode, F16 KV, 2k-budget
      sanity 391 CORRECT).
      VERIFIED 2026-10-17 (file + routing audit, gguf-dump):
      the Q8_0 FILE IS MTP-COMPLETE. qwen35.nextn_predict_layers=1;
      blk.32.nextn.{eh_proj[5120x2560],enorm,hnorm,shared_head_
      norm} on the last block, which per the llama.cpp convention
      IS the MTP layer body (block_count=33 = 32 main + 1 MTP;
      tied embeddings, shared head). Same convention as the 35B
      (qwen35moe: block_count=41 = 40 main + 1 MTP at blk.40 with
      the same 4 nextn tensors). CLASS ROUTING: arch qwen35moe
      (35B) -> models/qwen3_next.rs (full attach_speculative
      override; MTP live in prod, accept ~57-74% at n_predict=2);
      arch qwen35 text-only (4B) -> vision_models/qwen3_5/text.rs
      Qwen3_5TextModel (recurrent-state overrides only, attach =
      default bail). KEY FINDING: the Qwen3.5 family ALREADY HAS
      the MTP wiring written for its multimodal wrapper
      (vision_models/qwen3_5/speculative.rs, Qwen3_5Model) - the
      4B item is exposing/reusing that in-tree machinery for the
      text-only path, not a port from qwen3_next. No cross-model
      draft mechanism exists anywhere in the stack (no draft_model
      config; mtp_draft_* keys configure the draft HEAD's ISQ), so
      "4B as draft for larger models" is not a thing here - each
      MTP model carries its own head.
- [x] ~M FA2 path proof on the cuda role (compute 8.6: FA v2 yes, v3
      no): confirm the kernels actually dispatch there; try the
      `cudnn` feature as challenger. Pairs with the cuda-MTP port.
      CLOSED 2026-10-17 by the flash-attn landing (build 51fa55078):
      role-agnostic FA puts the CUTLASS FA2 kernels on the 3090 and
      they dispatch live - 4B decode 151.3 -> 195.1 tok/s (+29%,
      accept 80%), 27B dense prefill 1104 tok/s, prefill gains
      confirmed on every cuda-role probe. The `cudnn` challenger stays
      unexplored (optional, not blocking anything).
- [ ] ~L Split prefill/decode across the two GPUs (prefill-hungry on
      one device, latency decode on the other). Only if single-device
      saturates; most complex item here.
- [ ] ~M Utilization-aware device auto-select (opened 2026-10-09,
      follow-up to bare cuda/hip first-available): resolve bare specs
      (and eventually auto-select) to the least-loaded device instead
      of lowest ordinal. Needs a live cross-device memory/load policy
      consulted at resolve time (free VRAM/GTT per role? engine load?)
      plus answers to what "utilized" means for a lazy entry that has
      not loaded yet, and how it interacts with the demand-load
      router's own placement. Deliberately deferred: first-available
      is deterministic and debuggable; surprising placement is worse
      than predictable placement until the policy questions are settled.
- [x] ~M GGUF support for embedding models (opened 2026-10-09, found while
      wiring open-webui RAG to mistral.rs): the embedding pipeline
      (mistralrs-core/src/pipeline/embedding.rs, EmbeddingGemma +
      Qwen3Embedding loaders) consumes safetensors only
      (from_mmaped_safetensors/MmapedSafetensors), so the GGUF-centric
      fleet cannot serve quantized embedders (unsloth
      Qwen3-Embedding-0.6B-GGUF, nomic-embed, bge-m3 GGUF) that
      llama-server --embeddings handles routinely. Needs a GGUF loader
      arm (MmapedGguf + per-block dequant at forward or ISQ-at-load)
      plus format plumbing so `[[models]] kind = "embedding"` accepts
      `format.quantized_file`. Interim safetensors path staged:
      Qwen/Qwen3-Embedding-0.6B pulled to models/Qwen/ (1.2GB, 1024-dim,
      32k ctx, sentence_transformers layout auto-detector expects).
      Probe-footgun resolved same night (not a mistral.rs bug): the
      "silent exit 1" was the probe shell sourcing .rocm-env.sh with
      LOCALAI_ROOT unset - its guard prints to stderr and exit 1s the
      whole shell before the binary ever ran (a 2>/dev/null on the
      source swallowed the message). Export LOCALAI_ROOT before
      sourcing. SHIPPED 2026-10-09: probe re-ran clean with correct
      env (1024-dim, deterministic, batched, cosine 0.68/0.24),
      Engine::new unreachable!() fix cut to prod, RAG live in prod
      (webui rows set, bypass flipped by user).
      GGUF ARM SHIPPED 2026-10-19 (native-lane design, not the
      MmapedGguf sketch above): `kind = "embedding"` + `format.
      quantized_file` threads ModelSelected::Embedding through a new
      GGUF arm in the embedding pipeline that reuses the fork's
      native GGUF machinery (GgufWeightSource + GgufBindingMap, NOT
      legacy gguf pipelines): arch mapping qwen3/gemma3 -> the two
      embedding loaders, config synthesized from GGUF metadata
      (reuse synthesize_normal_config for qwen3, prepare_gemma3_
      text_config + architectures/use_bidirectional_attention
      stamps for gemma3), bindings = normal tables with the `model.`
      prefix stripped, pooling from `{arch}.pooling_type` (llama.cpp
      value space; mean/last only; defaults last for qwen3, mean for
      embeddinggemma) + Normalize when modules.json is absent,
      embedded-GGUF tokenizer with `tokenizer.ggml.add_eos_token`
      honored via an EOS-append post-processor (llama.cpp embed
      semantics; REQUIRED for Qwen3-Embedding last-token pooling
      parity - without it cosines to the safetensors lane sit at
      0.73-0.87, with it 0.996+). Bails loudly on ISQ-on-GGUF,
      UQFF/imatrix, topology, tensor parallelism. GGML still
      rejected; mmproj rejected. Validated end-to-end on :1244
      (CPU fast build): official Qwen/Qwen3-Embedding-0.6B-GGUF
      Q8_0 (bare repo, fully synthesized path) vs safetensors
      baseline: per-input cosine 0.996-0.997, retrieval ranking
      identical; synth-GGUF E2E unit test + 7 component tests in
      tree; 4 shapes checked. EmbeddingGemma caveat: its ST layout
      has 2 trained Dense heads NOT in the GGUF - full parity
      needs model_id at the ST repo (modules.json + Dense
      safetensors load from there), bare-GGUF matches llama.cpp
      --embeddings instead (documented in use-embeddings.mdx).
      FOLLOW-UPS 2026-10-19: (1) EmbeddingModelBuilder::
      with_quantized_file threads the GGUF arm through the Rust SDK
      (Vec filenames -> ModelSelected join ';', reload path included,
      2 builder tests); (2) mistralrs-pyo3 compiles in the default
      shape again - the 2026-10-18 expert_device/expert_hot_cache_mb
      fields never reached the pyo3 GGUFSpecificConfig site; the fix
      exposed a latent UPSTREAM test bug (48be6c1e2: the MTP
      reservation test's tempdir ships no config.json while its own
      peek_config requires one; pyo3 tests are not in upstream CI) -
      the test now writes a non-DFlash config.json. pyo3 9/9,
      mistralrs 17/17, cli 152/152; 4 shapes checked; docs updated.
      PROD ADOPTION 2026-10-21: the embedding lane flipped safetensors
      -> GGUF Q8_0 (same model_id dir; tokenizer/config/modules.json
      still from the ST repo so pooling is unchanged; the safetensors
      file stays on disk, unused). Measured before the flip on the
      probe lane with the EXACT prod layout: per-input cosine vs
      safetensors 0.993-0.998 (one short-input 0.9928 outlier; the
      broader 32-input retrieval set stayed >=0.996), designed-
      relevant docs at identical top-1/rank on all 8 test queries
      (only irrelevant-tail near-ties shuffled, margins ~0.004),
      embeds ~3.3x faster (singles ~110ms -> ~25ms on the CPU probe).
      INCIDENT (config-before-binary): the first flip attempt hit the
      still-running Oct 18 binary (pre-feature) - its dry-run gate
      rejects quantized_file on embedding ("Embedding models do not
      support GGUF or GGML format") and the unit crash-looped 28x
      (~4 min outage, zero traffic, restored in 1s on revert; commit
      6f37d03a3). LESSON: ship the binary BEFORE the config. Fixed
      order: release build (cuda,rocm,flash-attn, 16m35s, md5
      242c4c5f, old binary backstopped as mistralrs.pre-gguf-emb),
      dry-run validated against the flipped toml, then stop/swap/
      flip/start. LIVE: demand-load via the reload machinery
      ("reloaded successfully" - also end-to-end-validates the
      ModelSelected::Embedding quantized_filename threading), first
      request 803ms incl. load, singles ~22ms on hip:0, GTT delta
      +168MB at load, prod-vs-probe cosines 0.995-0.998 (cross
      device), prod-vs-safetensors-baseline min 0.998, dmesg +
      journal clean. NOTE: hip normalize yields norms 1.000 +/-0.005
      (CPU lanes are exact 1.0) - device kernel property, harmless
      for cosine retrieval; whether the old safetensors-on-hip lane
      was exactly 1.0 was never measured, so this may be pre-existing.
- [x] ~M Reranker lane + /v1/rerank endpoint (opened 2026-10-19, closed
      2026-10-24). Qwen3-Reranker-0.6B first cut, GGUF-only lane mirroring
      the embedding lane. A/B vs llama-server --reranking on the probe lane
      (same ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF artifact): ranking parity
      2/2 cases, scores within ~0.002 (mistral.rs 0.9922/0.9844/0.0674/0.0293
      vs llama.cpp 0.9941/0.9836/0.0698/0.0316). /v1/embeddings refuses
      reranker checkpoints (400 "incompatible for this model's category");
      /v1/chat/completions and /v1/completions refused the same way.
      GGUF metadata gotcha (caught live): llama.cpp writes the classifier
      labels under the ARCH-PREFIXED key `qwen3.classifier.output_labels`
      (and `qwen3.pooling_type` for RANK), not the unprefixed
      `classifier.output_labels` the spec assumed - same convention as
      pooling_type. Fixed in mistralrs-quant/src/gguf/embedding.rs.
      Template gotcha (caught live): the official baked
      `tokenizer.chat_template.rerank` carries an EMPTY think block
      (`assistant\n<think>\n\n\n\n</think>\n\n\n`); the first cut
      omitted it and scored 0.053 vs 0.994. format_rerank_prompt in
      mistralrs-core/src/request.rs now reproduces the artifact byte-for-byte
      (unit test `rerank_prompt_matches_official_gguf_template` locks it).
      Synthetic GGUF e2e test (loads_qwen3_reranker_gguf_and_scores) +
      refusal test + score-head binding test all pass; 4 shapes clean.
      LATER CUTS (unchanged): 4B/8B rerankers, bge-reranker (XLMRoberta
      cross-encoder, true classification head, different loader), custom
      instruction field (llama.cpp lacks it too; Qwen eval says +1-5%).
      PROD LIVE TEST 2026-10-25 (against :1235, build f3b2d3576): all
      green. Lazy demand-load fired on the first request ("reloaded
      successfully", journal line names labels ["yes","no"]); first call
      907ms incl. load, warm 4-doc rerank 71ms (~18ms/doc on hip:0).
      Jina-style plain-string docs AND Cohere-style {"text":...} objects
      both accepted (openwebui's reranking engine sends the Jina shape);
      top_n truncation works; empty documents returns the structured
      400 (invalid_request_error/param/code) in <0.1ms. Rankings
      semantically correct, score family matches the probe-lane A/B
      (relevant ~0.99, irrelevant ~0.03). Zero journal errors; main
      lanes unaffected. USER-SIDE FLIP DONE 2026-10-25 (user
      configured the openwebui Reranking Engine slot; connection
      verified from their side). Lane fully live end to end; first
      openwebui-originated rerank traffic will appear on the first
      RAG query against a rerank-enabled collection.
      TEI-SHAPE COMPAT (found 2026-10-26, FIXED same day):
      llama.cpp's /v1/rerank accepts the documents array under TWO
      keys - `texts` (TEI, Text Embeddings Inference shape) or
      `documents` (Jina); ours took `documents` only. FIXED:
      RerankRequest now takes `texts` as a serde alias for
      `documents` (plus `truncate` as an alias for
      truncate_sequence), gained a `return_text` flag, and
      RerankResult carries an optional `text` echo
      (skip_serializing_if None - byte-identical responses for
      existing clients). Two serde unit tests lock the shapes;
      docs/openapi.json regenerated (staleness gate green);
      use-rerankers.mdx documents the aliases; crate fmt clean
      (incl. a leftover import wrap from the rerank lane commit
      itself). openwebui unaffected (Jina sender). Prod serves
      the change after the next binary swap - the running
      f3b2d3576 binary predates it; no config change needed.
- [ ] ~L qwen4exp model-arch port: serve Qwen3.8-Flash-Next (opened
      2026-10-22, expanded 2026-10-24). 95 GB GGUF on disk, lazy alias
      `qwen3.8-flash-next` already registered. Loader gate for sharded
      GGUFs PASSED 2026-10-25 - multi-part reads already work end to
      end (see item 7), so the port starts at the arch registry. Then
      QSA sparse attention, HyperConnections, PLE.
      2026-10-21, user demand: the orcarouter uncensored GGUF is
      already on disk - 95GB on a 98%-full disk carrying it until
      this lands - and currently unloadable). EVIDENCE: shard-1 GGUF
      header parsed directly - general.architecture=qwen4exp, GGUF
      v3, 317 tensors across the split; normal_registry.rs
      (CanonicalGgufArchitecture, 26 archs, tops out at
      qwen3next/qwen35/qwen35moe) rejects it with
      UnknownArchitecture on demand-load (lazy alias fails clean
      per-request, fleet unaffected); repo-wide "qwen4" grep = 0
      hits; upstream EricLBuehler master ALSO lacks it (models/
      ends at qwen3_next.rs, checked 2026-10-21) - no rebase
      shortcut, the port is ours. ASSETS IN PLACE:
      models/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF/ -
      IQ4_XS-MTP 3-shard split + mmproj-F16, sizes byte-verified
      vs repo metadata; lazy alias qwen3.8-flash-next registered in
      mistralrs.toml (Q8_0 KV, mtp_n_predict=2 placeholder
      mirroring the 27B, sweep after port; rode into commit
      fd4a9a829). FAMILY PRECEDENT: third iteration of the
      qwen3next -> qwen35moe hybrid pattern. REUSABLE (proven
      in-tree): GDN kernels (mistralrs-core/src/gdn/ + packed
      hipcc GDN build), hybrid LayerImpl::LinearAttention
      (GatedDeltaNet) wiring (models/qwen3_next.rs,
      vision_models/qwen3_5_moe/text.rs), stacked-expert MoE,
      MTP attach lanes (live on 35B/27B/4B), mmproj family
      registry (multimodal_config.rs). NEW PIECES: (1) registry
      entry + builder; (2) QSA micro-block sparse attention; (3)
      HyperConnections replacing layer norms; (4) PLE n-gram hash
      embeddings - per_layer_token_embd [160, 320001536], 51.2B
      params, quant axis 160 NOT 256-divisible so the tensor stays
      Q8_0 even in K-quant builds (loader must take the mixed
      Q8_0 tensor; same quirk is why Q6_K/Q8_0 splits carry a
      ~54GB PLE shard over CloudFront's 50GB single-request cap);
      (5) MTP: self-speculation path for -MTP files, or pair a
      plain quant with the repo's separate MTP-draft.gguf (~4GB)
      via the existing spec-draft-model machinery; (6) mmproj
      family entry for vision; (7) multi-part split support:
      RESOLVED 2026-10-25 - the loader ALREADY handled sharded GGUFs
      end to end (mistral.rs's multi-reader Content wrapper,
      gguf/content.rs: validates split.count against the reader set,
      merges metadata, retrieves tensors across shards). Live-proven:
      a 4-shard llama-gguf-split of the 4B Q8_0 (441 tensors) serves
      CORRECT 1776 through BOTH the CLI serve path and the
      from-config path. AUTO-EXPANSION LANDED same day (user
      request): naming ANY ONE shard in quantized_file now expands
      to the full sibling set at get_model_paths time (gguf::
      parse_split_gguf_shard + paths::expand_split_gguf_names;
      repo dir scan via list_repo_files, loud error when the
      sibling set is incomplete, explicit lists dedupe, empty
      listing falls through to the old explicit behavior). The
      prod flash-next alias uses the single-shard spelling.
      REFS: llama.cpp #27742 (arch, in
      mainline since 2026-08-27) + #27880 follow-up; #27836 +
      #30592 (MTP self-speculation; ~0.77 draft acceptance on
      this IQ4_XS-MTP file per the orcarouter README). SERVING
      SHAPE: 512 experts, 10+1 active/token - decode is cheap but
      all 95GB weights must sit in RAM/VRAM or be mmap'd, so the
      natural lane is the expert-split + HotExpertCache infra
      (18800233e, fa996df07) proven on the 35B. VALIDATION: probe
      lane first (policy), output-parity A/B vs llama.cpp master
      (which supports it today), then release probe + the single
      prod restart batch rule.
- [ ] ~L multi-backend runtime: CPU + NVIDIA + AMD + Intel in ONE
      process, per-model device assignment. Design agreed 2026-09-26;
      UPDATE 2026-10-01: an RTX 3090 eGPU (sm86, 24GB, CUDA 13.3 UMD) is
      now attached - the cuda role has real hardware for the first time,
      so dual-build two-vendor validation is unblocked (serving stays on
      the AMD role by doctrine; auto-selection confirms Hip as the serving
      device with the 3090 present). UPDATE 2026-10-06: dual two-vendor
      serving proven end-to-end (4B across cpu/rocm/cuda and 27B
      cuda/rocm ISQ-Q4K, journaled with benches); cudarc-nvidia vendored
      in the candle repo like cudarc-hip; decode graphs live on both
      roles. Remaining for this item: Intel/Vulkan roles and the
      device-keyed FlashInfer gate below. Known refinement needed before the
      3090 can claim FlashInfer in a dual build: the supports_layer gate
      is BUILD-shape-keyed (all(cuda, not(rocm)), commit 94e1171c5) not
      device-keyed - correct for the hip serving role, suboptimal if a
      model is ever served on the 3090 via a dual build. FA2 buildable
      (flash-attn feature), FA3 not (Hopper-only).
      precedent = the ggml/llama.cpp architecture (separate backend
      registration incl. a fully independent Vulkan backend that
      covers all three GPU vendors; their "ROCm backend is the CUDA
      implementation compiled under HIP" mirrors our either/or
      features today). Stages after the spike:
      (1) candle Device/Storage re-architecture to true per-backend
      variants - the big one; Device::Cuda currently means "the one
      compiled GPU backend" and hundreds of match sites assume it.
      (2) Vulkan backend for candle (llama.cpp's Vulkan shaders prove
      LLM inference on all vendors; Intel arrives via Vulkan; a native
      SYCL/Level-Zero path is a later option).
      (3) BackendCaps per device - MANDATORY CONTRACT: paged
      attention, decode-graph capture, mmq/mmvq, MTP are retained on
      every device that supports them; per-device defaults degrade
      gracefully (a Vulkan-served model simply has no graph-capture
      decode) via caps-keyed dispatch, never silently. The GpuArch
      tables are the mechanism: each arch arm encodes what THAT arch
      measured best as.
      (4) Per-model toml `device = "cuda:0" | "hip:0" |
      "vulkan:1" | "cpu"` + per-device memory budgets/eviction in the
      router (the estimator currently assumes a single GPU).
      Constraints: one model = one backend (cross-vendor P2P absent,
      host-bounce copies; nobody does it, we won't). The CUDA-shaped
      API surface (CudaDevice, Device::Cuda) is NOT renamed: HIP is
      CUDA-shaped by design and renaming means forking away from
      candle upstream (llama.cpp accepts the same shape).
      VULKAN IS NOT "THE INTEL BACKEND" (user directive 2026-09-26):
      Vulkan is a vendor-agnostic API - the same physical card is
      reachable as hip:0 AND vulkan:0 (or cuda:0 and vulkan:0), so
      backends are keyed by API, not vendor, caps are per (API,
      device) - the same card via two APIs has different caps -
      and enumeration is per-API. GpuArch::Vulkan resolves the
      underlying vendor from VkPhysicalDeviceProperties (vendorID
      0x10DE/0x1002/0x8086 + deviceName parse) so Vulkan policy
      code can apply per-vendor quirks (ggml-vulkan precedent).
      Selection precedence: native API by default (cuda/hip),
      Vulkan when explicitly chosen or on Intel-only boxes.
      DESIGN NOTE 2026-10-02 (dispatch vs selection; vendor equality):
      vendors stay first-class - the AMD-primary state of dual builds is
      porting-history accident (the fused stack is hipcc-only today), not
      doctrine, and the role-keying work is what makes parity incremental
      (each new NVIDIA kernel flips one gate). Two layers, not one: (1)
      DISPATCH asks "can this run here?" - capability/transport-keyed,
      vendor-blind where possible, degrades loudly. This is
      device_has_fused_ops() (single GPU role in single-vendor; AMD
      role only in dual) + BackendCaps. "AMD-only in dual" means exactly
      this: the fused launchers exist solely as hipcc binaries, so only
      Device::Hip tensors may enter them. (2) SELECTION asks "what runs
      fastest here?" - irreducibly vendor+arch+SDK-aware: wavefront 64
      vs warp 32 vs Intel subgroups, MFMA vs Tensor Cores vs XMX, LDS
      sizes, rocBLAS/hipBLASLt vs cuBLAS/cuBLASLt vs oneMKL heuristics,
      CK vs FlashInfer vs eager (three maths, same answer), per-SDK bugs
      and tuning knobs. GpuArch (vendor+arch+transport) feeds selection;
      the predicate feeds dispatch. REJECTED: is_rocm()/is_vulkan()
      Device methods - in rocm-only builds Device::Cuda IS the AMD
      device, so any vendor-named method has shape-dependent semantics
      (the is_cuda() confusion with flipped polarity); vendor questions
      go to GpuArch/cfg, role questions to the predicate. Vulkan fits
      this split: transport (SPIR-V, portable) feeds dispatch, while
      VkPhysicalDeviceProperties vendorID (0x10DE/0x1002/0x8086) feeds
      per-vendor selection quirks and tuning. Capabilities decide
      whether, vendors decide how fast.
      SLICE PLAN (code survey 2026-09-26): candle's
      DeviceLocation/Device enums are the precedented extension
      point (Metal already coexists); cuda_backend is ONE module
      bound to either vendor via the extern-crate alias (it already
      dual-compiles) -> macro-instantiate it per vendor rather than
      duplicating; candle-kernels build.rs has BOTH paths (CORRECTED
      2026-09-26: rocm feature -> hand-rolled hipcc; else ->
      cudaforge, a crates.io nvcc wrapper - "nvcc" never appears
      literally in build.rs, the initial grep read was wrong), but a
      dual-GPU build still needs the kernels crate instantiated
      twice (candle-kernels-cuda twin keeping the cudaforge branch;
      today's build.rs takes the rocm branch on feature precedence,
      so a both-features build would silently produce HIP kernels
      only). Baseline candle ops are RUNTIME-loaded modules (PTX on
      CUDA, cubin on HIP) -> no link-time collision for the
      baseline; the only link-time collision is the static
      libmoe.a FFI archive (mmq/mmvq/moe fast paths, same C symbols
      from both vendors) - and BackendCaps makes it a non-blocker:
      a dual-build CUDA side ships with mmq/moe caps false and
      serves via dequant GEMM until symbol prefixing lands.
      NVIDIA side init is probe-gated (is_culib_present, one
      guarded attempt, never probe->init cycles). Slice order: (S1)
      dual-GPU candle compile - lift the compile_error, macro-
      instantiate the backend module, kernels twin crate (PTX
      only, no static FFI), Device::Hip + DeviceLocation::Hip arms,
      alloc/copy ops verified on hip on this box with the cuda side
      probe-gated; (S2) full candle GPU
      test suite through the dual build; (S3) mistral.rs routing:
      device strings + BackendCaps contract + graceful per-device
      degradation, serving smoke on hip:0; (S4) Vulkan backend
      (largest; llama.cpp shader precedent), GpuArch::Vulkan
      vendor-detect, vulkan caps floor (no decode graphs initially).
      mistral.rs fast paths (mmq/gdn/paged-attn/cuda-graph) meet
      Device::Hip in S3 - trait or caps-carried dispatch, design
      deferred to S3.
      S3a DONE 2026-09-26 (the dual-compile leg): the ENTIRE
      mistral.rs stack compiles under cuda+rocm - mistralrs-quant
      (launcher modules fast_mmq/mmvq/cuda replaced by *_hip.rs
      stand-ins: pure policy mirrored, launcher entries bail until
      S2's QStorage-on-hip; gemv + cublaslt gated with bail stubs;
      gemv's plain launchers cfg'd not(dual) since the fork FFI
      stream args differ), mistralrs-paged-attn (role alias,
      hip_fwd CustomOp twin, as_role_storage conversions, per-role
      cuda_fwd_t/workspace_ensure twins), mistralrs-core (FA3/
      FlashInfer/MLA/sinks NVIDIA-only paths tightened to
      all(cuda, not(rocm)) - paged-attn exports those symbols
      not(rocm) only; cuda_graph decode-graph stack fully
      role-bound with per-vendor sys/stream imports and the arena
      path widened from all(rocm, not(cuda)) to rocm; gdn.rs
      role-bound with Storage-arm twins, role_storage() helper, 2
      trait-method hip_fwd twins, resolve/resolve_hip split;
      device_map/debug/memory_usage/normal/multimodal/execution/
      sampling/ops Hip arms or role bindings), and mistralrs-cli
      green in all three shapes (dual, rocm, default).
      VERIFICATION: full regression bar green - quant 317/317
      (fp8 roundtrip fixed 2026-10-23, gated on feature=rocm),
      fast_mmq 5/5, gdn 55/55, cuda_graph 31+8,
      paged-attn 8/8, mmq bench, candle default/rocm/dual checks
      zero-error with dual tests 5/5, clippy parity on every
      changed crate (core 3=3, quant 5=5, paged-attn 2<=3).
      STRUCTURAL LESSONS: (1) cfg(feature="cuda") in mistral.rs
      means NVIDIA-ORIGINAL (upstream legacy) - under dual those
      paths reference not(rocm)-exported symbols and must tighten
      to all(cuda, not(rocm)); (2) all(rocm, not(cuda)) single-
      vendor guards break dual (the arena path) - serving-truth
      guards are plain feature="rocm"; (3) cfg CANNOT splice
      Storage enum variants into expressions - use a
      role_storage() helper fn per file, cfg-paired arms for
      matches, let-pairs for destructures; (4) rename trait-method
      cuda_fwd -> hip_fwd ONLY for fns inside impl blocks - free
      helper fns named cuda_fwd keep their names (an over-broad
      rename broke 16 call sites); (5) script hygiene: paren
      counts are 1-for-1 when swapping fn names in calls
      (role_storage( vs Storage::Cuda(), keep the closers), and
      destructure conversions must re-emit else-bodies.
      S3b DONE 2026-09-26 (candle + mistral.rs): device strings
      + BackendCaps + auto-selection, all probe-gated.
      CANDLE: cudarc dep swapped dynamic-linking -> dynamic-loading
      (dual binaries start with no NVIDIA runtime - the load-time
      libcuda DT_NEEDED is gone); cudarc-hip gains a link-mode
      is_culib_present() (true: a running process implies the
      linked runtime) so callers gate uniformly across binding
      modes; the dual test's NVIDIA arm is now the probe-gated
      pattern.
      MISTRALRS: mistralrs-core/device_spec.rs - DeviceSpec
      parse/serde (string form "cpu"|"cuda:N"|"hip:N", vulkan:N
      reports the S4 status), resolve() probe-gated per build
      shape (dual cuda:N without a driver refuses, naming hip:N),
      auto_select() (probe the serving role, log the choice; dual
      + NVIDIA-only machine refuses honestly instead of drowning
      on CPU), BackendCaps {arch, graphs, paged_attn,
      quantized_gguf, mtp} with a per-device startup log line and
      ensure_gguf() - the S2 boundary is a loud refusal at load
      entry. Wiring: --device + global toml device + PER-MODEL
      [[models]] device (untagged ModelDevice enum accepting the
      flat string OR the legacy [models.device] sub-table),
      init_device takes the spec, per-model resolution in
      multi-model, device-mapper pretty-prints hip[N]. Sweep:
      every remaining pure cfg(feature="cuda") module in
      mistralrs-quant (lora/dynamic, blockwise_fp8, vector/scalar
      fp8, gptq marlin, cublaslt users) + core get_dtypes
      (nvidia-smi probe!) tightened to all(cuda, not(rocm)) with
      dual fallbacks - those kernels are cuda-block machinery
      with no hipcc build. BUILD-SCRIPT LESSON: ar crs only
      replaces same-named members - switching feature shapes in
      one OUT_DIR leaks the other vendor's objects (stale nvcc
      members dragged cudart symbols); build_rocm now removes
      the archive first, and cargo clean -p does NOT purge
      build-script OUT_DIRs.
      VERIFICATION: mistralrs-cli builds release in DUAL (first
      full dual link) and checks zero-error in all three shapes;
      device_spec tests 4/4; full bar green (quant 316/317 - the
      1 is the documented pre-existing fp8; fast_mmq 5/5; gdn
      55/55; cuda_graph 31+8; paged-attn 8/8; mmq bench); the
      dual release binary SERVES: auto-select logs the serving
      role Hip, caps line "Rocm{gfx:(11,5,1)}: graphs=on
      paged_attn=on gguf=OFF mtp=on", device mapper routes
      "Layers 0-63: hip[0]", weight loading proceeds. Full GGUF
      serving on hip:0 remains gated on S2; the caps contract
      makes that visible and enforced. NEXT: S2 (QStorage-on-hip
      replacing the stand-ins; also place the multimodal gate),
      then S4 Vulkan.
      S3c HOT-PLUGGABLE BACKENDS (in progress 2026-09-27): dual binaries
      must start with no GPU hardware present - every GPU backend
      runtime-loads via dlopen, probe-gated, CPU always available.
      PIECES 1+2 DONE in one line: `cuda+rocm` now enables
      `cudarc-hip/dynamic-loading` (candle-core/Cargo.toml) - cudarc-hip's
      build.rs already skipped ALL link-lib emissions in that mode, so the
      same wire drops libamdhip64 + hipblas + hipblaslt + hiprand DT_NEEDED
      at once. The hipcc static archive links clean without them. Proof:
      fresh dual test binary has ZERO GPU DT_NEEDED (yesterday's has all
      four); 5/5 dual tests pass through the dlopen path. Single-vendor
      rocm keeps dynamic-linking DELIBERATELY (prod behavior unchanged).
      PIECE 3 DONE: Device::new_cuda/_with_stream and new_hip/_with_stream
      probe-gate (clean Err naming the missing runtime, never the
      first-driver-call panic); `crate::cudarc` binds per shape so one gate
      covers every shape; cuda_is_available() is now runtime-aware (same
      per-shape binding; link-mode probes report true so single-vendor is
      unchanged); new hip_is_available() for the dual AMD role. Tests 7/7
      (cuda-role refuses cleanly without driver + CPU fallback;
      hip-role gated on probe). B wall hit + DECISION NEEDED: the
      mistralrs-quant hipcc archive (libmistralrsquant.a) references HIP
      runtime symbols DIRECTLY from C++ (hipLaunchKernel,
      __hipPush/PopCallConfiguration, __hipRegisterFatBinary) - dropping
      its dylib=amdhip64 breaks the dual link, and the fatbin registrar
      runs in static init so lazy-PLT tricks die at startup. Reverted to
      keep the tree green. OPTIONS: (B) backend plugin .so (dlopen the
      quant/GDN archives as a companion shared lib - true hot-plug, big
      slice); (H) weak-symbol stubs + RTLD_GLOBAL dlopen (surgical but
      fragile, touches dlopen semantics); (C) accept libamdhip64 DT_NEEDED
      (binary needs ROCm FILES present, no GPU hardware required -
      probes still gate all use). Candle side is fully clean regardless.
      DECISION 2026-09-27: combine B+S2 as PHASED S2 (neither B-now plus
      S2-later nor one undifferentiated blob). Phase 1: plugin shell +
      dlsym loader + call-site indirection around CURRENT kernels
      (including the bail-stubs); verified by the existing bar plus the
      bare-metal startup proof - the green checkpoint. Phase 2: stub to
      real launchers (QStorage-on-hip, loader paths, managed uploads,
      multimodal gate) through the proven loader; verified by GGUF
      serving on hip:0 with caps gguf=on. Phase 1 wraps what exists,
      Phase 2 extends - no call site rewritten twice.
      PHASE 1 DONE 2026-09-27 (candle + mistral.rs): every hipcc kernel
      set ships as a runtime-loaded companion library; the dual binary
      carries ZERO GPU DT_NEEDED (readelf-proven on both fast and release
      profiles) and starts in a stripped env with no ROCm variables.
      Companions: libmoe_hip.so (candle-kernels, hip-plugin feature),
      libmistralrsquant_hip.so, libmistralrscuda_hip.so (core GDN/CK/
      graph), libmistralrspagedattention_hip.so. Design: one
      kernel_decl.rs macro per crate declares entries BOTH ways - plain
      extern "C" against the static archive in every single-vendor shape,
      dlsym-cached wrappers through the companion in dual; one
      hip_plugin.rs loader per crate (env override -> OUT_DIR-baked env!()
      -> exe sibling, process-lifetime handle, named panics). Call sites
      untouched - same names in all shapes. Rust test seams: gpu tests use
      test_gpu_device() (Hip in dual) and 5 hip_fwd twins landed with the
      conversion (BitWise, FusedGlu, NonZero, Softcap, FusedSplitGlu) +
      candle Storage::copy_strided_src Hip arm (a real S1b gap found by a
      twin test). 9 runtime-blocked tests gated with S2 pointers (hqq
      embedding/capture family, managed uploads, bitpack, capture).
      VERIFICATION: dual green through the plugins (quant 303, candle 7,
      core device_spec 4 + cuda_graph 31+8 + gdn 55, paged-attn 8);
      stripped-env startup OK; serving smoke identical (auto-select hip,
      caps line, weight loading); prod-parity bar EXACT (quant 316/1
      documented pre-existing fp8, fast_mmq 5/5, gdn 55/55, cuda_graph
      31+8, paged-attn 8/8); release gate links. LESSON: paged-attn was
      misclassified cuda-only - ATTRIBUTE undefined symbols to archive
      members (rust-lld 'referenced by' lines) before classifying a
      vendor block. BUILD INFRA: [profile.fast] in both workspaces
      (cgu=32, no LTO, incremental) - 4m full / seconds-per-iteration
      vs 9-10m release; sccache wired globally via ~/.cargo/config.toml
      (rustc-wrapper; cross-workspace candle-core hits); mold installed
      for cmake-only links (Rust stays on rust-lld - a mid-stream
      RUSTFLAGS change splits cargo's fingerprint cache into two
      universes); ccache-over-hipcc TRIED AND REJECTED: TheRock's hipcc
      is a binary driver whose clang++ child-exec breaks under ccache's
      two-phase invocation (--version passes, compiles fail). NEXT:
      Phase 2 (QStorage-on-hip + real launchers through the proven
      loader; multimodal gate placement), then the serving smoke with
      real GGUF weights and caps gguf=on.
      PHASE 2 MILESTONE PROVEN 2026-09-28 (committed 792958ff5): GGUF SERVES ON
      HIP:0. Qwen3.5-4B Q8_0 through the fully hot-pluggable dual
      binary: weights via QStorage::Hip + companion plugins, rms-norm
      + rope + softmax + GLU + dequant GEMM all on hip, correct answer
      ("1776", finish=stop, coherent thinking), 13.7 tok/s decode /
      67 tok/s prefill (4B Q8 sharing the APU). Caps line: gguf=on.
      LANDED (candle d93ca2ae): candle QStorage-on-hip (quantized/cuda.rs
      split into role shells + shared body per the S1 pattern;
      QStorage::Hip variant + ~20 dispatch-arm twins; ggml_file
      creation; QMatMul hip_fwd; role-typed device_ptr_with_guard;
      InplaceOpN blanket impls gained cfg'd hip forwarding - the gap
      that silently defaulted every InplaceOp to bail); candle-nn
      RmsNorm/LayerNorm/Sigmoid/SoftmaxLastDim hip_fwd twins +
      unified cuda+rocm feature; mistral.rs: caps flip
      (dual hip quantized_gguf=true + test), 15 quant + 3 core
      CustomOp hip_fwd twins (incl bitsandbytes dispatch_hip_kernel),
      full rope hip support (RotaryLaunchHip + launch_rotary_hip +
      InplaceOp3 twins + is_hip dispatch gates) with the
      hip_rope_matches_cpu kernel-correctness test, test un-gates
      (managed uploads + capture now pass on hip; hqq family +
      bitpack re-gated pending hip-side hqq op coverage - since
      resolved by 973394523; only the chunk-embedding test stays
      gated, pre-existing shared chunk-path race). Dual
      cascade at last healthy run: quant 307/0, candle 30/0, core
      device_spec 4/4.
      INCIDENT 2026-09-28 (recovery pending reboot - READ FIRST):
      Production went down 01:45 and the GPU driver is now wedged;
      BOTH ARE MY FAULT, sequence:
      (1) The Phase-1 release-gate build wrote the DUAL binary to
          target/release/mistralrs - prod's binary path - silently
          replacing the 13:45 rocm build.
      (2) Every smoke test ran `pkill -x mistralrs` first, which
          matched prod too. Prod died at 01:45 with the first smoke;
          systemd did not restart it (signal=TERM treated as clean).
          Down ~4.5h before I noticed. Healthcheck timer did not
          alert - CHECK ITS NOTIFY WIRING.
      (3) On restart prod ran the dual binary and hit the KNOWN
          multimodal projector gap (visual.patch_embed shape) - the
          same error every dual smoke shows; pre-existing, NOT caused
          by Phase 2.
      (4) While restoring I ran a malformed `mv ... /dev/null` that
          destroyed the just-rebuilt binary AND replaced /dev/null
          with a regular file. /dev/null was repaired immediately
          (mknod c 1 3, verified discard). Binary rebuilt (rocm
          shape, HEAD = Phase-1 tree 94207d08c, bar-verified at
          commit time - this IS a prod upgrade vs 13:45).
      (5) The repeated GPU-process kills left amdgpu degraded:
          svm_range_restore_work storms (dmesg), new GPU allocations
          hang system-wide. Prod loads weights (22.8GB RSS) then
          hangs on first forward. GPU unit tests hang or return
          garbage values (the hqq 'numerics' failures: BOTH repos
          at HEAD fail tests that passed at the same HEAD hours
          earlier - GPU STATE, NOT CODE; stash-bisects proved the
          trees innocent).
      RESOLVED 2026-09-28. Root cause of the prod hang was NOT the
      driver and NOT Phase 2: commit 8c0c5d6ef left
      `role_storage()` in mistralrs-core/src/ops.rs calling ITSELF
      in single-vendor builds (should return Storage::Cuda(s));
      release opt turned the tail-call into `jmp $self` inside
      qk_rms_norm_mrope_layout (caught via gdb: identical +14127
      self-jmp on two builds; fast-profile debug info named
      try_cuda_qk_rms_norm_rope at ops.rs:5953 -> role_storage at
      ops.rs:65). Dual builds were immune (separate definition);
      unit tests never touch that path; every serving smoke since
      8c0c5d6ef was dual - only rocm prod could trip it, and only
      on Qwen3.5-family models. One-line fix, prod rebuilt +
      restarted, verified serving (37x48=1776, stop, 18.5 tok/s).
      HQQ SOLO FAILURES also root-caused: latent UPSTREAM ROCm
      stream-ordering race in the 4-bit chunked embedding path
      (code from a31c74f74, pre-arc): chunks 3+5 miscompute solo
      (Eight-bit + CPU pass; reference dequant matches truth;
      explicit device sync after per-chunk uploads fixes it).
      In-suite ambient GPU work masks it; the parallel bar is the
      contract and is green. Do NOT add per-chunk syncs to the hot
      path; documented here instead.
      POST-REBOOT BAR (healthy GPU, HEAD + role_storage fix):
      prod serving OK; quant rocm 316/1 (fp8 documented);
      fast_mmq 5/5; gdn 55/55; cuda_graph 31+8; paged-attn 8/8;
      candle-core rocm 27/1. PHASE-2 RE-VERIFIED on final tree:
      all shapes zero-error checks; dual cascade quant 307/0,
      candle 30/0, device_spec 4/4; dual fast GGUF smoke on hip:0
      re-confirmed (1776, stop, 13.9 tok/s, gguf=on); dual
      RELEASE gate rebuilt with zero GPU DT_NEEDED. Remaining:
      doc-claim sweep (gguf=on), then commits.
      PROCESS FIXES (adopt now): smoke cleanup by port/pid, NEVER
      pkill by binary name; prod serves a COPY under deploy/ so
      release builds cannot clobber the running binary; smokes stay
      on --profile fast (separate target dir); consider the healthcheck
      alert path a TODO.
      LAUNCHER PORT LANDED 2026-09-28 (committed 97875e4b9): fused MMVQ +
      MMQ serve the hip role. The kernels were already hipcc-built on
      both sides; the port is dispatch wiring only: real launchers in
      fast_mmvq_hip.rs (plain/fused_glu/fused_qkv) and fast_mmq_hip.rs
      (plain/fused_qkv/fused_glu/fused_ffn via shared_lhs), a
      device_has_fused_gguf helper (CUDA everywhere + Hip in dual),
      and gate flips in gguf/mod.rs + three try_fused_* gates. The
      #[path] replacement makes callees resolve per-shape, so flips
      are predicate-only. MoE grouped entries stay on the CPU fallback
      (signatures kept NVIDIA-typed for the core callers); candle's
      gpu_quant_fused stays off (quant owns fused routing). 6 new hip
      correctness tests (fused-vs-chute) pass; dual cascade 313/0 +
      30/0 + 4/4; serving A/B on hip:0 (Qwen3.5-4B Q8_0, correct
      1776 both): prefill 55->105 tok/s (+91%), decode 15.0->16.5
      (+10%) via MISTRALRS_NO_FAST_MMVQ/MMQ=1 chute baseline; dual
      release rebuilt with zero GPU DT_NEEDED.
      GDN FUSED-KERNEL PORT LANDED 2026-09-29 (committed aab9e0291): the
      is_cuda() gates that kept the fused GDN kernels off the hip
      role are gone; a single gdn_device_supported predicate (cuda
      everywhere, +hip in dual) now guards all 24 call sites
      (backend gating/recurrence/conv x5, layer spec/fp8/stash x5,
      packed cu_seqlens x2, norm rmsnorm-gated x1, plus the module's
      internal support checks). The launchers were already
      role-typed (RoleDevice/RoleStorage since Phase 1); the 27
      gdn kernel tests now build role-aware devices (new_hip in
      dual) so the whole suite runs on the hip role.
      RESULTS: dual gdn suite 21/26 with --include-ignored; the 5
      failures are IDENTICAL on rocm-only gfx1151 (pre-existing):
      2x sm90-class arch gates (ValueMajor4 unsupported on gfx1151
      by design), 1x stale spec-transition test (slot-capacity
      validation from the 8445372b8 OOB fix, vendor-independent),
      2x fp8-quantized rmsnorm/norm kernels return zeros on
      gfx1151 BOTH roles (fp8-GDN models only; standard bar skips
      them via #[ignore]). Serving: Qwen3.5-4B dual-hip prefill
      105->148.6 tok/s, decode 16.5->28.6 tok/s (+73%), correct
      1776. Qwen3.6-35B MoE decode 0.78->0.81 (noise): the
      dominant MoE-model cost is the EXPERTS path -
      cpu_indexed_moe_forward dequantizes the full 512-expert
      weight set per call (gguf/mod.rs is_cuda gate at :513) while
      rocm takes cuda::qmatmul_indexed_moe_forward (2.69 decode /
      114 prefill vs 0.81 / 4.5 dual). gguf/cuda_hip.rs is the
      187-line stub shell (hip_bail!) - THE next port, same shape
      as fast_mmq_hip; also unlocks forward_grouped
      (grouped_moe_mmq*) whose NVIDIA-typed signatures were kept in
      the launcher port.
      MOE INDEXED PORT LANDED 2026-09-29 (committed ec5cdda76): stage 1 of
      the MoE port - the gather-path entries in gguf/cuda_hip.rs are
      real (qtensor/qmatmul_indexed_moe_forward: quantize the
      activation to Q8_1 once in a per-device leaked workspace, then
      launch the hipcc-built indexed MoE kernels via the same FFI
      surface; all 11 weight dtypes). The gather gate (gguf/mod.rs)
      and the three gather-SHAPING gates in moe/experts/backends.rs
      (forward_gather + gather_lora input/down reshaping) now use
      the all-shape device_has_fused_gguf predicate (made pub for
      core) - on hip they take the 3D/2D cuda-style shapes the
      indexed kernel contract requires. Dev-typed entries
      (moe_dispatch_build, weighted reduce, grouped prequantized,
      fused decode, LoRA decode) stay NVIDIA-typed stubs until the
      backends.rs fast paths become role-aware (stage 2).
      RESULTS (Qwen3.6-35B dual hip:0, mtp off, paged-attn off):
      prefill 4.5 -> 119.8 tok/s (26x), decode 0.81 -> 20.2 tok/s
      (25x), correct 1776. Combined with the GDN port the MoE model
      now outruns the (pressure-degraded) rocm-only baseline.
      Bars: quant dual 316/0; core dual moe/gdn --include-ignored
      24 passed + 11 pre-existing failures (6x new_cuda probe-gated
      by design, 5x the known gfx1151 gdn set, identical on
      rocm-only); rocm quant 316/1 (fp8 documented) + moe 6/0
      (flips are no-ops there); all three cargo check shapes clean.
      MOE STAGE 2 LANDED 2026-09-29 (committed 62e33b6b0): fused decode +
      grouped prefill serve on the hip role. cuda_hip.rs gained
      device-free hip entries (dev from weights, routing as plain
      tensors): indexed_moe_fused_decode_hip (gemv fused gate_up +
      down aggregate, all 11 dtypes), moe_dispatch_build_hip,
      quantize_input_q8_1_hip (owned buffer; workspace variant
      renamed quantize_input_q8_1_workspace), grouped_moe_gemm_
      prequantized_hip, weighted reduces (flat/bf16/same-dtype).
      fast_mmq_hip.rs gained mmq_moe_launcher + six grouped_hip
      bodies (pair_packed, from_glu packed/sorted_pair/pair, plain
      grouped, pair) on the existing MMQ/FIXUP workspaces; routing
      tables cross via the hip_u32_ptrs! macro (utils; bare
      statements, storage borrows must stay caller-scope - a match
      arm returning the guard does NOT compile). backends.rs:
      select_cuda_fast_path flipped to device_has_fused_gguf,
      forward_decode_hip + forward_grouped_hip, LoRA decode stays
      Ok(None) on hip (gather_lora covers it). Tests: 9 hip MoE
      tests green (fused decode Q4K + Q4K/Q5K, grouped Q4K +
      Q4K/Q5K + F32-reduce + Q8_1-prequantized). Serving
      Qwen3.6-35B dual hip:0: decode 20.2 -> 23.4, short prefill
      119.8 -> 214.9, long prefill (230 tok) 610.1 tok/s, correct
      1776. Bars: quant dual 316/0; core dual 30 passed + same 11
      pre-existing (6 new_cuda probe-gated, 5 gfx1151 gdn); rocm
      quant 316/2 (fp8 documented + hqq_embedding_chunks, the
      latter fails on the PRISTINE tree too - environmental, not
      this change); rocm core 27/5 (known gdn set); all three
      check shapes clean. (The hqq rocm failure passed on re-run:
      flaky/environmental, fails pristine too.)
      LORA MOE DECODE LANDED 2026-09-29 (committed c0653b5b6):
      IndexedMoeRoutingHip + IndexedMoeLoraDecodeHip in cuda_hip.rs
      (weights struct reused device-free; dev cloned from gate;
      gate_up pair + lora down launches, F32/F16/BF16 outputs via a
      MoeLoraOutputHip trait; device_ptr() instead of the
      stream-typed device_ptr_with_guard). backends.rs:
      forward_decode_lora_hip wired into forward_decode_lora.
      Tests: hip_fast_decode_lora + hip_grouped_prefill_lora
      (f32/f16/bf16) green vs gather_lora - 13/13 hip MoE tests.
      Bars: quant dual 316/0; core dual 34 passed + same 11
      pre-existing; rocm quant 316/1 (fp8), core 27/5; all three
      check shapes clean. The MoE hip port is now COMPLETE
      (gather, fused decode, grouped prefill, LoRA decode,
      grouped LoRA). HQQ HIP PORT LANDED 2026-09-29 (committed 973394523):
      the GPU dequantize() was the only Cuda-only hqq op (bitpack
      closures already fall back to tensor ops with hip_fwd twins,
      proven by layer construction on hip). dequant_for_dtype_hip!
      mirrors the macro on hip types (15 arms generated from the
      CUDA body, then hand-fixed); dequantize_hip branches off
      is_hip. Test gates opened (test_device now new_hip in dual):
      quant dual 320/0 (hqq4/hqq8/uqff/apply_isq green; single-
      chunk embedding exact, max_diff=0). LESSON: generating the
      copy by script duplicated the just-added is_hip branch into
      the copy -> infinite self-recursion (TCO loop); always diff
      generated copies. The chunk test stayed gated at the time;
      RESOLVED 2026-09-30 (see the CHUNK RACE entry below): NULL-stream
      kernel launches, fixed fleet-wide. MTP-ON/PAGED-ON SMOKE 2026-09-29
      (Qwen3.6-35B dual hip:0, defaults: paged_attn=on mtp=on):
      short prefill 198.5, long prefill (230 tok) 616.6, decode
      23.3 tok/s, correct 1776 - breaks even vs mtp-off/paged-off
      (214.9/610.1/23.4). MTP only wins on draft acceptance; these
      short deterministic probes show no uplift. Paged-attn pays
      under multi-turn/concurrency load, not single-stream speed.
      DECODE ARC 2026-10-02 RESOLVED (uncommitted: attention/mod.rs CK gate +
      pipeline/mod.rs paged-arm reset + device_spec.rs rocm:N alias; candle:
      cuda_backend/body/device.rs dedicated stream + positions_probe/
      stomp_probe examples): THREE bugs, three fixes, both roles green.
      (1) WARMUP RESIDUE: the paged-scheduler path never resets the live
      normal KV cache (no set_none_cache/clone calls in the PagedAttention
      arm of submit_step) - harmless on the hip role (PA layers ignore
      it), fatal on the degraded-eager NVIDIA role (kv 17 vs 16 keys,
      layer-0 mask mismatch). Fix mirrors the default scheduler's
      fresh-prompt Reset, caps-gated (PA roles untouched) and
      token_offset==0-guarded (continuation chunks keep prior keys).
      (2) CROSS-THREAD ORDERING (THE root cause of the NaN/garbage):
      positions upload lands clean (creation readback [0..10) correct,
      pre/post-synchronize() reads IDENTICAL - not a stomp) but the
      rope gather later reads 0x01010101 garbage -> index_select
      bounds assert (indexing.cu:63) -> NaN logits -> poisoned
      context (CUBLAS_STATUS_INTERNAL_ERROR). Thread IDs prove the
      migration: upload on tokio worker 51, consume alternating on
      rayon workers 34/35, and candle's BackendDevice::new used
      cuStreamPerThread (0x2) - a DIFFERENT physical stream per
      calling thread with no ordering between them; work migrated
      mid-request and read in-flight uploads. Evidence: tracer D2H
      syncs OR CUDA_LAUNCH_BLOCKING=1 served 3/3 coherent; clean
      async build NaN'd 4/4 deterministically. FIX (candle): one
      dedicated stream per device (context.new_stream()) shared by
      all threads - restores total order; the lost inter-thread
      concurrency was never sound without explicit events anyway.
      All probes that EXONERATED everything else: rope math 7/7 vs
      CPU, expert gather mm bit-exact, sort fallback, U32
      upload/cast/index_select, rms/GEMM/softmax canary (stomp_probe),
      sanitizer route DEAD (shadow memory OOMs the 3090 on a 16B
      serve). (3) CK GATE - and the correction lesson: the first fix
      keyed the CK flash-attn gate on device_has_fused_ops, which
      correctly blocks the dual NVIDIA role (bare is_cuda() had fired
      hipcc FmhaFwdKernel against NVIDIA pointers - async HIP fault
      aborting the process) but UNLOCKED it for Device::Hip - and
      ck_flash_attn's launcher is Storage::Cuda-shaped
      (as_cuda_device() bails on hip: "expected a cuda device, got
      hip" at first decode step, backtrace via RUST_BACKTRACE=1:
      rocm/mod.rs:47 <- attention/mod.rs:336 <- PA forward). KEY
      INSIGHT: in dual builds CK was ALWAYS OFF for hip (the old
      is_cuda() gate is false for Device::Hip) - dual prod prefill
      silently used the eager/flash path all along; CK only ever
      served single-vendor rocm. LESSON: device_has_fused_ops is for
      kernels that EXIST per role; CK's launcher is single-vendor-
      shaped, so the gate is cfg-keyed: fires only in single-vendor
      rocm builds (!cfg!(all(cuda,rocm)) && is_cuda()); a
      Storage::Hip twin (the 5-op hip_fwd pattern) is the follow-up
      if CK-on-hip-in-dual is ever wanted.
      VERIFICATION MATRIX (final, clean tree, no tracers, fully
      async): cuda:0 4/4+3/3 coherent, 12.4-12.5 tok/s decode, zero
      errors (eager attention, dequant GEMM, host topk - the full
      degradation stack working); hip:0 3/3 coherent, 101-102 tok/s,
      18 graphs captured+replaying, zero errors (identical to the
      pre-change control - no regression); all three build shapes
      check clean (mistralrs dual/cpu/rocm 0 errors; candle dual/cpu
      clean). ALSO LANDED: "rocm:N" accepted as a parse alias of
      "hip:N" in DeviceSpec (users reach for the ecosystem name; the
      cargo feature is rocm too; enum variant stays API-named per
      the design note - Device::Rocm would conflate the API layer
      with the vendor layer that GpuArch::Rocm owns; revisit the
      variant naming at the stage-1 per-backend re-architecture).
      27B ON THE 3090 VIA RUNTIME ISQ 2026-10-02 (user-directed;
      uncommitted tail below): Qwen3.8-27B Q6_K (~22GB, too big for the
      24.5GB usable) served on cuda:0 with [models.quantization]
      in_situ_quant = "Q4K" (~15GB). ISQ plan logs "quantizing to q4k,
      sensitive tensors q6k". Three more dual-blind gates fell on the
      GDN path (this model is hybrid GDN + full attn): (1)
      RecurrentStatePool::checkout took the in-place pool path on
      is_cuda() -> hip-companion clamp_slot_table status 1; gated to
      device_has_fused_ops so the role uses the gathered copy +
      generic-scatter commit (already covered). (2)
      rms_norm_forward_add + the two fused-norm gates in layers.rs
      (rms_norm_residual, residual_then_norm) claimed the hipcc
      cuda_add_rms_norm kernels via bare is_cuda(); same predicate
      swap, eager add+norm follows. (3) fused_glu() (utils/ops.rs)
      has a cuda-ONLY fast path (all(cuda, not(rocm))) so dual fell
      into FusedGlu::cuda_fwd -> ffi::fused_glu_f16 (hipcc binary) ->
      "invalid argument" during ISQ stats forwards; dual NVIDIA role
      now serves it with upstream unary+mul (all in the PTX twin),
      numerically identical up to op ordering.
      27B BENCH cuda:0 ISQ-Q4K: prefill 47.7 (69 tok) / 52.8 (208) /
      52.6 (1174) T/s; decode 5.3-5.5 T/s; arithmetic correct
      (37x48=1776); essays coherent. xlarge (8700 tok) OOMs on the
      prompt step: the eager full-attention scores [1,32,8700,8700]
      plus transients vs 24.5GB (hip serves 262k via the 2GB PA
      pool) - practical 3090/27B context ~= <=4k tokens, journaled
      as a degraded-path limit (chunked prefill could lift it;
      not pursued for a bench datapoint).
      NVIDIA-ROLE SLICE 3: PAGED-ATTENTION CUDA COMPANION (in progress
      2026-10-02; 318a2d301 + 642b77861 committed, dispatch uncommitted):
      the dual NVIDIA role gets real paged attention instead of eager.
      (1) NVCC FIX: q4_codebook.cuh kept the Lloyd-Max LUTs as
      namespace-scope constexpr - hipcc accepts, nvcc rejects
      (undefined in device code). Now __device__ __constant__ (all
      uses are in kernels); all 9 dense kernels compile under both
      nvcc 13.3 and hipcc. (2) BUILD: build_cuda_companion() emits
      nvcc-built libmistralrspagedattention_cuda.so in dual (dlopened,
      nothing linked - same symbols as the hip companion, separate
      RTLD_LOCAL handles, no prefixing needed); cuda_plugin.rs loader
      + ffi_cuda.rs decls (16 dense entries, cuda_backend CUstream).
      (3) TWIN: cuda_role/ backend (cache/gather/paged_attention,
      generated from backend/ + hand fixes: CudaView has no stream()
      upstream so offset pointers stay on the slice, single workspace
      copy, cuda_fwd not hip_fwd) + role_dispatch.rs *_auto wrappers
      routing Storage::Cuda -> twin, else -> hip path. (4) CORE:
      layers/paged_attention.rs + windowed_pool.rs call the auto
      wrappers under cfg-dual (single-vendor untouched). REMAINING:
      twin unit tests on the 3090 (6/6 green: copy_blocks x2,
      reshape+gather f32-exact, Q8, Q4, Q4+QJL), caps flip
      (paged_attention=true for dual-CUDA), 16B serve smoke on cuda:0:
      caps paged_attn=on, 2GB pool (1365 blocks, 43k ctx), HEALTH-OK,
      coherent thinking traces, zero hard errors. PARITY VERDICT:
      the 37x48 prompt loops identically under PA and eager
      (separate-alone servers, temp 0.3) - pre-existing eager-role
      MoE behavior (vendor-variance class), NOT a PA regression.
      Decode 6.9-7.0 T/s (no graphs yet - next slice), short prefill
      47-66 T/s. NOTE: two 16B residents OOM the 3090 (24.1/24.5GB)
      - A/B servers one at a time. ALSO: graphs=OFF still attempts
      precapture then warns "expected CUDA input ids" - the graphs
      slice should skip the chain entirely when caps.graphs is false.
      COMMITTED: slice-3 companion+twin+dispatch+caps (this queue).
      NVIDIA-ROLE SLICE 4: DECODE GRAPHS (2026-10-03; machinery landed,
      enablement BLOCKED - see verdict): new graph_role.rs module makes
      the whole capture/staging/mempool stack role-aware: GraphStream/
      GraphEvent/GraphPinnedHost are aliases in single-vendor builds and
      role-dispatching enums in dual, with ~15 normalized driver fns
      (SUCCESS==0 both drivers). cuda_graph.rs swept onto them (hip path
      preserved verbatim); execution.rs lookahead extracts the hip arm;
      gdn.rs phase timers + normal/multimodal replay paths role-keyed;
      candle re-exports CudaGraphHtodCacheGuard. THREE REAL BUGS fixed
      in the dispatcher out-params (end_capture/instantiate/mempool):
      `.cast()` on a `&mut`-local auto-derefs to the null POINTER VALUE
      instead of the slot address -> NULL out-params -> INVALID_VALUE
      with null pool; the same slip existed in the single-vendor arm and
      WOULD HAVE BROKEN PROD ROCM GRAPHS on next rebuild. Regression
      test pins the mempool out-param. ENABLEMENT VERDICT: the NVIDIA
      role reaches capture (mempool guard + end-capture + instantiate
      all work - mempools ARE supported on the 3090 eGPU, attr 115 in
      cudarc-0.19.8, my earlier attr-118 probe read the wrong ID), but
      the captured forward dies in 1ms: "moe experts forward ->
      to_cpu_storage during CUDA graph capture" - the eager grouped MoE
      gather (the degraded-role path) copies routing tables to HOST for
      per-expert grouping every forward; structurally illegal inside a
      capture. Scope: NOT plumbing - needs device-side routing (the
      fused MoE companion, slice 5) or a capture-friendly grouping
      variant. Role enablement reverted (is_cuda_graph_device hip-only
      again, caps graphs=false) so no request pays the failed first
      lazy capture;       the flip is a 2-line change once MoE is capture-safe.
      NVIDIA-ROLE SLICE 5: FUSED MOE DECODE COMPANION (2026-10-03, in
      flight): libmistralrsquant_cuda.so - nvcc-compiled indexed_moe.cu
      (the ONE TU carrying the whole decode set: Q8_1 quantize
      bf16/f16/f32 + fused gate_up for 11 dtypes + down_aggregate for
      11 dtypes - self-contained, compiled clean under nvcc 13.3 on the
      first try). cuda_plugin.rs loader + ffi_cuda.rs dlsym entries +
      gguf/cuda_role.rs twin (workspaces + quantize + fused decode,
      cuda_backend-typed) + backends.rs: forward_decode_cuda_role twin
      caller + select_cuda_fast_path grants the NVIDIA role DECODE
      only (grouped prefill stays hip-gated). RESULT: 16B decode on
      the 3090 6.9 -> 42.8 T/s (6.2x), zero hard errors, text behavior
      unchanged (same vendor-variance loop as eager/PA parity). Graphs
      flip re-applied after this (mempool capture now legal - MoE is
      device-side); smoke pending. ALSO: the lib.rs cfg-attr detach
      bug class struck again (hip_plugin cfg stolen by cuda_plugin
      insert) - ALWAYS check attribute adjacency after module-list
      edits.
      GRAPHS FLIP FORENSICS (2026-10-03, flip reverted in the same
      commit as the MoE companion): with MoE device-side the capture
      machinery runs end to end on the 3090 - mempool guard, capture,
      instantiate, upload all succeed - but the first cuGraphLaunch of
      the model-scale graph SEGFAULTS the driver (gdb: rax=0,
      mov (rax),rax four frames deep in libcuda; same with
      AUTO_FREE_ON_LAUNCH flag 0 and 1; a pre-launch context bind on
      the replay thread does not fix it). EVERY ISOLATED COMPONENT
      CAPTURES+LAUNCHES CLEAN: trivial PTX graph; 600 synthetic
      memalloc+memset+memcpy nodes; the PA companion reshape+gather
      pair (after warm + event-tracking disable - two real capture
      rules: dlopen/cudart-static-init inside a capture window
      invalidates it, and cuEventCreate from tracking-enabled guards
      does too); the fused MoE twin (eager == replayed output bit-for-
      bit within 1e-3). Remaining suspects: the PA DECODE kernel
      (v1/v2 - the pair probe did not cover it; its launcher may make
      capture-hostile occupancy/attr queries), the htod-cache interplay
      (rope uploads pass through enable_cuda_graph_htod_cache during
      capture), or a driver bug at this graph composition on 610.57
      eGPU. NEXT for the follow-up: extend the paged-attn capture
      probe to the v1 decode kernel, then bisect the captured model by
      layer count. The capture probes are durable #[ignore] tests
      (graph_role scale probe, paged-attn companion probe, quant
      fused-decode probe).
      NVIDIA-ROLE SLICE 6: GROUPED-PREFILL MMQ + DENSE MMQ COMPANION
      (2026-10-03): the nvcc companion grew the full prefill kernel set
      - moe_grouped.cu (dispatch + grouped gemms + weighted reduce),
      mmq_quantize.cu, and the ten per-dtype mmq_instance_*.cu TUs
      (there is no umbrella mmq_gguf.cu - first build attempt chased that
      phantom). ffi_cuda.rs +45 extracted entries (macro now takes an
      optional -> ret for the int-returning reduce kernels);
      fast_mmq_cuda.rs = scripted twin of fast_mmq_hip.rs (1689 lines:
      cuda_backend types, Storage::Cuda, nv_slice_ptr/nv_u32_ptrs utils
      twins, GpuArch::resolve, one-time attr logic intact); cuda_role.rs
      grew moe_dispatch_build_cuda + weighted-reduce twins; backends.rs
      gained forward_grouped_cuda_role (MMQ-only route; grouped lora and
      non-MMQ dtypes fall back to the eager gather loudly) and
      select_cuda_fast_path now grants the NVIDIA role GroupedPrefill.
      Dense MMQ wired too: gguf/mod.rs try_fast_forward routes NVIDIA-role
      batches > 8 through fast_mmq_cuda::plain (decode batches 1-8 stay
      on the dequant path until an MMVQ companion slice).
      RESULTS (16B, 3090): prefill ~64 -> 2157 T/s at 1939 tok (33x,
      past the hip role's 1014 reference), decode steady 42, zero hard
      errors, long-context reasoning grounded in the full prompt.
      27B ISQ-Q4K: prefill 47-53 -> 240.6 T/s (~5x), decode 5.4 -> 8.45
      T/s - the grouped path also serves runtime-ISQ weights.
      Correctness: cuda_grouped_prefill_matches_dequant_reference oracle
      on the 3090 (1% relative = the two-stacked-Q8_1-activation-quantize
      budget; tolerance split out of assert_close for it) + the dense
      twin's 6 tests green on hardware.
      BUG CLASSES THIS SLICE: (1) the cfg-attr detach bug struck a METHOD
      - inserting forward_grouped_cuda_role above forward_grouped_hip
      stole its #[cfg], breaking cpu/rocm-only checks (14/29 errors);
      ALWAYS re-check the attribute above any replaced signature.
      (2) QTensor::device_ptr_with_guard is role(hip)-typed in dual and
      its nvidia arm bails 'not implemented' - the twin uses the
      dual-aware device_ptr() (weights live for the model lifetime).
      KNOWN DUAL GAP (pre-existing, now journaled): fp8/f8q8 tests abort
      the dual test binary on the NVIDIA role (hip-compiled kernels
      dlsym'd via kernel_decl with no role gate; CUDA_CHECK exit()s the
      process). Not touched by this slice (diff disjoint); needs the
      same role-gating treatment as the MoE/MMQ paths or companion
      twins.
      NVIDIA-ROLE SLICE 7: MMVQ COMPANION + GEMV LANDMINE FIX + FP8
      ROLE-GATING (2026-10-04): (A) mmvq_gguf.cu joined the nvcc
      companion (nvcc-clean first try); ffi_cuda grew the 93 MMVQ
      entries (30 fused_glu + 30 fused_qkv pulled from the declare-
      macro signatures + 33 plain); fast_mmvq_cuda.rs twin (1215
      lines, same transform; the 6 QTensor::device_ptr_with_guard
      sites needed the dual-aware device_ptr() like MMQ);
      try_fast_forward_cuda_role now routes dense decode batches
      1-8 through MMVQ. HONEST RESULT (A/B with MISTRALRS_NO_FAST_
      MMVQ): decode 42.5 -> 43.8 T/s (+3%) single-stream - the
      dense GEMVs were already memory-bound and the fused MoE + PA
      dominate; the value is family parity (mmvq decode + mmq
      prefill + fused MoE both phases now all present) and should
      grow with batched decode. (B) LANDMINE: gemv::should_use_gemv
      matched Device::Cuda (the NVIDIA role in dual) but gemv() is
      a hard dual bail-stub - any unquantized F32/BF16 linear at
      decode (the 27B GDN b/a gates class) would have crashed
      instead of falling back; the gate is now dual-wide false
      (matching the stub) and unquantized linear falls back to
      plain GEMM everywhere in dual; a gemv companion twin is the
      journaled follow-up. Found by the twin's inherited unquant
      test, not by serving. (C) fp8 role-gating: the five fp8
      cuda_fwd handlers (scalar_fp8 x2, vector_fp8, blockwise x2)
      receive the NVIDIA storage in dual but launch hipcc-built
      companion kernels with a CUDA stream - the invalid-argument
      CUDA_CHECK exit()s the process. All five now early-bail
      loudly via dual_cuda_role_bail! ("serve the AMD role..."),
      fp8-family tests retarget to the working role through
      fp8_family_test_device (hip in dual), and a contract test
      pins the loud bail on the NVIDIA role. CONSEQUENCE: the fp8
      abort had been killing the dual test binary at 'fp8::'
      alphabetically - the ENTIRE rest of the suite never ran;
      it now completes: 343 passed / 0 failed (one pre-existing
      flake surfaced: grouped_large_scale absolute 1e-4 tolerance
      vs Q8_1-activation-quantize accumulation noise at |v| ~ 76 -
      made relative). fp8 NVIDIA-role parity stays a journaled
      follow-up (portable-FMA TUs would likely nvcc-compile, but
      no fp8 models in the serving set).
      GRAPHS-CRASH FOLLOW-UP (2026-10-04, flip still reverted - but the
      crash is now DECOMPOSED INTO THREE UNDERSTOOD SUB-BUGS):
      (1) THE ORIGINAL DRIVER SEGFAULT WAS CUBLAS-IN-CAPTURE. The A/B
      with MISTRALRS_NO_FAST_MMVQ=1 + graphs on brought the raw
      SIGSEGV back - proof: the old dense-decode path put cuBLAS GEMVs
      inside the captured step (handle/workspace allocs corrupt the
      exec) and the MMVQ companion slice REMOVED cublas from it. The
      segfault is FIXED by MMVQ; what remains with MMVQ is a
      catchable CUDA_ERROR_ILLEGAL_ADDRESS.
      (2) BROADCAST-OP PARAM UPLOADS MUST NOT BE CAPTURED. Minimal
      repro (mmvq_plain_bmul_capture_launch_matches_eager in the
      mmvq twin): plain()+exact-shape bmul replays bit-exact;
      plain()+BROADCAST bmul replays corrupt (block-boundary zeros,
      one wild value, ulp noise) - broadcast bmul passes dims/strides
      via SlicePtrOrNull::params_from_vec, and a FRESH clone_htod
      upload inside capture bakes an unreliable memcpy-node dependency
      on this driver. With the htod cache enabled (the engine's
      design: warmup populates CUDA_PARAM_CACHE, capture reads the
      fixed cached buffer, miss-in-capture bails loudly) the replay
      is bit-exact. The engine already enforces this; the repro now
      pins it as a durable test.
      (3) THE REMAINING MODEL-SCALE FAULT: k-side rope bmul (exactly
      256 threads = 4 KV heads x 64) reads a wild pointer at first
      replay; compute-sanitizer names bmul_bf16+0x1ac0, address 8.5GB
      past every live allocation; single-graph repros all pass - the
      mechanism is MULTI-BUCKET VA COLLISION: the precapture bakes
      placeholder memalloc VAs from the SHARED capture pool into
      several graph execs; each launch re-materializes its own
      graph's allocations over the same pool blocks, so execs trash
      each other's buffers. THIS IS WHY THE HIP ROLE IS IMMUNE (its
      arena gives each graph a private range) and matches upstream
      vLLM's static-workspace design. NEXT SLICE: per-graph
      allocation isolation for the NVIDIA role (dedicated mempool
      per capture / arena port to the nvidia stream / static
      workspaces), then re-flip.
      ALSO THIS SESSION: flags=0 (no AUTO_FREE) with memfree nodes
      SEGFAULTS the driver - AUTO_FREE_ON_LAUNCH=1 is required and
      restored; the post-capture trim is env-gated
      (MISTRALRS_NO_POST_CAPTURE_TRIM) and exonerated; a graph node
      inventory dump (MISTRALRS_DUMP_GRAPH_NODES, dlsym'd
      cuGraphGetNodes) is wired after instantiate; the PA v1 decode
      kernel capture probe passes (companion exonerated); the
      cuda_role workspaces got leak-on-growth (a captured graph may
      hold the old pointer; the hip arena's keepalive semantics,
      reproduced for pool allocs); the htod cache is exonerated by
      reading (capture cannot contain uncached H2D at all). The
      gemv #[ignore] crossover benchmarks now hit the loud dual bail
      (expected; run them single-vendor).
      GRAPHS ROUND 2 (2026-10-04 later, flip still reverted): the
      multi-bucket VA-collision theory from round 1 is DEAD - the
      two-graph collision probe (capture batch-1 + batch-2 graphs
      from the shared pool, AUTO_FREE, interleaved launches, read
      the earlier exec's output after the other's launch) is BIT-
      EXACT. Also exonerated by new passing probes: cuBLAS matmul
      in capture (the 16B lm_head IS BF16/unquantized so cublas is
      still in the graph - but a cublas+broadcast-bmul capture
      replays exactly); the engine's exact pool-guard attributes
      (release threshold u64::MAX + the three reuse attrs, mirrored
      into the mmvq probe - still bit-exact); the PA v1 decode
      kernel (capture probe passes). The raw-dlsym node inventory
      dump is REMOVED - both call forms misbehave on this driver
      (count form: INVALID_VALUE; capacity form: SIGSEGV) - bad
      tooling is worse than none; if needed again use the runtime
      API or cudarc-hip's wrappers. DIFFERENTIAL: the 27B with
      graphs on does NOT reach launch - its capture is INVALIDATED
      by the eager GDN path (host-copy class, same as the original
      MoE gather blocker) and cudarc unwraps the error -> SIGABRT;
      the 27B needs the GDN companion (or a capture-safe GDN
      fallback) before its graphs can even be attempted. WHERE THE
      16B STANDS: the fault is a k-side rope bmul (256 threads =
      4 KV heads x 64) reading a wild pointer at first replay, with
      every isolated mechanism exonerated; the nearest tracked
      allocation to the wild read is the 303,872-byte LOGITS
      buffer (batch x 151,936 vocab bf16) - the wild pointer is on
      the logits side of the VA space. NEXT (the one remaining
      move): engine-level pointer correlation - instrument the
      capture to record every allocation's (ptr, size, tensor) and
      at first-launch check the bmul node params against them to
      name exactly which buffer went wild; short of that, capture
      the real forward with a stripped config (no PA scheduler /
      FA3 schedules) to bisect the engine machinery itself.
      NVIDIA-ROLE SLICE 8: GDN COMPANION (2026-10-04): the fourth
      companion twin - and the first CORE-crate nvcc companion
      (libmistralrscuda_cuda.so; gdn.cu's 6000-line portable TU
      nvcc-compiled clean first try, all 25 externs). Pieces:
      cuda_plugin.rs loader (MISTRALRS_CORE_CUDA_PLUGIN), ffi_cuda
      with PLAIN names (so the launcher fn-pointer tables swap by
      path alone), gdn_cuda.rs - a 10.9k-line twin of the GDN kernel
      layer (the file turned out hip-only BY CONSTRUCTION:
      as_role_device bails for CUDA tensors in dual - 29 sites -
      so the established twin pattern applied). Twin surprises: the
      28 layer-facing context types must be IMPORTED from
      super::gdn (inherent impls removed - duplicates collide);
      as_role_storage sites; GpuArch::resolve; two int-returning
      kernels; private-field widenings (GdnFp8OutputSpec/Layout).
      Dispatch: 23 role-keyed early-routes at gdn.rs's pub entries
      (probe = the first in-scope tensor), v_major_state_supported
      got role-aware device-properties (the FIRST smoke died at
      hybrid-cache creation: 'expected a hip device, got cuda'),
      and gdn_device_supported now claims both roles in dual.
      RESULTS (27B, 3090, ISQ-Q4K): prefill 240.6 -> 901 T/s
      (3.7x), decode 8.45 -> 27.8 T/s (3.3x) - the eager GDN
      fallback (host copies) was the 27B's dominant NVIDIA-role
      cost. Versus this arc's start: prefill 47-53 -> 901
      (~17-19x), decode 5.3-5.5 -> 27.8 (~5x). Output grounded,
      zero hard errors, 58 dual gdn tests green. CONSEQUENCE FOR
      GRAPHS: the 27B's capture invalidation blocker is GONE - it
      now reaches capture; the remaining role-wide graphs blocker
      is the 16B's wild-read (the pointer-correlation
      instrumentation remains the next diagnostic).
      GRAPHS POINTER-CORRELATION INSTRUMENTATION (2026-10-04): built the
      full diagnostic stack, then chased the fault through eleven probes to
      a driver-side prime suspect. Durable keepers (committed): the
      engine-level census (MISTRALRS_GRAPH_PTR_CENSUS, zero-cost off) with
      capture-time VA map + pre-first-launch sweep (moved VA bails loud
      naming the buffer) + U32 content dumps + SIGSEGV auto-namer; seven
      dual capture probes (rope eager chain, bmul/cublas/companion/htod
      relaunch, 400-node, double-instantiate pinning the 801 single-use
      rule). All green - every isolated mechanism replays bit-exact.
      FINDINGS: sanitizer-precise wild read (bmul_bf16, dense, base VA
      ~8.5GB from any live allocation, first launch, reads-only); complete
      bake audit (720 capture bakes incl. all 192 k-rope-256, kernel sig
      matches, arg lifetimes sound, no custom launchers, no graph edits);
      sweep passes + staged content byte-identical; /proc/maps puts the
      fault region outside all reservations incl. the pool's 95GB window
      (garbage-from-birth, deterministic 0x1cd2 region across runs).
      ELIMINATED along the way: trim (twice, incl. a no-trim sanitizer
      fault), hit-state/prefix-cache (confounded by 1-token prefill-only
      requests that never launch), relaunch-count, pool churn/pressure,
      memfree nodes, double-ownership (reframed: nothing capture-malloc'd
      drops post-capture). 27B graphs: separate earlier blocker (hybrid
      checkpoint restore from_vec inside the capture window invalidates;
      needs hoisting, model-side). DETOURS REVERTED: keepalive-drop
      interposition (manual Drop on CudaStorage segfaults; derived clean),
      temp graphs flip + lookahead neuter + candle bake log + rope VA log.
      REMAINING SUSPECT: driver instantiate-patching of malloc nodes at
      model scale (unprobed, unauditable on 610.57); arena port (no malloc
      nodes) is the structural fix. POLICY CHANGE (user-ordered): all
      shapes supported (cpu/rocm/cuda + comma combinations + future
      vulkan, no compounds); AGENTS.md updated, three-shape rule retired.
      CUDA VS ROCM SERVING BENCH (2026-10-05, same dual binary, batch 1,
      Q8_0 KV, 150 completion tokens): 16B Q4_K_M (1939-tok prompt):
      prefill hip 740 vs cuda 2171 (2.9x), decode hip 39.2 (graphs on)
      vs cuda 44.9 (graphs OFF); 27B ISQ-Q4K (1983-tok prompt): prefill
      hip 337 vs cuda 907 (2.7x), decode hip 9.5 vs cuda 27.7 (2.9x).
      Prefill gap is FLOPS (compute-bound, expected); 16B decode is
      close (launch-overhead-bound at batch 1, and cuda leads DESPITE
      missing graphs). 27B hip decode (9.5) is the outlier: AMD-role
      GDN/hybrid decode is the next hip-side optimization target.
      cuda-alone is structurally broken (pre-existing candle-upstream-cudarc
      API drift: CudaGraphExec et al.) - revival is its own queued slice,
      current work closes first. The twin's
      inherited duplicate test module was deleted (tests live in
      gdn.rs where they run against the serving role). The dual quant suite must be run with fp8/f8q8 filtered
      until then.
      Also fixed here: rocm-only shape compiled with the same out-param
      bug class; the dual-nvcc test env needs TMPDIR on disk (tmpfs
      quota exhausted by 49G of target dirs; target-dual moved to
      /home/pstreet/LocalAI/.target-dual-disk permanently).
      COMMITTED: acbf47097 (mistral.rs: reset + CK gate + alias) and
      87ce9a6f (candle: dedicated stream + probes). THE THREE-WAY 16B
      BENCH (Qwen3-16B-A3B Q4_K_M, SAME dual binary both roles, bench
      methodology identical): ROCm hip:0 full stack (PA + 18 decode
      graphs, fused everything): prefill 175 (25 tok) / 779 (165) /
      1014 (1138) / 867 (3362) T/s; decode 53.1 / 49.7 / 41.0 / 29.7
      T/s (context-dependent); essays 54 T/s. CUDA cuda:0 degraded
      stack (eager attention, dequant GEMM, host topk): prefill 21.3
      (25) / 9.9 (165) T/s; decode 6.8-6.9 T/s (server-reported 12.5
      tok/s over its own accounting; the bench measures wall-per-token
      incl. response overhead - use one methodology per comparison,
      never mix). GAP: ~8x prefill short, ~6-8x decode - the price of
      the degradation stack until NVIDIA kernels land. KNOWN LIMIT
      (journaled): CUDA large prefills (>=~2900 tok) OOM in the eager
      MoE gather - the non-fused arm materializes tokens x topk COPIES
      of expert weights via index_select (2900x8x6.3MB = 146GB) while
      the fused grouped-prefill kernel (hip-only) avoids it; 165-tok
      prompts fit (~8GB). FIX DIRECTION: group-by-expert eager gather
      (narrow x rows per selected expert + matmul, O(64 expert copies)
      not O(tokens x topk)) - also speeds up CPU serving which shares
      this path. CPU: skipped per user directive (0.6-0.9 T/s prefill,
      0.1 T/s decode - measured, unusable). Both smoke instances
      cleaned down; prod still stopped by user.
      GROUPED EAGER GATHER 2026-10-02 (uncommitted tail below):
      unblocked CUDA large prefills (2900 tok: OOM -> 472 T/s;
      8700 tok: OOM -> 659 T/s; short/medium also jump 21->63 and
      10->232 T/s - the 64 dense GEMMs beat one strided-batched GEMM
      over an 8 GB materialized copy). Replaces the O(tokens*t*
      expert_size) materializing gather (146 GB at 2900 tok) with
      per-distinct-expert GEMMs on narrow weight views + index_add
      into a zeroed accumulator: O(tokens*t*(in+out)) activation
      copies only. Wired first in both gguf/cpu.rs arms (quantized
      QTensor + plain Tensor, F32 and native dtypes), materializing
      path kept as fallback for unhandled layouts. Tests: exact vs
      materializing on CPU (broadcast/per-slot/single-slot/all-to-
      one-expert), 4D/5D normalization shapes, serving-GPU role,
      plus large-scale (1138 tok, real 64x768x2048 dims) cuda-vs-cpu
      agreement. Full quant bar 327/0. LESSON: candle's
      scatter_add_set enforces stricter rank alignment than index_add
      and rejects this shape - stayed with the copying index_add
      (memcpy-class traffic, dwarfed by the GEMMs) with the reason
      in the comment.
      LARGE-PREFILL "GIBBERISH" VERDICT - VENDOR VARIANCE, NOT A BUG:
      cuda large-prefill outputs degenerate (repetition collapse)
      while hip reasons, yet every eager op proves exact at scale
      (mask bit-exact to 1138; softmax rows sum to 1.0; KV growth
      copies bit-exact; grouped gather == CPU at 1138 tok; per-layer
      absmax tracks within 2%; zero NaN/inf anywhere). Full-logits
      dump comparison (151936 f32 both roles, same 65-tok prompt):
      top-1 AGREES, KL 0.0014 (same distribution shape), but median
      abs diff 0.52 with p1/p99 at -2.9/+3.8. Mechanism: hip prefill
      runs the FUSED MMQ kernels (quantized-domain math, no dequant)
      while cuda runs dequant+GEMM - two different algorithms whose
      rounding differs at O(1) on outlier activations (hidden states
      carry ~500-magnitude outliers), exactly the known llama.cpp
      mmq-vs-dequant spread. Greedy decoding amplifies it chaotically
      (one flipped argmax -> divergent context -> loops); sampling
      masks it at short contexts. TF32 ruled out
      (NVIDIA_TF32_OVERRIDE=0 changes nothing). No code fix on the
      eager path - true parity needs NVIDIA fused kernels (the known
      backlog). Bench throughput numbers stand (that is what was
      being measured); cross-backend greedy text equality is not a
      valid oracle for MoE models across BLAS vendors.
      NVIDIA-ROLE SLICE 2: CUSTOM-OP ROLE-KEYING (committed 65de2242f;
      candle probes committed 4dae95e4) + DECODE-ARC SCOPED
      2026-10-01: the ops.rs arc landed. (1) ROLE-KEYING
      SWEEP: device_has_fused_gguf generalized to device_has_fused_ops
      (all fused C kernels - gguf launchers, rope, MoE router/experts,
      GDN, on-device sampling - share one availability rule: single GPU
      role in single-vendor builds, AMD role only in dual) and 20
      dual-blind gates swept: rotary dispatchers in mistralrs-quant (THE
      original fault: the is_cuda() arm fed NVIDIA tensors to hip
      companion kernels - memory fault on the AMD), 9 ops.rs gates, 2
      sampler gates, 3 pipeline/sampling.rs submit gates, gdn.rs (whose
      as_role_storage sites were .expect() panics, not errors),
      hybrid_cache copy_rows launches, backends.rs renames. (2) TENSOR
      ROPE: the "eager fallback" for rope was host-slices-only
      (cpu_apply_* reads literal CPU memory; the NVIDIA role bailed
      "RoPE positions must be CPU u32"), so wrote a pure-op
      tensor_apply_rotary_qk mirroring the host contract (per-token 3D
      and 2D cos, position-indexed caches, both neox and interleaved
      styles, partial rotation) - 7 unit tests vs the CPU reference, all
      pass, plus the real-GPU comparison via the new
      MISTRALRS_TEST_DEVICE=cuda:0 override (hip_rope_matches_cpu and
      the rope tests green on the 3090). (3) PA CAPABILITY GATE: the
      pipeline built PagedAttention layers purely from config
      (attention_mechanism = config.is_some()); all four decision sites
      (normal/multimodal/embedding/diffusion) now consult BackendCaps
      and degrade loudly - verified: "PagedAttention is configured but
      this device's role in this build has no paged-attention kernels;
      serving with eager attention". This was the third hard fault
      (reshape_and_cache_q8 as_role_storage bail at first attention
      write). (4) TOPK SORT-LIMIT FIX (real upstream limit): candle's
      asort kernel needs next_pow2(ncols)*4 bytes of dynamic shared
      memory; at CUDA's 48KiB static limit (no opt-in attribute) that
      caps GPU-servable rows at 8192 - vocab-sized rows (sampling:
      151,936 -> 1MiB) fail the launch outright with
      CUDA_ERROR_INVALID_VALUE (isolated by sort_probe/op_bisect_probe:
      n=4096 OK, n=151,936 FAIL). The TopKLastDimOp fallback now routes
      oversize rows through a host sort, moving back only the top-k;
      unit test added (vocab-row topk on the NVIDIA role == CPU).
      batched_mm_probe validated the expert gather matmul
      ((8,1,2048)@(8,2048,768) batched F32) bit-correct on the GPU.
      VALIDATION MATRIX (CORRECTED 2026-10-02 - the original entry
      misread warmup traces as request traces): the "sane prefill with
      sane first sample" below was the STARTUP WARMUP ("hello", 1
      token), not the request. The request prefill NEVER completed in
      slice 2: with warmup residue in the live normal cache (kv 17 for
      a 16-token prompt) it died in layer 0 on broadcast_add
      [1,32,16,17] vs [16,16]. Same binary serves the 16B coherently on
      hip:0 (no AMD-role regression from the sweep); the 16B on cuda:0
      loads (709 PTX fns). The ~512 hidden-state outliers are
      plausible Qwen3 activation behavior, not the bug.
      REMAINING BLOCKER - THE DECODE ARC (see DECODE ARC entry
      2026-10-02 below for the resolution path): the warmup-residue
      mask mismatch. NOTE: prefix_cache_n=0 does NOT remove it (a nopc
      run still showed kv_len=17) - the residue is in the live normal
      cache via the paged-scheduler path, independent of the prefix
      cacher. Bisect queue for next sitting: kv-cache read path ops,
      prefix cache machinery on the normal cache (the [16,17]
      residue), the m=1 decode GEMM (cuBLAS internal error could be
      the async surfacing), and role-gate the companion-kernel TEST
      utilities (quant bar on cuda:0: gguf/dequant tests pass, but
      count_nonzero/pack_8bit tests launch HIP companion kernels with
      CUDA pointers - the same dual-blind class, test-side; ops.cu CUB
      reduce + an SIGSEGV). Housekeeping: prod stopped by user for
      host memory; smoke instances cleaned. All temporary tracer
      scaffolding since REMOVED (qwen3_moe.rs stage/MoE tracers,
      gguf/cpu.rs gather tracer, sampler tracer - none stashed;
      /tmp is ephemeral, do not reference it as durable).
      NVIDIA-ROLE SERVING SLICE 1 + 16B BACKEND BENCH 2026-10-01
      (committed 31d7511a9; candle PTX fix committed cffcc107): the 16B three-backend bench
      (user ask) surfaced the first real NVIDIA-role serving work; four
      gates fixed and verified: (1) backend_caps: the dual NVIDIA role
      now reports quantized_gguf=true (served by the upstream
      QStorage::Cuda path, PTX from the twin crate) and CPU GGUF is
      un-refused (the gguf::cpu loader compiled unconditionally; the
      all-false caps were just conservative - verified serving, 0.6-0.9
      T/s prefill / 0.1 decode, unusable and skipped by user directive);
      (2) device_map: the dual NVIDIA device enumeration un-bailed;
      (3) CANDLE PTX BUG (real, fixed in candle): the shared body's
      cfg(feature="rocm") arm loaded PTX TEXT as a binary image on the
      dual NVIDIA role - cuModuleLoadData reads PTX source to NUL, so it
      read past the unterminated string into garbage and 5/11 modules
      failed JIT with INVALID_PTX (6/11 loaded by allocator luck); fixed
      role-keyed with a gpu_ptx_load! shell macro (the gpu_fork_graphs
      pattern: NVIDIA role -> Ptx::from Src CString; hip role ->
      from_binary; single-vendor shapes preserved); the new
      candle-core/examples/ptx_probe.rs reproduces and verifies (11/11
      modules, 709 functions load on the sm86 3090); (4)
      device_has_fused_gguf role-keyed: a bare is_cuda() claimed the
      hipcc-built fused gguf launchers for the NVIDIA role and the first
      model step died with "weight must live on Hip"; dual now keys on
      is_hip() only, so the NVIDIA role degrades quant matmuls to the
      plain dequant-GEMM path (the BackendCaps contract) - this gates
      BOTH the mmvq and mmq branches at try_fast_forward's entry, so no
      per-path whack-a-mole. REMAINING for the CUDA datapoint (the known
      "until symbol prefixing lands" arc, now precisely scoped): the
      custom-op layer assumes the hip role in dual - mistralrs-core/
      src/ops.rs has 16 as_role_storage sites; the first model step
      faults at ROPE (the hip rope kernel launched with NVIDIA pointers
      ON THE AMD GPU - memory fault, then "expected the serving role's
      storage"; as_role_storage bails for Storage::Cuda in dual); fix
      pattern = each fused op falls back to its eager implementation
      (the device-agnostic candle code that serves CPU today); rope +
      the MoE dispatch (indexed_moe_forward Hip-bail) are load-bearing
      for the 16B (Qwen3 MoE, no GDN layers). 16B BENCH NUMBERS (Q4_K_M,
      no MTP tensors in the GGUF so all backends compare plain): ROCm
      (hip role, PA 2GB pool + 15 decode graphs, 1.25s capture):
      prefill 157 T/s @25tok, 717 @165, 984 @1138, 874 @3362; decode
      45-47 T/s short-context, 26.5 T/s at 3.4k context. CPU: 0.6-0.9
      prefill, 0.1 decode. CUDA: pending the ops.rs arc. 27B on the
      3090: infeasible at Q6_K (22GB weights + draft + mmproj + pool >
      24.5GB usable) - needs a Q4_K_XL conversion (~15-30 min llama.cpp
      quantize) when the role is wired.
      COMPOUND FEATURE REMOVED 2026-10-01 (user-directed architecture
      change): `--features cuda,rocm` (plain comma) is now THE way to
      build any feature combination; the `cuda+rocm` compound name is
      gone from every Cargo.toml in both repos, and adding `vulkan`
      later means just `--features rocm,cuda,vulkan`. Findings along the
      way: (1) the compound was cargo's only AND-logic workaround - it
      carried dep:libloading (dual dlsym) and forwarded names, while all
      178 combination cfg sites (35 mistral.rs + 143 candle-core)
      already keyed on all(feature="cuda", feature="rocm") and work
      unchanged under comma syntax. (2) candle-core deliberately
      compile_error!-rejected plain cuda+rocm ("opt into cuda+rocm
      explicitly") - that guard is deleted; dual now "just works" and
      the portability machinery makes surprise combinations harmless.
      (3) LATENT QUIRK EXPOSED: the cli-level compound never enabled the
      cli's OWN cuda/rocm features, so every #[cfg(feature="cuda")]
      CLI code path - including the bench cudaProfiler externs - was
      silently compiled OUT of every dual build ever shipped; proper
      features compile it in, which surfaced the unresolved profiler
      externs; fixed by runtime-dlopening the driver's profiler symbols
      in bench.rs (portable, probe-gated, no new DT_NEEDED - the knob
      now actually WORKS in dual builds when an NVIDIA runtime exists).
      (4) USER-APPROVED SEMANTIC CHANGE: rocm builds are now uniformly
      portable - cudarc-hip/dynamic-loading + candle-kernels/hip-plugin
      moved onto the plain rocm feature, and mistralrs-core/quant/
      paged-attn's rocm build branches ship ALL kernels as companion
      .so files (the kernel_decl dlsym arm now serves every rocm build;
      the extern arm is cuda-only). Single-vendor rocm previously
      linked static kernel archives + libamdhip64; now every GPU
      artifact has zero GPU DT_NEEDED. VALIDATED: 4-shape checks clean
      (cpu/rocm/dual-comma; cuda-only still nvcc-broken PRE-EXISTING -
      gather_kv_cache_kernel.cu compile failure, unrelated to this
      change); rocm-only artifact ldd = zero GPU DT_NEEDED + serves
      the 4B end to end (1776, paged attention through the companion,
      40.6 T/s completion) + quant bar 317/0/1-ignored through the
      flipped loader; dual release built with `--features cuda,rocm`,
      deployed to prod: 29 graphs captured in 15.8s, 9 resident,
      1776 3/3, arithmetic decode 16.6-17.0, essay 12.3 - identical to
      the compound-built binary's numbers. `--features cuda+rocm` now
      fails cleanly ("does not contain this feature"). Candle-side:
      cuda attaches the PTX twin crate (compiled-but-unused in
      cuda-only, the monotone cost of no-AND); rocm attaches
      dynamic-loading + hip-plugin; the cuda_backend kernel re-export
      pair is unchanged (not(all(cuda,rocm)) -> candle_kernels,
      dual -> twin).
      DOC PASS 2026-10-01: environment-variables.md gained the missing
      functional knobs (MISTRALRS_NO_FAST_MMVQ, MISTRALRS_MOE_BACKEND,
      MISTRALRS_FORCE_AVX2) plus the expert/debug cluster
      (MISTRALRS_NAN_TRACE/NAN_PROBE/LAYER_PROFILE, MRS_ZERO_WS,
      MRS_POISON_NAN) and a corrected MISTRALRS_DECODE_ARENA_OVERRIDE row
      (x1.25 + geometric regrow; the old row said 6x). The wrapper's stale
      HANDOFF pointer now names TODO.md. The 35B "ACTIVE default" toml
      comment corrected; mistralrs-test.toml header now declares it the
      pre-router smoke config. TODO items closed: 35B Q4/Q5 follow-up
      (resolved: the divergence was the broken symlink), stale deploy
      comments, env-knob documentation (audited: half the listed knobs
      were already documented or deleted; ROCm-docs remainder split into
      its own item). The multi-backend item gained the 3090 status note
      (cuda-role hardware now real; FlashInfer gate needs a device-keyed
      refinement before a dual-served 3090 can claim it).
      DECODE-GRAPH CAPTURE CHAIN FIXED 2026-09-30 (committed 55b3d270d): the
      dual-prod 27B captured ZERO decode graphs (866 incompatible_shape
      skips) and hybrid+MTP precapture crashed; six fixes, each surfacing
      the next. (1) The multimodal dispatch device gate never persisted:
      the earlier session's edit script died on a LATER assertion before
      the file write, silently reverting ALL its replacements (the
      follow-up script fixed only the precapture site) - lesson: multi-
      replacement edit scripts write once at the end; grep EVERY site
      after. (2) Arena retry growth anchored to the ABORTED PREFIX: an
      overflowing arena fails the alloc hard so the forward aborts at the
      first failed allocation, and consumed counts only what ran; the
      old +12.5%+32MB single retry cannot bridge. Now max(prefix*9/8+MIN,
      2x current) per retry, 4 attempts. (3) The fresh-bucket x6 estimate
      pinned 6x-oversized arenas (measured capture-consumed == warmup
      within 0.1% on this model) - 48GB of pinning starved later buckets;
      now x2+MIN. (4) The arena clamp used MemoryUsage's iGPU "available"
      (host RAM x fraction), which collapses once the KV pool pins
      GTT-backed host pages - it zeroed arenas, a guaranteed failure; now
      the clamp and the guard use the device-true free (mem_get_info: GTT
      for hip, VRAM for cuda) with a DECODE_GRAPH_MIN_ARENA_BYTES floor.
      (5) Unbounded precapture marched pinned arenas into a FATAL hip OOM
      inside reshape_and_cache_kernel.cu (C-side check = process death);
      now a DECODE_GRAPH_PRECAPTURE_MIN_FREE_BYTES (8GB) guard before
      each bucket breaks cleanly, keeping captures, remaining shapes
      lazy. (6) One bucket's capture failure no longer clears ALL
      captured graphs (per-bucket tolerance, both pipelines). VALIDATED
      (27B smoke on the release binary, 4GB pool): 18 graphs through
      batch bucket 32 in 50.85s, guard stopped at 3.99GB free keeping
      them, resident=18, replay cache_hit=88, 1776 3/3, decode 15.4-15.5
      T/s vs 14.1-14.3 pre-graphs (+8%). PROD FINDING: the uncapped
      [paged_attn] pool (utilization 0.9 -> 77.6GB, 3.12M-token capacity)
      leaves ~2.1GB host available after the 27B load, so the guard
      correctly declines all capture; pool cap sizing is a serving-
      posture decision (max-model-context upper bound + two-resident
      budget, user-directed).
      POSTURE APPLIED 2026-10-01 (uncommitted tail): [paged_attn] now sets
      context_len = 262144 (fleet max native GGUF context: 256k for the
      27B/35B/coder-next/gemma-4/4B; 128k glimmer/llama; 40k qwen3-16b)
      -> the 27B pool is 6.5GB / 262,112 tokens instead of 77.6GB /
      3.12M tokens (~26.1 KB/token at K=Q8_0/V=Q4_0). Two follow-on fixes
      after the first capped-pool load CRASHED: (a) fresh-bucket arena
      estimate x2 -> x1.25 of the warmup (every measurement says
      capture-consumed == warmup within 0.1%; the regrow path exists to
      absorb undershoot) and the precapture floor 8GB -> 16GB - the crash
      was the SPEC phase's reshape_and_cache launch OOMing at the 8GB
      floor with the host 4.3GB into swap; the C-side CUDA_CHECK macro in
      mistralrs-paged-attn EXITS on OOM instead of returning an error
      the tolerance could catch (making kernel-launch OOM a process
      death) - journaled as the follow-up FFI surgery. (b) The decode-graph
      speculative-state budget (default: largest-entry x10 capped at 8%
      of device total ~= 10GB) kept only 2 of 29 captured shapes resident
      (per-shape spec state ~= 4.5GB with FullCheckpoints on the 27B MTP);
      MISTRALRS_CUDA_GRAPH_SPEC_STATE_BUDGET_BYTES=16GiB in the wrapper
      keeps 9 shapes resident (single-stream + small concurrent batches)
      while preserving ~38GB host available - the 32GiB first attempt
      left only 11GB and broke the two-resident budget, 16GiB is the
      calibrated balance (27B ~30.5GB fixed + 35B ~34GB fixed + graphs
      + ~12-16GB transients/headroom fits 122GB host).
      FINAL PROD STATE 2026-10-01: 27B demand-loads with 28 graphs
      captured in ~17s, 9 resident, replaying (1776 3/3; arithmetic-probe
      decode 16.5-16.9 T/s; essay 12.3-12.4 T/s - both +16-17% over the
      no-graph morning numbers 14.1-14.3 / 10.4-10.7). Prefill ~186 T/s
      short (69 tok). MTP acceptance ~62%. Historical essay 13.9-17.8
      remains unmatched but predates the hybrid-GDN serving path
      (apples-to-oranges). Session notes: five machine reboots (user
      eGPU bring-up) repeatedly wiped /tmp/opencode - validation moved
      onto the durable release binary directly; an RTX 3090 (sm86, 24GB,
      CUDA 13.3 UMD) is now attached for the CUDA role - auto-selection
      still picks Hip as serving role; FlashInfer's build-shape gate
      means dual+3090 will not claim it (device-aware refinement =
      follow-up); FA2 buildable (flash-attn feature), FA3 not (Hopper).
      UPDATE 2026-10-17 (commit 2f55fd62f): flash-attn is now ROLE-AGNOSTIC
      and part of the prod shape (cuda,rocm,flash-attn). The feature no
      longer forces the NVIDIA role (5 crates un-chained); ON enables FA on
      every GPU role present - Cuda via CUTLASS FA2 kernels (incl. the FA2
      paged-prefill path, now dual-compiled for the 3090), Hip via the
      ROCm-native CK flash path re-gated from solo-rocm-only to
      using_flash_attn(); OFF disables it on both roles (solo-rocm no longer
      gets CK for free). DFlash drafter tightened to solo-cuda (its graph +
      flashinfer windowed-KV sections bind NVIDIA-solo APIs; eager core
      stays dual). Expectation: prefill/TTFT gains on both roles, decode
      flat (GDN-hybrid mix: ~1 in 4 blocks full attention; decode rides the
      paged kernels; CK requires seq_len>1). Big FA payoff stays contingent
      on a dense/MLA model landing on the 3090.
      VALIDATED LIVE 2026-10-17 (build 51fa55078, cuda,rocm,flash-attn,
      restarted in a verified idle window; backstops: build23-mtp4b,
      build24-cpupa): 35B hip 63.6 tok/s decode (from 48-51, +25-32%),
      accept 77.4%; 4B cuda 195.1 tok/s (from 151.3, +29%), accept 80.0%;
      4B CPU PA+MTP end-to-end (dtype fix verified): 26.5 tok/s (from
      19.4 eager, +37%), accept 81.8%; zero journal errors. Decode gains
      came from CK/FA2 firing on multi-token MTP verify steps
      (seq_len>1) plus prefill - bigger than the predicted flat decode.
      Binary grew ~130MB -> ~379MB (CUTLASS kernel set, static).
      Prefill (4.4k-token prompts): 35B hip 1440 tok/s, 4B cuda
      5067 tok/s, 27B dense hip 396 / 3090 1104 tok/s (dense 27B is
      the FA sweet spot: every block pays attention; ~2.8x hip->cuda).
      OPERATIONAL: 27B hip load while 35B + 4B-cpu resident
      OOM-killed the server (load transient beat the evict estimate);
      systemd self-healed in ~6s. 35B+27B coexist (~64GB); the
      ~6GB 4B-cpu lane is the marginal one - unload it when both
      big lanes run.
      DUAL-SERVING BRING-UP + 27B BENCHMARK 2026-09-30 (committed 94e1171c5):
      restarting prod for an orcarouter-27B benchmark revealed the
      Sep 28 16:20 rebuild had put the DUAL binary into prod (boot log:
      Device::Hip + git rev 3d7c712c5) - and the 27B (default model,
      hybrid GDN + mmproj, MTP n=2, split Q8_0/Q4_0 KV) could NOT load:
      a chain of role-blind gates, each fixed and verified by the next
      blocker appearing: (1) server-core's "cuda+rocm" feature did NOT
      expand to cuda+rocm individually (unlike mistralrs-core) - every
      all(cuda,rocm) cfg in server-core was dead code; a PAPROBE warn
      run was invisible under the wrapper's RUST_LOG=mistralrs_core
      (probe via a spare-port from-config instance). (2)
      configure_paged_attn had no hip arm -> PA silently off ->
      demand-load reserved no KV pool (need ~24327 vs the healthy
      ~40711 MB fingerprint) -> MTP attach failed (the gemma4-MTP
      merge made MTP REQUIRE paged attention). (3) split-KV and
      block-quant KV cache validation gates were is_cuda()-only. (4)
      post-load device sync + memory estimation skipped on hip. (5)
      flashinfer supports_layer claimed FlashInfer in dual via
      cfg!(feature="cuda") although the hipcc paged-attn set EXCLUDES
      flashinfer (RDNA lacks ldmatrix/cp.async; documented in
      mistralrs-paged-attn/build.rs) -> models silently resolved the
      FlashInferHnd layout -> gated with the established
      all(cuda, not(rocm)) NVIDIA-role cfg. (6) decode-graph dispatch/
      precapture/host-staging gates were is_cuda()-only (930
      incompatible_shape skips, 0 resident graphs). (7) the graph
      spec-state mem probe matched Device::Cuda only. (8) graph
      state-index padding used GDN_PAD_SLOT (u32::MAX), predating the
      8445372b8 OOB-fix invariant enforced by set_state_indices_tensors
      -> precapture CRASHED the server; padded with the dedicated
      GraphPad row instead (hybrid_graph_slots carries pad_row).
      BENCHMARK (Q6_K 27B, dual hip:0, PA on, MTP n=2, temp 0.3):
      37x48 -> 1776 3/3 stop; short prefill (69 tok) ~184 T/s; long
      prefill (346 tok) 80.4 T/s; arithmetic-probe decode 14.1-14.3;
      essay 600tok decode 10.4-10.7 T/s vs the historical 13.9-17.8
      (which predates the hybrid-GDN serving path AND ran with decode
      graphs). REMAINING PERF GAPS: decode graphs still not captured -
      "CUDA graph capture arena overflowed on retry" (hybrid+MTP
      capture sizing; shape-independent, never exercised before
      because the earlier gates blocked the chain) + one more
      dispatch gate (866 incompatible_shape during the bench); both
      journaled as the follow-up. NOTE: /tmp/opencode/mtp_bench.py
      referenced by the toml no longer exists; recreate for MTP A/Bs.
      CONFIG SYMLINK RESTORED 2026-10-01: models/mistralrs.toml had become
      an unlinked regular file sometime before Sep 28, so the live config
      (35B Q4->Q5 upgrade, prefill 2048->4096, context_len sizing) drifted
      from the versioned deploy copy and nothing config-side was being
      committed. Re-synced the deploy copy to the live content
      (reconstructed exactly - 7187 bytes - after a merge-script failure
      led to a premature rm of the live file; the full diff captured in
      the session plus the git-clean deploy base reproduced it and the
      byte count verified the reconstruction) and restored the symlink
      models/mistralrs.toml -> ../mistral.rs/deploy/strix-halo-gfx1151/
      mistralrs.toml. Process lesson (same class as the edit-script one):
      destructive filesystem steps (rm/symlink swaps) must be chained with
      && behind the content step, never newline-separated. The stale
      device-selection comment block (predates the dual-serve world) and
      the 27B '#mtp = false' note are preserved in git history if wanted.
      HEALTHCHECK MARKER FIX (same window): the stop marker was
      consumed on sight, so during the ~10 min release-build window
      the watchdog re-started prod on the OLD binary inode mid-build;
      marker now means "deliberately stopped right now" - fresh (<4h)
      marker + down = quiet AND KEPT, service active = consume,
      stale = expire + alert. Stub-harness verified on 5 marker
      scenarios.
      HEALTHCHECK ALERT WIRING LANDED 2026-09-29 (committed 09bd4d872).
      GFX1151 GAPS CLOSED 2026-09-29 (uncommitted): the 5 --include-
      ignored gdn failures + the fp8 roundtrip are all green now
      (gdn 81/0 both roles; quant rocm 316/0, dual 320/0).
      (1) fp8-quantized norm/spec-norm returned zeros: ROOT CAUSE in
      gdn.cu, not the kernels - `#if CUDART_VERSION >= 11080` is
      false under hipcc (no CUDART_VERSION), so gdn_fp8_e4m3 fell
      into the `value(0)` stub arm even though the rocm_compat
      cuda_fp8.h shim was already included two lines above. One-line
      fix: `|| defined(USE_ROCM)` (verified __hip_fp8_e4m3 has a
      device-side float ctor on gfx1151: OCP+FNUZ both on).
      (2) 2x sm90 tests now early-skip (vacuous pass + note) when
      !gdn_vmajor_arch, same conditional idiom as the file's other
      arch gates; still fully exercised on Hopper.
      (3) stale spec test: the post-8445372b8 slot-capacity
      validation (base+len <= capacity) turned the test's own slots
      OOB (17+8 > 23) - regenerated `% (capacity - seq_len)`,
      shuffling intent preserved.
      (4) fp8 roundtrip: scalar_fp8 ops had hip_fwd twins with the
      right bodies but cuda-only gates (`all(cuda, not(rocm))` on
      BOTH methods, so no GPU path in rocm-only OR dual) plus a
      cuda-only ffi module gate; the twins were also still typed
      CudaStorage. Flipped cuda_fwd->any/hip_fwd->dual/ffi->any and
      ported both hip_fwd signatures to HipStorage (shared-body
      alloc_zeros exists on both roles). Roundtrip passes on hip
      (mean abs diff 0.0116). NOTE: one dual quant run showed 2
      transient hqq embedding failures that pass in isolation and
      on re-run - flake on this box, root-caused 2026-09-30 as the
      NULL-stream launches (see the CHUNK RACE entry), fixed
      fleet-wide.
      HEALTHCHECK ALERT WIRING LANDED 2026-09-29 (uncommitted):
      the 01:45 pkill incident mechanics verified live: SIGTERM kill
      shows Result=success/ExecMainStatus=15, identical to a deliberate
      stop, so Restart=on-failure restarts nothing; the watchdog then
      exited silently on the inactive service (is-active || exit 0) -
      4.5h of silence. No alert path existed at all (restarts only
      echoed to the check's own journal). Fix: alert() via local mail
      ($USER) + ntfy (topic mistralrs-strix-halo; knobs
      MISTRALRS_ALERT_MAIL/_NTFY_TOPIC), deduped per cause with a 30m
      cooldown; failed state -> restart + alert; inactive with no
      marker -> start + alert; activating-class states exit quiet;
      wedge restarts now alert too. mistralrs-server-stop.sh leaves
      the userstop marker so intentional stops stay quiet (marker
      consumed on sight). Live file is a symlink to the deploy copy,
      same convention as the wrapper. Verified with a stub harness
      (fake systemctl/journalctl/curl/mail, 6 scenarios: down,
      marked-stop, failed, activating, healthy, wedge-x2 + cooldown)
      without touching prod.
      CHUNK RACE ROOT-CAUSED AND FIXED 2026-09-30 (uncommitted): the
      hqq chunked-embedding corruption (pre-existing on BOTH roles,
      pristine tree too) was NULL-STREAM LAUNCHES: 32 kernels across
      5 files (ops.cu bitwise/leftshift, hqq.cu dequant x15, afq x9,
      gptq q_gemm x4, plus a streamless cudaMemset in the afq
      quantizer) launched with <<<grid, block>>> and NO stream arg -
      the NULL/legacy stream, which has no ordering against candle's
      CU_STREAM_NON_BLOCKING compute stream. In the chunk loop,
      bitwise_and read its `packed` input while the index_select
      filling it was still pending -> stale/garbage nibbles -> the
      per-chunk wrong-token values. THE HUNT (lessons): the
      deterministic failure was localized with a CPU dequant value
      catalog matcher (each wrong element matched SOME other token's
      exact dequant -> stale-row reads, matcher coincidences included)
      and a production-vs-standalone per-chunk diff (standalone
      exact, production wrong -> cross-op ordering, not inputs);
      upload-ordering / params-cache / keepalive / pinned-upload fixes
      ALL changed the failure rate but none fixed it - they only
      perturbed host-GPU timing (the red herring era). The device-side
      assertion in candle's indexing kernel (ids OOB) after partial
      fixes + a grep for `<<<...>>>` without a 4th stream arg found
      the class. Runtime primitives verified INNOCENT via ctypes
      microbench: hipMemcpy and hipMemcpyAsync-on-stream both 300/300
      correct (the earlier cudarc-hip blocking-copy experiment was
      reverted; candle experiments fully reverted, tree clean).
      FIX: cudaStream_t threaded through all 32 launches +
      cudaMemsetAsync; declares gained `stream: *mut c_void`
      (role-neutral, kernel_decl dual-dlsym compatible); call sites
      pass dev.cuda_stream().cu_stream(). hqq chunks test UN-GATED
      fleet-wide. BARS: chunks 30/30 + 30/30 (rocm-only + dual, was
      flaky 1/3..10/10 cold); quant rocm 316-317/0 (1 transient in
      one run, 6/6 clean re-runs single-threaded); dual 321/0; core
      gdn 81/0; checks x3 clean. NOTE: candle tree carries fmt-only
      drift in 5 files (4 appeared mid-session + candle-nn/ops.rs) -
      NOT from this fix, left untouched.
      PROJECTOR ITEM RESOLVED 2026-09-28 (not a code bug): the
      `visual.patch_embed` multimodal gap does NOT reproduce on the
      current tree - Qwen3.8-27B + mmproj serves text AND vision
      correctly on dual hip:0 (test image: "blue circle inside red
      square"). The incident-era journal claim was stale. Unit sweep
      of the qwen mmproj bindings: 123/0.
      INVESTIGATION 2026-09-28 (dual-hip MoE NaN, ROOT-CAUSED as
      environment, with two real bugs found and fixed on the way):
      Qwen3.6-35B + gemma4-26B "argmax received invalid logits" at
      first decode on DUAL ONLY; rocm-only serves the same files
      fine; dense models fine on dual. Probe chain
      (MISTRALRS_NAN_TRACE=1, new gdn-proj-x/ba probes in
      gdn/projection.rs): prefill healthy; at decode the residual
      entering the first GDN layer carries 16/2048 NaN (a 64-byte
      corruption) ONLY when GTT (shared APU memory) is near-exhausted
      (prod 22GB + 35B 22GB resident). The F32 b/a gate GEMV (small
      dense rocBLAS gemv) amplifies it to all-NaN - which is why
      gdn-ba always fired first. Reproduced deliberately: 35B +
      27B filler instance -> NaN; same server alone -> correct 1776.
      NOT a fork code bug: every suspect op passes at unit level on
      hip (argsort incl. 512-wide tie-free; router topk; MoE gather
      3-expert + mixed-kquant + 512-expert topk-8 scale; MMVQ plain
      + fused at real shapes Q4K/Q5K/Q6K k=2048; F32 [32,2048] gate
      projections via both GgufMatMul and UnquantLinear wrappers).
      WHY dual-only: is_cuda()-gated fused GDN kernels (gating,
      decode recurrence, chunked prefill) run on rocm-single but not
      on the hip role -> dual decode takes the generic path (rocBLAS
      gemv + generic recurrence, ~1.3 tok/s vs 18.5 fused) and only
      that path touches the pressure-fragile allocation pattern.
      Fix for dual = GDN fused-kernel port to the hip role (S4-scale
      follow-up, same shape as the launcher port). Guidance: do not
      smoke big models on dual with prod resident; the signature is
      gdn-ba nan=first + slow decode.
      FIXES LANDED 2026-09-28 (this tree): (1) candle ArgSort had no
      hip_fwd -> "no hip implementation for argsort" killed gemma4
      vision decode; hip twin added + tie-free wide-row tests (the
      first bf16 wide test failure was tie aliasing, not a kernel
      bug - odd-mantissa generator documented in the test).
      (2) startup race: the engine thread holds the pipeline mutex
      through warmup while main raced it with try_lock().unwrap()
      (lib.rs) -> panic on any slow warmup; now lock().await.
      (3) gdn-proj-x/ba-split probes added (env-gated, same pattern
      as the incident probes). Bars: quant dual 316/0, candle dual
      32/0, quant rocm 316/1 (fp8 documented), candle rocm 27/1
      (same), core hip router/moe/gather tests green; the 6
      Device::new_cuda MoE tests fail on this box in dual by design
      (probe-gated, no NVIDIA driver) and are outside the cascade.
      S1 DONE 2026-09-26 (candle): kernels twin builds standalone
      (11 PTX, symlink-free, sm_80 fallback GPU-less); dual check
      green alongside default + rocm; 4 dual tests pass (CPU op +
      hip arch/roundtrip/affine-module-load + graceful NVIDIA stub
      error in one binary); regressions green (fast_mmq 5/5, quant
      295, gdn 55, mmq bench, warning-parity with the original,
      candle-nn's `candle::builder_arg!` kept working via a crate-
      root macro_export). Structural lessons, all earned the hard
      way: (1) bodies must not define #[macro_export] macros (dual
      instantiation exports twice + poisons absolute-path imports);
      (2) shell `use` bindings are invisible to include-text inside
      modules DECLARED by expanded code - role bindings live in the
      role-module shells, bodies import via `super::`; (3) inherent
      impls on shared crate types (`impl Scalar`) conflict under
      dual instantiation - use a body-local trait; (4) macros cannot
      expand to struct fields or splice across their boundary - the
      dispatch gate uses an inverted companion macro instead.
      S1b DONE 2026-09-26 (candle): Device::Hip + Storage::Hip
      variants with the full dispatch layer - hand arms in device.rs
      (constructors, accessors, all storage-creation methods,
      synchronize), script-duplicated arms in storage.rs (17 single
      + tuple forms), to_device cross-product + conversions in
      tensor.rs, safetensors upload twins on hip_backend types,
      DeviceLocation::Hip with a role-owned location macro in the
      shared body, Display arms, defaulted hip_fwd (+aliased) on all
      six CustomOp traits (opt-in per op, Msg bail), quantized
      loaders + ggml_file routed to explicit S2 bails. Dual check
      green; dual tests 5/5 with a tensor-level Hip test (zeros/add/
      dtoh/hip->cpu/location); regressions green (rocm + default
      checks zero-error with warning parity, fast_mmq 5/5, quant
      295, gdn 55). Structural note: tuple match arms need paren-
      aware patterns, and arm-tracking scripts must reset per match
      (cross-match contamination renamed two Cuda arms mid-work -
      caught by the type checker, fixed by content search).
      REMAINING (S2): hip-side quantized coverage (QStorage-on-hip,
      currently explicit bails), safetensors/GGUF managed uploads in
      dual (single-vendor-gated), full candle GPU suite through the
      dual build; symbol prefixing for the static libmoe.a FFI (S3
      fast paths); the upstream-cudarc fork port restoring
      graphs/Lt/managed caps on the NVIDIA role (follows BackendCaps
      degradation until then); cuda-only builds stay broken
      (pre-existing fork-vs-upstream divergence, out of scope).
- [ ] _S semantic check post-merge (was §6): upstream `804d361e` bf16 vectorized
      `fast_sum` gated `__CUDA_ARCH__ >= 800` will be live in ROCm JIT builds
      (rocm_compat pins 1030) and is untested on gfx1151. Cheap safety check.
- [ ] ~S X-LoRA gate (was §7). 11 loaders panic with `todo!()`
      (`mistralrs-core/src/pipeline/loaders/normal_loaders.rs:1899,2763,3069,3376,3580,3783,4021,4218,4519,4819,5058`)
      and 20 models have `unimplemented!()` in `xlora_forward`.
      Gate the `--xlora` CLI path to the implemented set, or drop the claims.
      Do after upstream merge (touches overlapping model files).
- [x] _S mark downstream-facing FFI as such (was §1, one comment) so cleanup does
      not delete it: `cuMemPoolTrimTo`, `cuMemcpy{HtoD,DtoH}Async_v2`, `alloc_pinned`,
      `CudaEvent::elapsed_ms`, `CUDA_ERROR_NOT_SUPPORTED`
      (`cudarc-hip/src/driver/sys.rs:55,260,292,310`). Trivial insurance.
      CLOSED 2026-10-25: all five sites carry the marker (cudarc-hip b6016fbd
      marked cuMemPoolTrimTo; the rest - both cuMemcpy*_v2 wrappers +
      CUDA_ERROR_NOT_SUPPORTED in sys.rs, alloc_pinned +
      CudaEvent::elapsed_ms in driver/safe/core.rs - marked today).
      Downstream usage re-verified per symbol: memory_usage.rs matches
      the variant, graph_role.rs pins via alloc_pinned, gemv/cutile/
      routed_lora bench via elapsed_ms.
- [ ] ~M ROCm CI, remaining checklist (was §3, after P0 build-only):
    - [ ] Set up ROCm runner (self-hosted or `rocm/actions` if available)
    - [ ] Add test job running basic inference on ROCm
    - [ ] Cache ROCm build artifacts
    - [ ] Test matrix: gfx1100 (RDNA3), gfx1151 (RDNA3.5), gfx942 (CDNA3)

## P2. Merge prep — time-box, do after P0/P1 hygiene (large effort)

Order matters: rebase candle first (mistral.rs pins it by path), then pre-converge
mistral.rs hot zones, then merge.

### candle (26 behind at audit)

- [ ] ~L `candle-kernels/build.rs` graft (worst candle conflict). Fully
      fork-rewritten; upstream `e3f026d2` (cutile MoE) rewrote it too. Plan: keep
      `build_rocm`/`FFI_KERNELS`, add `CUTILE_FEATURE`/`moe_sources`. Same zone:
      `candle-nn/src/moe.rs` (fork `y_q8_1_scratch` FFI param vs upstream cutile MoE).
- [ ] ~L reconcile add/add `quantized/repack.rs` + `repack_x86.rs`: fork `e85cc63c`
      vs upstream `f80854c5` (#3697) look like the same aarch64 repack work.
      Diff the two, converge to one impl.
- [ ] ~L decide `candle-ug` keep vs drop before upstream `ddf1b879` deletes it.
- [ ] _S ROCm docs (remainder of the old §7): build instructions,
      `CANDLE_ROCM_PATH`/`CANDLE_ROCM_ARCH` (fix the hardcode-vs-knob issue
      in P3 first), ROCm example in `examples/`, known limitations vs CUDA
      (§Deferred + appendices). Can ride along with merge.

### mistral.rs (17 behind at audit, 59-file overlap)

- [ ] ~L pre-converge the `.cuh` macro names BEFORE the merge:
      `mistralrs-paged-attn/src/cuda/pagedattention.cuh:1089` still uses
      `VLLM_DevFuncAttribute_...` while upstream `0a29442f9` renamed the surface to
      `CUDA_CHECK` (fork already defines its own `CUDA_CHECK` at line 50).
      Cheap conflict avoidance; do first inside P2.
- [ ] ~L `cuda/gdn.rs`: worst merge in the repo. Fork +352/-140 on a 10.8k-line file
      with NaN probes woven in; upstream `ea9884815` adds cuTile GDN prefill + fp8
      tensor-core GEMV in the same files (`gdn.rs`, `gdn.cu`, `ffi.rs`).
      Consider stripping NaN probes (P0) before merging.
- [ ] ~L `pipeline/cuda_graph.rs`: fork +377/-16 (ROCm decode-graph stack) vs upstream
      `0a29442f9` +352 (CUDA-graph attention metadata/scheduling) and `ea9884815`
      `needs_logits` threading.
- [ ] ~L `paged_attention/layers/paged_attention.rs`: fork K/V split + Q8_0/Q4_0
      dispatch (+288/-29) vs upstream NVFP4 cache type (+411); `scheduler.rs`
      local +11 vs upstream +235; `config.rs` cache-type enum both sides.

## P3. Polish — low value, do last (small effort unless noted)

- [x] _S candle book fix (was §1): `candle-book/src/rocm/README.md` is a sketch;
      cites nonexistent knob `MISTRALRS_ROCM_COMPAT_INCLUDE` (build.rs hardcodes the
      include) and has a mangled sentence at lines 29-32. Fix, or retitle and shrink.
      (Do NOT document the knob until it exists.)
      CLOSED 2026-10-23 (candle a4651d9e): sentence fixed, knob ref removed.
- [ ] _S paged-attn proposed micro-opts (was §4), all reverted prototypes — measure
      before committing, none are approved yet:
    - `__ldg()` for KV cache loads: guard with `#if defined(USE_ROCM)` only; do NOT
      reference `__CUDA_ARCH__` in HIP path; verify gfx macro (`__gfx11__` likely
      never defined; HIP defines per-arch macros like `__gfx1100__`).
    - Async query load on RDNA: needs a single dynamic-shared-memory declaration;
      `__builtin_memcpy` is synchronous, so measure before/after.
    - PARTITION_SIZE 1024 for head_size 128: changes v2 grid shape; audit CUDA-graph
      partition math in `inputs_processor.rs` first, keep CUDA on existing path.
- [ ] _L dep drift (was §6): fork pins parquet 59 / fancy-regex 0.18, upstream moved
      to 60 / 0.19 (`1bbda281`, `4cd2f2f7`). Reappears every merge; batch with merge.
- [ ] _L 14 model files in overlap (was §6, qwen3_next MTP/GDN edits vs upstream tuner
      churn); 8 model files with xlora `unimplemented!()` also in overlap. Handle
      during merge, not before.
- [x] _S `qvm_split_k` metal `todo!` inside a `/* */` block (was §7,
      `mistralrs-quant/src/metal_kernels/mod.rs:1069`): finish or move to a tracking issue.
      CLOSED 2026-10-23 (mistral.rs a6075cafd): dead commented-out block removed.
- [x] _S cosmetic: feature-table indentation in `candle-nn/Cargo.toml`,
      `candle-examples/Cargo.toml`, `candle-transformers/Cargo.toml`. Absolute last.
      CLOSED 2026-10-23 (candle d8024a20).

## Deferred. Do not do on RDNA (record, revisit only with CDNA)

- FlashInfer ROCm integration — DEPRIORITIZED 2026-09-22. Not viable on gfx1151
  (RDNA lacks MFMA; AMD fork targets gfx942/CDNA only), no Rust bindings exist,
  ~2-3 weeks effort, and 0.6B measurements show it would not fix the actual decode
  bottleneck (launch overhead across 28 layers, not attention). AMD fork needs its
  full CMake/jinja build (simple kernel copy failed — config headers are generated).
  Revisit only for CDNA hardware. Proper path when revisited: build AMD FlashInfer
  as shared lib (`FLASHINFER_HIP_ARCHITECTURES=gfx942 python -m pip wheel . ...`),
  create `flashinfer-sys` bindgen crate + safe wrapper matching current FFI,
  link in `mistralrs-paged-attn/build.rs`, remove `#[cfg(not(feature = "rocm"))]`
  guards in `mistralrs-paged-attn/src/cuda/mod.rs` + `ffi.rs`, flip
  `mistralrs-core/src/flashinfer/mod.rs` cfgs to `any(cuda, rocm)`, enable
  FlashInfer decode in `paged_attention/layers/paged_attention.rs`. Unlocks FA3
  decode, paged KV, MLA decode on MI300X/MI325X.
- FP8 / MXFP4 kernels on ROCm — DEPRIORITIZED for RDNA 2026-09-22. MI300X has native
  FP8 MMA (2x BF16) but gfx1151 shows Q8 ISQ zero decode gain and Q4 *slower*
  (dequant overhead beats bandwidth savings; no FP8 tensor cores on RDNA). PagedAttn
  FP8 path works; quant-specific kernels don't. When revisited (CDNA): audit
  `mistralrs-quant/src/blockwise_fp8/ops.rs`, `pertensor_fp8/ops.rs`,
  `scalar_fp8/ops.rs`, `mxfp4/ops.rs` (reference `metal_kernels` for MXFP4), reuse
  `candle-kernels` ROCm FP8 primitives, guard `any(cuda, rocm)`.
- Speculative dflash on ROCm — UPDATED 2026-10-25 after parity audit;
  REFINED same day on a code re-read: dflash is NOT hard-gated off the
  hip role. DFlashDraftModel::load + attach_dflash compile and run on
  every build shape; enable_windowed_kv returns false on non-solo-cuda,
  so the drafter falls back to the EAGER zero-padded attention arm of
  draft_hidden_batch (dflash.rs:3323) ON THE SERVING GPU - functional,
  but never live-validated on hip and slow at long contexts (pads every
  draft step to max_ctx). The FAST path (bounded paged-windowed flash +
  drafter CUDA graphs) is solo-cuda+flash-attn only. Reasons + remaining
  work:
  (1) Graph capture already works on the hip role (graph_role dual
      module: Hip/Cuda enum, begin/end capture, launch, mempool, pinned
      host staging) — this is NOT a blocker.
  (2) Standard flash attention already works on the hip role via CK
      (rocm::ck_flash_attn, hdim 128/256, BF16) — NOT a blocker.
  (3) BLOCKER: dflash's attention core calls
      mistralrs_flash_attn::flash_attn_varlen_paged_windowed
      (dflash.rs:3691, paged_attention.rs:1841). That kernel is cutlass-based
      and cuda-only; mistralrs-flash-attn is #[cfg(feature = "cuda")].
  (4) CK KERNEL INVESTIGATION + TU LANDED 2026-10-25 (same day):
      API SURVEY of our CK (rocm-install): `fmha_batch_prefill`
      (the flexible-page paged API) is CDNA-ONLY - its async pipeline
      hard-wires buffer_load_*_lds (no RDNA instruction) with no
      Policy::AsyncCopy fallback, and amd_direct_load_global_to_lds
      is a SILENT NO-OP on non-gfx9 (a batch-prefill TU was written
      and compiles only on CDNA; removed, not in-tree).
      `fmha_fwd_splitkv` IS the RDNA path: both splitkv pipelines
      carry real __gfx11__ sections (PermuteWarpGemmCToA handles the
      P-tensor warp-gemm handoff on RDNA - the exact mechanism the
      async pipeline lacks), page size is a RUNTIME arg (the runner's
      128-divisibility rule is dispatcher-tile conservatism), and
      pool page 32 == kK1 tile = exactly one tile per page - NO POOL
      CHANGE NEEDED. LANDED: rocm_ck_flash_attn/
      fmha_fwd_splitkv_d128_bf16_mask_paged_gfx115.cpp - the gfx1151
      instantiation of the splitkv main kernel (FmhaFwdSplitKVKernel,
      BlockFmhaFwdSplitKVPipelineQRKSVS, TileFmhaFwdSplitKVTraits:
      kIsPagedKV=true, kHasUnevenSplits=true, kStoreLSE=true) PLUS
      the combine kernel (FmhaFwdSplitKVCombineKernel). COMPILES on
      gfx1151 (device code verified) and the dual fast build is green
      with it. Config learnings: WAVE32 forces combine kN1=32 +
      kLogMaxSplits=4 (the wave64/CDNA defaults fail the combine
      policy asserts); warp tiles 16x16x16 on both gemms;
      SimplifiedGenericAttentionMask<true> for runtime masks/windows.
      REMAINING updated after the 2026-10-25 bring-up attempt:
      PLUMBING LANDED (dispatcher ck_flash_attn_splitkv_fwd + hpp ABI,
      FFI decl, rocm::ck_flash_attn_splitkv Rust wrapper with o_acc/
      lse_acc workspaces + sm-count split heuristic capped by KV
      chunks (splits beyond the chunk count hand CTAs empty ranges),
      paged/varlen/windowed parity test) - but the HARDWARE VERDICT
      IS NEGATIVE: the splitkv MAIN kernel memory-faults on gfx1151
      in its BASE group-varlen path. Proven with a standalone harness
      (LocalAI/scratch/splitkv-debug/, preserved; knobs are -D flags:
      PAGED/HQ/HKV/QLEN/KVLEN/BATCH/WINDOW_L/WINDOW_R/MASK_T): faults
      identically with (a) paged AND non-paged kv, (b) no-mask AND
      masked, (c) splits=1 AND multi-split, (d) full 64-row Q tiles
      (no Q padding), main-kernel-only (combine skipped). RULED OUT
      along the way: empty-split ranges (KV-chunk cap + 1-split both
      fault) and seqlen_k_ptr null-reads (the ABI gained a nullable
      per-batch seqlen_k; populating it changed nothing). The COMBINE
      kernel is UNTESTED - the main faults first. Fault address is
      small and shifts with the D-pad config (LDS-space signature;
      the launch requests only 9216B group memory while the K(4KB)+
      V(8KB) tiles suggest ~12KB+ - smem under-budget suspected,
      unproven). The
      NWarpSShuffle splitkv variant does not compile at these tiles
      (MWarp==1 + gfx11 wmma-utility asserts - a separate config hunt).
      AMD ships ZERO gfx115/gfx1151 splitkv instantiations anywhere
      (their generate.py emits none) - the family is unvalidated on
      RDNA, like batch_prefill. The test is #[ignore]d with the
      verdict; the wrapper carries a do-not-call note; dispatcher/
      TU/FFI ship inert so a CK fix or working config is a small diff.
      OPTIONS IF REVISITED: (i) root-cause the smem budget in
      BlockFmhaPipelineQXKSVSCustomPolicy vs actual LDS use and try a
      custom Policy, (ii) sshuffle-variant tile hunt, (iii) the third
      paged API (fmha_fwd_pagedkv), (iv) a newer CK drop. Until one of
      those pays off, DFlash-on-hip keeps the EAGER fallback
      (functional; see the status refinement above) and the CK paged
      kernel lane is PARKED with a precise blocker record.
  (5) The functional fallback ALREADY EXISTS in code (the eager arm
      above); remaining functional work is only a live hip validation,
      not new machinery.
  Generic speculative driver/verifier already support ROCm
  (`any(cuda, rocm)`); the cuda-only surface is the dflash fast path
  (dflash.rs windowed arm, paged_attention.rs:1841). Custom kernels
  (dflash_context.cu, dflash_selector.cu, dynamic_conv.cu) are
  hipcc-portable via the existing compat shim.
  Revisit trigger: a DFlash draft model in hand, or after the CK
  paged-windowed kernel lands.

## Appendix A. Measured baselines (ROCM_PLAN 2026-09-22, keep for context)

`mistralrs bench auto -m Qwen/Qwen3-0.6B`, dev profile (`opt-level=3`), gfx1151:

| Config | TTFT | Decode TPOT |
|--------|------|-------------|
| BF16, batch 1, graphs on | 30.7ms (128 tok) / 75.7ms (1024 tok) | 10.88ms |
| BF16, batch 1, graphs off (`MISTRALRS_CUDA_GRAPHS=0`) | 29.4ms | 12.26ms |
| Q8_0 ISQ | same | 10.88ms (no change) |
| Q4_K_M ISQ | 27.0ms | 11.58ms (worse) |
| BF16, batch 4 | 30.5ms | 10.81ms (no scaling) |
| CK prefill (`MRS_DEBUG_CK=1`) | HIT every layer | SKIP by design (seq_len=1) |
| Release binary, BF16 batch 1 | 30.26ms | 10.44ms (~4% over dev) |

Conclusions at the time: prefill healthy (CK hits every layer, 8x tokens -> 2.5x
time); decode overhead-dominated not bandwidth-bound (Q8 changes nothing, batch 4
changes nothing, Q4 dequant hurts; ~11ms/token is per-layer launch overhead +
small-GEMM inefficiency across 28 layers); HIP graphs work (+12% decode, on by
default); FlashInfer would not fix this bottleneck class (attention is a small
fraction of decode at 0.6B).

Status snapshot at merge: core kernels (top-k, MoE router, MoE gemm, rotary, sort,
graph) via `hipcc` + candle `rocm_compat` headers = working; CK FlashAttention
prefill (BF16, head_dim 128/256) = working; quant (GPTQ/AWQ/HQQ/GGUF/bnb/FP8/MXFP4)
via candle = mostly working; LoRA/X-LoRA = working; PagedAttention core (decode
v1/v2, reshape_and_cache, gather_kv, copy/swap blocks, FP8 KV via `-DENABLE_FP8`,
`mistralrs-paged-attn/build.rs:250-352`) = working; device selection
(`cuda_if_available(0)` via `cudarc-hip`) = already worked. CUDA-only: FlashInfer
decode/FA3/MLA/Sinks, GDN FlashInfer SM90 prefill, Cutlass/DeepGEMM/Marlin MoE,
most `mistralrs-quant` FP8 kernels, speculative dflash/FA3 paths.

## Appendix B. Build health (re-verified 2026-10-06; policy: all shapes supported)

Toolchain (2026-10-08, measured): mold links everything (RUSTFLAGS
`-fuse-ld=mold` + build.rs `.so` links, thin-LTO verified); sccache
wraps rustc + host CC/CXX (100G); ccache wraps hipcc/nvcc `-c` steps
(100G, opt-in `MISTRALRS_CCACHE_GPU=1`); nvcc must NOT go through
sccache (phantom hits on failed compiles). One-crate release edit:
17m -> 7m05s after the toolchain landed. Probe lane policy (same
day): diagnostics build `--profile fast` + standalone server on a
spare port (1244), never a prod restart, never benchmarked;
`release` reserved for benches + prod swaps.

| Configuration | Status |
|---------------|--------|
| CPU-only (`cargo check -p mistralrs-core -p mistralrs-quant`) | Pass (4B serves on cpu, benched) |
| ROCm (`cargo check -p mistralrs-core -p mistralrs-quant --features rocm`) | Pass, prod serves on :1235 |
| Dual (`cargo check/build --features cuda,rocm`) | Pass, serves 4B/16B/27B on the 3090 and the APU; decode graphs live on both roles (NVIDIA role: non-hybrid models; hybrid models gated eager pending the restore hoist) |
| CUDA-only (`--features cuda`) | Pass (revived 2026-10-07, role-identity Phase 1): checks clean, release binary links with zero GPU DT_NEEDED, serves 27B ISQ-Q4K correctly (quality verified, resident memory matches dual). Known gaps, all journaled: ~3x decode-vs-dual perf gap (open tune item, companions exonerated by swap test), 4B decode-graph capture aborts (workaround MISTRALRS_CUDA_GRAPHS=0, debug queued), MTP off (role caps, like dual-NVIDIA) |

CPU fallback notes: `mistralrs-paged-attn` is a required dependency of
`mistralrs-core` (types always available, GPU kernels still feature-gated),
`indexed_copy` clamp is cfg-gated, `PAGED_ATTENTION_V2_PARTITION_SIZE` has a
fallback const for non-CUDA/ROCm builds (also fixes Metal-only builds).
Re-verify after the merge in section P2.

## Appendix C. Architecture notes (condensed from ROCM_PLAN)

- PagedAttention ROCm build: CUDA `.cu` files compiled with `hipcc` via
  `mistralrs-paged-attn/build.rs:250-352`. Works because kernels use basic warp
  shuffles + fp16/bf16 math, no CUDA tensor-core intrinsics. Enabled: v1/v2 paged
  attention, reshape_and_cache, copy_blocks, gather_kv, FP8 (`-DENABLE_FP8`).
  Excluded: FlashInfer (ldmatrix/cp.async), FA3, MLA, Sinks (need tensor cores
  RDNA lacks).
- Future ROCm-native path (not started): separate `.hip`/`.cpp` with CK
  (Composable Kernel), rocBLAS/hipBLASLt, rocRAND/hipRAND. Current transpilation
  gets ~80% compatibility with minimal maintenance.
- Feature flags (`mistralrs-core`, `mistralrs-paged-attn`, `mistralrs-quant`):
  `cuda` (NVIDIA), `rocm` (AMD HIP), `metal` (Apple) — mutually exclusive per binary.
- Tracking milestones: M1 CPU fallback + gfx1151 baseline (done); M2 CK
  sliding-window + bigger-model bench (done); M3 MoE optimization + CI (router
  done, CI open — P0/P1).
- References: FlashInfer ROCm, AMD Composable Kernel, candle-kernels ROCm,
  hipBLASLt docs, `mistralrs-paged-attn/build.rs:250-352`, `src/rocm_ck_flash_attn/`.

## Kernel-level paged attention optimizations (from Gemini analysis, Oct 31)

### [H] Coalesced memory access in paged attention cache lookup
- **File:** mistralrs-paged-attn/src/cuda/pagedattention.cuh
- **Issue:** Threads in same warp read from different physical blocks at
  BLOCK_SIZE*x stride (non-coalesced). For Q8_0 KV (x=16, block_size=32),
  consecutive threads access memory 512 bytes apart.
- **Fix:** Reorder memory access so threads in same warp read contiguous
  elements of K/V vectors. Requires changing cache layout from
  [num_blocks, num_kv_heads, (head_dim/x), block_size, x] to
  [num_blocks, num_kv_heads, block_size, (head_dim/x), x] or similar.
- **Impact:** Better L2 cache utilization, 10-20% decode improvement.
- **Effort:** High (cache engine + kernel + block table changes)
- **Status:** Not started. Needs profiling with rocprof-compute first.

### [M] Dual-backend thread layout (32 vs 64 threads)
- **File:** mistralrs-paged-attn/src/cuda/pagedattention.cuh
- **Issue:** WARP_SIZE hardcoded to 32. AMD wavefronts are 64 threads.
  NUM_THREADS=128 gives 2 wavefronts on AMD (fine) but tiling assumes
  32-thread warps. Thread group size MAX(WARP_SIZE/BLOCK_SIZE, 1) may
  not be optimal for 64-thread wavefronts.
- **Fix:** Parameterize WARP_SIZE and thread block dimensions by backend
  (32 for CUDA, 64 for ROCm). Adjust tiling accordingly.
- **Impact:** Better CU utilization on AMD, 5-10% improvement.
- **Effort:** Medium (template parameterization)
- **Status:** Not started.

### [M] Shared memory tiling for attention scores
- **File:** mistralrs-paged-attn/src/cuda/pagedattention.cuh
- **Issue:** Block table lookups and KV cache reads go directly to global
  memory. No prefetching of next block's metadata.
- **Fix:** Prefetch block table entries and KV scale data into shared
  memory (LDS on ROCm) while processing current block. Overlap
  computation with memory fetch.
- **Impact:** Reduced global memory latency, 5-15% improvement.
- **Effort:** Medium (shared memory staging)
- **Status:** Not started.

### [H] Kernel memory access tuning (mmvq_q8_0_plain)
- **File:** candle-kernels/src/mmvq_gguf.cu
- **Issue:** mmvq_q8_0_plain at 242 GB/s vs 1221 GB/s peak (20% utilization).
  Uses 16-bit loads (get_int_from_int8) instead of 32/128-bit.
  Q8_0 block layout (34 bytes) is not 16-byte aligned.
- **Fix:** Vectorized loads where alignment allows, shared memory tiling,
  warp-level reduction optimization, occupancy tuning.
- **Impact:** 10-30% decode improvement (dominant kernel).
- **Effort:** Very high (profiling + iterative tuning)
- **Status:** Not started. Needs rocprof-compute profiling first.
