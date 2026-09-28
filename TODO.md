# TODO / PLAN: candle + mistral.rs forks

Tracking doc for local fork work on `wip/pstreet-rocm` (both repos).
Audited 2026-09-25. Merged `mistral.rs/ROCM_PLAN.md` (dated 2026-09-22) here on
2026-09-25; `ROCM_PLAN.md` deleted after merge. Sorted by impact x ease on
2026-09-25 (old section numbers in parens). ROCm measurements are on Radeon
8060S gfx1151 unless noted.
Divergence at audit time:

- `candle`: 56 commits ahead / 26 behind `huggingface/candle:main`
- `mistral.rs`: 111 ahead / 17 behind `EricLBuehler/mistral.rs:master`

Re-check before relying on the merge section:

```bash
cd candle && git fetch origin main && git rev-list --count origin/main..HEAD && git rev-list --count HEAD..origin/main
cd mistral.rs && git fetch origin master && git rev-list --count origin/master..HEAD && git rev-list --count HEAD..origin/master
```

Legend: [ ] open  [x] done  ~ high/medium value (tags collapsed in this file —
priority order below is authoritative, not the prefix)  _ low value
S/M/L = effort. P0 = do first, P1 = do next, P2 = merge prep (time-boxed),
P3 = polish, Deferred = do not do on RDNA.

---

## P0. Do first — highest impact per effort (days)

- [x] ~M 35B trio quant A/B: Q4_K_XL vs Q5_K_XL vs Q8_0 (done 2026-09-25, was §4). Retargeted: no Q4_K_XL
      exists for the orcarouter 27B serving default, so ran the on-disk 35B trio
      (same base, isolates quant): essay 600tok temp0.3, MTP n=2 greedy, graphs on,
      Q8_0 KV 2048MB, port 1236, prod stopped for the window.
      Q4_K_XL (22.85GB): decode 50.81 T/s (51.55/50.88/50.01), prefill 725 T/s.
      Q5_K_XL (27.16GB): decode 47.11 T/s (47.71/47.40/46.22), prefill ~630 T/s.
      Q8_0 (37.80GB): decode 45.47 T/s (45.91/44.96/45.53), prefill 578 T/s.
      +65% bytes (Q4->Q8) buys only -10.5% decode: strongly sublinear, decode is
      overhead-dominated on 35B MoE, bandwidth a minor component. Q4 wins on speed
      AND memory.
      Quality 4/4 correct on all three arms (primes, is_prime, 72-min train,
      bullets), no low-quant regression.
      Loader lesson (worth keeping): Q8 heap-staging (`MISTRALRS_GGUF_NO_MMAP=1`)
      transiently holds weights twice (~76GB) and fails device mapping even solo
      (avail 57-62GB vs need ~73GB); mmap path (`NO_MMAP=0`) loads the same 37.8GB
      fine. Decode config identical across arms (MTP n=2 greedy, graphs on, Q8_0
      KV 2048MB), so tok/s numbers are comparable; only the load path differed.
      Follow-up: live `models/mistralrs.toml` runs 35B at Q5_K_XL while the deploy
      copy already points at Q4_K_XL — flip prod to Q4 (+8% decode, quality-clean)
      or record why Q5 stays. Follow-up: live `models/mistralrs.toml` runs 35B at
      Q5_K_XL while the deploy copy already points at Q4_K_XL — flip prod to Q4
      (+8% decode, quality-clean) or record why Q5 stays.
- [x] ~S `ATTN_PROBE` removed (done 2026-09-25, was §1). Verified push-only
      across the crate (no reader; the "end-of-forward dump" in the comment never
      existed — aspirational since `506f72b8a`). Deleted the static + both push
      sites in `vision_models/qwen3_5/text.rs`; the live `trace_layer_nan` path
      stays for future debugging. `cargo check -p mistralrs-core` clean.
- [x] ~S candle arena overflow/consumed getters (closed 2026-09-25, was §1).
      Audit premise was stale: both getters have live callers and the
      grow-and-retry loop already exists in
      `mistralrs-core/src/pipeline/cuda_graph.rs:2544-2616` (2 attempts,
      post-forward overflow check, regrow consumed + consumed/8 + MIN,
      re-capture, per-bucket observation for next sizing). `consumed` also feeds
      the `[cudarc]` diagnostic logs; the internal alloc-count counter feeds the
      overflow log line. No code change; kept as-is.
- [x] ~M unit tests for the demand-load router estimator (done 2026-09-25,
      was §3). 13 tests in `mistralrs-core/src/lib.rs` `mod tests` (10 for
      `estimate_load_bytes`: weights+headroom, heap-staging doubling, shard+mmproj
      sums, Mb/BestEffort/Utilization KV incl. 16GiB cap, ContextSize/missing
      unknowns; 3 for the eviction loop early returns: disabled policy, unknown
      model, idle noop) plus a `ModelSelected::gguf_for_tests` ctor (fields are
      private to `model_selected`). Env-var reads serialized via lock + guard.
      `cargo test -p mistralrs-core --lib` targeted run green; fmt clean.
- [x] ~M ROCm CI job, build-only slice (done 2026-09-25, was §3).
      Added `.github/workflows/ci_rocm.yaml` (mirrors `ci_cuda.yaml` gating:
      same-repo PRs + dispatch): `cargo check -p mistralrs --features rocm`
      then `cargo test -p mistralrs-paged-attn --features rocm`. Both commands
      validated on gfx1151. Two lessons baked in: link needs
      `LIBRARY_PATH=$ROCM_DIR/lib` (`LD_LIBRARY_PATH` is not enough for rust-lld;
      derived from `which hipcc`, no `/opt/rocm` assumption), and the
      `copy_blocks_uses_device_metadata_and_view_offsets` test is skipped (fails
      deterministically on ROCm, see P1 GPU-tests item). Runner labels
      `[self-hosted, Linux, X64, gpu, rocm]` — no such runner exists yet, so the
      remaining checklist stays in P1 (runner, inference test job, artifact
      cache, gfx1100/1151/942 matrix).
- [x] ~M per-call `env::var` cached with OnceLock (done 2026-09-25, was §2,
      template `fast_mmq_allowed()` already existed). Converted all 8 sites to
      function-local statics / shared helpers: `nan_probe`+call-site gate
      (text.rs), `MRS_PROFILE` (engine), `debug_pa()` helper for x3 PA sites,
      `debug_ck()` in attention (rocm-gated, shared with rocm/mod.rs),
      `fused_ffn/qkv_disabled()` (ops.rs, cfg-gated), `MRS_GDN_KERNEL`
      Option<String> (gdn), `MRS_ARENA_ALLOC_LOG` (cudarc-hip), `CANDLE_LT_COMPUTE`
      (candle, matching the existing `LT_MIN` idiom; the other read at
      mod.rs:2274 is a cold error path, left). Default + rocm checks green with
      unchanged warning baselines (6 / 54); fmt clean.
- [x] ~M unconditional `eprintln!` in production capture path (done 2026-09-25,
      was §1). All 6 converted to `tracing::debug!(target: "mistralrs", ...)`
      (off by default, visible with `-v`); the arena-grew line was removed as a
      dup of the existing debug line with the same numbers. Docs updated:
      `MRS_DECODE_ARENA_OVERRIDE` row now says debug-logging. Verified under
      `--features rocm` (the file is cfg-gated out of default builds — default
      check was vacuous): check green, all 30 `cuda_graph::tests` pass.

## P1. Do next — high impact, slightly more work (week)

- [x] ~M attention projection follow-up (answered 2026-09-25, was §4). Fresh
      rocprofv3 decode-only profile of the 35B Q4_K_XL (attach mode needs
      `ROCP_TOOL_ATTACH=1` on the target + `ptrace_scope=0`): premise stale.
      mmvq Q8_0 attention projections (attn_qkv/qkvz 17.8MB, ssm_out/gate/
      attn_output 8.9MB per layer, all Q8_0 in UD-Q4_K_XL) run 105-237 GB/s,
      comparable or better than moe_gemv (113); no 40 GB/s pathology, no
      tile-GEMM. Only in_proj_ba-style small calls lag (~105 GB/s, worth ~0.5%
      if heroic; skipped). The real fat found instead: the MTP drafter reads
      the full 540MB Q8_0 lm_head at batch 1, 2x per cycle (n=2 is sequential,
      cannot merge), 2.28ms each = 4.6ms/cycle ~10% of decode, at 237 GB/s
      (kernel already near peak; bytes are the cost). Fix shipped: new
      `mtp_draft_lm_head_isq` TOML/CLI knob (RuntimeOptions -> MtpConfig ->
      server builder no longer clobbers an explicit value with the global ISQ).
      Rejection sampling keeps output quality provably target-exact. A/B
      (same binary, 35B Q4_K_XL, essay 600tok temp0.3, 3 trials): draft head
      Q4K 53.12 T/s vs Q8_0 51.76 (+2.6%), acceptance flat 66.6% vs 66.2%
      (argmax robust to quant noise), quality 4/4 (primes/is_prime/72min/
      bullets; text diffs are benign batch-shape fp butterfly). Optional prod
      follow-up: set `mtp_draft_lm_head_isq = "Q4K"` in prod mistralrs.toml.
      gemma-4 dense-MLP half of the item stays open for gemma-4 serving work.
- [x] ~M decode-graph capture on ROCm: real test, not just the 2 CPU unit tests
      (done 2026-09-25, was §3, `cuda_graph.rs:3867-3910`). 4 new GPU tests
      (`--ignored --test-threads=1`, `Device::new_cuda(0)` works under the HIP
      fork): prelaunch-panic keeps LRU entry + reports Err, arena overflow
      regrows via `MRS_DECODE_ARENA_OVERRIDE` squeeze + records observation,
      memory-pressure eviction releases oldest first + survivor replays,
      arena clamp stays within device cap. All 8 ignored + 31 unit pass.
      Fixed alongside: `replay()` prelaunch now `catch_unwind`s, reinserts the
      entry and returns Err (was: LRU slot lost + outer `StdMutex` poisoned,
      wedging graphs; note the `CUgraph` itself was never orphaned, `Drop`
      destroys it). `capture_cuda_decode_graph` forward now discards the
      in-flight capture on panic before reporting. Err paths already fall back
      to eager via `disable_cuda_decode_graph`. Parallel GPU tests on one
      device abort in the HIP driver (stream-capture collision), hence serial.
      Negative control verified: new panic test fails without the fix
      (panics at the epoch `expect`), passes with it.
- [x] ~M GPU tests for Q8_0/Q4_0/QJL KV-cache read/write paths (done 2026-09-25,
      was §3). 4 round-trip tests in `mistralrs-paged-attn/.../backend/
      paged_attention.rs` (write + `gather_kv_cache` + compare): native f32
      exact, Q8_0 < 0.01, Q4_0 < 0.1, Q4_0+QJL residuals < 0.1. Found + fixed
      2 real Q4 bugs: (a) gather K strides used elementwise x-math on packed
      nibbles, reading head>=1 128B too far (`gather_kv_cache_kernel.cu`,
      new `k_q4_*_stride`); (b) plain-path quant rounded negatives wrong,
      `uint8_t((q-0.5)+8)` truncates toward zero after biasing, every
      negative off by one and [-8,-7.5) UB (`reshape_and_cache_kernel.cu`
      x4, `q4_utils.cuh` x2; now int-truncate then bias, Q8-style).
      `block_scales.rs` registry: 3 CPU tests (roundtrip, clone/offset-view
      identity, no collision). `copy_blocks` ROCm failure fixed, not skipped:
      HIP fork mints a fresh Arc per alloc so `Arc::ptr_eq` on slice streams
      never holds; compare `cu_stream()` handles like `gdn.rs:724`.
      Un-skipped in `ci_rocm.yaml`; full `cargo test -p mistralrs-paged-attn
      --features rocm` green (8/8, incl. CPU mirror + both copy_blocks).
- [x] ~S MTP speculative decoding unit test (done 2026-09-25, was §3). 8 new
      CPU tests. `qwen3_5/speculative.rs`: `mrope_at` per-dim reads + both
      missing-slot errors, `capture_view` rank-3 mrope passthrough / rank-2
      hidden unsqueeze / both rank bails (struct now derives Debug),
      `dflash_speculative_batch` Tokens arm -> one proposal per row.
      `speculative/proposer.rs`: `sample_draft_rows` greedy argmax picks
      per-row tokens, appends contexts, q = argmax softmax prob (NOT 1.0 -
      greedy drafts carry the real proposal probability, e.g. 0.88 for a
      3.0-logit spike over 4 tokens; verifier relies on this), plus the
      batch-mismatch bail. Sequence built via the sequence.rs-test
      `new_waiting` boilerplate (32 args; tokio Mutex for the group).
      Full `speculative` slice green: 92 rocm / 86 default.
- [x] ~S `CANDLE_LT_COMPUTE` silently ignored by large-batch Q6_K prefill
      (done 2026-09-25, was §2). `dequantize_matmul_f16` forced
      `set_gemm_reduced_precision_f16(true)` unconditionally (and it is the
      flag's only setter: sticky-global, F16 accumulation for every f16 GEMM
      once any large-batch quantized matmul runs). Now honors
      `CANDLE_LT_COMPUTE=32` (OnceLock-cached): =32 keeps F32 accumulation
      there too, unset keeps the llama.cpp-matching F16 WMMA default.
      candle README updated: the knob covers hipBLASLt AND the dequant-to-F16
      quantized matmul. Behavior note: the prod wrapper exports =32, so prod
      prefill (all quant dtypes with b*m > 8; fast_mmq off) now accumulates
      F32 instead of F16 - strictly more precise, possibly slower WMMA;
      A/B answered 2026-09-25 (same 10:10 binary, 35B Q4_K_XL, prod-style
      env, 3 trials short + 1 cold long-prompt; prefix cache invalidates
      repeats, only cold counts): LT=32 vs unset, short 760 vs 762, long
      1404 vs 1417 T/s - FLAT, F32 accumulation is perf-neutral, prod flip
      safe. Third arm (no env at all, fast_mmq on, defaults) measured the
      prod-style routing itself: 896 short / 1462 long / decode 53.5 -
      i.e. prod's MRS_NO_FAST_MMQ=1 + DMM_F16_MIN=8 dequant routing costs
      ~15% short-prompt prefill (~3% at 1600-row prompts) vs fast_mmq.
      18k-token follow-up (covers the old 23k-scale claim and the 4096
      chunk cap): 1492 vs 1492 T/s, tie. Wrapper updated 2026-09-25:
      MRS_NO_FAST_MMQ/CANDLE_NO_FAST_MMQ/CANDLE_DMM_F16_MIN removed
      (fast_mmq back on by default; the MMQ-off rationale predates the
      fork's mmq_gguf work), CANDLE_LT_COMPUTE=32 kept (Lt stability,
      never tested without). Applies at next prod restart, which also
      picks up today's binary (graph panic fix, Q4 KV fixes, draft-head
      knob, LT_COMPUTE honored on dequant).
      Status: fully live. The healthcheck restarted prod at 10:42:25 onto
      the new binary + fast_mmq; the user manually restarted at 11:11 to
      pick up the final wrapper (CANDLE_LT_COMPUTE gone, only
      ROCBLAS_USE_HIPBLASLT=1 remains). Smoke: 27B decode 19.4 T/s (best
      recorded; 13.9-17.8 historical), prefill 259 short-prompt.
      Follow-up same day: the "=32 required" claim itself is stale. New
      `candle-core/examples/lt_probe.rs` swept 80 Lt shapes (b=1 m 256-8192
      k/n 2048-16384, 4 odd, b 3-16 batched) x 16F vs 32F: zero heuristic
      failures either arm, 16F mean +1.2% (not slower). The INTERNAL_ERROR
      story dates from the pre-TheRock hipBLASLt. CANDLE_LT_COMPUTE removed
      from the wrapper too (keep the knob + lt_probe for post-ROCm-update
      regression checks); candle README rationale fixed. Remaining
      untested wrapper knob: ROCBLAS_USE_HIPBLASLT=1 (rocBLAS-internal).
      Also answered same day: serving A/B flat (888/51.4/1491 vs
      896/53.5/1492 var off - the 12s decode profile has ZERO plain
      rocBLAS GEMM kernels; all-quantized weights never route there,
      activations are BF16 throughout via custom kernels). Microbench
      (lt_probe through plain rocBLAS): +13% f16 / +19% bf16 mean when
      GEMMs do reach rocBLAS - e.g. dense BF16 safetensors models. Kept
      as no-op insurance with an evidence-based comment.
      CORRECTION later same day: the fast_mmq removal above was
      validated on the 35B (MoE Q4/Q8) only and was a 43% long-prompt
      prefill REGRESSION on the 27B (dense Q6_K, the prod default):
      18k cold 27B A/B, MMQ off 426 T/s vs on 243 T/s. See the GDN
      prefill entry below for the profile that caught it. RESOLVED
      same day (b19e4dea2): per-dtype MMQ dispatch - Q6K hands
      batches > 256 rows to candle's dequant-to-F16 + hipBLASLt path,
      keeps MMQ below (m-sweep on the 27B: mmq wins at ~112 rows
      139 vs 82, dequant wins from ~260 up 252 vs 148), all other
      dtypes keep MMQ at every batch (35B evidence). Sweep verify:
      199/216/301/389/444/448/426 T/s at 115..15032 rows - dominates
      both static arms at every point. 35B unchanged (893 short,
      1425 18k), quality 4/4 both models. Wrapper trio removed again
      (only ROCBLAS_USE_HIPBLASLT=1 remains); prod restarted onto it
      12:50 and verified: short 258 T/s (mmq side), 18k cold 423 T/s
      (dequant side), decode 18.6. Interim revert commit 0788e6898
      kept in history.
      Default + rocm `cargo check` green, fmt clean.
- [x] 2026-09-25 GDN prefill on ROCm (was §4, P1 #6): CLOSED, measured
      not worthwhile. rocprofv3 on an 18k-token prefill of the 27B (the
      GDN-heavy prod default; 48 GDN layers x 5 chunks): the key-major
      warp recurrence kernel is only 4.93s / 5.6% of 88.1s GPU time.
      Even a perfect GDN prefill kernel saves ~5% of prefill. Old
      premise doubly stale: the fork already has native vmajor1/2/4/8
      prefill kernels (cuda/gdn.rs), gated to compute major 9 -
      NVIDIA Hopper sm_90 (mistral.rs is CUDA-first; the numeric 9
      coincidentally also matches CDNA3 gfx942 under HIP's synthetic
      caps, untested there) - on gfx1151 (synthetic major 11)
      v_major_state_supported is false, so models use GdnKeyMajor +
      the backend.rs warp/chunked dispatcher (MISTRALRS_GDN_KERNEL
      knob). Revisit only if a GDN model's prefill profile ever shows
      the recurrence above ~20%.
      The SAME profile found the real prefill bottleneck: mul_mat_q
      Q6_K at 82.5% (72.7s, 2040 launches, ~11 effective TFLOPS vs
      ~30 TFLOPS from dequant->F16->hipBLASLt per lt_probe). 18k cold
      A/B on the 27B (dense Q6_K): fast_mmq OFF 426 T/s (42.4s) vs
      ON 243 T/s (74.1s) = fast_mmq costs the 27B 43% long-prompt
      prefill, while it GAINS the 35B (MoE Q4/Q8) +15% short. The
      wrapper A/B had validated on the 35B only - wrong model for the
      prod default. RESOLVED same day via the per-dtype dispatch
      (b19e4dea2, see the wrapper-cleanup entry above for the full
      evidence and prod verification).
- [x] 2026-09-25 long-context prefill baseline (both prod models,
      post-dispatch binary, cold prompts, max_tokens=1, test server):
      35B (MoE Q4/Q5): 18k 1425 T/s / 12.7s, 32k 1247 / 26.0s,
      48k 1116 / 43.4s (-22% at 48k vs 18k; MoE GEMM cost dominates,
      O(n^2) attention share stays small at ~12 chunks of KV).
      27B (dense Q6K): 18k 423 / 42.7s, 32k 325 / 99.9s, 48k 361 /
      133.9s (32k below 48k is single-shot noise, ~+-10%). Scaling is
      as gentle as the 35B, but absolute TTFT at range is 3-4x worse:
      100-134s vs 26-43s. Routing note: document-scale prompts belong
      on the 35B; the 27B's case is decode (19.4-19.6 T/s best-ever)
      and short-context latency. No prefill regression from the
      dispatch change at any length (18k points in-band with the
      morning arms).
- [x] 2026-09-25 hygiene sweep (c2cf6261e): default-feature
      `cargo clippy --workspace --tests --examples -- -D warnings`
      green for the first time on the branch (plus rocm workspace
      check, fmt, and the test suites: paged-attn 8/8, quant 295/295,
      cuda_graph 31+8 GPU, speculative 92/14-ignored, block_scales
      3/3, estimator 10/10). All fixes behavior-preserving; GPU-
      generic paths stay compiled under rocm per review call (cfg
      tightenings reverted). Deliberately NOT pursued:
      `--features rocm` clippy -D (fork-wide style debt in GPU-gated
      code; fixing it invites path-removal churn). Blanket full-
      workspace test skipped per the no-download rule (HF
      integration tests); targeted slices cover all touched paths.
      Binary note: prod runs the 12:37 build; these lint-equivalent
      fixes ride the next natural rebuild.
- [x] 2026-09-25 env namespace cleanup (candle ff97e7b8, mistral.rs
      61dc90f13): repo ownership rules - candle reads only CANDLE_*,
      mistral.rs reads only MISTRALRS_*. candle: MISTRALRS_MANAGED_
      WEIGHTS->CANDLE_MANAGED_WEIGHTS (allocator side; wrapper sets
      both), FORCE_AVXVNNI/AVX2 and MMQ_DEBUG renamed to CANDLE_,
      dequant_f16_min_rows() exposed so the Q6K handoff consumes the
      resolved threshold via API (no more duplicated 256 const).
      mistral.rs: all 16 MRS_ vars -> MISTRALRS_*, 5 unprefixed
      vars prefixed (NAN_PROBE, LAYER_PROFILE, RING_CONFIG,
      KEEP_ALIVE_INTERVAL, MCP_CONFIG_PATH), CANDLE_NO_FAST_MMQ reads
      dropped from the dispatch (MISTRALRS_NO_FAST_MMQ is the sole
      override), wrapper + 9 doc files updated. Prod restarted on the
      08:49 build and verified: zero MRS_ vars in env, decode 19.5 /
      short 260 / 18k 421 T/s all in-band, RSS 23.3GB confirms
      CANDLE_MANAGED_WEIGHTS active. Tests green: paged-attn 8/8,
      quant 295/295, cuda_graph 31+8, speculative 92, estimator 10/10,
      candle quantized 6/6.
- [x] 2026-09-25 Q6K MMQ investigation (code read, kernels/mmq_gguf/):
      WHY Q6K mmq plateaus ~11 effective TFLOPS on gfx1151 while
      dequant->F16->hipBLASLt reaches ~30. On RDNA3.5 the AMD_WMMA
      path runs wmma over s8 tiles (int accumulators; load_tiles
      stores __vsubss4(ql|qh, 0x20202020)-biased int8). Q6K is the
      ONLY quant with 16-value scale groups, which forces a dedicated
      vecdot (vec_dot_q6_K_q8_1_mma): K=16-value mma tiles (16x4)
      vs K=32 (16x8) for every other dtype (Q4_K/Q5_K ride the shared
      vec_dot_q8_1_q8_1_mma; Q8_0 has its own; all 32-value, 2-mul
      fixup dmA.x*dsB.x*C.x[l]). The Q6K fixup is 3 muls with two
      smem scale fetches (sc[k01/4] per 16 values + x_df[i]) running
      at 2x the frequency: ~2x wmma issue rate + ~2x fragment
      traffic + ~3x scalar fixup, plus the heaviest extraction
      (ql/qh bit juggling, 2 stores per 4 values). Ceiling check:
      at dense large-M even Q8_0-class mmq only TIES the dequant
      path (35B 18k all-mmq vs all-dequant: 1492 vs 1492 T/s), so a
      perfect Q6K mmq cannot meaningfully beat the ~30 TFLOPS the
      dispatch already gets from dequant+Lt at large M. Residual gap
      is mid-M (256-1024 rows) where dequant fixed cost bites.
      Battle scar on file: mmq_x=24 miscompiles on gfx1151 (skipped
      in the launcher). Structural fix if ever needed: lossless-ish
      load-time requant Q6K -> Q8_0-style (6-bit values fit int8;
      pair the 16-value scales per 32, rounding error well under the
      6-bit grid step) to ride the generic mma path - costs +29%
      weight memory (22.4 -> 28.7GB for the 27B, does not fit the
      APU budget alongside KV). NOT pursued. Follow-up candidate
      (~S): per-dtype mmq microbench at dense m=512..4096 to close
      the dispatch table (Q4_K/Q5_K dense-M are structurally Q8_0-
      class but unmeasured; 35B evidence is MoE-grouped and does not
      cover dense M). DONE same day, see the next entry.
- [x] 2026-09-25 per-dtype MMQ bench + dispatch closure (new
      mistralrs-quant #[ignore] test mmq_dtype_bench, cc-tagged;
      n=k=4096, m sweep 4..4096, per-dtype correctness vs dequantized
      f32 reference): gfx1151 mmq TFLOPS at m=4096 - Q8_0 24, Q5K 21,
      Q4K 21, Q3K 21, Q6K 15, Q2K 0.7 (pathological at EVERY m);
      dequant GEMM 28-31 at m>=2048, ties mmq at 512-1024. Verdict:
      Q6K carve-out validated; Q2K added to it; Q3K/Q4K/Q5K/Q8_0
      keep MMQ everywhere (dense large-m dequant edge is +18-25%,
      not worth handoff churn; no dense Q8_0 model sees long prompts
      in this fleet). PORTABILITY (user directive): all carve-outs
      are arch-gated via dequant_handoff_rows(dtype, cc) - active
      only on cc major 11 (RDNA3/3.5, the measured family with the
      shared AMD_WMMA path); unmeasured archs keep the MMQ-everywhere
      llama.cpp default; unit test q2k_q6k_hand_off_is_arch_gated
      pins 1150/942/1200/1000 behavior; extend the table only after
      benching new hardware. CANDLE BUG FOUND AND GATED: candle's own
      fast_mmq (stale kernel copy in candle-kernels/src/mmq_gguf,
      13-arg pre-type_dst ABI, diverged from mistral.rs's maintained
      copy) returns GARBAGE for every accepted dtype on gfx1151 -
      verified max_diff up to f32 max (Q2K/Q3K/Q4K/Q5K/Q6K/Q8_0)
      while mistral's copy is exact on identical inputs; the
      "impossible" 60-140 TFLOPS cells were wrong-result fiction.
      Dead code in prod (mistral's dispatch intercepts everything
      the fleet runs; the old Q2K<=128/Q6K<=256 gates diverted the
      rest) but a silent-corruption landmine. should_use_mmq now
      returns false on ALL archs until the kernel copy is resynced
      (fail-safe: dequant GEMM serves candle-side callers, correct
      everywhere); candle's fast_mmvq verified CORRECT at batch<=8.
      Recorded follow-up (~M): resync candle's mmq kernel copy from
      mistral.rs's, re-enable arch by arch via the bench. Known
      pre-existing (clean-tree repro): rocm-feature fp8 roundtrip
      ("no cuda implementation for dtype-to-fp8") and 2-3 hqq GPU
      tests fail under --features rocm; upstream capability gaps,
      unrelated to the dispatch work.
- [x] 2026-09-26 GpuArch: typed device identity for GPU policy
      decisions. DONE (see completion notes at the end of this
      entry). DESIGN agreed 2026-09-26; supersedes the raw-cc gates from
      de90887d9 (same behavior on gfx1151, clearer + multi-backend
      ready everywhere else).
      candle-core gains `pub enum GpuArch { Cuda { sm: (u8,u8) },
      Rocm { gfx: (u8,u8,u8) }, Vulkan { .. } }`, resolved ONCE per
      device. Rationale: the fork currently runs TWO conflicting
      numeric conventions, both from HIP's SYNTHETIC compute caps -
      the mmq layer's cc=maj*100+min (gfx1151 -> 1150) and the GDN
      layer's compute_major==9 (reads like Hopper sm_90 but means the
      CDNA gfx9xx family); synthetic caps also collide gfx1150 vs
      gfx1151. AMD identity comes from HIP's authoritative
      gcnArchName instead. KEY UNKNOWN RESOLVED: vendored
      candle/cudarc-hip already calls hipGetDeviceProperties with an
      opaque [u8;1472] CUdeviceProp (driver/safe/core.rs:213
      name()); hipDevicePropS layout is name[256] then
      gcnArchName[256], so the gfx string ("gfx1151") is readable at
      bytes[256..512] - add an accessor to the vendored cudarc-hip
      and parse to (11,5,1). NVIDIA: compute caps -> (9,0) etc.
      Migrations: dequant_handoff_rows(dtype, GpuArch) with the
      Rocm{gfx:(11,_,_)} arm carrying the measured RDNA3/3.5 Q2K+Q6K
      carve-outs; candle should_use_mmq takes arch (stays false until
      the kernel-copy resync); GDN v_major_state_supported +
      prefill_kernel_supported move from compute_major==9 to an exact
      translation Cuda{sm:(9,_)} OR Rocm{gfx:(9,_,_)} - the 9 meant
      NVIDIA Hopper sm_90 (CUDA-first history; the AMD gfx9xx match is
      the HIP-synthetic coincidence, untested, preserved for behavior);
      device_cc -> device_arch; unify the duplicated
      get_device_info impls (mistralrs-quant + candle-core both
      query attributes independently). The HIP-synthetic cc ints
      survive ONLY at the kernel bridge (the C launchers keep the
      llama.cpp encoding - one documented conversion point). Unit
      tests construct enum values directly (no encoding knowledge).
      Vulkan variant reserved now; identity later via
      VkPhysicalDeviceProperties (deviceName/vendorID string parsing -
      the ggml-vulkan precedent for vendor quirk workarounds).
      Verification: behavior-identical on gfx1151 - mmq_dtype_bench
      table unchanged, fast_mmq + GDN test slices green, serving smoke
      in-band. Conventions (candle README + AGENTS.md: "Rust policy
      matches GpuArch, never bare ints; the encoding lives only at
      the kernel bridge") land with the implementation commit so
      they ride the upstream PR.
      COMPLETION 2026-09-26: candle-core gpu.rs (GpuArch + parse +
      resolve + tests); cudarc-hip CudaContext::gcn_arch_name(); mistral
      device_arch/dequant_handoff_rows/batch_supported on GpuArch;
      GDN gates via gdn_vmajor_arch/gdn_modern_arch (exact Cuda sm 9 /
      family>=9 translations, sm90 test names kept - honest NVIDIA
      history); get_device_info (both copies) carry arch and became
      fallible. DISCOVERY: the offset-256 gcnArchName plan was WRONG -
      modern hipDevicePropS puts gcnArchName AFTER ~87 CUDA-compat
      fields + reserved[63] + hipReserved[32], and the unversioned
      hipGetDeviceProperties symbol only fills the LEGACY struct (the
      vendored 1472-byte blob is legacy-sized; the R0600 struct is
      bigger). hipDeviceAttributeGcnArchName was also removed
      (hipDeviceAttributeUnused5). FIX: cudarc-hip declares
      hipGetDevicePropertiesR0600 (exported by the runtime @@hip_6.0)
      and gcn_arch_name() scans an oversized zeroed buffer for the
      "gfx" string - no offset assumptions, ROCm-version-proof.
      Probe/#[ignore] test gcn_arch_name_resolves pins the contract;
      bench header now prints the true resolved arch (Rocm
      {gfx:(11,5,1)}). Feature matrix: compile_error on cuda+rocm
      both (clear pointer to this roadmap), CPU-only default builds
      verified green. Conventions landed in candle README + AGENTS.md.
      VERIFICATION: bench table identical on gfx1151 (mmq 0.7/21.2/
      20.4/21.5/14.7/23.5 vs fallback 28-30 at m=4096), fast_mmq
      slice 5/5, GDN gates 55/55 default, cuda_graph 31+8, paged-attn
      8/8, quant 295/295, fmt/clippy/check green both repos both
      feature sets. candle should_use_mmq now takes GpuArch (still
      false pending the kernel resync).
- [x] 2026-09-26 CUDA+HIP coexistence spike - the decision gate for
      multi-backend. VERDICT: GO - coexistence is clean; multi-backend
      is plumbing. Executed as a throwaway crate (cudarc =0.19.8 with
      candle's exact feature set + the fork's cudarc-hip, one binary,
      interleaved vendor calls) after installing CUDA 13.3 toolkit
      (nvcc for build.rs version detection only - cudarc 0.19
      compiles kernels via runtime NVRTC; stubs for link; no NVIDIA
      GPU or driver needed). Full evidence matrix, both of cudarc's
      vendor-binding modes:
      A) dynamic-linking (candle's current mode): links
         libcuda.so.1 + libamdhip64.so.7 side by side. With the
         toolkit stub as libcuda.so.1: PASS - cu* dispatch to the
         NVIDIA stub (CUDA_ERROR_STUB_LIBRARY, NOT into HIP), AMD
         side does real work (name/gcnArchName/4096-u32 htod-dtoh
         roundtrip), interleaved retries clean. WITHOUT any libcuda:
         the loader refuses to start the binary (DT_NEEDED).
      B) dynamic-loading: NO libcuda in DT_NEEDED at all - NVIDIA
         becomes a runtime dlopen. Same PASS with the stub present.
         Without libcuda: binary runs, but cudarc PANICS on the
         first driver call (not graceful) - must gate on
         cudarc::driver::sys::is_culib_present() (exists upstream
         in 0.19.8) first.
      Interposition: this stack's libamdhip64 exports ZERO cu*
      symbols (TheRock ships no CUDA-driver shims), so no global
      namespace overlap; and mode B dlopens via libloading with
      RTLD_LOCAL by construction, immune even on stacks that DO
      ship a ROCm libcuda.so.1 compat shim.
      DESIGN CONSEQUENCE for Stage 2: NVIDIA side must use
      dynamic-loading + is_culib_present() gate (AMD-only boxes
      then need zero NVIDIA bits and never fail to start). Per-
      vendor processes NOT needed. CPU-ONLY CASE (checked after
      the matrix): the HIP side has the same hazard mode A had
      for NVIDIA - libamdhip64.so.7 is a hard DT_NEEDED, so on a
      box with no HIP runtime the binary cannot even START (it
      only ran under a stripped env here because the TheRock
      lib dir is in the system loader cache). And the ungated
      NVIDIA dlopen PANICS on a GPU-less box - the probe gate is
      mandatory, not hygiene. CONSEQUENCE: the HIP side gets the
      symmetric treatment - port cudarc's libloading pattern
      into cudarc-hip (dynamic-loading mode + is_hiplib_present;
      see the new task below) so one binary starts anywhere
      (CPU-only, AMD-only, NVIDIA-only, mixed), probes both
      runtimes, and activates what it finds; candle's CPU backend
      is already first-class so per-model "cpu" assignment needs
      no extra work.
      Runtime half (real NVIDIA GPU) remains unverified on this
      box - hardware follow-up, but the link/load/dispatch risks
      are all answered.
- [x] 2026-09-26 ~M cudarc-hip dynamic-loading mode (Stage 2 enabler,
      from the coexistence spike). DONE. Ported upstream cudarc's
      libloading pattern across all 5 sys modules (driver 62 bindings,
      cublaslt 13, cublas 6, curand 9, nvrtc 5 hoisted out of a
      function-local extern block): a `dynamic-loading` cargo feature
      resolves every binding through a OnceLock<libloading::Library>
      (libloading 0.9, upstream's pin) + is_culib_present() probe
      mirroring cudarc::driver::sys's name; build.rs emits NO link
      lines under the feature. Dynamic-linking stays the default - prod
      builds compile the identical extern blocks (byte-for-byte, only
      cfg-gated).
      KEY DISCOVERY (cost a segfault): a probe that opens the library
      and drops the handle DL CLOSES the HIP runtime, whose static
      initializers cannot survive an unload/reload cycle - the next
      real init segfaults. Upstream's is_culib_present has exactly this
      shape. FIX in the fork: probe and loader share one
      OnceLock<Result<&'static Library, candidates>> - whichever runs
      first keeps the library loaded for the process lifetime.
      VERIFICATION: dual-dlopen spike binary has ZERO vendor libs in
      DT_NEEDED (readelf), starts under a stripped env, probe->init
      ->probe->init sequences clean, full htod/dtoh roundtrip +
      gcnArchName resolution through the dlopen'd runtime; nvlink +
      hip-dlopen mixed mode links; prod path unchanged (candle rocm
      check, gcn_arch_name_resolves, fast_mmq 5/5, mmq bench - all
      green, no new clippy lints in either mode). NOTE for Stage 2 on
      the NVIDIA side: upstream's probe dlcloses libcuda too - the
      stub survives it (trivial lib), but a REAL driver may share
      HIP's unload fragility, so prefer one guarded init attempt over
      probe->init cycles, and validate on hardware.
      Stage-2 wiring note (from the dynamic-loading port): the HIP
      probe/loader never dlclose()s (shared OnceLock - see the cudarc-hip
      task above); on the NVIDIA side, upstream cudarc's probe DOES
      dlclose, so wrap NVIDIA backend selection in ONE guarded init
      rather than probe->init cycles, pending hardware validation.
- [ ] ~L multi-backend runtime: CPU + NVIDIA + AMD + Intel in ONE
      process, per-model device assignment. Design agreed 2026-09-26;
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
      VERIFICATION: full regression bar green - quant 316/317
      (1 = documented pre-existing fp8 roundtrip, clean-tree
      verified), fast_mmq 5/5, gdn 55/55, cuda_graph 31+8,
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
      PHASE 2 MILESTONE PROVEN 2026-09-28 (uncommitted, see INCIDENT
      below for why the tree is not yet committed): GGUF SERVES ON
      HIP:0. Qwen3.5-4B Q8_0 through the fully hot-pluggable dual
      binary: weights via QStorage::Hip + companion plugins, rms-norm
      + rope + softmax + GLU + dequant GEMM all on hip, correct answer
      ("1776", finish=stop, coherent thinking), 13.7 tok/s decode /
      67 tok/s prefill (4B Q8 sharing the APU). Caps line: gguf=on.
      LANDED (uncommitted): candle QStorage-on-hip (quantized/cuda.rs
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
      bitpack re-gated pending hip-side hqq op coverage). Dual
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
      PHASE 2 REMAINING: hqq hip-side op coverage (gated tests),
      multimodal projector fix, launcher port (fused MMQ/MMVQ perf),
      release gate + full bar, commits.
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
- [ ] _S mark downstream-facing FFI as such (was §1, one comment) so cleanup does
      not delete it: `cuMemPoolTrimTo`, `cuMemcpy{HtoD,DtoH}Async_v2`, `alloc_pinned`,
      `CudaEvent::elapsed_ms`, `CUDA_ERROR_NOT_SUPPORTED`
      (`cudarc-hip/src/driver/sys.rs:55,260,292,310`). Trivial insurance.
- [ ] _S stale deploy comments (was §7, trivial):
    - `deploy/strix-halo-gfx1151/mistralrs.toml:73-75` still labels 35B MoE
      "ACTIVE default" (it is lazy-loaded; 27B is the entry default)
    - `mistralrs-test.toml:1-15` describes the pre-router world
      ("loads ALL [[models]] entries on startup"); `:39` `memory_mb = 512`
      contradicts the comment block above it. Verify which is authoritative.
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
- [ ] _S document the ~15 functional but undocumented MRS_/MISTRALRS_ env knobs (was §7)
      in `docs/src/content/docs/reference/environment-variables.md`
      (MRS_GDN_KERNEL, MRS_NO_FUSED_QKV/FFN, MRS_NO_FAST_MMVQ, MRS_ATTENTION_DEBUG,
      MRS_DIFFUSION_DEBUG_DUMP, MISTRALRS_MOE_BACKEND, MISTRALRS_FORCE_AVX2,
      MISTRALRS_CPU_KV_F32, the standalone-GGUF MISTRALRS_*_GGUF vars, ...).
      A single batch "debug and tuning envs" section works for the debug leftovers
      (MRS_PROFILE, LAYER_PROFILE, MRS_TRACE_LAYERS, NAN_PROBE, MRS_DEBUG_PA/CK,
      MRS_MMQ_DEBUG, MRS_ZERO_WS, MRS_POISON_NAN, MRS_NAN_TRACE, MRS_ATTENTION_DEBUG):
      either document as expert/knob or delete, now that the NaN hunt is over.
      Plus ROCm docs: build instructions, `CANDLE_ROCM_PATH`/`CANDLE_ROCM_ARCH`
      (fix the hardcode-vs-knob issue in P3 first), ROCm example in `examples/`,
      known limitations vs CUDA (§Deferred + appendices). Can ride along with merge.

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

- [ ] _S candle book fix (was §1): `candle-book/src/rocm/README.md` is a sketch;
      cites nonexistent knob `MISTRALRS_ROCM_COMPAT_INCLUDE` (build.rs hardcodes the
      include) and has a mangled sentence at lines 29-32. Fix, or retitle and shrink.
      (Do NOT document the knob until it exists.)
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
- [ ] _S `qvm_split_k` metal `todo!` inside a `/* */` block (was §7,
      `mistralrs-quant/src/metal_kernels/mod.rs:1069`): finish or move to a tracking issue.
- [ ] _S cosmetic: feature-table indentation in `candle-nn/Cargo.toml`,
      `candle-examples/Cargo.toml`, `candle-transformers/Cargo.toml`. Absolute last.

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
- Speculative dflash on ROCm — SKIPPED 2026-09-22. Generic speculative
  driver/verifier already support ROCm (`any(cuda, rocm)`); only dflash
  (`mistralrs-core/src/speculative/dflash.rs`, `paged_rows.rs`) is CUDA-gated. Its
  custom kernels are portable but its attention core is doubly FlashInfer-dependent
  (FA2 varlen-paged + FlashInfer KV layout) with no ROCm equivalent — 3-6 weeks,
  high risk. Generic draft-model speculative covers the use case; revisit only with
  a DFlash draft model in hand.

## Done (kept for context)

- [x] ~M CK sliding-window support (done 2026-09-22). CK's generic mask already
      honored `window_size_left/right`; only the args (`-1` hardcoded) and the
      dispatch gate (`sliding_window.is_none()`) blocked it. Threaded the window
      through C ABI, Rust FFI, `ck_flash_attn`; causal sliding dispatches,
      non-causal sliding stays eager. GPU reference test extended with square +
      gathered sliding cases. Measured release gfx1151: gemma-4 TTFT@128
      1082ms -> 382ms (2.8x), TTFT@1024 921ms; glimmer TTFT@128 1746ms -> 539ms
      (3.2x). Decode unchanged (seq_len=1 skips CK by design). Gemma-4 full
      layers (head_dim 512) stay eager; only hdim 128/256 kernels exist.
- [x] ~M bigger-model bench (answered 2026-09-22). Serving default Qwen3.8-27B
      (hybrid GDN + full attn every 4th layer, 65 layers, Q6_K, ~19.2GB):
      decode ~100% GPU busy, no host slack, no BLAS mistakes, no router, no eager
      attention; aggregate ~186 GB/s = 73% of 256 GB/s peak; MLP fused_glu alone
      ~216 GB/s (84% of practical peak), near-optimal; projection GEMVs
      ~170-190 GB/s, heroic tuning caps at ~10% total. Verdict: kernel work on
      this model exhausted; real levers are bytes/token and MTP n=2 (already on,
      +36%).
- [x] ~S prod `MISTRALRS_GGUF_NO_MMAP` 1 -> 0 (done 2026-09-25). Heap staging
      doubled every load transient (Q8_0 needed ~76GB, bailed solo); mmap +
      `DROP_HOST_AFTER_LOAD` (path fadvise) + `MANAGED_WEIGHTS` reaches the same
      single-copy steady state (27B RSS 23.2GB) with ~1x transient. Router gate
      confirmed: 27B need 40.7GB vs 62GB before. Smoke-tested inference OK.
- [x] ~M MoE router GEMV via rocBLAS fixed (done 2026-09-22). rocprofv3
      decode-only slice (Qwen3-16B, 64 tok @ d128): `moe_gemv` 6.4ms/token at
      113 GB/s vs 256 peak (44%), Q2_K gate_up laggard (~101 GB/s), Q4_K down
      ~131 GB/s. Router ran through candle Linear -> rocBLAS 128x128x32 tile GEMM
      for a 1-row GEMV: 65.8us/call x 48 calls/token = 3.16ms/token (15% of decode
      GPU busy) to read 512KB (~8 GB/s). Added `moe_router_gemv` (block-reduce
      dot-product in sort.cu, FFI, `ops::moe_router_gemv` fast path for <=16 rows,
      mixed F32/BF16/F16 xs/weight, leading dims folded). Wired: qwen3_moe,
      qwen3_next (router + shared-expert gate), mixtral, gemma4 router. Measured
      TPOT: Qwen3-16B 20.78 -> 17.87ms (14%), Qwen3.6-35B 32.5 -> 24.0ms (26%),
      gemma-4 30.4 -> 28.0ms (8%), Llama-30B flat (BLAS already picked sane skinny
      GEMV). rocprof confirms tile-GEMM router calls gone (0.6% prefill residue).
      The planned CK grouped-GEMM port is dead for this fleet: decode already uses
      custom quantized GEMV (`moe_gemv.cu`); CK grouped GEMM targets the BF16 path
      GGUF models never take at decode.

---

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

## Appendix B. Build health (verified 2026-09-22)

| Configuration | Status |
|---------------|--------|
| CPU-only (`cargo check -p mistralrs --no-default-features`) | Pass |
| ROCm (`cargo check -p mistralrs --features rocm`) | Pass |
| ROCm binary (`cargo build -p mistralrs-cli --features rocm`) | Pass, bench runs on 8060S |

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
