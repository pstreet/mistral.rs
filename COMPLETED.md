# COMPLETED: candle + mistral.rs fork work

Condensed record. Full evidence lives in git history (`git log -p TODO.md`).

## Oct 16 (top-k submit twin live, lookahead engaged)
- Resident top-k sampling submit twin (commit `384894552`, build 19):
  the last NVIDIA-role sampling gap in the resident decode path.
  `ops_cuda_role` gained the CudaTopKSamplingWorkspace family (pinned
  param slots, async token ring reuse, `sample_ranked_topk` launch -
  the kernel already lived in the nvcc companion, sort.cu; only the
  Rust machinery was missing) with dual dispatchers in ops.rs for
  submit/`_into`/wait_on/release_after (both stream roles)/complete/
  cancel, take_topk_{nvidia,hip} role guards, solo-build aliases, and
  the sampler + CudaTokenBatchSubmission wrapper carrying the role
  enums (wait_on_cuda/release_after_cuda/is_nvidia). The dual host-
  fallback gate in try_submit_cuda_token_batch_inner is GONE: non-
  argmax resident plans now submit on the NVIDIA role.
- Lookahead engagement traffic FOUND: the one-token lookahead gate is
  `device().is_cuda()`, i.e. NVIDIA-role only in dual builds, and
  pre-twin every sampled plan on that role bailed to host. So 4B-cuda
  decode traffic is the engagement trigger; first nonzero counter
  reading {role="cuda",outcome="engaged"}=16 over 3 requests (previously
  zero on both roles).
- Live verification (build 19 swap, rollback saved as
  mistralrs.pre-384894552-build18): 220tok temp-0.7/topk-40 coherent
  through the twin on qwen3.5-4b-cuda, greedy top1 and hip-role lanes
  unchanged, invalid-token guard silent, journal clean. Four shapes
  compile warning-free in touched files; CUDA_TOPK_SAMPLING_PARAM_WIDTH
  hoisted pub(crate); pre-existing dead imports removed.
- BENEFIT MEASURED (2026-10-16): brief A/B via the build 18 rollback
  binary (same lane, same prompts, warm): sampled decode on
  qwen3.5-4b-cuda 88.7-89.2 tok/s host-fallback -> 111.2-112.0 tok/s
  twin (+25%), greedy control flat (~112) - sampled decode now
  matches in-graph argmax. Engagement structure mapped: greedy argmax
  samples inside the graph replay (no submission); temp runs the
  lease chain with per-step on-device top-k (counter only marks
  chain re-starts, ~1/27 steps); logprobs stays host-side by design;
  concurrent max_seqs=2 pair 199 tok/s combined; prod restored to
  build 19 and re-verified (rate 111.9, counter re-engaged,
  journal clean). A/B artifacts in scratch/topk-twin/deltas/.
- Still host-side on NVIDIA (by design, unreachable on the resident
  path): the sync logprob top-k packed family
  (cuda_topk_logits_f32_packed_batched). Optional follow-up: a
  chained-step counter for per-step engagement visibility.

## Week of Oct 14-15 (serving sized, retention closed, twins live)
- 4B CUDA OOM root-caused: Utilization pool (15.4GB) fit but starved graphs;
  ISQ-aware `estimate_load_bytes` (scale by target/source bits, clamp to
  on-disk) + `fit_ctx` honored on Utilization budgets + 15% runtime reserve
  at all three Utilization sites (`94ffc9d98`, `8ffad71a9`, `78ea77e73`).
  4B verified at 12042MB pool, ~6GB runtime headroom, graphs capture.
- Mapping-fit regression (pre-existing): Utilization arm converted to
  MbAmount before the model-context cap, reserving a 65GB phantom pool for
  the 35B (real pool 10496MB) - old code failed worse (9.6GB over), reserve
  only shrank it. Fixed with the context clamp; verified eager load.
- Mapping KV 2x overcharge (pre-existing): K/V elements priced at activation
  bytes (2B bf16 vs 1B Q8_0). Now exact packed-width pricing (`46bfc4f3f`).
- Mapping saturation floors: Utilization projection floored at FIT_CTX
  minimum (`4c8892b55`); post-load physical-fit fallback when resident
  weights saturate the fraction cap (`27453e873`, all budget tests
  bit-identical). 27B self-sizes to 585-1096MB, fit-ctx clamps loudly.
- Attach-aware reservations (`ee31bc015`): MTP + entry ISQ quantizes a draft
  lm_head copy at attach (~4.2GB peak measured on 27B, 4259-fail/4535-pass
  MiB); reservation = vocab x hidden x (2B + 2x isq bits) + 25%, added to the
  primary reservation in both pipeline paths. 27B `context_len` REMOVED -
  zero manual context knobs left; `max_seqs=2` stays (concurrency).
- Unload retention (`94ef16f99`): teardown freed into the caching pool, no
  engine-tail trim, retry loops wedged 300s. Engine thread trims after the
  last Arc drops, failed reloads trim, both wait loops trim per poll;
  ContextSize pools estimate from GGUF headers (evict instead of blind).
  Verified: 23.9GB stranding -> 567MB; 19.9GB eviction converges in 0.5s;
  iGPU full GTT return in 5s.
- Twin parameterization (`020f390e4`, `007a8c2da`): pool introspection per
  driver twin (nvidia_pool/amd_pool) + `device_has_driver_pool()` gates
  (is_cuda excludes Hip - the Oct-12 "sync-only" note was wrong, the leg
  was fully inert). First hip trim in a dual build observed live ([pool-trim]
  50->51). Attributes degrade to None on unsupported runtimes.
- On-device eviction (`2509a1fe3`): LRU pick went global (evicted the iGPU
  35B to make room on the 3090). Both paths share a tested device-scoped
  helper; busy-wait scoped to same-device activity. Verified: exactly one
  eviction (the on-device 27B), 35B untouched. TTL eviction stays global.
- Build-shape lesson: prod is dual cuda,rocm - rocm-only builds silently
  drop the NVIDIA role (129.7MB -> 96.3MB tell). AGENTS.md corrected; builds
  need .rocm-env.sh + nvcc in the same shell (two rebuilds lost to this).
- Test debt (`11f60f4fb`, `b616e5da8`): GPU-shape lib targets never compiled
  (3 stale modules fixed); 19 device-env failures gated, q4k decode test bug
  fixed ((2,1)->(1,1)), 2 sm_90-only FA3 tests gated, arena tests aligned to
  shipped formulas. Lib green in all four shapes.

## Serving foundations (Oct 6-13)
- Role-identity cutover Oct 8 (`071c650d`, `587be98d2`, `bdb0f0169`, candle
  `77048535`): one shared backend body, thin role shells, dual prod binary.
- Graphs/MTP parity: NVIDIA MTP port, per-model device_spec fix (4B 70 tok/s
  on 3090), 27B fit + first MTP attach Oct 8 (since re-verified with graphs
  live 51.7 tok/s), topk twins, ctx-cap dual bug (+10.3GB), 3-backend
  concurrency + eviction architecture, cuda-only GDN gap, hybrid graphs-on,
  FlashInfer A/B (+3%, default ON), Q8-KV+FI loud-error, device sampling,
  capture-abort arena fix, speculative_rejection port, gemv twin, fp8
  scalar/vector (blockwise deferred).
- 35B graphs + sporadic fault hunt (CLOSED Oct 12-14): +69% graphs, MTP+graphs
  48-51 tok/s; footprint checks + settle-sync + gather guard; arena
  right-sized to measured x1.25 (`801aef85c`).
- 27B/4B live verifications Oct 13-15: 35B sanity CORRECT throughout; 27B
  MTP attach + 391 stop; 4B round-trips through eviction cleanly.

## Benches, quant and policy decisions
- MTP n=2 kept fleet-wide (`fe8e6ea1d`); 35B keeps Q5_K_XL, 27B stays Q6_K
  (Q4K ISQ -19%); 35B split KV K=Q8_0/V=Q4_0 kept (2.6GB freed); prefix
  caching n=4 live (48x multi-turn); per-model overrides + fit-ctx shipped;
  CPU serving uncapped (user policy); draft-head Q4K knob (+2.6%).
- GpuArch typed identity; CUDA+HIP coexistence GO (`is_culib_present`);
  cudarc-hip dlopen (no dlclose); Q6K MMQ not pursued; hygiene + env
  namespace cleanups; long-context + GDN-prefill baselines recorded.
- Test/hygiene/ops: healthcheck generation probe (level-trigger interlock),
  SEGV cores to scratch spool (LimitCORE=infinity), cuda-alone revival
  validation, ExecStartPre TOML guard, degraded-hardware boot,
  ATTN_PROBE removal, eprintln removal, per-call env::var OnceLock cache,
  stale deploy comments, env-knob docs audit (environment-variables.md).

## Dropped
- freedesktop drm/amd upstream filing (wild-writer vmcore forensics):
  dropped 2026-10-15, not worth the upstream cycle; forensics retained in
  the journal (NaN/stability sections of TODO.md).

## Archive: TODO Done section (moved verbatim 2026-10-15)

## Done (kept for context)

- [x] Ops 2026-10-11 ~01:20 UTC: iGPU 100%-busy/max-clocks wedge revisited
      (see 2026-10-09 entry below). New evidence overturns the "firmware
      latch, reboot-only" verdict: the wedge is INTERMITTENT and
      PROCESS-ATTRIBUTABLE, not stuck firmware. Provenance: mistral.rs
      load/inference bursts wedge 4/5 times, Chatterbox TTS startup load
      2/2, plain-torch bursts and idle never; `systemctl --user stop`
      of the owning service clears it every time (0% verified), no
      reboot needed. While wedged, SERVING IS UNAFFECTED (correct
      embeddings @200, 67 tok/s chat) - cost is ~40W + heat soak only.
      Signature while wedged: sysfs gpu_busy 100%, sclk pinned 2.9GHz,
      radeontop shows all shader blocks 0% with `cr` 100% (NB radeontop
      prints "Unknown Radeon card" on Strix Halo, block attribution
      suspect), owner CPU ~idle, VRAM modest. Ruled out: CUDA graphs
      (pegs with MISTRALRS_CUDA_GRAPHS=0 too), new mistral.rs binary
      specifically (torch/Chatterbox bursts wedge identically; Oct 9
      entry predates those builds), external pollers (journal shows only
      manual requests; healthcheck timer inactive). Leftover hypothesis:
      KFD queue/CP race on bursty weight uploads on this APU (TheRock
      10.1/HIP 7.16), occasionally wedged per burst. Follow-ups, all
      open: (a) watchdog tripwire "100% busy + zero requests ->
      restart offender" (current watchdog only watches 500s, and wedged
      requests still 200); (b) gdb stacks while wedged to pin the spin
      site; (c) note the Oct-09 "reboot cleared it" may just have been
      the process teardown that came with it - stopping the server was
      never tried then.
      ROOT-CAUSED 2026-10-11 ~02:10 UTC (same night, debug session):
      trigger matrix fully bisected with /tmp/opencode/bisect{1,2,3}.py
      (fresh-proc torch steps + 4-5s-settled sysfs gpu_busy reads).
      Findings: (1) `torch.cuda.init()` clean, `torch.empty(cuda)`
      malloc clean, but ANY real GPU work - 4-byte H2D, `fill_` kernel
      launch, tiny matmul - pegs 100%/41W; results always CORRECT.
      (2) Solo torch process: CLEAN (0%). torch + idle-ComfyUI: clean.
      torch + fresh-mistralrs (models lazy): clean. torch-new-work +
      mistralrs-MODEL-RESIDENT: WEDGES deterministically (embedding AND
      chat models, graphs on AND off). torch+torch+torch (chatterbox
      resident + comfyui + bisect): clean - mistralrs/candle-HIP is the
      required party. (3) SIGSTOP of any/all GPU procs does NOT clear
      (fully GPU-autonomous, no host thread driving it); SIGTERM/exit
      of EITHER party clears within ~30s (incl. mistralrs restart while
      chatterbox stays loaded). (4) Power 41W + sclk pinned max + zero
      shader-block activity + zero owner CPU = stuck amdgpu activity
      counter pinning DPM at max clocks, not real CU execution; serving
      unaffected throughout. Verdict: KFD/amdgpu multi-process activity-
      counter bug on gfx1151 (kernel 7.2.6-070206-generic, vbios
      113-STRXLGEN-001, TheRock 10.1/HIP 7.16, custom rocm-install/):
      a resident candle-HIP context + foreign-process dispatch latches
      the busy counter until either queue set is destroyed. No app-level
      fix (any H2D/kernel triggers it; can't serialize serving bursts).
      Mitigation: watchdog tripwire "100% busy + zero in-flight ->
      restart mistralrs" (45s, verified to clear); upstream report to
      TheRock/amdgpu with this matrix. Production wrapper untouched
      (MISTRALRS_CUDA_GRAPHS=1 restored after A/B).

- [x] Ops 2026-10-09 ~08:20 UTC: iGPU stuck at 100% busy / max clocks /
      ~40W with zero attributable load (all GPU processes proven idle
      via gdb + py-spy + frozen counters + empty nvtop list; likely
      post-burst DPM/firmware latch). DPM toggle obeyed but auto
      re-maximized. Reboot CLEARED it (post-boot: busy 0, sclk 617MHz,
      16.7W). BUT the same reboot dropped the RTX 3090 off the PCI
      bus (lspci clean, NVRM No NVIDIA GPU found, rescan empty -
      warm-reset enumeration failure; cold boot pending, user-owned).
      Prod crash-loops meanwhile (dual binary, see gap below) and
      self-recovers when the card returns.
      RECOVERY 2026-10-09: user updated Nvidia drivers + rebooted;
      3090 back on the bus (95:00.0, driver 615.71.09), prod self-
      recovered after 3 early-boot retries, iGPU latch STILL clear
      (busy 0, sclk 600MHz), prefix re-verified live on :1235
      (identical 3808-tok 9.84s -> 0.30s, hit/recurrent). PCI bus
      renumbered across the reboot (AMD card0, NVIDIA 95:00.0).
      NOTE: open-webui.service was disabled (preset enabled) - did
      not survive reboots; ENABLED 2026-10-09 (user-approved), both
      units now enabled.
- [x] ~S Dual binary degrades when a role's hardware is absent
      (found 2026-10-09, 3090 missing post-reboot; SHIPPED same day):
      lazy entries whose device resolve fails stay unregistered with
      a loud warn (`Skipping lazy model ... device unavailable ...
      restart with the hardware present`) instead of aborting boot;
      requests to them fail ModelNotFound, eager entries still abort
      loudly. Both builder loops (seeded + all-lazy). VALIDATED on
      the probe lane: hidden-CUDA + lazy cuda:0 entry boots, alias
      absent, hip serves, skipped alias 400s cleanly; symmetric
      hidden-HIP + lazy hip:0 entry boots, CPU embedder serves
      1024-dim, skipped alias 400s cleanly. 4-shape matrix green.

- [x] ~S ExecStartPre TOML guard + `from-config --dry-run` (2026-10-09,
      from the 3000-line journal review: Oct 8 ~70 crash-loop restarts
      on a duplicate `mtp` key, Oct 9 ~65 on lowercase `auto`
      cache_type - Restart=on-failure turns any bad edit into a 5s
      loop). `--dry-run` parses with the real serde schema and prints
      a serve summary (host:port, per-model alias/kind/lazy), exits 0
      without binding; both historical repros fail with their exact
      original errors. Unit file (repo-side deploy copy, daemon-
      reloaded, no restart) runs it pre-start; pre-start failure does
      not trigger Restart=. Validated against the live 12-model TOML.

- [x] ~L Role-identity Phase 1, acceptance battery (2026-10-06/07,
      continued): the cuda-only release binary links clean and is
      fully portable - ldd shows zero GPU DT_NEEDED (no libcudart,
      no libcuda); drivers and every kernel set resolve via dlopen.
      The last upstream linkage holdouts, retired or converted:
      (a) three build scripts still emitted static kernel archives +
          dylib=cudart (mistralrs-quant's 200-line inline static
          section with the marlin/fp8/mxfp4 probes, mistralrs-core's
          static block, candle-kernels' upstream archive) - retired;
          the probes died with them, which is role-true: the dual
          NVIDIA role runs without them. Capability-symmetric future
          (cutlass-MoE, mxfp4-marlin, fp8-sm90 on the NVIDIA role in
          both shapes) = benched follow-ups.
      (b) kernel_decl in mistralrs-quant and mistralrs-core flipped:
          the not-rocm arm now dlopens the cuda companion (the core
          one keeps a plain-decl arm for cpu where its ffi module
          still expands).
      (c) companions completed to full coverage: quant mirrors the
          old static TU set (30 TUs: ops/rotary/gemv/hqq/afq/gptq/
          moe_align/gelu/mmq/mmvq/indexed_moe/moe_grouped/fp8s/lora
          pair); core compiles the full src/cuda set minus the
          AMD-side moe_gemm trio; paged-attn gets its own CUDA list
          (the 9 dense + flashinfer pair + mla pair + kvscales +
          sinks) - the AMD companion legitimately excludes the
          flashinfer TUs (RDNA lacks the paths), so sharing its list
          starved the NVIDIA role.
      (d) the lora ffi files converted from raw extern "C" to the
          dlopen macro (exact signatures preserved), and the dynamic
          + routed MoE lora TUs joined the quant companion - the
          NVIDIA role in cuda-only now HAS working lora kernels,
          ahead of the dual NVIDIA role (widening there = follow-up).
      (e) the hybrid-models graphs gate widened to any cuda shape
          (was dual-only interim) - cuda-only hybrid models serve
          eager like the dual role.
      Serving observations: the 4B (multimodal pipeline, non-hybrid
      cache) aborts on its first decode-graph capture in cuda-only
      (pool-restore warn + poisoned-capture clone at cudarc 957) -
      reproducible with FlashInfer on and off; the dual 4B bench ran
      with graphs env-off so this path was never dual-validated for
      the 4B either. Debug queued. The 27B (hybrid, gated to eager)
      is the parity bench target: decode vs the dual 28.85 tok/s /
      prefill ~4.6k within noise = the Phase 1 proof.
- [x] ~L Role-identity Phase 1 (2026-10-06/07): cuda-only compiles
      green for the first time; all four shapes check clean (cpu,
      cuda, dual, rocm), post-fmt verified. Root-cause chain:
      (a) the candle workspace's [patch.crates-io] cudarc entry was
          silently inert - the resolver prefers the highest version
          satisfying ^0.19.8, so registry 0.19.9 always displaced the
          vendored 0.19.8 fork and the lock never re-resolved; every
          candle-workspace check was compiling against upstream
          cudarc (the true origin of the old "candle expects APIs
          upstream lacks" note). Fixed with version = "=0.19.8".
      (b) the NVIDIA shell's capability macros were shape-keyed
          (not(dual) expands - written when single-vendor meant the
          hip alias) instead of role-keyed; re-keyed the
          cuda_backend and quantized/cuda shells: kernels = the PTX
          twin, Lt/managed tiers dropped for the NVIDIA role, the
          rocm alias preserved bit-for-bit until the phase 3 cutover.
      (c) mistralrs re-keys: graph machinery (arena capture ON for
          any cuda shape, no-mode begin_capture, bookkeeping gates
          and warmup/clamp helpers un-gated), device_spec caps
          (cuda-only mirrors the dual NVIDIA row, mtp=false), the
          gemv/gguf/cublaslt/fp8 tiers in mistralrs-quant (alias
          launchers rocm-solo, NVIDIA twins feature=cuda, the
          never-compiled hip_fwd mangles and hip debug blocks
          deleted), the MoE dispatch in backends.rs, and paged-attn:
          the nvcc companion now builds for any cuda shape (the
          upstream static-archive + dylib-cudart path retires,
          kernel_decl dlopens cuda_plugin, has_fp8 emitted to match
          the AMD side, the cuda_role twins widened, FlashInfer
          kernels reachable through the companion).
      Safety: dual and rocm stayed green throughout - every re-key
      was a dead arm or additive for the working shapes.
      Remaining for Phase 1 acceptance: release cuda-only build,
      serve sanity + graphs on the 3090, 27B parity bench vs the
      dual numbers (decode 28.85 / prefill ~4.6k), FlashInfer-on
      observation in cuda-only. The dual FlashInfer claiming re-key
      + A/B bench stays queued with Phase 2 (the MTP port).
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
- [x] ~M NVIDIA-role decode graphs (fixed 2026-10-05). Model-scale capture
      replayed half the rope math: `tensor_apply_rotary_qk`
      (`mistralrs-quant/src/rotary/mod.rs`) ran q/k branches via `rayon::join`,
      and graph capture is thread-bound, so the worker-thread branch never
      landed in the capture; replay silently returned first-element
      replication (small-scale) / wild device read (model-scale, N>=4 layers).
      Proved by bisection (single/double/sequential chains green, rayon qk
      red). Fix: sequential launches on the GPU tensor path (CPU keeps its
      join). Full 16B graphs green at full scale in-request, 0 faults,
      coherent text. Companion changes: CUDA capture arena via the
      cudarc-nvidia fork (`graph_role.rs`, `cuda_graph.rs`), hybrid-restore
      `catch_unwind` hardening (`multimodal.rs`), lookahead bail -> warn +
      host-sampling fallback for the non-serving role (`execution.rs`),
      `SlicePtrOrNull::Null` static dims (`candle`). Regression:
      `rope_qk_capture_launch_matches_eager`.
      REMAINING (parked): first decode of the SECOND request on the same
      server fails at its first graph replay with bare
      `CUDA_ERROR_INVALID_VALUE` while the first request's ~9 replays and
      all eager work are clean; same-prompt repro, not sticky (5/6 requests
      pass with twin-workspace reset active; sticky without it), no memory
      faults under compute-sanitizer (it caught a synchronous API error:
      "Parameter ordinal must be between 0 and 1"), twin workspaces
      exonerated (fresh-alloc probe still fails once), stuck
      capture-flag/arena exonerated (per-step state probe false/false at
      failure), upload-skip exonerated. Surface: replay-lifecycle of the
      exec across requests (launch N vs 1..N-1). Parked with
      `BackendCaps.graphs=false` + `is_cuda_graph_device` hip-only; serving
      without graphs is excellent (16B decode 44.9 T/s cuda vs 39.2 hip).
      Next slice: instrument replay prelaunch/launch per stage, try
      per-request exec re-instantiation, or driver-level exec-node dump.
      cuda-alone still broken (pre-existing fork drift), revival queued.
- [x] ~M NVIDIA-role decode graphs UNPARKED (fixed 2026-10-05, later).
      Root cause of the cross-request failure: compute-sanitizer host
      backtraces pinned it to `cuStreamWaitEvent` - "Event is not valid.
      This can be related to a graph capture that is no longer running" -
      called from the MoE twin's workspace write guard. cudarc's
      `SyncOnDrop` event guards recorded slice events UNCONDITIONALLY on
      drop, including inside the capture window (tracking disabled), which
      bound the twin workspaces' eager-created events to the capture; the
      next request's first eager touch of those slices waited on a dead
      event. (The "Parameter ordinal" sanitizer line was startup device
      enumeration noise.) Fix: gate the record in
      `SyncOnDrop::record_event` on `is_event_tracking()` (cudarc-nvidia
      fork) - one choke point for all six guard sites. Verified: 5/5
      same-prompt requests, 0 faults, capture + replays all green; graphs
      re-enabled for the NVIDIA role (`BackendCaps.graphs=true`,
      `is_cuda_graph_device` hip-or-cuda).
      Open perf note (harmless, journaled): the captured graph carries 48
      MEMALLOC + 48 MEMSET + 48 MEMFREE nodes (3175 total: 2935 kernel,
      96 memcpy) - the twin's per-layer `final_out` `alloc_zeros` bypasses
      the capture arena (why candle temps carve but this alloc does not is
      unresolved); with AUTO_FREE_ON_LAUNCH these re-allocate per launch
      and serve correctly. Also: cudarc-nvidia vendored into
      `candle/cudarc-nvidia` (two commits: pristine 0.19.8 baseline, then
      the fork) matching the cudarc-hip precedent; patch paths updated in
      both workspaces.
- [x] ~S 4B (Qwen3.5-4B-MTP Q8_0) bench across cpu / rocm / cuda
      (2026-10-05, release dual build, single binary / three device arms,
      same methodology as the journaled arms: essay-600tok temp0.3 decode
      x3, 228-tok prefill x3, batch 1):
      - CUDA (3090): decode 68.9 tok/s (68.76/68.91/69.50), warm prefill
        ~10.5k tok/s (10441/10621; first-run 2819 = warmup). Settings: PA
        Q8_0 KV 2048MB, graphs OFF, MTP off.
      - ROCm (gfx1151 APU, prod paused via `systemctl --user`, graphs ON):
        decode 32.5 tok/s (32.49/32.50; trial 1 errored while prod's 27B
        was still draining - stop takes >5s), prefill ~1610 tok/s.
      - CPU (32 threads, rayon defaults): decode 18.0 tok/s (18.00-18.17),
        prefill ~228 tok/s. No PA/graphs/MTP.
      Quality: cuda arm answers "1776" (256-tok budget, finish=stop;
      64-tok sanity budgets are consumed by reasoning before content).
      Findings:
      (a) release + hybrid + cuda-role graphs ON aborts the process: fork
          unwrap (cudarc-nvidia core.rs:957) on
          CUDA_ERROR_STREAM_CAPTURE_INVALIDATED - the hybrid
          restore-in-capture blocker now has abort-class severity in
          release (the multimodal catch_unwind cannot cover fork-side
          unwraps). Working best for hybrid models on the NVIDIA role:
          MISTRALRS_CUDA_GRAPHS=0 until the restore is hoisted
          post-end_capture (top graphs follow-up).
      (b) the 4B refuses MTP ("This model does not support speculative
          decoding") despite the -MTP GGUF name and dual-hip caps
          mtp=true; prod's explicit `mtp = false` for it is the correct
          setting (the 27B is the MTP-bearing model).
      (c) ops: mistralrs-server.service is a systemd USER unit
          (~/.config/systemd/user/); pause with `systemctl --user stop
          mistralrs-server-healthcheck.service;
          systemctl --user stop mistralrs-server-healthcheck.timer
          mistralrs-server.service` (sudo queries the wrong manager), poll
          until :1235 is truly down (graceful drain of the 27B takes
          seconds), and restore both after.
- [x] ~M 27B (Qwen3.8-27B Q6_K + ISQ Q4K) bench, cuda vs rocm, best
      working settings (2026-10-06, release dual build, same harness:
      essay-600tok temp0.3 x3, 270-tok prefill x3):
      - CUDA (3090, graphs on -> loud hybrid skip -> eager, MTP off):
        decode 28.85 tok/s (28.48/28.85/28.98), prefill ~4.6k tok/s warm
        (270-tok prompt; first-run 938). Resident 21.4GB / 24GB.
      - ROCm (APU, graphs ON live - hybrid capture + replay clean,
        MTP off): decode 11.40 tok/s (11.38/11.40/11.40), prefill
        ~337 tok/s. Prod-paused window; sanity 1776 both arms.
      Parity gaps found and recorded (both block MTP-on today):
      (a) CUDA + MTP + 27B does not fit 24GB: PA reports "Num GPU
          blocks is 0" at BOTH 2048MB and 1024MB KV (drafter + Q4K
          weights); mtp-off fits at 21.4GB. NVIDIA-role MTP is also
          caps-gated (speculative kernels are hip-side).
      (b) NVIDIA-role graphs on hybrid models ABORT in release: the
          multimodal-pipeline capture (not normal-pipeline - the 27B/4B
          route through MultimodalPipeline) hits the poisoned-capture
          slice-clone unwrap (cudarc core.rs:957, panic=abort in
          release) - fixed with loud gates at both multimodal dispatch
          sites (skip + warn, eager decode) until the recurrent
          restore is hoisted post-capture.
      (c) indexed_copy dual-role port (copy_rows + clamp_slot_table):
          hip_fwd twins + role-aware gates; previously copy_rows bailed
          "requires CUDA tensors" on the dual hip role (the 27B+MTP
          rollback failure) and clamp silently no-op'd. Re-exported the
          hip slice ptr helpers from mistralrs-quant. This unblocks the
          hybrid MTP rollback on dual hip.
      (d) REMAINING for MTP-on-dual-hip: speculative_rejection.rs port
          (pinned staging, stream events, 3 kernel-launch ops, dense
          row packing - all CudaStorage-typed; the MTP verify path).
          That plus (a) is the next parity slice.
      Notes: hip 27B hybrid graphs now capture+replay cleanly in the
      dual build (previously the rollback poisoned them); hip decode
      11.40 mtp-off/dual-graphs-on vs prod single-vendor 11.10
      mtp-on/n=2 (methodologies differ: 600tok here, 1500tok prod
      sweep - not directly comparable).

---
