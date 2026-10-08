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

### Role-identity architecture (agreed 2026-10-06)

Target state: ONE shared backend body (`cuda_backend/body/`),
textually included by two thin role shells. Feature = role
linkage (`cuda` -> NVIDIA shell over vendored cudarc-nvidia,
`rocm` -> AMD shell over cudarc-hip; `vulkan` someday = a third
shell over the same body, the vendor-agnostic slot). Shell =
plain capability constants, no shape-keyed macros. Device =
runtime behavior (serving role, FlashInfer claimed on
Device::Cuda only, caps). Loud absence bails for unlinked roles.
Vendor features are never shape-hostage: the AMD role keeps the
full fork API in every shape; the NVIDIA role keeps arena graphs
+ FlashInfer + the nvcc companion in every shape (FlashInfer
wherever cuda is linked - standing directive). Capability
symmetry is the obligation; internal fast-path tiers (hipBLASLt
heuristics, managed uploads) are per-role optimization freedom,
ported only when a bench justifies. Module names stay
(`cuda_backend`/`hip_backend`) for merge health; shell doc
headers get rewritten as role definitions.

Three deviations from target (all identified, all understood):
(a) solo-rocm instantiates the AMD role through the NVIDIA-named
    shell (crate-root `cudarc` alias -> hip fork; AMD hardware
    appears as Device::Cuda) - the "ROCm leverages CUDA when
    built solo" issue;
(b) the NVIDIA shell's capability macros are shape-keyed
    (`not(dual)` expands) instead of role-keyed - the exact bug
    that broke cuda-only (the "present" arm was written when
    single-vendor meant hip-alias; it promises APIs the bound
    crate lacks: stream arena alloc, managed htod clones, fork
    cublasLt - the five errors of the 2026-10-06 check);
(c) ~17 downstream arms keyed on build shape (caps rows,
    FlashInfer claiming, serving-role resolution, graph arms).

Phases (sequencing endorsed 2026-10-06; the prod constraint is
that prod is MTP-locked and MTP only works through the alias
facade today, so the MTP port gates the cutover):
1. NVIDIA shell role-keying + cuda-only revival + FlashInfer
   re-key (device-keyed claiming) + caps.mtp enforcement + shell
   docs. No prod risk. Acceptance: 4-shape checks green (cpu,
   rocm, cuda, dual), release cuda-only serves 4B/16B on the
   3090, FlashInfer on/off A/B in dual (it activates a previously
   inert default-on path there), parity bench cuda-only vs dual
   NVIDIA role within noise; cuda-only joins the standing shape
   matrix (it rotted because nothing compiled it).
2. MTP role-awareness (speculative_rejection port; the
   indexed_copy rollback port landed 2026-10-06). Completes
   dual-AMD MTP and gates phase 3.
3. Solo-rocm cutover: hip_backend serves every rocm shape
   (Device::Hip in solo builds), the alias and its shape-keyed
   arms retire, "cuda:N" in rocm-only builds loudly redirects
   (prod auto-selects, config unaffected), full prod
   re-validation with the locked config + parity bench vs the
   current binary before cutover.
   PLAN 2026-10-07 (surveyed both repos, subagent inventories in
   session): (a) candle gates - lib.rs mod/use gates, role.rs arms,
   cuda_backend alias branches retire; (b) compiler-guided rekey -
   every Hip arm `all(cuda,rocm)` -> `rocm`, Cuda-is-AMD assumptions
   -> Hip, incl. Phase-2 port gates which must widen the same way
   (else solo-rocm MTP verify loses hip_fwd); (c) mistral.rs -
   device_spec resolve/caps/redirect, tuning/device_map/
   diagnostics enumeration, pipeline fan-outs, CK gate, gdn
   role_storage, graph arms, tests; (d) verify - 4-shape checks,
   solo-rocm smoke + MTP serve on hip hw, parity bench vs
   pre-cutover binary, prod re-validation protocol with user-owned
   prod restart. Prod :1235 untouched until (d).
   STATUS 2026-10-07: (a) DONE (candle 071c650d, 4-shape green).
   (b)+(c) DONE compile-side: mistral.rs-quant/-paged-attn/-core
   all check green in solo-rocm (10 compiler-guided rounds; the
   inventory: quant 17 files incl. hqq restructures + gptq cuda
   gating + gemv stub; core 15+ files incl. gdn 7k-line re-key,
   ops.rs ~180 gates, speculative_rejection full re-gate - hip
   twins rocm, Cuda items cuda, dispatchers split 3-way, so
   solo-rocm MTP verify NOW ROUTES TO HIP; moe/cuda role re-key;
   graph_role/CK/moe.rs hip re-keys; memory_usage pool_backend
   alias; diagnostics/device_map fan-outs; 5 pipeline fan-outs
   role-typed). Known solo-rocm behavior deltas, all loud-fallback:
   fused FP8 GLU quantize -> unfused path (mirrors dual hip role);
   gemv -> plain GEMM (mirrors dual); hqq bitpack -> CPU pack at
   load (dequant stays device-side). Full 4-shape matrix
   verification in flight; then hardware proof on the APU.
   LANDED 2026-10-07 (compile side): candle 071c650d + mistral.rs
   587be98d2, both post-fmt 0/0/0/0 on the full matrix. NEXT per
   plan (d): solo-rocm fast/release build + smoke + MTP serve on the
   APU (hip hw), parity bench vs the pre-cutover binary, then the
   user-owned prod re-validation. Prod :1235 still untouched.
   HARDWARE PROOF 2026-10-07 (solo-rocm fast binary, APU gfx1151,
   prod paused and restored clean):
   - 4B (Q8_0, mtp off): serves on Device::Hip; hybrid cache
     reserved + 2048MB PA allocated + 1024 GPU blocks; coherent
     reasoning trace, 400 tok, 0 errors.
   - 27B (Q6_K+ISQ Q4K, MTP n=2 greedy, port 1242): "Speculative
     decoding enabled" - the line the alias era NEVER produced
     (old binary refused at model attach). Metrics: 250 drafts,
     750 proposed, 362 accepted (48.3%; pos0/1/2 = 191/111/61),
     sparse_gpu_verify_total 247, ZERO cpu fallbacks, 0 errors,
     coherent essay.
   - THE PHASE 2+3 GATE IS PROVEN: solo-rocm MTP verify runs on
     hip GPU kernels (Phase 2's speculative_rejection hip twins
     + Phase 3's re-gates). Essay 600tok 57.8s on the fast build
     (never benched; release parity pending).
   Four stale gates caught by the proof (probe-guided), fixed,
   post-fmt 0/0/0/0 again: (1) candle-nn RmsNorm et al. hip_fwd
     twins dual-gated -> rocm (rms-norm bail); (2) mistralrs-
     server-core configure_paged_attn Hip arm dual-gated -> rocm
     (PA silently dropped at builder entry - a crate the compile
     survey never covered); (3) cache_engine is_hip_role dual-
     gated -> rocm (quantized-KV refused on Hip); (4) quant
     device_has_fused_ops solo-rocm arm still is_cuda() ->
     is_hip() (the alias-era check). Lesson journaled: the
     compiler matrix can't catch runtime-only gates in crates
     whose SOLO arm compiles but behaves wrong.
   RELEASE PARITY SUITE 2026-10-08 (release builds, APU, prod
   paused/restored per arm, standing harness: essay-600tok temp0.3
   x3 + 270-tok prefill x3; OLD = pre-cutover alias binary,
   NEW = cutover binary):
   - Round 1 caught TWO more silent eager regressions (probe +
     dispatch-metrics guided): (a) is_cuda_graph_device's solo
     arm still checked is_cuda() -> graphs never armed on Hip;
     (b) qwen3_5 TEXT model's supports_cuda_decode_graphs was
     cfg(feature=cuda)-gated -> trait default false -> 4B decode
     model_unsupported-skipped every step (the multimodal wrapper
     was already any(cuda,rocm), which is why the 27B graphed).
     Both widened; the dual-hip graph-capture fixes (Oct 6) carry
     over cleanly to solo-rocm.
   - FINAL NUMBERS (new binary ahead on every metric):
     4B decode 41.76/41.76/41.67 vs OLD 38.2 (+9.3%); 27B
     decode 11.31/11.31/11.28 vs OLD 10.6 (+6.7%); 4B prefill
     ~1606 vs ~1515 (+6%); 27B prefill ~339 vs ~327 (+4%).
     27B MTP n=2 (new capability): 10.41/9.49/9.62 - roughly
     mtp-off parity on this saturated APU at ~48% acceptance
     (MTP's win shows at higher-acceptance workloads / prod's
     35B n=2 sweep).
   Phase 3 parity acceptance MET: the cutover binary is
   performance-positive vs the alias binary on both models.
   35B PARITY SUITE 2026-10-08 (release builds, prod's own model,
   Q5_K_XL MTP n=2 greedy, Q8_0 KV 2048MB, prod paused/restored
   per arm): first arm FAILED on the new binary - every request
   died with "moe experts forward". Root cause (probe-guided):
   the 35B is the FIRST MoE model exercised on solo-rocm (the
   4B/27B are dense hybrids that never touch the MoE experts
   backend), and the Phase-3 re-point had made solo-rocm's
   gguf::cuda module the dual hip twin, whose un-suffixed MoE
   entries are stub bails ("stage 2 of the MoE port") - the real
   implementation (which the alias binary served prod from) was
   compiled in NO shape. Fixes:
   (a) DECODE: restored the real cuda.rs for solo-rocm (mod re-
       point + 19 Device/Storage matches re-keyed to Hip; file
       only compiles solo-rocm where every GPU tensor IS Hip).
   (b) GROUPED PREFILL: the un-suffixed fast_mmq grouped names
       are dual-era stubs; routed solo-rocm through the proven
       _hip twins instead - forward_grouped_hip widened from
       dual-only to every rocm build, its dispatch arm likewise,
       solo stub-block deleted, cuda_hip.rs compiled as a second
       module for solo (real _hip grouped entries) + both lib.rs
       export gates (own-name and grouped_moe_mmq_*_hip aliases)
       widened to feature=rocm.
   FINAL 35B NUMBERS (new binary ahead on both arms):
   decode 32.97/34.80/34.49 (~34.1) vs OLD 21.68/22.73/21.92
   (~22.1, +54% - the alias path apparently never armed
   MTP+graphs together on the MoE model; the cutover hip path
   does); prefill 627/635/634 vs ~593-604 (+5%); MTP 1046/1632
   accepted (64%), sparse_gpu_verify 792, ZERO cpu fallbacks,
   0 errors. All four shapes check clean post-fmt.
   Phase 3 FULL parity: 4B +9%, 27B +7%, 35B +54% decode; all
   three prefills ahead; MTP (impossible in the alias era) live
   on hip kernels for the 27B AND prod's 35B. Prod cutover now
   awaits only the user's re-validation window.
   CUTOVER EXECUTED 2026-10-08 ~07:00 UTC: fresh dual release
   (cuda,rocm) built from the tip (bdb0f0169 + candle 77048535)
   into the standard target mistral.rs/target/release/mistralrs
   (36m build, doctor: revision bdb0f0169, hardware match PASS);
   Oct-1 alias binary kept as target/release/mistralrs.pre-cutover-oct1
   for instant rollback. Service restarted on the dual binary:
   auto-selected Hip gfx1151 with graphs=on paged_attn=on gguf=on
   mtp=on; sanity "1776" correct; MTP drafting live (315 drafts,
   328/630 accepted); graphs replaying (317 replay vs 3 eager
   population); single first-boot decode probe 13.64 tok/s on
   model="default" (includes demand-load + graph capture - expect
   it to settle; full 3-trial methodology on a quiet box is the
   confirming measure). Healthcheck restored, prod serving.
   Dual binary is now prod: cpu/hip/cuda all addressable
   (device_spec "hip:N"/"cuda:N"/"cpu", auto = hip:0).
4. Vulkan: future third shell; the include!-body pattern is the
   extension point. No work yet.

### Graphs/MTP parity queue (opened 2026-10-06, from the 4B/27B benches)
- [x] ~M MTP on the NVIDIA role (CLOSED 2026-10-08, port of Phase 2).
      Kernels were already in the nvcc companion (both sparse_rejection
      entries exported from libmistralrscuda_cuda.so); the Rust side
      refused the role. Port: ffi_cuda declarations, dual cuda_fwd
      through ffi_cuda (use-alias, no duplication), staging dual-bails
      narrowed to solo-rocm, eligibility hip||cuda in dual, caps
      mtp=true on the dual-NVIDIA row (commit 00203776e, matrix 0/0/0/0).
      HARDWARE PROOF on prod (dual binary, RTX 3090 cuda:0 alias,
      27B ISQ-Q4K + K=Q8_0/V=Q4_0, essay-600tok x3 + 270-tok prefill
      x3, same methodology): "Speculative decoding enabled
      (built-in, n_predict=2)"; MTP-ON decode 12.74/12.60/13.23
      vs MTP-OFF 11.18/11.21/11.21 (+15%); acceptance 991/1718 (58%),
      sparse_gpu_verify 838, ZERO fallbacks; sanity "1776" both arms;
      prefill ~330 on vs ~337 off (draft overhead, expected). MTP now
      serves on both roles. Uplift smaller than hip's published
      +36-77%: follow-up work is graphs-capture proof + FA2 dispatch
      audit on the NVIDIA role (backlog P1).
      CORRECTION 2026-10-08: the numbers above are VOID as a cuda
      proof. nvidia-smi showed 1 MiB on the 3090 after the alias
      loaded+served, no Cuda caps line at any boot, prefill identical
      to hip: the alias served on the iGPU. Root cause:
      `build_all_lazy` (all prod entries are lazy) built one shared
      device from the global spec and never read per-model
      device_spec - the flat `device` key parsed and converted fine,
      then was silently dropped at registration (only the seeded path
      resolved it). Fix: per-entry resolve_and_report in the lazy
      loop (mirrors the seeded path, loud caps line per entry).
      RE-RUN QUEUED with nvidia-smi as device witness + a boot-log
      assertion on the Cuda caps line before any trial counts.
      RE-RUN RESULTS 2026-10-08 (5 rounds, all witnessed): the 3090
      serves (4B Q8_0 on cuda:0, 8.5GB VRAM, decode 70.25/70.34/70.31
      vs 41.76 hip, +68% - first real NVIDIA-role serving), but MTP
      verify remains UNPROVEN on hardware: the 4B is hybrid and MTP
      never attached (no "Speculative decoding enabled", empty
      counters - same recurrent-restore gap as graphs-eager, now
      observed on the NVIDIA role too), and the 27B cannot load on
      24GB under the global 262k-context config (15.5GB ISQ weights +
      6.5GB KV + fixed reservations overflow; the pool budget
      MbAmount/ContextSize hard-fails instead of fitting, and
      per-model max_model_len does not shrink the pool). Graphs-off
      and 32k-ctx variants fail identically, so the consumer is fixed
      reservations, not KV/MTP/arena alone. 27B-on-3090 needs
      spill-aware KV fitting or a per-model paged budget (backlog P1).
      Efficient proof path: cuda-shape GPU unit tests for
      sparse_rejection submit/complete on cuda:0 tensors (tiny, no
      24GB constraint) instead of full serving. All bench aliases
      removed from prod TOML; global context restored 262144.
      27B-ON-3090 SOLVED IN THREE CONFIG STEPS 2026-10-08 late
      (probe-guided via [cache-fit]/[res-split], probes removed
      before commit): (1) per-model context_len=32768 sizes the pool
      correctly (requested 6.5GB -> 816MB); (2) prefix_cache_n: the
      TOML said "no prefix caching" but the line was COMMENTED OUT,
      so the default 16 silently won and reserved 2511MB of hybrid
      recurrent prefix snapshots (17 x ~148MB) against the KV
      budget - on hip too, contradicting documented intent;
      uncommented (KEPT, helps both roles); (3) max_seqs: default
      32 -> capacity 33 x 3 MTP checkpoint lanes x ~148MB ~ 14.5GB
      recurrent state made the MTP attach OOM on 24GB; TEMP
      max_seqs=2 shrank it to ~1.3GB and the attach SUCCEEDED.
      MILESTONE 20:32 UTC: "Speculative decoding enabled: MTP
      assistant built-in with n_predict=2" - the first MTP attach
      on the NVIDIA role, ever. 27B ISQ-Q4K + Q8/Q4 KV + MTP n=2
      resident at 19.6GB/24GB on the 3090.
      FIRST FORWARD STEP FOUND THE REMAINING PORT GAP:
      "cuda_top1_logits_submit_batched requires CUDA logits" -
      ops.rs topk/top1 storage arms are (cuda,not-rocm)->Cuda /
      (rocm)->Hip, so DUAL builds have no Cuda arm and NVIDIA-role
      tensors bail. The verify-path dispatch was widened in Phase 2
      but the topk ops underneath (proposer top1 submit, verifier
      ranked-packed batched + workspace, logits packed batched)
      were not. Port shape per op: accept Storage::Cuda in dual,
      launch via ffi_cuda (all kernel symbols ALREADY in the nvcc
      companion - verified by nm: top1_large_f32_packed_batched,
      sample_ranked_topk, sparse_rejection_topk_f32), workspace
      types dual-handled. Rust-only, no build.rs changes.
      max_seqs restored (TEMP removed); alias kept for the
      continuation; probes still in the working tree (remove before
      commit). BACKLOG: per-model max_seqs + a loud log when
      recurrent-prefix reservations eat the KV budget.
      TOPK PORT HARDWARE-PROVEN 2026-10-08 ~21:34 UTC (probe lane,
      fast build + standalone :1244, graphs off, 3090):
      (a) GREEDY RUN (mtp=false): sanity "1776" correct + 600-tok
      essay 28.8 tok/s, ZERO errors. CORRECTION (2026-10-08,
      post-commit code check): this run did NOT exercise the twin -
      in dual builds device_sampling_supported gates
      try_sample_batch_cuda to hip devices (device_has_fused_ops
      returns is_hip()), so NVIDIA-role batch sampling, greedy
      included, degrades to host sampling. The twin's hardware
      proof is (b) alone.
      (b) FULL MTP LOOP PROOF (mtp=true, lazy, text-only load,
      max_seqs=1, context 16k): "Speculative decoding enabled
      (built-in, n_predict=2)" + sanity "1776" CORRECT through MTP +
      metrics drafts_total 20 / accepted_total 32 - draft, verify,
      and accept ALL on cuda kernels (top1 twin + ranked twin +
      sparse rejection). Resident 19.6GB. First MTP tokens ever
      generated on the NVIDIA role.
      STRUCTURAL NOTE FOR THE 3090 BENCH: NVIDIA-role non-MTP
      sampling runs host-side (full-vocab copy + sync per token)
      while MTP-on samples on-device via the verify twins, so the
      on/off delta carries that component - interpret accordingly.
      Non-MTP sampling parity needs the topk-sampling twin
      (sample_ranked_topk, categorical_large) - backlog, sharpened
      by this finding.
      ATTACH-HEADROOM FINDINGS (probe series b-f): eager attach OOMs
      (ISQ staging held at attach instant); lazy attach at
      max_seqs=2/32k also OOM'd on the probe (23.9GB held vs
      parity14's 19.6GB success in prod - the delta remains
      unexplained; mmproj removal did NOT switch pipelines, the
      GGUF metadata keeps it multimodal). max_seqs=1 + 16k cleared
      it. BACKLOG: settle-then-attach (retry the attach after the
      post-load settle), free ISQ staging pre-attach, and the
      unexplained 4GB eager-vs-prod delta. The port itself is
      exonerated: every failure was OOM at a specific allocation,
      never a kernel or correctness fault.
      PROBE-LANE VALIDATION: the fast+spare-port policy carried six
      probe rounds with ZERO prod restarts; 4m22s fast build vs 17m
      release; ccache 81% kernel hit rate across the rounds.
      PROD SWAP 2026-10-08 22:24 UTC: release rebuild (dual,
      7m08s with the caches) with the topk port live - the settle-
      retry era probes are gone from the artifact. Rollback copy:
      mistralrs.pre-topk-port (the 20:24 settle-retry build). 
      Verified: health in 10s, both caps lines (Hip gfx1151 serving
      role + Cuda sm8.6, graphs/paged_attn/gguf/mtp all on), 
      default-model sanity 200 (thinking-model empty-content quirk),
      3090 idle at 275MiB, no startup errors. Known state: the
      27B-cuda alias still attach-OOMs at default max_seqs=32
      (capacity backlog: settle-then-attach / per-model max_seqs);
      its forward-step bug IS fixed by this binary, proven on the
      probe lane tonight.
      MEMORY AUDIT 2026-10-08 ~23:00 UTC (user asked why the box
      was full): 72G used + 36.6G swap decomposed. 45G = amdgpu
      GTT (kernel-accounted, invisible to ps): the hip role
      serving the 27B - weights + 262k-context KV pool + MTP
      checkpoint pool at capacity 33 (~14.5G) + graphs + 
      workspaces. 24G = mistralrs anon RSS (host runtime +
      managed copies; zero swap). 36.6G swap = /tmp TMPFS holding
      ~35G of scratch CARGO_TARGET_DIRs from the session's shape
      checks (twincheck/fitctx/settle/lazydev/mtpcuda x 
      cuda/rocm/dual) - deleted, swap fell to 3.9G instantly.
      POLICY FIX (AGENTS.md): scratch target dirs go on disk
      under the repo, never /tmp (tmpfs = RAM). The GTT 45G is
      prod's real serving footprint, not a leak; per-model 
      max_seqs would shrink the checkpoint pool share.
      CTX-CAP BUG (found 2026-10-08 ~23:30 via user question,
      fixed same night): calculate_cache_config caps the KV pool
      at the model's native context on unified-memory devices,
      but the gate (is_integrated_gpu) has cfg arms only for
      solo shapes - in the dual build Device::Hip falls to
      _ => false, so the cap has been DEAD on the hip role since
      the dual cutover. Fleet GGUF context_length declarations:
      27B/35B/4B/coder-next/gemma-26b all 262144 (matches the
      global context_len, no current waste), but Llama3.2-30B
      and Muse-30B declare 131072 (2x over-alloc) and Qwen3-16B
      declares 40960 (6.4x, ~6GB unified RAM per load for a pool
      it cannot use past 40k). Fix is surgical: cap_to_model_
      context ||= cache_engine::is_hip_role(device) (the
      dual-correct role helper). The OTHER is_integrated_gpu
      callers are deliberately untouched: memory accounting
      (GTT-query branch), ISQ staging, device-map heuristics -
      all recent capacity tuning was measured under
      hip-as-discrete; flipping those is a deliberate change
      (AUDIT ITEM), not a bugfix.
      YaRN note: the cap target is GGUF context_length metadata
      (the model's declared native). No fleet model carries
      rope_scaling metadata; if one ever does, the cap clamps
      at declared native - the conservative direction.
      PROD SWAP 2 2026-10-08 23:28 UTC: ctx-cap fix live in prod
      (9877bd866, release 7m07s). Rollback chain: 
      mistralrs.pre-ctxcap (topk-port, 22:24) -> pre-topk-port
      (settle-retry, 20:24). Verified: health 10s, 27B sanity
      200 with its full 256k pool (6528MB; declares 262144 so
      correctly UNCAPPED), and the 16B PROOF: "capping KV cache
      from 12288 MB to 1920 MB (40960 tokens)" - 10.3GB saved
      per 16B load. 16B serves (HTTP 200); GTT 45.2 -> 47.6GB
      means its total footprint is ~2.4GB vs ~12.7GB uncapped.
      The 128k-native 30Bs (Llama3.2, Muse) will cap to ~half
      pool on their next demand-load. GTT steady state for the
      27B serving: 45.2GB.

- [x] ~M cuda-only 3x decode gap (CLOSED 2026-10-07, root-caused, not tuned).
      It was never host overhead: cuda-only ran the GDN recurrence on CPU
      (`gdn_device_supported` hard-`false` in solo shapes, a stale hip-side
      comment predating the nvcc twin port) while dual serves device GDN
      on both roles (dual arm returns true for either role). Per-layer
      6x DTOH + CPU compute + uploads = the ~3x. Flip (GDN-only,
      solo-cuda, loud plugin error if kernels missing):
      essay-600tok fast-binary 26.0s -> 8.0s (~75 tok/s, vs dual control
      69.7), "1776" sanity correct, 0 errors. Dual/hip untouched.
      Release-bench + parity formalities still queued (methodology).
      PARITY 2026-10-07 (release, graphs-off posture matching the dual
      control): decode 77.9/77.6/77.8 tok/s (dual control 69.7 - ahead,
      config KV-type differences likely), warm prefill ~11.8k tok/s
      (dual ~10.5k).Gap closed and reversed; the CPU recurrence was all
      of it.
      27B PARITY 2026-10-07 (release, FI-decode off to match the dual
      control which never claims FI; Q8 KV): decode 600 tok in 20.0s =
      30.0 tok/s vs dual 28.85 (+4%, within noise + config margin),
      0 errors, essay output fully coherent. Short-budget sanity empty
      (thinking-model quirk, same as 4B - reasoning trace present).
      27B prefill (270-tok prompt x3): 838 cold, 4791 / 4851 warm vs
      dual ~4.6k warm - parity.
      EXTERNAL CHECK 2026-10-07 (published Qwen3.8-27B / RTX 3090
      single-card, no-MTP decode), CORRECTED: ISQ Q4K dequantizes and
      requantizes every layer at load (`apply_isq`: only already-Q4K is
      preserved), so runtime traffic is Q4-class, not Q6_K - the file
      quant is irrelevant. Fair bar is llama.cpp Q4_K_M ~36-41 tok/s
      and vLLM no-spec 46; we are at 30.0, ~20-30% under. Roof math:
      ~16GB/token over 936GB/s = ~58 tok/s max; we run 52% of roof vs
      their ~68%. Suspects for the remainder, all consistent with the
      evidence: eager decode (no graphs - every published CUDA number
      is graphed; per-token launch overhead across ~65 layers),
      CPU-fallback sampling (full ~0.5MB bf16 logits DTOH + device
      sync per token; the fused top-k gate is hip-side), Q8_0 KV
      (~57KB/tok vs published q4_0 ~29KB/tok). MTP remains the big
      lever (+36-77% published) = Phase 2; graphs-on is the remaining
      efficiency follow-up (device sampling done 2026-10-07, see below).
      Phase 1 parity acceptance MET on both models.
- [ ] ~M Hybrid graphs-on (was "restore hoist", theory superseded above).
      Hardening landed: missing `end_cuda_capture_discard` on the
      forward-Err arm (the pool-guard drop + checkpoint rollback then ran
      on the still-open poisoned stream) + `try_clone` in
      `cached_htod_slice` (was `Clone::unwrap`). Abort -> loud eager
      verified. Debug aid kept: `MISTRALRS_DEBUG_HYBRID_GRAPHS=1`
      bypasses the eager gates (default off). Re-entry as noted above.
- [x] ~S FlashInfer on/off A/B on cuda-only (done 2026-10-07, 4B
      release, graphs-off): ON decode 77.7/77.4 tok/s, OFF 75.5/75.0
      (~+3% for FI); warm prefill identical ~11.3k both arms; 0 errors
      either way. FI decode path is live on the NVIDIA role (kernels
      present, output correct) - the Phase 1 "previously inert" path
      is now observed working. Default stays ON.
- [ ] ~S Q8-KV + FlashInfer-decode is unimplemented on the NVIDIA role
      (found 2026-10-07 via the 27B parity bench): cuda-only genuinely
      claims FI (`supports_layer` true) -> FlashInferHnd layout -> the
      block-quant gate bails "requires the Standard layout". Dual never
      reaches it (FI claim is cfg'd solo-only, so dual runs Standard).
      Parities used FI-decode off to match the control. Queued: either
      Q8+FI support or a loud config error (FI A/B done above).
- [x] ~S Device sampling on the solo NVIDIA role (done 2026-10-07,
      third stale-gate flip): the top-1/top-k submit path was hip-gated
      (`device_has_fused_ops`), so solo-cuda copied full logits to host
      + synced per token. New `device_sampling_supported` helper
      (fused-ops OR solo-cuda), used at the 3 sampling gates; MoE
      router, norms etc. untouched, dual untouched. Verified fast
      binary: no plugin errors (kernels present in nvcc companion),
      temp-0 sanity byte-identical `1776` (device top-1 agrees with
      CPU), essay-600tok 8.0s -> 6.85s (~14%, sync + 0.5MB DTOH per
      token removed). Suspect list in the external check updated
      accordingly; CPU sampling struck.

- [ ] ~M 27B/4B cuda-only capture abort (correctness, open):
      CORRECTION 2026-10-07: the 4B IS hybrid (NormalPipeline, text-only
      GGUF path; "Recurrent state pool reserved" in the serve log) - the
      earlier "non-hybrid" label was wrong. So this item and the hoist
      below are one bug, and the aborting pipeline was normal.rs, which
      (unlike multimodal.rs) had NO hybrid eager-skip gate.
      REPRODUCED 2026-10-07, twice, cuda-only release (rev 0f61a5210,
      graphs on, smoke4b-cuda.toml): first decode-graph capture attempt
      logs "Failed to restore CUDA graph memory pool threshold ...
      CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED" then aborts at
      cudarc-nvidia safe/core.rs:957 (`Clone::clone` unwrap on
      CUDA_ERROR_STREAM_CAPTURE_INVALIDATED; release panic=abort, core
      dumped). Chain: an illegal op inside the capture window (restore
      `from_vec` family) poisons the stream; the next slice clone
      unwraps instead of erroring.
      FIX (correctness, in-tree, pending verify): mirrored the multimodal
      hybrid eager-skip gate into NormalPipeline::try_cuda_decode_graph_
      forward + precapture impl (cfg(feature=cuda), warn-once, dispatch
      Skipped/ModelUnsupported). Hybrid models on the NVIDIA role now
      skip capture loudly in BOTH pipelines; true graphs-on needs the
      hoist below. Workaround MISTRALRS_CUDA_GRAPHS=0 no longer needed
      for correctness (kept as a perf posture knob).
      VERIFIED 2026-10-07: fast cuda-only binary serves the 4B smoke with
      graphs ON - the "hybrid models keep eager decode" warn fires from
      pipeline::normal, clean serve, 0 panic/abort lines. (Empty short-
      budget content is the known thinking-model quirk, not a
      regression.)
- [ ] ~M Hybrid restore hoist: move the recurrent checkpoint restore out
      of the capture window (post-`end_capture`) so hybrid models (27B/4B)
      can capture on the NVIDIA role; then drop the multimodal eager-skip
      gates. SUPERSEDED 2026-10-07 by backtrace evidence: the pipeline
      restore already runs post-capture and is safe. The real in-window
      DTOH offenders were (1) the CPU-fallback GDN recurrence
      (`gated_delta_rule_recurrence` host round-trips - FIXED by the
      device-GDN flip below, which also closed the 3x gap 26s -> 8s);
      (2) the recurrent pool scatter `to_vec1` (FIXED by the solo
      device-scatter flip). Remaining: with (1)+(2) fixed, lazy capture
      on the 4B reaches replay but the following sampling DTOH fails
      with a bare `CUDA_ERROR_INVALID_VALUE` (no capture error, no
      disable-warn, stream presumably left capturing - leak path
      unidentified; re-entry: info-level bypass run + replay-path
      audit). Eager gates stay until capture demonstrably works.
- [x] ~M `speculative_rejection.rs` dual-role port (done 2026-10-07,
      commit 3c24dda25): hip_fwd twins for the kernel launch + both
      pinned staging ops, parallel hip workspace/submission, hip
      staging/submit/complete/validate/dense paths, dispatch enums
      with solo aliases, verifier gates widened to the hip role.
      PROOF (dual fast binary, 27B hip:0, MTP n=2 greedy):
      "Speculative decoding enabled", 130 sparse GPU verifications
      on-device, 402 drafts proposed with pos0/1/2 accepts 101/52/27,
      ZERO cpu fallbacks, 0 errors; ~19-20 tok/s sustained vs 11.4
      mtp-off (+70%). The MTP verify path runs on hip kernels.
- [ ] ~M NVIDIA-role MTP: (a) memory - 27B Q4K + drafter = PA "0 GPU
      blocks" at BOTH 2048MB and 1024MB KV (mtp-off fits at 21.4GB/24GB);
      try smaller KV + `mtp_draft_lm_head_isq = "Q4K"` before anything
      else; (b) speculative kernels are hip-side (role caps mtp=false).
- [ ] ~S 16B graph perf note: the twin's per-layer `final_out`
      `alloc_zeros` bypasses the capture arena (48 MEMALLOC + MEMSET +
      MEMFREE nodes in the captured graph; harmless under
      AUTO_FREE_ON_LAUNCH but re-allocates per launch). Find the bypass
      or accept-and-document.
- [ ] ~S gemv companion twin (the gemv landmine fix is gated dual-wide
      false; the NVIDIA role takes the plain GEMM in dual builds -
      journal entry at the 2026-10-05 MMVQ work).
- [ ] ~S fp8 NVIDIA-role parity (currently a loud dual bail; port the
      fp8 kernels to the nvcc companion).
- [ ] ~M cuda-alone revival = role-identity Phase 1 (the NVIDIA
      shell's shape-keyed "present" arms promise APIs the bound
      crate lacks: stream arena alloc, managed htod clones, fork
      cublasLt - the five errors of the 2026-10-06 cuda-only
      check; the fix is role-keying the shell, not porting APIs).
      STATUS 2026-10-07: all four shapes check green post-fix;
      release build + serve sanity + 27B parity bench pending.
- [ ] ~S optional prod knob: `mtp_draft_lm_head_isq = "Q4K"` in prod
      mistralrs.toml (measured +2.6% decode on the 35B; harmless to
      quality).

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
      RESOLVED 2026-10-01: the Q4/Q5 "divergence" was the broken config
      symlink - the deploy copy was a stale unlinked file frozen at Sep
      28, the live file was canonical and deliberately runs Q5_K_XL.
      Symlink restored + deploy re-synced to live (fd3ef4b8e); Q5 stays.
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

### Perf-opportunity backlog (opened 2026-10-08, ideas, none started)

- [ ] ~M Expert-split MoE placement (llama.cpp `--cpu-moe` style):
      attention/router/shared-experts on GPU, cold experts in RAM.
      Unlocks the 35B on the 3090's 24GB. Requires new work: split
      placement in the MoE experts backend + cross-device dispatch per
      token (placement today is whole-layer only, verified: no
      lru/offload/evict machinery in `mistralrs-core/src/moe/`).
- [ ] ~S Solo-rocm loud fallbacks: fused FP8 GLU quantize (unfused
      path today), gemv -> plain GEMM, HQQ CPU bitpack-at-load.
      Pure wins, no architecture risk.
- [ ] ~S UQFF vs GGUF shootout: same weights through the CLI
      `quantize`/`uqff` path, bench load + decode on this box.
      Unmeasured; native format may win both.
- [ ] ~S Re-sweep MTP n on the cutover binary: n=2 was locked on the
      old binary; verify is faster now and acceptance sits 63-67%,
      so n=3 may win where it lost before. Include draft sampling
      greedy-vs-auto recheck.
- [ ] ~S MTP on the 4B: `Qwen3.5-4B-MTP-GGUF` carries the head but is
      pinned `mtp=false` in prod. Small-model MTP is usually
      high-acceptance; bench before pinning.
- [ ] ~S Graph-capture audit on prod metrics: `graph_dispatch`
      replay vs eager-population vs skipped in steady state. Tells
      whether decode is actually graphed or re-capturing.
- [ ] ~S KV precision on the 35B: 27B already runs split K=Q8_0/
      V=Q4_0; try the same (or lower) on the 35B with a quality check.
      Decode is bandwidth-bound; KV bytes are decode time.
- [ ] ~S KV pool right-sizing: prod `context_len = 262144` reserves
      ~7GB+ real traffic may never touch. Reclaim to a second resident
      or a bigger graph pool; measure resident-set vs Evict behavior.
      Extends to per-model paged budget (or spill-aware KV fitting):
      2026-10-08 the 27B could not load on the 24GB 3090 under the
      global pool (15.5GB weights + 6.5GB KV + fixed reservations
      overflow; MbAmount/ContextSize hard-fails, per-model
      max_model_len does not shrink the pool, graphs-off changes
      nothing). Small GPUs need either per-model `context_len`/budget
      or a mapper that spills more layers to make KV room.
- [ ] ~S Re-bench quants on the cutover binary: Q6_K vs Q4_K_M (27B),
      Q5_K_XL vs Q4_K_XL (35B). The +54% came from kernels, not
      weights, so the quant tradeoff curve moved.
- [ ] ~M FA2 path proof on the cuda role (compute 8.6: FA v2 yes, v3
      no): confirm the kernels actually dispatch there; try the
      `cudnn` feature as challenger. Pairs with the cuda-MTP port.
- [ ] ~M CUDA-graph capture proof on the NVIDIA role: caps claim
      graphs=true but the capture-bug fixes were hip-side; needs its
      own proof run on the 3090.
- [ ] ~S Prefix-caching revisit: `prefix_cache_n` is off to match
      llama-server; multi-turn traffic may win it back. Measure.
- [ ] ~L Split prefill/decode across the two GPUs (prefill-hungry on
      one device, latency decode on the other). Only if single-device
      saturates; most complex item here.
- [x] ~M Per-model paged overrides + `--fit-ctx` (SHIPPED 2026-10-08,
      matrix 0/0/0/0, hardware validation queued on the 3090).
      `[[models]]` gains `context_len`, `memory_mb`, `fit_ctx` (same
      names/semantics as `[paged_attn]`, entry wins; flows through the
      existing precedence incl. eager multi-model entries). Global
      `--fit-ctx` / `[paged_attn] fit_ctx`: post-load hard-budget
      (`MbAmount`/`ContextSize`) misses shrink to available (floor 2048
      tokens, WARN line with requested vs fitted) instead of refusing;
      off by default, refusal path byte-identical (unit-tested both).
      Motivation: 27B+262k-global cannot load on the 24GB 3090 and the
      pool budget was global-only. Sizing audit result: no static
      consumer explains it (activations log 0, base reservation 0,
      MTP-builtin adds nothing, affine 0 in dual) - VRAM timeline
      peaks 17.3GB vs 15.6 settled, so the post-load query reads pool
      pressure (ISQ/upload transients) as permanent occupancy.
- Policy (user-locked 2026-10-08): CPU serving capped at 4B. A CPU
      mirror alias is wanted only for `qwen3.5-4b`; everything larger
      stays GPU-pinned (no such alias = loud absence, never
      silent crawl).

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
      ("no cuda implementation for dtype-to-fp8"); the hqq GPU tests
      pass under --features rocm except the chunk-embedding test,
      which stays gated fleet-wide (pre-existing shared chunk-path
      race, fails pristine rocm-only too). Upstream capability gaps,
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
- [ ] _S mark downstream-facing FFI as such (was §1, one comment) so cleanup does
      not delete it: `cuMemPoolTrimTo`, `cuMemcpy{HtoD,DtoH}Async_v2`, `alloc_pinned`,
      `CudaEvent::elapsed_ms`, `CUDA_ERROR_NOT_SUPPORTED`
      (`cudarc-hip/src/driver/sys.rs:55,260,292,310`). Trivial insurance.
- [x] _S stale deploy comments (was §7; closed 2026-10-01): the 35B MoE
      "ACTIVE default" label corrected to "lazy; the orcarouter 27B is the
      entry default" in the re-synced toml; `mistralrs-test.toml` header
      now states it is a pre-router smoke config (eager loads, no
      demand-loading, no [router] policy) and points at mistralrs.toml
      as the live router config. The `memory_mb = 512` there is
      intentional for smoke runs.
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
- [x] _S document the functional MRS_/MISTRALRS_ env knobs (was §7; done
      2026-10-01): the audit showed the old list half stale - `MRS_GDN_KERNEL`
      (now `MISTRALRS_GDN_KERNEL`), `MISTRALRS_NO_FAST_MMQ`,
      `CANDLE_NO_FAST_MMQ`, and the GDN/FlashInfer selectors were already
      documented, and MRS_NO_FUSED_QKV/FFN, MRS_ATTENTION_DEBUG,
      MRS_DIFFUSION_DEBUG_DUMP, MRS_PROFILE, MRS_TRACE_LAYERS, MRS_DEBUG_PA,
      MRS_MMQ_DEBUG no longer exist in the tree. Newly documented in
      `docs/src/content/docs/reference/environment-variables.md`:
      `MISTRALRS_NO_FAST_MMVQ`, `MISTRALRS_MOE_BACKEND`,
      `MISTRALRS_FORCE_AVX2`, and the expert/debug cluster
      (`MISTRALRS_NAN_TRACE`, `MISTRALRS_NAN_PROBE`,
      `MISTRALRS_LAYER_PROFILE`, `MRS_ZERO_WS`, `MRS_POISON_NAN`); the
      `MISTRALRS_DECODE_ARENA_OVERRIDE` row was corrected to the measured
      x1.25 default + geometric regrow (was "6x the warmup").
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
