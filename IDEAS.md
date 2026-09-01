# Ideas

Candidate work items captured from the current repo state (branch landscape, `todo!()` sites, and feature gaps). Ordered roughly from "pick up active WIP" to "infra". Each entry notes the concrete hook in the codebase so it is grounded, not speculative.

Status: brainstorm, not a plan. Nothing here is scoped yet.

---

## 1. Pick up active WIP (current branch: `wip/pstreet-rocm`)

- **ROCm decode-graph capture.** Commit `75c15b4d` (`wip(rocm): decode-graph capture arena sizing, overflow retry, LRU eviction`) is still a WIP. Scopes: arena sizing policy, overflow retry path, LRU eviction of captured graphs.
- **`qvm_split_k` kernel.** `mistralrs-quant/src/metal_kernels` still has `todo!("qvm_split_k")` (a metal/AMD quant path that is unimplemented).

## 2. Quantization

Lots of unmerged feature branches here; these are genuine open areas.

- **EXL2.** A full `exl2_quant` branch exists but EXL2 is not wired into `Cargo` features or the quant registry. Task: land the engine end-to-end (load, matmul, CLI `--quant` detection, docs).
- **IQ quants.** `q2_k`, `q4_k`, `q6_k` are present, but the metal `qvm_split_k` path is unfinished. Task: complete/validate the IQ support surface across backends.
- **FP8 GEMM consolidation.** Branches `fp8_gemm`, `faster_fp8_kernel`, `cublaslt_vec32_u8` each touch the FP8 path. The tree already has `scalar_fp8`, `vector_fp8`, `blockwise_fp8`, `f8q8`. Opportunity: consolidate into one clean executor with a single dispatch surface.
- **More UQFF.** `more_uqff` branch. Task: extend the in-situ quantization (UQFF) format coverage.
- **Bits-and-Bytes.** `bitsandbytes` engine dir exists with `bitsandbytes_gemv` and `bitsandbytes_isq` branches. Task: complete + wire bnb support.

## 3. Models

`mistralrs-core/src/models/` has 27 model impls in.

- **Mamba2 / SSM.** SSM support is currently only partial (present in `gpt_oss.rs` and `granite.rs`). A `mamba2` branch hints at a dedicated port.
- **T5 / seq2seq.** `t5_seq2seq` branch only; there is no `t5` engine. Encoder-decoder support is a distinct gap.
- **DeepSeek MLA / flash-MLA.** `deepseek_mla` + `flash_mla` branches, unmerged.
- **Prune dead loaders.** `mistralrs-core/src/pipeline/loaders/normal_loaders.rs` has 11 `todo!()` sites spread across architecture loaders that are wired into detection but unimplemented. Either complete them or remove them from the dispatch paths so they stop advertising support.
- **Other branch-only models:** `qwen2.5_omni`, `minicpmo_whisper`, `gpt_oss` refinements, `qwen3_moe_faster` / `qwen3_grouped_moe_testing`.

## 4. Perf / serving

- **Grouped GEMM for MoE.** `grouped_gemm_impl` and `grouped_gemm_proper` branches, targeting MoE decode throughput.
- **KV cache quantization.** `kv_quant` branch; reduces memory for long contexts.
- **Prefix cache to disk.** `prefix_cache_disk` branch; warm caches that survive restarts.
- **Async decode runner.** `perf/async-decode-runner` branch; overlap transfer/sampling with compute.
- **Paged attention profiling.** `paged_attn_profile` branch; instrumentation to find the real bottlenecks.

## 5. Infra / low-risk

- **`todo!`/`FIXME` triage.** 53 `todo!`/`FIXME`/`unimplemented!()` sites across the tree. A pass to either resolve them or turn the load-bearing ones into tracked coverage.
- **OpenTelemetry.** `opentelem` branch; distributed tracing for the server.
- **Stress test harness.** `stresstest` branch + CI reliability (`ci/release-reliability`); add a repeatable soak/stress target to CI.

---

Note: branch names reflect what exists on the remotes at capture time and will drift. Verify before relying on any single entry.