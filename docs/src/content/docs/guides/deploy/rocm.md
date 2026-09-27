---
title: Deploy on ROCm
description: Build mistral.rs from source with the rocm feature and serve models on AMD GPUs.
---

There are no prebuilt ROCm binaries: running on AMD GPUs means building
from source with the `rocm` cargo feature against a ROCm toolkit and a
Candle checkout carrying the ROCm backend. This page covers prerequisites,
environment, build, and a systemd-based serving setup. It reflects a
working deployment on Strix Halo (`gfx1151`); adjust the architecture
string for your GPU.

## Prerequisites

- Rust 1.94+ via [rustup](https://rustup.rs).
- A ROCm toolkit installation: distro packages under `/opt/rocm`, or a
  local build (for example TheRock) at a path of your choice. `hipcc`
  must work.
- Your GPU's gfx target for kernel compilation, e.g. `gfx1151`
  (Strix Halo), `gfx1100` (RX 7900 series), `gfx942` (MI300 series).
  `rocminfo` reports it; when in doubt, check what your ROCm release
  supports.
- A Candle checkout with the ROCm backend, placed as a sibling of the
  mistral.rs checkout (the workspace depends on it by path, e.g.
  `candle-core = { path = "../candle/candle-core" }`).

## Environment

The kernel build script locates ROCm via `CANDLE_ROCM_PATH`,
`ROCM_HOME`, or `ROCM_PATH`, falling back to `hipcc` on `PATH` and then
`/opt/rocm`. `CANDLE_ROCM_ARCH` selects the compilation target.

```bash
export ROCM_PATH=/opt/rocm          # or your local ROCm install
export CANDLE_ROCM_ARCH=gfx1151     # your GPU's gfx target
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
```

Keep these exports in one place (an env file sourced by both your shell
and the service wrapper below) so interactive builds and the server
never disagree about the toolchain.

## Build

```bash
git clone https://github.com/EricLBuehler/mistral.rs.git
cd mistral.rs
cargo build --release --locked -p mistralrs-cli --features rocm
```

The binary lands at `target/release/mistralrs`. Kernel compilation takes
a while on the first build; later builds are incremental. See
[build from source](/developer/from-source/) for the general flag
reference.

### Dual-vendor (`cuda+rocm`) builds

`--features cuda+rocm` builds one binary carrying both vendors. The AMD
role serves (every C kernel is hipcc-built); the NVIDIA side is compiled in
for interop with each driver call gated on cudarc's `is_culib_present()`
probe, and cudarc's dynamic-loading mode means the binary starts and serves
with no NVIDIA runtime on the machine. A `cuda`-only build is not maintained
in this fork.

The NVIDIA half needs the CUDA toolkit even without an NVIDIA GPU: the
default compute-capability detection needs a driver, so GPU-less build boxes
pin it explicitly.

```bash
export CUDA_PATH=/usr/local/cuda-13.3
export PATH=/usr/local/cuda-13.3/bin:$PATH
export CUDA_COMPUTE_CAP=80
cargo build --release --locked -p mistralrs-cli --features cuda+rocm
```

## Run

Smoke-test with a single model, then move to a config file for anything
long-lived:

```bash
# Single model
./target/release/mistralrs serve -m ./model.gguf

# Config file (multiple models, KV cache, speculative decoding, router)
./target/release/mistralrs from-config -f mistralrs.toml
```

TOML serving is documented under [`from-config`](/reference/cli/from-config/)
and the [TOML reference](/reference/cli-toml-config/). MTP speculative
decoding works on ROCm with paged attention; see
[speculative decoding](/guides/perf/speculative-decoding/).

## Device selection

The `--device` flag, `device` under the TOML `[global]` section, or a
per-`[[models]]` `device` key select the backend: `"cpu"`, `"cuda:<N>"`, or
`"hip:<N>"` (indices are per-API enumeration). Omitted, auto-select probes
the serving role and logs the choice plus a capability line
(`graphs/paged_attn/gguf/mtp`). `hip:<N>` also works in `rocm`-only builds
as an alias for the compiled vendor, so one config ports across shapes.
GGUF weights on the hip role refuse cleanly at load time until S2
(QStorage-on-hip) lands; that gate is the `BackendCaps` contract, not a
silent fallback.

## Serve under systemd

Model paths in the TOML are resolved relative to the working directory,
so wrap the server in a small script that sets the ROCm environment and
`cd`s to a stable root before execing `from-config`:

```bash
#!/usr/bin/env bash
set -euo pipefail
source "$HOME/LocalAI/.rocm-env.sh"   # ROCM_PATH, PATH, LD_LIBRARY_PATH, ...
export HIP_VISIBLE_DEVICES=0
cd "$HOME/LocalAI"
exec "$HOME/LocalAI/mistral.rs/target/release/mistralrs" from-config -f "$HOME/LocalAI/models/mistralrs.toml"
```

A user unit then looks like:

```ini
[Unit]
Description=mistral.rs server (ROCm)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/LocalAI
ExecStart=%h/.config/mistralrs-server-wrapper.sh
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

Enable it with `systemctl --user enable --now mistralrs-server`. If the
model needs a tokenizer or config fetch from the Hugging Face Hub at
boot (GGUF `tok_model_id`), the first attempt can fail before the
network is up; `Restart=always` covers that, and standalone GGUF models
avoid the fetch entirely.

## ROCm runtime knobs

These environment variables tune the ROCm path; the full definitions
live in the [environment variables reference](/reference/environment-variables/).

| Variable | Effect |
|---|---|
| `MISTRALRS_MANAGED_WEIGHTS=1` + `CANDLE_MANAGED_WEIGHTS=1` | HIP managed (host-visible, prefetched) weights instead of device-only allocations; the mistral.rs flag gates its loader, the candle flag gates candle's allocator, so a deployment sets both. On unified-memory APUs the device-only default keeps weights out of process RSS. |
| `MISTRALRS_GGUF_NO_MMAP=1` | Read GGUF shards with `read()` instead of `mmap`; avoids a duplicate file-cache copy on shared-memory machines. |
| `MISTRALRS_GGUF_DROP_HOST_AFTER_LOAD=1` | Release host shard buffers once weights are on-device (pairs with `NO_MMAP`). |
| `MISTRALRS_CUDA_GRAPHS=0` | Disable decode-graph capture on the ROCm path if a capture crash recurs. |
| `MISTRALRS_DECODE_ARENA_OVERRIDE=<bytes>` | Raise the decode-graph capture arena if capture fails with an overflow; the log prints the size actually needed. |
| `CANDLE_LT_COMPUTE=32` | Force F32 accumulation in F16 hipBLASLt GEMMs. The F16 heuristic failures that once required this on RDNA parts no longer occur on current ROCm stacks, and F32 is performance-neutral; keep only as a fallback for old stacks. |
| `MISTRALRS_NO_FAST_MMQ=1` | Override the per-dtype MMQ dispatch: route every supported dtype's large-batch quantized GEMMs through dequantize + hipBLASLt. By default only Q6K batches above candle's dequant-GEMM row threshold (`CANDLE_DMM_F16_MIN`, default 256) take that path (MMQ Q6K plateaus far below the dequant GEMM at large batches); Q4K/Q5K/Q8_0 and small batches use the fused MMQ kernels. |
| `CANDLE_DMM_F16_MIN=<rows>` | Minimum `b*m` rows for the dequantize-to-F16 GEMM path (default 256). |

## Verify

```bash
curl http://127.0.0.1:1235/v1/models | python3 -m json.tool
curl http://127.0.0.1:1235/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<alias>","messages":[{"role":"user","content":"What is 37 times 48?"}],"temperature":0,"max_tokens":300}'
```

Responses include `usage.avg_compl_tok_per_sec` (decode throughput).
`GET /metrics` exposes graph capture/replay counters. Compare decode
throughput with MTP on and off to confirm the draft head is helping; see
[speculative decoding](/guides/perf/speculative-decoding/) for the flags.

## Troubleshooting

- **Kernel build cannot find ROCm:** set `CANDLE_ROCM_PATH` explicitly,
  or ensure `hipcc` is on `PATH` at build time.
- **Wrong-architecture kernels (launch failures at runtime):**
  `CANDLE_ROCM_ARCH` did not match the serving GPU. Rebuild with the
  right `gfxXXX` value; the arch is baked in at compile time.
- **Decode-graph capture overflow:** raise `MISTRALRS_DECODE_ARENA_OVERRIDE`
  to the size printed on the `[cudarc] CAPTURE` log line, or set
  `MISTRALRS_CUDA_GRAPHS=0` to run eager.
- **hipBLASLt `INTERNAL_ERROR` on some shapes:** set
  `CANDLE_LT_COMPUTE=32`; Candle falls back to plain rocBLAS on LT
  errors, so residual failures cost speed, not availability.

For the production hardening around this setup (health checks,
observability, update flow), see the
[production checklist](/guides/deploy/production-checklist/) and
[observability](/guides/deploy/observability/) guides.
