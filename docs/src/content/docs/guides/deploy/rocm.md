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
| `MISTRALRS_MANAGED_WEIGHTS=1` | HIP managed (host-visible, prefetched) weights instead of device-only allocations. On unified-memory APUs the device-only default keeps weights out of process RSS. |
| `MISTRALRS_GGUF_NO_MMAP=1` | Read GGUF shards with `read()` instead of `mmap`; avoids a duplicate file-cache copy on shared-memory machines. |
| `MISTRALRS_GGUF_DROP_HOST_AFTER_LOAD=1` | Release host shard buffers once weights are on-device (pairs with `NO_MMAP`). |
| `MISTRALRS_CUDA_GRAPHS=0` | Disable decode-graph capture on the ROCm path if a capture crash recurs. |
| `MRS_DECODE_ARENA_OVERRIDE=<bytes>` | Raise the decode-graph capture arena if capture fails with an overflow; the log prints the size actually needed. |
| `CANDLE_LT_COMPUTE=32` | Force F32 accumulation in hipBLASLt; required on some RDNA parts where the F16 path errors heuristically. |
| `MRS_NO_FAST_MMQ=1` / `CANDLE_NO_FAST_MMQ=1` | Route large-batch quantized GEMMs through dequantize+hipBLAS instead of the fused MMQ kernels. |

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
- **Decode-graph capture overflow:** raise `MRS_DECODE_ARENA_OVERRIDE`
  to the size printed on the `[cudarc] CAPTURE` log line, or set
  `MISTRALRS_CUDA_GRAPHS=0` to run eager.
- **hipBLASLt `INTERNAL_ERROR` on some shapes:** set
  `CANDLE_LT_COMPUTE=32`; Candle falls back to plain rocBLAS on LT
  errors, so residual failures cost speed, not availability.

For the production hardening around this setup (health checks,
observability, update flow), see the
[production checklist](/guides/deploy/production-checklist/) and
[observability](/guides/deploy/observability/) guides.
