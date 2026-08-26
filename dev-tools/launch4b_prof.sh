#!/bin/bash
# Fully detached 4B profiler launcher. Writes to /tmp/opencode/serve4b_prof.log
LOG=/tmp/opencode/serve4b_prof.log
(
  setsid env LD_LIBRARY_PATH=/home/pstreet/LocalAI/rocm-install/lib \
    LAYER_PROFILE=1 GDN_PROFILE=1 MRS_PROFILE=1 \
    bash -c 'cd /home/pstreet/LocalAI/models/unsloth/Qwen3.5-4B-MTP-GGUF && exec /home/pstreet/LocalAI/mistral.rs/target/release/mistralrs serve -p 18323 -m . -f Qwen3.5-4B-Q8_0.gguf --prefix-cache-n 0' \
    > "$LOG" 2>&1 < /dev/null &
  echo $! > /tmp/opencode/serve4b_prof.pid
)
echo "launcher done"
