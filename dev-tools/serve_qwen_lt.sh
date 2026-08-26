#!/bin/bash
# 27B with hipBLASLt debug prints (CANDLE_LT_DEBUG=1)
export LD_LIBRARY_PATH=/home/pstreet/LocalAI/rocm-install/lib
export CANDLE_LT_DEBUG=1
cd /home/pstreet/LocalAI/models/orcarouter/Qwen3.8-27B-Uncensored-GGUF
exec /home/pstreet/LocalAI/mistral.rs/target/release/mistralrs serve -p 18322 -m . -f Qwen3.8-27B-Uncensored-Q6_K.gguf --mmproj mmproj-Qwen3.8-27B-Uncensored-f16.gguf --tok-model-id Qwen/Qwen3.8-27B --prefix-cache-n 0
