#!/bin/bash
# 4B Q8_0 dense test server (Qwen3.5 hybrid GDN), port 18323
# Tokenizer is embedded in the GGUF; do not pass --tok-model-id
export LD_LIBRARY_PATH=/home/pstreet/LocalAI/rocm-install/lib
cd /home/pstreet/LocalAI/models/unsloth/Qwen3.5-4B-MTP-GGUF
exec /home/pstreet/LocalAI/mistral.rs/target/release/mistralrs serve -p 18323 -m . -f Qwen3.5-4B-Q8_0.gguf --prefix-cache-n 0
