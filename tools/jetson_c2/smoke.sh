#!/usr/bin/env bash
# Smoke test: does llama-bench see the Orin GPU, and what does a tiny run cost?
set +e
export PATH=/usr/local/cuda/bin:$PATH
BIN="$HOME/c2/llama.cpp/build/bin"
MODEL="$HOME/c2/models/Qwen3-1.7B-Q8_0.gguf"
echo "== devices"; "$BIN/llama-bench" --list-devices 2>&1 | head -8
echo "== smoke"
timeout 300 "$BIN/llama-bench" -m "$MODEL" -c 2048 -ctk q8_0 -ctv q8_0 -p 64 -n 16 -r 1 -ngl 99 2>&1 | tail -8
echo "== mem"; free -m | head -2
