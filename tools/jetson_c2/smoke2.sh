#!/usr/bin/env bash
set +e
BIN="$HOME/c2/llama.cpp/build/bin"
MODEL="$HOME/c2/models/Qwen3-1.7B-Q8_0.gguf"
echo "== help: cache flags"
"$BIN/llama-bench" --help 2>&1 | grep -iE "ctk|ctv|cache-type|-fa|flash" | head -5
echo "== minimal run"
timeout 200 "$BIN/llama-bench" -m "$MODEL" -p 64 -n 16 -ngl 99 2>&1 | head -12
echo "== with ctk"
timeout 200 "$BIN/llama-bench" -m "$MODEL" -p 64 -n 16 -ngl 99 -ctk q8_0 -ctv q8_0 2>&1 | head -6
