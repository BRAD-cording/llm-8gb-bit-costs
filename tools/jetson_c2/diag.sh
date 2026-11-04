#!/usr/bin/env bash
# Why does a quantised V cache fail on the Orin? Try the variants one at a time.
set +e
BIN="$HOME/c2/llama.cpp/build/bin"
MODEL="$HOME/c2/models/Qwen3-1.7B-Q8_0.gguf"
try() {
  label="$1"; shift
  echo "===== $label :: $*"
  timeout 180 "$BIN/llama-bench" -m "$MODEL" -p 32 -n 8 -ngl 99 "$@" 2>&1 \
    | grep -vE "^ggml_cuda_init|^  Device 0" | head -8
  echo "     rc=$?"
}
try "k-only q8"   -c 2048 -ctk q8_0
try "v-only q8"   -c 2048 -ctv q8_0
try "both q8 fa-on" -c 2048 -ctk q8_0 -ctv q8_0 -fa on
try "both q8 c4096" -c 4096 -ctk q8_0 -ctv q8_0
echo "== dmesg tail"; dmesg 2>/dev/null | tail -5
echo "== mem"; free -m | head -2
