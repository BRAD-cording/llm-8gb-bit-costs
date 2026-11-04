#!/usr/bin/env bash
# Does the allocator mode explain the 952 MiB cudaMalloc failure?
set +e
BIN="$HOME/c2/llama.cpp/build/bin"
MODEL="$HOME/c2/models/Qwen3-1.7B-Q8_0.gguf"
OUT="$HOME/c2/jetson"; mkdir -p "$OUT"

try() {  # $1 tag, $2 env, $3.. args
  tag="$1"; envs="$2"; shift 2
  timeout 900 env $envs "$BIN/llama-cli" -m "$MODEL" -ngl 99 -n 1 -p "Hello" -v "$@" \
      < /dev/null > "$OUT/t_$tag.log" 2>&1
  rc=$?
  echo "--- $tag rc=$rc"
  grep -E "allocating [0-9.]+ MiB on device|cudaMalloc failed|failed to allocate buffer|KV self size" "$OUT/t_$tag.log" | tail -3
  grep -E "^0.*(eval time|prompt eval)" "$OUT/t_$tag.log" | tail -2
}

try "16k_q8_novmm" "GGML_CUDA_NO_VMM=1" -c 16384 -ctk q8_0 -ctv q8_0
try "32k_q8_novmm" "GGML_CUDA_NO_VMM=1" -c 32768 -ctk q8_0 -ctv q8_0
try "16k_f16_novmm" "GGML_CUDA_NO_VMM=1" -c 16384
try "16k_q8_vmm" "" -c 16384 -ctk q8_0 -ctv q8_0
