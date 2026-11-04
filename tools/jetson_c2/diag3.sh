#!/usr/bin/env bash
# Which f16 configurations allocate, and does NO_VMM change it?
set +e
BIN="$HOME/c2/llama.cpp/build/bin"
MODEL="$HOME/c2/models/Qwen3-1.7B-Q8_0.gguf"
run() {  # $1 label; $2 p; $3 extra args; $4 env prefix
  echo "===== $1   free=$(free -m | awk '/内存|Mem/{print $7}') MiB"
  timeout 420 env $4 "$BIN/llama-bench" -m "$MODEL" -p "$2" -n 16 -r 1 -ngl 99 $3 > /tmp/x.out 2>&1
  echo "rc=$?"; grep -E "NvMap|error|failed|t/s|pp[0-9]+" /tmp/x.out | head -3
  grep -E "llama_kv_cache|compute buffer size" /tmp/x.out | tail -2
}
run "f16 p128"        128  "" ""
run "f16 p128 NO_VMM" 128  "" "GGML_CUDA_NO_VMM=1"
run "f16 p4096"       4096 "" ""
run "f16 p4096 NO_VMM" 4096 "" "GGML_CUDA_NO_VMM=1"
run "q8_0 p4096"      4096 "-ctk q8_0 -ctv q8_0" ""
run "q4_0 p4096"      4096 "-ctk q4_0 -ctv q4_0" ""
