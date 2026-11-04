#!/usr/bin/env bash
# llama-bench takes the context from -p (same as the Windows driver). Find out
# why a quantised V cache trips the Tegra NvMap allocator, and whether an env switch fixes it.
set +e
BIN="$HOME/c2/llama.cpp/build/bin"
MODEL="$HOME/c2/models/Qwen3-1.7B-Q8_0.gguf"
show() { grep -E "error|NvMap|qwen3|t/s|failed|abort" "$1" | head -4; }

echo "===== f16 baseline"
timeout 240 "$BIN/llama-bench" -m "$MODEL" -p 128 -n 16 -r 1 -ngl 99 > /tmp/b.out 2>&1
echo "rc=$?"; show /tmp/b.out

echo "===== q8_0 KV"
timeout 240 "$BIN/llama-bench" -m "$MODEL" -p 128 -n 16 -r 1 -ngl 99 -ctk q8_0 -ctv q8_0 > /tmp/b.out 2>&1
echo "rc=$?"; show /tmp/b.out

echo "===== q8_0 KV + fa on"
timeout 240 "$BIN/llama-bench" -m "$MODEL" -p 128 -n 16 -r 1 -ngl 99 -ctk q8_0 -ctv q8_0 -fa on > /tmp/b.out 2>&1
echo "rc=$?"; show /tmp/b.out

echo "===== K only q8_0"
timeout 240 "$BIN/llama-bench" -m "$MODEL" -p 128 -n 16 -r 1 -ngl 99 -ctk q8_0 > /tmp/b.out 2>&1
echo "rc=$?"; show /tmp/b.out

echo "===== V only q8_0"
timeout 240 "$BIN/llama-bench" -m "$MODEL" -p 128 -n 16 -r 1 -ngl 99 -ctv q8_0 > /tmp/b.out 2>&1
echo "rc=$?"; show /tmp/b.out

echo "===== q8_0 KV with GGML_CUDA_NO_VMM=1"
GGML_CUDA_NO_VMM=1 timeout 240 "$BIN/llama-bench" -m "$MODEL" -p 128 -n 16 -r 1 -ngl 99 -ctk q8_0 -ctv q8_0 > /tmp/b.out 2>&1
echo "rc=$?"; show /tmp/b.out

echo "===== q8_0 KV with GGML_CUDA_DISABLE_GRAPHS=1"
GGML_CUDA_DISABLE_GRAPHS=1 timeout 240 "$BIN/llama-bench" -m "$MODEL" -p 128 -n 16 -r 1 -ngl 99 -ctk q8_0 -ctv q8_0 > /tmp/b.out 2>&1
echo "rc=$?"; show /tmp/b.out
