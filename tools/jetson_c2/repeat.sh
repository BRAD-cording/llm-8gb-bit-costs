#!/usr/bin/env bash
# How repeatable is an instantiation? Three attempts per borderline cell.
set -u
BIN="$HOME/c2/llama.cpp/build/bin"
MODEL="$HOME/c2/models/Qwen3-1.7B-Q8_0.gguf"
OUT="$HOME/c2/jetson"; mkdir -p "$OUT"
echo "repeat start $(date -Is)" | tee -a "$OUT/grid.log"

for cell in "8192 f16" "16384 f16" "16384 q8_0" "16384 q4_0" "32768 q8_0" "32768 q4_0"; do
  set -- $cell; L=$1; t=$2
  ok=0
  for i in 1 2 3; do
    timeout 300 "$BIN/llama-cli" -m "$MODEL" -ngl 99 -c "$L" -ctk "$t" -ctv "$t" \
        -n 1 -p "Hello" < /dev/null > "$OUT/rep_L${L}_${t}_$i.log" 2>&1
    [ $? -eq 0 ] && ok=$((ok+1))
  done
  echo "--- L=$L $t ok=$ok/3" | tee -a "$OUT/grid.log"
done
echo "repeat done $(date -Is)" | tee -a "$OUT/grid.log"
