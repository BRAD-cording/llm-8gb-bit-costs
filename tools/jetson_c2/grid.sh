#!/usr/bin/env bash
# Jetson (Orin Nano Super) grid: throughput + perplexity for the three cache types.
set -u
BIN="$HOME/c2/llama.cpp/build/bin"
MODEL="$HOME/c2/models/Qwen3-1.7B-Q8_0.gguf"
DATA="$HOME/c2/data/wikitext2_test.txt"
OUT="$HOME/c2/jetson"; mkdir -p "$OUT"
echo "grid start $(date -Is)" | tee "$OUT/grid.log"

for t in f16 q8_0 q4_0; do
  for L in 4096 8192; do
    tag="bench_L${L}_${t}"
    timeout 900 "$BIN/llama-bench" -m "$MODEL" -p "$L" -n 128 -r 2 -ngl 99 \
        -ctk "$t" -ctv "$t" -o json > "$OUT/$tag.json" 2> "$OUT/$tag.log"
    rc=$?
    echo "--- $tag rc=$rc $(grep -oE '"avg_ts": [0-9.]+' "$OUT/$tag.json" | tr '\n' ' ')" | tee -a "$OUT/grid.log"
  done
done

for t in f16 q8_0 q4_0; do
  tag="ppl_${t}"
  timeout 5400 "$BIN/llama-perplexity" -m "$MODEL" -f "$DATA" -c 4096 --chunks 16 \
      -ngl 99 -ctk "$t" -ctv "$t" > "$OUT/$tag.log" 2>&1
  rc=$?
  echo "--- $tag rc=$rc $(grep -E 'Final estimate' "$OUT/$tag.log")" | tee -a "$OUT/grid.log"
done
echo "grid done $(date -Is)" | tee -a "$OUT/grid.log"
