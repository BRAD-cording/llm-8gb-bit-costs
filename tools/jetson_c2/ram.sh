#!/usr/bin/env bash
# Peak unified memory while a cell runs, read from tegrastats.
set -u
BIN="$HOME/c2/llama.cpp/build/bin"
MODEL="$HOME/c2/models/Qwen3-1.7B-Q8_0.gguf"
OUT="$HOME/c2/jetson"; mkdir -p "$OUT"
echo "ram start $(date -Is)" | tee -a "$OUT/grid.log"

for cell in "4096 f16" "16384 q8_0" "32768 q4_0"; do
  set -- $cell; L=$1; t=$2
  tegrastats --interval 200 > "$OUT/ram_L${L}_${t}.log" 2>&1 &
  tg=$!
  timeout 900 "$BIN/llama-bench" -m "$MODEL" -p "$L" -n 128 -r 2 -ngl 99 \
      -ctk "$t" -ctv "$t" -o json > "$OUT/rambench_L${L}_${t}.json" 2>/dev/null
  rc=$?
  sleep 1; kill $tg 2>/dev/null; wait $tg 2>/dev/null
  peak=$(grep -oE 'RAM [0-9]+/[0-9]+MB' "$OUT/ram_L${L}_${t}.log" | sed 's/RAM //;s|/.*||' | sort -n | tail -1)
  echo "--- L=$L $t rc=$rc peak_RAM=${peak:-?}MB $(grep -oE '"avg_ts": [0-9.]+' "$OUT/rambench_L${L}_${t}.json" | tr '\n' ' ')" | tee -a "$OUT/grid.log"
done
echo "ram done $(date -Is)" | tee -a "$OUT/grid.log"
