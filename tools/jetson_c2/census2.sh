#!/usr/bin/env bash
# Exact allocations at the two long contexts, plus the cells the census could
# not reach with llama-bench.
set -u
BIN="$HOME/c2/llama.cpp/build/bin"
MODEL="$HOME/c2/models/Qwen3-1.7B-Q8_0.gguf"
OUT="$HOME/c2/jetson"; mkdir -p "$OUT"
echo "census2 start $(date -Is)" | tee -a "$OUT/grid.log"

for L in 8192 16384; do
  for t in f16 q8_0 q4_0; do
    timeout 300 "$BIN/llama-cli" -m "$MODEL" -ngl 99 -c "$L" -ctk "$t" -ctv "$t" \
        -n 1 -p "Hello" -v < /dev/null > "$OUT/cli2_L${L}_${t}.log" 2>&1
    rc=$?
    line=$(grep -m1 -oE "size = [0-9.]+ MiB \( *[0-9]+ cells.*" "$OUT/cli2_L${L}_${t}.log")
    err=$(grep -m1 -oE "allocating [0-9.]+ MiB on device 0: .*" "$OUT/cli2_L${L}_${t}.log")
    echo "--- L=$L $t rc=$rc | ${line:-none} | ${err:-}" | tee -a "$OUT/grid.log"
  done
done

for L in 16384 32768; do
  for t in q4_0 q8_0; do
    tag="bench_L${L}_${t}"
    timeout 900 "$BIN/llama-bench" -m "$MODEL" -p "$L" -n 128 -r 2 -ngl 99 \
        -ctk "$t" -ctv "$t" -o json > "$OUT/$tag.json" 2> "$OUT/$tag.log"
    rc=$?
    echo "--- $tag rc=$rc $(grep -oE '"avg_ts": [0-9.]+' "$OUT/$tag.json" | tr '\n' ' ')" | tee -a "$OUT/grid.log"
  done
done
echo "census2 done $(date -Is)" | tee -a "$OUT/grid.log"
