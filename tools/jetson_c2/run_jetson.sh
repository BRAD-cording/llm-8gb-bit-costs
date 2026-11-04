#!/usr/bin/env bash
# C2 on the Jetson Orin Nano Super: throughput + allocation + peak unified memory
# for Qwen3-1.7B-Q8_0 with f16 / q8_0 / q4_0 KV cache at 4k / 16k / 32k.
# Mirrors src/eval/c2_llamacpp.py: same binary flags, same log regexes.
set +e
HOME_DIR="$HOME/c2"
BIN="$HOME_DIR/llama.cpp/build/bin"
MODEL="$HOME_DIR/models/Qwen3-1.7B-Q8_0.gguf"
OUT="$HOME_DIR/jetson"
mkdir -p "$OUT"

peak_ram() {  # $1 = tegrastats log -> MiB used
  awk '/RAM/ {for(i=1;i<=NF;i++) if($i=="RAM"){split($(i+1),a,"/"); gsub(/MB/,"",a[1]); if(a[1]+0>m)m=a[1]+0}} END{printf "%d", m}' "$1"
}

run_cell() {
  ctx="$1"; ct="$2"; to="${3:-900}"
  tag="L${ctx}_${ct}"
  echo "== $tag $(date +%H:%M:%S) available=$(free -m | awk '/内存|Mem/{print $7}') MiB"
  tegrastats --interval 500 > "$OUT/teg_$tag.log" 2>/dev/null &
  TG=$!
  timeout "$to" "$BIN/llama-bench" -m "$MODEL" -c "$ctx" -ctk "$ct" -ctv "$ct" \
        -p 512 -n 128 -r 2 -ngl 99 -o json \
        > "$OUT/bench_$tag.json" 2> "$OUT/bench_$tag.log"
  rc=$?
  sleep 1; kill $TG 2>/dev/null; wait $TG 2>/dev/null
  # allocation, straight from the runtime's own log
  grep -h "llama_kv_cache" "$OUT/bench_$tag.log" | grep "size =" | tail -2 > "$OUT/alloc_$tag.txt"
  grep -h "compute buffer size" "$OUT/bench_$tag.log" | tail -2 >> "$OUT/alloc_$tag.txt"
  grep -h "model buffer size" "$OUT/bench_$tag.log" | tail -1 >> "$OUT/alloc_$tag.txt"
  echo "tag=$tag ctx=$ctx cache=$ct rc=$rc peak_ram_mib=$(peak_ram "$OUT/teg_$tag.log") timeout=$to" > "$OUT/result_$tag.txt"
  cat "$OUT/result_$tag.txt"
  sleep 8
}

for ctx in 4096 16384 32768; do
  run_cell "$ctx" f16 900
  run_cell "$ctx" q8_0 900
  run_cell "$ctx" q4_0 900
done

echo "== ppl"
for ct in f16 q8_0 q4_0; do
  echo "== ppl $ct $(date +%H:%M:%S)"
  timeout 3600 "$BIN/llama-perplexity" -m "$MODEL" -f "$HOME_DIR/data/wikitext2_test.txt" \
      -c 4096 --chunks 16 -ctk "$ct" -ctv "$ct" -ngl 99 \
      > "$OUT/ppl_$ct.log" 2>&1
  echo "ppl_$ct rc=$?" >> "$OUT/bench.tsv"
  grep -E "Final estimate" "$OUT/ppl_$ct.log" | tail -1
done

echo "== DONE $(date +%H:%M:%S)"
free -m | head -2
