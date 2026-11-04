#!/usr/bin/env bash
# Where exactly does the f16 path stop on the Orin, and do the packed caches
# survive the long contexts we need?
set +e
BIN="$HOME/c2/llama.cpp/build/bin"
MODEL="$HOME/c2/models/Qwen3-1.7B-Q8_0.gguf"
OUT="$HOME/c2/jetson"; mkdir -p "$OUT"

one() {  # $1 tag  $2 p  $3 cache-args
  tag="$1"
  timeout 600 "$BIN/llama-bench" -m "$MODEL" -p "$2" -n 16 -r 1 -ngl 99 $3 \
      > "$OUT/diag_$tag.log" 2>&1
  rc=$?
  kv=$(grep -m1 "llama_kv_cache.*size =" "$OUT/diag_$tag.log" | sed 's/.*size = //; s/ (.*//')
  cb=$(grep -m1 "compute buffer size" "$OUT/diag_$tag.log" | sed 's/.*size = //')
  last=$(grep -vE "NvMap" "$OUT/diag_$tag.log" | tail -1 | cut -c1-70)
  echo "$tag rc=$rc kv=${kv:-?} compute=${cb:-?} | $last"
}

echo "== f16 bisect"
for p in 512 1024 2048 3072 4096; do one "f16_p$p" "$p" ""; done
echo "== packed caches at the contexts we need"
one "q8_p16384" 16384 "-ctk q8_0 -ctv q8_0"
one "q8_p32768" 32768 "-ctk q8_0 -ctv q8_0"
one "q4_p32768" 32768 "-ctk q4_0 -ctv q4_0"
echo "== mem"; free -m | head -2
