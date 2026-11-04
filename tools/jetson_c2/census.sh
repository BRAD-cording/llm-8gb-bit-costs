#!/usr/bin/env bash
# Can each (L, cache) configuration even be instantiated on the Orin?
# llama-cli prints the allocator's own buffer sizes at load; stdin from /dev/null
# makes it exit after the single generated token.
set +e
BIN="$HOME/c2/llama.cpp/build/bin"
MODEL="$HOME/c2/models/Qwen3-1.7B-Q8_0.gguf"
OUT="$HOME/c2/jetson"; mkdir -p "$OUT"

census() {
  L="$1"; ct="$2"
  tag="cli_L${L}_${ct}"
  timeout 900 "$BIN/llama-cli" -m "$MODEL" -c "$L" -ctk "$ct" -ctv "$ct" -ngl 99 \
      -n 1 -p "Hello" -v < /dev/null > "$OUT/$tag.log" 2>&1
  rc=$?
  kv=$(grep -m1 "llama_kv_cache.*size =" "$OUT/$tag.log" | sed 's/^.*size = //; s/ (.*//')
  cb=$(grep -m1 "compute buffer size" "$OUT/$tag.log" | sed 's/.*= *//')
  mb=$(grep -m1 "model buffer size" "$OUT/$tag.log" | sed 's/.*= *//')
  err=$(grep -iE "NvMap|failed to allocate|error:" "$OUT/$tag.log" | head -1 | cut -c1-55)
  echo "L=$L $ct rc=$rc | kv=${kv:-?} compute=${cb:-?} model=${mb:-?} ${err}"
}

for L in 4096 16384 32768; do
  for ct in f16 q8_0 q4_0; do census "$L" "$ct"; done
done
free -m | head -2
