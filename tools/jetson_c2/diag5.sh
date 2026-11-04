#!/usr/bin/env bash
# The batch, not the KV, may be what blows up: llama-bench builds a graph for the
# whole -p prompt. Cap it and retry the long contexts.
set +e
BIN="$HOME/c2/llama.cpp/build/bin"
MODEL="$HOME/c2/models/Qwen3-1.7B-Q8_0.gguf"
OUT="$HOME/c2/jetson"; mkdir -p "$OUT"

one() {
  tag="$1"; p="$2"; shift 2
  timeout 900 "$BIN/llama-bench" -m "$MODEL" -p "$p" -n 128 -r 2 -ngl 99 "$@" > "$OUT/b_$tag.log" 2>&1
  rc=$?
  kv=$(grep -m1 "llama_kv_cache.*size =" "$OUT/b_$tag.log" | sed 's/.*size = //; s/ (.*//')
  cb=$(grep -m1 "compute buffer size" "$OUT/b_$tag.log" | sed 's/.*= *//; s/ MiB//' | tr '\n' ' ')
  ts=$(grep -E "pp[0-9]+|tg[0-9]+" "$OUT/b_$tag.log" | sed 's/.*| *\([a-z0-9]*\).*| *\([0-9.]*\) ±.*/\1=\2/' | tr '\n' ' ')
  echo "$tag rc=$rc kv=${kv:-none} [$ts]"
}

echo "== does -b/-ub rescue the long prompt?"
one "q8_p16384_b512" 16384 -b 512 -ub 512 -ctk q8_0 -ctv q8_0
one "f16_p4096_b512"  4096 -b 512 -ub 512
one "q8_p32768_b512" 32768 -b 512 -ub 512 -ctk q8_0 -ctv q8_0
echo "== mem"; free -m | head -2
