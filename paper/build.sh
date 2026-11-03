#!/usr/bin/env bash
# Build both manuscripts with the portable MiKTeX that produced the last submission.
# Usage: ./build.sh            -> paper/main.pdf (IEEEtran master)
#        ./build.sh jsa        -> paper/jsa/main.pdf (elsarticle copy)
set -euo pipefail

BIN="/c/texportable/texmfs/install/miktex/bin/x64"
DIR="${1:-.}"

cd "$(dirname "$0")"
[ "$DIR" != "." ] && cd "$DIR"

export PATH="$BIN:$PATH"

echo "== pdflatex(1) in $(pwd)"
pdflatex -interaction=nonstopmode -file-line-error main.tex > /tmp/tex1.log 2>&1 || true

if [ -f main.aux ] && grep -qF '\bibdata' main.aux; then
  echo "== bibtex"
  bibtex main > /tmp/bibtex.log 2>&1 || true
fi

echo "== pdflatex(2)"
pdflatex -interaction=nonstopmode -file-line-error main.tex > /tmp/tex2.log 2>&1 || true
echo "== pdflatex(3)"
pdflatex -interaction=nonstopmode -file-line-error main.tex > /tmp/tex3.log 2>&1 || true

echo "--- errors / undefined"
grep -n "^\./\|^!\|Undefined control sequence\|LaTeX Error" /tmp/tex3.log | head -20 || true
grep -c "Overfull" /tmp/tex3.log || true
grep -n "Output written" /tmp/tex3.log || true
