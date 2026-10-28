# Artifact: what is here and how to reproduce it

This repository contains every measurement and every script behind the paper.
Nothing is hand-transcribed: the tables are generated from JSON that the scripts
in `src/bench/` write.

## 1. Layout

```
paper/            main.tex, refs.bib          the manuscript (7 pages, compiles clean)
                  check_tex.py                static LaTeX checks (no TeX needed)
                  audit_numbers.py            manuscript numbers vs result files
                  figures/                    generated PDFs + rendered PNGs
lit/              corpus.csv (108 rows)       the survey, with derived columns
                  GAP_MATRIX.md               the gap analysis
                  CITATION_AUDIT.md           87/87 references machine-verified
                  SEARCH_PROTOCOL.md          queries, windows, inclusion rules
src/bench/        device_probe.py             B_eff, P_eff, rho
                  regime_sweep.py             TTFT/TPOT vs context length
                  synth_sweep.py              same, on an architecture-exact
                                              random-weight model
                  gqa_patch.py                the attention dispatch fix
src/quant/        fakequant.py                weight + KV simulated quantisers
src/eval/         ppl.py                      perplexity + layer sensitivity
                  kv_bits_sweep.py            cache quantisation curve
                  weight_bits_sweep.py        weight quantisation curve
                  adversarial_weight.py       strengthening the weight side
                  bit_cost_ci.py              48-window paired bootstrap CIs
src/alloc/        build_items.py              measured curves -> allocator items
                  dp_alloc.py                 the allocation solver
experiments/      collect.py                  JSON -> tidy CSV, deduplicated
                  make_figures.py             all three figures from data
tools/            http_get.py                 generic resumable downloader
                  ms_download.py              resumable ModelScope fetcher
                  jetson.py                   SSH/SFTP helper
                  verify_citations.py         arXiv title-match audit
                  make_bib.py                 refs.bib generator
tests/            test_alloc.py               solver invariants
                  test_alloc_optimality.py    brute-force optimality (gap = 0)
artifacts/        reproduce_gqa_fallback.py   the 32x finding, model-free
results/          every measurement + the notes that interpret them
```

## 1b. Rebuilding the paper

Everything that is generated can be rebuilt with one command:

```bash
python experiments/run_all.py
```

It regenerates the figures, the tidy CSV, the allocator item list, then runs the
solver invariants, the brute-force optimality check, the quantiser self-test, the
model-free GQA reproduction, the LaTeX static checks, the numeric audit, the
table-versus-data comparison, the documentation path check, the stale-number
guard and the exchange-rate check, and finally rebuilds the PDF. Sixteen stages;
the current state is all green.

### Obtaining a specific state

The submission snapshot is tagged:

```bash
git checkout v0.1-submission
```

Line endings are normalised to LF by `.gitattributes`. That is deliberate: an
earlier revision uploaded scripts to the Jetson with CRLF and bash tried to
execute `$'/home/brad\r'`, so anything that runs on a Linux target must be LF in
the repository rather than on the developer's disk.

Individual pieces:

```bash
cd paper
python check_tex.py main.tex        # structure, tables, math, refs, cites
python audit_numbers.py             # every factor/percent claim vs the data
bibtex main && pdflatex main && pdflatex main
```

The TeX toolchain is a portable MiKTeX; `docs/ENVIRONMENT.md` §4b records how it
was installed without administrator rights and which mirror was used.

## 2. Reproducing the system results

**No model download is needed** for the headline system findings. The
architecture-exact random-weight path reproduces them because latency and memory
depend on tensor shapes and dtypes, not on weight values.

```bash
python src/bench/synth_sweep.py --config-from Qwen/Qwen3-0.6B \
    --precision fp16 --attn sdpa --lengths 1024,4096,8192,16384 \
    --out results/synth_qwen06_fp16.json

python src/bench/synth_sweep.py --config-from Qwen/Qwen3-0.6B \
    --precision fp16 --attn gqa_sdpa --lengths 1024,4096,8192,16384 \
    --out results/synth_qwen06_fp16_gqa.json

python experiments/collect.py      # -> results/system_matrix.csv + the pivot table
```

Expected pivot (RTX 4060 Laptop, Windows, torch 2.6.0+cu124):

```
      L       sdpa   gqa_sdpa  speedup
    128      29.44      25.39     1.2x
   1024     162.67      62.36     2.6x
   4096    2030.15     358.85     5.7x
   8192      spill     988.65        -
  16384        OOM    2801.49        -
```

## 3. The model-free reproduction of the 32x finding

```bash
python artifacts/reproduce_gqa_fallback.py
```

Prints, for identical tensors: fused-with-expansion ~84 MB, math ~2,693 MB, and
verifies the two agree numerically. This is the artifact a reviewer should run
first.

## 4. Three measurement traps this code was written to avoid

Each of these produced plausible-looking but wrong numbers before it was caught,
and each is now either fixed or actively detected.

1. **Timing outside `torch.inference_mode()`.** Autograd retains activations for
   a backward pass that never runs; the measured footprint inflates ~4x and
   spurious OOMs appear at moderate context. `regime_sweep.sweep_one` wraps every
   timed call.

2. **Logits over the whole prompt.** With a 151936-token vocabulary the
   `lm_head` projection over all positions is gigabytes. Real engines project
   one position; `synth_sweep` times the base model for prefill.

3. **Silent spill on Windows.** WDDM backs an over-budget allocation with system
   RAM instead of raising OOM; throughput collapses ~30x with no error. Every row
   is checked against `0.92 * total_vram` and marked `spilled` rather than
   silently reported. `collect.py` prints such rows as `spill`.

## 5. Known limits of this artifact

- The `nf4` path is bitsandbytes, whose dequantisation is not fused into the
  matmul. The "no speedup" result is a statement about this stack.
- Layer-wise sensitivity curves in `dp_alloc.py` are placeholders until
  `src/eval/ppl.py --calibrate` is run on real weights with a corpus. Provisional
  allocation numbers are labelled as such in `results/CAPACITY_TABLE.md`.
- The Jetson install pulls modular CUDA wheels (cudnn alone is 651 MB); on a slow
  link this dominates setup time. `docs/ENVIRONMENT.md` records why the package
  index is a Chinese mirror.

## 6. Hardware and software recorded for every run

`docs/ENVIRONMENT.md` holds the exact stack. Each measurement JSON records
device, precision, torch version and (for the Jetson) the power mode, because
power mode was already shown to shift results on this class of device.
