# Measurement artifact: bit costs and device-dependent capacity for LLM inference on 8 GB devices

Artifact for the manuscript *"The cache is cheaper than the weights: measured bit
costs and device-dependent capacity for LLM inference on 8 GB devices"*
(desk-rejected by the *Journal of Systems Architecture* on 2026-11-04 for lack of
novelty; repositioned and targeted at the *IEEE Internet of Things Journal*).

Every number, table and figure in the paper is produced by a script in this
repository, and every table is generated from a measurement file rather than
transcribed. The measurements themselves take hours and are committed under
`results/`; a single command rebuilds everything derived from them --- the
figures, the tidy CSV, the allocator inputs, the paper-side checks and the
PDF --- and then runs the checks. Pass `--with-measurements` to also re-run the
three fast local measurements.

```bash
python experiments/run_all.py             # 15 stages + the PDF; ALL OK on success
python experiments/run_all.py --with-measurements   # + the three fast local measurements
```

## What the paper claims, and where to check it

| Claim | Script | Result file |
|---|---|---|
| The serving stack sets the memory budget before any bit does ($32\times$ attention peak, $5.8\times$ prefill) | `src/bench/gqa_patch.py`, `src/bench/regime_sweep.py` | `results/stack_*.json` |
| 4-bit weights buy little or no time | `src/bench/*`, `src/eval/deploy_compare.py` | `results/deploy_*.json` |
| One cache bit costs less accuracy than one weight bit ($9.0\times$, $13.3\times$) | `src/eval/strength_matched.py`, `src/eval/gptq_win.py`, `src/eval/kivi_official_eval.py` | `results/strength_*.json`, `results/gptq_official_*.json`, `results/kivi_official_*.json` |
| The exchange rate is bit-dependent, not an artefact of one 4-bit point ($9.5\times$ at 4--6 weight bits, $0.5\times$ at 6--8) | `src/eval/bit_cost_cross.py` | `results/cross_06b_rtn.json` |
| At 4k tokens the shipped recipe (4-bit weights, fp16 cache) costs $+36.67\%$ in 766 MB against $+1.49\%$ in 869 MB for 8-bit weights with an 8-bit cache | `src/eval/bit_cost_cross.py` (`--seq-len 4096`) | `results/cross_06b_4k.json` |
| Same ordering on WikiText-2 ($10.9\times$, $7.7\times$) | `src/eval/strength_matched.py --text data/wikitext2_test.txt` | `results/strength_*_wikitext.json` |
| At 4 bits the cache scheme does not change long-context retrieval; at 2 bits it decides (5/5 vs 0/5) | `src/eval/passkey_eval.py`, `src/eval/passkey_long.py` | `results/passkey_*.json` |
| Packed 4-bit cache is $3.2\times$ smaller, measured on the allocator | `src/eval/packed_cache_footprint.py` | `results/packed_cache_06b.json` |
| The recommended policy executed on a shipped runtime: 8-bit weights, packed cache, 32k context that fp16 cannot reach (9,033 MiB allocated vs 6,873 MiB) | `src/eval/c2_llamacpp.py`, `experiments/make_c2_summary.py` | `results/c2_summary.json`, `results/c2_alloc.txt`, `results/C2_LLAMACPP.md` |
| The same policy on the unified-memory device: each halving of the per-token cache buys one context doubling (8k/16k/32k at fp16/q8_0/q4_0), and the device refuses a 1,792 MiB allocation while its peak sits at 5,235 of 7,620 MB | `tools/jetson_c2/*`, `experiments/make_c2_jetson_summary.py` | `results/c2_jetson/`, `results/c2_jetson_summary.json`, `results/C2_JETSON.md` |
| Device-dependent capacity, and the edge module's measured 3k ceiling | `src/alloc/dp_alloc.py`, `results/jetson_kv16_limit.json` | `results/capacity_table.json` |

## Layout

```
src/bench/      attention dispatch, regime sweeps, the GQA reproduction
src/quant/      fake-quant reference implementations, AWQ-style, KIVI-style,
                our GPTQ, and the adapter around KIVI's own quantiser
src/eval/       perplexity, bit-cost, strength-matched, GPTQ, KIVI, passkey,
                packed-footprint, deployment and the llama.cpp executed-policy
                sweep
src/alloc/      the multiple-choice knapsack solver and its inputs
experiments/    the pipeline plus the individual checkers
results/        measurement files, and short notes explaining the failures
paper/          IEEEtran master; paper/jsa is generated from it
third_party/    vendored third-party code, with its licence
docs/           environment notes, including the ones that cost us a day
```

## Third-party code

`third_party/KIVI/utils_quant.py` is vendored unmodified from
[github.com/jy-yuan/KIVI](https://github.com/jy-yuan/KIVI) at commit `876b4d2`,
under Apache-2.0; the licence is kept beside it. The published GPTQ implementation
(`gptqmodel` 4.2.5) is not vendored — it is installed from PyPI; see
`docs/ENVIRONMENT.md` for the two compatibility fixes it needs.

## Reproducing the environment

`docs/ENVIRONMENT.md` records the exact versions, the package indices that were
reachable, and the three workarounds that were necessary:

* gptqmodel needs `--no-build-isolation`, a `pcre` shim on the edge module
  (`artifacts/pcre_shim.py`), and — on Windows — `tools/gptq_compat.py`, which
  re-exports a name that moved in transformers 5.x and disables `torch._dynamo`
  where there is no Triton.
* The edge module refuses single allocations above roughly 1.2 GB, and aborts the
  allocator rather than paging, which is why the measured context ceiling is
  3k tokens rather than the 67k the cost model predicts.
* The laptop's driver spills to system memory instead of raising, so an
  over-budget configuration runs slowly rather than failing.

## Data

`data/calib_arxiv.txt` is 2,000 arXiv abstracts collected for this study;
`data/wikitext2_test.txt` is the WikiText-2 test split (CC BY-SA), converted from
the Hugging Face parquet release via `hf-mirror.com`. Only relative perplexity
increases are meaningful across corpora; absolute values are not comparable to
published numbers.

## License

Code is released under the MIT licence. The manuscript, figures and measurement
files are released under CC BY 4.0. Third-party components keep their own
licences.
