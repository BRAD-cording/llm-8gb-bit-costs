"""
One command that rebuilds every generated artefact that does not need a GPU
run: the figures, the tidy CSV, the allocator items, the paper-side checks and
the PDF.

Deliberately does NOT re-run measurements -- those take hours and their outputs
are committed under results/. Pass --with-measurements to also re-run the fast
CPU-side ones.

Usage:
    python experiments/run_all.py
    python experiments/run_all.py --with-measurements
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
TEXBIN = Path(r"C:\texportable\texmfs\install\miktex\bin\x64")


def run(label: str, cmd: list[str], cwd: Path | None = None) -> bool:
    t0 = time.time()
    print(f"\n=== {label} ===")
    print("  " + " ".join(str(c) for c in cmd))
    p = subprocess.run(cmd, cwd=str(cwd or ROOT))
    ok = p.returncode == 0
    print(f"  {'ok' if ok else 'FAILED'} ({time.time()-t0:.1f}s)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-measurements", action="store_true")
    args = ap.parse_args()

    steps: list[tuple[str, list[str], Path | None]] = []

    if args.with_measurements:
        steps += [
            ("device probe", [PY, "src/bench/device_probe.py", "--out",
                              "results/probe_4060.json"], None),
            ("bit cost with CIs", [PY, "src/eval/bit_cost_ci.py",
                                   "--model", "C:/models/Qwen3-0.6B",
                                   "--text", "data/calib_arxiv.txt",
                                   "--windows", "48"], None),
        ]

    steps += [
        ("collect measurements -> CSV", [PY, "experiments/collect.py"], None),
        ("figures", [PY, "experiments/make_figures.py"], None),
        ("build allocator items", [PY, "src/alloc/build_items.py",
                                   "--model", "C:/models/Qwen3-0.6B",
                                   "--tag", "qwen06v2",
                                   "--sens", "results/sensitivity_qwen06_v2.json",
                                   "--wbits", "results/w_bits_qwen06_v2.json",
                                   "--kvbits", "results/kv_bits_qwen06.json",
                                   "--out", "results/items_qwen06_v2.json"], None),
        ("solver invariants", [PY, "tests/test_alloc.py"], None),
        ("solver optimality", [PY, "tests/test_alloc_optimality.py"], None),
        ("quantiser self-test", [PY, "src/quant/fakequant.py"], None),
        ("GQA reproduction", [PY, "artifacts/reproduce_gqa_fallback.py"], None),
        ("LaTeX static checks", [PY, "paper/check_tex.py", "paper/main.tex"], None),
        ("numeric audit", [PY, "paper/audit_numbers.py"], None),
        ("table vs data",
         [PY, "experiments/check_table_vs_data.py"], None),
        ("documentation paths",
         [PY, "experiments/check_doc_paths.py"], None),
        ("stale-number guard",
         [PY, "experiments/check_stale_numbers.py"], None),
        ("exchange-rate consistency",
         [PY, "experiments/check_exchange_rate.py"], None),
    ]

    if TEXBIN.exists():
        steps += [
            ("bibtex", [str(TEXBIN / "bibtex.exe"), "main"], ROOT / "paper"),
            ("pdflatex 1", [str(TEXBIN / "pdflatex.exe"),
                            "-interaction=nonstopmode", "main.tex"], ROOT / "paper"),
            ("pdflatex 2", [str(TEXBIN / "pdflatex.exe"),
                            "-interaction=nonstopmode", "main.tex"], ROOT / "paper"),
        ]
    else:
        print(f"\n[skip] no TeX at {TEXBIN}; paper checks still run")

    results = [(label, run(label, cmd, cwd)) for label, cmd, cwd in steps]

    print("\n=== summary ===")
    for label, ok in results:
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}")
    bad = [l for l, ok in results if not ok]
    print(f"\n{'ALL OK' if not bad else str(len(bad)) + ' FAILED: ' + ', '.join(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
