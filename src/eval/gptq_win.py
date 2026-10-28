"""
Official GPTQ on Windows with the 4060, in the same stages as the Jetson script.

Windows is possible after all: the earlier failure was a stale `pkg_resources`
in the 3.13 environment, not the Python version. Two workarounds are needed and
both are recorded in the artifact:

  * `tools/gptq_compat` re-exports `no_init_weights`, which moved from
    `transformers.modeling_utils` to `transformers.initialization` in 5.x.
  * gptqmodel is installed from its sdist with `--no-build-isolation`.

Usage:
    python src/eval/gptq_win.py --stage fp16
    python src/eval/gptq_win.py --stage quantise
    python src/eval/gptq_win.py --stage eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import tools.gptq_compat  # noqa: E402,F401  (must precede gptqmodel)


def load_windows(tok, path, seq_len, n):
    text = open(path, encoding="utf-8", errors="ignore").read()
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n = min(len(ids) // seq_len, n)
    return ids[: n * seq_len].view(n, seq_len)


@torch.inference_mode()
def nll(model, win):
    out = []
    for i in range(win.shape[0]):
        x = win[i: i + 1].to("cuda")
        out.append(float(model(x, labels=x).loss))
    return out


def rel(a, b):
    d = sum(y - x for x, y in zip(a, b)) / len(a)
    return 100 * (pow(2.718281828459045, d) - 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fp16", "quantise", "eval"])
    ap.add_argument("--model", default="C:/models/Qwen3-1.7B")
    ap.add_argument("--save", default="C:/models/Qwen3-1.7B-gptq")
    ap.add_argument("--load", default=None, help="checkpoint to evaluate in the eval stage")
    ap.add_argument("--text", default="data/calib_arxiv.txt")
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--cal-windows", type=int, default=0,
                    help="calibration texts drawn from text BEYOND the eval "
                         "windows; 0 keeps the old (2-text) behaviour")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--out", default="results/gptq_official_17b.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    win = load_windows(tok, args.text, args.seq_len, args.windows)
    ev = win[: max(1, args.windows - max(2, args.windows // 8))]
    if args.cal_windows:
        # Calibrate on text after the evaluation block so the two never overlap.
        # Official GPTQ uses 128 x 2048 tokens; our first runs used two texts,
        # which biases the weight side downwards and therefore favours our own
        # conclusion, so this knob exists to test that.
        text = open(args.text, encoding="utf-8", errors="ignore").read()
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        start = args.windows * args.seq_len
        need = args.cal_windows * args.seq_len
        seg = ids[start: start + need]
        n = seg.shape[0] // args.seq_len
        cal = seg[: n * args.seq_len].view(n, args.seq_len)
    else:
        cal = win[len(ev):]
    n_cal = int(cal.shape[0])
    torch.save({"cal": cal.cpu(), "ev": ev.cpu()}, ".tmp/gptq_win_windows.pt")
    print(f"{ev.shape[0]} eval + {cal.shape[0]} calibration windows", flush=True)

    if args.stage == "fp16":
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="cuda",
            trust_remote_code=True).eval()
        vals = nll(m, ev)
        payload = {"stage": "fp16", "windows": int(ev.shape[0]),
                   "mean_nll": round(sum(vals) / len(vals), 6)}
        json.dump(payload, open(".tmp/gptq_win_fp16.json", "w"), indent=2)
        print("fp16 mean NLL", payload["mean_nll"])
        return 0

    if args.stage == "quantise":
        from gptqmodel import GPTQModel, QuantizeConfig
        print("loading for quantisation...", flush=True)
        from gptqmodel.utils.backend import BACKEND
        # Windows has no usable Triton, and gptqmodel's AUTO picks a triton
        # kernel there; TORCH is the pure-PyTorch path (the same one the Jetson
        # selected on its own).
        model = GPTQModel.load(args.model, QuantizeConfig(bits=args.bits,
                                                          group_size=args.group),
                               backend=BACKEND.TORCH)
        calib = [tok.decode(cal[i]) for i in range(cal.shape[0])]
        print(f"quantising, {len(calib)} calibration texts...", flush=True)
        model.quantize(calib)
        model.save(args.save)
        print("saved to", args.save)
        return 0

    from gptqmodel import GPTQModel
    print("loading quantised model...", flush=True)
    from gptqmodel.utils.backend import BACKEND
    m = GPTQModel.load(args.load or args.save, backend=BACKEND.TORCH)
    hf = getattr(m, "model", None)
    if hf is None:
        raise SystemExit("no underlying HF model")
    try:
        hf.to("cuda")
    except Exception as e:
        print("to(cuda):", type(e).__name__, str(e)[:100])
    hf.eval()
    vals = nll(hf, ev)
    fp16 = json.load(open(".tmp/gptq_win_fp16.json"))["mean_nll"]
    base = [fp16] * len(vals)
    payload = {"model": args.model, "bits": args.bits, "group": args.group,
               "windows": int(ev.shape[0]), "cal_windows": n_cal,
               "impl": "gptqmodel 4.2.5 (official), Windows + 4060",
               "fp16_mean_nll": fp16,
               "gptq_mean_nll": round(sum(vals) / len(vals), 6),
               "weights_gptq_rel_pct": round(rel(base, vals), 3)}
    os.makedirs("results", exist_ok=True)
    json.dump(payload, open(args.out, "w"), indent=2)
    print("GPTQ weights:", payload["weights_gptq_rel_pct"], "%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
