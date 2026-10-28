"""
Build the allocator's item list from MEASURED sensitivity curves.

Inputs
  results/sensitivity_qwen06.json   per-linear-layer PPL delta at 2/3/4 bits
  results/w_bits_qwen06.json        whole-model weight curve
  results/kv_bits_qwen06.json       whole-cache curve
  the model config                  parameter counts per module

Output
  results/items_<tag>.json          one entry per decision unit

Two normalisations, both documented because they are the part a reviewer will
question:

* Per-layer weight deltas are noisier than the signal for most layers (78 of 197
  came out NEGATIVE at 4 bits; the noise floor is ~0.12% relative). We keep the
  measured RANKING but rescale so the sum matches the end-to-end measured curve,
  and we clamp negatives to a small positive floor. This preserves what is
  measured (which layers are sensitive, and how much the model costs in total)
  and discards what is not (the per-layer absolute value of a layer that sits
  inside the noise).

* The cache curve was measured globally, not per layer, so its cost is spread
  uniformly across KV groups. That is an assumption, stated as one.

Usage:
    python src/alloc/build_items.py --model C:/models/Qwen3-0.6B --tag qwen06
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

BITS = (2, 3, 4, 8)
FLOOR_REL = 0.0005          # 0.05% relative floor for noise-dominated layers


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def param_shapes(model_id: str) -> dict[str, int]:
    """Parameter counts per linear module, read from config on the meta device."""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.dim() == 2:
            out[name] = int(mod.weight.numel())
    del model
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sens", default="results/sensitivity_qwen06.json")
    ap.add_argument("--wbits", default="results/w_bits_qwen06.json")
    ap.add_argument("--kvbits", default="results/kv_bits_qwen06.json")
    ap.add_argument("--kv-groups", type=int, default=8,
                    help="KV decision units per layer")
    ap.add_argument("--out", default=None)
    ap.add_argument("--w4-override", type=float, default=None,
                    help="replace the end-to-end 4-bit weight cost (relative) "
                         "with a stronger quantiser's, for a sensitivity run")
    args = ap.parse_args()

    sens = load_json(ROOT / args.sens)
    wcur = load_json(ROOT / args.wbits)
    kcur = load_json(ROOT / args.kvbits)
    shapes = param_shapes(args.model)

    # ---- whole-model curves: relative increase per bit width ----------------
    E_w = {r["w_bits"]: r["relative"] for r in wcur["rows"]}
    E_kv = {r["kv_bits"]: r["relative"] for r in kcur["rows"]}
    E_w.setdefault(8, 0.0)
    if args.w4_override is not None:
        print(f"[override] 4-bit weight cost {E_w[4]:.4f} -> {args.w4_override:.4f}")
        E_w[4] = args.w4_override
    E_kv.setdefault(8, 0.0)

    # ---- per-layer measured deltas ------------------------------------------
    per_layer: dict[str, dict[int, float]] = {}
    for r in sens["rows"]:
        base = sens["baseline_ppl"]
        rel = (r["ppl"] - base) / base
        per_layer.setdefault(r["layer"], {})[r["bits"]] = rel

    layers = [n for n in shapes if n in per_layer]
    if not layers:
        print("no overlap between sensitivity rows and model modules", file=sys.stderr)
        return 1

    # rescale per-layer curve so its sum equals the measured end-to-end curve
    items = []
    for bits in (2, 3, 4):
        raw = {n: max(per_layer[n].get(bits, 0.0), 0.0) for n in layers}
        total = sum(raw.values())
        target = E_w.get(bits, 0.0)
        scale = (target / total) if total > 0 else 0.0
        for n in layers:
            per_layer[n][bits] = max(raw[n] * scale, FLOOR_REL)

    for n in layers:
        items.append({
            "name": n, "kind": "w", "n_params": shapes[n],
            "err": {str(b): round(per_layer[n].get(b, 0.0), 8) for b in BITS},
        })

    # ---- KV groups: global curve spread over the groups ---------------------
    cfg = json.loads((Path(args.model) / "config.json").read_text(encoding="utf-8"))
    n_layers = cfg["num_hidden_layers"]
    kv_heads = cfg.get("num_key_value_heads") or cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    per_group_tokens = 2 * kv_heads * head_dim // args.kv_groups   # K and V
    n_kv_items = n_layers * args.kv_groups
    for l in range(n_layers):
        for g in range(args.kv_groups):
            items.append({
                "name": f"kv.{l}.{g}", "kind": "kv",
                "n_params": per_group_tokens,
                "err": {str(b): round(E_kv.get(b, 0.0) / n_kv_items, 8) for b in BITS},
            })

    out = Path(args.out or f"results/items_{args.tag}.json")
    payload = {"model": args.model, "tag": args.tag,
               "note": "per-layer ranking measured, magnitude rescaled to the "
                       "end-to-end curve; KV cost spread uniformly",
               "E_w": E_w, "E_kv": E_kv,
               "n_weight_items": len(layers), "n_kv_items": n_kv_items,
               "items": items}
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {out}: {len(items)} items "
          f"({len(layers)} weight, {n_kv_items} KV)")
    print(f"  E_w  = {E_w}")
    print(f"  E_kv = {E_kv}")
    top = sorted(layers, key=lambda n: -per_layer[n][4])[:5]
    print("  most sensitive (rescaled, 4-bit):")
    for n in top:
        print(f"    {n:44s} {per_layer[n][4]*100:6.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
