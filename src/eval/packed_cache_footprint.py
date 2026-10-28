"""
Materialise the cache saving instead of modelling it.

The paper's central mechanism is that a cache bit is cheaper than a weight bit,
and every memory number in the capacity table comes from a cost model. Our
attention patch quantises and immediately dequantises, so it pays the arithmetic
but stores nothing smaller --- a reviewer is entitled to ask whether the saving
exists at all.

This script answers that directly: it runs a prefill at a given length, captures
each layer's key and value tensors, packs them to 4 bits into real device
allocations (two values per byte, plus a per-group scale and minimum), and reports
what the allocator actually holds for the packed cache against the fp16 original.

It does not run attention on the packed form. That needs a chunked dequantising
attention, which is a kernel-level change we did not make; what is measured here is
therefore the resident footprint, which is the quantity the capacity argument uses.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, _ROOT)


def pack4(x: torch.Tensor, group: int = 32):
    """Asymmetric 4-bit packing along the last dim, two values per byte."""
    flat = x.reshape(-1, x.shape[-1])
    n, d = flat.shape
    pad = (-d) % group
    if pad:
        flat = torch.cat([flat, flat.new_zeros(n, pad)], dim=1)
    g = flat.reshape(n, -1, group)
    mn = g.amin(dim=-1, keepdim=True)
    mx = g.amax(dim=-1, keepdim=True)
    scale = (mx - mn).clamp_min(1e-8) / 15.0
    q = ((g - mn) / scale).round_().clamp_(0, 15).to(torch.uint8)
    q = q.reshape(n, -1)
    hi = q[:, 0::2] << 4
    lo = q[:, 1::2]
    packed = (hi | lo).contiguous()
    return packed, scale.half().contiguous(), mn.half().contiguous()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="8192,16384,32768")
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--passkey", default="48213")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True).eval()
    from bench.gqa_patch import enable_gqa_sdpa
    enable_gqa_sdpa(model, kv_bits=None)

    out_rows = []
    for L in [int(x) for x in args.lengths.split(",")]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()

        ids = torch.randint(100, 2000, (1, L), device="cuda")
        toks = tok.decode(ids[0].tolist())
        enc = tok(toks, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        with torch.inference_mode():
            res = model(enc, use_cache=True)
        torch.cuda.synchronize()
        cache = res.past_key_values
        # transformers 5.x DynamicCache: iterate .layers, fall back to indexing
        layers = getattr(cache, "layers", None)
        captured = []
        if layers is None:
            layers = [cache[i] for i in range(len(cache))]
        for i, layer in enumerate(layers):
            if layer is None:
                continue
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if k is None:
                k, v = layer[0], layer[1]
            captured.append((i, (k, v)))

        fp16_bytes = 0
        packed_bytes = 0
        scales_bytes = 0
        for _idx, kv in captured:
            for t in kv:
                if t is None:
                    continue
                fp16_bytes += t.numel() * t.element_size()
                p, s, mn = pack4(t, args.group)
                packed_bytes += p.numel() * p.element_size()
                scales_bytes += s.numel() * s.element_size() + mn.numel() * mn.element_size()

        # hold the packed tensors so the allocator really has them resident
        held = []
        for _idx, kv in captured:
            for t in kv:
                if t is None:
                    continue
                held.extend(pack4(t, args.group))
        torch.cuda.synchronize()
        packed_resident = torch.cuda.memory_allocated() - base
        peak = torch.cuda.max_memory_allocated()

        row = {"seq_len": L, "layers_captured": len(captured),
               "fp16_kv_MB": round(fp16_bytes / 1024 ** 2, 0),
               "packed_nibble_MB": round(packed_bytes / 1024 ** 2, 0),
               "scales_mn_MB": round(scales_bytes / 1024 ** 2, 0),
               "packed_total_MB": round((packed_bytes + scales_bytes) / 1024 ** 2, 0),
               "compression": round(fp16_bytes / max(1, packed_bytes + scales_bytes), 2),
               "measured_resident_delta_MB": round(packed_resident / 1024 ** 2, 0),
               "peak_MB": round(peak / 1024 ** 2, 0)}
        out_rows.append(row)
        print(json.dumps(row), flush=True)
        del held
        torch.cuda.empty_cache()

    payload = {"model": args.model,
               "kv_per_token_MB": round(2 * 2 * cfg.num_hidden_layers
                                        * cfg.num_key_value_heads * cfg.head_dim
                                        / 1024 ** 2, 5),
               "group": args.group, "rows": out_rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(payload, open(args.out, "w"), indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
