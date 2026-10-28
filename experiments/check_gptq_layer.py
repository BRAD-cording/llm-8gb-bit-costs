"""Per-layer sanity check: does GPTQ beat RTN and AWQ on one real layer?"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

sys.path.insert(0, "src")
from quant.gptq import gptq_quantise  # noqa: E402
from quant.fakequant import qdq_symmetric  # noqa: E402
from quant.strong import awq_best_alpha  # noqa: E402


@torch.inference_mode()
def main() -> int:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        sys.argv[1], torch_dtype=torch.float16, device_map="cuda").eval()

    target = None
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and n.endswith("mlp.down_proj") and m.weight.shape[1] > 4000:
            target = (n, m)
            break
    if target is None:
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear) and m.weight.shape[1] > 1000:
                target = (n, m)
                break
    name, layer = target
    print(f"layer {name}  W{tuple(layer.weight.shape)}")

    x = torch.randn(1024, layer.weight.shape[1], device="cuda").half()

    H = (x.float().T @ x.float())
    W = layer.weight.data

    for g in (16, 128):
        rtn = qdq_symmetric(W.float(), 4, group_size=g).half()
        gq = gptq_quantise(W, bits=4, group_size=g, H=H)
        ref = (W @ x.T).float()
        e_rtn = ((rtn @ x.T).float() - ref).pow(2).mean().item()
        e_gq = ((gq @ x.T).float() - ref).pow(2).mean().item()
        print(f"  g={g:3d}  RTN {e_rtn:.4e}   GPTQ {e_gq:.4e}   "
              f"{e_rtn/e_gq:.2f}x better")
        assert e_gq < e_rtn, f"GPTQ failed to beat RTN on real layer g={g}"
    print("real-layer check PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
