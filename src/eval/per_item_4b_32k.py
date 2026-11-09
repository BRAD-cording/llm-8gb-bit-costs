import io, json, os, sys, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench.gqa_patch import enable_gqa_sdpa
from eval.ppl import load_tokens, perplexity, quantised_weights, linear_layers

MODEL = r'C:\models\Qwen3-4B'
TEXT = os.path.join(ROOT, 'data', 'calib_arxiv.txt')
S, W, BITS = 32768, 3, (2, 3, 4, 8)   # 32k, 3 windows, per-item via per-layer quantisation


def layer_items(model):
    out = []
    for name, mod in linear_layers(model):
        if 'lm_head' in name:
            continue
        out.append((name, mod))
    return out


def nll_of(model, windows, kind, idx):
    """Quantise exactly one module at one width, keep the rest fp16."""
    items = layer_items(model)
    name, mod = items[idx]
    target = dict(items)
    orig = mod.weight.data.clone()
    try:
        mod.weight.data = kind_round(mod.weight.data, kind)
        tot = 0.0
        n = 0
        for w in windows:
            with torch.no_grad():
                out = model(w.unsqueeze(0).to('cuda'))
                logits = out.logits
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = w[1:].unsqueeze(0).to('cuda')
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1), reduction='sum')
            tot += loss.item()
            n += shift_labels.numel()
        return tot / n
    finally:
        mod.weight.data = orig


def kind_round(w, kind):
    b = kind
    qm = w.abs().max() if w.numel() else 1.0
    s = qm / (2 ** (b - 1) - 1)
    return torch.clamp(torch.round(w / s), -(2 ** (b - 1)), 2 ** (b - 1) - 1) * s
