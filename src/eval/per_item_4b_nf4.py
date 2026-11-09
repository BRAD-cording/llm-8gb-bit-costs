"""Per-item quantisation curves for Qwen3-4B on an nf4 base, 8k windows.

For every Linear (lm_head excluded) we replace that one module by an RTN
quantised copy of its dequantised weight at b in {2,3,4,8} bits, keep every
other module at nf4, and record the perplexity. The output is the item table
the allocator consumes:

    {"items": [{"name","kind","n_params","err": {"2":d,"3":d,"4":d,"8":d}}, ...]}

err[b] is the relative perplexity increase of quantising only that module to
b bits, measured against the all-nf4 base.
"""
import argparse, json, math, os, sys, time, types
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes.functional as bnbF

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from eval.ppl import load_tokens, linear_layers
from bench.gqa_patch import enable_gqa_sdpa


def quant_rtn(w, b):
    """Per-output-channel symmetric RTN (one scale per row)."""
    lo, hi = -(2 ** (b - 1)), 2 ** (b - 1) - 1
    if w.numel() == 0:
        return w.clone()
    s = w.abs().amax(dim=1, keepdim=True) / hi
    s = torch.where(s == 0, torch.ones_like(s), s)
    return torch.clamp(torch.round(w / s), lo, hi) * s


def dequant_module(mod):
    """Return the module's effective weight as fp16 on cpu."""
    if hasattr(mod, 'quant_state') and mod.quant_state is not None:
        w = bnbF.dequantize_4bit(mod.weight.data, mod.quant_state)
    else:
        w = mod.weight.data
    return w.detach().to('cpu', torch.float16).clone()


def iter_targets(model):
    for name, mod in linear_layers(model):
        if 'lm_head' in name:
            continue
        yield name, mod


def build(model_path):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb, device_map='cuda',
        trust_remote_code=True)
    return m.eval()


def nll(model, windows):
    tot, n = 0.0, 0
    for w in windows:
        ids = w.unsqueeze(0).to('cuda')
        with torch.no_grad():
            logits = model(ids).logits
        sl = logits[:, :-1].contiguous().view(-1, logits.size(-1)).float()
        lab = ids[:, 1:].contiguous().view(-1)
        tot += F.cross_entropy(sl, lab, reduction='sum').item()
        n += lab.numel()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--text', required=True)
    ap.add_argument('--seq-len', type=int, default=8192)
    ap.add_argument('--windows', type=int, default=3)
    ap.add_argument('--bits', default='2,3,4,8')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--log', default='')
    ap.add_argument('--names', default='')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    BITS = [int(b) for b in args.bits.split(',')]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    wins = load_tokens(tok, args.text, args.seq_len, args.windows)
    model = build(args.model)
    enable_gqa_sdpa(model)
    base = nll(model, wins)
    base_ppl = math.exp(base)
    print(f'base nll {base:.6f} ppl {base_ppl:.4f} windows {len(wins)}', flush=True)

    items = []
    t0 = time.time()
    targets = list(iter_targets(model))
    if args.names:
        keep = {l.strip() for l in open(args.names, encoding='utf-8') if l.strip()}
        targets = [(n, m) for n, m in targets if n in keep]
    if args.limit:
        targets = targets[:args.limit]
    for i, (name, mod) in enumerate(targets):
        w16 = dequant_module(mod)
        n_par = int(w16.numel())
        bias = getattr(mod, 'bias', None)
        b16 = None if bias is None else bias.detach().clone()
        orig_forward = mod.forward
        err = {}
        for b in BITS:
            wq = (quant_rtn(w16, b) if b < 16 else w16).to('cuda')
            def fwd(self, x, _w=wq, _b=b16):
                y = F.linear(x, _w.to(x.dtype))
                return y if _b is None else y + _b.to(x.dtype)
            mod.forward = types.MethodType(fwd, mod)
            try:
                d = nll(model, wins) - base
            finally:
                mod.forward = orig_forward
            err[str(b)] = round(100 * (math.exp(d) - 1), 3)
        vals = [(b, err[str(b)]) for b in BITS if str(b) in err]
        suspect = any(vals[i + 1][1] > vals[i][1] + 0.5
                      for i in range(len(vals) - 1))
        del w16
        torch.cuda.empty_cache()
        rec = {'name': name, 'kind': 'w', 'n_params': n_par, 'err': err,
               'suspect_nonmonotone': suspect}
        items.append(rec)
        if args.log:
            with open(args.log, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(rec) + chr(10))
        if i % 10 == 0:
            print(f'[{i}] {name} {err} {time.time()-t0:.0f}s', flush=True)

    out = {'model': args.model, 'base_nll': base, 'base_ppl': base_ppl,
           'seq_len': args.seq_len, 'windows': args.windows,
           'bits': BITS, 'items': items}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('WROTE', args.out, len(items), 'items', f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
