"""Diagnostic: what does the cal16 checkpoint actually score on the 0.6B windows?"""
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)
import tools.gptq_compat  # noqa: F401,E402

import json  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def load_windows(tok, path, seq_len, n):
    text = open(path, encoding="utf-8", errors="ignore").read()
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n = min(len(ids) // seq_len, n)
    return ids[: n * seq_len].view(n, seq_len)


@torch.inference_mode()
def nll(model, win):
    return [float(model(win[i:i + 1].to("cuda"),
                        labels=win[i:i + 1].to("cuda")).loss)
            for i in range(win.shape[0])]


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "C:/models/Qwen3-0.6B-gptq-cal16"
    tok = AutoTokenizer.from_pretrained("C:/models/Qwen3-0.6B", trust_remote_code=True)
    win = load_windows(tok, "data/calib_arxiv.txt", 1024, 16)
    ev = win[:14]

    from gptqmodel import GPTQModel
    from gptqmodel.utils.backend import BACKEND
    m = GPTQModel.load(model_dir, backend=BACKEND.TORCH)
    hf = getattr(m, "model", None)
    print("checkpoint:", model_dir)
    print("config hidden:", hf.config.hidden_size, "vocab:", hf.config.vocab_size)
    try:
        hf.to("cuda")
    except Exception as e:
        print("to(cuda):", type(e).__name__, str(e)[:80])
    hf.eval()
    vals = nll(hf, ev)
    print("quantised NLL:", round(sum(vals) / len(vals), 6))

    m2 = json.load(open(".tmp/gptq_win_fp16.json"))
    print("fp16 baseline file:", m2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
