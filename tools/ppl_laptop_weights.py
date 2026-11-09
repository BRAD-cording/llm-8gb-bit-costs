"""Weight-bit curve on Device A, llama.cpp path, same files as B and C.

Same protocol as Device C: 512-token chunks, 20 chunks, cache held at 16 bits.
This makes the three-device table apples to apples: one set of gguf files,
one corpus, one command line.
"""
import json, os, re, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = r'C:\Users\HP\c2\bin\llama-perplexity.exe'
MDIR = r'C:\Users\HP\c2\models'
CORPUS = os.path.join(ROOT, 'data', 'calib_arxiv.txt')
URL = 'https://modelscope.cn/models/unsloth/Qwen3-1.7B-GGUF/resolve/master/'

FILES = {
    'Q8_0': 'Qwen3-1.7B-Q8_0.gguf',
    'Q4_K_M': 'Qwen3-1.7B-Q4_K_M.gguf',
    'Q2_K': 'Qwen3-1.7B-Q2_K.gguf',
}

rows = {}
for name, fn in FILES.items():
    path = os.path.join(MDIR, fn)
    if not os.path.exists(path):
        print(name, 'downloading', flush=True)
        subprocess.run(['curl', '-sL', '-o', path, URL + fn], check=False, timeout=1800)
    print(name, 'size', os.path.getsize(path) if os.path.exists(path) else 'MISSING', flush=True)
    if not os.path.exists(path):
        rows[name] = {'ppl': None, 'reason': 'download failed'}
        continue
    cp = subprocess.run([BIN, '-m', path, '-f', CORPUS, '-c', '512', '--chunks', '20',
                         '-t', '8', '-ngl', '99'],
                        capture_output=True, text=True, timeout=3600)
    m = re.findall(r'Final estimate:\s*PPL\s*=\s*([\d.]+)', cp.stdout + cp.stderr)
    ppl = float(m[-1]) if m else None
    print(name, 'rc', cp.returncode, 'ppl', ppl, flush=True)
    rows[name] = {'ppl': ppl, 'rc': cp.returncode}

base = rows.get('Q8_0', {}).get('ppl')
for k, v in rows.items():
    if v.get('ppl') and base:
        v['rel_pct'] = round(100 * (v['ppl'] / base - 1), 2)

out = {'device': 'A', 'model': 'Qwen3-1.7B', 'runtime': 'llama.cpp', 'kv': 'f16', 'rows': rows}
with open(os.path.join(ROOT, 'results', 'ppl_laptop_weights.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1)
print('ROWSTABLE', json.dumps(rows))
