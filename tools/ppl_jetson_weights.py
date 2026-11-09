"""Whole-model weight-bit cost on Device B (Jetson), llama.cpp path.

Cache fixed at 16 bits; only the weight file changes. Coarser than the
PyTorch per-module table but cross-stack: it checks whether "weights are the
expensive side" survives off the 4060.
"""
import json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from jetson import creds, connect  # noqa: E402

MODELS = [
    ('Q8_0', '/home/brad/c2/models/Qwen3-1.7B-Q8_0.gguf'),
    ('Q4_K_M', '/home/brad/c2/models/Qwen3-1.7B-Q4_K_M.gguf'),
    ('Q2_K', '/home/brad/c2/models/Qwen3-1.7B-Q2_K.gguf'),
]
CORPUS = '/home/brad/c2/calib.txt'
BIN = '/home/brad/c2/llama.cpp/build/bin/llama-perplexity'
ENV = ('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/brad/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib; '
       'export PYTHONIOENCODING=utf-8; ')

cli = connect(creds())


def run(cmd, timeout=900):
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout, get_pty=False)
    txt = stdout.read().decode('utf-8', 'replace')
    rc = stdout.channel.recv_exit_status()
    return rc, txt


out = {'device': 'B', 'model': 'Qwen3-1.7B', 'runtime': 'llama.cpp', 'kv': 'f16', 'rows': {}}
for name, path in MODELS:
    rc, ls = run(f'ls -1 {path}')
    if rc != 0:
        print(name, 'MISSING', flush=True)
        out['rows'][name] = {'rc': rc, 'ppl': None, 'missing': True}
        continue
    cmd = (ENV + 'pkill -x llama-server; sleep 20; sync; '
           f'{BIN} -m {path} -f {CORPUS} -c 512 --chunks 20 -t 4 2>&1 | tail -40')
    rc, txt = run(cmd)
    m = re.findall(r'Final estimate:\s*PPL\s*=\s*([\d.]+)', txt)
    ppl = float(m[-1]) if m else None
    print(name, 'rc', rc, 'ppl', ppl, flush=True)
    if ppl is None:
        print(txt[-600:], flush=True)
    out['rows'][name] = {'rc': rc, 'ppl': ppl}

base = out['rows'].get('Q8_0', {}).get('ppl')
for k, v in out['rows'].items():
    if v.get('ppl') and base:
        v['rel_pct'] = round(100 * (v['ppl'] / base - 1), 2)

with open('results/ppl_jetson_weights.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1)
print('WROTE', json.dumps(out['rows']))
