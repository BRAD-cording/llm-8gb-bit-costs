"""Q2_K on Device C (Pi 5): complete the third point of the weight curve.

Fetches the gguf from ModelScope directly on the board (the laptop cannot
reach Hugging Face), then runs llama-perplexity with the same protocol used
for Q8_0 / Q4_K_M on this device: 512-token chunks, 20 chunks, 4 threads.
"""
import json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pi5 import conn, run  # noqa: E402

URL = ('https://modelscope.cn/models/unsloth/Qwen3-1.7B-GGUF/resolve/master/'
       'Qwen3-1.7B-Q2_K.gguf')
DST = '/home/rpi5/models/Qwen3-1.7B-Q2_K.gguf'
BIN = '/home/rpi5/llama.cpp/build/bin/llama-perplexity'
CORPUS = '/home/rpi5/calib.txt'

cli = conn()
rc, out = run(cli, f'ls -l {DST} 2>/dev/null | head -2', 60)
print('existing:', out.strip() or 'none', flush=True)

if 'Q2_K' not in out:
    rc, out = run(cli, f'curl -sL -o {DST} "{URL}" && ls -l {DST}', 1800)
    print('download rc', rc, out[-300:], flush=True)

rc, out = run(cli, f'{BIN} -m {DST} -f {CORPUS} -c 512 --chunks 20 -t 4 2>&1 | tail -30', 3600)
m = re.findall(r'Final estimate:\s*PPL\s*=\s*([\d.]+)', out)
ppl = float(m[-1]) if m else None
print('Q2_K rc', rc, 'ppl', ppl, flush=True)
if ppl is None:
    print(out[-800:], flush=True)

path = 'results/pi5_weight_curve.json'
d = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}
d.setdefault('device', 'C')
d.setdefault('model', 'Qwen3-1.7B')
d.setdefault('rows', {})
d['rows']['Q2_K'] = {'ppl': ppl, 'rc': rc}
json.dump(d, open(path, 'w', encoding='utf-8'), indent=1)
print('ROWS', json.dumps(d['rows']))
