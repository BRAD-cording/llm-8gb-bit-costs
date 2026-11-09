"""Q8_0 on Device B with CPU backend: the GPU path cannot allocate its
compute buffers alongside the resident server, and PPL is a numerical
quantity, so the backend changes speed rather than the value.
"""
import json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from jetson import creds, connect  # noqa: E402

BIN = '/home/brad/c2/llama.cpp/build/bin/llama-perplexity'
ENV = ('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/brad/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib; '
       'export PYTHONIOENCODING=utf-8; ')

cli = connect(creds())
_, stdout, _ = cli.exec_command(
    ENV + 'pkill -x llama-server; sleep 30; sync; ' +
    f'{BIN} -m /home/brad/c2/models/Qwen3-1.7B-Q8_0.gguf -f /home/brad/c2/calib.txt '
    '-c 512 --chunks 20 -t 4 -ngl 0 2>&1 | tail -25',
    timeout=2400, get_pty=False)
txt = stdout.read().decode('utf-8', 'replace')
rc = stdout.channel.recv_exit_status()
m = re.findall(r'Final estimate:\s*PPL\s*=\s*([\d.]+)', txt)
ppl = float(m[-1]) if m else None
print('Q8_0 cpu rc', rc, 'ppl', ppl)
if ppl is None:
    print(txt[-700:])

path = 'results/ppl_jetson_weights.json'
d = json.load(open(path, encoding='utf-8'))
d['rows']['Q8_0'] = {'rc': rc, 'ppl': ppl, 'backend': 'cpu'}
for k, v in d['rows'].items():
    if v.get('ppl') and ppl:
        v['rel_pct'] = round(100 * (v['ppl'] / ppl - 1), 2)
json.dump(d, open(path, 'w', encoding='utf-8'), indent=1)
print('ROWS', json.dumps(d['rows']))
