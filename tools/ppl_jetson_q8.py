"""Q8_0 baseline for Device B, after clearing the resident server."""
import json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from jetson import creds, connect  # noqa: E402

BIN = '/home/brad/c2/llama.cpp/build/bin/llama-perplexity'
ENV = ('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/brad/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib; '
       'export PYTHONIOENCODING=utf-8; ')

cli = connect(creds())


def run(cmd, timeout=1200):
    _, stdout, _ = cli.exec_command(cmd, timeout=timeout, get_pty=False)
    return stdout.channel.recv_exit_status(), stdout.read().decode('utf-8', 'replace')


rc, mem = run('pkill -x llama-server; sleep 40; sync; free -m | head -3; nvidia-smi '
              '--query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true')
print(mem, flush=True)

cmd = (ENV + f'{BIN} -m /home/brad/c2/models/Qwen3-1.7B-Q8_0.gguf -f /home/brad/c2/calib.txt '
       '-c 512 --chunks 20 -t 4 -ngl 99 2>&1 | tail -30')
rc, txt = run(cmd)
m = re.findall(r'Final estimate:\s*PPL\s*=\s*([\d.]+)', txt)
ppl = float(m[-1]) if m else None
print('Q8_0 rc', rc, 'ppl', ppl, flush=True)
if ppl is None:
    print(txt[-800:], flush=True)

path = 'results/ppl_jetson_weights.json'
d = json.load(open(path, encoding='utf-8'))
d['rows']['Q8_0'] = {'rc': rc, 'ppl': ppl}
base = ppl
for k, v in d['rows'].items():
    if v.get('ppl') and base:
        v['rel_pct'] = round(100 * (v['ppl'] / base - 1), 2)
json.dump(d, open(path, 'w', encoding='utf-8'), indent=1)
print('ROWS', json.dumps(d['rows']))
