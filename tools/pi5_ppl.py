import os, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from pi5 import conn, run  # type: ignore

ppl = '/home/rpi5/llama.cpp/build/bin/llama-perplexity'
env = 'export LD_LIBRARY_PATH=/home/rpi5/llama.cpp/build/bin:$LD_LIBRARY_PATH; '
M8 = '/home/rpi5/models/Qwen3-1.7B-Q8_0.gguf'
M4 = '/home/rpi5/models/Qwen3-1.7B-Q4_K_M.gguf'

for tag, model, kv in [
    ('Q8/f16', M8, 'f16'), ('Q8/q8_0', M8, 'q8_0'), ('Q8/q4_0', M8, 'q4_0'),
    ('Q4/f16', M4, 'f16'), ('Q4/q8_0', M4, 'q8_0'), ('Q4/q4_0', M4, 'q4_0'),
]:
    cmd = (env + f'timeout 2400 {ppl} -m {model} -f /home/rpi5/calib.txt -c 512 '
           f'--chunks 20 -t 4 -ctk {kv} -ctv {kv} 2>&1 | tail -6')
    c = conn()
    out, _ = run(c, cmd, timeout=2600)
    c.close()
    m = re.findall(r'Final estimate: PPL = ([\d.]+)', out)
    print('%-10s PPL %s' % (tag, m[-1] if m else 'NA'))
    if not m:
        print(out.strip()[-300:])
