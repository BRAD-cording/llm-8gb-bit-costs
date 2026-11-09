import os, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from pi5 import conn, run  # type: ignore

bench = '/home/rpi5/llama.cpp/build/bin/llama-bench'
M17 = '/home/rpi5/models/Qwen3-1.7B-Q4_K_M.gguf'
M4 = '/home/rpi5/models/Qwen3-4B-Q4_K_M.gguf'

for tag, model, kv in [
    ('1.7B/f16', M17, 'f16'), ('1.7B/q4_0', M17, 'q4_0'),
    ('4B/f16', M4, 'f16'), ('4B/q4_0', M4, 'q4_0'),
]:
    for L in (32768, 49152, 65536, 98304, 131072):
        cmd = (f'timeout 900 {bench} -m {model} -p {L} -n 8 -r 1 -t 4 '
               f'-ctk {kv} -ctv {kv} 2>&1 | tail -4')
        c = conn()
        out, _ = run(c, cmd, timeout=1000)
        c.close()
        ok = 'failed' not in out and 'error' not in out.lower()
        pp = re.findall(r'\|\s*pp\s*\|\s*([\d.]+)', out)
        tg = re.findall(r'\|\s*tg\s*\|\s*([\d.]+)', out)
        print('%-10s L=%-6d %s pp=%s tg=%s' % (tag, L, 'OK' if ok else 'FAIL',
                                             pp[-1] if pp else '-', tg[-1] if tg else '-'))
        if not ok:
            print('   ', out.strip().splitlines()[-1][:160])
            break
