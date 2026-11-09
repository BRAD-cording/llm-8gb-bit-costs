import os, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from pi5 import conn, run  # type: ignore

bb = '/home/rpi5/llama.cpp/build/bin/llama-batched-bench'
M17 = '/home/rpi5/models/Qwen3-1.7B-Q4_K_M.gguf'

for tag, kv in [('f16', 'f16'), ('q4_0', 'q4_0')]:
    cmd = (f'timeout 1800 {bb} -m {M17} -c 4096 -b 4 -n 32 -p 512 -t 4 '
           f'-ctk {kv} -ctv {kv} 2>&1 | tail -14')
    c = conn()
    out, _ = run(c, cmd, timeout=1900)
    c.close()
    print('===== cache', tag)
    print(out.strip())
