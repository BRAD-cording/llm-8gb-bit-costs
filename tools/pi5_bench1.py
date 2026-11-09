import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from pi5 import conn, run  # type: ignore

b = '/home/rpi5/llama.cpp/build/bin/llama-bench'
env = 'export LD_LIBRARY_PATH=/home/rpi5/llama.cpp/build/bin:$LD_LIBRARY_PATH; '

def bench(model, p, n, kv, tag):
    cmd = (env + f'{b} -m {model} -p {p} -n {n} -r 2 -t 4 -ub 128 -b 1024 '
           f'-ctk {kv} -ctv {kv} 2>&1 | tail -4')
    c = conn()
    out, err = run(c, cmd, timeout=2800)
    c.close()
    print('=== ' + tag)
    print(out.strip()[-700:])

M17 = '/home/rpi5/models/Qwen3-1.7B-Q4_K_M.gguf'
bench(M17, 4096, 64, 'f16', '1.7B 4k f16')
bench(M17, 4096, 64, 'q4_0', '1.7B 4k q4_0')
