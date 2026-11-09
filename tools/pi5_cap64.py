import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from pi5 import conn, run  # type: ignore

cli = '/home/rpi5/llama.cpp/build/bin/llama-cli'
env = 'export LD_LIBRARY_PATH=/home/rpi5/llama.cpp/build/bin:$LD_LIBRARY_PATH; '
M4 = '/home/rpi5/models/Qwen3-4B-Q4_K_M.gguf'

def arm(ctx, kv, tag):
    cmd = (env + f'echo hi | timeout 900 {cli} -m {M4} -c {ctx} -n 2 -t 4 '
           f'-ctk {kv} -ctv {kv} > /tmp/c2.out 2>/tmp/c2.err; '
           'echo rc=$?; tail -3 /tmp/c2.err; '
           'grep -E "Prompt:|Generation:" /tmp/c2.out | tail -1')
    c = conn()
    out, _ = run(c, cmd, timeout=1000)
    c.close()
    print('=== ' + tag)
    print(out.strip()[-500:])

arm(65536, 'q4_0', '4B 64k q4_0')
arm(65536, 'f16', '4B 64k f16')
