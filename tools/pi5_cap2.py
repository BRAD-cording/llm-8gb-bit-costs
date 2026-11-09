import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from pi5 import conn, run  # type: ignore

cli = '/home/rpi5/llama.cpp/build/bin/llama-cli'
M17 = '/home/rpi5/models/Qwen3-1.7B-Q4_K_M.gguf'
M4 = '/home/rpi5/models/Qwen3-4B-Q4_K_M.gguf'
GREP = 'grep -iE "create_context|out of memory|KV buffer size" | head -3'

for tag, model, kv in [('1.7B/f16', M17, 'f16'), ('1.7B/q4_0', M17, 'q4_0'),
                       ('4B/f16', M4, 'f16'), ('4B/q4_0', M4, 'q4_0')]:
    for L in (32768, 49152, 65536, 98304, 131072, 196608):
        cmd = ('timeout 180 %s -m %s -c %d -n 1 -p hi -t 4 --no-warmup '
               '-ctk %s -ctv %s 2>&1 | %s' % (cli, model, L, kv, kv, GREP))
        c = conn(); out, _ = run(c, cmd, timeout=200); c.close()
        low = out.lower()
        bad = ('failed' in low) or ('out of memory' in low)
        print('%-9s L=%-7d %s %s' % (tag, L, 'FAIL' if bad else 'ok', out.strip()[:90]))
        if bad:
            break
