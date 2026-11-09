import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from pi5 import conn, run  # type: ignore

TAR = 'C:' + chr(92) + 'Users' + chr(92) + 'HP' + chr(92) + 'lcpp.tgz'

c = conn()
run(c, 'pkill -f build_lcpp 2>/dev/null; pkill -f "lcpp.tgz" 2>/dev/null; rm -rf $HOME/llama.cpp $HOME/lcpp.tgz; true', timeout=30)
s = c.open_sftp()
t = time.time()
s.put(TAR, '/home/rpi5/lcpp.tgz')
s.close()
print('upload %.0fs size=%d' % (time.time() - t, os.path.getsize(TAR)))

SCRIPT = (
    '#!/bin/bash\nset -x\ncd $HOME\n'
    'tar xzf lcpp.tgz || exit 12\n'
    'rm -rf llama.cpp; mv llama.cpp-master llama.cpp || exit 13\n'
    'cd llama.cpp\n'
    'cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF || exit 21\n'
    'cmake --build build -j4 --target llama-bench llama-cli || exit 22\n'
    'echo BUILD_DONE\n'
)
s = c.open_sftp()
with s.file('/home/rpi5/build2.sh', 'w') as f:
    f.write(SCRIPT)
s.close()
out, _ = run(c, 'chmod +x $HOME/build2.sh; nohup setsid $HOME/build2.sh > $HOME/lcpp2.log 2>&1 < /dev/null & echo STARTED $!', timeout=30)
print(out.strip())
c.close()
