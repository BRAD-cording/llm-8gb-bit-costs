import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from pi5 import conn, run  # type: ignore

SCRIPT = r'''#!/bin/bash
set -x
cd $HOME
if [ ! -d llama.cpp ]; then
  curl -sL --max-time 900 https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.tar.gz -o lcpp.tgz || exit 11
  tar xzf lcpp.tgz || exit 12
  mv llama.cpp-master llama.cpp || exit 13
fi
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF || exit 21
cmake --build build -j4 --target llama-bench llama-cli || exit 22
ls -l build/bin/llama-bench build/bin/llama-cli
echo BUILD_DONE
'''

c = conn()
sftp = c.open_sftp()
with sftp.file('/home/rpi5/build_lcpp.sh', 'w') as f:
    f.write(SCRIPT)
sftp.close()
out, err = run(c, 'chmod +x $HOME/build_lcpp.sh; nohup setsid $HOME/build_lcpp.sh > $HOME/lcpp.log 2>&1 < /dev/null & echo STARTED $!', timeout=30)
print(out.strip())
c.close()
