import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from pi5 import conn, run  # type: ignore

SCRIPT = r'''#!/bin/bash
cd $HOME/models
BASE="https://modelscope.cn/api/v1/models"
get () { [ -f "$2" ] || curl -sL --max-time 3600 -o "$2" "$1"; ls -l "$2"; }
get "$BASE/unsloth/Qwen3-1.7B-GGUF/repo?Revision=master&FilePath=Qwen3-1.7B-Q8_0.gguf" Qwen3-1.7B-Q8_0.gguf
get "$BASE/unsloth/Qwen3-4B-Instruct-2507-GGUF/repo?Revision=master&FilePath=Qwen3-4B-Instruct-2507-Q4_K_M.gguf" Qwen3-4B-Q4_K_M.gguf
echo DL_DONE
'''
c = conn()
sftp = c.open_sftp()
with sftp.file('/home/rpi5/dl_models.sh', 'w') as f:
    f.write(SCRIPT)
sftp.close()
out, err = run(c, 'chmod +x $HOME/dl_models.sh; nohup setsid $HOME/dl_models.sh > $HOME/models/dl.log 2>&1 < /dev/null & echo STARTED $!', timeout=30)
print(out.strip())
c.close()
