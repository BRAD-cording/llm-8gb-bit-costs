"""Upload the eval corpus to Device B once, then it can be reused."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from jetson import creds, connect  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = os.path.join(ROOT, 'data', 'calib_arxiv.txt')
print('local size', os.path.getsize(src))

cli = connect(creds())
sftp = cli.open_sftp()
sftp.put(src, '/home/brad/c2/calib.txt')
print('remote size', sftp.stat('/home/brad/c2/calib.txt').st_size)
sftp.close()
cli.close()
