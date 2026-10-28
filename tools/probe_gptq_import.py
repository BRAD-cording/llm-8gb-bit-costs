"""Iteratively find what else gptqmodel 4.2.5 needs from transformers 5.x."""
import subprocess
import sys

CODE = """
import tools.gptq_compat
try:
    from gptqmodel import GPTQModel, QuantizeConfig
    import gptqmodel
    print('OK', gptqmodel.__version__)
except Exception as e:
    print(type(e).__name__ + ': ' + str(e)[:300])
"""

for i in range(6):
    r = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True,
                       cwd=r"C:\Users\HP\OneDrive\Desktop\Dev\harness\paper1")
    out = (r.stdout + r.stderr).strip()
    print(f"--- attempt {i+1} ---")
    print(out[-600:])
    if "OK " in out:
        break
