"""Install gptqmodel's remaining deps in the main Python 3.13 env, then import.

Mirrors the Jetson installation: each missing module is installed without pulling
its dependencies, so torch/transformers stay exactly as the rest of the project
expects them.
"""
import re
import subprocess
import sys

MIRROR = ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
          "--trusted-host", "pypi.tuna.tsinghua.edu.cn"]
CANDIDATES = ["logbar", "device_smi", "defuser", "random_word", "tokenicer",
              "addict", "threadpoolctl"]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    r = run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
             "--no-build-isolation", *MIRROR, *CANDIDATES])
    print("batch install rc:", r.returncode)
    if r.returncode:
        print(r.stderr.strip().splitlines()[-3:])

    for attempt in range(1, 9):
        r = run([sys.executable, "-c",
                 "from gptqmodel import GPTQModel, QuantizeConfig; "
                 "import gptqmodel; print('OK', gptqmodel.__version__)"])
        out = (r.stdout + r.stderr).strip().splitlines()
        tail = out[-1] if out else ""
        print(f"attempt {attempt}: {tail[:120]}")
        if "OK" in tail:
            return 0
        m = re.search(r"No module named '([^']+)'", tail)
        if not m:
            print("\n".join(out[-8:]))
            return 1
        mod = m.group(1).split(".")[0]
        print(f"  installing {mod} ...")
        r2 = run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                  "--no-build-isolation", *MIRROR, mod])
        print("   rc", r2.returncode)
    return 1


if __name__ == "__main__":
    sys.exit(main())
