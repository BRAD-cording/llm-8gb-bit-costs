#!/usr/bin/env bash
# Probe the Jetson Orin Nano Super: OS, JetPack/CUDA, power mode, toolchain.
set +e
echo "== uname"; uname -a
echo "== L4T"; cat /etc/nv_tegra_release 2>/dev/null
echo "== os"; . /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"
echo "== nvpmodel"; nvpmodel -q 2>&1 | head -5
echo "== mem"; free -m
echo "== disk"; df -h / /home 2>/dev/null | head -4
echo "== cpus"; nproc
echo "== toolchain"; for t in gcc g++ cmake ninja make git python3 pip3 nvcc; do
  printf '%-8s ' "$t"; command -v "$t" || echo MISSING
done
echo "== gcc"; gcc --version 2>/dev/null | head -1
echo "== cuda"; ls -d /usr/local/cuda* 2>/dev/null
/usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -2
echo "== tegra"; command -v tegrastats && echo TEGRASTATS_OK
echo "== apt"; apt-cache policy cmake 2>/dev/null | head -4
echo "== net"; timeout 8 curl -sS -o /dev/null -w '%{http_code} github\n' https://github.com 2>&1 | tail -1
echo "== done"
