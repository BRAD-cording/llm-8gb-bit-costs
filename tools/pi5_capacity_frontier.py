"""Capacity frontier on the Raspberry Pi 5 (Device C).

Cheap probe: context allocation happens in create_context, so a one-token
generation at -c L is enough to test whether L is reachable. No prefill
throughput is measured here.

Writes results/pi5_capacity_frontier.json
"""
import json
import os
import re
import sys
import time

import paramiko

HOST = os.environ.get("PI_HOST", "150.158.17.46")
PORT = int(os.environ.get("PI_PORT", "17004"))
USER = os.environ.get("PI_USER", "rpi5")
PASS = os.environ["PI_PASS"]

BIN = "/home/rpi5/llama.cpp/build/bin/llama-cli"
MODELS = {
    "1.7B_Q4_K_M": "/home/rpi5/models/Qwen3-1.7B-Q4_K_M.gguf",
    "4B_Q4_K_M": "/home/rpi5/models/Qwen3-4B-Q4_K_M.gguf",
}
ARMS = [("1.7B_Q4_K_M", "f16"), ("1.7B_Q4_K_M", "q4_0"),
        ("4B_Q4_K_M", "f16"), ("4B_Q4_K_M", "q4_0")]
LADDER = [32768, 49152, 65536, 98304, 131072]

FAIL = re.compile(r"failed to create_context|error:|failed to allocate|"
                  r"out of memory|GGML_ASSERT", re.I)


def conn(n=6):
    last = None
    for i in range(n):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, port=PORT, username=USER, password=PASS,
                      timeout=25, banner_timeout=25, auth_timeout=25)
            return c
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 + 2 * i)
    raise SystemExit(f"connect failed: {last}")


def run(cli, cmd, timeout=900):
    _, out, err = cli.exec_command(cmd, timeout=timeout)
    o = out.read().decode("utf-8", "ignore")
    e = err.read().decode("utf-8", "ignore")
    return o + e


def main():
    cli = conn()
    rows = []
    for tag, kv in ARMS:
        model = MODELS[tag]
        reached = 0
        for L in LADDER:
            run(cli, "pkill -x llama-server; sleep 5; sync", 60)
            cmd = (f"{BIN} -m {model} -c {L} -n 1 -p hi -st "
                   f"-ctk {kv} -ctv {kv} -t 4 -ngl 0 </dev/null 2>&1 | tail -25")
            t0 = time.time()
            out = run(cli, cmd)
            dt = round(time.time() - t0, 1)
            ok = ("error" not in out.lower()) and not FAIL.search(out)
            rows.append({"model": tag, "kv": kv, "ctx": L, "ok": ok,
                         "secs": dt, "tail": out.strip().splitlines()[-3:]})
            print(f"{tag:11s} {kv:5s} {L:7d} {'OK ' if ok else 'FAIL'} {dt:6.1f}s",
                  flush=True)
            if ok:
                reached = L
            else:
                break
        print(f"== {tag} {kv}: reached {reached}", flush=True)
    cli.close()
    json.dump({"device": "Pi 5 (Device C)", "probe": "llama-cli -c L -n 1",
               "ladder": LADDER, "rows": rows},
              open("results/pi5_capacity_frontier.json", "w"), indent=1)
    print("WROTE results/pi5_capacity_frontier.json")


if __name__ == "__main__":
    main()
