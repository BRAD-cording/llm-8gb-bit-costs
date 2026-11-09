# -*- coding: utf-8 -*-
"""Derive mean board power and energy per decoded token from tegrastats logs.

Method (stated in the paper): tegrastats prints VDD_IN (rail power in mW) once a
second. We take the samples where the GPU is actually working (GR3D_FREQ > 0 and
resident RAM within 5% of the run maximum) as the steady-decode window, average
VDD_IN there, and divide by the tokens/s that llama.cpp reports for the same
configuration. Idle power comes from samples where GR3D_FREQ == 0.
"""
import glob
import json
import os
import re

RE_TS = re.compile(r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})")
RE_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
RE_GR3D = re.compile(r"GR3D_FREQ (\d+)%")
RE_VDD = re.compile(r"VDD_IN (\d+)mW")

BENCH = {
    "L4096_f16": "bench_L4096_f16.json",
    "L16384_q8_0": "bench_L16384_q8_0.json",
    "L32768_q4_0": "bench_L32768_q4_0.json",
}


def parse(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = RE_TS.match(line)
            if not m:
                continue
            ram = RE_RAM.search(line)
            gr3d = RE_GR3D.search(line)
            vdd = RE_VDD.search(line)
            if not (ram and gr3d and vdd):
                continue
            rows.append(
                {
                    "t": m.group(1),
                    "ram": int(ram.group(1)),
                    "gr3d": int(gr3d.group(1)),
                    "mw": int(vdd.group(1)),
                }
            )
    return rows


def tok_s(name):
    p = os.path.join("results", "c2_jetson", BENCH[name])
    d = json.load(open(p))
    ns = d[1]["samples_ns"]
    gen = d[1]["n_gen"]
    per = [gen / (n / 1e9) for n in ns]
    return sum(per) / len(per), per


def analyse(name):
    rows = parse("results/c2_jetson/ram_%s.log" % name)
    if not rows:
        return {"config": name, "error": "no samples"}
    rmax = max(r["ram"] for r in rows)
    steady = [r for r in rows if r["gr3d"] > 0 and r["ram"] >= 0.95 * rmax]
    idle = [r for r in rows if r["gr3d"] == 0]
    mean = lambda xs: sum(x["mw"] for x in xs) / len(xs)
    p_steady = mean(steady) / 1000.0 if steady else None
    p_idle = mean(idle) / 1000.0 if idle else None
    ts, per = tok_s(name)
    out = {
        "config": name,
        "samples": len(rows),
        "ram_peak_MB": rmax,
        "steady_samples": len(steady),
        "idle_samples": len(idle),
        "P_steady_W": round(p_steady, 3) if p_steady else None,
        "P_idle_W": round(p_idle, 3) if p_idle else None,
        "tok_s": [round(x, 2) for x in per],
        "tok_s_mean": round(ts, 2),
        "mJ_per_token_total": round(1000.0 * p_steady / ts, 1) if p_steady else None,
        "mJ_per_token_marginal": (
            round(1000.0 * (p_steady - p_idle) / ts, 1)
            if (p_steady and p_idle)
            else None
        ),
    }
    return out


if __name__ == "__main__":
    res = [analyse(k) for k in BENCH]
    print(json.dumps(res, indent=1))
    with open(os.path.join("results", "energy_jetson_17b.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("wrote results/energy_jetson_17b.json")
