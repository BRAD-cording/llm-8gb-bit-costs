"""
Minimal SSH/SFTP helper for the Jetson. Windows' OpenSSH cannot take a password
non-interactively, so we use paramiko.

Credentials: read from environment (JETSON_HOST / JETSON_USER / JETSON_PASS) or
from tools/.jetson_creds.json (gitignored). Never hard-coded here.

Usage:
    python tools/jetson.py run "uname -a"
    python tools/jetson.py run "nvpmodel -q" --timeout 60
    python tools/jetson.py put src/bench/device_probe.py /home/brad/probe.py
    python tools/jetson.py get /home/brad/out.json results/jetson_out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CRED_FILE = Path(__file__).parent / ".jetson_creds.json"

# the Jetson reports Chinese localised group names; the Windows console is cp1252
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def creds() -> dict:
    host = os.environ.get("JETSON_HOST")
    if host:
        return {"host": host,
                "user": os.environ.get("JETSON_USER", "brad"),
                "password": os.environ.get("JETSON_PASS", "")}
    if CRED_FILE.exists():
        return json.loads(CRED_FILE.read_text(encoding="utf-8"))
    raise SystemExit("no credentials: set JETSON_HOST/USER/PASS or create "
                     f"{CRED_FILE}")


def connect(c: dict):
    import paramiko
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(c["host"], port=c.get("port", 22), username=c["user"],
                password=c.get("password"), timeout=20,
                allow_agent=False, look_for_keys=False)
    return cli


def do_run(cli, cmd: str, timeout: int) -> int:
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout, get_pty=False)
    for line in iter(stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
    rc = stdout.channel.recv_exit_status()
    err = stderr.read().decode("utf-8", "replace")
    if err.strip():
        sys.stderr.write(err)
    print(f"\n[exit {rc}]", file=sys.stderr)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("command")
    r.add_argument("--timeout", type=int, default=300)

    f = sub.add_parser("runfile", help="upload a local shell script and execute it "
                                      "(avoids all Windows->SSH quoting issues)")
    f.add_argument("script")
    f.add_argument("--timeout", type=int, default=300)

    p = sub.add_parser("put")
    p.add_argument("local")
    p.add_argument("remote")

    g = sub.add_parser("get")
    g.add_argument("remote")
    g.add_argument("local")

    args = ap.parse_args()
    cli = connect(creds())

    if args.cmd == "run":
        return do_run(cli, args.command, args.timeout)

    if args.cmd == "runfile":
        sftp = cli.open_sftp()
        remote = "/tmp/_dsh_script.sh"
        # Windows editors write CRLF; bash on the target then tries to execute
        # "$'/home/brad\r'". Normalise here so every uploaded script is LF-only.
        text = open(args.script, encoding="utf-8").read().replace("\r\n", "\n")
        with sftp.open(remote, "w") as fh:
            fh.write(text)
        sftp.close()
        return do_run(cli, f"bash {remote}", args.timeout)

    sftp = cli.open_sftp()
    if args.cmd == "put":
        sftp.put(args.local, args.remote)
        print(f"uploaded {args.local} -> {args.remote}")
    else:
        Path(args.local).parent.mkdir(parents=True, exist_ok=True)
        sftp.get(args.remote, args.local)
        print(f"downloaded {args.remote} -> {args.local}")
    sftp.close()
    cli.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
