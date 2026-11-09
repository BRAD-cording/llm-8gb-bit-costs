import os, time, paramiko

HOST, PORT, USER = '150.158.17.46', 17004, 'rpi5'


def conn(tries=6):
    last = None
    for _ in range(tries):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, port=PORT, username=USER,
                      password=os.environ['PI_PASS'], timeout=25,
                      banner_timeout=25, auth_timeout=25)
            return c
        except Exception as e:  # transient refusals observed
            last = e
            time.sleep(6)
    raise last


connect = conn


def run(c, cmd, timeout=180):
    i, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode('utf-8', 'replace'), e.read().decode('utf-8', 'replace')
