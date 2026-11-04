#!/usr/bin/env bash
# 952 MiB of cudaMalloc fails with 4.4 GiB "free" — is something capping the
# process (cgroup / ulimit / Jetson carve-out)?
set +e
echo "== cgroup v2"
cat /proc/self/cgroup
for f in memory.max memory.high memory.current memory.swap.max; do
  [ -r "/sys/fs/cgroup/$f" ] && echo "$f = $(cat /sys/fs/cgroup/$f)"
done
echo "== systemd limits"
systemctl show --property=MemoryMax,MemoryHigh,MemoryCurrent user-1001.slice 2>/dev/null
systemctl show --property=MemoryMax,MemoryHigh "user@1001.service" 2>/dev/null
echo "== limits"
ulimit -a | grep -E "virtual|max memory|data"
echo "== meminfo"
head -3 /proc/meminfo
grep -E "CmaTotal|CmaFree|SwapTotal|SwapFree" /proc/meminfo
echo "== tegrastats one shot"
timeout 3 tegrastats --interval 1000 | head -2
echo "== existing big processes"
ps -eo rss,args --sort=-rss | head -4
