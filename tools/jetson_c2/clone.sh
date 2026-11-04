#!/usr/bin/env bash
# What is holding 5 GB, and can we get the llama.cpp source?
set +e
echo "== top memory"
ps aux --sort=-rss | head -6 | awk '{printf "%-6s %5s%% %8s KB  %s\n", $1, $4, $6, substr($0, index($0,$11), 90)}'
echo "== jetson_clocks / power"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null
echo "== clone"
mkdir -p "$HOME/c2"
cd "$HOME/c2" || exit 1
if [ -d llama.cpp/.git ]; then
  echo "already cloned"
else
  timeout 900 git clone --depth 1 --branch b12063 https://github.com/ggml-org/llama.cpp 2>&1 | tail -5
fi
ls -d llama.cpp 2>/dev/null && git -C llama.cpp describe --tags 2>/dev/null
echo "== free after"
free -m | head -2
