#!/usr/bin/env bash
# Clear the leftover llama-server that is holding 3.5 GB, record what it was,
# then configure and start the b12063 CUDA build in the background.
set +e
echo "== leftover server"
ps -eo pid,rss,etime,args | grep -E "llama-(server|cli)" | grep -v grep
ls -la "$HOME/llama-deploy"/*.gguf 2>/dev/null
echo "== stop it (frees the memory budget we will report)"
pkill -f "llama-server" && sleep 3
pgrep -af llama-server || echo "no llama-server running"
free -m | head -2
echo "== configure"
export PATH=/usr/local/cuda/bin:$PATH
cd "$HOME/c2/llama.cpp" || exit 1
rm -rf build
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DLLAMA_CURL=OFF \
      -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -12
echo "== start build in background"
nohup bash -c 'cmake --build build -j4 --target llama-bench llama-cli llama-perplexity > '"$HOME"'/c2/build.log 2>&1; echo EXIT=$? >> '"$HOME"'/c2/build.log' >/dev/null 2>&1 &
sleep 5
echo "started; log:"; tail -3 "$HOME/c2/build.log" 2>/dev/null
