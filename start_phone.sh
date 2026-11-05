#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
LAB="$HOME/gemma4-aw"
MODEL="$LAB/models/gemma-4-E4B_q4_0-it.gguf"
if [ ! -f "$MODEL" ]; then echo 'Run setup_phone.sh first.' >&2; exit 2; fi
if command -v termux-wake-lock >/dev/null 2>&1; then termux-wake-lock; fi
echo 'Starting Gemma 4 E4B on this Android phone, CPU backend, port 8080.'
echo 'Leave Termux running. Stop with Ctrl+C.'
exec "$LAB/llama.cpp/build-phone/bin/llama-server" \
  -m "$MODEL" --alias gemma4e4b \
  -c 8192 -b 128 -ub 64 -t 4 -ngl 0 \
  --parallel 1 --jinja --reasoning-budget 0 \
  --host 127.0.0.1 --port 8080
