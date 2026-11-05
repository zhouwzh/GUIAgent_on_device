#!/data/data/com.termux/files/usr/bin/bash
# Run in the Termux app, not in adb shell.
set -euo pipefail
KIT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
LAB="$HOME/gemma4-aw"
SRC="$LAB/llama.cpp"
REV=749db7f034d43d6de01a01f6d8c872d02afa6c7b
MODEL=gemma-4-E4B_q4_0-it.gguf
if [ ! -f "$KIT_DIR/$MODEL" ]; then
  echo "Missing model: $KIT_DIR/$MODEL" >&2
  exit 2
fi
pkg update -y
pkg upgrade -y
pkg install -y git cmake clang ninja libandroid-spawn
mkdir -p "$LAB/models"
if [ ! -d "$SRC/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$SRC"
  git -C "$SRC" checkout "$REV"
fi
ACTUAL_REV=$(git -C "$SRC" rev-parse HEAD)
if [ "$ACTUAL_REV" != "$REV" ]; then
  echo "Source revision differs from this guide. Expected: $REV; got: $ACTUAL_REV" >&2
  exit 3
fi
cmake -S "$SRC" -B "$SRC/build-phone" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DLLAMA_OPENSSL=OFF \
  -DGGML_OPENCL=OFF -DGGML_VULKAN=OFF -DGGML_HEXAGON=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF
cmake --build "$SRC/build-phone" --parallel 2 --target llama-server llama-bench
cp "$KIT_DIR/$MODEL" "$LAB/models/$MODEL"
cp "$KIT_DIR/start_phone.sh" "$LAB/start_phone.sh"
printf '%s\n' "$ACTUAL_REV" > "$LAB/llama_revision.txt"
sha256sum "$LAB/models/$MODEL" > "$LAB/model.sha256"
echo
echo 'PHONE SETUP COMPLETE'
echo "Start the phone server with: bash $LAB/start_phone.sh"
