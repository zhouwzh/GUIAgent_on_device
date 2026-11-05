#!/usr/bin/env bash
# Run in Ubuntu under WSL2. The Windows emulator and Windows adb server stay on Windows.
set -euo pipefail
KIT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
LAB="$HOME/gemma4-aw-host"
CONDA_DIR="$LAB/miniconda3"
AW_DIR="$LAB/android_world"
AW_REV=55ddaf5edc6f6c25673aed343907d5b9dca66f84
mkdir -p "$LAB" "$HOME/Android/Sdk"
sudo apt-get update
sudo apt-get install -y curl unzip git ffmpeg libgl1 libglib2.0-0
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  curl --fail --location --retry 3 \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o "$LAB/miniconda-installer.sh"
  bash "$LAB/miniconda-installer.sh" -b -p "$CONDA_DIR"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"
if [ ! -d "$CONDA_DIR/envs/aw-gemma" ]; then
  conda create -y -n aw-gemma --override-channels -c conda-forge python=3.11.8
fi
conda activate aw-gemma
curl --fail --location --retry 3 https://dl.google.com/android/repository/platform-tools-latest-linux.zip -o "$LAB/platform-tools.zip"
unzip -q -o "$LAB/platform-tools.zip" -d "$HOME/Android/Sdk"
if [ ! -d "$AW_DIR/.git" ]; then
  git clone https://github.com/google-research/android_world.git "$AW_DIR"
  git -C "$AW_DIR" checkout "$AW_REV"
fi
ACTUAL_REV=$(git -C "$AW_DIR" rev-parse HEAD)
if [ "$ACTUAL_REV" != "$AW_REV" ]; then
  echo "AndroidWorld source differs from the guide: $ACTUAL_REV" >&2
  exit 3
fi
cd "$AW_DIR"
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e .
python setup.py generate_protos
cp "$KIT_DIR/run_gemma_androidworld.py" "$AW_DIR/run_gemma_androidworld.py"
echo
echo 'ANDROIDWORLD SETUP COMPLETE'
echo "source $CONDA_DIR/etc/profile.d/conda.sh"
echo 'conda activate aw-gemma'
echo "export PATH=\"\$HOME/Android/Sdk/platform-tools:\$PATH\""
echo 'export ADB_SERVER_SOCKET=tcp:127.0.0.1:5037'
echo "cd $AW_DIR"
