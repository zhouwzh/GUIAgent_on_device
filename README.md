# Gemma 4 on AndroidWorld

Run AndroidWorld's T3A agent with Gemma 4 E4B (Q4_0) served from a physical Android phone. Model inference runs on the phone's CPU through Termux and llama.cpp; GUI tasks run in a Windows-hosted Android emulator, with the evaluation runner in WSL2. T3A uses text UI trees. There is no cloud-model fallback.

## Requirements

- An ARM64 Android phone with USB debugging, sufficient model storage, and Termux.
- Windows 11 with Android Studio, ADB, and WSL2 Ubuntu 22.04 using mirrored networking.
- A Pixel 6 / API 33 Google APIs x86_64 emulator, started with console port `5554` and gRPC port `8554`.

## Files

| File | Purpose |
| --- | --- |
| `prepare_windows.ps1` | Download Termux and the pinned GGUF model. |
| `install_phone.ps1` | Install Termux and transfer the model and scripts via ADB. |
| `setup_phone.sh` | Build llama.cpp and prepare the model in Termux. |
| `start_phone.sh` | Serve the model on `127.0.0.1:8080`. |
| `setup_androidworld.sh` | Install the AndroidWorld environment in WSL2. |
| `run_gemma_androidworld.py` | Probe the model, initialize the emulator, and evaluate tasks. |
| `versions.json` | Record pinned versions and validation status. |
| `操作指南.txt` | Detailed setup and execution instructions in Chinese. |

## Usage

Follow `操作指南.txt` to complete setup. Keep the phone server and emulator running, and forward the model port from Windows:

```powershell
adb -s <PHONE_SERIAL> forward tcp:8080 tcp:8080
```

In WSL2, activate the installed environment:

```bash
source ~/gemma4-aw-host/miniconda3/etc/profile.d/conda.sh
conda activate aw-gemma
export PATH="$HOME/Android/Sdk/platform-tools:$PATH"
export ADB_SERVER_SOCKET=tcp:127.0.0.1:5037
cd ~/gemma4-aw-host/android_world

python run_gemma_androidworld.py --probe-only --output runs/probe
python run_gemma_androidworld.py --setup-only --output runs/setup
python run_gemma_androidworld.py --tasks ClockStopWatchRunning --min-successes 1 --output runs/smoke01
```

Use `--tasks all` for the full suite and `--combinations N` for multiple instances per task. Choose a fresh output directory for each evaluation; use `--resume` to continue an interrupted run.

## Results

Evaluation outputs include `results.csv`, `summary.json`, `llm_calls.jsonl`, and task checkpoints (`*.pkl.gz`). Check task success and exceptions in the summary; a healthy model endpoint alone does not establish task success.

The recorded validation covers syntax and mock checks. Phone-to-emulator end-to-end validation is marked as not run in `versions.json`.
