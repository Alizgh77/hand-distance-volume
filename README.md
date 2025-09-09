# Hand Distance Volume Control

Control your system volume with the normalized distance between **thumb tip** and **index tip** using your webcam (MediaPipe Hands + OpenCV).

## Features
- Normalized finger distance → 0..100% volume
- EMA smoothing + hysteresis/deadzone
- Live calibration (C)
- Cross-platform volume backend (Windows/macOS/Linux)

## Install
```bash
python -m venv .venv
pip install -r requirements.txt
puyhon hand_control.py
