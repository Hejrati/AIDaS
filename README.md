# AIDaS Prerequisites

This document lists what you need before running AIDaS.

## 1. Operating System
- Linux, macOS, or Windows
- This project is currently developed and tested on Linux

## 2. Python
- Python 3.9 or newer
- `pip` available for package installation

## 3. System Packages
AIDaS uses Tkinter for the desktop GUI.

On Ubuntu/Debian, install:

```bash
sudo apt update
sudo apt install -y python3-tk
```

## 4. Python Dependencies
Install project dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Dependencies:
- `numpy`
- `Pillow`

## 5. Input Data Requirements
Step 1 currently expects:
- A directory containing `.sdb` files

Optional sidecar files (if available):
- `.hdr` files with matching names

## 6. Run the App
From the project root:

```bash
python run_aidas.py
```

