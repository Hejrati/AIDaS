# AIDaS

A desktop application for processing OCT images.

## Quick Start (Windows)

The easiest way to run AIDaS on Windows is to use the standalone executable:

1. Download `AIDaS.exe` from the `dist/` folder
2. Double-click `AIDaS.exe` to launch the application
3. **No Python installation required!**

## Prerequisites for Running from Source

This document lists what you need before running AIDaS as a Python application.

### 1. Operating System
- Linux, macOS, or Windows
- This project is currently developed and tested on Linux

### 2. Python
- Python 3.9 or newer
- `pip` available for package installation

### 3. System Packages
AIDaS uses Tkinter for the desktop GUI.

**On Windows:**
- Tkinter is included with Python by default
- No additional system packages needed

**On Ubuntu/Debian, install:**

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

## 6. Run the App (From Source)
From the project root:

**On Windows:**
```bash
python run_aidas.py
```

**On Linux/macOS:**
```bash
python run_aidas.py
```

## Building the Windows Executable

To create a standalone Windows executable, use PyInstaller:

```bash
pip install pyinstaller
python -m PyInstaller --onefile --windowed --name AIDaS run_aidas.py
```

The executable will be created in the `dist/` folder as `AIDaS.exe`.

### Troubleshooting: `pyinstaller` not recognized (Windows PowerShell)

If you see this error:

```text
pyinstaller : The term 'pyinstaller' is not recognized as the name of a cmdlet, function, script file, or operable program.
```

PyInstaller is usually installed, but its script path is not on your `PATH`.
Use the module form instead:

```bash
python -m PyInstaller --onefile --windowed --name AIDaS run_aidas.py
```

