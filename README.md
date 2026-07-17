# AIDaS

A desktop application for processing OCT images.

## Quick Start (Windows)

The easiest way to run AIDaS on Windows is to use the release installer:

1. Download `AIDaS-Setup-<version>.exe` from the GitHub Releases page
2. Run the installer, then launch AIDaS from the Start menu
3. **No Python installation required!**

## Prerequisites for Running from Source

This document lists what you need before running AIDaS as a Python application.

### 1. Operating System
- Windows 10 version 1903 or newer is recommended for DirectML GPU inference
- Linux and macOS use the ONNX Runtime CPU execution provider

### 2. Python
- Python 3.11 is recommended for the main AIDaS app
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
Install the main AIDaS app and ONNX inference runtime from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Alternatively, create the conda environment from `environment.yml`. This uses
conda only for Python itself; app packages are installed by pip from
`requirements.txt`:

```bash
conda env create -f environment.yml
conda activate aidas
```

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

## OCT Retinal Layer Segmenter

AIDaS includes an OCT retinal layer segmenter in `OCT Segmenter/AI_ForAIDAS`.
It performs automated segmentation of 6 retinal layer boundaries in OCT images,
with optional foveal centre marking, and supports Analyze 7.5 `.img`/`.hdr`
input and output files.

### Installation

The installed Step 2 segmenter uses the ONNX model and is fully covered by the
single application requirements file:

```bash
python -m pip install -r requirements.txt
```

Or create the complete Conda environment:

```bash
conda env create -f environment.yml
conda activate aidas
```

The legacy model-training and ONNX-export scripts are developer tools, not
parts of the installed application. Install their large optional dependencies
only on a machine used to retrain or export the model:

```bash
python -m pip install torch onnx
```

Then launch the segmenter:

```bash
cd "OCT Segmenter/AI_ForAIDAS"
python app.py
```

You can also double-click `OCT Segmenter.bat` if using the packaged launcher.

### File Overview

| File | Purpose |
|------|---------|
| `app.py` | One-click GUI: segment, batch segment, gather data, retrain |
| `train.py` | Train the 6-boundary U-Net model |
| `segment_img_via_tiff.py` | Core inference pipeline for `.img` files |
| `segment.py` | Inference for TIFF files |
| `segment_analyze.py` | Alternative direct `.img` inference without TIFF round-trip |
| `gather_training_data.py` | Collect annotated `.img` pairs into the training folder |
| `backup/` | Clean copies of `train.py` and `segment.py` before modifications |

### Segmenter GUI

**Run Segmentation**
- Browse to a single `.img` file
- An image viewer opens; drag the yellow vertical line to the foveal centre, then click Confirm or Skip
- Output is saved as `<name>_segmented.img` and `<name>_segmented.hdr` next to the input

**Batch Segment Folder**
- Pick a folder; the app scans all subfolders for files named exactly `Light.img` or `Dark.img`
- For each file, the image picker appears for vertical-line placement
- Output is saved as `Light_MARKED.img` / `Dark_MARKED.img` next to each original

**Gather Training Data**
- Pick a source folder; the app scans recursively for raw and MARKED `.img` pairs
- Copies and renames them into `test/gathered_data/` as `scan_001`, `scan_002`, and so on
- Safe to run multiple times; new files are appended and existing files are not overwritten

**Retrain Models**
- Retrains the boundary model (`model_img.pth`, 60 epochs)
- Run after gathering more training data
- Progress appears in the log area

### Training Data

Annotated pairs live in:

```text
test/gathered_data/
  scan_001.img / scan_001.hdr
  scan_001_MARKED.img / scan_001_MARKED.hdr
  scan_002.img / ...
```

Boundary encoding in MARKED files:

| Pixel value | Boundary |
|-------------|----------|
| 255 | ILM, top, thick |
| 254 | NFL/GCL, thick |
| 253 | IPL/INL |
| 252 | INL/OPL |
| 250 | IS/OS |
| 249 | RPE, bottom |
| 243 | Vertical reference line, foveal centre |

### Boundary Model

`train.py` trains a fully-convolutional U-Net.

- Input: `(1, H, W)` grayscale OCT slice, any resolution
- Output: `(6, H, W)` per-boundary heatmap logits
- Features: `(32, 64, 128, 256)`, roughly 1.9M parameters
- Padding: input padded to multiple of 16, output cropped back to original size
- Loss: soft cross-entropy between predicted column-wise softmax and Gaussian soft labels
- Boundary y-coordinates extracted via soft-argmax
- Optimizer: Adam, lr=1e-3, ReduceLROnPlateau
- Best checkpoint saved based on validation loss
- The installed app runs `model_img.onnx`; `model_img.pth` remains the training checkpoint

```bash
python train.py --epochs 60 --save-path model_img.pth
```

Key arguments: `--epochs`, `--lr`, `--save-path`, `--data-dir`

After training, export and verify the checkpoint from the repository root:

```bash
python tools/export_segmentation_onnx.py
```

On a Windows development computer, use `--verify-provider dml` to compare the
export against PyTorch while executing the ONNX graph through DirectML.

### Inference Pipeline

`segment_img_via_tiff.py` processes each slice of an Analyze `.img` file:

1. Load int16 slice, offset to uint16, and save as a temp TIFF
2. Load TIFF with `load_image()`, normalize to `[0, 1]`, and run U-Net
3. Convert the raw slice to uint8 background using `save_overlay()`
4. Draw 6 boundary lines using PIL ImageDraw at exact pixel values, 249 through 255
5. Draw the vertical line at the user-specified column with value 243
6. Stack all slices and write uint8 Analyze `.img` and `.hdr` files
7. Save boundary y-coordinates as CSV

```bash
python segment_img_via_tiff.py test/Light.img --model model_img.pth
```

Key arguments: `--model`, `--no-csv`

Output files are saved next to the input:

- `Light_segmented.img` and `Light_segmented.hdr`: uint8 Analyze files with boundaries overlaid
- `Light_pred_boundaries.csv`: boundary y-coordinates per column per slice

### Data Gathering

Standalone script, also accessible from the GUI:

```bash
python gather_training_data.py
```

- Opens two folder pickers: source and destination
- Finds every `name.img` that has a matching `name_MARKED.img` in the same subfolder
- Copies both files plus their `.hdr` files, numbered sequentially from where the last run left off
- Safe to run on another computer; only Python standard library and tkinter are required

### Models

| File | Description |
|------|-------------|
| `model.pth` | Original model trained on TIFF images |
| `model_img.pth` | Current boundary model trained on `.img` data |
| `model_img.onnx` | Exported runtime model bundled with AIDaS |
| `model_img_baseline.pth` | Saved baseline before further modifications |

### Segmenter Folder Structure

```text
OCT Segmenter/AI_ForAIDAS/
  app.py                    GUI application
  train.py                  Boundary model training
  segment_img_via_tiff.py   Main inference script
  segment.py                TIFF inference
  segment_analyze.py        Direct .img inference
  gather_training_data.py   Data collection utility
  model_img.pth             Trained boundary model
  model_img.onnx            Exported model used by installed AIDaS
  model_img_baseline.pth    Baseline boundary model
  model.pth                 Original TIFF-trained model
  test/
    gathered_data/          Training pairs
    Light.img / Light.hdr   Test images
  AI training set/          Original TIFF training data
  backup/                   Pre-modification copies
  README.md                 Segmenter-specific README
```

### Segmenter Tips

- More data improves accuracy; aim for 100+ unique annotated `.img` files for reliable boundary detection
- Retrain after each new batch of data using the Retrain button
- The vertical line is user-placed; drag it to the foveal pit in the image picker
- Arrow keys nudge the vertical line 1 px; Shift+arrow nudges 10 px
- Both slices per `.img` file are identical, so only slice 0 is used for training
- Backups are in `backup/` if you need to restore the original scripts

## Application Package Structure

Runtime Python code is grouped by responsibility under `aidas/`:

```text
aidas/
|-- app.py                 Application composition and main window
|-- ai/                    Runtime inference and isolated AI worker
|-- canvas/                Reusable image display and annotation canvas
|-- core/                  Preferences and single-instance lifecycle
|-- services/              Update discovery, download, installation, and update UI
|-- steps/                 The four OCT workflow screens
`-- utils/                 Shared filesystem, image, I/O, logging, and UI helpers
```

`OCT Segmenter/AI_ForAIDAS/` remains a separate developer workspace for model
training and experimentation. The installed application uses the ONNX runtime
adapter in `aidas/ai/`, while PyInstaller bundles only the exported model from
that workspace. PyTorch and the `.pth` checkpoint are not included in releases.

### Step 2 GPU inference

On Windows, `onnxruntime-directml` allows the same ONNX model to run on recent
AMD, NVIDIA, and Intel DirectX 12 GPUs without CUDA or ROCm. Step 2 automatically
selects `DmlExecutionProvider` on adapter 0. `CPUExecutionProvider` is always
configured as the fallback, and AIDaS retries on CPU if DirectML cannot initialize
or execute the graph. The selected provider and any fallback reason are written
to the Step 2 segmentation log. A batch keeps one isolated worker process and
one optimized model session alive, so the executable and model are not reloaded
for every image.

DirectML remains supported by ONNX Runtime, although new Windows execution-provider
development is moving to WinML. The provider selection is isolated in
`aidas/ai/inference.py` so WinML-registered providers can be added later without
changing the model or Step 2. See the [ONNX Runtime DirectML documentation](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html).

Startup uses a live Tkinter splash window with status messages for module loading
and construction of each workflow step. No splash screenshot is generated or
stored. The Windows release uses PyInstaller's one-directory layout so the
application does not need to unpack a large archive on every launch.

## Building the Windows Executable

To create a standalone Windows build, use a clean virtual environment so Conda
or per-user packages cannot leak into the release. The dependency list includes
PyInstaller and its required Windows compatibility package, `pywin32-ctypes`:

```powershell
python -m venv .venv-release
.\.venv-release\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-release\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-release\Scripts\python.exe -m PyInstaller AIDaS.spec --clean --noconfirm
```

The build is created at `dist/AIDaS/`, with the launch executable at
`dist/AIDaS/AIDaS.exe`. The spec requires and bundles every Python runtime
package from `requirements.txt`, along with the app assets, release `.R` scripts,
AI_ForAIDAS `.onnx` model, and ONNX Runtime DirectML provider libraries. It
explicitly excludes the development-only PyTorch package. Users receive this
complete directory through the installer and do not need to install Python
packages, copy R scripts or model/data files, install CUDA/ROCm, or create a
separate conda environment.

Step 3 still requires the R interpreter and its required R packages. The app's
**Setup R and Packages** wizard detects or installs those runtime dependencies;
the workflow `.R` files are bundled with the application automatically.

Do not use a minimal PyInstaller command for release builds; it does not include
the model/data files or the reliable one-directory layout defined in
`AIDaS.spec`.

### Build with Spec File (Recommended)

Build from `AIDaS.spec` so app name, icon, bundled assets, ONNX model, and
DirectML provider DLLs stay consistent:

```bash
python -m PyInstaller AIDaS.spec --clean
```

### Troubleshooting: Desktop Icon Does Not Update (Windows)

If `AIDaS.exe` has the new icon but the desktop shortcut still shows the old icon, Windows is using cached icon data.

1. Delete the old desktop shortcut (do not delete the EXE)
2. Run `dist/AIDaS/AIDaS.exe` once directly
3. Create a new desktop shortcut from the new EXE
4. If still unchanged, clear icon cache and restart Explorer:

```powershell
Stop-Process -Name explorer -Force
Remove-Item "$env:LOCALAPPDATA\IconCache.db" -ErrorAction SilentlyContinue
Remove-Item "$env:LOCALAPPDATA\Microsoft\Windows\Explorer\iconcache*" -ErrorAction SilentlyContinue
Start-Process explorer.exe
```

If needed, rename the EXE once (for example, `AIDaS_new.exe`) to bypass filename-based icon caching.

### Troubleshooting: `pyinstaller` not recognized (Windows PowerShell)

If you see this error:

```text
pyinstaller : The term 'pyinstaller' is not recognized as the name of a cmdlet, function, script file, or operable program.
```

PyInstaller is usually installed, but its script path is not on your `PATH`.
Use the module form instead:

```bash
python -m PyInstaller AIDaS.spec --clean
```

### Troubleshooting: PyInstaller cannot import `pywintypes` or `win32api`

This error means `pywin32-ctypes`, a PyInstaller dependency on Windows, is
missing from the Python environment that runs the build. Activate the intended
environment and reinstall the complete project requirements with the same
Python interpreter:

```powershell
conda activate aidas
python -m pip install -r requirements.txt
python -c "from win32ctypes.pywin32 import pywintypes, win32api; print('pywin32-ctypes OK')"
python -m PyInstaller AIDaS.spec --clean
```

Use `python -m pip` rather than a standalone `pip` command so installation and
the PyInstaller build use the same interpreter.

## License

Copyright © 2026 Machine Vision and Pattern Recognition Lab, Wayne State University.

AIDaS is free software licensed under the [GNU Affero General Public License,
version 3 or later](LICENSE) (`AGPL-3.0-or-later`). You may use, study, modify,
and redistribute it under the terms of that license. Distributed versions must
preserve the license notices and provide the corresponding source code as
required by the AGPL.

