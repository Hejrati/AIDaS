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
- Python 3.11 is recommended so the main app and OCT segmenter use the same environment
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
Install all AIDaS and OCT segmenter dependencies from the root `requirements.txt`:

```bash
pip install -r requirements.txt
```

This single file includes the main app dependencies, OCT segmenter model
dependencies, and local OCT segmenter wheels.

Alternatively, create the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate oct-segmenter-env
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

Install from the root dependency file before launching the segmenter:

```bash
pip install -r requirements.txt
```

Or use the root conda environment:

```bash
conda env create -f environment.yml
conda activate oct-segmenter-env
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
| `train_vline.py` | Train the foveal vertical-line predictor |
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
- Retrains the boundary model (`model_img.pth`, 60 epochs), then the vertical-line model (`vline_model.pth`, 200 epochs)
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

```bash
python train.py --epochs 60 --save-path model_img.pth
```

Key arguments: `--epochs`, `--lr`, `--save-path`, `--data-dir`

### Vertical Line Model

`train_vline.py` trains a lightweight CNN regression model.

- Input: `(1, H, W)` grayscale OCT slice
- Output: single scalar in `[0, 1]`, the foveal x-coordinate as a fraction of image width
- Architecture: 5 strided convolution layers, AdaptiveAvgPool, fully connected layer, Sigmoid
- Loss: MSE on normalised x-coordinate
- Only one slice is used per `.img` file because slices are identical
- Optimizer: Adam, lr=1e-3, ReduceLROnPlateau
- Best checkpoint saved based on validation MAE in pixels

```bash
python train_vline.py --epochs 200 --save-path vline_model.pth
```

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

Key arguments: `--model`, `--vline-model`, `--no-csv`

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
| `model_img_baseline.pth` | Saved baseline before further modifications |
| `vline_model.pth` | Vertical line, foveal centre predictor |

### Segmenter Folder Structure

```text
OCT Segmenter/AI_ForAIDAS/
  app.py                    GUI application
  train.py                  Boundary model training
  train_vline.py            Vertical line model training
  segment_img_via_tiff.py   Main inference script
  segment.py                TIFF inference
  segment_analyze.py        Direct .img inference
  gather_training_data.py   Data collection utility
  model_img.pth             Trained boundary model
  model_img_baseline.pth    Baseline boundary model
  vline_model.pth           Trained vline model
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

## Building the Windows Executable

To create a standalone Windows executable, use PyInstaller:

```bash
pip install pyinstaller
python -m PyInstaller --onefile --windowed --name AIDaS run_aidas.py
```

The executable will be created in the `dist/` folder as `AIDaS.exe`.

### Build with Spec File (Recommended)

If you have `AIDaS.spec`, build from it so app name, icon, and bundled files stay consistent:

```bash
python -m PyInstaller AIDaS.spec --clean
```

### Troubleshooting: Desktop Icon Does Not Update (Windows)

If `AIDaS.exe` has the new icon but the desktop shortcut still shows the old icon, Windows is using cached icon data.

1. Delete the old desktop shortcut (do not delete the EXE)
2. Run `dist/AIDaS.exe` once directly
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
python -m PyInstaller --onefile --windowed --name AIDaS run_aidas.py
```

