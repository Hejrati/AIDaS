# OCT Retinal Layer Segmenter

Automated segmentation of 6 retinal layer boundaries in OCT images, with optional foveal centre marking. Supports Analyze 7.5 `.img`/`.hdr` input/output format.

---

## Installation (first time on a new computer)

1. Create the main AIDaS environment from the repository root:

   ```
   conda env create -f environment.yml
   conda activate aidas-env
   ```

   Or, with an existing Python 3.11 environment:

   ```
   python -m pip install -r requirements.txt
   ```

2. Double-click **OCT Segmenter.bat** to launch the app.

---

## Quick Start

Launch the GUI:
```
conda activate aidas-env
cd "OCT Segmenter\AI_ForAIDAS"
python app.py
```
Or double-click **OCT Segmenter.bat**.

---

## File Overview

| File | Purpose |
|------|---------|
| `app.py` | One-click GUI — segment, batch segment, gather data, retrain |
| `train.py` | Train the 6-boundary U-Net model |
| `train_vline.py` | Train the foveal vertical-line predictor |
| `segment_img_via_tiff.py` | Core inference pipeline for `.img` files |
| `segment.py` | Inference for TIFF files |
| `segment_analyze.py` | Alternative direct `.img` inference (no TIFF round-trip) |
| `gather_training_data.py` | Collect annotated `.img` pairs into the training folder |
| `backup/` | Clean copies of `train.py` and `segment.py` before modifications |

---

## GUI (`app.py`)

### Buttons

**Run Segmentation** (blue)
- Browse to a single `.img` file
- An image viewer opens — drag the yellow vertical line to the foveal centre, then click Confirm (or Skip)
- Output saved as `<name>_segmented.img` + `.hdr` next to the input

**Batch Segment Folder** (purple)
- Pick a folder — scans all subfolders for files named exactly `Light.img` or `Dark.img`
- For each file, shows the image picker for the vertical line
- Output saved as `Light_MARKED.img` / `Dark_MARKED.img` next to each original

**Gather Training Data** (green)
- Pick a source folder — scans recursively for raw + MARKED `.img` pairs
- Copies and renames them into `test/gathered_data/` as `scan_001`, `scan_002`, etc.
- Safe to run multiple times — new files are appended, existing ones never overwritten

**Retrain Models** (orange)
- Retrains the boundary model (`model_img.pth`, 60 epochs) then the vline model (`vline_model.pth`, 200 epochs)
- Run after gathering more training data
- Progress shown in the log area

---

## Training Data

Annotated pairs live in:
```
test/gathered_data/
  scan_001.img / scan_001.hdr
  scan_001_MARKED.img / scan_001_MARKED.hdr
  scan_002.img / ...
```

### Boundary encoding in MARKED files
| Pixel value | Boundary |
|-------------|----------|
| 255 | ILM (top, thick) |
| 254 | NFL/GCL (thick) |
| 253 | IPL/INL |
| 252 | INL/OPL |
| 250 | IS/OS |
| 249 | RPE (bottom) |
| 243 | Vertical reference line (foveal centre) |

---

## Boundary Model (`train.py`)

**Architecture:** Fully-convolutional U-Net  
- Input: `(1, H, W)` grayscale OCT slice, any resolution  
- Output: `(6, H, W)` per-boundary heatmap logits  
- Features: `(32, 64, 128, 256)` — ~1.9M parameters  
- Padding: input padded to multiple of 16, output cropped back to original size  

**Training:**
- Loss: soft cross-entropy between predicted column-wise softmax and Gaussian soft labels (σ=1.5px)
- Boundary y-coordinates extracted via soft-argmax (differentiable)
- Optimizer: Adam, lr=1e-3, ReduceLROnPlateau (patience=5, factor=0.5)
- Best checkpoint saved based on validation loss

```
python train.py --epochs 60 --save-path model_img.pth
```

Key arguments: `--epochs`, `--lr`, `--save-path`, `--data-dir`

---

## Vertical Line Model (`train_vline.py`)

**Architecture:** Lightweight CNN regression  
- Input: `(1, H, W)` grayscale OCT slice  
- Output: single scalar in `[0, 1]` — foveal x-coordinate as fraction of image width  
- 5 strided conv layers → AdaptiveAvgPool → FC → Sigmoid  

**Training:**
- Loss: MSE on normalised x-coordinate  
- Only one slice used per `.img` file (slices are identical)  
- Optimizer: Adam, lr=1e-3, ReduceLROnPlateau  
- Best checkpoint saved based on validation MAE (pixels)

```
python train_vline.py --epochs 200 --save-path vline_model.pth
```

---

## Inference Pipeline (`segment_img_via_tiff.py`)

Processes each slice of an Analyze `.img` file:

1. Load int16 slice → offset to uint16 → save as temp TIFF  
2. Load TIFF with `load_image()` (min-max normalisation to [0,1]) → run U-Net  
3. Convert raw slice to uint8 background using `save_overlay()` (pixel/137 + 5.7 formula)  
4. Draw 6 boundary lines using PIL ImageDraw at exact pixel values (249–255)  
5. Draw vertical line at user-specified column (value 243)  
6. Stack all slices → write uint8 Analyze `.img` + `.hdr`  
7. Save boundary y-coordinates as CSV  

```
python segment_img_via_tiff.py test/Light.img --model model_img.pth
```

Key arguments: `--model`, `--vline-model`, `--no-csv`

Output files (saved next to input):
- `Light_segmented.img` + `.hdr` — uint8 Analyze with boundaries overlaid
- `Light_pred_boundaries.csv` — boundary y-coordinates per column per slice

---

## Data Gathering (`gather_training_data.py`)

Standalone script (also accessible via GUI button):
```
python gather_training_data.py
```
- Opens two folder pickers: source and destination
- Finds every `name.img` that has a matching `name_MARKED.img` in the same subfolder
- Copies both + `.hdr` files, numbered sequentially from where the last run left off
- Safe to run on another computer (only needs Python standard library + tkinter)

---

## Models

| File | Description |
|------|-------------|
| `model.pth` | Original model trained on TIFF images |
| `model_img.pth` | Current boundary model trained on `.img` data |
| `model_img_baseline.pth` | Saved baseline before further modifications |
| `vline_model.pth` | Vertical line (foveal centre) predictor |

---

## Folder Structure

```
OCTSegmenterHuman/
├── app.py                      GUI application
├── train.py                    Boundary model training
├── train_vline.py              Vertical line model training
├── segment_img_via_tiff.py     Main inference script
├── segment.py                  TIFF inference
├── segment_analyze.py          Direct .img inference
├── gather_training_data.py     Data collection utility
├── model_img.pth               Trained boundary model
├── model_img_baseline.pth      Baseline boundary model
├── vline_model.pth             Trained vline model
├── model.pth                   Original TIFF-trained model
├── test/
│   ├── gathered_data/          Training pairs (scan_001, scan_002 ...)
│   └── Light.img / Light.hdr   Test images
├── AI training set/            Original TIFF training data
├── backup/                     Pre-modification copies of train.py, segment.py
└── README.md                   This file
```

---

## Tips

- **More data = better accuracy.** Aim for 100+ unique annotated `.img` files for reliable boundary detection.
- **Retrain after each new batch of data.** Use the orange Retrain button in the app.
- **Vertical line is user-placed.** Drag it to the foveal pit in the image picker. Arrow keys (← →) nudge 1px; Shift+arrow nudges 10px.
- **Both slices per `.img` file are identical** — only slice 0 is used for training.
- **Backups** are in `backup/` if you need to restore the original scripts.
