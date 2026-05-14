
# `main.py` OCT Processing Documentation

This document explains the Python translation in [`main.py`](main.py), block by block. The script mirrors the R workflow in `RAW_OCT_PROCESSING_2023_09SEP-05_WSU.R`: it loads four Analyze volumes, extracts the fovea and RPE from marked images, samples the retina along perpendicular lines, aligns the flattened data, detects retinal layer borders, spatially normalizes the strip, and writes Analyze, text, plot, and NPZ outputs.

## Shape Notation

The code uses NumPy arrays, but it intentionally follows the R script's indexing logic. Values are written as pixels first, with physical size in parentheses. There are two pixel grids:

1. Input pixels: the original OCT pixels, where `1 input pixel = 3.89 um`.
2. Resampled/flattened pixels: output grid locations created by the script. Along the flattened retinal-strip rows, `1 output row pixel = 1 um`. Along the first flattened depth axis, `500 output column pixels = 500 um`.

In this document:

| Symbol | Meaning | Typical value in this script |
|---|---:|---:|
| `Y` | image rows after loading into R layout | input dependent input pixels (`Y * 3.89 um` if measured directly in the source image) |
| `X` | image columns after loading into R layout | input dependent input pixels (`X * 3.89 um` if measured directly in the source image) |
| `Zdark` | number of DARK slices | `len(IMAGE_INDEX_DARK)`, default `2` slices |
| `Zlight` | number of LIGHT slices | `len(IMAGE_INDEX_LIGHT)`, default `2` slices |
| `N` | first flattened distance rows | `3201` output row pixels (`3201 um`, from `-200..3000 um`, 1 um per row) |
| `D` | first flattened perpendicular depth samples | `500` output column pixels (`500 um`, from about `-50..+450 um` relative to RPE, 1 um per column) |
| `R` | registered retinal-strip rows | `2851` output row pixels (`2851 um`, from `-100..2750 um`, 1 um per row) |
| `C` | cropped depth columns after vertex crop | `461` output column pixels (`461 um` in the flattened depth grid) |
| `P` | normalized depth columns | `90` output column pixels (`112.5%` normalized depth span, 1.25 percentage points per column) |

The most important shapes are:

| Data | Shape |
|---|---:|
| Loaded volume | `(Y input pixels, X input pixels, Z slices)` |
| Flattened raw/marker strip before registration | `(3201 px rows, 500 px columns, Z)` = `(3201 um, 500 um, Z)` |
| Registered and cropped strip, `*_RRC` | `(2851 px rows, 461 px columns, Z)` = `(2851 um, 461 um, Z)` |
| Normalized strip, `*_RRC_N` | `(2851 px rows, 90 normalized columns, Z)` = `(2851 um, 112.5% normalized depth, Z)` |
| Analyze export for flat strip | `(Z slices, 461 px columns, 2851 px rows)` = `(Z, 461 um, 2851 um)` |
| Analyze export for normalized strip | `(Z slices, 90 normalized columns, 2851 px rows)` = `(Z, 112.5% normalized depth, 2851 um)` |

Example: if the Analyze reader returns a raw stack shaped `(2 slices, 500 input pixels, 1024 input pixels)`, `load_analyze_volume_r_layout()` converts it to internal shape `(1024 input pixels, 500 input pixels, 2 slices)`.

## Input Size Requirements

These limits are for the current `main.py` as written, with the hard-coded R-script distances and default image indices.

| Requirement | Minimum for the code to run reliably | Maximum |
|---|---|---|
| Files | All four Analyze files must exist: `DARK_MARKED`, `LIGHT_MARKED`, `DARK`, `LIGHT` | no file-count maximum; this script uses these four names |
| Slice count | At least `2` DARK slices and `2` LIGHT slices with the default `IMAGE_INDEX_* = [1, 2]` | no hard maximum, but only `len(IMAGE_INDEX_*)` slices are processed |
| Shared input shape | All four loaded volumes should have the same `(Y input pixels, X input pixels, Z slices)` layout after loading | no hard maximum; limited by RAM and runtime |
| RPE distance coverage | The marked RPE must produce exactly `3201` output row pixels (`3201 um`) after filtering `-200..3000 um` | no useful maximum; extra RPE coverage outside the filter is ignored |
| Source-image distance along the RPE | About `823` input pixels minimum along the RPE curve (`823 * 3.89 = 3201 um`) | no hard maximum |
| Fovea placement along RPE | Fovea must be at least about `52` input pixels (`200 um`) from the negative/left RPE limit and about `772` input pixels (`3000 um`) from the positive/right RPE limit | no hard maximum |
| Perpendicular source depth around RPE | At least about `129` input pixels are needed along the perpendicular line (`500 um`); practically, the RPE needs about `13` input pixels (`50 um`) outward and `116` input pixels (`450 um`) inward inside the image | no hard maximum |
| Flattened depth columns | The script always creates `500` output column pixels (`500 um`) and later crops to `461` output column pixels (`461 um`) | fixed by code |
| Registered strip rows | The script always crops to `2851` output row pixels (`2851 um`, `-100..2750 um`) | fixed by code |
| Normalized strip columns | The script always creates `90` output column pixels (`112.5%` normalized depth, 1.25 percentage points each) | fixed by code |
| Required markers | Each processed marked slice needs RPE `255` and fovea `243`; the first DARK marked slice also needs layer markers `249`, `250`, `252`, `253`, `254` across enough rows for the moving windows | no marker-count maximum |

In code terms, the RPE-distance coordinate is built from `marked_slice.shape[0]`, so a nearly horizontal RPE needs the internal `Y` dimension to be at least about `823` input pixels (`3201 um`). The perpendicular source-depth requirement mostly depends on the internal `X` dimension, but a sloped or curved RPE consumes margin in both `Y` and `X`.

The most important practical minimum is not just image width or height; it is marker coverage. The input image may have enough pixels, but the code can still fail if the RPE marker does not cover the full `3201` output row pixels (`3201 um`) from `-200..3000 um`, or if the layer markers are missing from the moving windows used later.

If the fovea is exactly centered in the image and the marked RPE also spans the image, the minimum lateral image length is controlled by the larger side of the required distance range:

```text
left side needed:   200 um / 3.89 = about 52 input pixels
right side needed: 3000 um / 3.89 = about 772 input pixels
```

When the fovea is centered:

```text
left available = right available = image_length / 2
```

Therefore the image needs at least:

```text
minimum lateral length = 772 * 2 = about 1544 input pixels
1544 input pixels * 3.89 um/input pixel = about 6006 um
```

Practical centered-fovea examples:

| Lateral image length | Distance from centered fovea to each edge | Result |
|---:|---:|---|
| `1024 input pixels` (`3983 um`) | `512 input pixels` (`1992 um`) | Not enough, because the positive side needs `3000 um` |
| `1544 input pixels` (`6006 um`) | `772 input pixels` (`3003 um`) | Bare minimum if RPE marker spans the whole image |
| `1600 input pixels` (`6224 um`) | `800 input pixels` (`3112 um`) | Safer practical minimum |
| `2133 input pixels` (`8297 um`) | `1066.5 input pixels` (`4148.7 um`) | Enough for the current code if the fovea is near center |

This centered-fovea rule assumes the marked RPE reaches both lateral edges. If the RPE marker starts later or ends earlier than the image, use the RPE marker length and the fovea position along that RPE, not the full image length.

The height/depth requirement is different from the lateral fovea requirement. The code builds a `500 um` line perpendicular to the RPE at each RPE distance location:

```text
500 um / 3.89 um per input pixel = about 128.5 input pixels
```

So the absolute minimum source-image height/depth is about:

```text
129 input pixels = about 502 um
```

However, this `500 um` line is not centered on the RPE. It is split approximately like this:

```text
50 um outward from RPE  = about 13 input pixels
450 um inward from RPE  = about 116 input pixels
```

That means the RPE position inside the image matters:

| Height/depth situation | Minimum height/depth | Explanation |
|---|---:|---|
| RPE placed optimally near the outward edge | about `129 input pixels` (`502 um`) | Enough only if the RPE has about `13 px` outward margin and `116 px` inward margin |
| RPE exactly centered vertically | about `232 input pixels` (`902 um`) | Centered RPE gives equal margins, so both sides must be at least the larger required side: `116 px * 2` |
| Safer practical centered-RPE height | `240+ input pixels` (`934+ um`) | Gives a little room for RPE slope, curvature, rounding, and `floor()` sampling |

For example, an image with height/depth `128 input pixels` is approximately:

```text
128 * 3.89 = 497.92 um
```

That is right at the `500 um` sampling requirement. The code can still run because out-of-bounds sampling returns `NaN`, but for a fully valid perpendicular sample at every RPE location, a little more height/depth is safer.

There is no explicit maximum input size in `main.py`. Larger images and more slices mainly increase memory use and runtime. The working arrays are `float64`, so one `(3201, 500, 1)` flattened slice uses about `12.8 MB`, and the script keeps several DARK and LIGHT copies in memory during registration and normalization.

## Configuration Block

The top-level constants define the exact files and R settings being translated:

```python
REFERENCE_DARK = "DARK_MARKED"
REFERENCE_LIGHT = "LIGHT_MARKED"
TO_PROCESS_DARK = "DARK"
TO_PROCESS_LIGHT = "LIGHT"
PIXEL_WIDTH = 3.89
IMAGE_INDEX_LIGHT = [1, 2]
IMAGE_INDEX_DARK = [1, 2]
DFonINITIALspline = 10
DFforSECONDfit = 10
```

Input files are expected as Analyze pairs:

```text
DARK_MARKED.hdr / DARK_MARKED.img
LIGHT_MARKED.hdr / LIGHT_MARKED.img
DARK.hdr / DARK.img
LIGHT.hdr / LIGHT.img
```

Marker values follow the R comments:

| Marker value | Meaning |
|---:|---|
| `255` | RPE |
| `254` | ELM / OLM marker used by the script as the OLM-position layer |
| `253` | ONL/OPL border |
| `252` | INL/IPL border |
| `250` | RNFL/GCL border |
| `249` | RNFL/vitreous border |
| `243` | fovea center line |

`PIXEL_WIDTH = 3.89` means one pixel is treated as 3.89 um. A 500 um perpendicular sampling length becomes:

```python
pixel_move = round(500 / 3.89, 1)  # 128.5 pixels
```

## Orientation, Rotation, Padding, and Resizing Summary

`main.py` does not use a literal image resize operation on the input image and does not call `np.rot90()`. Instead, it uses these transformations:

| Operation | Where | Purpose | Shape effect |
|---|---|---|---|
| `np.transpose(volume, (2, 1, 0))` | `load_analyze_volume_r_layout()` | Convert local reader layout to R-like `(Y, X, Z)` | `(Z, X, Y)` -> `(Y, X, Z)` |
| `[:, ::-1, :]` | `load_analyze_volume_r_layout()` | Reverse the in-slice x axis to match R orientation | same shape |
| 500-point perpendicular resampling | flattening loops | Convert curved OCT geometry into a rectangular strip | `(Y input px, X input px)` -> `(3201 output px rows, 500 output px columns)` per slice (`3201 um, 500 um`) |
| Row shifting | `shift_rows_to_border()` | Vertically align the retinal strip to a border | same shape |
| Lateral crop | RRC block | Keep `2851` output row pixels (`-100..2750 um`) | `(3201 px, 500 px, Z)` -> `(2851 px, 500 px, Z)` = `(2851 um, 500 um, Z)` |
| Vertex crop | vertex block | Keep RPE-centered depth columns | `(2851 px, 500 px, Z)` -> `(2851 px, 461 px, Z)` = `(2851 um, 461 um, Z)` |
| Depth normalization | `build_main_normalized_strip()` | Map variable retinal thickness into 90 standard depth bins | `(2851 px, 461 px, Z)` -> `(2851 px, 90 bins, Z)` = `(2851 um, 112.5% depth, Z)` |
| `[:, ::-1, :]` | normalized builders and exports | Reverse depth axis to match R output orientation | same shape |
| `np.vstack()` with NaN pads | final sections | Re-align border-position matrices to full strip coordinates | adds rows |
| `np.transpose(..., (2, 1, 0))` | Analyze export | Write data in expected Analyze stack order | `(rows, cols, Z)` -> `(Z, cols, rows)` |

So when people say "resizing" in this pipeline, the closest operations are geometric resampling to 500 perpendicular samples and spatial normalization to 90 depth columns.

## Pipeline Blocks

### 1. CLI and Output Directories

`main()` parses:

```text
--input-dir
--output-dir
--more-outputs
--flat-npz
--done-npz
```

Normal mode loads Analyze files and runs the full translation. `--more-outputs` skips the full pipeline and uses saved `.npz` files to generate the follow-on tissue-border plots and thickness tables.

Example:

```powershell
python main.py --input-dir C:\Users\behzad\Desktop\flat --output-dir C:\Users\behzad\Desktop\flat\output
```

Output directory contents include:

```text
python_plots/
_flat_DARK.hdr / _flat_DARK.img
_flat_LIGHT.hdr / _flat_LIGHT.img
_flat-normed_DARK.hdr / _flat-normed_DARK.img
_flat-normed_LIGHT.hdr / _flat-normed_LIGHT.img
DARK__and__LIGHT__flat.npz
_done_DARK__and__LIGHT.npz
_dark_profiles_DARK.txt
_light_profiles_LIGHT.txt
_fovea_dark_profiles_DARK.txt
_fovea_light_profiles_LIGHT.txt
```

### 2. Load Analyze Volumes

Function: `load_input_volumes(input_dir)`

This loads the four required volumes:

```python
{
    "REF_DARK": DARK_MARKED,
    "REF_LIGHT": LIGHT_MARKED,
    "DARK": DARK,
    "LIGHT": LIGHT,
}
```

Each file is passed through `load_analyze_volume_r_layout(path)`.

Input shape example:

```text
read_analyze("DARK.hdr") -> (2 slices, 500 input pixels, 1024 input pixels)
```

Output shape example:

```text
load_analyze_volume_r_layout("DARK.hdr") -> (1024 input pixels, 500 input pixels, 2 slices)
```

The transformation is:

```python
volume = np.transpose(volume, (2, 1, 0))[:, ::-1, :]
```

This is the main orientation correction in `main.py`. It is equivalent to reordering axes into R's effective `(row, column, slice)` layout and flipping the in-slice x direction.

If a future reader returns a 4D array, the code keeps the first channel:

```python
volume = volume[:, :, :, 0]
```

### 3. Build Coordinate Grids

The first `REF_DARK` slice is used to build 1-based coordinate grids `xs` and `ys`, matching R matrix indexing.

Input:

```text
r = REF_DARK[:, :, 0]    # shape (Y, X)
```

Outputs:

```text
xs shape: (Y, X), each column contains 1..Y
ys shape: (Y, X), each row contains 1..X
```

Example with a small 3x4 image:

```text
xs =
[[1, 1, 1, 1],
 [2, 2, 2, 2],
 [3, 3, 3, 3]]

ys =
[[1, 2, 3, 4],
 [1, 2, 3, 4],
 [1, 2, 3, 4]]
```

These grids let the code multiply marker masks by coordinates to recover the marked point locations.

### 4. Find Fovea Center Marker

The fovea line is marked as value `243`.

Process:

```python
r[r < 243] = np.nan
r[r > 243] = np.nan
r[r == 243] = 1.0
xcoords = xs * r
ycoords = ys * r
fovea_line = np.column_stack((xcoords[mask], ycoords[mask]))
```

Input:

```text
marked slice: (Y, X)
xs, ys:       (Y, X)
```

Output:

```text
fovea_line: (F, 2)
```

`F` is the number of pixels marked with value `243`. The two columns are x and y positions in the R-style 1-based coordinate system.

Special case:

```python
if all fovea x values are identical:
    fovea_line[0, 0] += 0.1
```

That avoids a perfectly vertical line causing a degenerate linear fit.

### 5. Find RPE Marker

The RPE line is marked as value `255`.

Process:

```python
r[r < 255] = np.nan
r[r > 255] = np.nan
r[r == 255] = 1.0
rpe_line = np.column_stack((xcoords[mask], ycoords[mask]))
rpe_line = rpe_line sorted by x
```

Input:

```text
marked slice: (Y, X)
```

Output:

```text
rpe_line: (RPE_pixels, 2)
```

Example:

```text
rpe_line row = [145, 431]
```

This means the RPE marker exists at x position 145 and y/depth position 431 in the R-style coordinate grid.

### 6. Smooth the RPE and Build Distance Coordinates

Function: `fit_smooth_spline_like_r(x, y, df=DFonINITIALspline)`

This approximates R's:

```r
smooth.spline(RPE.line[,1], RPE.line[,2], df=DFonINITIALspline)
```

The spline is sampled every `0.02` pixels along x:

```python
pred_x = np.arange(0.0, marked_slice.shape[0] + 0.02, 0.02)
pred_y = rpe_sp(pred_x)
pred_dy = rpe_sp.derivative()(pred_x)
```

Input:

```text
rpe_line: (RPE_pixels, 2)
```

Intermediate output:

```text
rpe_spline: about (Y / 0.02 + 1, 3)
columns: [x_pix, y_pix, dy_dx]
```

Then the code computes distance along the RPE curve:

```python
rpe_spline_compare = np.vstack((rpe_spline[1:], rpe_spline[-1:]))
step_distance = sqrt(dx^2 + dy^2)
cumulative_distance = cumsum(step_distance)
```

Final `rpe_spline` shape:

```text
about (Y / 0.02 + 1, 5)
columns: [cumulative_distance, step_distance, x_pix, y_pix, dy_dx]
```

Detailed meaning of these columns:

| Column | Unit | Meaning |
|---|---|---|
| `cumulative_distance` | input pixels (`input pixels * 3.89 um`) | Running distance along the curved RPE spline from the left/start side. This is arc length, not just horizontal x distance. |
| `step_distance` | input pixels (`input pixels * 3.89 um`) | Distance from the current spline point to the next spline point. Because x is sampled every `0.02` input pixels (`0.0778 um`), this is usually very close to `0.02` input pixels (`0.0778 um`). |
| `x_pix` | input pixels (`input pixels * 3.89 um`) | Source-image x coordinate of the smoothed RPE point. |
| `y_pix` | input pixels (`input pixels * 3.89 um`) | Source-image y/depth coordinate of the smoothed RPE point. |
| `dy_dx` | pixel/pixel | Local slope of the smoothed RPE curve. This is unitless because both axes are pixels. |

Plain-language summary:

1. Block 6 starts from `RPE.line`, which is the rough RPE marker made from pixels with value `255`.
2. It fits a smooth curve through those marker pixels so the RPE is represented as one continuous line instead of many rough segmented marker pixels.
3. It samples that smooth curve every `0.02` input pixels (`0.0778 um`) along x.
4. It calculates the small distance from each sampled point to the next sampled point using `sqrt(dx^2 + dy^2)`.
5. It adds those small distances together to create `cumulative_distance`, which is the distance traveled along the curved RPE.
6. Later, Block 7 finds the fovea center on this cumulative-distance coordinate. Then the code converts distance from that fovea center into micrometers and keeps `-200..3000 um` as `RPE.info.2`.

Current real debug example from `debug_pipeline_all_blocks.py`:

```text
RPE.line input shape:       10696x2
Raw RPE.spline shape:       106651x3
Final RPE.spline shape:     106651x5
Spline sample spacing:      0.020000 input pixels (0.077800 um)
Step distance mean:         0.020013099 input pixels (0.077850956 um)
Final cumulative distance:  2134.417050 input pixels (8302.882324 um)
RPE.info.2 shape:           3201x4, distances -200..3000 um
```

Real `RPE.line` marker examples from the current input:

| Location | Row index | x pixel (`um`) | y pixel (`um`) |
|---|---:|---:|---:|
| first | 0 | `1.000000` (`3.890000 um`) | `41.000000` (`159.490000 um`) |
| 25% | 2674 | `533.000000` (`2073.370000 um`) | `19.000000` (`73.910000 um`) |
| 50% | 5348 | `1067.000000` (`4150.630000 um`) | `12.000000` (`46.680000 um`) |
| 75% | 8022 | `1600.000000` (`6224.000000 um`) | `33.000000` (`128.370000 um`) |
| last | 10695 | `2133.000000` (`8297.370000 um`) | `53.000000` (`206.170000 um`) |

Real raw smoothed spline examples before distance columns are added:

| Location | Row index | x pixel (`um`) | y pixel (`um`) | `dy_dx` |
|---|---:|---:|---:|---:|
| first | 0 | `0.000000` (`0.000000 um`) | `43.635653` (`169.742690 um`) | `-0.137850460` |
| 25% | 26662 | `533.240000` (`2074.303600 um`) | `19.899183` (`77.407821 um`) | `-0.036220256` |
| 50% | 53325 | `1066.500000` (`4148.685000 um`) | `14.173940` (`55.136628 um`) | `0.020212259` |
| 75% | 79988 | `1599.760000` (`6223.066400 um`) | `33.459921` (`130.159093 um`) | `0.032465602` |
| last | 106650 | `2133.000000` (`8297.370000 um`) | `52.469493` (`204.106329 um`) | `0.104419397` |

Real final `RPE.spline` examples after distance columns are added:

| Location | Row index | cumulative distance pixel (`um`) | step distance pixel (`um`) | x pixel (`um`) | y pixel (`um`) | `dy_dx` |
|---|---:|---:|---:|---:|---:|---:|
| first | 0 | `0.020189` (`0.078535 um`) | `0.020189` (`0.078535 um`) | `0.000000` (`0.000000 um`) | `43.635653` (`169.742690 um`) | `-0.137850460` |
| 25% | 26662 | `533.835602` (`2076.620492 um`) | `0.020013` (`0.077851 um`) | `533.240000` (`2074.303600 um`) | `19.899183` (`77.407821 um`) | `-0.036220256` |
| 50% | 53325 | `1067.212377` (`4151.456145 um`) | `0.020004` (`0.077816 um`) | `1066.500000` (`4148.685000 um`) | `14.173940` (`55.136628 um`) | `0.020212259` |
| 75% | 79988 | `1600.832293` (`6227.237619 um`) | `0.020011` (`0.077841 um`) | `1599.760000` (`6223.066400 um`) | `33.459921` (`130.159093 um`) | `0.032465602` |
| last | 106650 | `2134.417050` (`8302.882324 um`) | `0.000000` (`0.000000 um`) | `2133.000000` (`8297.370000 um`) | `52.469493` (`204.106329 um`) | `0.104419397` |

Important detail: the first `cumulative_distance` value is not zero because the code first measures the distance from the current spline point to the next point, then immediately applies `cumsum`. The last `step_distance` is zero because the last row is compared with itself.

Real `RPE.info.2` examples used by later flattening blocks:

| Distance from fovea (`um`) | Row index | x pixel (`um`) | y pixel (`um`) | perpendicular slope | stored distance |
|---:|---:|---:|---:|---:|---:|
| `-200 um` | 0 | `1158.580000` (`4506.876200 um`) | `16.457881` (`64.021157 um`) | `-34.320776622` | `-200 um` |
| `0 um` | 200 | `1209.960000` (`4706.744400 um`) | `18.070559` (`70.294475 um`) | `-29.737551416` | `0 um` |
| `500 um` | 700 | `1338.400000` (`5206.376000 um`) | `23.033367` (`89.599799 um`) | `-23.457314513` | `500 um` |
| `1000 um` | 1200 | `1466.820000` (`5705.929800 um`) | `28.561732` (`111.105136 um`) | `-24.218441905` | `1000 um` |
| `2000 um` | 2200 | `1723.740000` (`6705.348600 um`) | `37.205725` (`144.730272 um`) | `-34.046443665` | `2000 um` |
| `3000 um` | 3200 | `1980.660000` (`7704.767400 um`) | `45.806769` (`178.188331 um`) | `-27.485874828` | `3000 um` |

So Block 6 turns the rough RPE marker into a smooth RPE road map, measures distance along that road map, and prepares the exact RPE positions that later blocks use to flatten the retina.

Real sampled rows and plots are generated in:

```text
debug_outputs/pipeline_blocks_1_to_30/block_06_real_debug_examples.md
debug_outputs/pipeline_blocks_1_to_30/block_06_rpe_spline_y_and_derivative.png
debug_outputs/pipeline_blocks_1_to_30/block_06_rpe_step_and_cumulative_distance.png
debug_outputs/pipeline_blocks_1_to_30/block_06_rpe_info_2_overlay_on_dark.png
```

### 7. Find the Fovea Center Along the RPE

The code fits a straight line through `fovea_line`:

```python
fovea_curve = fit_line_coefficients(fovea_line)
```

Output:

```text
fovea_curve: (2,)
columns: [intercept, slope]
```

Then it finds the RPE spline point closest to that fovea line. That point becomes `CENTER.value`, the zero-distance reference along the RPE.

### 8. Build `RPE.info.2`

`rpe_info` keeps the useful RPE fields and converts distance to microns:

```python
rpe_info = rpe_spline[:, [2, 3, 4, 0]]
rpe_info[:, 3] = round((distance - CENTER.value) * PIXEL_WIDTH)
```

Only rows between about `-200` and `3000` um are kept:

```python
-200.9 < dist_um < 3000.9
```

Then duplicate distances are collapsed into one row per micron:

```text
rpe_info_2: (N, 4)
columns: [x_pix, y_pix, perpendicular_slope_pix, dist.on.spline.microns]
```

Typical size:

```text
N ~= 3201 rows, one for each integer micron from -200 to 3000
```

The slope is converted from RPE tangent slope to perpendicular slope:

```python
rpe_info_2[:, 2] = -1 / rpe_info_2[:, 2]
```

### 9. Apparent Angle Measurements

The script computes two linear angle estimates:

| Angle | Distance range | Stored in |
|---|---:|---|
| main retinal strip angle | `500..2750` um in the original R script; helper uses `0..2750` in one path | `APPARENT.ANGLES.FOR.*[:, 3]` equivalent |
| fovea angle | `-100..100` um | `APPARENT.ANGLES.FOR.*[:, 2]` equivalent |

The calculation is:

```python
slope = fit_line_coefficients(points)[1]
angle_degrees = arctan(slope) * 180 / pi
```

Input:

```text
RPE points in range: (M, 2)
```

Output:

```text
single float angle in degrees
```

The per-modality angle arrays have shape:

```text
apparent_angles_for_dark:  (Zdark, 3)
apparent_angles_for_light: (Zlight, 3)
columns: [image_index, fovea_angle_deg, main_angle_deg]
```

### 10. Build Perpendicular Sampling Lines

The RPE point, perpendicular slope, and 500 um move length define a sampling line through the retina.

The code first calculates unit movement along the perpendicular:

```python
deltas[:, 0] = cos(arctan(perpendicular_slope))
deltas[:, 1] = sin(arctan(perpendicular_slope))
deltas *= round(500 / PIXEL_WIDTH, 1)
```

It then determines the side pointing into the eye and defines:

```text
ADD: 450 um inward from RPE
SUB:  50 um outward from RPE
```

Output:

```text
retina_points: (N, 8)
columns:
  1 dist.on.spline.microns
  2 x_pix
  3 y_pix
  4 perpendicular_slope_pix
  5 end.x
  6 end.y
  7 start.x
  8 start.y
```

Example row:

```text
[500, 220.4, 431.2, -0.18, 242.5, 315.0, 218.0, 444.1]
```

This means distance 500 um from the fovea, with a perpendicular line sampled from `(end.x, end.y)` to `(start.x, start.y)`.

### 11. Sample Along Perpendiculars

Function: `build_floor_sample_line(start_x, end_x, start_y, end_y)`

This mirrors R:

```r
seq(start, end, by=(end-start)/500)
floor(...)
```

Input:

```text
four scalar coordinates
```

Output:

```text
line: (501, 2)
columns: [x, y]
```

The code samples 501 coordinates, then stores `values[1:]`, producing exactly 500 depth samples:

```text
flattened row: (500,)
```

Function: `get_recon_value(unwrapped_recon, upper_x, upper_y, point)`

The source image is flattened in Fortran order:

```python
unwrapped_recon = np.ravel(image_2d, order="F")
idx = int((col - 1) * upper_y + row - 1)
```

That matches the R vectorization:

```r
as.vector(image)
unwrapped.recon[((x - 1) * UpperY) + y]
```

Out-of-bounds samples become `np.nan`.

Real debug image:

```text
debug_outputs/pipeline_blocks_1_to_30/block_11_sample_along_perpendiculars_details.png
```

This image shows:

1. Several straight source sampling lines at selected RPE distances (`-200`, `0`, `500`, `1000`, `2000`, and `3000 um`).
2. The corresponding flattened marker strip, where each source line becomes one row location in the flattened output.
3. The real `0 um` sampled marker profile.
4. The exact indexing rule used to read the source image.

Important: Block 10 defines the geometric lines. Block 11 reads pixels along those lines. The line is geometrically straight; `floor()` only converts the continuous coordinates into discrete source-pixel indices before sampling.

### 12. Build `FLATTENED.MARKERS`

For the first marked dark slice:

```python
flattened_markers = np.full((retina_points.shape[0], 500), np.nan)
```

Input:

```text
REF_DARK[:, :, 0]: (Y, X)
retina_points:     (N, 8)
```

Output:

```text
FLATTENED.MARKERS: (N, 500)
```

Each row is one distance location along the RPE. Each column is one depth sample along the perpendicular line.

### 13. Flatten DARK and LIGHT Volumes

For each DARK and LIGHT slice:

1. Recompute fovea and RPE from that slice's marked reference.
2. Build slice-specific `Retina.Points`.
3. Sample the corresponding raw image volume along those perpendiculars.

Outputs:

```text
FLATTENED.DARK.RETINA:  (N, 500, Zdark)
FLATTENED.LIGHT.RETINA: (N, 500, Zlight)
```

Example with two slices:

```text
flattened_dark_retina.shape = (3201, 500, 2)
flattened_light_retina.shape = (3201, 500, 2)
```

Overlay plots are written for visual checking:

```text
python_plots/python_dark_rpe_info_2_overlay_slice_1.png
python_plots/python_light_rpe_info_2_overlay_slice_1.png
```

### 14. Convert Log-Scale OCT Values Back to Raw Scale

The R script converts log-transformed values back to linear reflectivity. The Python equivalent:

```python
flattened[np.isnan(flattened)] = -32768.0
flattened = flattened + 32768.0
flattened[flattened < 0] = 0.0
flattened_raw = np.power(2.0, flattened / 5000.0)
```

Input:

```text
flattened_dark_retina: (N, 500, Zdark)
```

Output:

```text
flattened_dark_retina_raw: (N, 500, Zdark)
```

The shape does not change. Only the intensity scale changes.

### 15. First Grand Mean

The first grand mean averages all DARK and LIGHT raw flattened slices.

Input:

```text
flattened_dark_retina_raw:  (N, 500, Zdark)
flattened_light_retina_raw: (N, 500, Zlight)
```

Output:

```text
FIRST.GRAND.MEAN: (N, 500)
```

Example:

```text
If Zdark=2 and Zlight=2, each pixel is average of 4 flattened images.
```

### 16. Rough Vitreous Position and Vertical Shift

The marker value `249` gives a rough vitreous/retina border in the flattened marker strip.

Output:

```text
ROUGH.VIT.RETINA.POSITION: (N, 2)
columns: [dist_um, rough_border_column]
```

The code then estimates vertical/depth shifts for DARK and LIGHT by comparing a local profile to the first grand mean.

Parameters:

```text
window_width_in_pixels = 400
start_move = 201
movement candidates = -10..10 columns
step between tested rows = 50
```

Outputs:

```text
SHIFT.POSITION.DARK:        (N, Zdark + 1)
SHIFT.POSITION.LIGHT:       (N, Zlight + 1)
SHIFT.POSITION.*.REFINED:   same shape
```

Column 1 is distance in microns. Remaining columns are the target border positions for each slice.

The sparse shift measurements are smoothed with `fit_smooth_spline_like_r(..., df=DFforSECONDfit)`, rounded, and extended at the left edge:

```python
shift_position_refined[:, 1:] = round(...)
shift_position_refined[:199, 1:] = shift_position_refined[199, 1:]
```

### 17. Shift Rows to a Common Border

Function: `shift_rows_to_border(data, borders, fill_value)`

This is the main padding/translation helper. It shifts each flattened row so the chosen border lands at column `450`.

Input examples:

```text
data:    (N, 500)       marker strip
data:    (N, 500, Z)    image stack
borders: (N,)
```

Output:

```text
same shape as data
```

Logic:

```text
border < 450: shift left, fill missing tail with fill_value
border = 450: keep row unchanged
border > 450: shift right, fill missing head with fill_value
```

Example:

```text
Input row:  [a, b, c, d, e]
border target requires shifting right by 2
Output row: [0, 0, a, b, c]       # simplified 5-column example
```

Actual output arrays:

```text
FLATTENED.DARK.RETINA.RAW.REFINED:  (N, 500, Zdark)
FLATTENED.LIGHT.RETINA.RAW.REFINED: (N, 500, Zlight)
FLATTENED.MARKERS.REFINED:          (N, 500)
```

### 18. Second Grand Mean and Lateral Registration

The second grand mean averages the vertically refined stacks:

```text
SECOND.GRAND.MEAN: (N, 500)
```

Then the code searches lateral row offsets from `-39..39` to maximize correlation with the grand mean.

Outputs:

```text
BEST.LAT.MOVE.DARK:  (Zdark, 2)
BEST.LAT.MOVE.LIGHT: (Zlight, 2)
columns: [slice_number, best_row_shift]
```

### 19. Crop to Registered Retinal Coordinates (`RRC`)

After lateral registration, the script crops to the main retinal distance range:

```python
(100 - crop_shift - 1):(2950 - crop_shift)
```

This produces:

```text
FLATTENED.DARK.RETINA.RRC:  (2851, 500, Zdark)
FLATTENED.LIGHT.RETINA.RRC: (2851, 500, Zlight)
FLATTENED.MARKERS.RRC:      (2851, 500)
```

Rows correspond approximately to:

```text
-100, -99, ..., 2750 um from the fovea
```

The final grand mean is:

```text
FINAL.GRAND.MEAN: (2851, 500)
```

### 20. Vertex Detection and Depth Crop

The code averages `FINAL.GRAND.MEAN` over rows to make a depth profile:

```text
GRAND.PROFILE before local crop: (500, 2)
columns: [column_index, mean_intensity]
```

It focuses on columns `434..466`, fits a spline, and uses the derivative to find `vertex`.

Then it crops depth columns around the vertex:

```python
vertex_start = int(vertex) - 431
vertex_stop = int(vertex) + 30
```

Because Python stop indices are exclusive, this keeps:

```text
C = 461 columns
```

Outputs:

```text
FLATTENED.DARK.RETINA.RRC:  (2851, 461, Zdark)
FLATTENED.LIGHT.RETINA.RRC: (2851, 461, Zlight)
FLATTENED.MARKERS.RRC:      (2851, 461)
```

Analyze flat exports are prepared as:

```python
export = np.transpose(np.nan_to_num(rrc[:, ::-1, :], nan=0.0), (2, 1, 0)).astype(np.float32)
```

Shape:

```text
(2851, 461, Z) -> (Z, 461, 2851)
```

The `[:, ::-1, :]` reverses the depth axis before writing, matching the R export orientation.

### 21. Identify Retinal Layer Borders

The code first reads manual layer markers from `FLATTENED.MARKERS.RRC`.

Output:

```text
HAND.BORDERS: (2851, 6)
columns:
  1 vitreous/RNFL marker from 249
  2 RNFL/GCL marker from 250
  3 INL/IPL marker from 252
  4 ONL/OPL marker from 253
  5 OLM/ELM marker from 254
  6 RPE fixed at 431
```

For DARK and LIGHT, the code computes refined borders using a moving row window:

```text
window_width_in_pixels = 40
start_move = 21
end_move = rows - 22
```

Outputs:

```text
TRUE.BORDERS.DARK:  (2851, 6, Zdark)
TRUE.BORDERS.LIGHT: (2851, 6, Zlight)
```

The per-border logic uses:

| Border | Detection idea |
|---|---|
| vitreous/RNFL | half-height crossing around marker 249 |
| RNFL/GCL | log profile, peak-to-half-height crossing |
| INL/IPL | half-height crossing around marker 252 |
| ONL/OPL | half-height crossing around marker 253 |
| OLM/ELM | local peak around marker 254, refined by a small spline |
| RPE | fixed at column 431 |

### 22. Border Refinement

Function: `refine_border_position_pass(...)`

This runs repeated cleaning passes on border-position matrices. It compares each layer to a reference layer, fits smooth splines, marks outlier rows using z-score thresholds, and replaces outliers with spline predictions.

Input:

```text
position_mats: dict of matrices, each (main_rows, Z)
order_names: ordered list of layer names
target_name: layer to update
z_threshold: usually 2.0, OLM uses 3.0
```

Output:

```text
updated position_mats[target_name]: same shape
plot paths: one PNG per slice
```

For each modality, the script refines:

```text
INL/IPL
ONL/OPL
vitreous/retina
OLM
RNFL/GCL
```

Before the refinement passes, the first 599 rows are removed:

```python
position = position[599:, :]
```

This leaves the main analysis strip:

```text
2252 rows for x = 499..2750 um
```

### 23. Smooth Final Layer Positions

Function: `smooth_position_matrix(position_matrix, x_values, df, output_dir, plot_prefix)`

Each layer-position matrix is smoothed over:

```text
x_values = 499..2750 um
df = 11
```

Input:

```text
position_matrix: (2252, Z)
x_values:        (2252,)
```

Output:

```text
revised position matrix: (2252, Z)
plot paths: one PNG per slice
```

RPE is handled separately by `fill_na_with_leading_non_na()`.

### 24. Pad Position Matrices Back to Full Strip Coordinates

After smoothing only the main strip, the code pads the layer-position matrices back toward full RRC row coordinates.

DARK:

```python
pad_dark = np.full((599, Zdark), np.nan)
position_dark = np.vstack((pad_dark, position_dark))
```

Shape:

```text
(2252, Zdark) -> (2851, Zdark)
```

LIGHT:

```python
pad_light = np.full((600, Zlight), np.nan)
position_light = np.vstack((pad_light, position_light))
```

Shape:

```text
(2252, Zlight) -> (2852, Zlight)
```

For normalized LIGHT construction, the code slices the matrices back to `flattened_light_retina_rrc.shape[0]`, so the normalized image still has `2851` rows.

This padding is intentional translation behavior from the R script's indexing sections.

### 25. Main Thickness Outputs

The main output matrices start as `(Z, 4)`:

```text
columns before final angle insertion:
  image_index
  whole retinal thickness = RPE - vitreous
  RPE to OLM distance = RPE - OLM
  RNFL thickness = RNFL/GCL - vitreous
```

After angle insertion:

```text
MAIN.DARK.OUTPUTS:  (Zdark, 5)
MAIN.LIGHT.OUTPUTS: (Zlight, 5)
columns:
  INDEX
  apparent.angle.retinal.strip..degrees
  whole.retinal.thick
  RPE.to.OLM.distance
  RNFL.thickness
```

Fovea outputs:

```text
MAIN.DARK.OUTPUTS.fovea:  (Zdark, 3)
MAIN.LIGHT.OUTPUTS.fovea: (Zlight, 3)
columns:
  INDEX
  apparent.angle.fovea..degrees
  RPE.to.OLM.distance
```

### 26. Build Main 90-Column Normalized Strip

Function: `build_main_normalized_strip(...)`

Input:

```text
harvest_stack: (2851, 461, Z)
layer positions: each about (2851, Z)
row_start = 601
```

Output:

```text
normalized strip: (2851, 90, Z)
```

Before the final depth-axis reversal, the 90 columns are filled like this:

| Columns | Samples |
|---:|---|
| `1..6` | RPE + 24, +20, +16, +12, +8, +4 um |
| `7..24` | 18 samples from RPE to OLM |
| `25..40` | 16 samples from OLM to ONL/OPL |
| `41..56` | 16 samples from ONL/OPL to INL/IPL |
| `57..78` | 22 samples from INL/IPL to RNFL/GCL |
| `79..86` | 8 samples from RNFL/GCL to vitreous |
| `87..90` | vitreous + 0, -4, -8, -12 um |

Then:

```python
normalized[np.isnan(normalized)] = 0.0
return normalized[:, ::-1, :]
```

The reversal makes the output orientation match the R output. The depth profile labels are:

```text
-3.75, -2.5, ..., 107.5
```

with step size `1.25`.

### 27. Build Fovea-Specific Normalized Strip

Function: `build_fovea_normalized_strip(...)`

The fovea block focuses on rows `51..151`, corresponding to about `-50..50 um` around the fovea.

Inputs:

```text
harvest_stack: (2851, 461, Z)
rpe_fovea:     (about 2851, Z)
olm_fovea:     (about 2851, Z)
row_start = 51
row_end = 151
```

Output:

```text
fovea normalized strip: (2851, 90, Z)
```

The OLM fovea border is smoothed over:

```text
x = -80..80 um
df = 3
```

Then only `-50..50 um` is inserted into the full normalized image:

```python
flattened_*_retina_rrc_n[49:152, :, :] = flattened_*_retina_rrc_n_fovea[49:152, :, :]
```

Fovea profile matrices are first `90 x (Z + 1)`, then cropped to rows `57..90`:

```text
FLATTENED.*.RRC.N.fovea.profiles: (34, Z + 1)
```

For final text export, a `56 x Z` NaN buffer is added above the fovea profile so it aligns to the 90-depth-column table.

### 28. Build Profile Matrices

Function: `build_profile_matrix(volume, image_indices, row_slice=None)`

Input:

```text
volume: (rows, 90, Z)
```

Output:

```text
profiles: (90, Z + 1)
column 1: percent/depth labels -3.75..107.5
columns 2..Z+1: mean intensity profile per image
```

Example:

```text
If volume shape is (2851, 90, 2), profile shape is (90, 3).
```

With `row_slice=slice(50, 151)`, only fovea rows are averaged.

### 29. Final Analyze Exports

Flat strip export:

```python
dark_export = np.transpose(np.nan_to_num(flattened_dark_retina_rrc[:, ::-1, :], nan=0.0), (2, 1, 0)).astype(np.float32)
write_analyze("_flat_DARK", dark_export)
```

Shape:

```text
(2851, 461, Zdark) -> (Zdark, 461, 2851)
```

Normalized strip export:

```python
dark_norm_export = np.transpose(np.nan_to_num(flattened_dark_retina_rrc_n[:, ::-1, :], nan=0.0), (2, 1, 0)).astype(np.float32)
write_analyze("_flat-normed_DARK", dark_norm_export)
```

Shape:

```text
(2851, 90, Zdark) -> (Zdark, 90, 2851)
```

The LIGHT exports follow the same shape pattern.

### 30. Final Text and NPZ Exports

Text exports:

```text
_dark_profiles_DARK.txt
_light_profiles_LIGHT.txt
_fovea_dark_profiles_DARK.txt
_fovea_light_profiles_LIGHT.txt
```

`write_object_table()` writes the transpose of the mixed object table, matching R:

```r
write(t(EXPORT), ...)
```

Compressed NumPy exports:

```text
DARK__and__LIGHT__flat.npz
_done_DARK__and__LIGHT.npz
```

The first NPZ stores the flattened registered strip and registration metadata. The second NPZ stores final metrics, normalized strips, profiles, and layer positions.

## Unified Debug Workflow

`debug_pipeline_all_blocks.py` is the single debug runner for this project. It is meant to make the Python pipeline easier to verify block by block without jumping between several temporary scripts. It follows the same major logic as `main.py`, saves representative intermediate arrays as plots/tables, and writes one combined report for Blocks 1-30.

Run it from the repository root:

```powershell
python debug_pipeline_all_blocks.py
```

Or explicitly pass the input and output folders:

```powershell
python debug_pipeline_all_blocks.py --input-dir "C:\Users\behzad\Desktop\flat" --output-dir "debug_outputs\pipeline_blocks_1_to_30"
```

The debug script expects the same four Analyze inputs as `main.py`:

```text
DARK_MARKED.hdr / DARK_MARKED.img
LIGHT_MARKED.hdr / LIGHT_MARKED.img
DARK.hdr / DARK.img
LIGHT.hdr / LIGHT.img
```

For Blocks 21-30, the debug script also looks for the final NPZ files in the input folder:

```text
DARK__and__LIGHT__flat.npz
_done_DARK__and__LIGHT.npz
```

If those NPZ files are missing, the script still debugs earlier blocks, but it reports that final layer/normalization blocks were skipped.

The main debug outputs are:

| Debug output | Meaning |
|---|---|
| `pipeline_blocks_1_to_30_debug_report.md` | One text report with real shapes, real values, and saved-plot names for each block group. Start here. |
| `pipeline_blocks_1_to_30_file_index.md` | File list generated by the debug runner, with each saved plot/table and size. |
| `block_*.png` | Visual checkpoints for images, marker overlays, flattened strips, registration, crops, layers, and normalized output. |
| `block_06_real_debug_examples.md` | Detailed numeric example file for RPE smoothing, spline distance, and `RPE.info.2`. |
| `block_10_perpendicular_sampling_lines_details.png` | One detailed visual explanation of Block 10, including formulas, real 0 um values, line endpoints, and the overlay on the DARK slice. |
| `_debug_flat-normed_DARK.hdr/.img` and `_debug_flat-normed_LIGHT.hdr/.img` | Debug Analyze exports of normalized strips. |
| `_debug_dark_profiles_DARK.txt` | Debug text-table preview for DARK profiles. |

Earlier debug versions generated `block_XX_debug_data.md` and `block_XX_debug_summary.png` files for every block. Those files are now intentionally not generated. The debug folder keeps the real plots, overlays, tables, Analyze exports, and focused detail files instead.

For Block 10 specifically, use `block_10_perpendicular_sampling_lines_details.png`. It contains the complete perpendicular-sampling explanation in one image: the overlay, the zoomed geometry, the real values, and the formulas.

The Block 10 detail image draws the continuous geometric sampling line as a straight line. The code still applies `floor()` before reading image pixels, but those floored coordinates are discrete sample locations, not the geometric line itself. If the floored coordinates are connected as a line, they can look stair-stepped; that is a plotting artifact, not the intended perpendicular geometry.

Current real debug run summary:

```text
Input volumes:            2133x128x2
fovea_line:               119x2
rpe_line:                 10696x2
RPE.spline final:         106651x5
RPE.info.2:               3201x4, -200..3000 um
Retina.Points:            3201x8
Flattened DARK/LIGHT:     3201x500x2
Registered/cropped RRC:   2851x461x2
Normalized output:        2851x90x2
Analyze normalized export: 2x90x2851
```

### Debug Outputs by Pipeline Block

Use the debug outputs in order. If an early block is wrong, later blocks can look wrong even if their code is working.

The table below lists the block-specific plots, images, tables, or exports to inspect.

| Block(s) | Debug file(s) | What to check |
|---|---|---|
| 1-2. Configuration and loading | `block_02_dark_slice_1.png`, `block_02_ref_dark_marked_slice_1.png`, report volume shapes | Confirms files load, orientation is correct, and marked/reference images match expected shape. Current data loads as `2133x128x2`. |
| 3. Coordinate grids | `block_03_xs_grid.png`, `block_03_ys_grid.png` | Confirms `Xs` ranges from `1..Y` and `Ys` ranges from `1..X` in the R-style coordinate system. |
| 4. Fovea marker | `block_04_fovea_marker_243_overlay.png` | Confirms marker value `243` is found and sits where the fovea should be. Current debug has `fovea_line = 119x2`. |
| 5. RPE marker | `block_05_rpe_marker_255_overlay.png` | Confirms marker value `255` forms a long enough RPE line. Current debug has `rpe_line = 10696x2`. |
| 6. RPE smoothing and distance | `block_06_real_debug_examples.md`, `block_06_rpe_spline_y_and_derivative.png`, `block_06_rpe_step_and_cumulative_distance.png`, `block_06_rpe_info_2_overlay_on_dark.png` | Confirms RPE spline shape, derivative, step distance, cumulative distance, and final `RPE.info.2` distance range. Expected `RPE.info.2` is `3201x4` from `-200..3000 um`. |
| 7. Fovea center on RPE | `block_07_fovea_center_on_rpe.png`, `block_07_center_distance_curve.png` | Confirms the code picked the RPE spline point closest to the fovea line. Current center is near `x=1210.080 px`, `y=18.075 px`. |
| 8-10. Angles and perpendicular sampling | `block_10_perpendicular_sampling_lines_details.png`, `block_10_sampling_line_0um_on_dark.png`, report `Retina.Points` shape | Confirms the perpendicular sample line at `0 um` crosses the retina correctly. The detail image includes the formula, real values, endpoint coordinates, zoomed geometry, and overlay. Expected sampling line shape is `501x2`; the stored flattened row uses `500` samples after dropping the first point. |
| 11-12. Marker flattening | `block_11_sample_along_perpendiculars_details.png`, `block_11_marker_sample_0um_profile.png`, `block_12_flattened_markers.png` | Confirms source pixels are sampled along perpendicular lines and placed into a rectangular `3201x500` strip. |
| 13. DARK/LIGHT flattening | `block_13_flattened_dark_slice_1_log.png` | Confirms raw OCT data follows the same curved-to-flat geometry as the markers. |
| 14. Raw conversion | `block_14_flattened_dark_slice_1_raw.png` | Confirms log/intensity values are converted back to raw scale. |
| 15. First grand mean | `block_15_first_grand_mean.png` | Confirms DARK and LIGHT slices are averaged into a common reference image. |
| 16. Vertical shift estimation | `block_16_shift_position_dark_refined.png` | Confirms the shift target varies smoothly along distance from fovea. Sudden large jumps usually mean border detection or marker sampling failed earlier. |
| 17. Row shifting | `block_17_dark_refined_slice_1.png` | Confirms flattened rows are vertically aligned to the common border. |
| 18. Lateral registration | `block_18_second_grand_mean.png`, report `BEST.LAT.MOVE.*` values | Confirms slices are laterally aligned. Current debug has lateral moves `[0.0, 0.0]` for both DARK and LIGHT. |
| 19. RRC lateral crop | `block_19_dark_rrc_slice_1_before_vertex_crop.png` | Confirms the strip is cropped to `-100..2750 um`, giving `2851` rows. |
| 20. Vertex crop | `block_20_grand_profile_full.png`, `block_20_dark_rrc_slice_1_after_vertex_crop.png` | Confirms the RPE vertex is detected and the depth crop keeps `461` columns. Current debug vertex is `444.0`. |
| 21-25. Layer positions and thickness outputs | `block_21_final_layers_dark_overlay.png` | Confirms detected/refined layers sit on the retinal strip in the expected order. |
| 26. Normalized strip | `block_26_dark_normalized_slice_1.png` | Confirms variable retinal layer thickness is mapped into `90` normalized columns. |
| 28. Profile matrices | `block_28_dark_normalized_profile_slice_1.png`, `_debug_dark_profiles_DARK.txt` | Confirms normalized-depth profiles are built with percent-depth labels. |
| 29. Analyze debug export | `_debug_flat-normed_DARK.hdr/.img`, `_debug_flat-normed_LIGHT.hdr/.img` | Confirms final normalized export shape is `(Z, 90, 2851)`. Current debug is `2x90x2851`. |
| 30. Final table preview | `block_30_main_dark_outputs_final_preview.png` | Confirms final per-slice output table has the expected columns and values. |

### How to Read the Debug Report

The report gives the fastest pass/fail checks:

```text
Blocks 1-2: loaded image shape and dtype
Blocks 4-5: fovea and RPE marker counts
Block 6: RPE spline shape and distance coverage
Block 7: selected fovea center on the RPE
Blocks 11-15: flattened strip shapes
Blocks 16-18: registration and shift shapes
Blocks 19-20: RRC crop and vertex crop shapes
Blocks 21-30: final normalized/layer output shapes
```

For the current dataset, these are the most important expected checkpoints:

| Checkpoint | Expected current value |
|---|---:|
| Loaded volume shape | `2133x128x2` |
| `RPE.info.2` | `3201x4` |
| `Retina.Points` | `3201x8` |
| `FLATTENED.DARK.RETINA` | `3201x500x2` |
| `FLATTENED.DARK.RETINA.RRC` after vertex crop | `2851x461x2` |
| `FLATTENED.DARK.RETINA.RRC.N` | `2851x90x2` |
| Final normalized Analyze export | `2x90x2851` |

### Common Debug Failure Patterns

| Symptom | Likely cause | Where to look first |
|---|---|---|
| `fovea_line` is empty or very small | Marker value `243` is missing or not loaded in the expected slice/orientation | `block_04_fovea_marker_243_overlay.png` |
| `rpe_line` is empty, fragmented, or too short | Marker value `255` does not cover enough RPE length | `block_05_rpe_marker_255_overlay.png` |
| `RPE.info.2` is not `3201x4` | RPE/fovea coverage is not enough for `-200..3000 um` | Block 6 report and `block_06_real_debug_examples.md` |
| Center point is not near the fovea | Fovea marker line or RPE spline fit is wrong | `block_07_fovea_center_on_rpe.png` |
| Flattened strip has many blank/black/NaN regions | Perpendicular sampling line goes outside the image, often because height/depth is too small or RPE is too close to an edge | `block_10_sampling_line_0um_on_dark.png`, `block_13_flattened_dark_slice_1_log.png` |
| Shift plot has sudden jumps | Rough vitreous/RPE border detection is unstable | `block_16_shift_position_dark_refined.png` |
| Second grand mean is blurry or duplicated | Lateral registration did not align slices well | `block_18_second_grand_mean.png`, `BEST.LAT.MOVE.*` values |
| Vertex crop cuts away the retina | RPE vertex detection or crop window is wrong | `block_20_grand_profile_full.png`, `block_20_dark_rrc_slice_1_after_vertex_crop.png` |
| Final layer lines are in the wrong order | Layer marker detection/refinement failed | `block_21_final_layers_dark_overlay.png` |
| Blocks 21-30 are skipped | Required NPZ files are missing from the input folder | Run `main.py` first so `DARK__and__LIGHT__flat.npz` and `_done_DARK__and__LIGHT.npz` exist |

The safest debugging order is:

1. Open `pipeline_blocks_1_to_30_debug_report.md`.
2. Check the first block where the shape or value differs from the expected value.
3. Open the PNG/table files for that block.
4. Fix the earliest failing block before interpreting later outputs.

## Function Reference

### Debug and Formatting Helpers

| Function | Input | Output | Purpose |
|---|---|---|---|
| `dbg(step, *parts)` | strings/objects | prints message | Mimics R debug messages and updates `DEBUG_STEP`. |
| `stop_at_boundary(step, *parts)` | strings/objects | raises `StopTranslationBoundary` | Lets translation stop safely at a known stage. |
| `format_number(value)` | scalar | string | Formats numbers as `%.6f`. |
| `format_dim(array)` | ndarray | string | Formats shape like `500x500x2`. |
| `show_scalar_stats(name, value)` | scalar | printed stats | Debug summary. |
| `show_vector_stats(name, values)` | list of ints | printed stats | Debug summary. |
| `show_array_stats(name, array)` | ndarray | printed stats | Debug summary with shape, min, max, mean, sum, NaN count. |

Example:

```python
format_dim(np.zeros((2851, 461, 2)))  # "2851x461x2"
```

### Analyze I/O Helpers

| Function | Input shape | Output shape | Purpose |
|---|---:|---:|---|
| `load_analyze_volume_r_layout(path)` | usually `(Z, X, Y)` from reader | `(Y, X, Z)` | Load and reorient Analyze data to match R. |
| `load_input_volumes(input_dir)` | folder path | dict of four `(Y, X, Z)` arrays | Load REF_DARK, REF_LIGHT, DARK, LIGHT. |

Example:

```python
volumes = load_input_volumes(Path("C:/data/flat"))
volumes["DARK"].shape  # e.g. (1024, 500, 2)
```

### Plot Helpers

| Function | Main input shape | Output |
|---|---:|---|
| `save_r_image_matlines_plot()` | image `(Y, X)`, line `(N, 4)` | PNG |
| `save_overlay_series_plot()` | volume `(Y, X, Z)`, list of line arrays | list of PNG paths |
| `save_shift_position_plot()` | x `(N,)`, y `(N,)`, spline `(N, 2)` | PNG |
| `save_profile_plot()` | profile `(M, 2)` | PNG |
| `save_border_positions_overview_plot()` | list of border vectors | PNG |
| `save_border_refinement_plot()` | relative positions `(M, K)`, splines `(M, K)` | PNG |
| `save_series_with_spline_line_plot()` | x/y vectors and spline `(M, 2)` | PNG |
| `save_tissue_border_plot()` | `flattened_rrc (2851, 461, Z)`, border arrays | PNG |

These functions do not change processing arrays; they only save diagnostic plots equivalent to R `plot()`, `image()`, `matlines()`, and `abline()`.

### Geometry and Sampling Helpers

| Function | Input | Output | Purpose |
|---|---|---|---|
| `get_recon_value()` | flattened image vector, bounds, point | scalar or `np.nan` | R-style matrix lookup from a vectorized image. |
| `build_floor_sample_line()` | four coordinates | `(501, 2)` | Build one floored sampling line. |
| `compute_retina_points_for_marked_slice()` | marked slice `(Y, X)`, `xs`, `ys` | `retina_points (N, 8)`, `rpe_info_2 (N, 4)`, two angles | Full RPE/fovea/perpendicular setup for one marked slice. |
| `shift_rows_to_border()` | `(N, 500)` or `(N, 500, Z)`, borders `(N,)` | same as input | Shift each row so a border aligns to column 450. |
| `slice_rows_1based()` | array, start, end | inclusive 1-based slice | Mimics R `a[start:end, ]`. |
| `first_closest_zero_crossing()` | check matrix | scalar position | Finds first requested sign or closest half-height crossing. |

Example:

```python
line = build_floor_sample_line(242.5, 218.0, 315.0, 444.1)
line.shape  # (501, 2)
```

### Spline and Statistics Helpers

| Function | Input | Output | Purpose |
|---|---|---|---|
| `correlation_estimate(x, y)` | vectors | float | Pearson correlation, ignoring NaNs. |
| `fit_line_coefficients(points)` | `(M, 2)` | `(2,)` | R-style linear model coefficients `[intercept, slope]`. |
| `fit_smooth_spline_like_r(x, y, df, degree=3)` | vectors | spline object | Approximate R `smooth.spline(df=...)`. |
| `r_style_sd(values)` | vector | float | Sample standard deviation like R `sd()`. |
| `spline_predict_from_series(values, df, split, valid_mask=None)` | vector | `(split,)` | Fit spline to a series and predict all rows. |
| `r_style_zscore(values)` | vector | vector | `(x - mean) / sd`, with R-like NaN behavior. |

Example:

```python
coeffs = fit_line_coefficients(np.array([[1, 2], [2, 4], [3, 6]]))
coeffs  # approximately [0, 2]
```

### Border and Normalization Helpers

| Function | Input shape | Output shape | Purpose |
|---|---:|---:|---|
| `refine_border_position_pass()` | layer matrices `(rows, Z)` | updates target matrix | Smooth and replace outlier border rows. |
| `smooth_position_matrix()` | `(2252, Z)` | `(2252, Z)` plus plot paths | Final spline smoothing per slice. |
| `fill_na_with_leading_non_na()` | vector | vector | Replace NaNs using available values, matching R behavior. |
| `nearest_depth_index()` | scalar, depth vector | scalar 1-based index | R `WHICH.INDEX()` equivalent. |
| `build_lookup_vector()` | position vector | 1-based lookup vector | Converts layer positions to depth-column indices. |
| `build_main_normalized_strip()` | `(2851, 461, Z)` plus layer positions | `(2851, 90, Z)` | Main retinal depth normalization. |
| `build_fovea_normalized_strip()` | `(2851, 461, Z)` plus RPE/OLM positions | `(2851, 90, Z)` | Fovea-specific normalization. |
| `build_profile_matrix()` | `(rows, 90, Z)` | `(90, Z + 1)` | Mean profile table. |

Example:

```python
normalized = build_main_normalized_strip(
    flattened_dark_retina_rrc,
    r_rpe_position_dark,
    r_olm_position_dark,
    r_onl_opl_position_dark,
    r_inl_ipl_position_dark,
    r_rnfl_gcl_position_dark,
    r_vitreous_retina_position_dark,
    row_start=601,
)
normalized.shape  # (2851, 90, Zdark)
```

### Export Helpers

| Function | Input | Output |
|---|---|---|
| `format_export_cell()` | scalar/object | string for space-delimited R-style exports |
| `format_tab_export_cell()` | scalar/object | string for tab-delimited follow-on exports |
| `write_object_table()` | mixed table | text file matching `write(t(EXPORT))` |
| `write_tab_object_table()` | mixed table | tab-delimited text file |
| `_load_npz_array()` | NPZ file and possible names | ndarray |
| `build_thickness_export()` | final layer positions | object table |
| `run_more_outputs_from_step3_npz()` | flat NPZ, done NPZ, output dir | dict of output paths |

## Follow-On `--more-outputs` Mode

`run_more_outputs_from_step3_npz()` is a Python translation of the follow-on `more_outputs_afterRAW...R` output script.

Inputs:

```text
DARK__and__LIGHT__flat.npz
_done_DARK__and__LIGHT.npz
```

Outputs:

```text
_tissueBorders__DARK.png
_tissueBorders__LIGHT.png
_thickness_vs_distance_from_fovea_DARK.txt
_thickness_vs_distance_from_fovea_LIGHT.txt
```

`build_thickness_export()` creates rows:

```text
Distance_from_Fundus_um
WholeRetina_um
RPE_to_OLM_um
OLM_to_ONL_OPLborder_um
ONL_OPLborder_to_INL_IPLborder_um
INL_IPLborder_to_RNFL_GCLborder_um
RNFL_GCLborder_to_vitreous_um
Summed_layers
```

Each row has one label column plus one value per distance point from `-100..2750 um`.

## End-to-End Shape Example

Assume:

```text
read_analyze(DARK.hdr) -> (2 slices, 500 input pixels, 1024 input pixels)
Zdark = 2
Zlight = 2
N = 3201
```

Then the main shape changes are:

```text
Raw Analyze reader:
  DARK                      (2 slices, 500 input pixels, 1024 input pixels)

After R-layout conversion:
  DARK                      (1024 input pixels, 500 input pixels, 2 slices)

After perpendicular sampling:
  FLATTENED.DARK.RETINA     (3201 px rows, 500 px columns, 2 slices)
                            (3201 um, 500 um, 2 slices)

After log-to-linear conversion:
  FLATTENED.DARK.RETINA.RAW (3201 px rows, 500 px columns, 2 slices)
                            (3201 um, 500 um, 2 slices)

After vertical refinement:
  FLATTENED.DARK.RETINA.RAW.REFINED
                            (3201 px rows, 500 px columns, 2 slices)
                            (3201 um, 500 um, 2 slices)

After lateral crop:
  FLATTENED.DARK.RETINA.RRC (2851 px rows, 500 px columns, 2 slices)
                            (2851 um, 500 um, 2 slices)

After vertex crop:
  FLATTENED.DARK.RETINA.RRC (2851 px rows, 461 px columns, 2 slices)
                            (2851 um, 461 um, 2 slices)

After normalized-depth mapping:
  FLATTENED.DARK.RETINA.RRC.N
                            (2851 px rows, 90 normalized columns, 2 slices)
                            (2851 um, 112.5% normalized depth, 2 slices)

Analyze flat export:
  _flat_DARK                (2 slices, 461 px columns, 2851 px rows)
                            (2 slices, 461 um, 2851 um)

Analyze normalized export:
  _flat-normed_DARK         (2 slices, 90 normalized columns, 2851 px rows)
                            (2 slices, 112.5% normalized depth, 2851 um)
```

LIGHT follows the same pattern.

## Important Implementation Notes

1. The code is intentionally 1-based in many calculations. Helpers like `slice_rows_1based()`, `nearest_depth_index()`, and `get_recon_value()` exist to preserve R behavior.
2. `np.ravel(..., order="F")` is used because R stores matrices column-major.
3. The output depth axis is reversed several times with `[:, ::-1, :]` to match R's saved image orientation.
4. Missing image samples become `np.nan` during processing, then are often written as `0.0` in final Analyze exports.
5. The main normalized image is not a simple resize. It is a layer-aware remapping: each retinal layer span gets a fixed number of samples.
6. DARK and LIGHT share most logic, but some padding and slicing differs because the Python code is preserving R's indexing behavior.
