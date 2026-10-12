# 3D OCT Stitcher

Interactive tools for cleaning, aligning, and stitching OCT scans named
`vol1.npy`, `vol2.npy`, … (single-volume `.npz` inputs are also supported by
the alignment and cleanup tools). The main pipeline now produces a stitched
**3-D NPY volume** that can be opened directly in Napari; the original PLY
surface workflow remains available when a mesh is needed.

## Main NPY pipeline

From the repository root, run:

```bash
python3 conversion_code/handler.py
```

The guided pipeline:

1. Finds numbered scans in `scans/` by default and lets you select scans by
   number, range, comma-separated list, or `all`.
2. Lets you use an existing position file or create new manual placements.
3. Supports alignment of each scan to either a complete reference volume or a
   progressively assembled composite of earlier scans.
4. Converts manual offsets, global poses, or pairwise poses into the pairwise
   4×4 transform format used for stitching.
5. Resamples every volume into a common `(Z, Y, X)` coordinate system.
6. Computes a bounding box for the complete stitch, estimates memory use, and
   stops before allocating an output larger than 32 GB (float32 estimate).
7. Blends overlaps by coverage-weighted averaging and saves the stitched NPY
   volume plus the global transforms actually applied.

By default, results are written to `results/final_stitch/`:

```text
results/final_stitch/
├── manual_offsets.csv
├── manual_pairwise_transforms.npy
├── applied_transforms.npy
└── final_stitched.npy
```

Open the final result in Napari, for example:

```bash
napari results/final_stitch/final_stitched.npy
```

### Guided run: every prompt

Start with all of the scans you intend to stitch in one folder and name them
`vol<number>.npy` (for example, `vol1.npy`, `vol2.npy`, `vol3.npy`). Then run
`handler.py` and answer the prompts as follows.

| Prompt | What to enter | Effect |
| --- | --- | --- |
| **Folder containing…** | `scans` or another scan folder | Where numbered input volumes are found. |
| **Scans to use** | `all`, `1,2,3`, `1-4`, or `1,2,5-7` | Chooses the ordered scans to align. The transform file must contain one pose per selected scan. |
| **Output folder** | An output directory | Receives the offsets, transforms, and final stitched volume. |
| **Volume downsample factor** | Positive integer; normally `4` | The transform-grid scale. Keep it consistent with the positions being used; the pipeline passes it to coordinate conversion. |
| **Manual alignment view downsample factor** | Positive integer; normally `4` | Reduces only the manual-alignment display for a more responsive window. It does not reduce the final NPY volume's resolution. |
| **Do you already have a position file?** | `y` or `n` | Reuse a CSV/NPY transform file, or open interactive placement windows. |
| **Position file path** | Path to `.csv` or `.npy` | Used only when reusing positions. See [Inputs and transform options](#inputs-and-transform-options). |
| **Do you have a complete reference volume…** | `y` or `n` | With `y`, every scan is placed against one reference. With `n`, scans are placed one after another against the growing composite. |
| **Reference volume…** | Path to a 3-D reference NPY/NPZ | Used only in reference-based alignment. The default is `scans/Complete_surface_reference.npy`. |

Use the same scan set when creating and reusing transforms. The stitcher reads
all `vol*.npy` files in its input folder and requires exactly `N - 1` pairwise
transforms for its `N` input volumes; keep unrelated volumes in a different
folder for a subset stitch.

### Recommended first run

1. Put only the desired numbered NPY scans in `scans/`.
2. Run `python3 conversion_code/handler.py`.
3. Accept `scans`, `results/final_stitch`, and scale `4` unless you have a
   reason to use another transform scale.
4. Choose `n` for an existing position file.
5. Choose reference alignment if a complete reference is available. For every
   scan, align the overlay, click **Save offset**, and close the window.
6. Wait for `final_stitched.npy` and inspect it in Napari.

If the final volume is unexpectedly huge or misaligned, stop and verify the
scan order, selected folder, transform count, transform scale, and manual
offsets before trying again.

### Inputs and transform options

The pipeline expects numbered scans such as `vol1.npy` and accepts 3-D arrays
in `(Z, Y, X)` order; a `(1, Z, Y, X)` NPY is also accepted by the stitcher.
The manual-alignment loader removes singleton dimensions and accepts a
single-array NPZ archive. For NPZ archives with multiple arrays, name the
volume `volume`, `stitched`, or `arr_0`.

At the position prompt, provide one of the following:

- A CSV of manual global offsets with columns `volume`, `z_vox`, `y_vox`, and
  `x_vox`.
- An NPY array shaped `(N - 1, 4, 4)` containing adjacent pairwise transforms.
- An NPY array shaped `(N, 4, 4)` containing global poses; it is converted to
  adjacent pairwise transforms automatically.

Manual offsets are saved in transform-grid voxel coordinates. The manual view
can use a separate downsample factor for responsiveness; its slider offsets
are automatically converted to the final transform scale.

### Manual alignment

For a reference-based run, each scan opens in a three-plane Matplotlib window.
The reference is magenta and the moving scan is green; shared features should
appear yellow/white when aligned. Set the Z, Y, and X offsets, click **Save
offset**, then close the window to move to the next scan.

Without a reference, the first scan starts on a blank canvas. Each later scan
is aligned against a magenta maximum-intensity composite of the scans already
placed. Its initial position is the previous scan's saved global offset, so
placements remain in one common coordinate system.

Starting a new manual run removes prior offset rows for the selected scans but
preserves rows for other scans in the same CSV.

## Napari cleanup: how it works

Napari cleanup belongs to the optional PLY workflow below, or can be run by
itself. It **does not modify the acquired input scan**: it writes a cleaned
copy to the output path supplied with `--output`.

```bash
python3 conversion_code/napari_visualization.py \
  --input scans/vol1.npy \
  --output results/cleaned_vol1.npy
```

The viewer opens a grayscale `Scan` layer and a red `Artifact centers` points
layer. It starts in 3-D on most systems. In the 3-D view, click the centre of
each artifact to add a red X. In 2-D mode, navigate the Z/Y/X slices and add
points at the artifact centre. Use the controls in the **Cleanup** dock, then
click **Save cleaned volume**. Closing Napari without saving returns an error
and produces no cleaned output.

### Cleanup controls

| Control | Default | What it does when saved |
| --- | --- | --- |
| **Background floor** | `5%` | Values at or below that fraction of the current contrast range are made black. With visible settings enabled, those voxels are set to zero in the saved copy. Raise it to remove more weak background/noise; lower it to preserve faint signal. |
| **Contrast limits** | Estimated from the preview | Defines the visible intensity range. With visible settings enabled, values below the low limit are removed and values above the high limit are clipped. Narrowing the range increases visual contrast but can discard dim or saturated information on save. |
| **Gamma** | Napari layer default | Changes brightness within the contrast range. With visible settings enabled, the gamma correction is baked into the saved voxel intensities. Leave it near the default unless that intensity change is deliberate. |
| **Apply visible floor, contrast limits, and gamma…** | On | Makes the three display settings permanent in the cleaned copy. Turn it off to use them for viewing only and preserve original retained intensities. |
| **Keep only selected depth band** | Off | Zeros all source slices outside the selected preview-Z interval. Use it to isolate one panel/surface depth layer; keep it off when the complete depth range is needed. |
| **Keep Z from / through** | Full Z range | The inclusive preview slice bounds retained by the depth-band option. Their order is corrected automatically. |
| **Removal radius (preview voxels)** | `8`, range `1–100` | Radius of the spherical artifact selection around each red X. See the radius explanation below. |

### Radius and preview resolution

The cleanup viewer normally shows a reduced preview rather than every source
voxel. It automatically chooses a `preview step` so the preview has at most
750,000 voxels. The dock displays the selected step: `step = 1` means full
resolution; `step = 2` means one preview voxel represents roughly a 2×2×2
source-voxel block.

The radius is measured in **preview voxels**, not original source voxels. A
radius of `r` covers a spherical region of approximately `r × step` source
voxels from the clicked centre in each direction, with selected preview cells
expanded to their whole source blocks. Thus, at `step = 4`, the default radius
`8` removes an area roughly 32 source voxels from the centre in each axis.
The exact boundary is block-shaped at source resolution, so start smaller than
the artifact, save, and increase only if residual artifact remains.

Every point uses the same radius when you save. Remove accidental points from
the `Artifact centers` layer before saving; otherwise each one creates a
permanent zeroed region in the cleaned copy.

### Preview and quality tradeoffs

`--preview-max-voxels` changes **only the interactive preview**, not the
underlying source scan. A lower cap makes Napari start and render faster, but
uses a larger preview step, makes point placement less precise, and makes each
radius unit cover more source voxels. A higher cap improves placement detail
and makes the radius more literal, at the cost of GPU memory and rendering
speed.

```bash
# Faster, coarser preview
python3 conversion_code/napari_visualization.py \
  --input scans/vol1.npy --output results/cleaned_vol1.npy \
  --preview-max-voxels 250000

# Higher-detail preview for careful cleanup
python3 conversion_code/napari_visualization.py \
  --input scans/vol1.npy --output results/cleaned_vol1.npy \
  --preview-max-voxels 2000000
```

On macOS, the program uses 2-D as a safe fallback when it detects Intel/Rosetta
Python. Use native ARM64 Python for reliable 3-D rendering. `--force-3d`
overrides that fallback but may be unstable on that configuration.

### Stitching behavior

The NPY stitcher uses the same transform convention as the mesh stitcher:

```text
T₀ = I
Tᵢ = Tᵢ₋₁ @ inverse(Pᵢ₋₁)
```

where `P` is an adjacent pairwise transform. Use `--coordinate-scale Z,Y,X`
when the transforms were measured on downsampled data (the interactive
pipeline supplies this automatically). The direct stitcher also supports
nearest-neighbor, linear (default), and cubic interpolation through
`--order 0`, `--order 1`, and `--order 3`.

To run it directly:

```bash
python3 conversion_code/stitch_npy_volumes.py \
  --input-dir scans \
  --initial-transforms results/final_stitch/manual_pairwise_transforms.npy \
  --output results/final_stitch/final_stitched.npy \
  --transforms-output results/final_stitch/applied_transforms.npy \
  --coordinate-scale 4,4,4
```

## Scan cleanup and PLY workflow

The previous mesh workflow is still available:

```bash
python3 conversion_code/raw_scans_to_final_ply.py
```

It preserves acquired scans, creates selection-specific cleaned copies under
`results/final_stitch/cleaned_volumes/<selected-scans>/`, exports individual
surface meshes, and writes `final_stitched.ply` to the chosen output folder.
A one-scan run performs cleanup and exports a cleaned individual PLY without
attempting alignment or stitching.

### PLY guided-run options

| Prompt | Recommended choice | Effect |
| --- | --- | --- |
| **How many scans should be manually cleaned…** | `all` | Selects one or more scans. Enter a number to confirm individual scans. One scan exports only a cleaned PLY; two or more continue to alignment and stitching. |
| **Reuse existing cleaned scans…** | `y` only after review | Skips Napari and reuses the selection-specific cleaned copies. Choose `n` to reopen Napari and replace them. |
| **Volume downsample factor** | `4` for a quick mesh, `1` for maximum detail | Samples every `scale` voxel in Z, Y, and X before surface extraction. See quality tradeoffs below. This scale must match manual placement and PLY export for the transform coordinates to remain correct. |
| **Manual alignment view downsample factor** | `4` or higher for responsiveness | Affects only the placement window, not final PLY detail. Saved offsets are converted to the final PLY scale. |
| **Do you already have a position file?** | `n` for a first run | Reuses a `.csv` offsets file or `.npy` transforms instead of manual placement. |
| **Do you have a complete reference volume…** | `y` when available | Opens reference-based overlays; otherwise opens sequential placed-scan composites. |
| **Optional extra PLY corner-artifact region** | `none` unless known artifact coordinates exist | Removes mesh vertices in the requested bounded region after export. |
| **Refine alignment using broad X panels…** | `n` unless every scan has that broad feature | Adjusts only the X translation to align detected broad X-facing panels. It is not a general registration method. |
| **Overlap merge distance** | `1` | Welds nearby overlapping mesh vertices. Set `0` to disable welding; increase cautiously because a large value can erase fine distinct surface detail. |

### PLY quality and speed

The PLY workflow's **Volume downsample factor** is the main quality setting.
It samples every `s`th input voxel on each axis, so increasing it from `1` to
`2` reduces linear sampling resolution by two and processes roughly one eighth
as many voxels. Increasing it from `1` to `4` processes roughly one sixty-fourth
as many voxels. This is much faster and reduces mesh size, but thin structures,
small defects, and fine curvature can disappear or become blockier.

Use `1` when maximum surface detail matters and memory/runtime allow it; use
`2–4` for iteration or large scans; use a larger factor only when a coarse
surface is acceptable. The manual-alignment view scale is separate: increasing
that value makes placement faster without lowering the final PLY's detail.

The PLY export script also exposes these direct options when needed:

```bash
python3 conversion_code/export_scans_to_ply.py \
  --input-dir scans --output-dir results/individual_ply \
  --scale 4 --oct-pose-clean \
  --smooth 1.0 --mesh-step 1 --spacing 1,1,1
```

- `--surface-level` fixes the normalized isosurface threshold; omit it to use
  an automatic threshold per scan.
- `--smooth` increases Gaussian smoothing before extraction. More smoothing
  reduces noise but softens edges and fine detail; `0` disables it.
- `--mesh-step` takes larger marching-cubes steps. Larger values are faster
  and make a coarser mesh.
- `--spacing Z,Y,X` supplies physical voxel spacing; it changes mesh geometry
  into physical units rather than changing image quality.
- `--oct-pose-clean` applies percentile intensity windowing, a 3×3 in-plane
  median filter, and low-signal suppression before meshing. It can reduce
  speckle, but should be reviewed if faint real structure is important.

Before export, the workflow opens each selected scan in Napari. Its cleanup
features include:

- A capped, downsampled preview (750,000 voxels by default) for responsive 3-D
  inspection; `--preview-max-voxels` adjusts the cap.
- Interactive background floor, contrast limits, and gamma controls. Applying
  visible settings is enabled by default, so blacked-out voxels are removed
  from the saved scan and remaining intensities receive the selected gamma.
- An optional depth-band mask to retain only the desired Z range.
- Clickable 3-D artifact centres and a configurable spherical removal radius.
  Preview selections are expanded back to the corresponding full-resolution
  source blocks when saved.
- Safe 2-D fallback on Intel/Rosetta Python for macOS stability; native ARM64
  Python is recommended for 3-D rendering. Use `--force-3d` only when needed.

For instructions on operating those controls, including the relationship
between preview resolution and removal radius, see
[Napari cleanup: how it works](#napari-cleanup-how-it-works).

The mesh export path can also apply conservative OCT preprocessing: robust
percentile windowing, 3×3 in-plane median filtering, and a low-signal floor
to suppress isolated speckle before isosurface extraction. It supports
physical voxel spacing, mesh step size, smoothing, and a fixed or automatic
surface level.

For multi-scan PLY runs, transforms may come from manual offsets or an
existing NPY pose file. Optional broad X-facing panel refinement can be
enabled when every cleaned mesh has that feature; otherwise the
manual/reference transforms are used. The final mesh stitch supports
conservative overlap welding through an adjustable merge distance.

## Requirements

Use a Python environment with:

```text
numpy
scipy
matplotlib
napari
qtpy
scikit-image
```

Napari also requires a Qt backend appropriate for your platform. For large
OCT acquisitions, ensure sufficient RAM for the input volumes, output volume,
and stitching accumulators.

## Supporting commands

The scripts in `conversion_code/` can be run independently when a workflow
needs to be repeated or automated:

- `manual_reference_alignment.py` — interactively place one scan against a
  reference or placed-scan composite.
- `manual_offsets_to_transforms.py` — convert saved manual offsets to pairwise
  transforms.
- `positions_to_pairwise_transforms.py` — validate/convert global or pairwise
  4×4 NPY transforms.
- `stitch_npy_volumes.py` — resample and blend volumes into one NPY array.
- `napari_visualization.py` — save a cleaned NPY/NPZ copy after interactive
  artifact removal.
- `manual_object_roi.py` — create a JSON square object ROI from a PLY depth
  projection. It automatically proposes the largest non-background depth
  cluster, lets you adjust the square interactively, and saves `z_min`,
  `z_max`, `y_min`, and `y_max` bounds.
- `export_scans_to_ply.py`, `clean_ply_panels.py`, and
  `stitch_ply_meshes.py` — PLY export, cleanup, and mesh stitching.
