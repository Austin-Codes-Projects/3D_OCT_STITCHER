#!/usr/bin/env python3
"""Review a scan in 3D and remove artifacts by clicking their centers."""

import argparse
import math
import platform
from pathlib import Path

import napari
import numpy as np


def read_volume(path):
    """Load an NPY or a single-volume NPZ while retaining its original shape/key."""
    path = Path(path)
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        keys = list(loaded.files)
        key = next((name for name in ("volume", "stitched", "arr_0") if name in keys), None)
        if key is None and len(keys) == 1:
            key = keys[0]
        if key is None:
            loaded.close()
            raise ValueError(f"{path} contains multiple arrays {keys}; expected a volume array")
        volume = loaded[key]
        loaded.close()
        return volume, key
    return loaded, None


def save_cleaned(path, volume, archive_key):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npz":
        np.savez_compressed(path, **{archive_key or "volume": volume})
    elif path.suffix.lower() == ".npy":
        np.save(path, volume)
    else:
        raise ValueError("Output must end in .npy or .npz")


def smooth_background_for_display(volume, low, high, floor_percent):
    """Make low-intensity background uniform without changing object voxels.

    The floor is only used by the Napari preview.  Values beneath it are shown
    as the black end of the contrast range; values above it are retained
    exactly, so the object itself is not blurred or filtered.
    """
    if not 0 <= floor_percent <= 100:
        raise ValueError("Background floor must be between 0 and 100 percent")
    cutoff = low + (high - low) * floor_percent / 100
    return np.where(volume <= cutoff, low, volume)


def keep_depth_band(volume, start, stop, background_value):
    """Return a display copy with every Z slice outside ``start:stop`` hidden."""
    if not 0 <= start <= stop < volume.shape[0]:
        raise ValueError("Depth band must lie within the preview volume")
    kept = np.full_like(volume, background_value)
    kept[start:stop + 1] = volume[start:stop + 1]
    return kept


def apply_visible_display_settings(volume, low, high, gamma, floor_percent):
    """Bake Napari's visible intensity settings into a cleaned volume.

    Values blacked out by the background floor/low contrast limit are removed.
    Retained values are clipped at the high contrast limit and gamma-corrected
    exactly as a luminance display range is.  Work one Z plane at a time so a
    large OCT acquisition does not need another full-volume float copy.
    """
    low, high = sorted((float(low), float(high)))
    gamma = float(gamma)
    if high <= low:
        raise ValueError("Napari contrast limits must span a non-zero range")
    if gamma <= 0:
        raise ValueError("Napari gamma must be positive")
    if not 0 <= floor_percent <= 100:
        raise ValueError("Background floor must be between 0 and 100 percent")

    cutoff = low + (high - low) * floor_percent / 100
    is_float = np.issubdtype(volume.dtype, np.floating)
    output_scale = high if np.issubdtype(volume.dtype, np.integer) and high > 0 else 1.0
    removed = 0
    for z_index in range(volume.shape[0]):
        plane = volume[z_index]
        keep = plane > cutoff
        if is_float:
            keep &= np.isfinite(plane)
        removed += int(plane.size - np.count_nonzero(keep))
        plane[~keep] = 0
        if not np.any(keep):
            continue
        normalized = np.clip(
            (plane[keep].astype(np.float32) - low) / (high - low), 0.0, 1.0
        )
        visible_intensity = np.power(normalized, gamma)
        if np.issubdtype(volume.dtype, np.integer):
            dtype_limits = np.iinfo(volume.dtype)
            plane[keep] = np.clip(
                np.rint(visible_intensity * output_scale),
                dtype_limits.min,
                dtype_limits.max,
            ).astype(volume.dtype)
        else:
            plane[keep] = visible_intensity
    return removed, low, high, cutoff, gamma


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="source .npy or .npz scan")
    parser.add_argument("--output", required=True, help="cleaned copy to save as .npy or .npz")
    parser.add_argument(
        "--preview-max-voxels", type=int, default=750_000,
        help="maximum voxel count in the interactive preview (default: 750 thousand)",
    )
    parser.add_argument(
        "--force-3d", action="store_true",
        help="enable 3D rendering even under Intel/Rosetta Python on macOS",
    )
    args = parser.parse_args()

    source, archive_key = read_volume(args.input)
    volume = np.squeeze(source)
    if volume.ndim != 3:
        raise SystemExit(f"Expected a 3-D scan after removing singleton axes; got {source.shape}")

    # Keep the editable preview and label layer small; marked preview voxels
    # are expanded back to their corresponding source blocks when saving.
    if args.preview_max_voxels < 1:
        parser.error("--preview-max-voxels must be positive")
    step = max(1, math.ceil((volume.size / args.preview_max_voxels) ** (1 / 3)))
    preview = volume[::step, ::step, ::step]
    # Estimate contrast from a small sample. Scanning/sorting every preview
    # voxel adds noticeable startup time on large OCT volumes.
    flat = preview.reshape(-1)
    stride = max(1, flat.size // 500_000)
    sample = flat[::stride]
    finite = sample[np.isfinite(sample)] if np.issubdtype(sample.dtype, np.floating) else sample
    if finite.size == 0:
        raise SystemExit("The selected scan contains no finite intensity values")
    low, high = np.percentile(finite, (1.0, 99.7))
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        raise SystemExit("The selected scan has constant intensity; nothing can be displayed")
    display = preview
    if np.issubdtype(preview.dtype, np.floating) and not np.isfinite(preview).all():
        display = np.nan_to_num(preview, nan=low, posinf=high, neginf=low)
    background_floor = 5
    display = smooth_background_for_display(display, low, high, background_floor)
    viewer = napari.Viewer(title=f"Clean {Path(args.input).name}")
    scan_layer = viewer.add_image(
        display, name="Scan", colormap="gray", contrast_limits=(low, high),
        rendering="mip", depiction="volume", interpolation3d="linear",
        blending="additive", visible=True,
    )
    points = viewer.add_points(
        np.empty((0, 3), dtype=np.float32), name="Artifact centers",
        size=8, face_color="red", border_color="white", symbol="x", ndim=3,
    )
    points.mode = "add"
    # Napari's Mac crashes can be caused by running an x86 Python through
    # Rosetta. Prefer 3D on native Apple Silicon with a smaller render volume;
    # retain 2D as the safe fallback for Intel/Rosetta environments.
    native_or_non_mac = platform.system() != "Darwin" or platform.machine() == "arm64"
    start_in_3d = native_or_non_mac or args.force_3d
    viewer.dims.ndisplay = 3 if start_in_3d else 2
    viewer.dims.current_step = tuple(size // 2 for size in preview.shape)
    viewer.layers.selection.active = points
    viewer.reset_view()
    saved = {"done": False}

    def save_cleaned_volume():
        centers = np.asarray(points.data, dtype=float)
        cleaned = np.array(source, copy=True)
        squeezed = np.squeeze(cleaned)
        removed = 0
        radius = radius_control.value()
        display_note = ""
        if visible_settings_control.isChecked():
            display_removed, contrast_low, contrast_high, cutoff, gamma = apply_visible_display_settings(
                squeezed, *scan_layer.contrast_limits, scan_layer.gamma,
                background_control.value(),
            )
            removed += display_removed
            display_note = (
                f"; applied floor {cutoff:g}, contrast {contrast_low:g}..{contrast_high:g}, gamma {gamma:g}"
            )
        if depth_band_control.isChecked():
            depth_start, depth_stop = selected_depth_band()
            raw_z0 = depth_start * step
            raw_z1 = min((depth_stop + 1) * step, volume.shape[0])
            squeezed[:raw_z0] = 0
            squeezed[raw_z1:] = 0
        for center in centers:
            cz, cy, cx = np.rint(center).astype(int)
            # Point coordinates and the radius belong to Napari's preview,
            # not the source array.  Select a sphere of preview voxels, then
            # expand every selected preview voxel to its exact source block.
            # This keeps a clicked artifact in the same location when the
            # cleaned source volume is later converted into a PLY.
            rz = min(radius, preview.shape[0])
            ry = min(radius, preview.shape[1])
            rx = min(radius, preview.shape[2])
            zlo, zhi = max(0, cz - rz), min(preview.shape[0], cz + rz + 1)
            ylo, yhi = max(0, cy - ry), min(preview.shape[1], cy + ry + 1)
            xlo, xhi = max(0, cx - rx), min(preview.shape[2], cx + rx + 1)
            zz, yy, xx = np.ogrid[zlo - cz:zhi - cz, ylo - cy:yhi - cy, xlo - cx:xhi - cx]
            sphere = zz * zz + yy * yy + xx * xx <= radius * radius
            for local_z, preview_z in enumerate(range(zlo, zhi)):
                raw_z0, raw_z1 = preview_z * step, min((preview_z + 1) * step, volume.shape[0])
                for local_y, preview_y in enumerate(range(ylo, yhi)):
                    raw_y0, raw_y1 = preview_y * step, min((preview_y + 1) * step, volume.shape[1])
                    x_mask = np.zeros(volume.shape[2], dtype=bool)
                    local_x = np.flatnonzero(sphere[local_z, local_y]) + xlo
                    for preview_x in local_x:
                        raw_x0 = preview_x * step
                        raw_x1 = min((preview_x + 1) * step, volume.shape[2])
                        x_mask[raw_x0:raw_x1] = True
                    if x_mask.any():
                        block = squeezed[raw_z0:raw_z1, raw_y0:raw_y1, :]
                        removed += int(np.count_nonzero(block[..., x_mask]))
                        block[..., x_mask] = 0
        save_cleaned(args.output, cleaned, archive_key)
        saved["done"] = True
        band_note = ""
        if depth_band_control.isChecked():
            depth_start, depth_stop = selected_depth_band()
            band_note = (
                f"; kept preview Z={depth_start}..{depth_stop}"
            )
        print(
            f"Saved cleaned scan: {args.output} "
            f"({removed:,} source voxels removed{display_note}{band_note})"
        )

    from qtpy.QtWidgets import QCheckBox, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

    def update_background_preview(floor_percent):
        """Refresh the preview after its visible background floor changes."""
        update_preview()

    def update_preview(*_):
        preview_display = preview
        if np.issubdtype(preview.dtype, np.floating) and not np.isfinite(preview).all():
            preview_display = np.nan_to_num(preview, nan=low, posinf=high, neginf=low)
        preview_display = smooth_background_for_display(
            preview_display, low, high, background_control.value()
        )
        if depth_band_control.isChecked():
            depth_start, depth_stop = selected_depth_band()
            preview_display = keep_depth_band(
                preview_display,
                depth_start,
                depth_stop,
                low,
            )
        scan_layer.data = preview_display

    def set_depth_band_enabled(enabled):
        depth_start_control.setEnabled(enabled)
        depth_stop_control.setEnabled(enabled)
        update_preview()

    def selected_depth_band():
        """Return ordered preview slice bounds even while a spin box is edited."""
        return tuple(sorted((depth_start_control.value(), depth_stop_control.value())))

    instructions = QLabel(
        f"The background floor, contrast limits, and gamma are applied to the saved scan and exported PLY by default.\n"
        f"This makes voxels that are black in the intensity display absent from the PLY.\n"
        f"Enable the depth band to keep only the cyan panel's Z range; it is applied to the saved scan.\n"
        f"On macOS, native ARM64 Python is recommended for 3D rendering.\n"
        f"Preview step: {step} source voxels per preview voxel (750K preview voxels max).\n"
        f"Adjust the radius below, then save the cleaned copy."
    )
    radius_control = QSpinBox()
    radius_control.setRange(1, 100)
    radius_control.setValue(8)
    radius_control.setPrefix("Removal radius (preview voxels): ")
    background_control = QSpinBox()
    background_control.setRange(0, 95)
    background_control.setValue(background_floor)
    background_control.setSuffix("%")
    background_control.setPrefix("Background floor (also applied when saving): ")
    background_control.valueChanged.connect(update_background_preview)
    visible_settings_control = QCheckBox("Apply visible floor, contrast limits, and gamma to saved scan and PLY")
    visible_settings_control.setChecked(True)
    depth_band_control = QCheckBox("Keep only selected depth band (used when saving)")
    depth_start_control = QSpinBox()
    depth_start_control.setRange(0, preview.shape[0] - 1)
    depth_start_control.setValue(0)
    depth_start_control.setPrefix("Keep Z from (preview slice): ")
    depth_stop_control = QSpinBox()
    depth_stop_control.setRange(0, preview.shape[0] - 1)
    depth_stop_control.setValue(preview.shape[0] - 1)
    depth_stop_control.setPrefix("Keep Z through (preview slice): ")
    depth_start_control.setEnabled(False)
    depth_stop_control.setEnabled(False)
    depth_band_control.toggled.connect(set_depth_band_enabled)
    depth_start_control.valueChanged.connect(update_preview)
    depth_stop_control.valueChanged.connect(update_preview)
    button = QPushButton("Save cleaned volume")
    button.clicked.connect(save_cleaned_volume)
    controls = QWidget()
    layout = QVBoxLayout(controls)
    layout.addWidget(instructions)
    layout.addWidget(background_control)
    layout.addWidget(visible_settings_control)
    layout.addWidget(depth_band_control)
    layout.addWidget(depth_start_control)
    layout.addWidget(depth_stop_control)
    layout.addWidget(radius_control)
    layout.addWidget(button)
    viewer.window.add_dock_widget(controls, name="Cleanup", area="right")
    napari.run()
    if not saved["done"]:
        print("No cleaned output saved; pipeline stopped for this scan.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
