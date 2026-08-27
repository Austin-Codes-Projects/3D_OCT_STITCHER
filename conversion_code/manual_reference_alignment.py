#!/usr/bin/env python3
"""Manually place one scan on a reference or a previously assembled puzzle.

Use the offset sliders until the green moving scan overlays the magenta
background in all three planes.  The background can be a complete reference
volume or a composite of scans that have already been placed.  Offsets are
stored in downsampled voxel coordinates.
"""

import argparse
import csv
import os
import sys

import matplotlib
# IDEs sometimes select Agg or an inline backend, which renders the figure
# without creating the separate slider window this workflow needs. On macOS,
# select the native GUI backend before pyplot is imported.
if sys.platform == "darwin" and any(token in matplotlib.get_backend().lower()
                                   for token in ("agg", "inline")):
    matplotlib.use("MacOSX", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from scipy.ndimage import shift as image_shift

from load_vol import load_volume


def load_volume_for_view(filename, scale):
    volume = load_volume(filename, scale)
    maximum = volume.max()
    if maximum <= 0:
        raise ValueError(f"{filename} contains no positive intensity values")
    return volume / maximum


def load_saved_offsets(filename):
    """Read positions saved by earlier manual-alignment windows."""
    if not filename or not os.path.exists(filename):
        return {}
    with open(filename, newline="") as handle:
        return {
            row["volume"]: tuple(float(row[key]) for key in ("z_vox", "y_vox", "x_vox"))
            for row in csv.DictReader(handle)
        }


def shifted_volume(volume, offsets):
    """Place a volume in global/reference coordinates without wraparound."""
    return image_shift(volume, offsets, order=1, mode="constant", cval=0.0)


def puzzle_background(filenames, offsets_file, scale, transform_scale, shape):
    """Combine previously placed scans into the background for the next scan."""
    background = np.zeros(shape, dtype=np.float32)
    saved = load_saved_offsets(offsets_file)
    for filename in filenames:
        name = os.path.basename(filename)
        if name not in saved:
            raise ValueError(f"No saved offset for puzzle-background scan: {name}")
        # Saved offsets use transform-grid coordinates, while this display may
        # be more downsampled for responsiveness.
        display_offset = np.asarray(saved[name]) * transform_scale / scale
        placed = shifted_volume(load_volume_for_view(filename, scale), display_offset)
        background = np.maximum(background, placed)
    return background


def shifted_slice(volume, plane, index, offsets):
    """Return a moving-volume slice in reference/global coordinates."""
    dz, dy, dx = offsets
    if plane == "axial":
        source_index, in_plane_shift = index - dz, (dy, dx)
        axis = 0
    elif plane == "coronal":
        source_index, in_plane_shift = index - dy, (dz, dx)
        axis = 1
    else:
        source_index, in_plane_shift = index - dx, (dz, dy)
        axis = 2
    if not 0 <= source_index < volume.shape[axis]:
        shape = tuple(size for i, size in enumerate(volume.shape) if i != axis)
        return np.zeros(shape, dtype=np.float32)
    source = np.take(volume, source_index, axis=axis)
    return image_shift(source, in_plane_shift, order=1, mode="constant", cval=0.0)


def overlay(reference, moving, vmin, vmax):
    scale = max(vmax - vmin, np.finfo(np.float32).eps)
    ref = np.clip((reference - vmin) / scale, 0, 1)
    mov = np.clip((moving - vmin) / scale, 0, 1)
    return np.dstack((ref, mov, np.zeros_like(ref)))


def save_offset(filename, volume_name, offsets):
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    rows = []
    if os.path.exists(filename):
        with open(filename, newline="") as handle:
            rows = list(csv.DictReader(handle))
    values = {"volume": volume_name, "z_vox": offsets[0], "y_vox": offsets[1], "x_vox": offsets[2]}
    rows = [row for row in rows if row.get("volume") != volume_name]
    rows.append(values)
    rows.sort(key=lambda row: row["volume"])
    with open(filename, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("volume", "z_vox", "y_vox", "x_vox"))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", required=True, help="scan to align, e.g. scans/vol1.npy")
    parser.add_argument("--reference", help="complete reference volume; omit when using --background-volumes")
    parser.add_argument("--background-volumes", nargs="*", default=[],
                        help="already placed scans to combine as a puzzle background")
    parser.add_argument("--scale", type=int, default=4,
                        help="downsample factor used only for the responsive alignment view")
    parser.add_argument("--transform-scale", type=int,
                        help="downsample factor used for the final PLY/transforms; defaults to --scale")
    parser.add_argument("--offsets", default="scans/manual_offsets.csv")
    args = parser.parse_args()
    if args.scale < 1:
        parser.error("--scale must be at least 1")
    if args.transform_scale is None:
        args.transform_scale = args.scale
    if args.transform_scale < 1:
        parser.error("--transform-scale must be at least 1")

    moving = load_volume_for_view(args.volume, args.scale)
    if args.reference and args.background_volumes:
        parser.error("Use either --reference or --background-volumes, not both")
    if args.reference:
        reference = load_volume_for_view(args.reference, args.scale)
        background_label = "reference"
        initial_offset = (0, 0, 0)
    elif args.background_volumes:
        reference = puzzle_background(args.background_volumes, args.offsets, args.scale, args.transform_scale, moving.shape)
        background_label = "placed-scan puzzle"
        # Manual offsets are global coordinates. Start the next scan at the
        # most recently placed scan's coordinate, rather than at zero, so the
        # slider adjustments are made in the same coordinate frame as the
        # magenta composite and transfer correctly into the stitch transform.
        saved = load_saved_offsets(args.offsets)
        initial_offset = tuple(np.asarray(saved[os.path.basename(args.background_volumes[-1])])
                               * args.transform_scale / args.scale)
    else:
        reference = np.zeros_like(moving)
        background_label = "blank canvas"
        initial_offset = (0, 0, 0)
    if reference.shape != moving.shape:
        raise ValueError(f"Shape mismatch: background {reference.shape}, moving {moving.shape}")
    # A blank canvas is valid for the first puzzle piece.  In that case, use
    # the moving scan to establish a useful grayscale range for the panels.
    foreground = reference[reference > 1e-3]
    if foreground.size == 0:
        foreground = moving[moving > 1e-3]
    vmin, vmax = np.percentile(foreground, (1, 99))
    center = [dimension // 2 for dimension in reference.shape]
    limits = [dimension - 1 for dimension in reference.shape]

    # This program is launched as a subprocess by the raw-to-PLY workflow.
    # Explicitly disable interactive mode and block at show() so the pipeline
    # cannot continue before the operator has used the movement controls.
    plt.ioff()
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.canvas.manager.set_window_title(f"Manual placement — {os.path.basename(args.volume)}")
    fig.subplots_adjust(bottom=0.31, hspace=0.34, wspace=0.18)
    images = [[None] * 3 for _ in range(3)]
    plane_names = ("Axial (Z)", "Coronal (Y)", "Sagittal (X)")

    def draw(indices, offsets):
        reference_slices = (
            reference[indices[0], :, :],
            reference[:, indices[1], :],
            reference[:, :, indices[2]],
        )
        moving_slices = (
            shifted_slice(moving, "axial", indices[0], offsets),
            shifted_slice(moving, "coronal", indices[1], offsets),
            shifted_slice(moving, "sagittal", indices[2], offsets),
        )
        for row, (name, ref_slice, moving_slice) in enumerate(zip(plane_names, reference_slices, moving_slices)):
            values = (ref_slice, moving_slice, overlay(ref_slice, moving_slice, vmin, vmax))
            headings = (background_label, "moving scan", "overlay: magenta=background, green=scan")
            for col, (value, heading) in enumerate(zip(values, headings)):
                if images[row][col] is None:
                    images[row][col] = axes[row, col].imshow(
                        value, cmap=None if col == 2 else "gray", vmin=None if col == 2 else vmin,
                        vmax=None if col == 2 else vmax, origin="lower",
                    )
                    axes[row, col].set_title(f"{name}: {heading}")
                else:
                    images[row][col].set_data(value)

    draw(center, initial_offset)
    location_axes = [fig.add_axes((0.20, y, 0.62, 0.018)) for y in (0.235, 0.205, 0.175)]
    offset_axes = [fig.add_axes((0.20, y, 0.62, 0.018)) for y in (0.125, 0.095, 0.065)]
    locations = [Slider(axis, label, 0, maximum, valinit=value, valstep=1)
                 for axis, label, maximum, value in zip(location_axes, ("Reference Z", "Reference Y", "Reference X"), limits, center)]
    offsets = [Slider(axis, label, -maximum, maximum, valinit=value, valstep=1)
               for axis, label, maximum, value in zip(offset_axes, ("Offset Z", "Offset Y", "Offset X"), limits, initial_offset)]
    status = fig.text(0.20, 0.015, "Adjust offsets until shared features are yellow/white, then Save offset.")

    def update(_):
        indices = tuple(int(slider.val) for slider in locations)
        offset_values = tuple(int(slider.val) for slider in offsets)
        draw(indices, offset_values)
        saved_values = tuple(value * args.scale / args.transform_scale for value in offset_values)
        status.set_text(f"View offset: {offset_values}; saved transform offset (Z, Y, X): {saved_values}")
        fig.canvas.draw_idle()

    for slider in locations + offsets:
        slider.on_changed(update)

    button_axis = fig.add_axes((0.84, 0.055, 0.11, 0.05))
    button = Button(button_axis, "Save offset")

    def save(_):
        view_values = tuple(int(slider.val) for slider in offsets)
        transform_values = tuple(value * args.scale / args.transform_scale for value in view_values)
        save_offset(args.offsets, os.path.basename(args.volume), transform_values)
        status.set_text(f"Saved transform offset {transform_values} to {args.offsets}")
        fig.canvas.draw_idle()

    button.on_clicked(save)
    print(f"Background: {background_label}; shape (scaled): {reference.shape}")
    print(f"Initial view offset (Z, Y, X): {initial_offset}")
    print("Align green moving features to magenta background features, then click Save offset.")
    print("Opening manual placement window now; close that window to continue the pipeline.")
    fig.canvas.manager.show()
    plt.show(block=True)


if __name__ == "__main__":
    main()
