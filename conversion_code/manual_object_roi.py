# #!/usr/bin/env python3
# """Select a complete object rectangle from a PLY depth projection."""

# import argparse
# import json
# import sys
# from pathlib import Path

# import matplotlib

# if sys.platform == "darwin" and any(item in matplotlib.get_backend().lower()
#                                     for item in ("agg", "inline")):
#     matplotlib.use("MacOSX", force=True)
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.widgets import Button, RectangleSelector

# from stitch_ply_meshes import load_ply


# def depth_projection(vertices, bin_width):
#     lateral = vertices[:, :2]
#     origin = lateral.min(axis=0)
#     bins = np.floor((lateral - origin) / bin_width).astype(np.int32)
#     shape = tuple(bins.max(axis=0) + 1)
#     image = np.full(shape, -np.inf, dtype=np.float32)
#     np.maximum.at(image, (bins[:, 0], bins[:, 1]), vertices[:, 2])
#     image[~np.isfinite(image)] = np.nan
#     return image, (origin[0], origin[0] + shape[0] * bin_width,
#                    origin[1], origin[1] + shape[1] * bin_width)


# def main():
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument("--input", required=True)
#     parser.add_argument("--output", required=True)
#     parser.add_argument("--bin-width", type=float, default=4.0)
#     parser.add_argument("--padding", type=float, default=32.0)
#     args = parser.parse_args()
#     if args.bin_width <= 0 or args.padding < 0:
#         parser.error("--bin-width must be positive and --padding non-negative")

#     vertices, _, _ = load_ply(Path(args.input))
#     image, extent = depth_projection(vertices, args.bin_width)
#     finite = image[np.isfinite(image)]
#     if finite.size == 0:
#         parser.error("PLY has no finite vertices")
#     figure, axis = plt.subplots(figsize=(12, 8))
#     figure.subplots_adjust(bottom=0.14)
#     axis.imshow(image.T, origin="lower", extent=extent, aspect="auto", cmap="turbo",
#                 vmin=np.percentile(finite, 2), vmax=np.percentile(finite, 98))
#     axis.set(xlabel="Z PLY coordinate", ylabel="Y PLY coordinate",
#              title="Drag a rectangle containing the complete object, then click Save ROI")
#     selection = {"bounds": None}
#     status = figure.text(0.12, 0.03, "No ROI selected yet.")

#     def on_select(start, end):
#         if None in (start.xdata, start.ydata, end.xdata, end.ydata):
#             return
#         z_low, z_high = sorted((start.xdata, end.xdata))
#         y_low, y_high = sorted((start.ydata, end.ydata))
#         selection["bounds"] = (z_low, z_high, y_low, y_high)
#         status.set_text(f"Selected Z={z_low:.1f}..{z_high:.1f}, Y={y_low:.1f}..{y_high:.1f}; padding={args.padding:g}")
#         figure.canvas.draw_idle()

#     RectangleSelector(axis, on_select, useblit=True, button=[1], interactive=True)
#     button = Button(figure.add_axes((0.78, 0.025, 0.14, 0.06)), "Save ROI")

#     def save(_):
#         if selection["bounds"] is None:
#             status.set_text("Drag a rectangle around the full object before saving.")
#             figure.canvas.draw_idle()
#             return
#         z_low, z_high, y_low, y_high = selection["bounds"]
#         roi = {"z_min": max(extent[0], z_low - args.padding), "z_max": min(extent[1], z_high + args.padding),
#                "y_min": max(extent[2], y_low - args.padding), "y_max": min(extent[3], y_high + args.padding)}
#         output = Path(args.output)
#         output.parent.mkdir(parents=True, exist_ok=True)
#         output.write_text(json.dumps(roi, indent=2) + "\n")
#         status.set_text(f"Saved {output}")
#         figure.canvas.draw_idle()

#     button.on_clicked(save)
#     print("Drag around the entire object, click Save ROI, then close the window to continue.")
#     plt.show(block=True)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""Select a complete object square from a PLY depth projection.

Restores the earlier working depth-projection UI and automatically centres a
square crop on the non-background (non-normal) depth region.  The square is
made large enough to contain the full object; the user can still drag to
adjust before saving.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
if sys.platform == "darwin" and any(
    token in matplotlib.get_backend().lower() for token in ("agg", "inline")
):
    matplotlib.use("MacOSX", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RectangleSelector
from scipy.ndimage import binary_opening, label

from stitch_ply_meshes import load_ply


def depth_projection(vertices, bin_width):
    """Max-depth projection in the lateral (Z, Y) plane."""
    lateral = vertices[:, :2]
    origin = lateral.min(axis=0)
    bins = np.floor((lateral - origin) / bin_width).astype(np.int32)
    shape = tuple(bins.max(axis=0) + 1)
    image = np.full(shape, -np.inf, dtype=np.float32)
    np.maximum.at(image, (bins[:, 0], bins[:, 1]), vertices[:, 2])
    image[~np.isfinite(image)] = np.nan
    extent = (
        origin[0],
        origin[0] + shape[0] * bin_width,
        origin[1],
        origin[1] + shape[1] * bin_width,
    )
    return image, extent


def estimate_square_object_roi(vertices, bin_width, padding, radius_scale=1.6):
    """Locate the non-background depth cluster and return a square ROI.

    Uses the same depth-mode separation as clean_ply_panels.estimate_depth_cluster_bounds
    (depth axis = X = coordinate 2).  The returned bounds are forced square and
    expanded so the full object remains visible.
    """
    depth_axis = 2  # X
    lateral_axes = [0, 1]  # Z, Y
    lateral = vertices[:, lateral_axes]
    origin = lateral.min(axis=0)
    bins = np.floor((lateral - origin) / bin_width).astype(np.int32)
    shape = tuple(bins.max(axis=0) + 1)

    furthest_depth = np.full(shape, -np.inf, dtype=np.float32)
    np.maximum.at(furthest_depth, (bins[:, 0], bins[:, 1]), vertices[:, depth_axis])

    sampled_depth = vertices[:, depth_axis]
    if sampled_depth.size < 20:
        raise ValueError("not enough depth samples to find the object")

    rounded = np.rint(sampled_depth).astype(np.int32)
    counts = np.bincount(rounded - rounded.min())

    # Background = dominant lower-depth mode; object = dominant higher-depth mode
    low_start = int(np.percentile(rounded, 5)) - rounded.min()
    low_end = int(np.percentile(rounded, 60)) - rounded.min()
    if low_end <= low_start:
        raise ValueError("no lower-depth background mode found")
    background_depth = int(
        np.argmax(counts[max(0, low_start):low_end]) + max(0, low_start) + rounded.min()
    )

    high_start = int(np.percentile(rounded, 70)) - rounded.min()
    higher = np.flatnonzero(counts[max(0, high_start):]) + max(0, high_start)
    if higher.size == 0:
        raise ValueError("no depth-different object cluster found")
    object_depth = int(higher[np.argmax(counts[higher])] + rounded.min())

    cutoff = (background_depth + object_depth) / 2
    object_high = object_depth + max(64, 0.20 * (object_depth - background_depth))
    candidate = (furthest_depth >= cutoff) & (furthest_depth <= object_high)

    # Remove thin streaks that would otherwise glue the object to the border
    candidate = binary_opening(candidate, structure=np.ones((3, 3), dtype=bool))
    labels, label_count = label(candidate, structure=np.ones((3, 3), dtype=np.uint8))
    if label_count == 0:
        raise ValueError("no connected depth-different object cluster found")

    areas = np.bincount(labels.ravel(), minlength=label_count + 1)
    areas[0] = 0
    chosen = labels == np.argmax(areas)
    cells = np.argwhere(chosen)

    # Centre of the object cluster (world coordinates)
    cell_centres = origin + cells * bin_width + bin_width / 2
    centre = cell_centres.mean(axis=0)  # (Z, Y)

    # Half-side of the axis-aligned bounding box, then force a square
    half_extents = (cells.max(axis=0) - cells.min(axis=0) + 1) * bin_width / 2
    half_side = float(max(half_extents[0], half_extents[1]))

    # Enlarge so the full object is comfortably inside (radius_scale > 1)
    # and add the user padding on top
    half_side = half_side * radius_scale + padding

    z_min = float(centre[0] - half_side)
    z_max = float(centre[0] + half_side)
    y_min = float(centre[1] - half_side)
    y_max = float(centre[1] + half_side)

    return {
        "z_min": z_min,
        "z_max": z_max,
        "y_min": y_min,
        "y_max": y_max,
        "centre_z": float(centre[0]),
        "centre_y": float(centre[1]),
        "half_side": half_side,
        "background_depth": background_depth,
        "object_depth": object_depth,
        "cluster_cells": int(chosen.sum()),
    }


def clamp_square_to_extent(roi, extent):
    """Keep the square inside the projection extent while preserving size if possible."""
    z0, z1, y0, y1 = extent
    side = min(roi["z_max"] - roi["z_min"], roi["y_max"] - roi["y_min"])
    cz = 0.5 * (roi["z_min"] + roi["z_max"])
    cy = 0.5 * (roi["y_min"] + roi["y_max"])

    half = side / 2
    # Shift centre if the square would go outside
    cz = max(z0 + half, min(z1 - half, cz))
    cy = max(y0 + half, min(y1 - half, cy))
    # If the image is smaller than the square, just clamp
    half = min(half, (z1 - z0) / 2, (y1 - y0) / 2)

    return {
        "z_min": float(cz - half),
        "z_max": float(cz + half),
        "y_min": float(cy - half),
        "y_max": float(cy + half),
    }


def force_square(z0, z1, y0, y1):
    """Expand the smaller side so the rectangle becomes a square (centred)."""
    cz = 0.5 * (z0 + z1)
    cy = 0.5 * (y0 + y1)
    half = 0.5 * max(z1 - z0, y1 - y0)
    return cz - half, cz + half, cy - half, cy + half


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bin-width", type=float, default=4.0)
    parser.add_argument("--padding", type=float, default=32.0,
                        help="Extra margin added outside the detected object (world units)")
    parser.add_argument("--radius-scale", type=float, default=1.6,
                        help="Multiplier applied to the detected object half-size "
                             "before padding (larger = more background kept)")
    args = parser.parse_args()
    if args.bin_width <= 0 or args.padding < 0 or args.radius_scale <= 0:
        parser.error("--bin-width, --padding and --radius-scale must be positive")

    vertices, _, _ = load_ply(Path(args.input))
    image, extent = depth_projection(vertices, args.bin_width)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise SystemExit("PLY has no finite vertices")

    # Auto-detect square ROI on the non-normal / non-background depth cluster
    try:
        auto = estimate_square_object_roi(
            vertices, args.bin_width, args.padding, args.radius_scale
        )
        auto = clamp_square_to_extent(auto, extent)
        print(
            f"Auto square ROI centred at Z={auto['z_min'] + (auto['z_max']-auto['z_min'])/2:.1f}, "
            f"Y={auto['y_min'] + (auto['y_max']-auto['y_min'])/2:.1f}  "
            f"half-side={ (auto['z_max']-auto['z_min'])/2 :.1f}"
        )
    except ValueError as exc:
        print(f"Auto detection failed ({exc}); start with empty selection.")
        auto = None

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(bottom=0.14)
    fig.canvas.manager.set_window_title(f"Object ROI — {Path(args.input).name}")

    vmin, vmax = np.percentile(finite, [2, 98])
    ax.imshow(
        image.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set(
        xlabel="Z PLY coordinate",
        ylabel="Y PLY coordinate",
        title="Non-background depth region (auto square).  "
              "Drag to adjust if needed, then click Save ROI",
    )

    selection = {"bounds": None}
    status = fig.text(0.12, 0.03, "No ROI selected yet.")

    def on_select(start, end):
        if None in (start.xdata, start.ydata, end.xdata, end.ydata):
            return
        z0, z1 = sorted((start.xdata, end.xdata))
        y0, y1 = sorted((start.ydata, end.ydata))
        # Force square immediately so the visual feedback matches what will be saved
        z0, z1, y0, y1 = force_square(z0, z1, y0, y1)
        selection["bounds"] = (z0, z1, y0, y1)
        side = z1 - z0
        status.set_text(
            f"Square  Z={z0:.1f}…{z1:.1f}   Y={y0:.1f}…{y1:.1f}   side={side:.1f}"
        )
        fig.canvas.draw_idle()

    selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        interactive=True,
        minspanx=5,
        minspany=5,
        spancoords="data",
        drag_from_anywhere=True,
        props=dict(facecolor="yellow", edgecolor="red", alpha=0.30, linewidth=2),
    )

    # Pre-load the auto-detected square so the user sees it immediately
    if auto is not None:
        selector.extents = (
            auto["z_min"],
            auto["z_max"],
            auto["y_min"],
            auto["y_max"],
        )
        selection["bounds"] = (
            auto["z_min"],
            auto["z_max"],
            auto["y_min"],
            auto["y_max"],
        )
        side = auto["z_max"] - auto["z_min"]
        status.set_text(
            f"Auto square  Z={auto['z_min']:.1f}…{auto['z_max']:.1f}   "
            f"Y={auto['y_min']:.1f}…{auto['y_max']:.1f}   side={side:.1f}"
        )

    btn_ax = fig.add_axes((0.78, 0.025, 0.14, 0.06))
    button = Button(btn_ax, "Save ROI")

    def save(_):
        if selection["bounds"] is None:
            status.set_text("Drag a square around the object first.")
            fig.canvas.draw_idle()
            return
        z0, z1, y0, y1 = selection["bounds"]
        # Final safety: force square again and clamp to the projection extent
        z0, z1, y0, y1 = force_square(z0, z1, y0, y1)
        z0 = max(extent[0], z0)
        z1 = min(extent[1], z1)
        y0 = max(extent[2], y0)
        y1 = min(extent[3], y1)
        # Re-square after clamping (may shrink if near the border)
        z0, z1, y0, y1 = force_square(z0, z1, y0, y1)

        roi = {
            "z_min": float(z0),
            "z_max": float(z1),
            "y_min": float(y0),
            "y_max": float(y1),
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(roi, indent=2) + "\n")
        status.set_text(f"Saved → {out}")
        fig.canvas.draw_idle()
        print(f"ROI saved to {out}")
        print(json.dumps(roi, indent=2))

    button.on_clicked(save)

    print("Window opened (restored depth-projection style).")
    print("Yellow/red square = auto non-background depth region (forced square).")
    print("Drag to adjust → Save ROI → close window.")
    plt.show(block=True)


if __name__ == "__main__":
    main()