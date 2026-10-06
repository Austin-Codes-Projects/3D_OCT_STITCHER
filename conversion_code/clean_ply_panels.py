#!/usr/bin/env python3
"""Clean planar OCT artifacts from a PLY mesh or point cloud.

Exact planes are removed only when explicitly requested.  Broad, axis-aligned
sheet detection is optional because real OCT anatomy can also be planar.  This
is intended for long rectangular sheets made at scan boundaries.  Use
--dry-run before writing a new file.
"""

import argparse
from pathlib import Path

import numpy as np

from imes4d.mesh_export import save_ply
from stitch_ply_meshes import load_ply


AXES = {"z": 0, "y": 1, "x": 2}


def parse_plane(value):
    try:
        axis, position = value.lower().split(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("plane must be AXIS:POSITION, e.g. z:min or x:1164") from error
    if axis not in AXES:
        raise argparse.ArgumentTypeError("axis must be z, y, or x")
    if position not in {"min", "max"}:
        try:
            position = float(position)
        except ValueError as error:
            raise argparse.ArgumentTypeError("position must be min, max, or a number") from error
    return axis, position


def parse_region(value):
    """Parse y:low:high,z:low:high; bounds can be numbers, min, or max."""
    region = []
    for part in value.lower().split(","):
        pieces = part.split(":")
        if len(pieces) != 3 or pieces[0] not in AXES:
            raise argparse.ArgumentTypeError("region must be AXIS:LOW:HIGH pairs, e.g. y:1890:max,z:min:713")
        axis, low, high = pieces
        try:
            low = low if low in {"min", "max"} else float(low)
            high = high if high in {"min", "max"} else float(high)
        except ValueError as error:
            raise argparse.ArgumentTypeError("region bounds must be min, max, or a number") from error
        region.append((axis, low, high))
    return region


def resolve_plane(vertices, plane):
    axis_name, position = plane
    values = vertices[:, AXES[axis_name]]
    if position == "min":
        position = float(values.min())
    elif position == "max":
        position = float(values.max())
    return axis_name, float(position)


def resolve_bound(values, bound):
    if bound == "min":
        return float(values.min())
    if bound == "max":
        return float(values.max())
    return float(bound)


def plane_vertices(vertices, position, axis, tolerance):
    return np.abs(vertices[:, axis] - position) <= tolerance


def candidate_sheet_vertices(vertices, normals, axis, bin_width, normal_alignment, min_points, min_coverage):
    """Find large, nearly axis-normal planar vertex groups."""
    coordinate = vertices[:, axis]
    bins = np.rint(coordinate / bin_width).astype(np.int64)
    unique_bins, inverse = np.unique(bins, return_inverse=True)
    bin_count = len(unique_bins)
    aligned = np.abs(normals[:, axis]) >= normal_alignment
    if not aligned.any():
        return np.zeros(len(vertices), dtype=bool), np.array([])

    aligned_count = np.bincount(inverse[aligned], minlength=bin_count)
    other_axes = [index for index in range(3) if index != axis]
    spans = []
    for other_axis in other_axes:
        low = np.full(bin_count, np.inf)
        high = np.full(bin_count, -np.inf)
        np.minimum.at(low, inverse[aligned], vertices[aligned, other_axis])
        np.maximum.at(high, inverse[aligned], vertices[aligned, other_axis])
        total_span = np.ptp(vertices[:, other_axis])
        spans.append((high - low) / total_span if total_span else np.zeros(bin_count))

    selected_bins = (aligned_count >= min_points) & (spans[0] >= min_coverage) & (spans[1] >= min_coverage)
    return selected_bins[inverse] & aligned, unique_bins[selected_bins] * bin_width


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="input binary PLY mesh or point cloud")
    parser.add_argument("--output", required=True, help="cleaned binary PLY")
    parser.add_argument("--plane", type=parse_plane, action="append", default=[],
                        help="remove an exact plane, e.g. z:min or x:1164; repeat as needed")
    parser.add_argument("--remove-region", type=parse_region, action="append", default=[],
                        help="remove vertices inside a bounded region, e.g. y:1890:max,z:min:713")
    parser.add_argument("--boundary-caps", action="store_true",
                        help="also remove triangles on all six outer bounding-box planes")
    parser.add_argument("--auto-sheets", action="store_true",
                        help="detect broad axis-aligned sheets (review with --dry-run first)")
    parser.add_argument("--axis", choices=[*AXES, "all"], default="all",
                        help="sheet-normal direction to inspect (default: all)")
    parser.add_argument("--bin-width", type=float, default=1.0, help="sheet grouping width (default: 1)")
    parser.add_argument("--normal-alignment", type=float, default=0.98,
                        help="minimum normal component along sheet axis (default: 0.98)")
    parser.add_argument("--min-points", type=int, default=10_000,
                        help="minimum vertices in one sheet group (default: 10000)")
    parser.add_argument("--min-coverage", type=float, default=0.70,
                        help="minimum coverage of both other axes (default: 0.70)")
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="exact-plane and cap tolerance (default: 1)")
    parser.add_argument("--points-only", action="store_true", help="write vertices only; discard all faces")
    parser.add_argument("--dry-run", action="store_true", help="report selection without saving")
    args = parser.parse_args()
    if args.bin_width <= 0 or args.tolerance < 0:
        parser.error("--bin-width must be positive and --tolerance must be non-negative")
    if not 0 <= args.normal_alignment <= 1 or args.min_points < 1 or not 0 < args.min_coverage <= 1:
        parser.error("invalid sheet-detection thresholds")

    vertices, normals, faces = load_ply(Path(args.input))
    remove = np.zeros(len(vertices), dtype=bool)

    if args.auto_sheets:
        axis_names = AXES if args.axis == "all" else [args.axis]
        for axis_name in axis_names:
            matches, centres = candidate_sheet_vertices(
                vertices, normals, AXES[axis_name], args.bin_width, args.normal_alignment,
                args.min_points, args.min_coverage,
            )
            remove |= matches
            print(f"{axis_name.upper()} sheets at {centres.tolist()}: {matches.sum():,} vertices selected")

    requested_planes = list(args.plane)
    if args.boundary_caps:
        requested_planes.extend((axis, bound) for axis in AXES for bound in ("min", "max"))
    for requested in requested_planes:
        axis_name, position = resolve_plane(vertices, requested)
        matches = plane_vertices(vertices, position, AXES[axis_name], args.tolerance)
        newly_removed = np.count_nonzero(matches & ~remove)
        remove |= matches
        print(f"{axis_name}={position:g}: {newly_removed:,} vertices selected")

    for region in args.remove_region:
        matches = np.ones(len(vertices), dtype=bool)
        description = []
        for axis_name, low, high in region:
            values = vertices[:, AXES[axis_name]]
            low, high = resolve_bound(values, low), resolve_bound(values, high)
            if low > high:
                # A fixed cleanup region may not overlap a smaller scan (for
                # example y:1890:max when its maximum Y is below 1890).  It
                # simply selects no vertices and must not stop the pipeline.
                description.append(f"{axis_name}=outside mesh")
                matches = np.zeros(len(vertices), dtype=bool)
                break
            matches &= (values >= low) & (values <= high)
            description.append(f"{axis_name}={low:g}..{high:g}")
        newly_removed = np.count_nonzero(matches & ~remove)
        remove |= matches
        print(f"region ({', '.join(description)}): {newly_removed:,} vertices selected")

    print(f"Total selected: {remove.sum():,} of {len(vertices):,} vertices")
    if args.dry_run:
        return
    kept = ~remove
    if args.points_only:
        save_ply(vertices[kept], np.empty((0, 3), dtype=np.int32), normals[kept], args.output)
        print(f"Saved point cloud {args.output} ({kept.sum():,} vertices)")
        return

    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[kept] = np.arange(kept.sum(), dtype=np.int32)
    kept_faces = faces[np.all(kept[faces], axis=1)]
    save_ply(vertices[kept], remap[kept_faces], normals[kept], args.output)
    print(f"Saved {args.output} ({kept.sum():,} vertices, {len(kept_faces):,} faces)")


if __name__ == "__main__":
    main()
