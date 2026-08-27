#!/usr/bin/env python3
"""Make each adjacent PLY pair share its first detected broad panel axis.

This adjusts only one translation component of existing manual pairwise
transforms.  It is intended for scans that share a repeated broad planar
feature: the first (lowest-coordinate) detected panel in every later scan is
made coincident with the first panel in the preceding scan.  The other manual
translation components are preserved, and no ICP is used.
"""

import argparse
import re
from pathlib import Path

import numpy as np

from clean_ply_panels import AXES, candidate_sheet_vertices
from stitch_ply_meshes import load_ply


def volume_key(path):
    match = re.fullmatch(r"vol(\d+)\.ply", path.name)
    if not match:
        raise ValueError(f"Expected vol<number>.ply, got {path.name}")
    return int(match.group(1))


def parse_scale(value):
    try:
        scale = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("coordinate scale must be Z,Y,X") from error
    if len(scale) != 3 or any(item <= 0 for item in scale):
        raise argparse.ArgumentTypeError("coordinate scale must contain three positive values")
    return scale


def first_panel_coordinate(vertices, normals, axis, bin_width, normal_alignment, min_points, min_coverage):
    _, centres = candidate_sheet_vertices(
        vertices, normals, axis, bin_width, normal_alignment, min_points, min_coverage,
    )
    if len(centres) == 0:
        raise ValueError("no broad panels detected")
    centres = np.sort(centres)
    # The two sides of one thin mesh sheet commonly occupy adjacent bins.
    first_group = [centres[0]]
    for centre in centres[1:]:
        if centre - first_group[-1] > bin_width * 1.5:
            break
        first_group.append(centre)
    return float(np.mean(first_group)), first_group, centres


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="directory containing vol1.ply, vol2.ply, ...")
    parser.add_argument("--initial-transforms", required=True, help="existing manual pairwise 4x4 transforms (.npy)")
    parser.add_argument("--output", required=True, help="adjusted pairwise transform output (.npy)")
    parser.add_argument("--axis", choices=AXES, default="x", help="panel-normal axis in volume Z,Y,X order (default: x)")
    parser.add_argument("--coordinate-scale", type=parse_scale, default=(1, 1, 1),
                        help="PLY coordinate scale relative to transforms, Z,Y,X (default: 1,1,1)")
    parser.add_argument("--bin-width", type=float, default=1.0)
    parser.add_argument("--normal-alignment", type=float, default=0.98)
    parser.add_argument("--min-points", type=int, default=10_000)
    parser.add_argument("--min-coverage", type=float, default=0.70)
    args = parser.parse_args()

    files = sorted(Path(args.input_dir).glob("vol*.ply"), key=volume_key)
    if len(files) < 2:
        parser.error(f"Need at least two vol*.ply files in {args.input_dir}")
    transforms = np.load(args.initial_transforms).astype(float, copy=True)
    if transforms.shape != (len(files) - 1, 4, 4):
        parser.error("--initial-transforms must contain one 4x4 transform per adjacent PLY pair")
    if args.bin_width <= 0 or not 0 <= args.normal_alignment <= 1 or args.min_points < 1 or not 0 < args.min_coverage <= 1:
        parser.error("invalid panel-detection thresholds")

    axis = AXES[args.axis]
    panel_coordinates = []
    for path in files:
        vertices, normals, _ = load_ply(path)
        try:
            coordinate, group, all_centres = first_panel_coordinate(
                vertices, normals, axis, args.bin_width, args.normal_alignment,
                args.min_points, args.min_coverage,
            )
        except ValueError as error:
            parser.error(f"{path.name}: {error}")
        panel_coordinates.append(coordinate)
        print(f"{path.name}: first {args.axis}-normal panel = {coordinate:g} (bins {group}; all {all_centres.tolist()})")

    for index, (first, second) in enumerate(zip(panel_coordinates[:-1], panel_coordinates[1:])):
        # P maps scan i into scan i+1; stitch_ply_meshes applies inv(P).
        # Therefore P's translation is second_panel - first_panel.
        old = transforms[index, axis, 3]
        transforms[index, axis, 3] = (second - first) / args.coordinate_scale[axis]
        print(f"{files[index].stem} -> {files[index + 1].stem}: {args.axis} translation {old:g} -> "
              f"{transforms[index, axis, 3]:g} (transform units)")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, transforms)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
