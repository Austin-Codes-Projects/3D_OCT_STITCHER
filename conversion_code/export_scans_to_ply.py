#!/usr/bin/env python3
"""Export every acquired OCT volume as an individual PLY mesh.

The files are deliberately kept separate: ``stitch_ply_meshes.py`` can then
register them as surfaces without changing the existing volume stitcher.
"""

import argparse
import re
from pathlib import Path

from imes4d.mesh_export import export_volume_as_ply
from load_vol import load_volume


def volume_key(path):
    match = re.fullmatch(r"vol(\d+)\.(?:npy|npz)", path.name)
    if not match:
        raise ValueError(f"Expected vol<number>.npy or vol<number>.npz, got {path.name}")
    return int(match.group(1))


def parse_spacing(value):
    try:
        spacing = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("spacing must be Z,Y,X, e.g. 0.01,0.02,0.02") from error
    if len(spacing) != 3 or any(item <= 0 for item in spacing):
        raise argparse.ArgumentTypeError("spacing must contain three positive Z,Y,X values")
    return spacing


def main():
    parser = argparse.ArgumentParser(description="Convert vol*.npy scans to separate PLY meshes.")
    parser.add_argument("--input-dir", default="scans")
    parser.add_argument("--output-dir", default="results/individual_ply")
    parser.add_argument("--volumes", nargs="*", help="specific vol*.npy/.npz files to export; default exports all")
    parser.add_argument("--scale", type=int, default=4, help="downsample factor; must match stitching transforms")
    parser.add_argument("--surface-level", type=float, help="fixed normalized isosurface level; default uses Otsu per scan")
    parser.add_argument("--smooth", type=float, default=1.0)
    parser.add_argument("--mesh-step", type=int, default=1)
    parser.add_argument("--spacing", type=parse_spacing, default=(1, 1, 1), help="physical Z,Y,X voxel spacing")
    args = parser.parse_args()

    if args.scale < 1:
        parser.error("--scale must be at least 1")
    files = [Path(filename) for filename in args.volumes] if args.volumes else [
        *Path(args.input_dir).glob("vol*.npy"), *Path(args.input_dir).glob("vol*.npz")
    ]
    files = sorted(files, key=volume_key)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        parser.error(f"Selected volume file not found: {', '.join(missing)}")
    if not files:
        parser.error(f"No vol*.npy or vol*.npz files found in {args.input_dir}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # A sampled voxel represents ``scale`` original voxels in every direction.
    spacing = tuple(value * args.scale for value in args.spacing)
    for source in files:
        volume = load_volume(source, args.scale)
        maximum = volume.max()
        if maximum <= 0:
            print(f"Skipping {source}: no positive voxels")
            continue
        output = output_dir / f"{source.stem}.ply"
        vertices, faces, level = export_volume_as_ply(
            volume / maximum, output, level=args.surface_level,
            smoothing_sigma=args.smooth, step_size=args.mesh_step, spacing=spacing,
        )
        print(f"Saved {output} ({vertices:,} vertices, {faces:,} faces; level={level:.4g})")


if __name__ == "__main__":
    main()
