#!/usr/bin/env python3
"""Convert manual-reference offsets into pairwise PLY-stitcher transforms.

The offsets written by ``manual_reference_alignment.py`` place every scan in
one common reference coordinate system.  This script derives the relative
4x4 transforms required by ``stitch_ply_meshes.py`` without blending any
volumes or creating an NPZ file.
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np


def volume_key(path):
    match = re.fullmatch(r"vol(\d+)\.ply", path.name)
    if match is None:
        raise ValueError(f"Expected a filename such as vol1.ply, got {path.name}")
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True,
                        help="directory containing the PLY scans to stitch (vol1.ply, vol2.ply, ...)")
    parser.add_argument("--offsets", default="scans/manual_offsets.csv",
                        help="CSV written by manual_reference_alignment.py")
    parser.add_argument("--output", default="results/manual_ply_initial_transforms.npy",
                        help="output pairwise 4x4 transform file")
    args = parser.parse_args()

    files = sorted(Path(args.input_dir).glob("vol*.ply"), key=volume_key)
    if len(files) < 2:
        parser.error(f"Need at least two vol*.ply files in {args.input_dir}")

    with Path(args.offsets).open(newline="") as handle:
        rows = {
            Path(row["volume"]).stem: np.array([float(row["z_vox"]), float(row["y_vox"]), float(row["x_vox"])])
            for row in csv.DictReader(handle)
        }
    names = [path.stem for path in files]
    missing = [name for name in names if name not in rows]
    if missing:
        parser.error(f"Missing manual offsets for: {', '.join(missing)}")

    origins = np.asarray([rows[name] for name in names])
    transforms = []
    for first, second in zip(origins[:-1], origins[1:]):
        transform = np.eye(4)
        # Map coordinates in the first scan into the next scan's coordinates.
        # stitch_ply_meshes.py inverts this to move each later scan into the
        # accumulated mesh coordinate system.
        transform[:3, 3] = first - second
        transforms.append(transform)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.asarray(transforms))
    print("Manual origins (scaled voxels, Z/Y/X):")
    for name, origin in zip(names, origins):
        print(f"  {name}: {tuple(origin)}")
    print(f"Saved {output} ({len(transforms)} pairwise transforms)")


if __name__ == "__main__":
    main()
