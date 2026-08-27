#!/usr/bin/env python3
"""Validate position transforms and convert global poses to stitcher pairwise poses."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help=".npy transforms shaped (N,4,4) global poses or (N-1,4,4) pairwise poses")
    parser.add_argument("--scan-count", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    transforms = np.load(args.input)
    if transforms.shape == (args.scan_count - 1, 4, 4):
        pairwise = transforms
        kind = "pairwise"
    elif transforms.shape == (args.scan_count, 4, 4):
        # stitch_ply_meshes applies inverse(pairwise[i]) after the prior pose.
        pairwise = np.asarray([np.linalg.inv(next_pose) @ previous_pose
                               for previous_pose, next_pose in zip(transforms[:-1], transforms[1:])])
        kind = "global poses converted to pairwise"
    else:
        parser.error(f"Expected ({args.scan_count},4,4) global or ({args.scan_count - 1},4,4) pairwise transforms; got {transforms.shape}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, pairwise)
    print(f"Saved {args.output} from {kind} transforms")


if __name__ == "__main__":
    main()
