#!/usr/bin/env python3

"""
Stitch NPY volumes into a single NPY volume.

Expected input:

    input_dir/
        vol1.npy
        vol2.npy
        vol3.npy
        ...

    pairwise transforms:
        (N - 1, 4, 4)

Output:

    final_stitched.npy

The transform convention matches stitch_ply_meshes.py:

    T_0 = I

    T_i = T_(i-1) @ inv(P_(i-1))

where P_i is the pairwise transform between adjacent scans.

Volumes are treated as:

    (Z, Y, X)

The resulting stitched volume is also:

    (Z, Y, X)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform


# =========================================================
# Filename utilities
# =========================================================

def volume_key(path: Path) -> int:
    """
    Extract the numeric part from:

        vol1.npy
        vol2.npy
        vol10.npy
    """

    match = re.fullmatch(r"vol(\d+)\.npy", path.name)

    if match is None:
        raise ValueError(
            f"Unexpected volume filename: {path.name}"
        )

    return int(match.group(1))


# =========================================================
# Volume loading
# =========================================================

def load_volume(path: Path) -> np.ndarray:
    """
    Load an NPY volume.

    Supported shapes:

        (Z, Y, X)

    or:

        (1, Z, Y, X)
    """

    volume = np.load(path)

    print(
        f"Loaded {path.name}: "
        f"shape={volume.shape}, "
        f"dtype={volume.dtype}"
    )

    # Handle (1, Z, Y, X)
    if volume.ndim == 4:

        if volume.shape[0] != 1:
            raise ValueError(
                f"{path.name} has shape {volume.shape}. "
                "Expected (1, Z, Y, X) for 4D volumes."
            )

        volume = volume[0]

    # Must ultimately be 3D
    if volume.ndim != 3:
        raise ValueError(
            f"{path.name} has shape {volume.shape}. "
            "Expected a 3D volume."
        )

    return volume


# =========================================================
# Homogeneous transforms
# =========================================================

def make_corners(shape: tuple[int, int, int]) -> np.ndarray:
    """
    Return the eight corners of a volume.

    Coordinates are:

        [Z, Y, X]

    Returned array:

        (8, 3)
    """

    z_max = float(shape[0] - 1)
    y_max = float(shape[1] - 1)
    x_max = float(shape[2] - 1)

    return np.array(
        [
            [0.0, 0.0, 0.0],
            [z_max, 0.0, 0.0],
            [0.0, y_max, 0.0],
            [0.0, 0.0, x_max],
            [z_max, y_max, 0.0],
            [z_max, 0.0, x_max],
            [0.0, y_max, x_max],
            [z_max, y_max, x_max],
        ],
        dtype=np.float64,
    )


def apply_transform(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """
    Apply a 4x4 homogeneous transform to Nx3 points.
    """

    points = np.asarray(points, dtype=np.float64)

    ones = np.ones(
        (points.shape[0], 1),
        dtype=np.float64,
    )

    homogeneous = np.concatenate(
        [points, ones],
        axis=1,
    )

    transformed = homogeneous @ transform.T

    return transformed[:, :3]


# =========================================================
# Coordinate scale
# =========================================================

def make_coordinate_scale(
    scale: tuple[float, float, float],
) -> np.ndarray:
    """
    Construct:

        diag(scale_z, scale_y, scale_x, 1)

    This follows the same idea used in stitch_ply_meshes.py.
    """

    matrix = np.eye(4, dtype=np.float64)

    matrix[0, 0] = scale[0]
    matrix[1, 1] = scale[1]
    matrix[2, 2] = scale[2]

    return matrix


def convert_pairwise_to_volume_coordinates(
    pairwise: np.ndarray,
    coordinate_scale: tuple[float, float, float],
) -> np.ndarray:
    """
    Convert transforms using the same coordinate-change
    convention as stitch_ply_meshes.py.

    In the PLY stitcher:

        C @ P @ inv(C)

    is used.

    We reproduce that here so the same transform files
    remain usable.
    """

    scale_matrix = make_coordinate_scale(
        coordinate_scale
    )

    inverse_scale = np.linalg.inv(scale_matrix)

    converted = np.asarray(
        [
            scale_matrix @ transform @ inverse_scale
            for transform in pairwise
        ],
        dtype=np.float64,
    )

    return converted


# =========================================================
# Global transform calculation
# =========================================================

def build_global_transforms(
    pairwise: np.ndarray,
) -> list[np.ndarray]:
    """
    Convert adjacent pairwise transforms into transforms
    relative to volume 0.

    Same convention as stitch_ply_meshes.py:

        T_0 = I

        T_i = T_(i-1) @ inv(P_(i-1))
    """

    pairwise = np.asarray(
        pairwise,
        dtype=np.float64,
    )

    if pairwise.ndim != 3:
        raise ValueError(
            "Pairwise transforms must be a 3D array."
        )

    if pairwise.shape[1:] != (4, 4):
        raise ValueError(
            "Pairwise transforms must have shape "
            "(N-1, 4, 4). "
            f"Got {pairwise.shape}."
        )

    transforms: list[np.ndarray] = [
        np.eye(4, dtype=np.float64)
    ]

    for pairwise_transform in pairwise:

        next_transform = (
            transforms[-1]
            @ np.linalg.inv(pairwise_transform)
        )

        transforms.append(next_transform)

    return transforms


# =========================================================
# Bounding box
# =========================================================

def calculate_output_bounds(
    volumes: list[np.ndarray],
    transforms: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """
    Calculate the bounding box containing all transformed
    volumes.
    """

    if len(volumes) != len(transforms):
        raise ValueError(
            "Number of volumes and transforms must match."
        )

    all_points = []

    for volume, transform in zip(
        volumes,
        transforms,
    ):

        corners = make_corners(volume.shape)

        transformed_corners = apply_transform(
            corners,
            transform,
        )

        all_points.append(
            transformed_corners
        )

    all_points = np.vstack(all_points)

    minimum = np.floor(
        np.min(all_points, axis=0)
    ).astype(np.int64)

    maximum = np.ceil(
        np.max(all_points, axis=0)
    ).astype(np.int64)

    output_shape = (
        maximum - minimum + 1
    ).astype(np.int64)

    output_shape = tuple(
        int(value)
        for value in output_shape
    )

    return minimum, maximum, output_shape


# =========================================================
# Resampling
# =========================================================

def transform_volume(
    volume: np.ndarray,
    transform: np.ndarray,
    output_shape: tuple[int, int, int],
    output_origin: np.ndarray,
    interpolation_order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform a volume into the common output coordinate
    system.

    Returns:

        transformed_volume
        transformed_weight

    The weight volume tells us which output voxels are
    actually covered by this input volume.
    """

    # -----------------------------------------------------
    # Convert global coordinates into output-array
    # coordinates.
    #
    # If the global minimum is:
    #
    #     [z_min, y_min, x_min]
    #
    # then:
    #
    #     output_coordinate =
    #         global_coordinate - output_origin
    # -----------------------------------------------------

    origin_transform = np.eye(
        4,
        dtype=np.float64,
    )

    origin_transform[:3, 3] = -output_origin

    forward_transform = (
        origin_transform @ transform
    )

    # scipy.ndimage.affine_transform expects:
    #
    #     output -> input
    #
    # while our transform is:
    #
    #     input -> output
    #
    inverse_transform = np.linalg.inv(
        forward_transform
    )

    matrix = inverse_transform[:3, :3]

    offset = inverse_transform[:3, 3]

    # -----------------------------------------------------
    # Transform actual image data
    # -----------------------------------------------------

    transformed = affine_transform(
        volume.astype(np.float32, copy=False),
        matrix=matrix,
        offset=offset,
        output_shape=output_shape,
        order=interpolation_order,
        mode="constant",
        cval=0.0,
        prefilter=(
            interpolation_order > 1
        ),
    )

    # -----------------------------------------------------
    # Transform a binary coverage mask
    # -----------------------------------------------------

    mask = np.ones(
        volume.shape,
        dtype=np.float32,
    )

    transformed_mask = affine_transform(
        mask,
        matrix=matrix,
        offset=offset,
        output_shape=output_shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    transformed_mask = np.clip(
        transformed_mask,
        0.0,
        1.0,
    )

    return transformed, transformed_mask


# =========================================================
# Stitching
# =========================================================

def stitch_volumes(
    volumes: list[np.ndarray],
    transforms: list[np.ndarray],
    output_origin: np.ndarray,
    output_shape: tuple[int, int, int],
    interpolation_order: int = 1,
) -> np.ndarray:
    """
    Transform all volumes into the common coordinate system
    and blend overlapping regions by weighted averaging.
    """

    print()
    print("=" * 60)
    print("STITCHING")
    print("=" * 60)

    print(
        f"Output origin: {output_origin}"
    )

    print(
        f"Output shape:  {output_shape}"
    )

    # -----------------------------------------------------
    # Accumulator
    # -----------------------------------------------------

    accumulator = np.zeros(
        output_shape,
        dtype=np.float64,
    )

    weights = np.zeros(
        output_shape,
        dtype=np.float64,
    )

    # -----------------------------------------------------
    # Process every volume
    # -----------------------------------------------------

    for index, (
        volume,
        transform,
    ) in enumerate(
        zip(volumes, transforms)
    ):

        print()
        print(
            f"[{index + 1}/{len(volumes)}] "
            f"Transforming vol{index + 1}"
        )

        print(
            f"Input shape: {volume.shape}"
        )

        transformed, transformed_mask = (
            transform_volume(
                volume=volume,
                transform=transform,
                output_shape=output_shape,
                output_origin=output_origin,
                interpolation_order=interpolation_order,
            )
        )

        # -------------------------------------------------
        # Accumulate weighted data
        # -------------------------------------------------

        accumulator += (
            transformed.astype(np.float64)
            * transformed_mask.astype(np.float64)
        )

        weights += (
            transformed_mask.astype(np.float64)
        )

        print(
            f"  coverage: "
            f"{np.count_nonzero(transformed_mask):,} "
            f"voxels"
        )

        del transformed
        del transformed_mask

    # -----------------------------------------------------
    # Normalize overlaps
    # -----------------------------------------------------

    result = np.zeros_like(
        accumulator,
        dtype=np.float64,
    )

    valid = weights > 1e-8

    result[valid] = (
        accumulator[valid]
        / weights[valid]
    )

    print()
    print(
        f"Covered output voxels: "
        f"{np.count_nonzero(valid):,}"
    )

    print(
        f"Total output voxels: "
        f"{result.size:,}"
    )

    return result


# =========================================================
# Output dtype
# =========================================================

def restore_dtype(
    result: np.ndarray,
    original_dtype: np.dtype,
) -> np.ndarray:
    """
    Convert the floating-point stitched result back to the
    input dtype when possible.
    """

    if np.issubdtype(
        original_dtype,
        np.integer,
    ):

        info = np.iinfo(
            original_dtype
        )

        result = np.rint(result)

        result = np.clip(
            result,
            info.min,
            info.max,
        )

        return result.astype(
            original_dtype
        )

    if np.issubdtype(
        original_dtype,
        np.floating,
    ):

        return result.astype(
            original_dtype
        )

    # Safe fallback
    return result.astype(
        np.float32
    )


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Stitch vol*.npy volumes into "
            "a single final NPY volume."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing vol1.npy, "
            "vol2.npy, etc."
        ),
    )

    parser.add_argument(
        "--initial-transforms",
        required=True,
        type=Path,
        help=(
            "Pairwise transform .npy file "
            "with shape (N-1,4,4)."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "Output stitched NPY path."
        ),
    )

    parser.add_argument(
        "--transforms-output",
        type=Path,
        default=None,
        help=(
            "Optional path for saving global "
            "transforms."
        ),
    )

    parser.add_argument(
        "--coordinate-scale",
        type=str,
        default="1,1,1",
        help=(
            "Z,Y,X coordinate scale. "
            "Example: 4,4,4"
        ),
    )

    parser.add_argument(
        "--order",
        type=int,
        choices=[0, 1, 3],
        default=1,
        help=(
            "Interpolation order: "
            "0=nearest, 1=linear, 3=cubic. "
            "Default: 1"
        ),
    )

    args = parser.parse_args()

    # =====================================================
    # Parse coordinate scale
    # =====================================================

    try:

        coordinate_scale = tuple(
            float(value.strip())
            for value in args.coordinate_scale.split(",")
        )

    except ValueError as exc:

        raise ValueError(
            "--coordinate-scale must look like "
            "4,4,4"
        ) from exc

    if len(coordinate_scale) != 3:
        raise ValueError(
            "--coordinate-scale must contain "
            "exactly three values: Z,Y,X"
        )

    if any(
        value <= 0
        for value in coordinate_scale
    ):
        raise ValueError(
            "Coordinate-scale values must "
            "be greater than zero."
        )

    # =====================================================
    # Locate volumes
    # =====================================================

    files = sorted(
        args.input_dir.glob("vol*.npy"),
        key=volume_key,
    )

    if not files:

        raise RuntimeError(
            f"No vol*.npy files found in "
            f"{args.input_dir}"
        )

    print("=" * 60)
    print("NPY VOLUME STITCHER")
    print("=" * 60)

    print()
    print("Input volumes:")

    for path in files:
        print(f"  {path}")

    # =====================================================
    # Load volumes
    # =====================================================

    volumes = []

    for path in files:

        volume = load_volume(path)

        volumes.append(volume)

    # Make sure all volumes have compatible dimensions.
    # They don't need to have identical shapes, but they
    # must all be 3D.
    for index, volume in enumerate(volumes):

        if volume.ndim != 3:
            raise ValueError(
                f"Volume {index} is not 3D."
            )

    # =====================================================
    # Load pairwise transforms
    # =====================================================

    print()
    print(
        f"Loading transforms: "
        f"{args.initial_transforms}"
    )

    pairwise = np.load(
        args.initial_transforms
    )

    pairwise = np.asarray(
        pairwise,
        dtype=np.float64,
    )

    expected_shape = (
        len(volumes) - 1,
        4,
        4,
    )

    if pairwise.shape != expected_shape:

        raise ValueError(
            "Incorrect transform shape.\n"
            f"Expected: {expected_shape}\n"
            f"Got:      {pairwise.shape}"
        )

    print(
        f"Pairwise transform shape: "
        f"{pairwise.shape}"
    )

    # =====================================================
    # Convert coordinate system
    # =====================================================

    print()
    print(
        "Coordinate scale:",
        coordinate_scale,
    )

    pairwise_volume = (
        convert_pairwise_to_volume_coordinates(
            pairwise,
            coordinate_scale,
        )
    )

    # =====================================================
    # Build global transforms
    # =====================================================

    print()
    print("=" * 60)
    print("GLOBAL TRANSFORMS")
    print("=" * 60)

    transforms = build_global_transforms(
        pairwise_volume
    )

    for index, transform in enumerate(
        transforms
    ):

        print()
        print(
            f"Volume {index + 1}:"
        )

        print(transform)

    # =====================================================
    # Save global transforms
    # =====================================================

    if args.transforms_output is not None:

        args.transforms_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            args.transforms_output,
            np.asarray(
                transforms,
                dtype=np.float64,
            ),
        )

        print()
        print(
            f"Saved global transforms to:"
            f"\n{args.transforms_output}"
        )

    # =====================================================
    # Calculate output bounding box
    # =====================================================

    print()
    print("=" * 60)
    print("OUTPUT BOUNDING BOX")
    print("=" * 60)

    minimum, maximum, output_shape = (
        calculate_output_bounds(
            volumes,
            transforms,
        )
    )

    print()
    print(
        "Minimum:",
        minimum,
    )

    print(
        "Maximum:",
        maximum,
    )

    print(
        "Output shape:",
        output_shape,
    )

    # =====================================================
    # Check for unreasonable output
    # =====================================================

    output_voxels = int(
        np.prod(output_shape)
    )

    print()
    print(
        f"Output voxel count: "
        f"{output_voxels:,}"
    )

    estimated_gb = (
        output_voxels * 4
        / (1024 ** 3)
    )

    print(
        f"Estimated float32 memory: "
        f"{estimated_gb:.2f} GB"
    )

    if estimated_gb > 32:

        raise MemoryError(
            "The requested stitched volume would "
            f"require approximately {estimated_gb:.1f} GB "
            "for one float32 array. "
            "Check the transforms/bounding box before "
            "continuing."
        )

    # =====================================================
    # Stitch
    # =====================================================

    result = stitch_volumes(
        volumes=volumes,
        transforms=transforms,
        output_origin=minimum,
        output_shape=output_shape,
        interpolation_order=args.order,
    )

    # =====================================================
    # Restore dtype
    # =====================================================

    result = restore_dtype(
        result,
        volumes[0].dtype,
    )

    # =====================================================
    # Save
    # =====================================================

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        args.output,
        result,
    )

    # =====================================================
    # Final information
    # =====================================================

    print()
    print("=" * 60)
    print("FINISHED")
    print("=" * 60)

    print()
    print(
        f"Final volume: {args.output}"
    )

    print(
        f"Shape:        {result.shape}"
    )

    print(
        f"Dtype:        {result.dtype}"
    )

    print()
    print(
        "You can now load this file directly "
        "with NumPy or Napari."
    )


if __name__ == "__main__":
    main()