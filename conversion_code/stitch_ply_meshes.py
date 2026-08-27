#!/usr/bin/env python3
"""Sequential ICP registration and merge for individual PLY meshes.

The output reuses the input mesh geometry; unlike the volume stitcher, it does
not blend voxels or reconstruct a surface from a stitched NPZ volume.
"""

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from imes4d.mesh_export import save_ply


VERTEX_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])
FACE_DTYPE = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])


def mesh_key(path):
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


def load_ply(path):
    """Read the binary little-endian PLY format written by mesh_export.py."""
    with path.open("rb") as handle:
        header = b""
        while b"end_header\n" not in header:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} has no PLY header")
            header += line
        lines = header.decode("ascii").splitlines()
        if "format binary_little_endian 1.0" not in lines:
            raise ValueError(f"{path} must be binary_little_endian PLY")
        counts = {line.split()[1]: int(line.split()[2]) for line in lines if line.startswith("element ")}
        vertices = np.fromfile(handle, VERTEX_DTYPE, counts.get("vertex", 0))
        raw_faces = np.fromfile(handle, FACE_DTYPE, counts.get("face", 0))
    return (np.column_stack((vertices["x"], vertices["y"], vertices["z"])),
            np.column_stack((vertices["nx"], vertices["ny"], vertices["nz"])),
            raw_faces["indices"])


def apply_transform(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def rigid_transform(source, target):
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def icp(source, target, initial, iterations, keep_fraction):
    """Robust point-to-point ICP.  A good initial transform is essential."""
    transform = initial.copy()
    tree = cKDTree(target)
    for _ in range(iterations):
        moved = apply_transform(source, transform)
        distances, indices = tree.query(moved, workers=-1)
        cutoff = np.quantile(distances, keep_fraction)
        keep = distances <= cutoff
        if keep.sum() < 6:
            break
        update = rigid_transform(moved[keep], target[indices[keep]])
        transform = update @ transform
        if np.linalg.norm(update[:3, 3]) < 1e-4 and np.linalg.norm(update[:3, :3] - np.eye(3)) < 1e-5:
            break
    moved = apply_transform(source, transform)
    distances, _ = tree.query(moved, workers=-1)
    return transform, float(np.median(distances))


def sample(points, count, rng):
    if len(points) <= count:
        return points
    return points[rng.choice(len(points), count, replace=False)]


def weld_overlaps(vertex_sets, normal_sets, faces, distance, normal_dot):
    """Average nearby vertices from different input scans and remap faces."""
    vertices = np.vstack(vertex_sets)
    normals = np.vstack(normal_sets)
    offsets = np.cumsum([0, *(len(points) for points in vertex_sets[:-1])])
    parent = np.arange(len(vertices), dtype=np.int32)
    earlier_vertices, earlier_normals = [], []
    merged_pairs = 0
    for index, (points, point_normals, offset) in enumerate(zip(vertex_sets, normal_sets, offsets)):
        if earlier_vertices:
            previous_vertices = np.vstack(earlier_vertices)
            previous_normals = np.vstack(earlier_normals)
            distances, nearest = cKDTree(previous_vertices).query(points, distance_upper_bound=distance, workers=-1)
            matched = np.isfinite(distances)
            if normal_dot > -1 and matched.any():
                dot = np.einsum("ij,ij->i", point_normals[matched], previous_normals[nearest[matched]])
                matched_indices = np.flatnonzero(matched)
                matched[matched_indices[dot < normal_dot]] = False
            # Each new vertex is linked only to an earlier scan, so assigning
            # its parent directly avoids expensive Python-level union loops.
            parent[offset + np.flatnonzero(matched)] = nearest[matched]
            merged_pairs += int(matched.sum())
        earlier_vertices.append(points)
        earlier_normals.append(point_normals)

    roots = parent.copy()
    while True:
        next_roots = roots[roots]
        if np.array_equal(next_roots, roots):
            break
        roots = next_roots
    _, inverse = np.unique(roots, return_inverse=True)
    count = np.bincount(inverse)
    averaged_vertices = np.zeros((len(count), 3), dtype=np.float64)
    averaged_normals = np.zeros((len(count), 3), dtype=np.float64)
    np.add.at(averaged_vertices, inverse, vertices)
    np.add.at(averaged_normals, inverse, normals)
    averaged_vertices /= count[:, None]
    lengths = np.linalg.norm(averaged_normals, axis=1)
    averaged_normals[lengths > 0] /= lengths[lengths > 0, None]

    remapped_faces = inverse[faces]
    non_degenerate = ((remapped_faces[:, 0] != remapped_faces[:, 1]) &
                      (remapped_faces[:, 0] != remapped_faces[:, 2]) &
                      (remapped_faces[:, 1] != remapped_faces[:, 2]))
    remapped_faces = remapped_faces[non_degenerate]
    canonical = np.sort(remapped_faces, axis=1)
    _, unique_indices = np.unique(canonical, axis=0, return_index=True)
    remapped_faces = remapped_faces[np.sort(unique_indices)]
    print(f"Welded {merged_pairs:,} cross-scan vertices into {len(averaged_vertices):,} vertices; "
          f"kept {len(remapped_faces):,} non-duplicate faces")
    return averaged_vertices.astype(np.float32), averaged_normals.astype(np.float32), remapped_faces.astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description="Register individual OCT PLY meshes with sequential ICP.")
    parser.add_argument("--input-dir", default="results/individual_ply")
    parser.add_argument("--output", default="results/stitched_mesh.ply")
    parser.add_argument("--transforms-output", default="results/ply_icp_transforms.npy")
    parser.add_argument("--initial-transforms",
                        help="optional pairwise 4x4 transforms from an external alignment process")
    parser.add_argument("--coordinate-scale", type=parse_scale, default=(1, 1, 1),
                        help="Z,Y,X scale from transform coordinates to PLY coordinates")
    parser.add_argument("--sample-points", type=int, default=20000)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--keep-fraction", type=float, default=0.7, help="nearest-neighbour pairs retained by ICP")
    parser.add_argument("--skip-icp", action="store_true",
                        help="apply --initial-transforms exactly without ICP refinement")
    parser.add_argument("--merge-distance", type=float, default=0,
                        help="average overlapping vertices from different scans within this PLY distance (default: disabled)")
    parser.add_argument("--merge-normal-dot", type=float, default=0.8,
                        help="minimum normal dot product for overlap averaging; use -1 to ignore normals (default: 0.8)")
    args = parser.parse_args()
    if not 0 < args.keep_fraction <= 1:
        parser.error("--keep-fraction must be in (0, 1]")
    if args.merge_distance < 0 or not -1 <= args.merge_normal_dot <= 1:
        parser.error("--merge-distance must be non-negative and --merge-normal-dot must be in [-1, 1]")

    files = sorted(Path(args.input_dir).glob("vol*.ply"), key=mesh_key)
    if len(files) < 2:
        parser.error("Need at least two vol*.ply files")
    meshes = [load_ply(path) for path in files]
    pairwise = np.load(args.initial_transforms) if args.initial_transforms else None
    if pairwise is not None and pairwise.shape != (len(meshes) - 1, 4, 4):
        parser.error("--initial-transforms must have one 4x4 matrix per adjacent mesh pair")
    if args.skip_icp and pairwise is None:
        parser.error("--skip-icp requires --initial-transforms")
    # The volume stitcher records transforms in downsampled voxel coordinates.
    # Conjugate them into the PLY coordinate system (which may be physical).
    coordinate_change = np.eye(4)
    coordinate_change[np.diag_indices(3)] = args.coordinate_scale
    if pairwise is not None:
        pairwise = np.asarray([coordinate_change @ item @ np.linalg.inv(coordinate_change)
                               for item in pairwise])
    if pairwise is None:
        print("WARNING: no initial transforms; using centroid alignment. "
              "For small overlaps, provide --initial-transforms from a PLY alignment process.")

    rng = np.random.default_rng(0)
    transforms = [np.eye(4)]
    merged_points = meshes[0][0].copy()
    for index, (points, _, _) in enumerate(meshes[1:], start=1):
        if pairwise is not None:
            initial = transforms[-1] @ np.linalg.inv(pairwise[index - 1])
        else:
            initial = np.eye(4)
            initial[:3, 3] = merged_points.mean(axis=0) - points.mean(axis=0)
        if args.skip_icp:
            transform, error = initial, None
        else:
            transform, error = icp(sample(points, args.sample_points, rng), sample(merged_points, args.sample_points, rng),
                                   initial, args.iterations, args.keep_fraction)
        transforms.append(transform)
        merged_points = np.vstack((merged_points, apply_transform(points, transform)))
        if error is None:
            print(f"{files[index].name}: applied supplied transform without ICP refinement")
        else:
            print(f"{files[index].name}: median ICP distance = {error:.4g}")

    all_vertices, all_normals, all_faces = [], [], []
    offset = 0
    for (vertices, normals, faces), transform in zip(meshes, transforms):
        all_vertices.append(apply_transform(vertices, transform))
        all_normals.append(normals @ transform[:3, :3].T)
        all_faces.append(faces + offset)
        offset += len(vertices)
    all_vertices = np.vstack(all_vertices)
    all_normals = np.vstack(all_normals)
    all_faces = np.vstack(all_faces)
    if args.merge_distance:
        all_vertices, all_normals, all_faces = weld_overlaps(
            [apply_transform(vertices, transform) for (vertices, _, _), transform in zip(meshes, transforms)],
            [normals @ transform[:3, :3].T for (_, normals, _), transform in zip(meshes, transforms)],
            all_faces, args.merge_distance, args.merge_normal_dot,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_ply(all_vertices, all_faces, all_normals, output)
    Path(args.transforms_output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.transforms_output, np.asarray(transforms))
    print(f"Saved {output}; open it in MeshLab. Saved refined transforms to {args.transforms_output}")


if __name__ == "__main__":
    main()
