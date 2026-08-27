#!/usr/bin/env python3
"""Interactive one-command pipeline from raw vol*.npy/.npz scans to a final PLY."""

import re
import subprocess
import sys
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def ask(prompt, default=None):
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or default


def run(script, *args):
    print("\n> " + " ".join([sys.executable, str(ROOT / script), *map(str, args)]))
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], check=True)


def scan_key(path):
    match = re.fullmatch(r"vol(\d+)\.(?:npy|npz)", path.name)
    return int(match.group(1)) if match else None


def choose_scans(available):
    """Choose every scan or confirm individual scans until a requested count."""
    while True:
        requested = ask(
            f"How many scans should be stitched? Enter all for every available scan ({len(available)})",
            "all",
        ).lower()
        if requested in {"all", "total"}:
            return available
        try:
            count = int(requested)
        except ValueError:
            print("Enter a whole number of scans, or all.")
            continue
        if not 2 <= count <= len(available):
            print(f"Choose between 2 and {len(available)} scans, or all.")
            continue

        chosen = []
        for scan in available:
            if ask(f"Include {scan.name}? (y/n)", "y").lower().startswith("y"):
                chosen.append(scan)
                if len(chosen) == count:
                    return chosen
        print(f"Only confirmed {len(chosen)} of {count} requested scans; choosing again.")


def clear_selected_offsets(filename, scans):
    """Start a manual run cleanly without retaining a different scale's poses."""
    if not filename.exists():
        return
    selected = {scan.name for scan in scans}
    with filename.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("volume") not in selected]
    with filename.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("volume", "z_vox", "y_vox", "x_vox"))
        writer.writeheader()
        writer.writerows(rows)
    print("Cleared previous manual offsets for this selection; save every scan again in this run.")


def main():
    print("=" * 68)
    print("Raw scan to final stitched PLY")
    print("This preserves the raw files and writes every result to one output folder.")
    print("=" * 68)
    input_dir = Path(ask("Folder containing vol1.npy/vol1.npz, vol2..., etc.", "scans"))
    available_scans = sorted([path for path in input_dir.iterdir() if scan_key(path) is not None], key=scan_key)
    if len(available_scans) < 2:
        raise SystemExit("Need at least two files named vol<number>.npy or vol<number>.npz")
    print(f"Found {len(available_scans)} available scans: " + ", ".join(path.name for path in available_scans))
    scans = choose_scans(available_scans)
    print(f"Selected {len(scans)} scans: " + ", ".join(path.name for path in scans))
    output_dir = Path(ask("Output folder", "results/final_stitch"))
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = ask("Volume downsample factor (must stay the same for manual placement and PLY export; normally 4)", "4")
    try:
        scale_value = int(scale)
        if scale_value < 1:
            raise ValueError
    except ValueError:
        raise SystemExit("Scale must be a positive integer")
    scale_vector = f"{scale_value},{scale_value},{scale_value}"
    view_scale = ask(
        "Manual alignment view downsample factor (does not reduce final PLY detail; normally 4)",
        str(max(4, scale_value)),
    )
    try:
        view_scale_value = int(view_scale)
        if view_scale_value < 1:
            raise ValueError
    except ValueError:
        raise SystemExit("Manual alignment view scale must be a positive integer")

    has_positions = ask("Do you already have a position file? (y/n)", "n").lower().startswith("y")
    # A selection-specific folder prevents PLYs from unselected scans (or an
    # earlier run with a different selection) from entering this stitch.
    selection_name = "_".join(path.stem for path in scans)
    raw_ply = output_dir / "individual_raw_ply" / selection_name
    clean_ply = output_dir / "individual_clean_ply" / selection_name
    offsets = output_dir / "manual_offsets.csv"
    if has_positions:
        position_file = Path(ask("Position file path (.npy transforms or .csv manual offsets)"))
        if not position_file.is_file() or position_file.suffix.lower() not in {".npy", ".csv"}:
            raise SystemExit("Position file must exist and be a .npy or .csv file")
    else:
        # Manual positions are tied to the chosen final/view scales. A fresh
        # manual run must not inherit values created at a different scale.
        clear_selected_offsets(offsets, scans)
        has_reference = ask("Do you have a complete reference volume to align against? (y/n)", "y").lower().startswith("y")
        if has_reference:
            reference = Path(ask("Reference volume for manual alignment", "scans/Complete_surface_reference.npy"))
            if not reference.is_file():
                raise SystemExit(f"Reference volume not found: {reference}")
            print("\nA manual-alignment window opens for every scan. Align green scan features to the magenta reference, Save offset, then close the window.")
            for scan in scans:
                run("manual_reference_alignment.py", "--volume", scan, "--reference", reference,
                    "--scale", view_scale_value, "--transform-scale", scale_value, "--offsets", offsets)
        else:
            print("\nBlank-canvas placement: the manual movement window opens for vol1 at global offset (0, 0, 0). After you Save offset and close that window, every later scan opens over a magenta composite of all previously saved scans, with its sliders initialized to the prior scan's global offset. Adjust from there, Save offset, then close the window to continue.")
            for index, scan in enumerate(scans):
                args = ["--volume", scan, "--scale", view_scale_value,
                        "--transform-scale", scale_value, "--offsets", offsets]
                if index:
                    args.extend(["--background-volumes", *scans[:index]])
                run("manual_reference_alignment.py", *args)
        position_file = offsets

    run("export_scans_to_ply.py", "--input-dir", input_dir, "--output-dir", raw_ply,
        "--scale", scale_value, "--volumes", *scans)
    region = ask("Corner artifact region to remove (Enter uses reference-cleaned rule; type none to skip)",
                 "y:1890:max,x:min:713")
    for scan in scans:
        source = raw_ply / f"{scan.stem}.ply"
        args = ["--input", source, "--output", clean_ply / source.name]
        if region.lower() != "none":
            args.extend(["--remove-region", region])
        run("clean_ply_panels.py", *args)

    manual_pairwise = output_dir / "manual_pairwise_transforms.npy"
    if position_file.suffix.lower() == ".csv":
        run("manual_offsets_to_transforms.py", "--input-dir", clean_ply, "--offsets", position_file, "--output", manual_pairwise)
    else:
        run("positions_to_pairwise_transforms.py", "--input", position_file, "--scan-count", len(scans), "--output", manual_pairwise)
    panel_transforms = output_dir / "first_panel_reference_transforms.npy"
    run("align_first_panel_transforms.py", "--input-dir", clean_ply, "--initial-transforms", manual_pairwise,
        "--output", panel_transforms, "--axis", "x", "--coordinate-scale", scale_vector)
    merge_distance = ask("Overlap merge distance in PLY units (1 is conservative; 0 disables)", "1")
    try:
        float(merge_distance)
    except ValueError:
        raise SystemExit("Merge distance must be a number")
    final_ply = output_dir / "final_stitched.ply"
    run("stitch_ply_meshes.py", "--input-dir", clean_ply, "--output", final_ply,
        "--transforms-output", output_dir / "applied_transforms.npy", "--initial-transforms", panel_transforms,
        "--coordinate-scale", scale_vector, "--skip-icp", "--merge-distance", merge_distance)
    print(f"\nFinished. Open: {final_ply}")


if __name__ == "__main__":
    main()
