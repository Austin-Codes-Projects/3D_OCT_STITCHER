#!/usr/bin/env python3

"""
Interactive NPY-only OCT stitching pipeline.

Run:

    python3 npy_pipeline.py

Pipeline:

    vol*.npy
       |
       v
    manual alignment
       |
       v
    manual_offsets.csv
       |
       v
    manual_pairwise_transforms.npy
       |
       v
    stitch_npy_volumes.py
       |
       v
    final_stitched.npy

The final output is a 3D NPY volume that can be opened directly
with Napari.

This intentionally removes the PLY stages from the original pipeline.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------

def ask(prompt: str, default=None) -> str:
    """Ask the user for input, optionally providing a default."""
    if default is not None:
        value = input(f"{prompt} [{default}]: ").strip()
        return value if value else str(default)

    return input(f"{prompt}: ").strip()


def ask_yes_no(prompt: str, default="n") -> bool:
    """Ask a yes/no question."""
    answer = ask(prompt, default).lower()

    if answer in ("y", "yes"):
        return True

    if answer in ("n", "no"):
        return False

    print("Please enter y or n.")
    return ask_yes_no(prompt, default)


def run(script: str, *args) -> None:
    """
    Run one of the pipeline scripts using the same Python interpreter
    that launched this pipeline.
    """
    script_path = ROOT / script

    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not find pipeline script:\n{script_path}"
        )

    command = [
        sys.executable,
        str(script_path),
        *[str(arg) for arg in args],
    ]

    print()
    print("=" * 70)
    print("RUNNING")
    print("=" * 70)
    print(" ".join(command))
    print()

    subprocess.run(command, check=True)


def scan_key(path: Path) -> int | None:
    """
    Extract the numeric scan index from:

        vol1.npy
        vol2.npy
        vol10.npz

    Returns None for anything that is not a volume.
    """
    match = re.fullmatch(r"vol(\d+)\.(?:npy|npz)", path.name)

    if match is None:
        return None

    return int(match.group(1))


# ---------------------------------------------------------
# Scan selection
# ---------------------------------------------------------

def choose_scans(available: list[Path]) -> list[Path]:
    """Let the user select which volume scans to stitch."""

    print()
    print("=" * 70)
    print("AVAILABLE SCANS")
    print("=" * 70)

    for index, path in enumerate(available, start=1):
        print(f"  {index}. {path.name}")

    print()
    print("You can enter:")
    print("  - all")
    print("  - 1,2,3")
    print("  - 1-4")
    print("  - 1,2,5-7")
    print()

    selection = ask("Scans to use", "all").lower()

    if selection == "all":
        return available

    selected_indices: set[int] = set()

    for part in selection.split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            pieces = part.split("-")

            if len(pieces) != 2:
                raise ValueError(
                    f"Invalid scan range: {part}"
                )

            start = int(pieces[0])
            end = int(pieces[1])

            if start > end:
                start, end = end, start

            selected_indices.update(
                range(start, end + 1)
            )

        else:
            selected_indices.add(int(part))

    selected = []

    for index, path in enumerate(available, start=1):
        if index in selected_indices:
            selected.append(path)

    if not selected:
        raise ValueError("No scans were selected.")

    return selected


# ---------------------------------------------------------
# Existing manual-offset handling
# ---------------------------------------------------------

def clear_selected_offsets(
    filename: Path,
    scans: list[Path],
) -> None:
    """
    Remove previous offset entries belonging to the scans being
    processed.

    This preserves offsets for other scans in the same CSV.
    """

    if not filename.exists():
        return

    selected_names = {
        scan.stem
        for scan in scans
    }

    try:
        with filename.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        print(
            f"Warning: could not read existing offsets file:\n"
            f"{filename}"
        )
        return

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    if not fieldnames:
        return

    filtered = []

    for row in rows:
        row_text = " ".join(
            str(value)
            for value in row.values()
        )

        belongs_to_selected_scan = any(
            name in row_text
            for name in selected_names
        )

        if not belongs_to_selected_scan:
            filtered.append(row)

    if len(filtered) == len(rows):
        return

    with filename.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(filtered)

    print()
    print(
        f"Removed previous offset entries for selected scans "
        f"from:\n{filename}"
    )


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def main():

    print()
    print("=" * 70)
    print("3D OCT STITCHER — NPY PIPELINE")
    print("=" * 70)
    print()
    print(
        "This pipeline takes vol*.npy / vol*.npz scans and produces"
    )
    print(
        "a final stitched NPY volume for visualization in Napari."
    )
    print()

    # -----------------------------------------------------
    # Input directory
    # -----------------------------------------------------

    input_dir = Path(
        ask(
            "Folder containing vol1.npy/vol1.npz, vol2..., etc.",
            "scans",
        )
    )

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist:\n{input_dir}"
        )

    available_scans = sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file()
            and scan_key(path) is not None
        ],
        key=lambda path: scan_key(path),
    )

    if not available_scans:
        raise RuntimeError(
            f"No vol*.npy or vol*.npz files found in:\n"
            f"{input_dir}"
        )

    scans = choose_scans(available_scans)

    print()
    print("Selected scans:")

    for scan in scans:
        print(f"  {scan.name}")

    # -----------------------------------------------------
    # Output directory
    # -----------------------------------------------------

    output_dir = Path(
        ask(
            "Output folder",
            "results/final_stitch",
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Downsampling
    # -----------------------------------------------------

    scale = ask(
        "Volume downsample factor",
        "4",
    )

    try:
        scale_value = int(scale)
    except ValueError as exc:
        raise ValueError(
            "Downsample factor must be an integer."
        ) from exc

    if scale_value <= 0:
        raise ValueError(
            "Downsample factor must be greater than zero."
        )

    view_scale = ask(
        "Manual alignment view downsample factor",
        str(max(4, scale_value)),
    )

    try:
        view_scale_value = int(view_scale)
    except ValueError as exc:
        raise ValueError(
            "Manual alignment view downsample factor "
            "must be an integer."
        ) from exc

    if view_scale_value <= 0:
        raise ValueError(
            "Manual alignment view downsample factor "
            "must be greater than zero."
        )

    # -----------------------------------------------------
    # Files
    # -----------------------------------------------------

    offsets = output_dir / "manual_offsets.csv"

    manual_pairwise = (
        output_dir /
        "manual_pairwise_transforms.npy"
    )

    final_npy = (
        output_dir /
        "final_stitched.npy"
    )

    applied_transforms = (
        output_dir /
        "applied_transforms.npy"
    )

    # -----------------------------------------------------
    # Position file
    # -----------------------------------------------------

    has_positions = ask_yes_no(
        "Do you already have a position file?",
        "n",
    )

    if has_positions:

        position_file = Path(
            ask(
                "Position file path (.npy transforms or .csv manual offsets)"
            )
        )

        if not position_file.exists():
            raise FileNotFoundError(
                f"Position file does not exist:\n"
                f"{position_file}"
            )

        # ---------------------------------------------
        # Existing CSV offsets
        # ---------------------------------------------

        if position_file.suffix.lower() == ".csv":

            run(
                "manual_offsets_to_transforms.py",
                "--input-dir",
                input_dir,
                "--offsets",
                position_file,
                "--output",
                manual_pairwise,
            )

        # ---------------------------------------------
        # Existing NPY transforms
        # ---------------------------------------------

        else:

            run(
                "positions_to_pairwise_transforms.py",
                "--input",
                position_file,
                "--scan-count",
                len(scans),
                "--output",
                manual_pairwise,
            )

    else:

        # -------------------------------------------------
        # Start a new alignment
        # -------------------------------------------------

        clear_selected_offsets(
            offsets,
            scans,
        )

        has_reference = ask_yes_no(
            "Do you have a complete reference volume "
            "to align against?",
            "y",
        )

        if has_reference:

            reference = Path(
                ask(
                    "Reference volume for manual alignment",
                    "scans/Complete_surface_reference.npy",
                )
            )

            if not reference.exists():
                raise FileNotFoundError(
                    f"Reference volume does not exist:\n"
                    f"{reference}"
                )

            # ---------------------------------------------
            # Align every selected scan against reference
            # ---------------------------------------------

            for scan in scans:

                print()
                print("=" * 70)
                print(
                    f"MANUAL ALIGNMENT: {scan.name}"
                )
                print("=" * 70)

                run(
                    "manual_reference_alignment.py",
                    "--volume",
                    scan,
                    "--reference",
                    reference,
                    "--scale",
                    view_scale_value,
                    "--transform-scale",
                    scale_value,
                    "--offsets",
                    offsets,
                )

        else:

            # ---------------------------------------------
            # Sequential manual alignment
            #
            # vol1 = first/background
            # vol2 aligned against vol1
            # vol3 aligned against vol1 + vol2
            # etc.
            # ---------------------------------------------

            for index, scan in enumerate(scans):

                print()
                print("=" * 70)
                print(
                    f"MANUAL ALIGNMENT: {scan.name}"
                )
                print("=" * 70)

                args = [
                    "--volume",
                    scan,
                    "--scale",
                    view_scale_value,
                    "--transform-scale",
                    scale_value,
                    "--offsets",
                    offsets,
                ]

                if index:

                    args.extend(
                        [
                            "--background-volumes",
                            *scans[:index],
                        ]
                    )

                run(
                    "manual_reference_alignment.py",
                    *args,
                )

        # -------------------------------------------------
        # Convert manual offsets into transforms
        # -------------------------------------------------

        if not offsets.exists():
            raise FileNotFoundError(
                "Manual alignment finished, but the expected "
                f"offset file was not created:\n{offsets}"
            )

        run(
            "manual_offsets_to_transforms.py",
            "--input-dir",
            input_dir,
            "--offsets",
            offsets,
            "--output",
            manual_pairwise,
        )

    # -----------------------------------------------------
    # Verify transform file
    # -----------------------------------------------------

    if not manual_pairwise.exists():
        raise FileNotFoundError(
            "The pairwise transform file was not created:\n"
            f"{manual_pairwise}"
        )

    # -----------------------------------------------------
    # Coordinate scale
    # -----------------------------------------------------

    scale_vector = (
        f"{scale_value},"
        f"{scale_value},"
        f"{scale_value}"
    )

    # -----------------------------------------------------
    # NPY stitching
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL NPY VOLUME STITCH")
    print("=" * 70)
    print()
    print("Input scans:")

    for scan in scans:
        print(f"  {scan}")

    print()
    print(f"Pairwise transforms: {manual_pairwise}")
    print(f"Output volume:       {final_npy}")
    print(f"Coordinate scale:    {scale_vector}")
    print()

    run(
        "stitch_npy_volumes.py",
        "--input-dir",
        input_dir,
        "--initial-transforms",
        manual_pairwise,
        "--output",
        final_npy,
        "--transforms-output",
        applied_transforms,
        "--coordinate-scale",
        scale_vector,
    )

    # -----------------------------------------------------
    # Finished
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print()
    print(f"Final stitched NPY:")
    print(f"  {final_npy}")
    print()
    print(f"Transforms:")
    print(f"  {applied_transforms}")
    print()
    print("You can now open final_stitched.npy in Napari.")
    print()


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print()
        print("Pipeline cancelled by user.")
        sys.exit(1)

    except subprocess.CalledProcessError as exc:
        print()
        print("=" * 70)
        print("PIPELINE FAILED")
        print("=" * 70)
        print()
        print(
            f"A pipeline step exited with code {exc.returncode}."
        )
        sys.exit(exc.returncode)

    except Exception as exc:
        print()
        print("=" * 70)
        print("PIPELINE FAILED")
        print("=" * 70)
        print()
        print(str(exc))
        sys.exit(1)