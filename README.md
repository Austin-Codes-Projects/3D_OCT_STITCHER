# Raw scans to final stitched PLY

This project has one supported workflow: convert raw `vol*.npy` or
`vol*.npz` scans into cleaned individual PLYs and a final overlap-welded
stitched PLY.

Run this command from the repository root:

```bash
python3 conversion_code/raw_scans_to_final_ply.py
```

The interactive menu asks for:

1. The folder containing `vol1`, `vol2`, and further raw scans.
2. How many scans to stitch. Enter `all` to use every available `vol*` scan,
   or enter a number and confirm/decline individual scans until that number is
   selected.
3. An output folder.
4. The final PLY downsample factor and, separately, the manual-alignment view
   downsample factor. Use `1` for maximum final-mesh detail and `4` for a
   responsive alignment window; offsets are converted automatically.
5. A position file, if available:
   - `.npy` containing global `(N, 4, 4)` or adjacent-pair `(N-1, 4, 4)` transforms.
   - `.csv` containing manual offsets from a previous alignment run.
6. Or, if no position file is available, choose manual placement:
   - With a complete reference volume, each scan is aligned to that reference.
   - Without one, `vol1` opens on a blank canvas the same size as the scan.
     Each later scan opens one at a time over a composite of every earlier
     saved scan, so its placement uses the offsets already set. Set Z/Y/X,
     click **Save offset**, and close the window before continuing.
   In either mode, click **Save offset** before closing a scan's window.
7. The panel-corner region and overlap merge distance.

The pipeline exports each scan to PLY, removes the reference-style corner
artifact, aligns the first broad X-reference panel between scans, averages
overlapping cross-scan vertices, removes duplicate faces, and writes
`final_stitched.ply` in your chosen output folder.

The merge step reduces duplicated overlap; it does not create or invent new
surface detail.
