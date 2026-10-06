"""OCT volume preparation adapted from ``OCT-pose-main`` preprocessing.

The pose project clips a black/white intensity window before treating bright
voxels as signal.  This module keeps that useful, model-independent portion
of the workflow for surface extraction: it uses a robust, per-volume window
instead of hard-coded values and suppresses isolated speckle with a small 3-D
median filter.  No learned pose model or checkpoint is required.
"""

import numpy as np
from scipy.ndimage import median_filter


def prepare_volume(volume, black_percentile=1.0, white_percentile=99.8,
                   median_size=3, signal_floor=0.02):
    """Return a finite, contrast-normalized, speckle-reduced OCT volume.

    ``black_percentile`` and ``white_percentile`` mirror the black/white
    window used by OCT-pose-main while adapting to each acquisition's dynamic
    range.  ``signal_floor`` removes only near-black background after that
    normalization, so it is deliberately conservative.
    """
    if not 0 <= black_percentile < white_percentile <= 100:
        raise ValueError("black percentile must be below white percentile in [0, 100]")
    if median_size < 1 or median_size % 2 != 1:
        raise ValueError("median size must be a positive odd integer")
    if not 0 <= signal_floor < 1:
        raise ValueError("signal floor must be in [0, 1)")

    image = np.asarray(volume, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("OCT volume contains no finite voxels")
    # A manual Napari mask uses zero for excluded voxels.  Those voxels are
    # background, not intensity samples for the OCT window.  Including them
    # can collapse both percentiles to zero and reject an otherwise valid
    # manually constrained scan as a constant volume.
    window_samples = finite[finite != 0]
    if window_samples.size == 0:
        return np.zeros_like(image, dtype=np.float32), (0.0, 0.0)
    black, white = np.percentile(window_samples, (black_percentile, white_percentile))
    if white <= black:
        # A single retained intensity still defines a useful binary surface
        # against the zeroed background created by manual cleanup.
        binary = np.where(np.isfinite(image) & (image != 0), 1.0, 0.0)
        return binary.astype(np.float32, copy=False), (float(black), float(white))
    image = np.nan_to_num(image, nan=black, posinf=white, neginf=black)
    image = np.clip(image, black, white)
    image = (image - black) / (white - black)
    if median_size > 1:
        # Filtering each OCT B-scan independently avoids the large temporary
        # work arrays of a full 3-D median filter on multi-gigabyte scans.
        # It still removes the isolated in-plane speckle that becomes noisy
        # mesh vertices.
        image = median_filter(image, size=(1, median_size, median_size), mode="nearest")
    if signal_floor:
        image[image < signal_floor] = 0.0
    return image.astype(np.float32, copy=False), (float(black), float(white))
