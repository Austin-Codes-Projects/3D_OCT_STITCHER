import numpy as np


# def load_volume(filename: str, scale: int = 2):
#     """.npy file loader """
#     vol = np.load(filename)
#     if scale > 1:
#         vol = vol[::scale, ::scale, ::scale]
#     return vol.astype(np.float32)

def load_volume(filename: str, scale: int = 2):
    """Load a .npy or single-array .npz volume and remove extra dimensions."""
    # Raw OCT arrays are several gigabytes.  Mapping .npy input prevents the
    # full acquisition from being copied into RAM before downsampling.
    loaded = np.load(filename, mmap_mode="r" if str(filename).lower().endswith(".npy") else None)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        keys = list(loaded.files)
        preferred = next((key for key in ("volume", "stitched", "arr_0") if key in keys), None)
        if preferred is None and len(keys) == 1:
            preferred = keys[0]
        if preferred is None:
            loaded.close()
            raise ValueError(f"{filename} has multiple arrays {keys}; name the volume 'volume' or provide one array")
        vol = loaded[preferred]
        loaded.close()
    else:
        vol = loaded

    # Remove any size-1 dimensions (e.g. (1, Z, Y, X) → (Z, Y, X))
    vol = np.squeeze(vol)

    if scale > 1:
        vol = vol[::scale, ::scale, ::scale]

    return vol.astype(np.float32)
