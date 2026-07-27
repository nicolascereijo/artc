import numpy as np
from numpy.typing import NDArray


def adjust_length(*ndarrays: NDArray[np.float32]) -> tuple[NDArray[np.float32], ...]:
    min_length = min(map(lambda lst: len(lst), ndarrays))
    adjusted_ndarrays = tuple(np.array(ndarr[:min_length], dtype=np.float32) for ndarr in ndarrays)

    return adjusted_ndarrays


def adjust_dimensions(*ndarrays: NDArray) -> list[NDArray]:
    """Truncate arrays along their last axis (frames) to the shortest one, as copies

    Works uniformly for 1D vectors (frames along axis 0) and 2D feature
    matrices (frames along axis 1), and preserves each array's original
    dtype (some analysis modules compare raw complex FFT values, not floats).
    """
    min_frames = min(array.shape[-1] for array in ndarrays)
    return [np.array(array[..., :min_frames]) for array in ndarrays]
