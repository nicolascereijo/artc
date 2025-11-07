import numpy as np
from numpy.typing import NDArray


def adjust_length(*ndarrays: NDArray[np.float32]) -> tuple[NDArray[np.float32], ...]:
    min_length = min(map(lambda lst: len(lst), ndarrays))
    adjusted_ndarrays = tuple(np.array(ndarr[:min_length], dtype=np.float32) for ndarr in ndarrays)

    return adjusted_ndarrays


def adjust_dimensions(*ndarrays: NDArray[np.float32]) -> list[NDArray[np.float32]]:
    min_frames = min(array.size for array in ndarrays)
    return [array[:min_frames] for array in ndarrays]


def normalize_btw_0_1(*iterables: NDArray[np.float32]) -> tuple[list, ...]:
    all_values = np.hstack(iterables)
    min_val = np.min(all_values)
    max_val = np.max(all_values)

    normalized_ndarrays = tuple(map(lambda lst: list(
        map(lambda x: (x - min_val) / (max_val - min_val), lst)), iterables))

    return normalized_ndarrays
