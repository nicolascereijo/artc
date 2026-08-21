import numpy as np
from numpy.typing import NDArray


def adjust_dimensions(*ndarrays: NDArray) -> list[NDArray]:
    """Truncate arrays along their last axis (frames) to the shortest one, as copies

    Works uniformly for 1D vectors (frames along axis 0) and 2D feature
    matrices (frames along axis 1), and preserves each array's original
    dtype (some analysis modules compare raw complex FFT values, not floats).
    """
    min_frames = min(array.shape[-1] for array in ndarrays)
    return [np.array(array[..., :min_frames]) for array in ndarrays]


def check_matching_sample_rates(sample_rate1: float, sample_rate2: float) -> None:
    """Raise if two signals were sampled at different rates

    adjust_dimensions truncates feature matrices by frame index rather than
    by real time, so frame k of one signal and frame k of the other only
    represent the same instant when both signals were sampled at the same
    rate. Comparing signals at different sample rates would otherwise
    silently align mismatched instants instead of failing.

    Raises:
        ValueError: If sample_rate1 and sample_rate2 differ.
    """
    if sample_rate1 != sample_rate2:
        raise ValueError(
            f"Cannot compare signals with different sample rates "
            f"({sample_rate1} != {sample_rate2}): frame-by-frame alignment "
            f"assumes both signals share the same sample rate."
        )
