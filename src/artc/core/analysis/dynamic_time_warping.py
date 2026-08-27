import numpy as np
import numpy.typing as npt
from librosa.sequence import dtw

import artc.core.configurations as config

from .mfcc import calculate_mfcc


def compare_two_dtw(
    audio_signal1: npt.NDArray[np.float32],
    audio_signal2: npt.NDArray[np.float32],
    sample_rate1: float,
    sample_rate2: float,
    /,
    *,
    n_fft: int | None = None,
) -> float:
    """Computes DTW based similarity between two audio signals by aligning
    their MFCC feature sequences, computed via 'calculate_mfcc', and returning
    a normalized similarity score.

    Args:
        audio_signal1: First audio time series array.
        audio_signal2: Second audio time series array.
        sample_rate1: Sampling rate, in Hz, of the first signal.
        sample_rate2: Sampling rate, in Hz, of the second signal.
        n_fft: FFT window length used for MFCC extraction. Defaults to the
            'dynamic_time_warping' entry of '[metric.window_parameter]' in the
            TOML.

    Returns:
        Similarity score between 0 and 1, where 1 indicates a perfect
        match.
    """
    if n_fft is None:
        n_fft = int(
            config.read_config(("window_parameter", "dynamic_time_warping"))
        )
    mfcc1 = calculate_mfcc(audio_signal1, sample_rate1, n_fft=n_fft)
    mfcc2 = calculate_mfcc(audio_signal2, sample_rate2, n_fft=n_fft)

    distance, _ = dtw(X=mfcc1, Y=mfcc2)
    dtw_distance = distance[-1, -1] / (mfcc1.shape[0] * mfcc2.shape[0])
    max_distance = max(mfcc1.shape[1], mfcc2.shape[1])

    similarity = (dtw_distance / max_distance) if max_distance > 0 else 1.0
    return max(0.0, 1 - similarity)


def compare_multiple_dtw(
    audio_signals: list[npt.NDArray[np.float32]],
    sample_rates: list[float],
    /,
    *,
    n_fft: int | None = None,
) -> float:
    """Computes average DTW based similarity for all unique signal pairs using
    'compare_two_dtw', reflecting overall sequence alignment coherence.

    Args:
        audio_signals: List of audio time series arrays.
        sample_rates: Corresponding sampling rates of each signal.
        n_fft: FFT window length used for MFCC extraction.

    Returns:
        Mean similarity score across all unique pairwise comparisons.

    Raises:
        ValueError: If the number of signals does not match the number of
            sampling rates.
    """
    if len(audio_signals) != len(sample_rates):
        raise ValueError(
            "The number of signals must match the number of sampling rates"
        )

    num_signals = len(audio_signals)
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(num_signals):
        for j in range(i + 1, num_signals):
            total_similarity += compare_two_dtw(
                audio_signals[i],
                audio_signals[j],
                sample_rates[i],
                sample_rates[j],
                n_fft=n_fft,
            )
            num_comparisons += 1

    return (
        total_similarity / num_comparisons if num_comparisons > 0 else 0.0
    )
