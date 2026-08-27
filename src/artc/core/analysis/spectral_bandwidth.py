from typing import Any

import numpy as np
import numpy.typing as npt
from librosa.feature import spectral_bandwidth

import artc.core.configurations as config

from ..datastructures.harmonize import (
    adjust_dimensions,
    check_matching_sample_rates,
)


def calculate_spectral_bandwidth(
    audio_signal: npt.NDArray[np.float32],
    sample_rate: float,
    /,
    *,
    n_fft: int | None = None,
) -> npt.NDArray[np.number[Any]]:  # pyright: ignore[reportExplicitAny]
    """Computes the spectral bandwidth of the audio signal and returns its
    frequency domain representation.

    Args:
        audio_signal: Time series array of the audio signal.
        sample_rate: Sampling rate, in Hz, of the audio signal.
        n_fft: Length of the FFT window for spectral analysis. Defaults to the
            'spectral_bandwidth' entry of '[metric.window_parameter]' in the
            TOML.

    Returns:
        The FFT of the spectral bandwidth sequence.
    """
    if n_fft is None:
        n_fft = int(
            config.read_config(("window_parameter", "spectral_bandwidth"))
        )
    bandwidth = spectral_bandwidth(
        y=audio_signal, sr=sample_rate, n_fft=n_fft
    )
    return np.fft.fft(bandwidth)


def compare_two_spectral_bandwidth(
    audio_signal1: npt.NDArray[np.float32],
    audio_signal2: npt.NDArray[np.float32],
    sample_rate1: float,
    sample_rate2: float,
    /,
    *,
    n_fft: int | None = None,
) -> float:
    """Compares spectral bandwidth between two audio signals by computing
    their bandwidth FFTs and returning a normalized similarity score.

    Args:
        audio_signal1: First audio time series array.
        audio_signal2: Second audio time series array.
        sample_rate1: Sampling rate, in Hz, of the first signal.
        sample_rate2: Sampling rate, in Hz, of the second signal.
        n_fft: Length of the FFT window for spectral analysis.

    Returns:
        A similarity score between 0 and 1, where 1 indicates perfect
        alignment.
    """
    check_matching_sample_rates(sample_rate1, sample_rate2)

    bandwidth_1 = calculate_spectral_bandwidth(
        audio_signal1, sample_rate1, n_fft=n_fft
    )
    bandwidth_2 = calculate_spectral_bandwidth(
        audio_signal2, sample_rate2, n_fft=n_fft
    )

    bandwidth_1_adjusted, bandwidth_2_adjusted = adjust_dimensions(
        bandwidth_1, bandwidth_2
    )

    distance = np.linalg.norm(
        np.abs(bandwidth_1_adjusted) - np.abs(bandwidth_2_adjusted)
    )
    max_distance = np.linalg.norm(
        np.abs(bandwidth_1_adjusted)
    ) + np.linalg.norm(np.abs(bandwidth_2_adjusted))

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_spectral_bandwidth(
    audio_signals: list[npt.NDArray[np.float32]],
    sample_rates: list[float],
    /,
    *,
    n_fft: int | None = None,
) -> float:
    """Computes the average spectral bandwidth similarity for all unique signal
    pairs using 'compare_two_spectral_bandwidth', reflecting overall
    spectral dispersion coherence.

    Args:
        audio_signals: List of audio time series arrays.
        sample_rates: Corresponding sampling rates of each signal.
        n_fft: Length of the FFT window for spectral analysis.

    Returns:
        The mean similarity score across all unique pairwise comparisons.

    Raises:
        ValueError: If the number of signals does not match the number of
            sample rates.
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
            total_similarity += compare_two_spectral_bandwidth(
                audio_signals[i], audio_signals[j],
                sample_rates[i], sample_rates[j],
                n_fft=n_fft
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
