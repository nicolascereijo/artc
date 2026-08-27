from typing import Any

import numpy as np
import numpy.typing as npt
from librosa.feature import tempogram
from librosa.onset import onset_strength

import artc.core.configurations as config

from ..datastructures.harmonize import (
    adjust_dimensions,
    check_matching_sample_rates,
)


def calculate_wct(
    audio_signal: npt.NDArray[np.float32],
    sample_rate: float,
    /,
    *,
    hop_length: int | None = None,
) -> npt.NDArray[np.number[Any]]:  # pyright: ignore[reportExplicitAny]
    """Computes the weighted cyclic tempogram, WCT, of the audio signal,
    combining onset strength with tempogram autocorrelation.

    Args:
        audio_signal: Time series array of the audio signal.
        sample_rate: Sampling rate, in Hz, of the audio signal.
        hop_length: Number of samples between successive analysis
            frames. Defaults to the 'weighted_cyclic_tempogram' entry of
            '[metric.window_parameter]' in the TOML.

    Returns:
        The FFT of the cyclic tempogram matrix.
    """
    if hop_length is None:
        hop_length = int(
            config.read_config(
                ("window_parameter", "weighted_cyclic_tempogram")
            )
        )
    onset_envelope = onset_strength(
        y=audio_signal, sr=sample_rate, hop_length=hop_length
    )
    cyclic_tempogram = tempogram(
        y=audio_signal,
        sr=sample_rate,
        hop_length=hop_length,
        onset_envelope=onset_envelope,
    )
    return np.fft.fft(cyclic_tempogram)


def compare_two_wct(
    signal1: npt.NDArray[np.float32],
    signal2: npt.NDArray[np.float32],
    sample_rate1: float,
    sample_rate2: float,
    /,
    *,
    hop_length: int | None = None,
) -> float:
    """Compares weighted cyclic tempograms of two audio signals by their FFTs.

    Args:
        signal1: First audio time series array.
        signal2: Second audio time series array.
        sample_rate1: Sampling rate, in Hz, of the first signal.
        sample_rate2: Sampling rate, in Hz, of the second signal.
        hop_length: Number of samples between successive analysis frames.

    Returns:
        Similarity score between 0 and 1, where 1 means identical cyclic
        patterns.
    """
    check_matching_sample_rates(sample_rate1, sample_rate2)

    cyclic_tempogram1 = calculate_wct(
        signal1, sample_rate1, hop_length=hop_length
    )
    cyclic_tempogram2 = calculate_wct(
        signal2, sample_rate2, hop_length=hop_length
    )

    cyclic_tempogram1_adjusted, cyclic_tempogram2_adjusted = adjust_dimensions(
        cyclic_tempogram1, cyclic_tempogram2
    )

    distance = np.linalg.norm(
        np.abs(cyclic_tempogram1_adjusted) -
        np.abs(cyclic_tempogram2_adjusted)
    )
    max_distance = (
        np.linalg.norm(np.abs(cyclic_tempogram1_adjusted)) +
        np.linalg.norm(np.abs(cyclic_tempogram2_adjusted))
    )

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_wct(
    audio_signals: list[npt.NDArray[np.float32]],
    sample_rates: list[float],
    /,
    *,
    hop_length: int | None = None,
) -> float:
    """Computes average weighted cyclic tempogram similarity across every
    unique pair of signals.

    Args:
        audio_signals: List of audio time series arrays.
        sample_rates: Sampling rate, in Hz, of each signal.
        hop_length: Number of samples between successive analysis frames.

    Returns:
        Mean similarity score across every unique pairwise comparison.

    Raises:
        ValueError: If the number of signals does not match the number of
            sample rates.
    """
    if len(audio_signals) != len(sample_rates):
        raise ValueError(
            "The number of signals must match the number of sample rates"
        )

    num_signals = len(audio_signals)
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(num_signals):
        for j in range(i + 1, num_signals):
            total_similarity += compare_two_wct(
                audio_signals[i], audio_signals[j],
                sample_rates[i], sample_rates[j],
                hop_length=hop_length,
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
