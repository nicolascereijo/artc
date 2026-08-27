from typing import Any

import numpy as np
import numpy.typing as npt
from librosa.feature import tempogram

import artc.core.configurations as config

from ..datastructures.harmonize import (
    adjust_dimensions,
    check_matching_sample_rates,
)


def calculate_tempogram(
    audio_signal: npt.NDArray[np.float32],
    sample_rate: float,
    /,
    *,
    hop_length: int | None = None,
) -> npt.NDArray[np.number[Any]]:  # pyright: ignore[reportExplicitAny]
    """Computes the tempogram matrix of the audio signal based on onset
    strength autocorrelation and returns its frequency domain representation.

    Args:
        audio_signal: Time series array of the audio signal.
        sample_rate: Sampling rate, in Hz, of the audio signal.
        hop_length: Number of samples between successive analysis frames.
            Defaults to the 'tempogram' entry of '[metric.window_parameter]' in
            the TOML.

    Returns:
        The FFT of the tempogram matrix.
    """
    if hop_length is None:
        hop_length = int(
            config.read_config(("window_parameter", "tempogram"))
        )
    temp = tempogram(y=audio_signal, sr=sample_rate, hop_length=hop_length)
    return np.fft.fft(temp)


def compare_two_tempogram(
    signal1: npt.NDArray[np.float32],
    signal2: npt.NDArray[np.float32],
    sample_rate1: float,
    sample_rate2: float,
    /,
    *,
    hop_length: int | None = None,
) -> float:
    """Compares tempogram representations of two audio signals by computing
    their FFTs and returning a normalized similarity score.

    Args:
        signal1: First audio time series array.
        signal2: Second audio time series array.
        sample_rate1: Sampling rate, in Hz, of the first signal.
        sample_rate2: Sampling rate, in Hz, of the second signal.
        hop_length: Number of samples between successive analysis frames.

    Returns:
        A similarity score between 0 and 1, where 1 indicates a perfect
        alignment.
    """
    check_matching_sample_rates(sample_rate1, sample_rate2)

    tempogram1 = calculate_tempogram(
        signal1, sample_rate1, hop_length=hop_length
    )
    tempogram2 = calculate_tempogram(
        signal2, sample_rate2, hop_length=hop_length
    )

    tempogram1_adjusted, tempogram2_adjusted = adjust_dimensions(
        tempogram1, tempogram2
    )

    distance = np.linalg.norm(
        np.abs(tempogram1_adjusted) - np.abs(tempogram2_adjusted)
    )
    max_distance = np.linalg.norm(
        np.abs(tempogram1_adjusted)
    ) + np.linalg.norm(np.abs(tempogram2_adjusted))

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_tempogram(
    audio_signals: list[npt.NDArray[np.float32]],
    sample_rates: list[float],
    /,
    *,
    hop_length: int | None = None,
) -> float:
    """Computes the average tempogram similarity for every unique signal pair,
    using 'compare_two_tempogram', reflecting the overall rhythmic periodicity
    coherence.

    Args:
        audio_signals: List of audio time series arrays.
        sample_rates: Sampling rate, in Hz, of each signal, in the same order
            as 'audio_signals'.
        hop_length: Number of samples between successive analysis frames.

    Returns:
        The mean similarity score across every unique pairwise comparison.

    Raises:
        ValueError: If 'audio_signals' and 'sample_rates' have different
            lengths.
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
            total_similarity += compare_two_tempogram(
                audio_signals[i],
                audio_signals[j],
                sample_rates[i],
                sample_rates[j],
                hop_length=hop_length,
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
