from typing import Any

import numpy as np
import numpy.typing as npt
from librosa.onset import onset_strength

import artc.core.configurations as config

from ..datastructures.harmonize import (
    adjust_dimensions,
    check_matching_sample_rates,
)


def calculate_temporal_flux(
    audio_signal: npt.NDArray[np.float32],
    sample_rate: float,
    /,
    *,
    hop_length: int | None = None,
) -> npt.NDArray[np.number[Any]]:  # pyright: ignore[reportExplicitAny]
    """Computes the temporal flux of the audio signal, from its onset strength.

    Args:
        audio_signal: Time series array of the audio signal.
        sample_rate: Sampling rate, in Hz, of the audio signal.
        hop_length: Number of samples between successive analysis frames.
            Defaults to the 'temporal_flux' entry of
            '[metric.window_parameter]' in the TOML.

    Returns:
        The FFT of the temporal flux sequence.
    """
    if hop_length is None:
        hop_length = int(
            config.read_config(("window_parameter", "temporal_flux"))
        )
    temporal_flux = onset_strength(
        y=audio_signal, sr=sample_rate, hop_length=hop_length
    )
    return np.fft.fft(temporal_flux)


def compare_two_temporal_flux(
    audio_signal1: npt.NDArray[np.float32],
    audio_signal2: npt.NDArray[np.float32],
    sample_rate1: float,
    sample_rate2: float,
    /,
    *,
    hop_length: int | None = None,
) -> float:
    """Compares temporal flux sequences of two audio signals by their FFTs.

    Args:
        audio_signal1: First audio time series array.
        audio_signal2: Second audio time series array.
        sample_rate1: Sampling rate, in Hz, of the first signal.
        sample_rate2: Sampling rate, in Hz, of the second signal.
        hop_length: Number of samples between successive analysis frames.

    Returns:
        Similarity score between 0 and 1, where 1 means identical flux
        patterns.
    """
    check_matching_sample_rates(sample_rate1, sample_rate2)

    flux_1 = calculate_temporal_flux(
        audio_signal1, sample_rate1, hop_length=hop_length
    )
    flux_2 = calculate_temporal_flux(
        audio_signal2, sample_rate2, hop_length=hop_length
    )

    flux_1_adjusted, flux_2_adjusted = adjust_dimensions(flux_1, flux_2)

    distance = np.linalg.norm(
        np.abs(flux_1_adjusted) - np.abs(flux_2_adjusted)
    )
    max_distance = (
        np.linalg.norm(np.abs(flux_1_adjusted)) +
        np.linalg.norm(np.abs(flux_2_adjusted))
    )

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_temporal_flux(
    audio_signals: list[npt.NDArray[np.float32]],
    sample_rates: list[float],
    /,
    *,
    hop_length: int | None = None,
) -> float:
    """Computes average temporal flux similarity across every unique pair of
    signals.

    Args:
        audio_signals: List of audio time series arrays.
        sample_rates: Sampling rate, in Hz, of each signal.
        hop_length: Number of samples between successive analysis frames.

    Returns:
        Mean similarity score across every unique pairwise comparison.

    Raises:
        ValueError: If the number of signals does not match the number
            of sample rates.
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
            total_similarity += compare_two_temporal_flux(
                audio_signals[i], audio_signals[j],
                sample_rates[i], sample_rates[j],
                hop_length=hop_length,
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
