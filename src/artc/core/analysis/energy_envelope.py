from typing import Any

import numpy as np
import numpy.typing as npt
from librosa.feature import rms

import artc.core.configurations as config

from ..datastructures.harmonize import adjust_dimensions


def calculate_energy_envelope(
    audio_signal: npt.NDArray[np.float32],
    /,
    *,
    hop_length: int | None = None,
) -> npt.NDArray[np.number[Any]]:  # pyright: ignore[reportExplicitAny]
    """Computes the energy envelope of the audio signal using RMS and returns
    its frequency domain representation.

    Args:
        audio_signal: Time series array of the audio signal.
        hop_length: Number of samples between successive analysis
            frames. Defaults to the 'energy_envelope' entry of
            [metric.window_parameter] in the TOML.

    Returns:
        FFT of the energy envelope sequence.
    """
    if hop_length is None:
        hop_length = int(
            config.read_config(("window_parameter", "energy_envelope"))
        )
    energy_envelope = rms(y=audio_signal, hop_length=hop_length)
    return np.fft.fft(energy_envelope)


def compare_two_energy_envelope(
    audio_signal1: npt.NDArray[np.float32],
    audio_signal2: npt.NDArray[np.float32],
    /,
    *,
    hop_length: int | None = None,
) -> float:
    """Compares energy envelopes between two audio signals by computing their
    energy envelope FFTs and returning a normalized similarity score.

    Args:
        audio_signal1: First audio time series array.
        audio_signal2: Second audio time series array.
        hop_length: Number of samples between successive analysis frames.

    Returns:
        Similarity score between 0 and 1, where 1 indicates identical energy
        envelopes.
    """
    energy_envelope1 = calculate_energy_envelope(
        audio_signal1, hop_length=hop_length
    )
    energy_envelope2 = calculate_energy_envelope(
        audio_signal2, hop_length=hop_length
    )

    energy1_adjusted, energy2_adjusted = adjust_dimensions(
        energy_envelope1, energy_envelope2
    )

    distance = np.linalg.norm(energy1_adjusted - energy2_adjusted)
    max_distance = (
        np.linalg.norm(energy1_adjusted) +
        np.linalg.norm(energy2_adjusted)
    )

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_energy_envelope(
    audio_signals: list[npt.NDArray[np.float32]],
    /,
    *,
    hop_length: int | None = None,
) -> float:
    """Computes average energy envelope similarity for all unique signal
    pairs, reflecting overall dynamic coherence.

    Args:
        audio_signals: List of audio time series arrays.
        hop_length: Number of samples between successive analysis frames.

    Returns:
        Mean similarity score across all unique pairwise comparisons.
    """
    num_signals = len(audio_signals)
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(num_signals):
        for j in range(i + 1, num_signals):
            total_similarity += compare_two_energy_envelope(
                audio_signals[i], audio_signals[j],
                hop_length=hop_length
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
