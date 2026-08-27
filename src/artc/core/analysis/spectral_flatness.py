from typing import Any

import numpy as np
import numpy.typing as npt
from librosa.feature import spectral_flatness

import artc.core.configurations as config

from ..datastructures.harmonize import adjust_dimensions


def calculate_spectral_flatness(
    audio_signal: npt.NDArray[np.float32],
    /,
    *,
    n_fft: int | None = None,
) -> npt.NDArray[np.number[Any]]:  # pyright: ignore[reportExplicitAny]
    """Computes the spectral flatness of the audio signal using an FFT based
    measure and returns its frequency domain representation.

    Args:
        audio_signal: Time series array of the audio signal.
        n_fft: Length of the FFT window for spectral analysis. Defaults
            to the 'spectral_flatness' entry of '[metric.window_parameter]'
            in the TOML.

    Returns:
        The FFT of the spectral flatness sequence.
    """
    if n_fft is None:
        n_fft = int(
            config.read_config(("window_parameter", "spectral_flatness"))
        )
    return np.fft.fft(spectral_flatness(y=audio_signal, n_fft=n_fft))


def compare_two_spectral_flatness(
    audio_signal1: npt.NDArray[np.float32],
    audio_signal2: npt.NDArray[np.float32],
    /,
    *,
    n_fft: int | None = None,
) -> float:
    """Compares spectral flatness between two audio signals by computing their
    flatness FFTs and returning a normalized similarity score.

    Args:
        audio_signal1: First audio time series array.
        audio_signal2: Second audio time series array.
        n_fft: Length of the FFT window for spectral analysis.

    Returns:
        A similarity score between 0 and 1, where 1 indicates identical
        flatness patterns.
    """
    flatness_1 = calculate_spectral_flatness(audio_signal1, n_fft=n_fft)
    flatness_2 = calculate_spectral_flatness(audio_signal2, n_fft=n_fft)

    flatness_1_adjusted, flatness_2_adjusted = adjust_dimensions(
        flatness_1, flatness_2
    )

    distance = np.linalg.norm(
        np.abs(flatness_1_adjusted) - np.abs(flatness_2_adjusted)
    )
    max_distance = np.linalg.norm(
        np.abs(flatness_1_adjusted)
    ) + np.linalg.norm(np.abs(flatness_2_adjusted))

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_spectral_flatness(
    audio_signals: list[npt.NDArray[np.float32]],
    /,
    *,
    n_fft: int | None = None,
) -> float:
    """Computes the average spectral flatness similarity for all unique signal
    pairs using 'compare_two_spectral_flatness', reflecting overall noisiness
    coherence.

    Args:
        audio_signals: List of audio time series arrays.
        n_fft: Length of the FFT window for spectral analysis.

    Returns:
        The mean similarity score across all unique pairwise comparisons.
    """
    num_signals = len(audio_signals)
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(num_signals):
        for j in range(i + 1, num_signals):
            total_similarity += compare_two_spectral_flatness(
                audio_signals[i], audio_signals[j],
                n_fft=n_fft
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
