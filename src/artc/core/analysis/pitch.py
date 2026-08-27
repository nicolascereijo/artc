from typing import Any

import numpy as np
import numpy.typing as npt
from librosa.core import piptrack

import artc.core.configurations as config

from ..datastructures.harmonize import (
    adjust_dimensions,
    check_matching_sample_rates,
)


def calculate_pitch(
    audio_signal: npt.NDArray[np.float32],
    sample_rate: float,
    /,
    *,
    n_fft: int | None = None,
) -> npt.NDArray[np.number[Any]]:  # pyright: ignore[reportExplicitAny]
    """Extracts the predominant pitch contour from the audio signal using the
    piptrack algorithm and returns its frequency domain representation.

    Args:
        audio_signal: Time series array of the audio signal.
        sample_rate: Sampling rate, in Hz, of the audio signal.
        n_fft: FFT window length for pitch tracking analysis. Defaults to the
            'pitch' entry of '[metric.window_parameter]' in the TOML.

    Returns:
        FFT of the pitch sequence, in Hz, extracted per frame.
    """
    if n_fft is None:
        n_fft = int(config.read_config(("window_parameter", "pitch")))
    pitches, magnitudes = piptrack(
        y=audio_signal, sr=sample_rate, n_fft=n_fft
    )
    frames = np.arange(magnitudes.shape[1])
    return np.fft.fft(pitches[magnitudes.argmax(axis=0), frames])


def compare_two_pitch(
    audio_signal1: npt.NDArray[np.float32],
    audio_signal2: npt.NDArray[np.float32],
    sample_rate1: float,
    sample_rate2: float,
    /,
    *,
    n_fft: int | None = None,
) -> float:
    """Compares pitch contours between two audio signals by computing
    their pitch FFTs and returns a normalized similarity score.

    Args:
        audio_signal1: First audio time series array.
        audio_signal2: Second audio time series array.
        sample_rate1: Sampling rate, in Hz, of the first signal.
        sample_rate2: Sampling rate, in Hz, of the second signal.
        n_fft: FFT window length for pitch tracking analysis. See
            'calculate_pitch'.

    Returns:
        Similarity score between 0 and 1, where 1 means a perfect match.
    """
    check_matching_sample_rates(sample_rate1, sample_rate2)

    pitch1 = calculate_pitch(audio_signal1, sample_rate1, n_fft=n_fft)
    pitch2 = calculate_pitch(audio_signal2, sample_rate2, n_fft=n_fft)

    pitch1_adjusted, pitch2_adjusted = adjust_dimensions(pitch1, pitch2)

    distance = np.linalg.norm(pitch1_adjusted - pitch2_adjusted)
    max_distance = (
        np.linalg.norm(pitch1_adjusted) + np.linalg.norm(pitch2_adjusted)
    )

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(max(0.0, similarity))


def compare_multiple_pitch(
    audio_signals: list[npt.NDArray[np.float32]],
    sample_rates: list[float],
    /,
    *,
    n_fft: int | None = None,
) -> float:
    """Computes the average pitch similarity for every unique signal pair using
    'compare_two_pitch', reflecting overall melodic coherence.

    Args:
        audio_signals: List of audio time series arrays.
        sample_rates: Sampling rate, in Hz, of each signal, in the same order
            as 'audio_signals'.
        n_fft: FFT window length for pitch tracking analysis. See
            'calculate_pitch'.

    Returns:
        Mean similarity score across every unique pairwise comparison.

    Raises:
        ValueError: If 'audio_signals' and 'sample_rates' have different
            lengths.
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
            total_similarity += compare_two_pitch(
                audio_signals[i],
                audio_signals[j],
                sample_rates[i],
                sample_rates[j],
                n_fft=n_fft,
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
