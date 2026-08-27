from typing import Any

import numpy as np
import numpy.typing as npt
from librosa.feature import spectral_rolloff

import artc.core.configurations as config

from ..datastructures.harmonize import (
    adjust_dimensions,
    check_matching_sample_rates,
)


def calculate_spectral_roll_off(
    audio_signal: npt.NDArray[np.float32],
    sample_rate: float,
    /,
    *,
    n_fft: int | None = None,
    roll_percent: float = 0.5,
) -> npt.NDArray[np.number[Any]]:  # pyright: ignore[reportExplicitAny]
    """Computes the spectral roll off frequency for each frame and returns its
    frequency domain representation.

    Args:
        audio_signal: Time series array of the audio signal.
        sample_rate: Sampling rate, in Hz, of the audio signal.
        n_fft: Length of the FFT window for spectral analysis. Defaults to the
            'spectral_roll_off' entry of '[metric.window_parameter]' in the
            TOML.
        roll_percent: Fraction of spectral energy below the roll off frequency.
            Not read from the TOML, always uses its default unless passed
            explicitly.

    Returns:
        The FFT of the spectral roll off sequence.
    """
    if n_fft is None:
        n_fft = int(
            config.read_config(("window_parameter", "spectral_roll_off"))
        )
    roll_off = spectral_rolloff(
        y=audio_signal, sr=sample_rate, n_fft=n_fft, roll_percent=roll_percent
    )
    return np.fft.fft(roll_off)


def compare_two_spectral_roll_off(
    audio_signal1: npt.NDArray[np.float32],
    audio_signal2: npt.NDArray[np.float32],
    sample_rate1: float,
    sample_rate2: float,
    /,
    *,
    n_fft: int | None = None,
    roll_percent: float = 0.5,
) -> float:
    """Compares spectral roll off between two audio signals by computing their
    roll off FFTs and returning a normalized similarity score.

    Args:
        audio_signal1: First audio time series array.
        audio_signal2: Second audio time series array.
        sample_rate1: Sampling rate, in Hz, of the first signal.
        sample_rate2: Sampling rate, in Hz, of the second signal.
        n_fft: Length of the FFT window for spectral analysis.
        roll_percent: Fraction of spectral energy below the roll off frequency.

    Returns:
        A similarity score between 0 and 1, where 1 indicates identical
        roll off patterns.
    """
    check_matching_sample_rates(sample_rate1, sample_rate2)

    roll_off_1 = calculate_spectral_roll_off(
        audio_signal1, sample_rate1, n_fft=n_fft, roll_percent=roll_percent
    )
    roll_off_2 = calculate_spectral_roll_off(
        audio_signal2, sample_rate2, n_fft=n_fft, roll_percent=roll_percent
    )

    roll_off_1_adjusted, roll_off_2_adjusted = adjust_dimensions(
        roll_off_1, roll_off_2
    )

    distance = np.linalg.norm(
        np.abs(roll_off_1_adjusted) - np.abs(roll_off_2_adjusted)
    )
    max_distance = np.linalg.norm(
        np.abs(roll_off_1_adjusted)
    ) + np.linalg.norm(np.abs(roll_off_2_adjusted))

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_spectral_roll_off(
    audio_signals: list[npt.NDArray[np.float32]],
    sample_rates: list[float],
    /,
    *,
    n_fft: int | None = None,
    roll_percent: float = 0.5,
) -> float:
    """Computes the average spectral roll off similarity for every unique
    signal pair, using 'compare_two_spectral_roll_off', reflecting the overall
    spectral shape coherence.

    Args:
        audio_signals: List of audio time series arrays.
        sample_rates: Sampling rate, in Hz, of each signal, in the same order
            as 'audio_signals'.
        n_fft: Length of the FFT window for spectral analysis.
        roll_percent: Fraction of spectral energy below the roll off frequency.

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
            total_similarity += compare_two_spectral_roll_off(
                audio_signals[i],
                audio_signals[j],
                sample_rates[i],
                sample_rates[j],
                n_fft=n_fft,
                roll_percent=roll_percent,
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
