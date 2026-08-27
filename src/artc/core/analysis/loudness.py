from typing import Any

import librosa
import numpy as np
import numpy.typing as npt

from ..datastructures.harmonize import (
    adjust_dimensions,
    check_matching_sample_rates,
)


def calculate_loudness(
    audio_signal: npt.NDArray[np.float32], sample_rate: float,
) -> npt.NDArray[np.number[Any]]:  # pyright: ignore[reportExplicitAny]
    """Computes A weighted loudness of the audio signal by converting its
    magnitude spectrogram to decibels, applying A weighting and returning the
    frequency domain representation.

    Args:
        audio_signal: Time series array of the audio signal.
        sample_rate: Sampling rate, in Hz, of the audio signal.

    Returns:
        FFT of the A weighted decibel spectrogram.
    """
    magnitude_spectrogram = np.abs(librosa.stft(audio_signal))
    db_spectrogram = librosa.amplitude_to_db(
        magnitude_spectrogram, ref=np.max
    )

    frequencies = librosa.fft_frequencies(sr=sample_rate)
    frequencies[frequencies == 0] = 1e-6  # Avoids log10(0).

    weighting = librosa.A_weighting(frequencies)[:, np.newaxis]
    return np.fft.fft(db_spectrogram * weighting)


def compare_two_loudness(
    audio_signal1: npt.NDArray[np.float32],
    audio_signal2: npt.NDArray[np.float32],
    sample_rate1: float,
    sample_rate2: float,
    /,
) -> float:
    """Compares loudness profiles between two audio signals by computing their
    A weighted spectrogram FFTs and returning a normalized similarity score.

    Args:
        audio_signal1: First audio time series array.
        audio_signal2: Second audio time series array.
        sample_rate1: Sampling rate, in Hz, of the first signal.
        sample_rate2: Sampling rate, in Hz, of the second signal.

    Returns:
        Similarity score between 0 and 1, where 1 indicates identical loudness
        patterns.
    """
    check_matching_sample_rates(sample_rate1, sample_rate2)

    loudness1 = calculate_loudness(audio_signal1, sample_rate1)
    loudness2 = calculate_loudness(audio_signal2, sample_rate2)

    loudness1_adjusted, loudness2_adjusted = adjust_dimensions(
        loudness1, loudness2
    )

    distance = np.linalg.norm(loudness1_adjusted - loudness2_adjusted)
    max_distance = (
        np.linalg.norm(loudness1_adjusted) +
        np.linalg.norm(loudness2_adjusted)
    )

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_loudness(
    audio_signals: list[npt.NDArray[np.float32]],
    sample_rates: list[float],
    /,
) -> float:
    """Computes average loudness similarity for all unique signal pairs,
    reflecting overall loudness pattern coherence.

    Args:
        audio_signals: List of audio time series arrays.
        sample_rates: Corresponding sampling rates of each signal.

    Returns:
        Mean similarity score across all unique pairwise comparisons.
    """
    num_signals = len(audio_signals)
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(num_signals):
        for j in range(i + 1, num_signals):
            total_similarity += compare_two_loudness(
                audio_signals[i], audio_signals[j],
                sample_rates[i], sample_rates[j]
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
