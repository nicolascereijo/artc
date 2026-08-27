from typing import Any

import numpy as np
import numpy.typing as npt
from librosa.feature import chroma_stft

import artc.core.configurations as config

from ..datastructures.harmonize import (
    adjust_dimensions,
    check_matching_sample_rates,
)


def calculate_chroma_stft(
    audio_signal: npt.NDArray[np.float32],
    sample_rate: float,
    /,
    *,
    n_fft: int | None = None,
) -> npt.NDArray[np.number[Any]]:  # pyright: ignore[reportExplicitAny]
    """Extracts the Chroma STFT feature matrix from the audio signal using the
    short time Fourier transform and returns its frequency domain
    representation.

    Args:
        audio_signal: Time series array of the audio signal.
        sample_rate: Sampling rate, in Hz, of the audio signal.
        n_fft: Length of the FFT window for STFT analysis. Defaults to the
            'chroma_stft' entry of '[metric.window_parameter]' in the TOML.

    Returns:
        FFT of the Chroma STFT matrix.
    """
    if n_fft is None:
        n_fft = int(config.read_config(("window_parameter", "chroma_stft")))
    chr_stft = chroma_stft(y=audio_signal, sr=sample_rate, n_fft=n_fft)
    return np.fft.fft(chr_stft)


def compare_two_chroma_stft(
    audio_signal1: npt.NDArray[np.float32],
    audio_signal2: npt.NDArray[np.float32],
    sample_rate1: float,
    sample_rate2: float,
    /,
    *,
    n_fft: int | None = None,
) -> float:
    """Compares Chroma STFT alignment between two audio signals by computing
    their Chroma STFT FFTs and calculating a normalized similarity score.

    Args:
        audio_signal1: First audio time series array.
        audio_signal2: Second audio time series array.
        sample_rate1: Sampling rate, in Hz, of the first signal.
        sample_rate2: Sampling rate, in Hz, of the second signal.
        n_fft: Length of the FFT window for STFT analysis.

    Returns:
        Similarity score between 0 and 1, where 1 indicates perfect alignment.
    """
    check_matching_sample_rates(sample_rate1, sample_rate2)

    chroma_1 = calculate_chroma_stft(audio_signal1, sample_rate1, n_fft=n_fft)
    chroma_2 = calculate_chroma_stft(audio_signal2, sample_rate2, n_fft=n_fft)

    matrix1_fft_adjusted, matrix2_fft_adjusted = adjust_dimensions(
        chroma_1, chroma_2
    )

    distance = np.linalg.norm(
        np.abs(matrix1_fft_adjusted) - np.abs(matrix2_fft_adjusted)
    )
    max_distance = np.linalg.norm(
        np.abs(matrix1_fft_adjusted)
    ) + np.linalg.norm(
        np.abs(matrix2_fft_adjusted)
    )

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_chroma_stft(
    audio_signals: list[npt.NDArray[np.float32]],
    sample_rates: list[float],
    /,
    *,
    n_fft: int | None = None,
) -> float:
    """Computes average Chroma STFT alignment similarity for all unique signal
    pairs using 'compare_two_chroma_stft', reflecting overall harmonic
    coherence.

    Args:
        audio_signals: List of audio time series arrays.
        sample_rates: Corresponding sampling rates of each signal.
        n_fft: Length of the FFT window for STFT analysis.

    Returns:
        Mean similarity score across all unique pairwise comparisons.

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
            total_similarity += compare_two_chroma_stft(
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
