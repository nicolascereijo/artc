import numpy as np
from librosa.feature import spectral_contrast


def calculate_spectral_contrast(audio_signal: np.ndarray, sample_rate: float,
                                /, *, hop_length: int = 2048) -> np.ndarray:
    """
        Computes spectral contrast of the audio signal using multiple frequency bands and returns
        its frequency-domain representation.

        Args:
            audio_signal (np.ndarray): Time-series array of the audio signal.
            sample_rate (float): Sampling rate (in Hz) of the audio signal.

        Keyword Arguments:
            hop_length (int): Number of samples between successive analysis frames.

        Returns:
            np.ndarray: FFT of the spectral contrast matrix.
    """
    contrast = spectral_contrast(y=audio_signal, sr=sample_rate, hop_length=hop_length)
    return np.fft.fft(contrast)


def compare_two_spectral_contrast(audio_signal1: np.ndarray, audio_signal2: np.ndarray,
                                  sample_rate1: float, sample_rate2: float,
                                  /, *, hop_length: int = 2048) -> float:
    """
        Compares spectral contrast between two audio signals by computing their FFTs and returning a
        normalized similarity score.

        Args:
            audio_signal1 (np.ndarray): First audio time-series array.
            audio_signal2 (np.ndarray): Second audio time-series array.
            sample_rate1 (float): Sampling rate (in Hz) of the first signal.
            sample_rate2 (float): Sampling rate (in Hz) of the second signal.

        Keyword Arguments:
            hop_length (int): Number of samples between successive analysis frames.

        Returns:
            float: Similarity score between 0 and 1, where 1 indicates perfect alignment.

        See Also:
            calculate_spectral_contrast
    """
    contrast_1 = calculate_spectral_contrast(audio_signal1, sample_rate1, hop_length=hop_length)
    contrast_2 = calculate_spectral_contrast(audio_signal2, sample_rate2, hop_length=hop_length)

    min_len = min(contrast_1.shape[1], contrast_2.shape[1])
    contrast_1_adjusted = contrast_1[:, :min_len]
    contrast_2_adjusted = contrast_2[:, :min_len]

    distance = np.linalg.norm(np.abs(contrast_1_adjusted) -
                              np.abs(contrast_2_adjusted))
    max_distance = (np.linalg.norm(np.abs(contrast_1_adjusted)) +
                    np.linalg.norm(np.abs(contrast_2_adjusted)))

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_spectral_contrast(audio_signals: list, sample_rates: list,
                                       /, *, hop_length: int = 2048) -> float:
    """
        Computes average spectral contrast similarity for all unique signal pairs using
        `compare_two_spectral_contrast`, reflecting overall tonal texture coherence.

        Args:
            audio_signals (list[np.ndarray]): List of audio time-series arrays.
            sample_rates (list[float]): Corresponding sampling rates of each signal.

        Keyword Arguments:
            hop_length (int): Number of samples between successive analysis frames.

        Returns:
            float: Mean similarity score across all unique pairwise comparisons.

        Raises:
            ValueError: If the number of signals does not match the number of sample rates.

        See Also:
            compare_two_spectral_contrast
    """
    if len(audio_signals) != len(sample_rates):
        raise ValueError("The number of signals must match the number of sampling rates")

    num_signals = len(audio_signals)
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(num_signals):
        for j in range(i + 1, num_signals):
            total_similarity += compare_two_spectral_contrast(
                audio_signals[i], audio_signals[j],
                sample_rates[i], sample_rates[j],
                hop_length=hop_length
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
