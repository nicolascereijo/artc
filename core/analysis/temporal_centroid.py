import numpy as np
from librosa import times_like
from librosa.onset import onset_strength


def calculate_temporal_centroid(audio_signal: np.ndarray, sample_rate: float, /) -> np.ndarray:
    """
        Computes the temporal centroid of the audio signal based on onset strength, returning the
        average time (in seconds) weighted by onset energy.

        Args:
            audio_signal (np.ndarray): Time-series array of the audio signal.
            sample_rate (float): Sampling rate (in Hz) of the audio signal.

        Returns:
            np.ndarray: 1D array containing the temporal centroid in seconds.
    """
    envelope = np.abs(onset_strength(y=audio_signal, sr=sample_rate))
    times = times_like(envelope, sr=sample_rate)

    temporal_centroid = np.sum(envelope * times) / np.sum(envelope)
    return np.array([temporal_centroid])


def compare_two_temporal_centroid(audio_signal1: np.ndarray, audio_signal2: np.ndarray,
                                  sample_rate1: float, sample_rate2: float, /) -> float:
    """
        Compares temporal centroids of two audio signals and returns a normalized similarity score.

        Args:
            audio_signal1 (np.ndarray): First audio time-series array.
            audio_signal2 (np.ndarray): Second audio time-series array.
            sample_rate1 (float): Sampling rate (in Hz) of the first signal.
            sample_rate2 (float): Sampling rate (in Hz) of the second signal.

        Returns:
            float: Similarity score between 0 and 1, where 1 indicates identical centroids.

        See Also:
            calculate_temporal_centroid
    """
    centroid_1 = calculate_temporal_centroid(audio_signal1, sample_rate1)
    centroid_2 = calculate_temporal_centroid(audio_signal2, sample_rate2)

    distance = np.linalg.norm(np.abs(centroid_1) -
                              np.abs(centroid_2))
    max_distance = (np.linalg.norm(np.abs(centroid_1)) +
                    np.linalg.norm(np.abs(centroid_2)))

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_temporal_centroid(audio_signals: list, sample_rates: list, /) -> float:
    """
        Computes average temporal centroid similarity for all unique signal pairs using
        `compare_two_temporal_centroid`, reflecting overall temporal balance coherence.

        Args:
            audio_signals (list[np.ndarray]): List of audio time-series arrays.
            sample_rates (list[float]): Corresponding sampling rates of each signal.

        Returns:
            float: Mean similarity score across all unique pairwise comparisons.

        Raises:
            ValueError: If the number of signals does not match the number of sampling rates.

        See Also:
            compare_two_temporal_centroid
    """
    if len(audio_signals) != len(sample_rates):
        raise ValueError("The number of signals must match the number of sampling rates")

    num_signals = len(audio_signals)
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(num_signals):
        for j in range(i + 1, num_signals):
            total_similarity += compare_two_temporal_centroid(
                audio_signals[i], audio_signals[j],
                sample_rates[i], sample_rates[j]
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
