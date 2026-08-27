import numpy as np
import numpy.typing as npt
from librosa import times_like
from librosa.onset import onset_strength


def calculate_temporal_centroid(
    audio_signal: npt.NDArray[np.float32], sample_rate: float, /,
) -> npt.NDArray[np.float64]:
    """Computes the temporal centroid of the audio signal, based on onset
    strength.

    Args:
        audio_signal: Time series array of the audio signal.
        sample_rate: Sampling rate, in Hz, of the audio signal.

    Returns:
        A 1D array holding the temporal centroid, the onset energy weighted
        average time, in seconds.
    """
    envelope = np.abs(onset_strength(y=audio_signal, sr=sample_rate))
    times = times_like(envelope, sr=sample_rate)

    # Avoids 0/0 on silence, a 'NaN' here would later be masked as a
    # perfect match.
    envelope_sum = np.sum(envelope)
    if envelope_sum > 0:
        temporal_centroid = np.sum(envelope * times) / envelope_sum
    else:
        temporal_centroid = 0.0
    return np.array([temporal_centroid])


def compare_two_temporal_centroid(
    audio_signal1: npt.NDArray[np.float32],
    audio_signal2: npt.NDArray[np.float32],
    sample_rate1: float,
    sample_rate2: float,
    /,
) -> float:
    """Compares temporal centroids of two audio signals.

    Args:
        audio_signal1: First audio time series array.
        audio_signal2: Second audio time series array.
        sample_rate1: Sampling rate, in Hz, of the first signal.
        sample_rate2: Sampling rate, in Hz, of the second signal.

    Returns:
        Similarity score between 0 and 1, where 1 means identical centroids.
    """
    centroid_1 = calculate_temporal_centroid(audio_signal1, sample_rate1)
    centroid_2 = calculate_temporal_centroid(audio_signal2, sample_rate2)

    distance = np.linalg.norm(np.abs(centroid_1) - np.abs(centroid_2))
    max_distance = (
        np.linalg.norm(np.abs(centroid_1)) +
        np.linalg.norm(np.abs(centroid_2))
    )

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_temporal_centroid(
    audio_signals: list[npt.NDArray[np.float32]],
    sample_rates: list[float],
    /,
) -> float:
    """Computes average temporal centroid similarity across every unique pair
    of signals.

    Args:
        audio_signals: List of audio time series arrays.
        sample_rates: Sampling rate, in Hz, of each signal.

    Returns:
        Mean similarity score across every unique pairwise comparison.

    Raises:
        ValueError: If the number of signals does not match the number of
            sample rates.
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
            total_similarity += compare_two_temporal_centroid(
                audio_signals[i], audio_signals[j],
                sample_rates[i], sample_rates[j],
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
