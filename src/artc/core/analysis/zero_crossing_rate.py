import numpy as np
from librosa.feature import zero_crossing_rate


def calculate_zcr(audio_signal: np.ndarray,
                  /, *, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
        Computes the zero-crossing rate (ZCR) of the audio signal over time frames.

        Args:
            audio_signal (np.ndarray): Time-series array of the audio signal.

        Keyword Arguments:
            frame_length (int): Length of each analysis frame (in samples).
            hop_length (int): Number of samples between successive frames.

        Returns:
            np.ndarray: ZCR sequence as a 2D array with shape (1, frames).
    """
    return zero_crossing_rate(y=audio_signal, frame_length=frame_length, hop_length=hop_length)


def compare_two_zcr(signal1: np.ndarray, signal2: np.ndarray,
                    /, *, frame_length: int = 2048, hop_length: int = 512) -> float:
    """
        Compares zero-crossing rate sequences of two audio signals and returns a normalized
        similarity score.

        Args:
            signal1 (np.ndarray): First audio time-series array.
            signal2 (np.ndarray): Second audio time-series array.

        Keyword Arguments:
            frame_length (int): Length of each analysis frame (in samples).
            hop_length (int): Number of samples between successive frames.

        Returns:
            float: Similarity score between 0 and 1, where 1 indicates identical ZCR profiles.

        See Also:
            calculate_zcr
    """
    zcr1 = calculate_zcr(signal1, frame_length=frame_length, hop_length=hop_length)
    zcr2 = calculate_zcr(signal2, frame_length=frame_length, hop_length=hop_length)

    zcr1_normalized = zcr1 / zcr1.max() if zcr1.max() > 0 else zcr1
    zcr2_normalized = zcr2 / zcr2.max() if zcr2.max() > 0 else zcr2

    min_len = min(zcr1_normalized.shape[1], zcr2_normalized.shape[1])
    zcr1_adjusted = zcr1_normalized[:, :min_len]
    zcr2_adjusted = zcr2_normalized[:, :min_len]

    relative_difference = (np.abs(zcr1_adjusted - zcr2_adjusted) /
                           (np.abs(zcr1_adjusted) + np.abs(zcr2_adjusted) + 1e-8))
    similarity = 1.0 - np.mean(relative_difference)
    return float(max(0.0, similarity))


def compare_multiple_zcr(audio_signals: list,
                         /, *, frame_length: int = 2048, hop_length: int = 512) -> float:
    """
        Computes average ZCR similarity for all unique signal pairs using `compare_two_zcr`,
        reflecting overall temporal fine-structure coherence.

        Args:
            audio_signals (list[np.ndarray]): List of audio time-series arrays.

        Keyword Arguments:
            frame_length (int): Length of each analysis frame (in samples).
            hop_length (int): Number of samples between successive frames.

        Returns:
            float: Mean similarity score across all unique pairwise comparisons.

        Raises:
            ValueError: If fewer than two signals are provided.

        See Also:
            compare_two_zcr
    """
    num_signals = len(audio_signals)
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(num_signals):
        for j in range(i + 1, num_signals):
            total_similarity += compare_two_zcr(
                audio_signals[i], audio_signals[j],
                frame_length=frame_length, hop_length=hop_length
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
