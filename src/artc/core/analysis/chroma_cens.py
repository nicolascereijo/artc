import numpy as np
from librosa.feature import chroma_cens


def calculate_chroma_cens(audio_signal: np.ndarray, sample_rate: float,
                          /, *, hop_length: int = 512) -> np.ndarray:
    """
        Extracts the Chroma CENS feature sequence from the audio signal using the CENS chroma
        algorithm and returns its frequency-domain representation.

        Args:
            audio_signal (np.ndarray): Time-series array of the audio signal.
            sample_rate (float): Sampling rate (in Hz) of the audio signal.

        Keyword Arguments:
            hop_length (int): Number of samples between successive analysis frames.

        Returns:
            np.ndarray: FFT of the Chroma CENS matrix (12 chroma bins × frames).
    """
    chr_cens = chroma_cens(y=audio_signal, sr=sample_rate, hop_length=hop_length)
    return np.fft.fft(chr_cens)


def compare_two_chroma_cens(signal1: np.ndarray, signal2: np.ndarray,
                            sample_rate1: float, sample_rate2: float,
                            /, *, hop_length: int = 512) -> float:
    """
        Compares Chroma CENS alignment between two audio signals by computing their Chroma CENS FFTs
        and calculating a normalized similarity score.

        Args:
            signal1 (np.ndarray): First audio time-series array.
            signal2 (np.ndarray): Second audio time-series array.
            sample_rate1 (float): Sampling rate (in Hz) of the first signal.
            sample_rate2 (float): Sampling rate (in Hz) of the second signal.

        Keyword Arguments:
            hop_length (int): Number of samples between successive analysis frames.

        Returns:
            float: Similarity score between 0 and 1, where 1 indicates perfect alignment.

        See Also:
            calculate_chroma_cens
    """
    chroma_cens1 = calculate_chroma_cens(signal1, sample_rate1, hop_length=hop_length)
    chroma_cens2 = calculate_chroma_cens(signal2, sample_rate2, hop_length=hop_length)

    min_len = min(chroma_cens1.shape[1], chroma_cens2.shape[1])
    chroma_cens1_adjusted = chroma_cens1[:, :min_len]
    chroma_cens2_adjusted = chroma_cens2[:, :min_len]

    distance = np.linalg.norm(np.abs(chroma_cens1_adjusted) -
                              np.abs(chroma_cens2_adjusted))
    max_distance = (np.linalg.norm(np.abs(chroma_cens1_adjusted)) +
                    np.linalg.norm(np.abs(chroma_cens2_adjusted)))

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_chroma_cens(audio_signals: list, sample_rates: list,
                                 /, *, hop_length: int = 512) -> float:
    """
        Computes average Chroma CENS alignment similarity for all unique signal pairs using
        `compare_two_chroma_cens`, reflecting overall harmonic coherence.

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
            compare_two_chroma_cens
    """
    if len(audio_signals) != len(sample_rates):
        raise ValueError("The number of signals must match the number of sampling rates")

    num_signals = len(audio_signals)
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(num_signals):
        for j in range(i + 1, num_signals):
            total_similarity += compare_two_chroma_cens(
                audio_signals[i], audio_signals[j],
                sample_rates[i], sample_rates[j],
                hop_length=hop_length
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
