import numpy as np
from librosa.feature import chroma_stft


def calculate_chroma_stft(audio_signal: np.ndarray, sample_rate: float,
                          /, *, n_fft: int = 2048) -> np.ndarray:
    """
        Extracts the Chroma STFT feature matrix from the audio signal using the short-time Fourier
        transform and returns its frequency-domain representation.

        Args:
            audio_signal (np.ndarray): Time-series array of the audio signal.
            sample_rate (float): Sampling rate (in Hz) of the audio signal.

        Keyword Arguments:
            n_fft (int): Length of the FFT window for STFT analysis.

        Returns:
            np.ndarray: FFT of the Chroma STFT matrix.
    """
    chr_stft = chroma_stft(y=audio_signal, sr=sample_rate, n_fft=n_fft)
    return np.fft.fft(chr_stft)


def compare_two_chroma_stft(audio_signal1: np.ndarray, audio_signal2: np.ndarray,
                            sample_rate1: float, sample_rate2: float,
                            /, *, n_fft: int = 2048) -> float:
    """
        Compares Chroma STFT alignment between two audio signals by computing their Chroma STFT FFTs
        and calculating a normalized similarity score.

        Args:
            audio_signal1 (np.ndarray): First audio time-series array.
            audio_signal2 (np.ndarray): Second audio time-series array.
            sample_rate1 (float): Sampling rate (in Hz) of the first signal.
            sample_rate2 (float): Sampling rate (in Hz) of the second signal.

        Keyword Arguments:
            n_fft (int): Length of the FFT window for STFT analysis.

        Returns:
            float: Similarity score between 0 and 1, where 1 indicates perfect alignment.

        See Also:
            calculate_chroma_stft
    """
    chroma_1 = calculate_chroma_stft(audio_signal1, sample_rate1, n_fft=n_fft)
    chroma_2 = calculate_chroma_stft(audio_signal2, sample_rate2, n_fft=n_fft)

    min_len = min(chroma_1.shape[1], chroma_2.shape[1])
    matrix1_fft_adjusted = chroma_1[:, :min_len]
    matrix2_fft_adjusted = chroma_2[:, :min_len]

    distance = np.linalg.norm(np.abs(matrix1_fft_adjusted) -
                              np.abs(matrix2_fft_adjusted))
    max_distance = (np.linalg.norm(np.abs(matrix1_fft_adjusted)) +
                    np.linalg.norm(np.abs(matrix2_fft_adjusted)))

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_chroma_stft(audio_signals: list, sample_rates: list,
                                 /, *, n_fft: int = 2048) -> float:
    """
        Computes average Chroma STFT alignment similarity for all unique signal pairs using
        `compare_two_chroma_stft`, reflecting overall harmonic coherence.

        Args:
            audio_signals (list[np.ndarray]): List of audio time-series arrays.
            sample_rates (list[float]): Corresponding sampling rates of each signal.

        Keyword Arguments:
            n_fft (int): Length of the FFT window for STFT analysis.

        Returns:
            float: Mean similarity score across all unique pairwise comparisons.

        Raises:
            ValueError: If the number of signals does not match the number of sample rates.

        See Also:
            compare_two_chroma_stft
    """
    if len(audio_signals) != len(sample_rates):
        raise ValueError("The number of signals must match the number of sampling rates")

    num_signals = len(audio_signals)
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(num_signals):
        for j in range(i + 1, num_signals):
            total_similarity += compare_two_chroma_stft(
                audio_signals[i], audio_signals[j],
                sample_rates[i], sample_rates[j],
                n_fft=n_fft
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
