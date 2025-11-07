import numpy as np
from librosa.feature import mfcc


def calculate_mfcc(audio_signal: np.ndarray, sample_rate: float,
                   /, *, n_fft: int = 8192) -> np.ndarray:
    """
        Calculates Mel-frequency cepstral coefficients (MFCCs) from the audio signal.

        Args:
            audio_signal (np.ndarray): Time-series array of the audio signal.
            sample_rate (float): Sampling rate (in Hz) of the audio signal.

        Keyword Arguments:
            n_fft (int): Length of the FFT window for STFT used in MFCC extraction.

        Returns:
            np.ndarray: Array of shape (n_mfcc, frames) containing MFCC features.
    """
    return mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13, n_fft=n_fft)


def compare_two_mfcc(audio_signal1: np.ndarray, audio_signal2: np.ndarray,
                     sample_rate1: float, sample_rate2: float,
                     /, *, n_fft: int = 8192) -> float:
    """
        Compares MFCC sequences between two audio signals by computing their FFTs and returning a
        normalized similarity score.

        Args:
            audio_signal1 (np.ndarray): First audio time-series array.
            audio_signal2 (np.ndarray): Second audio time-series array.
            sample_rate1 (float): Sampling rate (in Hz) of the first signal.
            sample_rate2 (float): Sampling rate (in Hz) of the second signal.

        Keyword Arguments:
            n_fft (int): Length of the FFT window for STFT used in MFCC extraction.

        Returns:
            float: Similarity score between 0 and 1, where 1 indicates perfect match.

        See Also:
            calculate_mfcc
    """
    mfcc1 = np.fft.fft(calculate_mfcc(audio_signal1, sample_rate1, n_fft=n_fft))
    mfcc2 = np.fft.fft(calculate_mfcc(audio_signal2, sample_rate2, n_fft=n_fft))

    min_len = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1_adjusted = mfcc1[:, :min_len]
    mfcc2_adjusted = mfcc2[:, :min_len]

    distance = np.linalg.norm(mfcc1_adjusted - mfcc2_adjusted)
    max_distance = (np.linalg.norm(mfcc1_adjusted) +
                    np.linalg.norm(mfcc2_adjusted))

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_mfcc(audio_signals: list, sample_rates: list,
                          /, *, n_fft: int = 8192) -> float:
    """
        Computes average MFCC-based similarity for all unique signal pairs using `compare_two_mfcc`,
        reflecting overall timbral coherence.

        Args:
            audio_signals (list[np.ndarray]): List of audio time-series arrays.
            sample_rates (list[float]): Corresponding sampling rates of each signal.

        Keyword Arguments:
            n_fft (int): Length of the FFT window for STFT used in MFCC extraction.

        Returns:
            float: Mean similarity score across all unique pairwise comparisons.

        Raises:
            ValueError: If the number of signals does not match the number of sample rates.

        See Also:
            compare_two_mfcc
    """
    if len(audio_signals) != len(sample_rates):
        raise ValueError("The number of signals must match the number of sampling rates")

    num_signals = len(audio_signals)
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(num_signals):
        for j in range(i + 1, num_signals):
            total_similarity += compare_two_mfcc(
                audio_signals[i], audio_signals[j],
                sample_rates[i], sample_rates[j],
                n_fft=n_fft
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
