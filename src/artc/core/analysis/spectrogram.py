import numpy as np
from librosa import stft


def calculate_spectrogram(audio_signal: np.ndarray,
                          /, *, n_fft: int = 4096) -> np.ndarray:
    """
        Computes magnitude spectrogram of the audio signal using short-time Fourier transform and
        returns its frequency-domain representation.

        Args:
            audio_signal (np.ndarray): Time-series array of the audio signal.

        Keyword Arguments:
            n_fft (int): Length of the FFT window for spectral analysis.

        Returns:
            np.ndarray: FFT of the magnitude spectrogram matrix.
    """
    spectrogram = np.abs(stft(audio_signal, n_fft=n_fft))
    return np.fft.fft(spectrogram)


def compare_two_spectrogram(audio_signal1: np.ndarray, audio_signal2: np.ndarray,
                            /, *, n_fft: int = 4096) -> float:
    """
        Compares spectrograms between two audio signals by computing their FFTs and returning a
        normalized similarity score.

        Args:
            audio_signal1 (np.ndarray): First audio time-series array.
            audio_signal2 (np.ndarray): Second audio time-series array.

        Keyword Arguments:
            n_fft (int): Length of the FFT window for spectral analysis.

        Returns:
            float: Similarity score between 0 and 1, where 1 indicates perfect match.

        See Also:
            calculate_spectrogram
    """
    spectrogram_1 = calculate_spectrogram(audio_signal1, n_fft=n_fft)
    spectrogram_2 = calculate_spectrogram(audio_signal2, n_fft=n_fft)

    min_len = min(spectrogram_1.shape[1], spectrogram_2.shape[1])
    spectrogram_1_adjusted = spectrogram_1[:, :min_len]
    spectrogram_2_adjusted = spectrogram_2[:, :min_len]

    distance = np.linalg.norm(np.abs(spectrogram_1_adjusted) -
                              np.abs(spectrogram_2_adjusted))
    max_distance = (np.linalg.norm(np.abs(spectrogram_1_adjusted)) +
                    np.linalg.norm(np.abs(spectrogram_2_adjusted)))

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_spectrogram(audio_signals: list,
                                 /, *, n_fft: int = 4096) -> float:
    """
        Computes average spectrogram similarity for all unique signal pairs using
        `compare_two_spectrogram`, reflecting overall spectral content coherence.

        Args:
            audio_signals (list[np.ndarray]): List of audio time-series arrays.

        Keyword Arguments:
            n_fft (int): Length of the FFT window for spectral analysis.

        Returns:
            float: Mean similarity score across all unique pairwise comparisons.

        See Also:
            compare_two_spectrogram
    """
    num_signals = len(audio_signals)
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(num_signals):
        for j in range(i + 1, num_signals):
            total_similarity += compare_two_spectrogram(
                audio_signals[i], audio_signals[j],
                n_fft=n_fft
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
