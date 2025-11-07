import numpy as np
import librosa


def calculate_loudness(audio_signal: np.ndarray, sample_rate: float) -> np.ndarray:
    """
        Computes A-weighted loudness of the audio signal by converting its magnitude spectrogram to
        decibels, applying A-weighting, and returning the frequency-domain representation.

        Args:
            audio_signal (np.ndarray): Time-series array of the audio signal.
            sample_rate (float): Sampling rate (in Hz) of the audio signal.

        Returns:
            np.ndarray: FFT of the A-weighted decibel spectrogram.
    """
    magnitude_spectrogram = np.abs(librosa.stft(audio_signal))
    db_spectrogram = librosa.amplitude_to_db(magnitude_spectrogram, ref=np.max)

    frequencies = librosa.fft_frequencies(sr=sample_rate)
    frequencies[frequencies == 0] = 1e-6  # Avoid log10(0)

    weighting = librosa.A_weighting(frequencies)[:, np.newaxis]
    return np.fft.fft(db_spectrogram * weighting)


def compare_two_loudness(audio_signal1: np.ndarray, audio_signal2: np.ndarray,
                         sample_rate1: float, sample_rate2: float, /) -> float:
    """
        Compares loudness profiles between two audio signals by computing their A-weighted
        spectrogram FFTs and returning a normalized similarity score.

        Args:
            audio_signal1 (np.ndarray): First audio time-series array.
            audio_signal2 (np.ndarray): Second audio time-series array.
            sample_rate1 (float): Sampling rate (in Hz) of the first signal.
            sample_rate2 (float): Sampling rate (in Hz) of the second signal.

        Returns:
            float: Similarity score between 0 and 1, where 1 indicates identical loudness patterns.

        See Also:
            calculate_loudness
    """
    loudness1 = calculate_loudness(audio_signal1, sample_rate1)
    loudness2 = calculate_loudness(audio_signal2, sample_rate2)

    min_len = min(loudness1.shape[1], loudness2.shape[1])
    loudness1_adjusted = loudness1[:, :min_len]
    loudness2_adjusted = loudness2[:, :min_len]

    distance = np.linalg.norm(loudness1_adjusted - loudness2_adjusted)
    max_distance = (np.linalg.norm(loudness1_adjusted) +
                    np.linalg.norm(loudness2_adjusted))

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_loudness(audio_signals: list, sample_rates: list, /) -> float:
    """
        Computes average loudness similarity for all unique signal pairs using
        `compare_two_loudness`, reflecting overall loudness pattern coherence.

        Args:
            audio_signals (list[np.ndarray]): List of audio time-series arrays.
            sample_rates  (list[float]): Corresponding sampling rates of each signal.

        Returns:
            float: Mean similarity score across all unique pairwise comparisons.

        See Also:
            compare_two_loudness
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
