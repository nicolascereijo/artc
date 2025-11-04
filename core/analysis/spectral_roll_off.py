import numpy as np
from librosa.feature import spectral_rolloff


def calculate_spectral_roll_off(audio_signal: np.ndarray, sample_rate: float,
                                /, *, n_fft: int = 512, roll_percent: float = 0.5) -> np.ndarray:
    """
        Computes spectral roll-off frequency for each frame and returns its frequency-domain
        representation.

        Args:
            audio_signal (np.ndarray): Time-series array of the audio signal.
            sample_rate (float): Sampling rate (in Hz) of the audio signal.

        Keyword Arguments:
            n_fft (int): Length of the FFT window for spectral analysis.
            roll_percent (float): Fraction of spectral energy below the roll-off frequency.

        Returns:
            np.ndarray: FFT of the spectral roll-off sequence.
    """
    roll_off = spectral_rolloff(y=audio_signal, sr=sample_rate,
                                n_fft=n_fft, roll_percent=roll_percent)
    return np.fft.fft(roll_off)


def compare_two_spectral_roll_off(audio_signal1: np.ndarray, audio_signal2: np.ndarray,
                                  sample_rate1: float, sample_rate2: float,
                                  /, *, n_fft: int = 512, roll_percent: float = 0.5) -> float:
    """
        Compares spectral roll-off between two audio signals by computing their roll-off FFTs and
        returning a normalized similarity score.

        Args:
            audio_signal1 (np.ndarray): First audio time-series array.
            audio_signal2 (np.ndarray): Second audio time-series array.
            sample_rate1 (float): Sampling rate (in Hz) of the first signal.
            sample_rate2 (float): Sampling rate (in Hz) of the second signal.

        Keyword Arguments:
            n_fft (int): Length of the FFT window for spectral analysis.
            roll_percent (float): Fraction of spectral energy below the roll-off frequency.

        Returns:
            float: Similarity score between 0 and 1, where 1 indicates identical roll-off patterns.

        See Also:
            calculate_spectral_roll_off
    """
    roll_off_1 = calculate_spectral_roll_off(
        audio_signal1, sample_rate1, n_fft=n_fft, roll_percent=roll_percent)
    roll_off_2 = calculate_spectral_roll_off(
        audio_signal2, sample_rate2, n_fft=n_fft, roll_percent=roll_percent)

    min_len = min(roll_off_1.shape[1], roll_off_2.shape[1])
    roll_off_1_adjusted = roll_off_1[:, :min_len]
    roll_off_2_adjusted = roll_off_2[:, :min_len]

    distance = np.linalg.norm(np.abs(roll_off_1_adjusted) -
                              np.abs(roll_off_2_adjusted))
    max_distance = (np.linalg.norm(np.abs(roll_off_1_adjusted)) +
                    np.linalg.norm(np.abs(roll_off_2_adjusted)))

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_spectral_roll_off(audio_signals: list, sample_rates: list,
                                       /, *, n_fft: int = 512, roll_percent: float = 0.5) -> float:
    """
        Computes average spectral roll-off similarity for all unique signal pairs using
        `compare_two_spectral_roll_off`, reflecting overall spectral shape coherence.

        Args:
            audio_signals (list[np.ndarray]): List of audio time-series arrays.
            sample_rates (list[float]): Corresponding sampling rates of each signal.

        Keyword Arguments:
            n_fft (int): Length of the FFT window for spectral analysis.
            roll_percent (float): Fraction of spectral energy below the roll-off frequency.

        Returns:
            float: Mean similarity score across all unique pairwise comparisons.

        Raises:
            ValueError: If the number of signals does not match the number of sampling rates.

        See Also:
            compare_two_spectral_roll_off
    """
    if len(audio_signals) != len(sample_rates):
        raise ValueError("The number of signals must match the number of sampling rates")

    num_signals = len(audio_signals)
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(num_signals):
        for j in range(i + 1, num_signals):
            total_similarity += compare_two_spectral_roll_off(
                audio_signals[i], audio_signals[j],
                sample_rates[i], sample_rates[j],
                n_fft=n_fft, roll_percent=roll_percent
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
