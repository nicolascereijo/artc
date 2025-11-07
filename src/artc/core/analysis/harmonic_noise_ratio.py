import numpy as np
from librosa import stft
from librosa.effects import hpss


def calculate_harmonic_noise_ratio(audio_signal: np.ndarray,
                                   /, *, n_fft: int = 512, hop_length: int = 512) -> float:
    """
        Computes the ratio of harmonic to noise components in the audio signal by separating
        harmonic and percussive parts and measuring their power.

        Args:
            audio_signal (np.ndarray): Time-series array of the audio signal.

        Keyword Arguments:
            n_fft (int): FFT window length for STFT analysis.
            hop_length (int): Number of samples between successive analysis frames.

        Returns:
            float: Harmonic-to-noise ratio (HNR), where higher values indicate
            greater harmonic dominance.
    """
    harmonic, percussive = hpss(y=audio_signal)

    harmonic_power = np.sum(np.abs(stft(harmonic, n_fft=n_fft, hop_length=hop_length))**2)
    percussive_power = np.sum(np.abs(stft(percussive, n_fft=n_fft, hop_length=hop_length))**2)

    total_power = harmonic_power + percussive_power
    return float(harmonic_power / total_power if total_power > 0 else 0.0)


def compare_two_hnr(audio_signal1: np.ndarray, audio_signal2: np.ndarray,
                    /, *, n_fft: int = 512, hop_length: int = 512) -> float:
    """
        Compares harmonic-to-noise ratios between two audio signals and returns a normalized
        similarity score.

        Args:
            audio_signal1 (np.ndarray): First audio time-series array.
            audio_signal2 (np.ndarray): Second audio time-series array.

        Keyword Arguments:
            n_fft (int): FFT window length for STFT analysis.
            hop_length (int): Number of samples between successive analysis frames.

        Returns:
            float: Similarity score between 0 and 1, where 1 indicates identical HNR.

        See Also:
            calculate_harmonic_noise_ratio
    """
    hnr1 = calculate_harmonic_noise_ratio(audio_signal1, n_fft=n_fft, hop_length=hop_length)
    hnr2 = calculate_harmonic_noise_ratio(audio_signal2, n_fft=n_fft, hop_length=hop_length)

    distance = abs(hnr1 - hnr2)
    max_distance = max(abs(hnr1), abs(hnr2))

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return similarity


def compare_multiple_hnr(audio_signals: list,
                         /, *, n_fft: int = 512, hop_length: int = 512) -> float:
    """
        Computes average HNR similarity for all unique signal pairs using `compare_two_hnr`,
        reflecting overall harmonic versus noise coherence.

        Args:
            audio_signals (list[np.ndarray]): List of audio time-series arrays.

        Keyword Arguments:
            n_fft (int): FFT window length for STFT analysis.
            hop_length (int): Number of samples between successive analysis frames.

        Returns:
            float: Mean similarity score across all unique pairwise comparisons.

        See Also:
            compare_two_hnr
    """
    num_signals = len(audio_signals)
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(num_signals):
        for j in range(i + 1, num_signals):
            total_similarity += compare_two_hnr(
                audio_signals[i], audio_signals[j],
                n_fft=n_fft, hop_length=hop_length
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
