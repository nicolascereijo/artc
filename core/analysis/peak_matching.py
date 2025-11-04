import numpy as np
import librosa


def calculate_peak_matching(audio_signal: np.ndarray, sample_rate: float,
                            /, *, n_fft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """
        Identifies prominent spectral peaks by computing the STFT magnitude, averaging across time,
        and picking peaks in the decibel domain.

        Args:
            audio_signal (np.ndarray): Time-series array of the audio signal.
            sample_rate (float): Sampling rate (in Hz) of the audio signal.

        Keyword Arguments:
            n_fft (int): Length of the FFT window for spectral analysis.

        Returns:
            tuple[np.ndarray, np.ndarray]: Array of peak frequencies and corresponding magnitudes.
    """
    spectrogram = np.abs(librosa.stft(audio_signal, n_fft=n_fft))
    one_dimensional_spectrogram = np.mean(spectrogram, axis=1)

    # Peak picking parameters:
    # - pre_max, post_max (typical range: 3-10): Number of samples before/after the current point
    # that must be smaller for it to be considered a peak.
    # - pre_avg, post_avg (typical range: 3-10): Number of samples before/after the current point to
    # compute the local average, helping to smooth the signal.
    # - delta (typical range: 0.1-1.0): Minimum amplitude threshold that a peak must have to be
    # considered as such, detecting only the most prominent peaks.
    # - wait (typical range: 1-10): Minimum number of samples between successive peaks, preventing
    # the detection of peaks that are too close to each other.
    spectral_peaks = librosa.util.peak_pick(
        librosa.amplitude_to_db(one_dimensional_spectrogram),
        pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=5)

    fft_frequencies = librosa.core.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    peak_frequencies = fft_frequencies[spectral_peaks]
    peak_magnitudes = one_dimensional_spectrogram[spectral_peaks]

    return peak_frequencies, peak_magnitudes


def compare_two_peak_matching(audio_signal1: np.ndarray, audio_signal2: np.ndarray,
                              sample_rate1: float, sample_rate2: float,
                              /, *, n_fft: int = 4096) -> float:
    """
        Compares spectral peak patterns between two audio signals by extracting their peak sets and
        computing an average frequency-magnitude similarity score.

        Args:
            audio_signal1 (np.ndarray): First audio time-series array.
            audio_signal2 (np.ndarray): Second audio time-series array.
            sample_rate1 (float): Sampling rate (in Hz) of the first signal.
            sample_rate2 (float): Sampling rate (in Hz) of the second signal.

        Keyword Arguments:
            n_fft (int): Length of the FFT window for spectral analysis.

        Returns:
            float: Similarity score between 0 and 1, where 1 indicates identical peak sets.

        See Also:
            calculate_peak_matching
    """
    peak_freq1, peak_mag1 = calculate_peak_matching(audio_signal1, sample_rate1, n_fft=n_fft)
    peak_freq2, peak_mag2 = calculate_peak_matching(audio_signal2, sample_rate2, n_fft=n_fft)

    min_len_freq = min(len(peak_freq1), len(peak_freq2))
    min_len_mag = min(len(peak_mag1), len(peak_mag2))
    peak_freq1_adjusted = peak_freq1[:min_len_freq]
    peak_freq2_adjusted = peak_freq2[:min_len_freq]
    peak_mag1_adjusted = peak_mag1[:min_len_mag]
    peak_mag2_adjusted = peak_mag2[:min_len_mag]

    distance_freq = np.linalg.norm(np.abs(peak_freq1_adjusted - peak_freq2_adjusted))
    max_distance_freq = (np.linalg.norm(np.abs(peak_freq1_adjusted)) +
                         np.linalg.norm(np.abs(peak_freq2_adjusted)))
    distance_mag = np.linalg.norm(np.abs(peak_mag1_adjusted - peak_mag2_adjusted))
    max_distance_mag = (np.linalg.norm(np.abs(peak_mag1_adjusted)) +
                        np.linalg.norm(np.abs(peak_mag2_adjusted)))

    similarity_freq = (1 - distance_freq / max_distance_freq) if max_distance_freq > 0 else 1.0
    similarity_mag = (1 - distance_mag / max_distance_mag) if max_distance_mag > 0 else 1.0

    similarity = (similarity_freq + similarity_mag) / 2
    return float(similarity)


def compare_multiple_peak_matching(audio_signals: list, sample_rates: list,
                                   /, *, n_fft: int = 4096) -> float:
    """
        Computes average spectral peak similarity across all unique signal pairs using
        `compare_two_peak_matching`, reflecting overall spectral feature coherence.

        Args:
            audio_signals (list[np.ndarray]): List of audio time-series arrays.
            sample_rates (list[float]): Corresponding sampling rates of each signal.

        Keyword Arguments:
            n_fft (int): Length of the FFT window for spectral analysis.

        Returns:
            float: Mean similarity score across all unique pairwise comparisons.

        Raises:
            ValueError: If the number of signals does not match the number of sample rates.

        See Also:
            compare_two_peak_matching
    """
    if len(audio_signals) != len(sample_rates):
        raise ValueError("The number of signals must match the number of sampling rates")

    num_signals = len(audio_signals)
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(num_signals):
        for j in range(i + 1, num_signals):
            total_similarity += compare_two_peak_matching(
                audio_signals[i], audio_signals[j],
                sample_rates[i], sample_rates[j],
                n_fft=n_fft
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
