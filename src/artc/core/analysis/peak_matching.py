import librosa
import numpy as np
import numpy.typing as npt

import artc.core.configurations as config

from ..datastructures.harmonize import (
    adjust_dimensions,
    check_matching_sample_rates,
)


def calculate_peak_matching(
    audio_signal: npt.NDArray[np.float32],
    sample_rate: float,
    /,
    *,
    n_fft: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Identifies prominent spectral peaks by computing the STFT magnitude,
    averaging across time and picking peaks in the decibel domain.

    Args:
        audio_signal: Time series array of the audio signal.
        sample_rate: Sampling rate, in Hz, of the audio signal.
        n_fft: Length of the FFT window for spectral analysis. Defaults to the
            'peak_matching' entry of '[metric.window_parameter]' in the TOML.

    Returns:
        A tuple of the peak frequencies and their corresponding magnitudes.
    """
    if n_fft is None:
        n_fft = int(
            config.read_config(("window_parameter", "peak_matching"))
        )
    spectrogram = np.abs(librosa.stft(audio_signal, n_fft=n_fft))
    one_dimensional_spectrogram = np.mean(spectrogram, axis=1)

    # Peak picking parameters. 'pre_max' and 'post_max' (typically 3 to 10) are
    # how many neighboring samples on each side must be smaller for a point to
    # count as a peak. 'pre_avg' and 'post_avg' (typically 3 to 10) set the
    # window used for the local average that smooths the signal. 'delta'
    # (typically 0.1 to 1.0) is the minimum amplitude a peak must clear to be
    # considered prominent. 'wait' (typically 1 to 10) is the minimum number of
    # samples required between two consecutive peaks.
    spectral_peaks = librosa.util.peak_pick(
        librosa.amplitude_to_db(one_dimensional_spectrogram),
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=0.5,
        wait=5,
    )

    fft_frequencies = librosa.core.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    peak_frequencies = fft_frequencies[spectral_peaks]
    peak_magnitudes = one_dimensional_spectrogram[spectral_peaks]

    return peak_frequencies, peak_magnitudes


def compare_two_peak_matching(
    audio_signal1: npt.NDArray[np.float32],
    audio_signal2: npt.NDArray[np.float32],
    sample_rate1: float,
    sample_rate2: float,
    /,
    *,
    n_fft: int | None = None,
) -> float:
    """Compares spectral peak patterns between two audio signals by matching
    their peak sets position by position, not by frequency proximity, and
    computing an average frequency and magnitude similarity score.

    Args:
        audio_signal1: First audio time series array.
        audio_signal2: Second audio time series array.
        sample_rate1: Sampling rate, in Hz, of the first signal.
        sample_rate2: Sampling rate, in Hz, of the second signal.
        n_fft: Length of the FFT window for spectral analysis. See
            'calculate_peak_matching'.

    Returns:
        Similarity score between 0 and 1, where 1 means identical peak sets.
    """
    check_matching_sample_rates(sample_rate1, sample_rate2)

    peak_freq1, peak_mag1 = calculate_peak_matching(
        audio_signal1, sample_rate1, n_fft=n_fft
    )
    peak_freq2, peak_mag2 = calculate_peak_matching(
        audio_signal2, sample_rate2, n_fft=n_fft
    )

    # Only equal if neither signal has peaks. Truncating only one side
    # below would otherwise hide a real difference as a trivial match.
    if len(peak_freq1) == 0 or len(peak_freq2) == 0:
        return 1.0 if len(peak_freq1) == len(peak_freq2) else 0.0

    peak_freq1_adjusted, peak_freq2_adjusted = adjust_dimensions(
        peak_freq1, peak_freq2
    )
    peak_mag1_adjusted, peak_mag2_adjusted = adjust_dimensions(
        peak_mag1, peak_mag2
    )

    distance_freq = np.linalg.norm(
        np.abs(peak_freq1_adjusted - peak_freq2_adjusted)
    )
    max_distance_freq = np.linalg.norm(
        np.abs(peak_freq1_adjusted)
    ) + np.linalg.norm(
        np.abs(peak_freq2_adjusted)
    )
    distance_mag = np.linalg.norm(
        np.abs(peak_mag1_adjusted - peak_mag2_adjusted)
    )
    max_distance_mag = np.linalg.norm(
        np.abs(peak_mag1_adjusted)
    ) + np.linalg.norm(
        np.abs(peak_mag2_adjusted)
    )

    similarity_freq = (
        (1 - distance_freq / max_distance_freq)
        if max_distance_freq > 0 else 1.0
    )
    similarity_mag = (
        (1 - distance_mag / max_distance_mag)
        if max_distance_mag > 0 else 1.0
    )

    similarity = (similarity_freq + similarity_mag) / 2
    return float(similarity)


def compare_multiple_peak_matching(
    audio_signals: list[npt.NDArray[np.float32]],
    sample_rates: list[float],
    /,
    *,
    n_fft: int | None = None,
) -> float:
    """Computes the average spectral peak similarity across every unique signal
    pair using 'compare_two_peak_matching', reflecting overall spectral feature
    coherence.

    Args:
        audio_signals: List of audio time series arrays.
        sample_rates: Sampling rate, in Hz, of each signal, in the same order
            as 'audio_signals'.
        n_fft: Length of the FFT window for spectral analysis. See
            'calculate_peak_matching'.

    Returns:
        Mean similarity score across every unique pairwise comparison.

    Raises:
        ValueError: If 'audio_signals' and 'sample_rates' have different
            lengths.
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
            total_similarity += compare_two_peak_matching(
                audio_signals[i],
                audio_signals[j],
                sample_rates[i],
                sample_rates[j],
                n_fft=n_fft,
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
