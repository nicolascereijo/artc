import numpy as np
from librosa.beat import beat_track

import artc.core.configurations as config
from ..datastructures.harmonize import adjust_dimensions, check_matching_sample_rates


def calculate_beat_alignment(audio_signal: np.ndarray, sample_rate: float,
                             /, *, hop_length: int | None = None) -> np.ndarray:
    """
        Detects beat positions in the audio signal using a beat tracking algorithm and returns the
        frequency-domain representation of the beat activation sequence.

        Args:
            audio_signal (np.ndarray): Time-series array of the audio signal.
            sample_rate (float): Sampling rate (in Hz) of the audio signal.

        Keyword Arguments:
            hop_length (int): Number of samples between successive analysis frames.
                Defaults to the 'beat_alignment' entry of [metric.window_parameter] in the TOML.

        Returns:
            np.ndarray: FFT of the detected beat position sequence.
    """
    if hop_length is None:
        hop_length = int(config.read_config(("window_parameter", "beat_alignment")))
    _, beats = beat_track(y=audio_signal, sr=sample_rate, hop_length=hop_length)
    # np.fft.fft raises on empty input. No detectable pulse means zero beats.
    if len(beats) == 0:
        return np.array([])
    return np.fft.fft(beats)


def compare_two_beat_alignment(audio_signal1: np.ndarray, audio_signal2: np.ndarray,
                               sample_rate1: float, sample_rate2: float,
                               /, *, hop_length: int | None = None) -> float:
    """
        Compares beat alignment between two audio signals by computing their beat FFTs and
        calculating a normalized similarity score.

        Args:
            audio_signal1 (np.ndarray): First audio time-series array.
            audio_signal2 (np.ndarray): Second audio time-series array.
            sample_rate1 (float): Sampling rate (in Hz) of the first signal.
            sample_rate2 (float): Sampling rate (in Hz) of the second signal.

        Keyword Arguments:
            hop_length (int): Number of samples between successive analysis frames.

        Returns:
            float: Similarity score between 0 and 1, where 1 indicates perfect alignment.

        See Also:
            calculate_beat_alignment
    """
    check_matching_sample_rates(sample_rate1, sample_rate2)

    beats_1 = calculate_beat_alignment(audio_signal1, sample_rate1, hop_length=hop_length)
    beats_2 = calculate_beat_alignment(audio_signal2, sample_rate2, hop_length=hop_length)

    # Only equal if neither signal has beats. One-sided truncation below
    # would otherwise hide a real difference as a trivial match.
    if len(beats_1) == 0 or len(beats_2) == 0:
        return 1.0 if len(beats_1) == len(beats_2) else 0.0

    beats_1_adjusted, beats_2_adjusted = adjust_dimensions(beats_1, beats_2)

    distance = np.linalg.norm(np.abs(beats_1_adjusted) -
                              np.abs(beats_2_adjusted))
    max_distance = (np.linalg.norm(np.abs(beats_1_adjusted)) +
                    np.linalg.norm(np.abs(beats_2_adjusted)))

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_beat_alignment(audio_signals: list, sample_rates: list,
                                    /, *, hop_length: int | None = None) -> float:
    """
        Computes average beat-alignment similarity for all unique signal pairs using
        `compare_two_beat_alignment`, reflecting overall rhythmic coherence.

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
            compare_two_beat_alignment
    """
    if len(audio_signals) != len(sample_rates):
        raise ValueError("The number of signals must match the number of sampling rates")

    num_signals = len(audio_signals)
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(num_signals):
        for j in range(i + 1, num_signals):
            total_similarity += compare_two_beat_alignment(
                audio_signals[i], audio_signals[j],
                sample_rates[i], sample_rates[j],
                hop_length=hop_length
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
