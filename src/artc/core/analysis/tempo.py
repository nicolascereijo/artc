import numpy as np
import numpy.typing as npt
from librosa.beat import beat_track

import artc.core.configurations as config


def calculate_tempo(
    audio_signal: npt.NDArray[np.float32],
    sample_rate: float,
    /,
    *,
    hop_length: int | None = None,
) -> float:
    """Estimates the global tempo, in BPM, of the audio signal using beat
    tracking.

    Args:
        audio_signal: Time series array of the audio signal.
        sample_rate: Sampling rate, in Hz, of the audio signal.
        hop_length: Number of samples between successive analysis frames.
            Defaults to the 'tempo' entry of '[metric.window_parameter]' in
            the TOML.

    Returns:
        The estimated tempo, in beats per minute, or 'NaN' if no pulse was
        detected, as opposed to a genuine 0 BPM reading.
    """
    if hop_length is None:
        hop_length = int(config.read_config(("window_parameter", "tempo")))
    tempo, beats = beat_track(
        y=audio_signal, sr=sample_rate, hop_length=hop_length
    )

    if isinstance(tempo, np.ndarray):
        tempo = np.mean(tempo)
    if len(beats) == 0:
        return float("nan")
    return float(tempo)


def compare_two_tempo(
    audio_signal1: npt.NDArray[np.float32],
    audio_signal2: npt.NDArray[np.float32],
    sample_rate1: float,
    sample_rate2: float,
    /,
    *,
    hop_length: int | None = None,
) -> float:
    """Compares the estimated tempo between two audio signals by computing
    their BPMs and returning a normalized similarity score.

    Args:
        audio_signal1: First audio time series array.
        audio_signal2: Second audio time series array.
        sample_rate1: Sampling rate, in Hz, of the first signal.
        sample_rate2: Sampling rate, in Hz, of the second signal.
        hop_length: Number of samples between successive analysis frames.

    Returns:
        A similarity score between 0 and 1, where 1 indicates an identical
        tempo.
    """
    tempo1 = calculate_tempo(
        audio_signal1, sample_rate1, hop_length=hop_length
    )
    tempo2 = calculate_tempo(
        audio_signal2, sample_rate2, hop_length=hop_length
    )

    # Only equal if neither signal has a detectable pulse. One 'NaN' and one
    # real tempo would otherwise compare as if both were genuinely 0 BPM.
    if np.isnan(tempo1) or np.isnan(tempo2):
        return 1.0 if np.isnan(tempo1) and np.isnan(tempo2) else 0.0

    distance = abs(tempo1 - tempo2)
    max_distance = max(tempo1, tempo2)

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return max(0.0, similarity)


def compare_multiple_tempo(
    audio_signals: list[npt.NDArray[np.float32]],
    sample_rates: list[float],
    /,
    *,
    hop_length: int | None = None,
) -> float:
    """Computes the average tempo similarity for every unique signal pair,
    using 'compare_two_tempo', reflecting the overall tempo coherence.

    Args:
        audio_signals: List of audio time series arrays.
        sample_rates: Sampling rate, in Hz, of each signal, in the same order
            as 'audio_signals'.
        hop_length: Number of samples between successive analysis frames.

    Returns:
        The mean similarity score across every unique pairwise comparison.

    Raises:
        ValueError: If 'audio_signals' and 'sample_rates' have different
            lengths.
    """
    if len(audio_signals) != len(sample_rates):
        raise ValueError(
            "The number of signals must match the number of sampling rates"
        )

    num_signals = len(audio_signals)
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(num_signals):
        for j in range(i + 1, num_signals):
            total_similarity += compare_two_tempo(
                audio_signals[i],
                audio_signals[j],
                sample_rates[i],
                sample_rates[j],
                hop_length=hop_length,
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
