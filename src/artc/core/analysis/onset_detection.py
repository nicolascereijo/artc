import numpy as np
import numpy.typing as npt
from librosa.onset import onset_strength

import artc.core.configurations as config

from ..datastructures.harmonize import (
    adjust_dimensions,
    check_matching_sample_rates,
)


def calculate_onset_detection(
    audio_signal: npt.NDArray[np.float32],
    sample_rate: float,
    /,
    *,
    hop_length: int | None = None,
) -> npt.NDArray[np.float64]:
    """Computes the onset strength envelope of the audio signal, indicating the
    likelihood of onsets, for example note attacks, over time.

    Args:
        audio_signal: Time series array of the audio signal.
        sample_rate: Sampling rate, in Hz, of the audio signal.
        hop_length: Number of samples between successive analysis frames.
            Defaults to the 'onset_detection' entry of
            '[metric.window_parameter]' in the TOML.

    Returns:
        Onset strength envelope as a 1D array of length 'frames'.
    """
    if hop_length is None:
        hop_length = int(
            config.read_config(("window_parameter", "onset_detection"))
        )
    return onset_strength(
        y=audio_signal, sr=sample_rate, hop_length=hop_length
    )


def compare_two_onset_detection(
    audio_signal1: npt.NDArray[np.float32],
    audio_signal2: npt.NDArray[np.float32],
    sample_rate1: float,
    sample_rate2: float,
    /,
    *,
    hop_length: int | None = None,
) -> float:
    """Compares onset strength envelopes between two audio signals and returns
    a normalized similarity score.

    Args:
        audio_signal1: First audio time series array.
        audio_signal2: Second audio time series array.
        sample_rate1: Sampling rate, in Hz, of the first signal.
        sample_rate2: Sampling rate, in Hz, of the second signal.
        hop_length: Number of samples between successive analysis frames. See
            'calculate_onset_detection'.

    Returns:
        Similarity score between 0 and 1, where 1 means identical onset
        patterns.
    """
    check_matching_sample_rates(sample_rate1, sample_rate2)

    onset_env1 = calculate_onset_detection(
        audio_signal1, sample_rate1, hop_length=hop_length
    )
    onset_env2 = calculate_onset_detection(
        audio_signal2, sample_rate2, hop_length=hop_length
    )

    onset_env1_adjusted, onset_env2_adjusted = adjust_dimensions(
        onset_env1, onset_env2
    )

    distance = np.linalg.norm(onset_env1_adjusted - onset_env2_adjusted)
    max_distance = np.linalg.norm(
        onset_env1_adjusted
    ) + np.linalg.norm(
        onset_env2_adjusted
    )

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return float(similarity)


def compare_multiple_onset_detection(
    audio_signals: list[npt.NDArray[np.float32]],
    sample_rates: list[float],
    /,
    *,
    hop_length: int | None = None,
) -> float:
    """Computes the average onset strength similarity for every unique signal
    pair using 'compare_two_onset_detection', reflecting overall onset pattern
    coherence.

    Args:
        audio_signals: List of audio time series arrays.
        sample_rates: Sampling rate, in Hz, of each signal, in the same order
            as 'audio_signals'.
        hop_length: Number of samples between successive analysis frames. See
            'calculate_onset_detection'.

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
            total_similarity += compare_two_onset_detection(
                audio_signals[i],
                audio_signals[j],
                sample_rates[i],
                sample_rates[j],
                hop_length=hop_length,
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
