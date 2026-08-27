import numpy as np
import numpy.typing as npt
from librosa.feature import zero_crossing_rate

import artc.core.configurations as config

from ..datastructures.harmonize import adjust_dimensions


def calculate_zcr(
    audio_signal: npt.NDArray[np.float32],
    /,
    *,
    frame_length: int = 2048,
    hop_length: int | None = None,
) -> npt.NDArray[np.float64]:
    """Computes the zero crossing rate, ZCR, of the audio signal over time
    frames.

    Args:
        audio_signal: Time series array of the audio signal.
        frame_length: Length of each analysis frame, in samples.
        hop_length: Number of samples between successive frames.
            Defaults to the 'zero_crossing_rate' entry of
            '[metric.window_parameter]' in the TOML.

    Returns:
        The ZCR sequence, as a 2D array of shape (1, frames).
    """
    if audio_signal.size == 0:
        return np.zeros((1, 0), dtype=np.float64)
    if hop_length is None:
        hop_length = int(
            config.read_config(("window_parameter", "zero_crossing_rate"))
        )
    return zero_crossing_rate(
        y=audio_signal, frame_length=frame_length, hop_length=hop_length
    )


def compare_two_zcr(
    signal1: npt.NDArray[np.float32],
    signal2: npt.NDArray[np.float32],
    /,
    *,
    frame_length: int = 2048,
    hop_length: int | None = None,
) -> float:
    """Compares zero crossing rate sequences of two audio signals.

    Args:
        signal1: First audio time series array.
        signal2: Second audio time series array.
        frame_length: Length of each analysis frame, in samples.
        hop_length: Number of samples between successive frames.

    Returns:
        Similarity score between 0 and 1, where 1 means identical ZCR
        profiles.
    """
    zcr1 = calculate_zcr(
        signal1, frame_length=frame_length, hop_length=hop_length
    )
    zcr2 = calculate_zcr(
        signal2, frame_length=frame_length, hop_length=hop_length
    )

    # A zero length input signal produces a zero length ZCR sequence, which
    # 'np.percentile' and 'max' can't handle. If both signals are empty they
    # are trivially identical. If only one is empty they are trivially as
    # different as possible, and nothing needs to be truncated.
    if zcr1.size == 0 or zcr2.size == 0:
        return 1.0 if zcr1.size == 0 and zcr2.size == 0 else 0.0

    # Clips each sequence to its own 95th percentile before normalizing.
    # A single isolated spike in ZCR, for example a brief burst of noise,
    # would otherwise dominate the max based reference, collapsing the rest of
    # that signal's normalized values toward zero and making it look
    # artificially dissimilar from a signal with the same overall shape but no
    # such spike.
    cap1 = np.percentile(zcr1, 95)
    cap2 = np.percentile(zcr2, 95)
    zcr1_clipped = np.clip(zcr1, None, cap1)
    zcr2_clipped = np.clip(zcr2, None, cap2)

    zcr1_normalized = (
        zcr1_clipped / zcr1_clipped.max()
        if zcr1_clipped.max() > 0
        else zcr1_clipped
    )
    zcr2_normalized = (
        zcr2_clipped / zcr2_clipped.max()
        if zcr2_clipped.max() > 0
        else zcr2_clipped
    )

    zcr1_adjusted, zcr2_adjusted = adjust_dimensions(
        zcr1_normalized, zcr2_normalized
    )

    relative_difference = np.abs(zcr1_adjusted - zcr2_adjusted) / (
        np.abs(zcr1_adjusted) + np.abs(zcr2_adjusted) + 1e-8
    )
    similarity = 1.0 - np.mean(relative_difference)
    return float(max(0.0, similarity))


def compare_multiple_zcr(
    audio_signals: list[npt.NDArray[np.float32]],
    /,
    *,
    frame_length: int = 2048,
    hop_length: int | None = None,
) -> float:
    """Computes average ZCR similarity across every unique pair of signals.

    Args:
        audio_signals: List of audio time series arrays.
        frame_length: Length of each analysis frame, in samples.
        hop_length: Number of samples between successive frames.

    Returns:
        Mean similarity score across every unique pairwise comparison.
    """
    num_signals = len(audio_signals)
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(num_signals):
        for j in range(i + 1, num_signals):
            total_similarity += compare_two_zcr(
                audio_signals[i],
                audio_signals[j],
                frame_length=frame_length,
                hop_length=hop_length,
            )
            num_comparisons += 1

    return total_similarity / num_comparisons if num_comparisons > 0 else 0.0
