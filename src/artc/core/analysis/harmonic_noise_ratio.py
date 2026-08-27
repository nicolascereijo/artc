import numpy as np
import numpy.typing as npt
from librosa import stft

import artc.core.configurations as config


def calculate_harmonic_noise_ratio(
    audio_signal: npt.NDArray[np.float32],
    /,
    *,
    n_fft: int | None = None,
    hop_length: int | None = None,
) -> float:
    """Computes the ratio of harmonic to noise components in the audio signal
    by separating harmonic and percussive parts and measuring their power.

    Args:
        audio_signal: Time series array of the audio signal.
        n_fft: FFT window length for STFT analysis.
        hop_length: Number of samples between successive analysis frames. Both
            'n_fft' and 'hop_length' default to the 'harmonic_noise_ratio'
            entry of '[metric.window_parameter]' in the TOML.

    Returns:
        Harmonic to noise ratio (HNR), where higher values indicate greater
        harmonic dominance.
    """
    # This import is deferred because librosa.effects.hpss pulls in
    # librosa.decompose, which in turn pulls in scikit-learn. That is a real
    # cost, and it should only be paid when HNR is actually computed, not on
    # every 'import artc.core'.
    from librosa.effects import hpss

    if n_fft is None:
        n_fft = int(
            config.read_config(("window_parameter", "harmonic_noise_ratio"))
        )
    if hop_length is None:
        hop_length = int(
            config.read_config(("window_parameter", "harmonic_noise_ratio"))
        )
    harmonic, percussive = hpss(y=audio_signal)

    harmonic_power = np.sum(
        np.abs(stft(harmonic, n_fft=n_fft, hop_length=hop_length)) ** 2
    )
    percussive_power = np.sum(
        np.abs(stft(percussive, n_fft=n_fft, hop_length=hop_length)) ** 2
    )

    total_power = harmonic_power + percussive_power
    return float(harmonic_power / total_power if total_power > 0 else 0.0)


def compare_two_hnr(
    audio_signal1: npt.NDArray[np.float32],
    audio_signal2: npt.NDArray[np.float32],
    /,
    *,
    n_fft: int | None = None,
    hop_length: int | None = None,
) -> float:
    """Compares harmonic to noise ratios between two audio signals and returns
    a normalized similarity score.

    Args:
        audio_signal1: First audio time series array.
        audio_signal2: Second audio time series array.
        n_fft: FFT window length for STFT analysis.
        hop_length: Number of samples between successive analysis frames.

    Returns:
        Similarity score between 0 and 1, where 1 indicates identical HNR.
    """
    hnr1 = calculate_harmonic_noise_ratio(
        audio_signal1, n_fft=n_fft, hop_length=hop_length
    )
    hnr2 = calculate_harmonic_noise_ratio(
        audio_signal2, n_fft=n_fft, hop_length=hop_length
    )

    distance = abs(hnr1 - hnr2)
    max_distance = max(abs(hnr1), abs(hnr2))

    similarity = (1 - distance / max_distance) if max_distance > 0 else 1.0
    return similarity


def compare_multiple_hnr(
    audio_signals: list[npt.NDArray[np.float32]],
    /,
    *,
    n_fft: int | None = None,
    hop_length: int | None = None,
) -> float:
    """Computes average HNR similarity for all unique signal pairs, reflecting
    overall harmonic versus noise coherence.

    Args:
        audio_signals: List of audio time series arrays.
        n_fft: FFT window length for STFT analysis.
        hop_length: Number of samples between successive analysis frames.

    Returns:
        Mean similarity score across all unique pairwise comparisons.
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
