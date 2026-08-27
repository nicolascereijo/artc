from collections.abc import Callable
from typing import TypedDict

from .beat_alignment import (
    compare_multiple_beat_alignment,
    compare_two_beat_alignment,
)
from .chroma_cens import compare_multiple_chroma_cens, compare_two_chroma_cens
from .chroma_stft import compare_multiple_chroma_stft, compare_two_chroma_stft
from .dynamic_time_warping import compare_multiple_dtw, compare_two_dtw
from .energy_envelope import (
    compare_multiple_energy_envelope,
    compare_two_energy_envelope,
)
from .harmonic_noise_ratio import compare_multiple_hnr, compare_two_hnr
from .harmonic_tempogram import (
    compare_multiple_harmonic_tempogram,
    compare_two_harmonic_tempogram,
)
from .loudness import compare_multiple_loudness, compare_two_loudness
from .mfcc import compare_multiple_mfcc, compare_two_mfcc
from .onset_detection import (
    compare_multiple_onset_detection,
    compare_two_onset_detection,
)
from .peak_matching import (
    compare_multiple_peak_matching,
    compare_two_peak_matching,
)
from .pitch import compare_multiple_pitch, compare_two_pitch
from .spectral_bandwidth import (
    compare_multiple_spectral_bandwidth,
    compare_two_spectral_bandwidth,
)
from .spectral_centroid import (
    compare_multiple_spectral_centroid,
    compare_two_spectral_centroid,
)
from .spectral_contrast import (
    compare_multiple_spectral_contrast,
    compare_two_spectral_contrast,
)
from .spectral_flatness import (
    compare_multiple_spectral_flatness,
    compare_two_spectral_flatness,
)
from .spectral_roll_off import (
    compare_multiple_spectral_roll_off,
    compare_two_spectral_roll_off,
)
from .spectrogram import compare_multiple_spectrogram, compare_two_spectrogram
from .tempo import compare_multiple_tempo, compare_two_tempo
from .tempogram import compare_multiple_tempogram, compare_two_tempogram
from .temporal_centroid import (
    compare_multiple_temporal_centroid,
    compare_two_temporal_centroid,
)
from .temporal_flux import (
    compare_multiple_temporal_flux,
    compare_two_temporal_flux,
)
from .weighted_cyclic_tempogram import compare_multiple_wct, compare_two_wct
from .zero_crossing_rate import compare_multiple_zcr, compare_two_zcr

__all__ = [
    'COMPARE_FUNCTIONS',
    'compare_multiple_beat_alignment',
    'compare_multiple_chroma_cens',
    'compare_multiple_chroma_stft',
    'compare_multiple_dtw',
    'compare_multiple_energy_envelope',
    'compare_multiple_harmonic_tempogram',
    'compare_multiple_hnr',
    'compare_multiple_loudness',
    'compare_multiple_mfcc',
    'compare_multiple_onset_detection',
    'compare_multiple_peak_matching',
    'compare_multiple_pitch',
    'compare_multiple_spectral_bandwidth',
    'compare_multiple_spectral_centroid',
    'compare_multiple_spectral_contrast',
    'compare_multiple_spectral_flatness',
    'compare_multiple_spectral_roll_off',
    'compare_multiple_spectrogram',
    'compare_multiple_tempo',
    'compare_multiple_tempogram',
    'compare_multiple_temporal_centroid',
    'compare_multiple_temporal_flux',
    'compare_multiple_wct',
    'compare_multiple_zcr',
    'compare_two_beat_alignment',
    'compare_two_chroma_cens',
    'compare_two_chroma_stft',
    'compare_two_dtw',
    'compare_two_energy_envelope',
    'compare_two_harmonic_tempogram',
    'compare_two_hnr',
    'compare_two_loudness',
    'compare_two_mfcc',
    'compare_two_onset_detection',
    'compare_two_peak_matching',
    'compare_two_pitch',
    'compare_two_spectral_bandwidth',
    'compare_two_spectral_centroid',
    'compare_two_spectral_contrast',
    'compare_two_spectral_flatness',
    'compare_two_spectral_roll_off',
    'compare_two_spectrogram',
    'compare_two_tempo',
    'compare_two_tempogram',
    'compare_two_temporal_centroid',
    'compare_two_temporal_flux',
    'compare_two_wct',
    'compare_two_zcr',
    'get_metric_names',
]


class MetricFunctions(TypedDict):
    """One metric's comparison callables and whether they need sample rates."""

    compare_two: Callable[..., float]
    compare_multiple: Callable[..., float]
    use_sample_rate: bool


COMPARE_FUNCTIONS: dict[str, MetricFunctions] = {
    "beat_alignment": {
        "compare_two": compare_two_beat_alignment,
        "compare_multiple": compare_multiple_beat_alignment,
        "use_sample_rate": True
    },
    "chroma_cens": {
        "compare_two": compare_two_chroma_cens,
        "compare_multiple": compare_multiple_chroma_cens,
        "use_sample_rate": True
    },
    "chroma_stft": {
        "compare_two": compare_two_chroma_stft,
        "compare_multiple": compare_multiple_chroma_stft,
        "use_sample_rate": True
    },
    "dynamic_time_warping": {
        "compare_two": compare_two_dtw,
        "compare_multiple": compare_multiple_dtw,
        "use_sample_rate": True
    },
    "energy_envelope": {
        "compare_two": compare_two_energy_envelope,
        "compare_multiple": compare_multiple_energy_envelope,
        "use_sample_rate": False
    },
    "harmonic_noise_ratio": {
        "compare_two": compare_two_hnr,
        "compare_multiple": compare_multiple_hnr,
        "use_sample_rate": False
    },
    "harmonic_tempogram": {
        "compare_two": compare_two_harmonic_tempogram,
        "compare_multiple": compare_multiple_harmonic_tempogram,
        "use_sample_rate": True
    },
    "loudness": {
        "compare_two": compare_two_loudness,
        "compare_multiple": compare_multiple_loudness,
        "use_sample_rate": True
    },
    "mfcc": {
        "compare_two": compare_two_mfcc,
        "compare_multiple": compare_multiple_mfcc,
        "use_sample_rate": True
    },
    "onset_detection": {
        "compare_two": compare_two_onset_detection,
        "compare_multiple": compare_multiple_onset_detection,
        "use_sample_rate": True
    },
    "peak_matching": {
        "compare_two": compare_two_peak_matching,
        "compare_multiple": compare_multiple_peak_matching,
        "use_sample_rate": True
    },
    "pitch": {
        "compare_two": compare_two_pitch,
        "compare_multiple": compare_multiple_pitch,
        "use_sample_rate": True
    },
    "spectral_bandwidth": {
        "compare_two": compare_two_spectral_bandwidth,
        "compare_multiple": compare_multiple_spectral_bandwidth,
        "use_sample_rate": True
    },
    "spectral_centroid": {
        "compare_two": compare_two_spectral_centroid,
        "compare_multiple": compare_multiple_spectral_centroid,
        "use_sample_rate": True
    },
    "spectral_contrast": {
        "compare_two": compare_two_spectral_contrast,
        "compare_multiple": compare_multiple_spectral_contrast,
        "use_sample_rate": True
    },
    "spectral_flatness": {
        "compare_two": compare_two_spectral_flatness,
        "compare_multiple": compare_multiple_spectral_flatness,
        "use_sample_rate": False
    },
    "spectral_roll_off": {
        "compare_two": compare_two_spectral_roll_off,
        "compare_multiple": compare_multiple_spectral_roll_off,
        "use_sample_rate": True
    },
    "spectrogram": {
        "compare_two": compare_two_spectrogram,
        "compare_multiple": compare_multiple_spectrogram,
        "use_sample_rate": False
    },
    "tempo": {
        "compare_two": compare_two_tempo,
        "compare_multiple": compare_multiple_tempo,
        "use_sample_rate": True
    },
    "tempogram": {
        "compare_two": compare_two_tempogram,
        "compare_multiple": compare_multiple_tempogram,
        "use_sample_rate": True
    },
    "temporal_centroid": {
        "compare_two": compare_two_temporal_centroid,
        "compare_multiple": compare_multiple_temporal_centroid,
        "use_sample_rate": True
    },
    "temporal_flux": {
        "compare_two": compare_two_temporal_flux,
        "compare_multiple": compare_multiple_temporal_flux,
        "use_sample_rate": True
    },
    "weighted_cyclic_tempogram": {
        "compare_two": compare_two_wct,
        "compare_multiple": compare_multiple_wct,
        "use_sample_rate": True
    },
    "zero_crossing_rate": {
        "compare_two": compare_two_zcr,
        "compare_multiple": compare_multiple_zcr,
        "use_sample_rate": False
    }
}


def get_metric_names() -> list[str]:
    """Returns the name of every metric registered in 'COMPARE_FUNCTIONS'."""
    return list(COMPARE_FUNCTIONS.keys())
