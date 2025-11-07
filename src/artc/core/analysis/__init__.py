from .beat_alignment import compare_two_beat_alignment, compare_multiple_beat_alignment
from .chroma_cens import compare_two_chroma_cens, compare_multiple_chroma_cens
from .chroma_stft import compare_two_chroma_stft, compare_multiple_chroma_stft
from .dynamic_time_warping import compare_two_dtw, compare_multiple_dtw
from .energy_envelope import compare_two_energy_envelope, compare_multiple_energy_envelope
from .harmonic_noise_ratio import compare_two_hnr, compare_multiple_hnr
from .harmonic_tempogram import compare_two_harmonic_tempogram, compare_multiple_harmonic_tempogram
from .loudness import compare_two_loudness, compare_multiple_loudness
from .mfcc import compare_two_mfcc, compare_multiple_mfcc
from .onset_detection import compare_two_onset_detection, compare_multiple_onset_detection
from .peak_matching import compare_two_peak_matching, compare_multiple_peak_matching
from .pitch import compare_two_pitch, compare_multiple_pitch
from .spectral_bandwidth import compare_two_spectral_bandwidth, compare_multiple_spectral_bandwidth
from .spectral_centroid import compare_two_spectral_centroid, compare_multiple_spectral_centroid
from .spectral_contrast import compare_two_spectral_contrast, compare_multiple_spectral_contrast
from .spectral_flatness import compare_two_spectral_flatness, compare_multiple_spectral_flatness
from .spectral_roll_off import compare_two_spectral_roll_off, compare_multiple_spectral_roll_off
from .spectrogram import compare_two_spectrogram, compare_multiple_spectrogram
from .tempo import compare_two_tempo, compare_multiple_tempo
from .tempogram import compare_two_tempogram, compare_multiple_tempogram
from .temporal_centroid import compare_two_temporal_centroid, compare_multiple_temporal_centroid
from .temporal_flux import compare_two_temporal_flux, compare_multiple_temporal_flux
from .weighted_cyclic_tempogram import compare_two_wct, compare_multiple_wct
from .zero_crossing_rate import compare_two_zcr, compare_multiple_zcr


COMPARE_FUNCTIONS = {
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


def get_metric_names() -> list:
    return list(COMPARE_FUNCTIONS.keys())
