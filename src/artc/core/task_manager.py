"""
Task manager and parallel execution utilities for ARtC.

This module handles the orchestration of audio comparison tasks, statistical
aggregation and system-level resource management.
It provides safe multiprocessing, memory constraints and efficient matrix
reconstruction utilities for similarity computation.

Module Structure
────────────────
artc.core.task_manager
│
├── [STATISTICS LAYER]
│   ├── _mean, _variance, _maximum, _minimum, _mean_of_mode_range
│   └── STAT_CALCULATION
│
├── [SYSTEM LAYER]
│   ├── _available_processes
│   ├── _available_memory
│   └── _set_memory_limit
│
├── [DATA UTILS LAYER]
│   ├── _audio_into_chunks
│   └── _build_symmetric_matrix
│
├── [COMPARATOR LAYER]
│   ├── _compare_all_chunks
│   ├── _comparator_builder
│   └── _comparator
│
└── [ORCHESTRATION LAYER]
    └── compare  ← public entrypoint

Author: Nicolás Cereijo Ranchal
Part of the ARtC (Audio Real-time Comparator) framework.
"""

import gc
import multiprocessing
import os
import resource
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import psutil

import artc.core.configurations as config
from artc.core import analysis, errors
from artc.core.datastructures import WorkingSet
from artc.types import (
    FloatScalar,
    NDArrayFloat,
    ScalarReduceFn,
    np_max,
    np_mean,
    np_min,
    np_ravel,
    np_var,
)

logger = errors.logger_config.LoggerSingleton().get_logger()


def _mean(values: NDArrayFloat) -> FloatScalar:
    """Compute the arithmetic mean of numeric values."""
    return np_mean(values)


def _variance(values: NDArrayFloat) -> FloatScalar:
    """Compute the variance of numeric values."""
    return np_var(values)


def _mean_of_mode_range(values: NDArrayFloat) -> FloatScalar:
    """Compute the mean value within the most populated range of the input data.

    The sequence is divided into 10 equal-width buckets (ranges of size 10).
    The bucket containing the most elements is identified, and the mean
    of values within that bucket is returned.

    Args:
        values: Numeric values to analyze.

    Returns:
        The mean of the densest range, or 0.0 if no data falls within any range.
    """
    flat = np_ravel(values).astype(FloatScalar)

    # Compute histogram across 10 equal-width bins covering [0, 100)
    hist, _ = np.histogram(flat, bins=np.arange(0, 101, 10))

    # Divide the input into 10 fixed ranges (0–10, 10–20, ..., 90–100)
    # and identify the range containing the highest density of values.
    best_range = int(np.argmax(hist))
    lo, hi = best_range * 10, (best_range + 1) * 10
    mask = (flat >= lo) & (flat < hi)

    return FloatScalar(np_mean(flat[mask]) if np.any(mask) else 0.0)


def _maximum(values: NDArrayFloat) -> FloatScalar:
    """Return the maximum value of the sequence."""
    return np_max(values)


def _minimum(values: NDArrayFloat) -> FloatScalar:
    """Return the minimum value of the sequence."""
    return np_min(values)


"""List of scalar statistical reduction functions applied during analysis."""
STAT_CALCULATION: list[ScalarReduceFn] = [
    _mean,
    _variance,
    _mean_of_mode_range,
    _maximum,
    _minimum,
]


# ─────────────────────────────────────────────────────────────
# Process, memory safety controls and resource limits
# ─────────────────────────────────────────────────────────────
def _available_processes(configuration_path: Path) -> int:
    """Determine how many concurrent processes can be safely used.

    Read the user configuration and validate the number of processes
    against the system's available CPU cores.
    """
    processes = config.read_config("processes", configuration_path)
    cpu_cores = os.cpu_count()

    if not (isinstance(processes, int) and isinstance(cpu_cores, int)):
        raise ValueError("Unable to query available processes or CPU cores.")

    if processes < 1 or processes > cpu_cores:
        raise ValueError(
            f"Selected processes ({processes}) must be between 1 and {cpu_cores}."
        )
    return processes


def _available_memory(configuration_path: Path) -> int:
    """Compute the safe memory allocation limit in bytes.

    The limit is derived from the system’s total memory and the percentage
    specified in the user configuration (max 80% allowed).
    """
    mem_limit = config.read_config("memory", configuration_path)
    total_memory = psutil.virtual_memory().total

    if not isinstance(mem_limit, int) or not (0 < mem_limit <= 80):
        raise ValueError(
            "System memory settings cannot be queried, or the selected amount "
            "exceeds the safety limit (<= 80%)."
        )
    return int(total_memory * (mem_limit / 100.0))


def _set_memory_limit(configuration_path: Path) -> None:
    """Apply an OS-level virtual memory limit based on configuration.

    This uses `resource.setrlimit()` to restrict the address space (RLIMIT_AS)
    available to the current process, preventing excessive memory consumption.
    """
    memory = _available_memory(configuration_path)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory, resource.RLIM_INFINITY))
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The memory configuration is not compatible with the operating system."
        ) from exc


# ─────────────────────────────────────────────────────────────
# Matrix reconstruction and audio segmentation helpers
# ─────────────────────────────────────────────────────────────
def _audio_into_chunks(
    audio: NDArrayFloat, samples_per_chunk: int
) -> list[NDArrayFloat]:
    """Split an audio signal into equal-length chunks.

    Handles both mono and multi-channel signals, producing non-overlapping
    contiguous segments of uniform size. Chunks smaller than the requested
    size are discarded to ensure consistent array shapes.

    Args:
        audio:
            Input audio array. For mono signals: shape (samples,).
            For multi-channel signals: shape (channels, samples).
        samples_per_chunk:
            Number of samples per chunk.

    Returns:
        A list of contiguous, equally sized audio chunks.
    """
    if samples_per_chunk <= 0:
        raise ValueError("Number of samples per chunk must be a positive integer.")

    if audio.ndim == 1:  # Mono
        chunks = [
            audio[i : i + samples_per_chunk]
            for i in range(0, len(audio), samples_per_chunk)
            if i + samples_per_chunk <= len(audio)
        ]
    else:  # Multi-channel
        chunks = [
            audio[:, i : i + samples_per_chunk]
            for i in range(0, audio.shape[1], samples_per_chunk)
            if i + samples_per_chunk <= audio.shape[1]
        ]

    return chunks


def _build_symmetric_matrix(values: Sequence[float]) -> NDArrayFloat:
    """Reconstruct a symmetric matrix from its upper-triangular elements.

    Given a list of values representing the upper-triangular part of a symmetric
    matrix (including the main diagonal):

        value1, value2, value3
             _, value4, value5
             _,      _, value6

    This function reconstructs the full symmetric matrix, whose side length (n)
    is unknown a priori:

        value1, value2, value3
        value2, value4, value5
        value3, value5, value6

    Let L be the number of known elements (the list length) and n the resulting
    matrix size. The relationship is:

        L = n * (n + 1) / 2

    Solving for n yields the quadratic equation:

        n² + n - 2L = 0
        n = (-1 + √(1 + 8L)) / 2

    Args:
        values:
            Flattened sequence representing the upper-triangular values
            (including the main diagonal).

    Returns:
        The reconstructed symmetric matrix as a NumPy array.
    """
    # Compute matrix side length using the inverse triangular-number formula
    n = int(round((-1 + (1 + 8 * len(values)) ** 0.5) / 2))

    if n * (n + 1) // 2 != len(values):
        raise ValueError(
            f"Input length ({len(values)}) does not correspond to a valid symmetric matrix."
        )

    matrix = np.zeros((n, n), dtype=float)
    row_idx, col_idx = np.triu_indices(n)

    # Fill upper-triangular values and mirror them to the lower half
    matrix[row_idx, col_idx] = values
    matrix[col_idx, row_idx] = values

    return matrix


# ─────────────────────────────────────────────────────────────
# Comparator construction
# ─────────────────────────────────────────────────────────────
def _compare_all_chunks(
    compare_func: Callable[..., FloatScalar],
    audio_1_chunks: list[NDArrayFloat],
    audio_2_chunks: list[NDArrayFloat],
    params: tuple,
    use_sr: bool,
) -> list[FloatScalar]:
    """Execute all pairwise chunk-to-chunk comparisons between two audio signals.

    Each comparison produces a similarity score between 0 and 1 (or 0–100 if scaled later).
    The function iterates over the upper-triangular region of the chunk pairs
    to avoid redundant computations.
    """
    similarity_scores: list[FloatScalar] = []

    for i, chunk_a in enumerate(audio_1_chunks):
        for j in range(i, len(audio_2_chunks)):
            chunk_b = audio_2_chunks[j]
            if use_sr:
                sr1, sr2 = params[3], params[4]
                similarity_scores.append(compare_func(chunk_a, chunk_b, sr1, sr2))
            else:
                similarity_scores.append(compare_func(chunk_a, chunk_b))

    return similarity_scores


def _comparator_builder(
    metric: str,
    parameters: tuple,
    configuration_path: Path,
) -> list[list[Callable[[], FloatScalar]]]:
    """Construct deferred comparison callables between all chunk pairs.

    Each callable represents a comparison operation between two specific
    audio chunks and can be executed later in parallel by worker processes.

    Args:
        metric: Name of the comparison metric as defined in `COMPARE_FUNCTIONS`.
        parameters: Tuple containing (compare_func, audio1, audio2, [sr1, sr2]).
        configuration_path: Path to the active configuration file.

    Returns:
        A nested list of zero-argument callables, grouped per audio pair.
    """
    samples_per_chunk = config.read_config("sampling", configuration_path)
    compare_func: Callable[..., FloatScalar] = parameters[0]
    audio_1: np.ndarray = parameters[1]
    audio_2: np.ndarray = parameters[2]

    if not isinstance(samples_per_chunk, int) or samples_per_chunk <= 0:
        raise ValueError(
            "Invalid 'sampling' configuration value (expected positive int)."
        )

    min_len = min(audio_1.shape[-1], audio_2.shape[-1])
    if samples_per_chunk > min_len:
        raise ValueError(
            "Samples per fragment cannot be queried, or the selected number is too large."
        )

    # Normalize indexing depending on channel layout (mono vs multi-channel)
    def _slice(x: np.ndarray) -> np.ndarray:
        return x[:min_len] if x.ndim == 1 else x[:, :min_len]

    audio1_chunks = _audio_into_chunks(_slice(audio_1), samples_per_chunk)
    audio2_chunks = _audio_into_chunks(_slice(audio_2), samples_per_chunk)

    use_sr = analysis.COMPARE_FUNCTIONS[metric]["use_sample_rate"]

    # Build all upper-triangular chunk comparisons
    if use_sr:
        sr1, sr2 = parameters[3], parameters[4]
        comparators_group = [
            partial(compare_func, chunk_a, chunk_b, sr1, sr2)
            for i, chunk_a in enumerate(audio1_chunks)
            for chunk_b in audio2_chunks[i:]
        ]
    else:
        comparators_group = [
            partial(compare_func, chunk_a, chunk_b)
            for i, chunk_a in enumerate(audio1_chunks)
            for chunk_b in audio2_chunks[i:]
        ]

    return [comparators_group]


def _comparator(comparison: Callable[[], FloatScalar]) -> FloatScalar:
    """Safely execute a comparison callable, scaling result to 0–100.

    Catches MemoryError and logs a warning if the operation cannot be completed.
    The result is scaled by 100 to express similarity as a percentage.

    Args:
        comparison: Callable returning a FloatScalar similarity value.

    Returns:
        The scaled comparison value, or -1.0 if memory constraints are hit.
    """
    try:
        return FloatScalar(comparison()) * 100.0
    except MemoryError:
        logger.warning("An operation was aborted due to insufficient memory")
        return FloatScalar(-1.0)


# ─────────────────────────────────────────────────────────────
# Public entrypoint
# ─────────────────────────────────────────────────────────────
def compare(
    metric: str,
    wset: WorkingSet,
    configuration_path: Path,
    *,
    set_to_use: str = "individual_files",
    stats: list[str] | None = None,
) -> list[tuple[str, NDArrayFloat]]:
    """Run pairwise audio comparisons and compute selected statistics.

    Executes all pairwise comparisons for a given metric, applies statistical
    aggregations (mean, variance, etc.), and returns one symmetric matrix per
    statistic, representing the pairwise similarity across the working set.
    """
    available_stats: list[str] = config.read_config("stats", configuration_path)
    processes = _available_processes(configuration_path)
    items = wset.working_set[set_to_use]
    results: list[tuple[str, NDArrayFloat]] = []

    if metric not in analysis.COMPARE_FUNCTIONS:
        raise ValueError(
            f"Invalid metric '{metric}'. Available metrics: "
            f"{list(analysis.COMPARE_FUNCTIONS.keys())}"
        )

    if stats:
        unknown = [s for s in stats if s not in available_stats]
        if unknown:
            raise ValueError(
                f"Invalid statistics: {unknown}. Available: {available_stats}"
            )
    selected_stats = stats or available_stats

    all_operations: list[Sequence[Callable[[], FloatScalar]]] = []
    for i, item_a in enumerate(items):
        for item_b in items[i:]:
            compare_func = analysis.COMPARE_FUNCTIONS[metric]["compare_two"]
            params: tuple = (
                compare_func,
                item_a.audio_signal_unloaded(),
                item_b.audio_signal_unloaded(),
            )
            if analysis.COMPARE_FUNCTIONS[metric]["use_sample_rate"]:
                params += (item_a.sample_rate, item_b.sample_rate)
            all_operations.extend(
                _comparator_builder(metric, params, configuration_path)
            )

    _set_memory_limit(configuration_path)

    with multiprocessing.Pool(processes=processes) as pool:
        pair_results = [pool.map(_comparator, group) for group in all_operations]

        pair_arrays = [np.array(r, dtype=FloatScalar, copy=False) for r in pair_results]

        for stat_name in selected_stats:
            idx = available_stats.index(stat_name)
            per_pair_stats = pool.map(STAT_CALCULATION[idx], pair_arrays)
            results.append(
                (stat_name, _build_symmetric_matrix([float(v) for v in per_pair_stats]))
            )

    gc.collect()
    return results
