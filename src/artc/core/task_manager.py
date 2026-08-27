"""
Task manager and parallel execution utilities for ARtC

This module handles the orchestration of audio comparison tasks, statistical
aggregation and system-level resource management
It provides safe multiprocessing, memory constraints and efficient matrix
reconstruction utilities for similarity computation

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
│   ├── _comparator_builder
│   └── _comparator
│
└── [ORCHESTRATION LAYER]
    ├── _mapper
    └── compare  ← public entrypoint

Author: Nicolás Cereijo Ranchal
Part of the ARtC (Audio Real-time Comparator) framework.
"""

import contextlib
import gc
import multiprocessing
import os
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from multiprocessing.pool import Pool
from typing import TypeVar

import numpy as np
import psutil

# Optional 'resource' module (POSIX only).
try:
    import resource  # type: ignore[attr-defined]
except ImportError:  # Windows
    resource = None  # type: ignore[assignment]

import artc.core.configurations as config
from artc.core import analysis
from artc.core.datastructures import WorkingSet
from artc.core.errors import logger_config
from artc.types import (
    FloatScalar,
    NDArrayFloat,
    NDArrayFloatCheck,
    ScalarReduceFn,
    np_max,
    np_mean,
    np_min,
    np_ravel,
    np_var,
)

logger = logger_config.LoggerSingleton().get_logger()


@NDArrayFloatCheck("values", level="full_checks")
def _mean(values: NDArrayFloat) -> FloatScalar:
    """Compute the arithmetic mean of numeric values."""
    return np_mean(values)


@NDArrayFloatCheck("values", level="full_checks")
def _variance(values: NDArrayFloat) -> FloatScalar:
    """Compute the variance of numeric values."""
    return np_var(values)


@NDArrayFloatCheck("values", level="full_checks")
def _mean_of_mode_range(values: NDArrayFloat) -> FloatScalar:
    """Compute the mean value within the most populated range of the input data.

    The sequence is divided into 10 equal-width buckets (ranges of size 0.1)
    The bucket containing the most elements is identified, and the mean
    of values within that bucket is returned.

    Args:
        values: Numeric values to analyze

    Returns:
        The mean of the densest range, or 0.0 if no data falls within any
        range.
    """
    flat = np_ravel(values).astype(FloatScalar)

    # A NaN never falls inside any bin below, so it would otherwise be
    # silently excluded instead of flagging the pair as affected.
    if np.isnan(flat).any():
        return FloatScalar(np.nan)

    # Compute histogram across 10 equal-width bins covering [0, 1).
    hist, _ = np.histogram(flat, bins=np.arange(0, 1.01, 0.1))

    # Divide the input into 10 fixed ranges (0.0–0.1, 0.1–0.2, ..., 0.9–1.0)
    # and identify the range containing the highest density of values.
    best_range = int(np.argmax(hist))
    lo, hi = best_range * 0.1, (best_range + 1) * 0.1
    mask = (flat >= lo) & (flat < hi)

    return FloatScalar(np_mean(flat[mask]) if np.any(mask) else 0.0)


@NDArrayFloatCheck("values", level="full_checks")
def _maximum(values: NDArrayFloat) -> FloatScalar:
    """Return the maximum value of the sequence."""
    return np_max(values)


@NDArrayFloatCheck("values", level="full_checks")
def _minimum(values: NDArrayFloat) -> FloatScalar:
    """Return the minimum value of the sequence."""
    return np_min(values)


"""Scalar statistical reduction functions applied during analysis, by name."""
STAT_CALCULATION: dict[str, ScalarReduceFn] = {
    "mean": _mean,
    "variance": _variance,
    "mean_of_mode_range": _mean_of_mode_range,
    "maximum": _maximum,
    "minimum": _minimum,
}


# ─────────────────────────────────────────────────────────────
# Process, memory safety controls and resource limits
# ─────────────────────────────────────────────────────────────
def _available_processes() -> int:
    """Determine how many concurrent processes can be safely used.

    Read the user configuration and validate the number of processes
    against the system's available CPU cores.
    """
    processes = config.read_config("processes")
    cpu_cores = os.cpu_count() or 1

    if not isinstance(processes, int):
        raise TypeError("Unable to query available processes or CPU cores")

    if processes < 1 or processes > cpu_cores:
        raise ValueError(
            f"Selected processes ({processes}) must be between 1 and " +
            f"{cpu_cores}"
        )
    return processes


def _available_memory() -> int:
    """Compute the safe memory allocation limit in bytes.

    The limit is derived from the system’s total memory and the percentage
    specified in the user configuration (max 80% allowed).
    """
    mem_limit = config.read_config("memory")
    total_memory = psutil.virtual_memory().total

    if not isinstance(mem_limit, int) or not (0 < mem_limit <= 80):
        raise ValueError(
            "System memory settings cannot be queried, or the selected " +
            "amount exceeds the safety limit (<= 80%)"
        )
    return int(total_memory * (mem_limit / 100.0))


def _set_memory_limit() -> None:
    """Apply an OS-level virtual memory limit based on configuration.

    This uses 'resource.setrlimit()' to restrict the address space
    ('RLIMIT_AS') available to the current process, preventing excessive
    memory consumption.

    Note:
        Memory-limit enforcement does not work on Windows because Windows
        sucks.
    """
    # Skip on Windows or platforms without 'RLIMIT_AS'.
    if resource is None or not hasattr(resource, "RLIMIT_AS"):
        logger.warning(
            "Memory limit enforcement via 'resource' is not supported on " +
            "this platform, skipping"
        )
        return

    memory = _available_memory()
    try:
        resource.setrlimit(
            resource.RLIMIT_AS, (memory, resource.RLIM_INFINITY)
        )
    except Exception as exc:
        raise RuntimeError(
            "The memory configuration is not compatible with the " +
            "operating system"
        ) from exc


# ─────────────────────────────────────────────────────────────
# Matrix reconstruction and audio segmentation helpers
# ─────────────────────────────────────────────────────────────
@NDArrayFloatCheck("audio", level="full_checks")
def _audio_into_chunks(
    audio: NDArrayFloat, samples_per_chunk: int
) -> list[NDArrayFloat]:
    """Split an audio signal into equal-length chunks.

    Handles both mono and multi-channel signals, producing non-overlapping
    contiguous segments of uniform size. Chunks smaller than the requested
    size are discarded to ensure consistent array shapes.

    Args:
        audio:
            Input audio array. For mono signals: shape (samples,)
            For multi-channel signals: shape (channels, samples)
        samples_per_chunk:
            Number of samples per chunk.

    Returns:
        A list of contiguous, equally sized audio chunks.
    """
    if samples_per_chunk <= 0:
        raise ValueError(
            "Number of samples per chunk must be a positive integer"
        )

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

    Given a list of values representing the upper-triangular part of a
    symmetric matrix (including the main diagonal):

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
    # Compute matrix side length using the inverse triangular-number formula.
    n = round((-1 + (1 + 8 * len(values)) ** 0.5) / 2)

    if n * (n + 1) // 2 != len(values):
        raise ValueError(
            f"Input length ({len(values)}) does not correspond to a " +
            "valid symmetric matrix"
        )

    matrix = np.zeros((n, n), dtype=FloatScalar)
    row_idx, col_idx = np.triu_indices(n)

    # Fill upper-triangular values and mirror them to the lower half.
    matrix[row_idx, col_idx] = values
    matrix[col_idx, row_idx] = values

    return matrix


# ─────────────────────────────────────────────────────────────
# Comparator construction
# ─────────────────────────────────────────────────────────────
@NDArrayFloatCheck("audio_1", level="frontier_checks")
@NDArrayFloatCheck("audio_2", level="frontier_checks")
def _comparator_builder(
    metric: str,
    compare_func: Callable[..., float],
    audio_1: NDArrayFloat,
    audio_2: NDArrayFloat,
    *,
    sr1: int | None = None,
    sr2: int | None = None,
) -> list[list[Callable[[], float]]]:
    """Construct deferred comparison callables between all chunk pairs.

    Each callable represents a comparison operation between two specific audio
    chunks and can be executed later, either sequentially or in parallel by
    worker processes. The actual execution mode is decided by the caller.

    Note on chunk pairing:
        When 'audio_1' and 'audio_2' are the same signal (e.g. when
        computing the diagonal of the similarity matrix), the
        chunk-to-chunk comparison matrix is symmetric, so only the
        upper-triangular region is built to avoid computing and storing
        redundant duplicate pairs.

        When 'audio_1' and 'audio_2' are different signals, no such
        symmetry exists. Pair '(i, j)' and pair '(j, i)' involve different
        chunk content on each side, so every combination of chunks must be
        compared.
    """
    samples_per_chunk = config.read_config("sampling")

    if not isinstance(samples_per_chunk, int) or samples_per_chunk <= 0:
        raise ValueError(
            "Invalid 'sampling' configuration value (expected positive int)"
        )

    min_len = min(audio_1.shape[-1], audio_2.shape[-1])
    if samples_per_chunk > min_len:
        raise ValueError(
            "Samples per fragment cannot be queried, or the selected " +
            "number is too large"
        )

    # Normalize indexing depending on channel layout (mono vs multi-channel).
    def _slice(x: NDArrayFloat) -> NDArrayFloat:
        return x[:min_len] if x.ndim == 1 else x[:, :min_len]

    audio1_chunks = _audio_into_chunks(_slice(audio_1), samples_per_chunk)
    audio2_chunks = _audio_into_chunks(_slice(audio_2), samples_per_chunk)

    use_sr = analysis.COMPARE_FUNCTIONS[metric]["use_sample_rate"]
    sr_args: tuple[int | None, ...] = (sr1, sr2) if use_sr else ()

    # Same underlying signal compared with itself. The chunk-pair matrix is
    # symmetric, so only the upper triangle (including the diagonal) is needed.
    same_signal = audio_1 is audio_2

    if same_signal:
        comparators_group = [
            partial(compare_func, chunk_a, chunk_b, *sr_args)
            for i, chunk_a in enumerate(audio1_chunks)
            for chunk_b in audio2_chunks[i:]
        ]
    else:
        comparators_group = [
            partial(compare_func, chunk_a, chunk_b, *sr_args)
            for chunk_a in audio1_chunks
            for chunk_b in audio2_chunks
        ]

    return [comparators_group]


def _comparator(comparison: Callable[[], float]) -> FloatScalar:
    """Safely execute a comparison callable.

    Catches any exception raised by the comparison and logs a warning instead
    of letting it propagate and abort every other pending comparison for the
    current metric.

    Args:
        comparison: Callable returning a similarity value as a plain float

    Returns:
        The comparison value, or NaN if the comparison failed.
    """
    try:
        return FloatScalar(comparison())
    except MemoryError:
        logger.warning("An operation was aborted due to insufficient memory")
        return FloatScalar(np.nan)
    # Broad catch is intentional, it isolates a failing comparison so the
    # rest of the pool keeps running.
    except Exception as e:  # noqa: BLE001
        logger.warning(f"An operation failed and was skipped: {e!r}")
        return FloatScalar(np.nan)


# ─────────────────────────────────────────────────────────────
# Public entrypoint
# ─────────────────────────────────────────────────────────────
_MapperIn = TypeVar("_MapperIn")
_MapperOut = TypeVar("_MapperOut")


def _mapper(
    pool: Pool | None,
    func: Callable[[_MapperIn], _MapperOut],
    values: Iterable[_MapperIn],
) -> list[_MapperOut]:
    """Apply 'func' over 'values', using 'pool' when one is available.

    Falls back to plain sequential mapping when 'pool' is 'None', so the
    reduction logic in 'compare()' only has to be written once regardless
    of whether multiprocessing is enabled.
    """
    if pool is not None:
        return pool.map(func, values)
    return [func(x) for x in values]


def compare(
    metric: str,
    wset: WorkingSet,
    *,
    set_to_use: str = "individual_files",
    stats: list[str] | None = None,
) -> list[tuple[str, NDArrayFloat]]:
    """Run pairwise audio comparisons and compute selected statistics.

    Executes all pairwise comparisons for a given metric, applies statistical
    aggregations (mean, variance, etc.), and returns one symmetric matrix per
    statistic, representing the pairwise similarity across the working set.
    """
    available_stats: list[str] = config.read_config("stats")
    processes = _available_processes()
    results: list[tuple[str, NDArrayFloat]] = []

    if metric not in analysis.COMPARE_FUNCTIONS:
        raise ValueError(
            f"Invalid metric '{metric}'. Available metrics: " +
            f"{list(analysis.COMPARE_FUNCTIONS.keys())}"
        )

    if set_to_use not in wset.working_set:
        raise ValueError(
            f"Unknown set '{set_to_use}' in working set '{wset.name}'. " +
            f"Available sets: {list(wset.working_set.keys())}"
        )
    items = wset.working_set[set_to_use]

    if stats is not None:
        unknown = [s for s in stats if s not in available_stats]
        if unknown:
            raise ValueError(
                f"Invalid statistics: {unknown}. Available: {available_stats}"
            )
    selected_stats = stats if stats is not None else available_stats

    unknown_stats = [s for s in selected_stats if s not in STAT_CALCULATION]
    if unknown_stats:
        raise ValueError(
            f"Unknown statistic(s) in config: {unknown_stats}. " +
            f"Available: {list(STAT_CALCULATION.keys())}"
        )

    _set_memory_limit()

    all_operations: list[Sequence[Callable[[], float]]] = []
    use_sr = analysis.COMPARE_FUNCTIONS[metric]["use_sample_rate"]
    for i, audio_signal_1 in enumerate(items):
        for audio_signal_2 in items[i:]:
            compare_func = analysis.COMPARE_FUNCTIONS[metric]["compare_two"]

            kwargs = (
                {
                    "sr1": audio_signal_1.sample_rate,
                    "sr2": audio_signal_2.sample_rate,
                }
                if use_sr
                else {}
            )

            all_operations.extend(
                _comparator_builder(
                    metric,
                    compare_func,
                    audio_signal_1.audio_signal_unloaded(),
                    audio_signal_2.audio_signal_unloaded(),
                    **kwargs,
                )
            )

    # A single pool (when processes > 1) is reused for both the pairwise
    # comparisons and every statistic's reduction, instead of spawning one
    # per statistic. '_mapper' falls back to plain sequential mapping when
    # running with a single process, so the reduction logic below only
    # exists once.
    pool_context = (
        multiprocessing.Pool(processes=processes)
        if processes > 1
        else contextlib.nullcontext()
    )
    with pool_context as pool:
        pair_results = [
            _mapper(pool, _comparator, group) for group in all_operations
        ]
        pair_arrays = [
            np.array(r, dtype=FloatScalar, copy=False) for r in pair_results
        ]

        for stat_name in selected_stats:
            stat_func = STAT_CALCULATION[stat_name]
            per_pair_stats = _mapper(pool, stat_func, pair_arrays)
            matrix = _build_symmetric_matrix(
                [float(v) for v in per_pair_stats]
            )
            results.append((stat_name, matrix))

    _ = gc.collect()
    return results
