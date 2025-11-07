import gc
import os
import multiprocessing
import resource
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import psutil

from artc.core import analysis, errors
import artc.core.configurations as config
from artc.core.datastructures import WorkingSet


logger = errors.logger_config.LoggerSingleton().get_logger()


def mean(values: list) -> float:
    return float(np.mean(values))


def variance(values: list) -> float:
    return float(np.var(values))


def mean_of_mode_range(values: list) -> float:
    flat_values = np.ravel(values)
    counts = [0] * 10

    for val in flat_values:
        counts[min(int(val // 10), 9)] += 1

    best_range = counts.index(max(counts))
    values_in_range = [
        val for val in flat_values if best_range * 10 <= val < (best_range + 1) * 10
    ]

    return float(np.mean(values_in_range)) if values_in_range else 0.0


def maximum(values: list) -> float:
    return np.max(values)


def minimum(values: list) -> float:
    return np.min(values)


STAT_CALCULATION = [mean, variance, mean_of_mode_range, maximum, minimum]


def comparator(comparison):
    try:
        return comparison() * 100
    except MemoryError:
        logger.warning("An operation was aborted due to insufficient memory")
        return -1


def available_processes(configuration_path: Path) -> int:
    processes = config.read_config("processes", configuration_path)
    cpu_cores = os.cpu_count()

    if not (
        isinstance(processes, int)
        and isinstance(cpu_cores, int)
        and processes <= cpu_cores
    ):
        raise ValueError(
            "Available processes cannot be queried, or selected amount exceeds safety limit"
        )
    return processes


def available_memory(configuration_path: Path) -> int:
    mem_limit = config.read_config("memory", configuration_path)
    total_memory = psutil.virtual_memory().total

    if not isinstance(mem_limit, int) or not mem_limit <= 80:
        raise ValueError(
            "System's available memory cannot be queried, or selected amount exceeds safety limit"
        )
    return int(total_memory * (mem_limit / 100))


def set_memory_limit(configuration_path: Path):
    memory = available_memory(configuration_path)

    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory, resource.RLIM_INFINITY))
    except Exception:
        raise RuntimeError(
            "The memory configuration is not compatible with the operating system"
        )


def build_symmetric_matrix(values: list) -> np.ndarray:
    matrix_size = int((-1 + (1 + 8 * len(values)) ** 0.5) / 2)
    matrix = np.zeros((matrix_size, matrix_size))

    idx = 0
    for i in range(matrix_size):
        for j in range(i, matrix_size):
            matrix[i][j] = values[idx]
            if i != j:
                matrix[j][i] = values[idx]
            idx += 1
    return matrix


def audio_into_chunks(audio: np.ndarray, samples_per_chunk: int) -> list:
    if audio.ndim == 1:  # Mono audio
        return [
            audio[i : i + samples_per_chunk]
            for i in range(0, len(audio), samples_per_chunk)
            if len(audio[i : i + samples_per_chunk]) == samples_per_chunk
        ]
    else:  # Stereo audio
        return [
            audio[:, i : i + samples_per_chunk]
            for i in range(0, audio.shape[1], samples_per_chunk)
            if audio[:, i : i + samples_per_chunk].shape[1] == samples_per_chunk
        ]


def comparator_builder(
    metric: str, parameters: tuple, processes: int, configuration_path: Path
) -> list:
    samples_per_chunk = config.read_config("sampling", configuration_path)
    audio_1 = parameters[1]
    audio_2 = parameters[2]
    operations = []

    min_len = min(audio_1.shape[-1], audio_2.shape[-1])
    if not isinstance(samples_per_chunk, int) or not samples_per_chunk <= min_len:
        raise ValueError(
            "Samples per fragment cannot be queried, or the selected number is too large"
        )

    if audio_1.ndim == 1:  # Mono audio
        audio1_chunks = audio_into_chunks(audio_1[:min_len], samples_per_chunk)
        audio2_chunks = audio_into_chunks(audio_2[:min_len], samples_per_chunk)
    else:  # Stereo audio
        audio1_chunks = audio_into_chunks(audio_1[:, :min_len], samples_per_chunk)
        audio2_chunks = audio_into_chunks(audio_2[:, :min_len], samples_per_chunk)

    with Pool(processes=processes) as pool:
        for i in range(len(audio1_chunks)):
            for j in range(len(audio1_chunks) - i):
                compare_func = parameters[0]

                new_parameters = (compare_func, audio1_chunks[i], audio2_chunks[i + j])
                if analysis.COMPARE_FUNCTIONS[metric]["use_sample_rate"] is True:
                    new_parameters += (
                        parameters[3],  # Sample rate 1
                        parameters[4],  # Sample rate 2
                    )

                operations.append(pool.map(comparator, [partial(*new_parameters)]))
    return operations


def compare(
    metric: str,
    wset: WorkingSet,
    configuration_path: Path,
    *,
    set_to_use: str = "individual_files",
    stats: list[str] = [],
) -> list[tuple[str, np.ndarray]]:
    available_stats = config.read_config("stats", configuration_path)
    processes = available_processes(configuration_path)
    set = wset.working_set[set_to_use]
    operations, results = [], []

    if metric not in analysis.COMPARE_FUNCTIONS:
        raise ValueError(
            f"Metric {metric} invalid, available metrics are the following:\n{
                list(analysis.COMPARE_FUNCTIONS.keys())
            }"
        )
    if stats and any(stat not in available_stats for stat in stats):
        raise ValueError(
            f"Invalid statistics, available stats are the following:\n{available_stats}"
        )
    elif not stats:
        stats = available_stats

    for i in range(len(set)):
        for j in range(len(set) - i):
            compare_func = analysis.COMPARE_FUNCTIONS[metric]["compare_two"]

            parameters = (
                compare_func,
                set[i].audio_signal_unloaded(),
                set[i + j].audio_signal_unloaded(),
            )
            if analysis.COMPARE_FUNCTIONS[metric]["use_sample_rate"] is True:
                parameters += (set[i].sample_rate, set[i + j].sample_rate)

            operations.append(
                comparator_builder(metric, parameters, processes, configuration_path)
            )

    set_memory_limit(configuration_path)

    with multiprocessing.Pool(processes=processes) as pool:
        for i, stat in enumerate(available_stats):
            if stat in stats:
                st = pool.map(STAT_CALCULATION[i], operations)
                results.append([stat, build_symmetric_matrix(st)])
    gc.collect()

    return results
