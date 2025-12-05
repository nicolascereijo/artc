import os
import time
from pathlib import Path

import numpy as np

from artc.core.ensembles import generate_forest


def get_csv_files(directory: Path):
    """Return a list of CSV filenames inside a directory."""
    csv_files = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            csv_files.append(file)
    return csv_files


def load_csv_as_ndarray(csv_files, directory: Path):
    """Load each CSV file into a NumPy ndarray."""
    ndarray_list = []
    for file in csv_files:
        try:
            matrix = np.loadtxt(directory / file, delimiter=",", skiprows=0)
            ndarray_list.append(matrix)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return ndarray_list


def main():
    start_time = time.time()
    current_path = Path(__file__)

    # Must match the directory where the first script saves its CSV output
    results_dir = current_path.parent / "TEMPORARY_demo_results"

    csv_files = get_csv_files(results_dir)
    ndarray_list = load_csv_as_ndarray(csv_files, results_dir)

    # Labels corresponding to each CSV file
    labels = [
        1, 1, 1, 1, 0,
        0, 1, 0, 1, 1,
        0, 0, 0, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 1, 0,
    ]

    # Optional: warn if number of files does not match number of labels
    if len(csv_files) != len(labels):
        print(f"Warning: {len(csv_files)} CSV files but {len(labels)} labels.")

    generate_forest(ndarray_list, csv_files, labels)

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
