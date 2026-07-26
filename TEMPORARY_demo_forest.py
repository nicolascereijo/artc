import os
import time
from pathlib import Path

import numpy as np

from artc.core.ensembles import generate_forest

# Audio directory used by TEMPORARY_demo_compare.py to build the comparison
# matrices. Audio order there follows WorkingSet.add_directory's
# sorted(os.listdir(...), key=str.lower)
AUDIO_DIR = Path(__file__).parent / "test_collection" / "TEMPORARY_papper_selection"

LABELS_BY_FILE: dict[str, int] = {
    '3-hoar-s-thunders.mp3': 0,
    'A-crackling-fireplace.mp3': 1,
    'A-lot-of-cicadas-and-other-insects.mp3': 0,
    'An-open-fire-in-fireplace.mp3': 1,
    'bangkok-thunderstorm.mp3': 0,
    'Car Drive By.mp3': 0,
    'Car Engine Idling Accelerating.mp3': 0,
    'Car Stop And Go.mp3': 0,
    'chicken-farm.mp3': 0,
    'City Road Traffic.mp3': 0,
    'close-rain-and-thunder.mp3': 0,
    'country-house-fireplace.mp3': 1,
    'Crackling-bark-wood-in-a-closed-fireplace.mp3': 1,
    'Daytime Forrest Bonfire.mp3': 1,
    'distant-thunder-and-rain-from-half-open-window-2.mp3': 0,
    'Dog Barking.mp3': 0,
    'Fire.mp3': 1,
    'Fireplace-and-the-flame-took-off.mp3': 1,
    'fireplace-close.mp3': 1,
    'Inside-a-chicken-house.mp3': 0,
    'Morning Highway in Distance.mp3': 0,
    'Muscle Car Driving Skid Out.mp3': 0,
    'Outdoor Farm Sounds .mp3': 0,
    'Outside Night.mp3': 0,
    'rain-and-distant-peals-of-thunder.mp3': 0,
    'sea-waves.mp3': 0,
    'Seagulls-nesting-on-a-cliff-at-Etretat-Northern-France.mp3': 0,
    'Small Stream Flowing.mp3': 0,
    'Small-fire-with-few-cracklings-in-fireplace.mp3': 1,
    'Splashing Water.mp3': 0,
    'Spring Day Forest.mp3': 0,
    'Walk On Dirt.mp3': 0,
    'Walk On Wet Cobble.mp3': 0,
    'Walking Barefoot over grass.mp3': 0,
    'Walking Cleats.mp3': 0,
    'Walking In Shallow Water.mp3': 0,
    'Walking on Gravel.mp3': 0,
    'warm-evening-outdoors.mp3': 0,
    'Wood-crackling-in-a-fireplay.mp3': 1,
    'Woodpecker Eating Distant.mp3': 0,
}


def build_labels(audio_dir: Path) -> list[int]:
    """Build the label list in the exact audio order used by WorkingSet.

    Re-derives the same sorted(os.listdir(...), key=str.lower) order that
    WorkingSet.add_directory used to build the comparison matrices, then
    maps each filename to its label via LABELS_BY_FILE. Raises if the audio
    directory and LABELS_BY_FILE disagree on which files exist, instead of
    silently misaligning labels with matrix rows.
    """
    audio_files = sorted(os.listdir(audio_dir), key=str.lower)

    missing = set(audio_files) - LABELS_BY_FILE.keys()
    extra = LABELS_BY_FILE.keys() - set(audio_files)
    if missing or extra:
        raise ValueError(
            f"LABELS_BY_FILE is out of sync with '{audio_dir}': "
            f"missing labels for {sorted(missing)}, stale labels for {sorted(extra)}"
        )

    return [LABELS_BY_FILE[f] for f in audio_files]


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
    # results_dir = current_path.parent / "TEMPORARY_papper_results"

    csv_files = get_csv_files(results_dir)
    ndarray_list = load_csv_as_ndarray(csv_files, results_dir)

    labels = build_labels(AUDIO_DIR)

    generate_forest(ndarray_list, csv_files, labels)

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
