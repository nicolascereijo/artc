import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colormaps
from matplotlib.colors import ListedColormap

import artc.core.datastructures as dt_structs
from artc.core import analysis, compare


def main():
    current_path = Path(__file__)
    start_time = time.time()
    configuration_path = (
        current_path.parent
        / "src"
        / "artc"
        / "core"
        / "configurations"
        / "artc_config.toml"
    )
    files_path = current_path.parent / "test_collection" / "TEMPORARY_sample_selection"

    example_set = dt_structs.WorkingSet("main_set")
    example_set.add_directory(
        path=files_path, configuration_path=configuration_path, group="main_set"
    )

    metric_names = analysis.get_metric_names()

    print("Starting test script execution...\n")
    for metric in metric_names:
        loop_start_time = time.time()
        print(f"> Comparing audios with the '{metric}' metric")

        results = compare(metric, example_set, set_to_use="main_set")
        print(f"> Heatmaps completed, saved to 'TEMPORARY_demo_results/{metric}.png'")

        for stat in results:
            stat_name = stat[0]
            result_array = np.array(stat[1])
            result_array = result_array.reshape(24, 24)

            viridis = colormaps.get_cmap("viridis")
            newcolors = viridis(np.linspace(0, 1, 256))
            red = np.array([1, 0, 0, 1])
            newcolors[:1, :] = red
            newcmp = ListedColormap(newcolors)

            plt.figure(figsize=(20, 16))
            sns.heatmap(
                result_array,
                cmap=newcmp,
                annot=True,
                cbar=True,
                fmt=".2f",
                xticklabels=[str(i) for i in range(1, 25)],
                yticklabels=[str(i) for i in range(1, 25)],
                vmin=-1,
                vmax=100,
            )
            plt.title(f"{metric} - {stat_name}")
            plt.xlabel("Audios")
            plt.ylabel("Audios")
            plt.tight_layout()
            plt.savefig(f"TEMPORARY_demo_results/{metric}_{stat_name}.png")
            plt.close()

            np.savetxt(
                f"TEMPORARY_demo_results/{metric}_{stat_name}.csv",
                result_array,
                delimiter=",",
                fmt="%.2f",
            )

        loop_end_time = time.time()
        print(
            f"> Time for '{metric}' metric: {loop_end_time - loop_start_time:.2f} seconds\n"
        )

    print("\nTest script execution completed!!")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
