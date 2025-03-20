# script to convert the old experiments file format into the new format
import json
from pathlib import Path

from autotuning_methodology.validators import validate_experimentsfile

# set input and output files
folderpath = Path(__file__).parent
old_file_path = folderpath / Path("../tests/autotuning_methodology/integration/mockfiles/test_import_runs.json")
new_file_path = folderpath / Path("../tests/autotuning_methodology/integration/mockfiles/test_import_runs_new.json")
encoding = "utf-8"
assert old_file_path.exists(), f"Old file does not exist at {old_file_path}"
assert not new_file_path.exists(), f"New file does already exists at {new_file_path}"

# read input file to dictionary
with old_file_path.open("r", encoding=encoding) as fp:
    old_experiment: dict = json.load(fp)

# convert the dictionary to the new format
new_experiment = {
    "version": "1.2.0",
    "name": old_experiment["name"],
    "parent_folder": f"./{old_experiment['folder_id']}",
    "experimental_groups_defaults": {
        "applications": [
            {
                "name": kernel,
                "input_file": f"{old_experiment['kernels_path']}/{kernel}",
                "folder": f"{old_experiment['visualization_caches_path']}/{kernel}",
            }
            for kernel in old_experiment["kernels"]
        ],
        "gpus": old_experiment["GPUs"],
        "pattern_for_full_search_space_filenames": {
            "regex": f"{old_experiment['bruteforced_caches_path']}/" + "${applications}/${gpus}.json"
        },
        "stochastic": old_experiment["strategy_defaults"]["stochastic"],
        "repeats": old_experiment["strategy_defaults"]["repeats"],
        "samples": old_experiment["strategy_defaults"]["iterations"],
        "minimum_fraction_of_budget_valid": old_experiment.get("minimum_fraction_of_budget_valid", 0.5),
        "minimum_number_of_valid_search_iterations": old_experiment["strategy_defaults"][
            "minimum_number_of_evaluations"
        ],
        "ignore_cache": False,
    },
    "search_strategies": [
        {
            "name": strategy["name"],
            "search_method": strategy["strategy"],
            "display_name": strategy["display_name"],
            "autotuner": (
                "KernelTuner" if strategy["name"] != "ktt_profile_searcher" else "KTT"
            ),  # Assuming autotuner is KernelTuner for all strategies
        }
        for strategy in old_experiment["strategies"]
    ],
    "statistics_settings": {
        "minimization": old_experiment["minimization"],
        "cutoff_percentile": old_experiment["cutoff_percentile"],
        "cutoff_percentile_start": old_experiment["cutoff_percentile_start"],
        "cutoff_type": old_experiment["cutoff_type"],
        "objective_time_keys": ["all"],  # Mapped to 'all'
        "objective_performance_keys": old_experiment["objective_performance_keys"],
    },
    "visualization_settings": {
        "plots": [
            {
                "scope": "aggregate" if "aggregated" in plottype else "searchspace",
                "style": "scatter" if "scatter" in plottype else "line",
                "x_axis_value_types": [plottype if plottype != "aggregated" else "time"],
                "y_axis_value_types": old_experiment["plot"]["plot_y_value_types"],
            }
            for plottype in old_experiment["plot"]["plot_x_value_types"]
        ],
        "resolution": old_experiment["resolution"],
        "confidence_level": old_experiment["plot"]["confidence_level"],
        "compare_baselines": old_experiment["plot"]["compare_baselines"],
        "compare_split_times": old_experiment["plot"]["compare_split_times"],
    },
}

# validate using schema
validate_experimentsfile(new_experiment, encoding=encoding)

# write converted dictionary to file
with new_file_path.open("w", encoding=encoding) as fp:
    json.dump(new_experiment, fp)
