""" Main experiments code """

from typing import Tuple
from math import ceil
import json
from jsonschema import validate
from importlib import import_module
from pathlib import Path
import sys
from argparse import ArgumentParser

from autotuning_methodology.runner import collect_results
from autotuning_methodology.caching import ResultsDescription
from autotuning_methodology.searchspace_statistics import SearchspaceStatistics


def get_args_from_cli(args) -> str:
    """Set the Command Line Interface arguments and return the argument values"""
    CLI = ArgumentParser()
    CLI.add_argument("experiment", type=str, help="The experiment.json to execute, see experiments/template.json")
    args = CLI.parse_args(args)
    filepath: str = args.experiment
    if filepath is None or filepath == "":
        raise ValueError("Invalid '-experiment' option. Run 'visualize_experiments.py -h' to read more about the options.")
    return filepath


def get_experiment_schema_filepath() -> Path:
    """Get the filepath to the JSON schema for experiment files"""
    schemafilepath = Path("src/autotuning_methodology/schema.json")
    assert schemafilepath.exists(), f"Path to schema.json does not exist, attempted path: {schemafilepath}"
    return schemafilepath


def get_experiment(filename: str) -> dict:
    """Validates and gets the experiment from the .json file"""
    folder_name = "experiment_files"
    folder = Path(folder_name)
    extension = ".json"
    if not filename.endswith(extension):
        filename = filename + extension
    if not filename.startswith(folder_name + "/"):
        path = folder / filename
    else:
        path = Path(filename)
    assert path.exists(), f"Path to experiment file does not exist, attempted path: {path}"
    schemafilepath = get_experiment_schema_filepath()
    with open(path) as file, open(schemafilepath) as schemafile:
        schema = json.load(schemafile)
        experiment = json.load(file)
        validate(instance=experiment, schema=schema)
        return experiment


def get_strategies(experiment: dict) -> dict:
    """Gets the strategies from an experiments file by augmenting it with the defaults"""
    strategy_defaults = experiment["strategy_defaults"]
    strategies = experiment["strategies"]
    # # get a baseline index if it exists
    # baseline_index = list(strategy_index for strategy_index, strategy in enumerate(strategies) if 'is_baseline' in strategy)
    # if len(baseline_index) != 1:
    #     raise ValueError(f"There must be exactly one baseline, found {len(baseline_index)} baselines")
    # if strategies[baseline_index[0]]['is_baseline'] != True:
    #     raise ValueError(f"is_baseline must be true, yet is set to {strategies[0]['is_baseline']}!")
    # # if the baseline index is not 0, put the baseline strategy first
    # if baseline_index[0] != 0:
    #     raise ValueError("The baseline strategy must be the first strategy in the experiments file!")
    #     # strategies.insert(0, strategies.pop(baseline_index[0]))

    # augment the strategies with the defaults
    for strategy in strategies:
        for default in strategy_defaults:
            if not default in strategy:
                strategy[default] = strategy_defaults[default]
    return strategies


def execute_experiment(filepath: str, profiling: bool) -> Tuple[dict, dict, dict]:
    """Executes the experiment by retrieving it from the cache or running it"""
    experiment = get_experiment(filepath)
    print(f"Starting experiment '{experiment['name']}'")
    experiment_folder_id = experiment.get("folder_id")
    minimization: bool = experiment.get("minimization", True)
    cutoff_percentile: float = experiment.get("cutoff_percentile", 1)
    cutoff_type: str = experiment.get("cutoff_type", "fevals")
    assert cutoff_type == "fevals" or cutoff_type == "time"
    curve_segment_factor: float = experiment.get("curve_segment_factor", 0.05)
    assert isinstance(curve_segment_factor, float)
    strategies = get_strategies(experiment)
    # add the kernel directory to path to import the module
    kernel_path = Path(experiment.get("kernel_path", ""))
    if not kernel_path.exists():
        raise FileNotFoundError(f"No such path {kernel_path}")
    sys.path.append(str(kernel_path))
    kernel_names = experiment["kernels"]
    kernels = list(import_module(kernel_name) for kernel_name in kernel_names)

    # variables for comparison
    objective_time_keys = experiment.get("objective_time_keys")
    objective_performance_keys = experiment.get("objective_performance_keys")

    # execute each strategy in the experiment per GPU and kernel
    results_descriptions: dict[str, dict[str, dict[str, ResultsDescription]]] = dict()
    gpu_name: str
    for gpu_name in experiment["GPUs"]:
        print(f" | running on GPU '{gpu_name}'")
        results_descriptions[gpu_name] = dict()
        for index, kernel in enumerate(kernels):
            kernel_name = kernel_names[index]
            searchspace_stats = SearchspaceStatistics(
                kernel_name=kernel_name,
                device_name=gpu_name,
                minimization=minimization,
                objective_time_keys=objective_time_keys,
                objective_performance_keys=objective_performance_keys,
            )

            # set cutoff point
            _, cutoff_point_fevals, cutoff_point_time = searchspace_stats.cutoff_point_fevals_time(cutoff_percentile)

            print(f" | - optimizing kernel '{kernel_name}'")
            results_descriptions[gpu_name][kernel_name] = dict()
            for strategy in strategies:
                strategy_name: str = strategy["name"]
                strategy_display_name: str = strategy["display_name"]
                stochastic = strategy["stochastic"]
                print(f" | - | using strategy '{strategy['display_name']}'")

                # setup the results description
                if not "options" in strategy:
                    strategy["options"] = dict()
                cutoff_margin = 1.1  # +10% margin, to make sure cutoff_point is reached by compensating for potential non-valid evaluations

                # TODO make sure this works correctly
                # if cutoff_type == 'time':
                #     strategy['options']['time_limit'] = cutoff_point_time * cutoff_margin
                # else:
                strategy["options"]["max_fevals"] = min(int(ceil(cutoff_point_fevals * cutoff_margin)), searchspace_stats.size)
                results_description = ResultsDescription(
                    experiment_folder_id,
                    kernel_name,
                    gpu_name,
                    strategy_name,
                    strategy_display_name,
                    stochastic,
                    objective_time_keys=objective_time_keys,
                    objective_performance_keys=objective_performance_keys,
                    minimization=minimization,
                )

                # if the strategy is in the cache, use cached data
                if "ignore_cache" not in strategy and results_description.has_results():
                    print(" | - |-> retrieved from cache")
                else:  # execute each strategy that is not in the cache
                    results_description = collect_results(kernel, strategy, results_description, profiling=profiling, error_value=1e20)

                # set the results
                results_descriptions[gpu_name][kernel_name][strategy_name] = results_description

    return experiment, strategies, results_descriptions


if __name__ == "__main__":
    experiment_filepath = get_args_from_cli(None)
    execute_experiment(experiment_filepath, profiling=False)
