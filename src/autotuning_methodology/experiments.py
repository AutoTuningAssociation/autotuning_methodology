"""Main experiments code."""

from __future__ import annotations  # for correct nested type hints e.g. list[str], tuple[dict, str]

import json
import sys
from argparse import ArgumentParser
from importlib import import_module
from importlib.resources import files
from math import ceil
from os import getcwd
from pathlib import Path

from jsonschema import validate

from autotuning_methodology.caching import ResultsDescription
from autotuning_methodology.runner import collect_results
from autotuning_methodology.searchspace_statistics import SearchspaceStatistics


def get_args_from_cli(args=None) -> str:
    """Set the Command Line Interface arguments definitions, get and return the argument values.

    Args:
        args: optional list of arguments for testing without CLI interaction. Defaults to None.

    Raises:
        ValueError: on invalid argument.

    Returns:
        The filepath to the experiments file.
    """
    CLI = ArgumentParser()
    CLI.add_argument("experiment", type=str, help="The experiment.json to execute, see experiments/template.json")
    args = CLI.parse_args(args)
    filepath: str = args.experiment
    if filepath is None or filepath == "":
        raise ValueError(
            "Invalid '-experiment' option. Run 'visualize_experiments.py -h' to read more about the options."
        )
    return filepath


def get_experiment_schema_filepath():
    """Obtains and checks the filepath to the JSON schema.

    Returns:
        the filepath to the schema in Traversable format.
    """
    schemafile = files("autotuning_methodology").joinpath("schema.json")
    assert schemafile.is_file(), f"Path to schema.json does not exist, attempted path: {schemafile}"
    return schemafile


def get_experiment(filename: str) -> dict:
    """Validates and gets the experiment from the experiments .json file.

    Args:
        filename: path to the experiments .json file.

    Returns:
        Experiment dictionary object.
    """
    # get the path to the experiment file
    # folder_name = "experiment_files"
    extension = ".json"
    if not filename.endswith(extension):
        filename = filename + extension
    path = Path(filename)
    # if not filename.startswith(folder_name + "/"):
    #     path = folder / filename
    # else:
    #     path = Path(filename)
    assert path.exists(), f"Path to experiment file does not exist, attempted path: {path}, CWD: {getcwd()}"

    # get the path to the schema
    schemafile = get_experiment_schema_filepath()

    # open the experiment file and validate using the schema file
    with open(path) as file, open(schemafile) as schemafile:
        schema = json.load(schemafile)
        experiment: dict = json.load(file)
        validate(instance=experiment, schema=schema)
        return experiment


def get_strategies(experiment: dict) -> dict:
    """Gets the strategies from an experiments file by augmenting it with the defaults.

    Args:
        experiment: the experiment dictionary object.

    Returns:
        The strategies in the experiment dictionary object, augmented where necessery.
    """
    strategy_defaults = experiment["strategy_defaults"]
    strategies = experiment["strategies"]
    # # get a baseline index if it exists
    # baseline_index = list(
    #     strategy_index for strategy_index, strategy in enumerate(strategies) if "is_baseline" in strategy
    # )
    # if len(baseline_index) != 1:
    #     raise ValueError(f"There must be exactly one baseline, found {len(baseline_index)} baselines")
    # if strategies[baseline_index[0]]["is_baseline"] is not True:
    #     raise ValueError(f"is_baseline must be true, yet is set to {strategies[0]['is_baseline']}!")
    # # if the baseline index is not 0, put the baseline strategy first
    # if baseline_index[0] != 0:
    #     raise ValueError("The baseline strategy must be the first strategy in the experiments file!")
    #     # strategies.insert(0, strategies.pop(baseline_index[0]))

    # augment the strategies with the defaults
    for strategy in strategies:
        for default in strategy_defaults:
            if default not in strategy:
                strategy[default] = strategy_defaults[default]
    return strategies


def execute_experiment(filepath: str, profiling: bool = False) -> tuple[dict, dict, dict]:
    """Executes the experiment by retrieving it from the cache or running it.

    Args:
        filepath: path to the experiments .json file.
        profiling: whether profiling is enabled. Defaults to False.

    Raises:
        FileNotFoundError: if the path to the kernel specified in the experiments file is not found.

    Returns:
        A tuple of the experiment dictionary, the strategies executed, and the resulting list of ``ResultsDescription``.
    """
    experiment = get_experiment(filepath)
    experiment_folderpath = Path(filepath).parent
    print(f"Starting experiment '{experiment['name']}'")
    experiment_folder_id: str = experiment["folder_id"]
    minimization: bool = experiment.get("minimization", True)
    cutoff_percentile: float = experiment.get("cutoff_percentile", 1)
    cutoff_type: str = experiment.get("cutoff_type", "fevals")
    assert cutoff_type == "fevals" or cutoff_type == "time", f"cutoff_type must be 'fevals' or 'time', is {cutoff_type}"
    curve_segment_factor: float = experiment.get("curve_segment_factor", 0.05)
    assert isinstance(curve_segment_factor, float), f"curve_segment_factor is not float, {type(curve_segment_factor)}"
    strategies = get_strategies(experiment)

    # add the kernel directory to the path to import the module, relative to the experiment file
    kernels_path = experiment_folderpath / Path(experiment["kernels_path"])
    if not kernels_path.exists():
        raise FileNotFoundError(f"No such path {kernels_path.resolve()}, CWD: {getcwd()}")
    sys.path.append(str(kernels_path))
    kernel_names = experiment["kernels"]
    kernels = list(import_module(kernel_name) for kernel_name in kernel_names)

    # variables for comparison
    objective_time_keys: list[str] = experiment["objective_time_keys"]
    objective_performance_keys: list[str] = experiment["objective_performance_keys"]

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
                bruteforced_caches_path=experiment_folderpath / experiment["bruteforced_caches_path"],
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
                if "options" not in strategy:
                    strategy["options"] = dict()
                cutoff_margin = 1.1  # +10% margin, to make sure cutoff_point is reached by compensating for potential non-valid evaluations  # noqa: E501

                # TODO make sure this works correctly
                # if cutoff_type == 'time':
                #     strategy['options']['time_limit'] = cutoff_point_time * cutoff_margin
                # else:
                strategy["options"]["max_fevals"] = min(
                    int(ceil(cutoff_point_fevals * cutoff_margin)), searchspace_stats.size
                )
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
                    visualization_caches_path=experiment_folderpath / experiment["visualization_caches_path"],
                )

                # if the strategy is in the cache, use cached data
                if "ignore_cache" not in strategy and results_description.has_results():
                    print(" | - |-> retrieved from cache")
                else:  # execute each strategy that is not in the cache
                    results_description = collect_results(
                        kernel, strategy, results_description, searchspace_stats, profiling=profiling
                    )

                # set the results
                results_descriptions[gpu_name][kernel_name][strategy_name] = results_description

    return experiment, strategies, results_descriptions


def entry_point():  #  pragma: no cover
    """Entry point function for Experiments."""
    experiment_filepath = get_args_from_cli()
    execute_experiment(experiment_filepath, profiling=False)


if __name__ == "__main__":
    entry_point()
