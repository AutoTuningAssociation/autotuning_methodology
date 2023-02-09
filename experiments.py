""" Main experiments code """

import argparse
from importlib import import_module
from pathlib import Path
import json
from jsonschema import validate
import os
import sys
from typing import Tuple
import pathvalidate
import numpy as np
from math import ceil

from runner import collect_results
from caching import ResultsDescription
from searchspace_statistics import SearchspaceStatistics


def change_directory(path: str):
    absolute_path = os.path.abspath(path)
    os.chdir(absolute_path)
    sys.path.append(absolute_path)


def get_experiment(filename: str) -> dict:
    """ Validates and gets the experiment from the .json file """
    folder = 'experiment_files/'
    extension = '.json'
    if not filename.endswith(extension):
        filename = filename + extension
    path = filename
    if not filename.startswith(folder):
        path = folder + filename
    safe_path = pathvalidate.sanitize_filepath(path)
    schemafilepath = folder + 'schema.json'
    with open(safe_path) as file, open(schemafilepath) as schemafile:
        schema = json.load(schemafile)
        experiment = json.load(file)
        validate(instance=experiment, schema=schema)
        return experiment


def get_strategies(experiment: dict) -> dict:
    """ Gets the strategies from an experiments file by augmenting it with the defaults """
    strategy_defaults = experiment['strategy_defaults']
    strategies = experiment['strategies']
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
    """ Executes the experiment by retrieving it from the cache or running it """
    experiment = get_experiment(filepath)
    print(f"Starting experiment \'{experiment['name']}\'")
    experiment_folder_id = experiment.get('folder_id')
    minimization: bool = experiment.get('minimization', True)
    cutoff_percentile: float = experiment.get('cutoff_percentile', 1)
    cutoff_type: str = experiment.get('cutoff_type', "fevals")
    assert cutoff_type == 'fevals' or cutoff_type == 'time'
    curve_segment_factor: float = experiment.get('curve_segment_factor', 0.05)
    assert isinstance(curve_segment_factor, float)
    strategies = get_strategies(experiment)
    # add the kernel directory to path to import the module
    kernel_path = Path(experiment.get('kernel_path', ""))
    if not kernel_path.exists():
        raise FileNotFoundError(f"No such path {kernel_path}")
    sys.path.append(str(kernel_path))
    kernel_names = experiment['kernels']
    kernels = list(import_module(kernel_name) for kernel_name in kernel_names)
    # cachefiles_path = kernel_path + '../cachefiles'
    # change_directory("cached_data_used" + kernel_path)

    # variables for comparison
    objective_time_keys = ['times']
    objective_value_key = 'time'
    objective_values_key = 'times'

    # execute each strategy in the experiment per GPU and kernel
    results_descriptions: dict[str, dict[str, dict[str, ResultsDescription]]] = dict()
    gpu_name: str
    for gpu_name in experiment['GPUs']:
        print(f" | running on GPU '{gpu_name}'")
        results_descriptions[gpu_name] = dict()
        for index, kernel in enumerate(kernels):
            kernel_name = kernel_names[index]
            searchspace_stats = SearchspaceStatistics(kernel_name=kernel_name, device_name=gpu_name,
                                                      minimization=minimization)    # TODO add objective_performance_keys and objective_times_keys

            # set cutoff point
            _, cutoff_point_fevals, cutoff_point_time = searchspace_stats.cutoff_point_fevals_time(cutoff_percentile)

            print(f" | - optimizing kernel '{kernel_name}'")
            results_descriptions[gpu_name][kernel_name] = dict()
            for strategy in strategies:
                strategy_name: str = strategy['name']
                strategy_display_name: str = strategy['display_name']
                stochastic = strategy['stochastic']
                print(f" | - | using strategy '{strategy['display_name']}'")

                # setup the results description
                if not 'options' in strategy:
                    strategy['options'] = dict()
                cutoff_margin = 2.0    # +10% margin, to make sure cutoff_point is reached by compensating for potential non-valid evaluations

                # TODO make sure this works correctly (but how could it?)
                # if cutoff_type == 'time':
                #     strategy['options']['time_limit'] = cutoff_point_time * cutoff_margin
                # else:
                strategy['options']['max_fevals'] = int(ceil(cutoff_point_fevals * cutoff_margin))
                results_description = ResultsDescription(experiment_folder_id, kernel_name, gpu_name, strategy_name, strategy_display_name, stochastic,
                                                         objective_time_keys, objective_value_key, objective_values_key, minimization)

                # if the strategy is in the cache, use cached data
                if 'ignore_cache' not in strategy and results_description.has_results():
                    print(" | - |-> retrieved from cache")
                else:    # execute each strategy that is not in the cache
                    results_description = collect_results(kernel, strategy, results_description, profiling=profiling, error_value=1e20)

                # set the results
                results_descriptions[gpu_name][kernel_name][strategy_name] = results_description

    return experiment, strategies, results_descriptions


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("experiment", type=str, help="The experiment.json to execute, see experiments/template.json")
    args = CLI.parse_args()
    experiment_filepath = args.experiment
    if experiment_filepath is None:
        raise ValueError("Invalid '-experiment' option. Run 'experiments.py -h' to read more about the options.")

    execute_experiment(experiment_filepath, profiling=False)
