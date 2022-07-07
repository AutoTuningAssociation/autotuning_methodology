""" Main experiments code """

import argparse
from importlib import import_module
import json
import os
import sys
from typing import Tuple, Any
import pathvalidate
from copy import deepcopy
import numpy as np

from runner import collect_results
from caching import CachedObject


def get_searchspaces_info_stats() -> dict[str, Any]:
    """ read the searchspaces info statistics dictionary from file """
    with open("../cached_data_used/kernel_info.json") as file:
        kernels_device_info_data = file.read()
    return json.loads(kernels_device_info_data)


def change_directory(path: str):
    absolute_path = os.path.abspath(path)
    os.chdir(absolute_path)
    sys.path.append(absolute_path)


def get_experiment(filename: str) -> dict:
    """ Gets the experiment from the .json file """
    folder = 'experiments/'
    extension = '.json'
    if not filename.endswith(extension):
        filename = filename + extension
    path = filename
    if not filename.startswith(folder):
        path = folder + filename
    safe_path = pathvalidate.sanitize_filepath(path)
    with open(safe_path) as file:
        experiment = json.load(file)
        return experiment


def get_strategies(experiment: dict) -> dict:
    """ Gets the strategies from an experiments file by augmenting it with the defaults """
    strategy_defaults = experiment['strategy_defaults']
    strategies = experiment['strategies']
    # get a baseline index if it exists
    baseline_index = list(strategy_index for strategy_index, strategy in enumerate(strategies) if 'is_baseline' in strategy)
    if len(baseline_index) != 1:
        raise ValueError(f"There must be exactly one baseline, found {len(baseline_index)} baselines")
    if strategies[baseline_index[0]]['is_baseline'] != True:
        raise ValueError(f"is_baseline must be true, yet is set to {strategies[0]['is_baseline']}!")
    # if the baseline index is not 0, put the baseline strategy first
    if baseline_index[0] != 0:
        raise ValueError("The baseline strategy must be the first strategy in the experiments file!")
        # strategies.insert(0, strategies.pop(baseline_index[0]))

    # augment the strategies with the defaults
    for strategy in strategies:
        for default in strategy_defaults:
            if not default in strategy:
                strategy[default] = strategy_defaults[default]
    return strategies


def create_expected_results() -> dict:
    """ Creates a dict to put the expected results into """
    expected_results = dict({
        'total_times': None,
        'cutoff_quantile': None,
        'curve_segment_factor': None,
        'num_function_evaluations': None,
        'best_found_objective_values': None,
        'interpolated_time': None,
        'interpolated_objective': None,
        'interpolated_objective_std': None,
        'interpolated_objective_error_lower': None,
        'interpolated_objective_error_upper': None,
    })
    return expected_results


def execute_experiment(filepath: str, profiling: bool, searchspaces_info_stats: dict) -> Tuple[dict, dict, dict]:
    """ Executes the experiment by retrieving it from the cache or running it """
    experiment = get_experiment(filepath)
    print(f"Starting experiment \'{experiment['name']}\'")
    kernel_path = experiment.get('kernel_path', "")
    cutoff_quantile = experiment.get('cutoff_quantile', 0.99)
    curve_segment_factor = experiment.get('curve_segment_factor', 0.05)
    assert isinstance(curve_segment_factor, float)
    time_resolution = experiment.get('resolution', 1e4)
    if int(time_resolution) != time_resolution:
        raise ValueError(f"The resolution must be an integer, yet is {time_resolution}.")
    time_resolution = int(time_resolution)
    change_directory("../cached_data_used" + kernel_path)
    strategies = get_strategies(experiment)
    kernel_names = experiment['kernels']
    kernels = list(import_module(kernel_name) for kernel_name in kernel_names)

    # execute each strategy in the experiment per GPU and kernel
    caches: dict[str, dict[str, Any]] = dict()
    gpu_name: str
    for gpu_name in experiment['GPUs']:
        caches[gpu_name] = dict()
        for index, kernel in enumerate(kernels):
            kernel_name = kernel_names[index]
            stats_info = searchspaces_info_stats[gpu_name]['kernels'][kernel_name]
            objective_value_at_cutoff_point = np.quantile(np.array(stats_info['sorted_times']), 1-cutoff_quantile)   # sorted in ascending order, so inverse quantile
            y_min = None
            y_median = None
            if 'absolute_optimum' in stats_info and 'median' in stats_info:
                y_min = stats_info['absolute_optimum']
                y_median = stats_info['median']
            print(f"  running {kernel_name} on {gpu_name}")
            # get or create a cache to write the results to
            cache = CachedObject(kernel_name, gpu_name, deepcopy(strategies))
            baseline_time_interpolated = None
            baseline_executed = False
            for strategy in strategies:
                print(f"    | with strategy {strategy['display_name']}")
                # if the strategy is in the cache, use cached data
                expected_results = create_expected_results()
                if 'ignore_cache' not in strategy and baseline_executed is False:
                    cached_data = cache.get_strategy_results(strategy['name'], strategy['options'], strategy['repeats'], expected_results)
                    if cached_data is not None and 'cutoff_quantile' in cached_data['results'] and cached_data['results']['cutoff_quantile'] == cutoff_quantile and 'curve_segment_factor' in cached_data['results'] and cached_data['results']['curve_segment_factor'] == curve_segment_factor:
                        print("| retrieved from cache")
                        if baseline_time_interpolated is None and 'is_baseline' in strategy and strategy['is_baseline'] is True:
                            baseline_time_interpolated = np.array(cached_data['results']['interpolated_time'])
                        continue

                # execute each strategy that is not in the cache
                strategy_results = collect_results(kernel, kernel_name, gpu_name, strategy, expected_results, profiling, objective_value_at_cutoff_point, time_resolution=time_resolution, time_interpolated_axis=baseline_time_interpolated, y_min=y_min, y_median=y_median, segment_factor=curve_segment_factor)
                if 'cutoff_quantile' in expected_results:
                    strategy_results['cutoff_quantile'] = cutoff_quantile
                if 'curve_segment_factor' in expected_results:
                    strategy_results['curve_segment_factor'] = curve_segment_factor

                # if this strategy is used as the baseline, keep its x-axis (time dimension) as the baseline along which the other values are interpolated
                if baseline_time_interpolated is None and 'is_baseline' in strategy and strategy['is_baseline'] is True:
                    baseline_time_interpolated = strategy_results['interpolated_time']
                    baseline_executed = True    # if the baseline has been (re)executed, the other cached strategies must be re-executed as the interpolated time axis has changed
                if baseline_time_interpolated is None:
                    raise ValueError(f"baseline_time_interpolated should not be None here, check whether the first strategy has 'is_baseline' set to True")
                # double check that the interpolated results are as expected
                assert np.array_equal(baseline_time_interpolated, strategy_results['interpolated_time'])
                assert len(baseline_time_interpolated) == len(strategy_results['interpolated_objective'])
                # write the results to the cache
                cache.set_strategy(deepcopy(strategy), strategy_results)
            caches[gpu_name][kernel_name] = cache

    return experiment, strategies, caches


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("experiment", type=str, help="The experiment.json to execute, see experiments/template.json")
    args = CLI.parse_args()
    experiment_filepath = args.experiment
    if experiment_filepath is None:
        raise ValueError("Invalid '-experiment' option. Run 'experiments.py -h' to read more about the options.")

    execute_experiment(experiment_filepath, profiling=False, searchspaces_info_stats=get_searchspaces_info_stats())
