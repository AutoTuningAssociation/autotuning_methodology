""" Interface to run an experiment on Kernel Tuner """
from cProfile import label
from copy import deepcopy
from math import sqrt, floor, ceil
import numpy as np
import progressbar
from typing import Any, Tuple, Dict
import time as python_time
import warnings
import yappi
from scipy.interpolate import interp1d

record_data = ['mean_actual_num_evals']


def remove_duplicates(res: list, remove_duplicate_results: bool):
    """ Removes duplicate configurations from the results """
    if not remove_duplicate_results:
        return res
    unique_res = list()
    for result in res:
        if result not in unique_res:
            unique_res.append(result)
    return unique_res


def tune(kernel, kernel_name: str, device_name: str, strategy: dict, tune_options: dict, profiling: bool) -> Tuple[list, int]:
    """ Execute a strategy, return the result, runtime and optional profiling statistics """

    def tune_with_kerneltuner():
        """ interface with kernel tuner to tune the kernel and return the results """
        if profiling:
            yappi.set_clock_type("cpu")
            yappi.start()
        res, env = kernel.tune(device_name=device_name, strategy=strategy['strategy'], strategy_options=strategy['options'], **tune_options)
        if profiling:
            yappi.stop()
        return res, env

    total_start_time = python_time.perf_counter()
    warnings.simplefilter("ignore", UserWarning)
    try:
        res, _ = tune_with_kerneltuner()
    except ValueError:
        print("Something went wrong, trying once more.")
        res, _ = tune_with_kerneltuner()
    warnings.simplefilter("default", UserWarning)
    total_end_time = python_time.perf_counter()
    total_time_ms = round((total_end_time - total_start_time) * 1000)
    # TODO when profiling, should the total_time_ms not be the time from profiling_stats? Otherwise we are timing the profiling code as well
    return res, total_time_ms


def collect_results(kernel, kernel_name: str, device_name: str, strategy: dict, expected_results: dict, profiling: bool, cutoff_point_fevals: int,
                    objective_value_at_cutoff_point: float, optimization_objective='time', remove_duplicate_results=True, time_resolution=1e4,
                    time_interpolated_axis=None, y_min=None, y_median=None, segment_factor=0.05) -> dict:
    """ Executes strategies to obtain (or retrieve from cache) the statistical data """
    print(f"Running {strategy['display_name']}")
    min_num_evals = strategy['minimum_number_of_evaluations']
    # TODO put the tune options in the .json in strategy_defaults?
    tune_options = {
        'verbose': False,
        'quiet': True,
        'simulation_mode': True
    }

    def report_multiple_attempts(rep: int, len_res: int, len_unique_res: int, strategy_repeats: int):
        """ If multiple attempts are necessary, report the reason """
        if len_res < 1:
            print(f"({rep+1}/{strategy_repeats}) No results found, trying once more...")
        elif len_unique_res < min_num_evals:
            print(f"Too few unique results found ({len_unique_res} in {len_res} evaluations), trying once more...")
        else:
            print(f"({rep+1}/{strategy_repeats}) Only invalid results found, trying once more...")

    # repeat the strategy as specified
    repeated_results = list()
    total_time_results = np.array([])
    for rep in progressbar.progressbar(range(strategy['repeats']), redirect_stdout=True):
        attempt = 0
        only_invalid = True
        while only_invalid or (remove_duplicate_results and len_unique_res < min_num_evals):
            if attempt > 0:
                report_multiple_attempts(rep, len_res, len_unique_res, strategy['repeats'])
            res, total_time_ms = tune(kernel, kernel_name, device_name, strategy, tune_options, profiling)
            # TODO continue here with confidence interval
            len_res: int = len(res)
            # check if there are only invalid configs in the first 10 fevals, if so, try again
            only_invalid = len_res < 1 or min(res[:10], key=lambda x: x['time'])['time'] == 1e20
            unique_res = remove_duplicates(res, remove_duplicate_results)
            len_unique_res: int = len(unique_res)
            attempt += 1
        # register the results
        repeated_results.append(unique_res)
        total_time_results = np.append(total_time_results, total_time_ms)

    # gather profiling data and clear the profiler before the next round
    if profiling:
        stats = yappi.get_func_stats()
        # stats.print_all()
        path = "../experiments/profilings/random/profile-v2.prof"
        stats.save(path, type="pstat")    # pylint: disable=no-member
        yappi.clear_stats()

    # create the interpolated results from the repeated results
    results = create_interpolated_results(repeated_results, expected_results, optimization_objective, cutoff_point_fevals, objective_value_at_cutoff_point,
                                          time_resolution, time_interpolated_axis, y_min, y_median, segment_factor)

    # check that all expected results are present
    for key in results.keys():
        if key == 'cutoff_quantile' or key == 'curve_segment_factor':
            continue
        if results[key] is None:
            raise ValueError(f"Expected result {key} was not filled in the results")
    return results
