""" Interface to run an experiment on Kernel Tuner """
from cProfile import label
import numpy as np
import progressbar
from typing import Any, Tuple, Dict
import time as python_time
import warnings
import yappi
from caching import ResultsDescription


def is_invalid_objective_value(objective_value: float, error_value) -> bool:
    """ Returns whether an objective value is invalid by checking against NaN and the error value """
    return np.isnan(objective_value) or objective_value == error_value


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


def collect_results(kernel, strategy: dict, results_description: ResultsDescription, profiling: bool, minimization: bool, error_value) -> ResultsDescription:
    """ Executes optimization algorithms to capture optimization algorithm behaviour """
    print(f"Running {strategy['display_name']}")
    min_num_evals: int = strategy['minimum_number_of_evaluations']
    # TODO put the tune options in the .json in strategy_defaults?
    tune_options = {
        'verbose': False,
        'quiet': True,
        'simulation_mode': True
    }

    def report_multiple_attempts(rep: int, len_res: int, strategy_repeats: int):
        """ If multiple attempts are necessary, report the reason """
        if len_res < 1:
            print(f"({rep+1}/{strategy_repeats}) No results found, trying once more...")
        elif len_res < min_num_evals:
            print(f"Too few results found ({len_res} of {min_num_evals} required), trying once more...")
        else:
            print(f"({rep+1}/{strategy_repeats}) Only invalid results found, trying once more...")

    # repeat the strategy as specified
    repeated_results = list()
    total_time_results = np.array([])
    for rep in progressbar.progressbar(range(strategy['repeats']), redirect_stdout=True):
        attempt = 0
        only_invalid = True
        while only_invalid or len_res < min_num_evals:
            if attempt > 0:
                report_multiple_attempts(rep, len_res, strategy['repeats'])
            res, total_time_ms = tune(kernel, results_description.kernel_name, results_description.device_name, strategy, tune_options, profiling)
            # TODO continue here with confidence interval
            len_res: int = len(res)
            # check if there are only invalid configs in the first min_num_evals, if so, try again
            only_invalid = len_res < 1 or min(res[:min_num_evals], key=lambda x: x['time'])['time'] == error_value
            attempt += 1
        # register the results
        repeated_results.append(res)
        total_time_results = np.append(total_time_results, total_time_ms)

    # gather profiling data and clear the profiler before the next round
    if profiling:
        stats = yappi.get_func_stats()
        # stats.print_all()
        path = "../experiments/profilings/random/profile-v2.prof"
        stats.save(path, type="pstat")    # pylint: disable=no-member
        yappi.clear_stats()

    # combine the results to numpy arrays and write to a file
    write_results(repeated_results, results_description, minimization, error_value=error_value)
    assert results_description.has_results()
    return results_description


def write_results(repeated_results: list, results_description: ResultsDescription, minimization: bool, error_value):
    """ Combine the results and write them to a numpy file """

    # get the objective value and time keys
    objective_time_keys = results_description.objective_time_keys
    objective_value_key = results_description.objective_value_key
    objective_values_key = results_description.objective_values_key

    # find the maximum number of function evaluations
    max_num_evals = max(len(repeat) for repeat in repeated_results)

    def get_nan_array() -> np.ndarray:
        """ get an array of NaN so they are not counted as zeros inadvertedly """
        return np.full((max_num_evals, len(repeated_results)), np.nan)

    # set the arrays to write to
    fevals_results = get_nan_array()
    time_results = get_nan_array()
    objective_time_results = get_nan_array()
    objective_value_results = get_nan_array()
    objective_value_best_results = get_nan_array()
    objective_value_stds = get_nan_array()

    # combine the results
    opt_func = np.nanmin if minimization is True else np.nanmax

    for repeat_index, repeat in enumerate(repeated_results):
        cumulative_total_time = 0
        cumulative_objective_time = 0
        objective_value_best = np.nan
        for evaluation_index, evaluation in enumerate(repeat):
            objective_value = evaluation[objective_value_key]
            if not is_invalid_objective_value(objective_value, error_value):
                # extract the objectives and time spent
                if not np.isnan(cumulative_total_time):
                    cumulative_total_time += sum(evaluation['times']) / 1000    # TODO the miliseconds to seconds conversion is specific to Kernel Tuner
                if not np.isnan(cumulative_objective_time):
                    cumulative_objective_time += sum(sum(evaluation[time_key]) for time_key in objective_time_keys) / 1000
                objective_value_best = opt_func([objective_value, objective_value_best])
                objective_value_std = np.std(list(e for e in evaluation[objective_values_key] if e is not error_value))
            else:
                # set the values to NaN
                cumulative_total_time = np.NaN
                cumulative_objective_time = np.NaN

            # write to the arrays
            fevals_results[evaluation_index, repeat_index] = evaluation_index
            time_results[evaluation_index, repeat_index] = cumulative_total_time
            objective_time_results[evaluation_index, repeat_index] = cumulative_objective_time
            if not is_invalid_objective_value(objective_value, error_value):    # if it is an error value, it must stay NaN
                objective_value_results[evaluation_index, repeat_index] = objective_value
                objective_value_stds[evaluation_index, repeat_index] = objective_value_std
            if not np.isnan(objective_value_best):
                objective_value_best_results[evaluation_index, repeat_index] = objective_value_best

    # write to file
    numpy_arrays = {
        'fevals_results': fevals_results,
        'time_results': time_results,
        'objective_time_results': objective_time_results,
        'objective_value_results': objective_value_results,
        'objective_value_best_results': objective_value_best_results,
        'objective_value_stds': objective_value_stds
    }
    return results_description.set_results(numpy_arrays)
