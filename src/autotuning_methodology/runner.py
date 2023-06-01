"""Interface to run an experiment on the auto-tuning frameworks."""

from __future__ import annotations  # for correct nested type hints e.g. list[str], tuple[dict, str]

import contextlib
import json
import os
import time as python_time
import warnings
from inspect import getfile
from pathlib import Path

import numpy as np
import progressbar
import yappi

from autotuning_methodology.caching import ResultsDescription

error_types_strings = ["", "InvalidConfig", "CompilationFailedConfig", "RuntimeFailedConfig"]
kernel_tuner_error_value = 1e20


@contextlib.contextmanager
def temporary_working_directory_change(new_WD: Path):
    """Temporarily change to the given working directory in a context. Based on https://stackoverflow.com/questions/75048986/way-to-temporarily-change-the-directory-in-python-to-execute-code-without-affect.

    Args:
        new_WD: path of the working directory to temporarily change to.
    """
    assert new_WD.exists()

    # save the current working directory so we can revert to it
    original_working_directory = os.getcwd()

    # potentially raises an exception, left to the caller
    os.chdir(new_WD)

    # yield control to the caller
    try:
        yield

    # change back to the original working directory
    finally:
        # potentially raises an exception, left to the caller
        os.chdir(original_working_directory)


def is_invalid_objective_performance(objective_performance: float) -> bool:
    """Returns whether an objective value is invalid by checking against NaN and the error value.

    Args:
        objective_performance: the objective performance value to check.

    Raises:
        TypeError: if the ``objective_performance`` value has an unexpected type.

    Returns:
        True if the ``objective_performance`` value is invalid, False otherwise.
    """
    if any(str(objective_performance) == error_type_string for error_type_string in error_types_strings):
        return True
    if not isinstance(objective_performance, (int, float)):
        raise TypeError(
            f"""Objective value should be of type float,
                but is of type {type(objective_performance)} with value {objective_performance}"""
        )
    return np.isnan(objective_performance) or objective_performance == kernel_tuner_error_value


def is_invalid_objective_time(objective_time: float) -> bool:
    """Returns whether an objective time is invalid.

    Args:
        objective_time: the objective time value to check.

    Returns:
        True if the ``objective_time`` value is invalid, False otherwise.
    """
    return np.isnan(objective_time) or objective_time < 0


def is_valid_config_result(config: dict) -> bool:
    """Returns whether a given configuration is valid.

    Args:
        config: the configuration to check.

    Returns:
         True if the ``config`` is valid, False otherwise.
    """
    return "invalidity" in config and config["invalidity"] == "correct"


def get_results_and_metadata(
    filename_results: str = "../last_run/_tune_configuration-results.json",
    filename_metadata: str = "../last_run/_tune_configuration-metadata.json",
) -> tuple[list, list]:
    """Load the results and metadata files (relative to kernel directory) in accordance with the defined T4 standards.

    Args:
        filename_results: filepath relative to kernel. Defaults to "../last_run/_tune_configuration-results.json".
        filename_metadata: filepath relative to kernel. Defaults to "../last_run/_tune_configuration-metadata.json".

    Returns:
        A tuple of the results and metadata lists respectively.
    """
    assert Path(filename_results).exists(), f"File {filename_results} does not exist relative to {os.getcwd()}"
    assert Path(filename_metadata).exists(), f"File {filename_metadata} does not exist relative to {os.getcwd()}"
    with open(filename_results, "r") as file_results:
        results: list = json.load(file_results)["results"]
    with open(filename_metadata, "r") as file_metadata:
        metadata: list = json.load(file_metadata)["metadata"]
    return metadata, results


def tune(
    kernel, kernel_name: str, device_name: str, strategy: dict, tune_options: dict, profiling: bool
) -> tuple[list, list, int]:
    """Tune a program using an optimization algorithm and collect the results.

    Optionally collects profiling statistics.

    Args:
        kernel: the program (kernel) to tune.
        kernel_name: the name of the program to tune.
        device_name: the device (GPU) to tune on.
        strategy: the optimization algorithm to optimize with.
        tune_options: a special options dictionary passed along to Kernel Tuner.
        profiling: whether profiling statistics must be collected.

    Raises:
        ValueError: if tuning fails multiple times in a row.

    Returns:
        A tuple of the metadata, the results, and the total runtime in miliseconds.
    """

    def tune_with_kerneltuner():
        """Interface with kernel tuner to tune the kernel and return the results."""
        # get the path to the directory the kernel is in; can't use importlib.resources.files because its not a package
        kernel_directory = Path(getfile(kernel)).parent
        assert kernel_directory.is_dir()

        # change CWD to the directory of the kernel
        with temporary_working_directory_change(kernel_directory):
            if profiling:
                yappi.set_clock_type("cpu")
                yappi.start()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res, env = kernel.tune(
                    device_name=device_name,
                    strategy=strategy["strategy"],
                    strategy_options=strategy["options"],
                    **tune_options,
                )
            if profiling:
                yappi.stop()
            metadata, results = get_results_and_metadata(
                filename_results=kernel.file_path_results, filename_metadata=kernel.file_path_metadata
            )
        if "max_fevals" in strategy["options"]:
            max_fevals = strategy["options"]["max_fevals"]
            if len(results) < max_fevals * 0.1:
                warnings.warn(f"Much fewer configurations were returned ({len(res)}) than the requested {max_fevals}")
            if len(results) < 2:
                raise ValueError("Less than two configurations were returned")
        return metadata, results

    def tune_with_BAT():
        """Interface to tune with the BAT benchmarking suite."""
        # TODO integrate with BAT

    total_start_time = python_time.perf_counter()
    warnings.simplefilter("ignore", UserWarning)
    try:
        metadata, results = tune_with_kerneltuner()
    except ValueError:
        print("Something went wrong, trying once more.")
        metadata, results = tune_with_kerneltuner()
    warnings.simplefilter("default", UserWarning)
    total_end_time = python_time.perf_counter()
    total_time_ms = round((total_end_time - total_start_time) * 1000)
    # be careful not to rely on total_time_ms when profiling, because it will include profiling time
    return metadata, results, total_time_ms


def collect_results(
    kernel, strategy: dict, results_description: ResultsDescription, profiling: bool
) -> ResultsDescription:
    """Executes optimization algorithms on tuning problems to capture their behaviour.

    Args:
        kernel: the program (kernel) to tune.
        strategy: the optimization algorithm to optimize with.
        results_description: the ``ResultsDescription`` object to write the results to.
        profiling: whether profiling statistics must be collected.

    Returns:
        The ``ResultsDescription`` object with the results.
    """
    min_num_evals: int = strategy["minimum_number_of_evaluations"]
    # TODO put the tune options in the .json in strategy_defaults? Make it Kernel Tuner independent
    tune_options = {"verbose": False, "quiet": True, "simulation_mode": True}

    def report_multiple_attempts(rep: int, len_res: int, strategy_repeats: int):
        """If multiple attempts are necessary, report the reason."""
        if len_res < 1:
            print(f"({rep+1}/{strategy_repeats}) No results found, trying once more...")
        elif len_res < min_num_evals:
            print(f"Too few results found ({len_res} of {min_num_evals} required), trying once more...")
        else:
            print(f"({rep+1}/{strategy_repeats}) Only invalid results found, trying once more...")

    # repeat the strategy as specified
    repeated_results = list()
    total_time_results = np.array([])
    for rep in progressbar.progressbar(
        range(strategy["repeats"]),
        redirect_stdout=True,
        prefix=" | - |-> running: ",
        widgets=[
            progressbar.PercentageLabelBar(),
            " [",
            progressbar.SimpleProgress(format="%(value_s)s/%(max_value_s)s"),
            ", ",
            progressbar.Timer(format="Elapsed: %(elapsed)s"),
            ", ",
            progressbar.ETA(),
            "]",
        ],
    ):
        attempt = 0
        only_invalid = True
        len_res: int = -1
        while only_invalid or len_res < min_num_evals:
            if attempt > 0:
                report_multiple_attempts(rep, len_res, strategy["repeats"])
            metadata, results, total_time_ms = tune(
                kernel,
                results_description.kernel_name,
                results_description.device_name,
                strategy,
                tune_options,
                profiling,
            )
            len_res = len(results)
            # check if there are only invalid configs in the first min_num_evals, if so, try again
            temp_res_filtered = list(filter(lambda config: is_valid_config_result(config), results))
            only_invalid = len(temp_res_filtered) < 1
            attempt += 1
        # register the results
        repeated_results.append(results)
        total_time_results = np.append(total_time_results, total_time_ms)

    # gather profiling data and clear the profiler before the next round
    if profiling:
        stats = yappi.get_func_stats()
        # stats.print_all()
        path = "../old_experiments/profilings/random/profile-v2.prof"
        stats.save(path, type="pstat")  # pylint: disable=no-member
        yappi.clear_stats()

    # combine the results to numpy arrays and write to a file
    write_results(repeated_results, results_description)
    assert results_description.has_results(), "No results in ResultsDescription after writing results."
    return results_description


def write_results(repeated_results: list, results_description: ResultsDescription):
    """Combine the results and write them to a NumPy file.

    Args:
        repeated_results: a list of tuning results, one per tuning session.
        results_description: the ``ResultsDescription`` object to write the results to.
    """
    # get the objective value and time keys
    objective_time_keys = results_description.objective_time_keys
    objective_performance_keys = results_description.objective_performance_keys

    # find the maximum number of function evaluations
    max_num_evals = max(len(repeat) for repeat in repeated_results)

    def get_nan_array() -> np.ndarray:
        """Get an array of NaN so they are not counted as zeros inadvertedly."""
        return np.full((max_num_evals, len(repeated_results)), np.nan)

    # set the arrays to write to
    fevals_results = get_nan_array()
    objective_time_results = get_nan_array()
    objective_performance_results = get_nan_array()
    objective_performance_best_results = get_nan_array()
    objective_performance_stds = get_nan_array()
    objective_time_results_per_key = np.full((len(objective_time_keys), max_num_evals, len(repeated_results)), np.nan)
    objective_performance_results_per_key = np.full(
        (len(objective_time_keys), max_num_evals, len(repeated_results)), np.nan
    )

    # combine the results
    opt_func = np.nanmin if results_description.minimization is True else np.nanmax
    for repeat_index, repeat in enumerate(repeated_results):
        cumulative_objective_time = 0
        objective_performance_best = np.nan
        for evaluation_index, evaluation in enumerate(repeat):
            # set the number of function evaluations
            fevals_results[evaluation_index, repeat_index] = (
                evaluation_index + 1
            )  # number of function evaluations are counted from 1 instead of 0

            # in case of an invalid config, there is nothing to be registered
            if str(evaluation["invalidity"]) == "constraints":
                warnings.warn("Invalid config found, this should have been caught by the framework constraints")
                continue

            # TODO continue here with implementing switch in output format
            # obtain the objective time per key
            objective_times_list = list()
            for key_index, key in enumerate(objective_time_keys):
                evaluation_times = evaluation["times"]
                assert (
                    key in evaluation_times
                ), f"Objective time key {key} not in evaluation['times'] ({evaluation_times})"
                value = evaluation_times[key]
                if value is not None and not is_invalid_objective_time(value):
                    value = value / 1000  # TODO this miliseconds to seconds conversion is specific to Kernel Tuner
                    objective_time_results_per_key[key_index, evaluation_index, repeat_index] = value
                    objective_times_list.append(value)
            # sum the objective times of the keys
            if len(objective_times_list) >= 1:
                objective_time = sum(objective_times_list)
                if not is_invalid_objective_time(objective_time):
                    cumulative_objective_time += objective_time
                    objective_time_results[evaluation_index, repeat_index] = cumulative_objective_time

            # obtain the objective performance per key (called 'measurements' in the T4 format)
            objective_performances_list = list()
            for key_index, key in enumerate(objective_performance_keys):
                evaluation_measurements = evaluation["measurements"]
                measurements = list(filter(lambda m: m["name"] == key, evaluation_measurements))
                assert (
                    len(measurements) > 0
                ), f"Objective performance key name {key} not in evaluation['measurements'] ({evaluation_measurements})"
                assert (
                    len(measurements) == 1
                ), f"""Objective performance key name {key} multiply defined
                        in evaluation['measurements'] ({evaluation_measurements})"""
                value = measurements[0]["value"]
                if value is not None and not is_invalid_objective_performance(value):
                    objective_performance_results_per_key[key_index, evaluation_index, repeat_index] = value
                    objective_performances_list.append(value)
            # sum the objective performances of the keys
            if len(objective_performances_list) >= 1:
                objective_performance = sum(objective_performances_list)
                if not is_invalid_objective_performance(objective_performance):
                    objective_performance_results[evaluation_index, repeat_index] = objective_performance
                    objective_performance_best = opt_func([objective_performance, objective_performance_best])

            # set the best objective performance
            if not is_invalid_objective_performance(objective_performance_best):
                objective_performance_best_results[evaluation_index, repeat_index] = objective_performance_best

    # write to file
    numpy_arrays = {
        "fevals_results": fevals_results,
        "objective_time_results": objective_time_results,
        "objective_performance_results": objective_performance_results,
        "objective_performance_best_results": objective_performance_best_results,
        "objective_performance_stds": objective_performance_stds,
        "objective_time_results_per_key": objective_time_results_per_key,
        "objective_performance_results_per_key": objective_performance_results_per_key,
    }
    results_description.set_results(numpy_arrays)
