"""Interface to run an experiment on the auto-tuning frameworks."""

from __future__ import annotations  # for correct nested type hints e.g. list[str], tuple[dict, str]

import contextlib
import json
import os
import time as python_time
import warnings
from pathlib import Path

# compression libraries if necessary for collecting results
import pickle
import gzip

import numpy as np
import progressbar
import yappi

from autotuning_methodology.caching import ResultsDescription
from autotuning_methodology.searchspace_statistics import SearchspaceStatistics, convert_from_time_unit
from autotuning_methodology.validators import (
    is_invalid_objective_performance,
    is_invalid_objective_time,
    is_valid_config_result,
    validate_T4,
)

# TODO this does not conform to new intended dicrectory structure
folder = Path(__file__).parent.parent.parent

# Imported runs must be remapped to have the same keys, values and order of parameters as the other runs.
# This mapping provides both the order and mapping, so all keys must be present.
# Default value is a tuple where the first element is the new parameter name and the second the mapped value.
# Arrays of tuples allow mapping from one parameter to multiple.
# 'None' values are skipped.
ktt_param_mapping = {
    "convolution": {
        "BLOCK_SIZE_X": ("block_size_x", lambda x: x),
        "BLOCK_SIZE_Y": ("block_size_y", lambda x: x),
        "HFS": [("filter_height", 15), ("filter_width", 15)],
        "READ_ONLY": ("read_only", lambda x: x),
        "TILE_SIZE_X": ("tile_size_x", lambda x: x),
        "TILE_SIZE_Y": ("tile_size_y", lambda x: x),
        "PADDING": ("use_padding", lambda x: x),
        "IMAGE_WIDTH": None,
        "IMAGE_HEIGHT": None,
    },
    "pnpoly": {
        "BETWEEN_METHOD": ("between_method", lambda x: x),
        "BLOCK_SIZE_X": ("block_size_x", lambda x: x),
        "TILE_SIZE": ("tile_size", lambda x: x),
        "USE_METHOD": ("use_method", lambda x: x),
        "VERTICES": None,
    },
}
ktt_param_mapping["mocktest_kernel_convolution"] = ktt_param_mapping["convolution"]


@contextlib.contextmanager
def temporary_working_directory_change(new_wd: Path):
    """Temporarily change to the given working directory in a context. Based on https://stackoverflow.com/questions/75048986/way-to-temporarily-change-the-directory-in-python-to-execute-code-without-affect.

    Args:
        new_wd: path of the working directory to temporarily change to.
    """
    assert new_wd.exists()

    # save the current working directory so we can revert to it
    original_working_directory = os.getcwd()

    # potentially raises an exception, left to the caller
    os.chdir(new_wd)

    # yield control to the caller
    try:
        yield

    # change back to the original working directory
    finally:
        # potentially raises an exception, left to the caller
        os.chdir(original_working_directory)


def load_json(path: Path):
    """Helper function to load a JSON file."""
    assert path.exists(), f"File {str(path)} does not exist relative to {os.getcwd()}"
    with path.open() as file_results:
        return json.load(file_results)


def get_kerneltuner_results_and_metadata(
    filename_results: str = f"{folder}../last_run/_tune_configuration-results.json",
    filename_metadata: str = f"{folder}../last_run/_tune_configuration-metadata.json",
) -> tuple[list, list]:
    """Load the results and metadata files (relative to kernel directory) in accordance with the defined T4 standards.

    Args:
        filename_results: filepath relative to kernel. Defaults to "../last_run/_tune_configuration-results.json".
        filename_metadata: filepath relative to kernel. Defaults to "../last_run/_tune_configuration-metadata.json".

    Returns:
        A tuple of the results and metadata lists respectively.
    """
    metadata: list = load_json(Path(filename_metadata))["metadata"]
    results: list = load_json(Path(filename_results))["results"]
    return metadata, results


def tune(
    input_file,
    application_name: str,
    device_name: str,
    group: dict,
    objective: str,
    objective_higher_is_better: bool,
    profiling: bool,
    searchspace_stats: SearchspaceStatistics,
) -> tuple[list, list, int]:
    """Tune a program using an optimization algorithm and collect the results.

    Optionally collects profiling statistics.

    Args:
        input_file: the json input file for tuning the application.
        application_name: the name of the program to tune.
        device_name: the device (GPU) to tune on.
        group: the experimental group (usually the search method).
        objective: the key to optimize for.
        objective_higher_is_better: whether to maximize or minimize the objective.
        profiling: whether profiling statistics should be collected.
        searchspace_stats: a ``SearchspaceStatistics`` object passed to convert imported runs.

    Raises:
        ValueError: if tuning fails multiple times in a row.

    Returns:
        A tuple of the metadata, the results, and the total runtime in milliseconds.
    """

    def tune_with_kerneltuner():
        """Interface with Kernel Tuner to tune the kernel and return the results."""
        from kernel_tuner import tune_kernel_T1

        samples = group["samples"]
        strategy_options = group.get("budget", {})
        if "custom_search_method_path" in group:
            # if a custom search method is specified, use it
            strategy_options["custom_search_method_path"] = group["custom_search_method_path"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metadata, results = tune_kernel_T1(
                input_file,
                objective=objective,
                objective_higher_is_better=objective_higher_is_better,
                simulation_mode=True,
                output_T4=True,
                iterations=samples,
                strategy_options=strategy_options,
            )
        if "max_fevals" in group["budget"]:
            max_fevals = group["budget"]["max_fevals"]
            num_results = len(results["results"])
            if num_results < max_fevals * 0.1:
                warnings.warn(
                    f"Much fewer configurations were returned ({num_results}) than the requested {max_fevals}"
                )
            if num_results < 2 and group["budget"]["max_fevals"] > 2:
                raise ValueError(
                    f"Less than two configurations were returned ({len(results['results'])}, budget {group['budget']}) \n"
                )
        return metadata, results

    def tune_with_BAT():
        """Interface to tune with the BAT benchmarking suite."""
        # TODO integrate with BAT
        raise NotImplementedError("This will be implemented in the future.")

    def tune_with_KTT():
        """Interface with KTT to tune the kernel and return the results."""
        raise NotImplementedError(
            "KTT is working on supporting the shared interface. The old conversions have been deprecated. An older build can be used to use these functions."
        )

    if group["autotuner"] == "KTT":
        metadata, results, total_time_ms = tune_with_KTT()
    elif group["autotuner"] == "KernelTuner":
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
    else:
        raise ValueError(f"Invalid autotuning framework '{group['autotuner']}'")

    # convert time units
    timeunit: str = results.get("metadata", {}).get("timeunit", "seconds")
    for result in results["results"]:
        for k, v in result["times"].items():
            result["times"][k] = convert_from_time_unit(v, timeunit)
        # performance should not be auto-converted
        # for i, m in enumerate(result["measurements"]):
        #     if "unit" in m and not isinstance(m["value"], str):
        #         result["measurements"][i]["value"] = convert_from_time_unit(m["value"], m["unit"])

    # be careful not to rely on total_time_ms when profiling, because it will include profiling time
    validate_T4(results)
    return metadata, results, total_time_ms


def collect_results(
    input_file,
    group: dict,
    results_description: ResultsDescription,
    searchspace_stats: SearchspaceStatistics,
    profiling_filename: str | None = None,
    compress: bool = True,
) -> ResultsDescription:
    """Executes optimization algorithms on tuning problems to capture their behaviour.

    Args:
        input_file: an input json file to tune.
        group: a dictionary with settings for experimental group.
        results_description: the ``ResultsDescription`` object to write the results to.
        searchspace_stats: the ``SearchspaceStatistics`` object, used for conversion of imported runs.
        profiling_filename: optional filename for profiling output. Defaults to None (no profiling).
        compress: whether the results should be compressed.

    Returns:
        The ``ResultsDescription`` object with the results.
    """
    if profiling_filename is not None:
        import psutil
        from os import getpid
        process = psutil.Process(getpid())
        warnings.warn(f"Profiling enabled, not writing results to file.")
        warnings.warn(f"Memory usage at start of collect_results: {process.memory_info().rss / 1e6:.1f} MB")
        yappi.set_clock_type("cpu")  # use CPU time for profiling, alternatively use wall time with "wall"
        yappi.start()

    # calculate the minimum number of function evaluations that must be valid
    minimum_fraction_of_budget_valid = group.get("minimum_fraction_of_budget_valid", None)
    if minimum_fraction_of_budget_valid is not None:
        assert isinstance(minimum_fraction_of_budget_valid, float)
        assert 0.0 < minimum_fraction_of_budget_valid <= 1.0
        max_fevals = None
        budget = group["budget"]
        if "max_fevals" in budget:
            max_fevals = budget["max_fevals"]
        elif "time_limit" in budget:
            time_limit = budget["time_limit"]
            time_per_feval = searchspace_stats.get_time_per_feval("mean_per_feval")
            max_fevals = max(round(time_limit / time_per_feval), 2)
        else:
            raise ValueError(f"Unkown budget {budget}, can not calculate minimum fraction of budget valid")
        min_num_evals = max(round(minimum_fraction_of_budget_valid * min(max_fevals, searchspace_stats.size)), 2)
        if "minimum_number_of_valid_search_iterations" in group:
            min_num_evals = max(min(min_num_evals, group["minimum_number_of_valid_search_iterations"]), 2)
            warnings.warn(
                f"Both 'minimum_number_of_valid_search_iterations' ({group['minimum_number_of_valid_search_iterations']}) and 'minimum_fraction_of_budget_valid' ({minimum_fraction_of_budget_valid}/{max_fevals=}) are set, the minimum ({min_num_evals}) is used."
            )
    else:
        min_num_evals: int = group["minimum_number_of_valid_search_iterations"]

    if len(results_description.objective_performance_keys) != 1:
        raise NotImplementedError(
            f"Multi objective tuning is not yet supported ({results_description.objective_performance_keys})"
        )
    objective = results_description.objective_performance_keys[0]
    objective_higher_is_better = not results_description.minimization

    def report_multiple_attempts(rep: int, len_res: int, group_repeats: int, attempt: int):
        """If multiple attempts are necessary, report the reason."""
        if len_res < 1:
            print(f"({rep + 1}/{group_repeats}) No results found, trying once more...")
        elif len_res < min_num_evals:
            print(
                f"Too few results found ({len_res} of {min_num_evals} required, attempt {attempt}), trying once more..."
            )
        else:
            print(f"({rep + 1}/{group_repeats}) Only invalid results found, trying once more...")

    def cumulative_time_taken(results: list) -> list:
        """Calculates the cumulative time taken for each of the configurations in results."""
        config_times = []
        cumulative_time_taken = 0
        for config in results:
            config_sum = 0
            for key in config["times"]:
                if key in searchspace_stats.objective_time_keys:
                    time = config["times"][key]
                    if isinstance(time, (list, tuple)):
                        time = sum(time)
                    config_sum += time
            cumulative_time_taken += config_sum
            config_times.append(cumulative_time_taken)
        return config_times

    # repeat the run as specified
    repeated_results = []
    total_time_results = np.array([])
    for rep in progressbar.progressbar(
        range(group["repeats"]),
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
                report_multiple_attempts(rep, len_res, group["repeats"], attempt)
            if attempt >= 20:
                raise RuntimeError(
                    f"Could not find enough results for {results_description.application_name} on {results_description.device_name} in {attempt} attempts ({'only invalid, ' if only_invalid else ''}{len_res}/{min_num_evals}), quiting..."
                )
            _, results, total_time_ms = tune(
                input_file,
                results_description.application_name,
                results_description.device_name,
                group,
                objective,
                objective_higher_is_better,
                profiling_filename is not None,
                searchspace_stats,
            )
            results = results["results"]

            # check without results that are beyond the cutoff times
            time_taken = cumulative_time_taken(results)
            cutoff_time = group["cutoff_times"]["cutoff_time"]
            cutoff_time_start = group["cutoff_times"]["cutoff_time_start"]
            temp_results = [res for res, time in zip(results, time_taken) if cutoff_time_start <= time <= cutoff_time]
            # if len(temp_results) < len(results):
            #     print(
            #         f"Dropped {len(results) - len(temp_results)} configurations beyond cutoff time {round(cutoff_time, 3)}, {len(temp_results)} left"
            #     )

            # check if there are only invalid configs in the first min_num_evals, if so, try again
            len_res = len(temp_results)
            temp_res_filtered = list(filter(lambda config: is_valid_config_result(config), temp_results))
            only_invalid = len(temp_res_filtered) < 2  # there must be at least two valid configurations
            attempt += 1

        # compress the results if necessary
        if compress:
            results = gzip.compress(pickle.dumps(results))

        # register the results
        repeated_results.append(results)
        total_time_results = np.append(total_time_results, total_time_ms)

        # report the memory usage
        if profiling_filename is not None:
            warnings.warn(f"Memory usage after iteration {rep}: {process.memory_info().rss / 1e6:.1f} MB")

    # gather profiling data and clear the profiler before the next round
    if profiling_filename is not None:
        stats = yappi.get_func_stats()
        # stats.print_all()
        path = results_description.run_folder / (profiling_filename + f"_{results_description.group_name}_{results_description.application_name}_{results_description.device_name}.prof")
        stats.save(path, type="pstat")  # pylint: disable=no-member
        yappi.clear_stats()
        warnings.warn(f"Memory usage before writing in collect_results: {process.memory_info().rss / 1e6:.1f} MB")

    # combine the results to numpy arrays and write to a file
    if profiling_filename is not None:
        warnings.warn(f"Skipping writing results to file because profiling is enabled, quitting. Profiling file at: {path}.")
        warnings.warn(f"Memory usage at end of of collect_results: {process.memory_info().rss / 1e6:.1f} MB")
        return None
    else:
        write_results(repeated_results, results_description, compressed=compress)
    assert results_description.has_results(), "No results in ResultsDescription after writing results."
    return results_description


def write_results(repeated_results: list, results_description: ResultsDescription, compressed=False):
    """Combine the results and write them to a NumPy file.

    Args:
        repeated_results: a list of tuning results, one per tuning session.
        results_description: the ``ResultsDescription`` object to write the results to.
        compressed: whether the repeated_results are compressed.
    """
    # get the objective value and time keys
    objective_time_keys = results_description.objective_time_keys
    objective_performance_keys = results_description.objective_performance_keys

    # find the maximum (reasonable) number of function evaluations
    num_evals = []
    for repeat in repeated_results:
        if compressed:
            repeat = pickle.loads(gzip.decompress(repeat))
        num_evals.append(len(repeat))
    max_num_evals = max(num_evals) if num_evals else 0
    mean_num_evals = np.mean(num_evals) if num_evals else 0
    if max_num_evals > mean_num_evals * 2:
        # the maximum number of evaluations is more than twice the mean, this is likely an outlier, cut to save memory
        max_num_evals = int(mean_num_evals * 2)
    if max_num_evals > 1e8:
        # more than 100 million evaluations, set to the mean number of evaluations
        max_num_evals = int(mean_num_evals)

    # set the dtype
    dtype = np.float64
    if max_num_evals * len(repeated_results) > 1e9:
        warnings.warn(
            f"More than 1 billion entries ({max_num_evals * len(repeated_results)}) in the results, using float16 to save memory."
        )
        dtype = np.float16
    elif max_num_evals * len(repeated_results) > 1e8:
        warnings.warn(
            f"More than 100 million entries ({max_num_evals * len(repeated_results)}) in the results, using float32 to save memory."
        )
        dtype = np.float32
    estimated_memory_usage = max_num_evals * len(repeated_results) * (
        8 if dtype == np.float64 else 2 if dtype == np.float16 else 4
    )  # 8 bytes for float64, 4 bytes for float32, 2 bytes for float16
    if estimated_memory_usage > 1e9*10:  # more than 10 GB
        warnings.warn(
            f"Estimated memory usage of {estimated_memory_usage / 1e9:.2f} GB for the results arrays, may go out of memory."
        )

    def get_nan_array() -> np.ndarray:
        """Get an array of NaN so they are not counted as zeros inadvertedly."""
        # return np.full((max_num_evals, len(repeated_results)), np.nan, dtype=dtype)
        arr = np.empty((max_num_evals, len(repeated_results)), dtype=dtype)
        arr.fill(np.nan)
        return arr

    # set the arrays to write to
    fevals_results = get_nan_array()
    objective_time_results = get_nan_array()
    objective_performance_results = get_nan_array()
    objective_performance_best_results = get_nan_array()
    objective_performance_stds = get_nan_array()
    objective_time_results_per_key = np.full((len(objective_time_keys), max_num_evals, len(repeated_results)), np.nan, dtype=dtype)
    objective_performance_results_per_key = np.full(
        (len(objective_time_keys), max_num_evals, len(repeated_results)), np.nan, dtype=dtype
    )

    # combine the results
    opt_func = np.nanmin if results_description.minimization is True else np.nanmax
    for repeat_index, repeat in enumerate(repeated_results):
        if compressed:
            repeat = pickle.loads(gzip.decompress(repeat))
        cumulative_objective_time = 0
        objective_performance_best = np.nan
        for evaluation_index, evaluation in enumerate(repeat):
            if evaluation_index >= max_num_evals:
                break

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
            objective_times_list = []
            for key_index, key in enumerate(objective_time_keys):
                evaluation_times = evaluation["times"]
                assert key in evaluation_times, (
                    f"Objective time key {key} not in evaluation['times'] ({evaluation_times})"
                )
                if isinstance(evaluation_times[key], list):
                    # this happens when runtimes are in objective_time_keys
                    value = sum(evaluation_times[key])
                else:
                    value = evaluation_times[key]
                if value is not None and not is_invalid_objective_time(value):
                    # value = value / 1000  # TODO this milliseconds to seconds conversion is specific to Kernel Tuner
                    objective_time_results_per_key[key_index, evaluation_index, repeat_index] = value
                    objective_times_list.append(value)
            # sum the objective times of the keys
            if len(objective_times_list) >= 1:
                objective_time = sum(objective_times_list)
                if not is_invalid_objective_time(objective_time):
                    cumulative_objective_time += objective_time
                    objective_time_results[evaluation_index, repeat_index] = cumulative_objective_time

            # obtain the objective performance per key (called 'measurements' in the T4 format)
            objective_performances_list = []
            for key_index, key in enumerate(objective_performance_keys):
                evaluation_measurements = evaluation["measurements"]
                measurements = list(filter(lambda m: m["name"] == key, evaluation_measurements))
                assert len(measurements) > 0, (
                    f"Objective performance key name {key} not in evaluation['measurements'] ({evaluation_measurements})"
                )
                assert len(measurements) == 1, f"""Objective performance key name {key} multiply defined
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
