"""Interface to run an experiment on the auto-tuning frameworks."""

from __future__ import annotations  # for correct nested type hints e.g. list[str], tuple[dict, str]

import contextlib
import json
import os
import time as python_time
import warnings
from pathlib import Path

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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metadata, results = tune_kernel_T1(
                input_file,
                objective=objective,
                objective_higher_is_better=objective_higher_is_better,
                simulation_mode=True,
                output_T4=True,
                iterations=samples,
                strategy_options=group["budget"],
            )
        if "max_fevals" in group["budget"]:
            max_fevals = group["budget"]["max_fevals"]
            num_results = len(results["results"])
            if num_results < max_fevals * 0.1:
                warnings.warn(
                    f"Much fewer configurations were returned ({num_results}) than the requested {max_fevals}"
                )
            if num_results < 2:
                raise ValueError("Less than two configurations were returned")
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
    profiling: bool,
) -> ResultsDescription:
    """Executes optimization algorithms on tuning problems to capture their behaviour.

    Args:
        input_file: an input json file to tune.
        group: a dictionary with settings for experimental group.
        results_description: the ``ResultsDescription`` object to write the results to.
        searchspace_stats: the ``SearchspaceStatistics`` object, only used for conversion of imported runs.
        profiling: whether profiling statistics must be collected.

    Returns:
        The ``ResultsDescription`` object with the results.
    """
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
            print(f"({rep+1}/{group_repeats}) No results found, trying once more...")
        elif len_res < min_num_evals:
            print(
                f"Too few results found ({len_res} of {min_num_evals} required, attempt {attempt}), trying once more..."
            )
        else:
            print(f"({rep+1}/{group_repeats}) Only invalid results found, trying once more...")

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
            _, results, total_time_ms = tune(
                input_file,
                results_description.application_name,
                results_description.device_name,
                group,
                objective,
                objective_higher_is_better,
                profiling,
                searchspace_stats,
            )
            results = results["results"]
            if attempt >= 10:
                raise RuntimeError(f"Could not find enough results in {attempt} attempts, quiting...")
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
        path = results_description.run_folder + "/profile-v2.prof"
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
            objective_times_list = []
            for key_index, key in enumerate(objective_time_keys):
                evaluation_times = evaluation["times"]
                assert (
                    key in evaluation_times
                ), f"Objective time key {key} not in evaluation['times'] ({evaluation_times})"
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
