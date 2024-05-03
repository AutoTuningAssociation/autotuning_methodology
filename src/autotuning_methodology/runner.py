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
from autotuning_methodology.searchspace_statistics import SearchspaceStatistics
from autotuning_methodology.validators import (
    is_invalid_objective_performance,
    is_invalid_objective_time,
    is_valid_config_result,
)

folder = Path(__file__).parent.parent.parent
import_runs_path = Path(folder, "cached_data_used/import_runs")


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


def load_json(path: Path):
    """Helper function to load a JSON file."""
    assert path.exists(), f"File {path.name} does not exist relative to {os.getcwd()}"
    with path.open() as file_results:
        return json.load(file_results)


def get_results_and_metadata(
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
    run_number: int,
    kernel,
    kernel_name: str,
    device_name: str,
    strategy: dict,
    tune_options: dict,
    profiling: bool,
    searchspace_stats: SearchspaceStatistics,
) -> tuple[list, list, int]:
    """Tune a program using an optimization algorithm and collect the results.

    Optionally collects profiling statistics.

    Args:
        run_number: the run number (only relevant when importing).
        kernel: the program (kernel) to tune.
        kernel_name: the name of the program to tune.
        device_name: the device (GPU) to tune on.
        strategy: the optimization algorithm to optimize with.
        tune_options: a special options dictionary passed along to the autotuning framework.
        profiling: whether profiling statistics should be collected.
        searchspace_stats: a ``SearchspaceStatistics`` object passed to convert imported runs.

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
            # check that the number of iterations is correct
            if "iterations" in strategy:
                for result in results:
                    if "runtime" in result:
                        num_iters = len(results[0]["runtimes"])
                        assert (
                            strategy["iterations"] == num_iters
                        ), f"Specified {strategy['iterations']=} not equal to actual number of iterations ({num_iters})"
                        break
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

    def import_from_KTT(use_param_mapping=True, use_bruteforce_objective=True):
        """Import a KTT output file."""
        # import the file
        assert import_runs_path.exists() and import_runs_path.is_dir()
        expected_filename = (
            f"t~'ktt'd~'{device_name}'k~'{kernel_name}'s~'{strategy['strategy']}'r~{run_number}.json".lower()
        )
        matching_runs: list[dict] = list()
        for file in import_runs_path.iterdir():
            if file.name == expected_filename:
                matching_runs.append(load_json(file))
        if len(matching_runs) < 1:
            raise FileNotFoundError(f"No files to import found with name '{expected_filename}'")
        if len(matching_runs) > 1:
            raise FileExistsError(
                f"{len(matching_runs)} files exist with name '{expected_filename}', there can be only one"
            )
        run = matching_runs[0]

        # map all timeunits to miliseconds
        ktt_timeunit_mapping = {
            "seconds": lambda x: x * 1000,
            "miliseconds": lambda x: x,
            "microseconds": lambda x: x / 1000,
        }
        ktt_status_mapping = {
            "ok": "correct",
            "devicelimitsexceeded": "compile",
            "computationfailed": "runtime",
        }

        # convert to the T4 format
        metadata = None  # TODO implement the metadata conversion when necessary
        results = list()
        run_metadata: dict = run["Metadata"]
        run_results: list[dict] = run["Results"]
        timemapper = ktt_timeunit_mapping[str(run_metadata["TimeUnit"]).lower()]
        total_time_ms = 0
        for config_attempt in run_results:

            # convert the configuration to T4 style dictionary for fast lookups in the mapping
            configuration_ktt = dict()
            for param in config_attempt["Configuration"]:
                configuration_ktt[param["Name"]] = param["Value"]

            # convert the configuration data with the mapping in the correct order
            configuration = dict()
            if use_param_mapping and kernel_name in ktt_param_mapping:
                param_map = ktt_param_mapping[kernel_name]
                assert len(param_map) == len(
                    configuration_ktt
                ), f"Mapping provided for {len(param_map)} params, but configuration has {len(configuration_ktt)}"
                for param_name, mapping in param_map.items():
                    param_value = configuration_ktt[param_name]
                    # if the mapping is None, do not include the parameter
                    if mapping is None:
                        pass
                    # if the mapping is a tuple, the first argument is the new parameter name and the second the value
                    elif isinstance(mapping, tuple):
                        param_mapped_name, param_mapped_value = mapping
                        if callable(param_mapped_value):
                            param_mapped_value = param_mapped_value(param_value)
                        configuration[param_mapped_name] = param_mapped_value
                    # if it's a list of tuples, map to multiple parameters
                    elif isinstance(mapping, list):
                        for param_mapped_name, param_mapped_value in mapping:
                            if callable(param_mapped_value):
                                param_mapped_value = param_mapped_value(param_value)
                            configuration[param_mapped_name] = param_mapped_value
                    else:
                        raise ValueError(f"Can not apply parameter mapping of {type(mapping)} ({mapping})")
            else:
                configuration = configuration_ktt

            # add to total time
            total_duration = timemapper(config_attempt["TotalDuration"])
            total_overhead = timemapper(config_attempt["TotalOverhead"])
            total_time_ms += total_duration + total_overhead

            # convert the times data
            times_runtimes = []
            duration = ""
            if len(config_attempt["ComputationResults"]) > 0:
                for config_result in config_attempt["ComputationResults"]:
                    times_runtimes.append(timemapper(config_result["Duration"]))
                if use_bruteforce_objective:
                    config_string_key = ",".join(str(x) for x in configuration.values())
                    duration = searchspace_stats.get_value_in_config(config_string_key, "time")
                else:
                    duration = np.mean(times_runtimes)
                assert (
                    "iterations" in strategy
                ), "For imported KTT runs, the number of iterations must be specified in the experiments file"
                if strategy["iterations"] != len(times_runtimes):
                    times_runtimes = [np.mean(times_runtimes)] * strategy["iterations"]
                    warnings.warn(
                        f"The specified number of iterations ({strategy['iterations']}) did not equal"
                        + f"the actual number of iterations ({len(times_runtimes)}). "
                        + "The average has been used."
                    )
            if (not isinstance(duration, (float, int, np.number))) or np.isnan(duration):
                duration = ""
            times_search_algorithm = timemapper(config_attempt.get("SearcherOverhead", 0))
            times_validation = timemapper(config_attempt.get("ValidationOverhead", 0))
            times_framework = timemapper(config_attempt.get("DataMovementOverhead", 0))
            times_benchmark = total_duration
            times_compilation = total_overhead - times_search_algorithm - times_validation - times_framework

            # assemble the converted data
            converted = {
                "configuration": configuration,
                "invalidity": ktt_status_mapping[str(config_attempt["Status"]).lower()],
                "correctness": 1,
                "measurements": [
                    {
                        "name": "time",
                        "value": duration,
                        "unit": "ms",
                    }
                ],
                "objectives": ["time"],
                "times": {
                    "compilation": times_compilation,
                    "benchmark": times_benchmark,
                    "framework": times_framework,
                    "search_algorithm": times_search_algorithm,
                    "validation": times_validation,
                    "runtimes": times_runtimes,
                },
            }
            results.append(converted)

        return metadata, results, round(total_time_ms)

    strategy_name = str(strategy["name"]).lower()
    if strategy_name.startswith("ktt_"):
        metadata, results, total_time_ms = import_from_KTT()
    elif strategy_name.startswith("kerneltuner_") or True:
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
        raise ValueError(f"Invalid autotuning framework '{strategy_name}'")
    # be careful not to rely on total_time_ms when profiling, because it will include profiling time
    return metadata, results, total_time_ms


def collect_results(
    kernel,
    strategy: dict,
    results_description: ResultsDescription,
    searchspace_stats: SearchspaceStatistics,
    profiling: bool,
) -> ResultsDescription:
    """Executes optimization algorithms on tuning problems to capture their behaviour.

    Args:
        kernel: the program (kernel) to tune.
        strategy: the optimization algorithm to optimize with.
        searchspace_stats: the ``SearchspaceStatistics`` object, only used for conversion of imported runs.
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
                rep,
                kernel,
                results_description.kernel_name,
                results_description.device_name,
                strategy,
                tune_options,
                profiling,
                searchspace_stats,
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
