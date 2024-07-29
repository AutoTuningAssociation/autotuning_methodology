"""Report results of the experiment without requiring visualization."""

from pathlib import Path

import numpy as np

from autotuning_methodology.baseline import (
    Baseline,
    ExecutedStrategyBaseline,
    RandomSearchCalculatedBaseline,
)
from autotuning_methodology.curves import Curve, StochasticOptimizationAlgorithm
from autotuning_methodology.searchspace_statistics import SearchspaceStatistics


def get_aggregation_data_key(gpu_name: str, kernel_name: str):
    """Utility function to get the key for data in the aggregation data dictionary.

    Args:
        gpu_name: the GPU name
        kernel_name: the kernel name

    Returns:
        The key as a string.
    """
    return f"{gpu_name}+{kernel_name}"


def get_aggregation_data(
    experiment_folderpath: Path,
    experiment: dict,
    strategies: dict,
    results_descriptions: dict,
    cutoff_percentile: float,
    cutoff_percentile_start=0.01,
    confidence_level=0.95,
    minimization: bool = True,
    time_resolution: int = 1e4,
    use_strategy_as_baseline=None,
):
    """Function to collect the aggregation data after the experiments have ran.

    Args:
        experiment_folderpath: _description_
        experiment: _description_
        strategies: _description_
        results_descriptions: _description_
        cutoff_percentile: _description_
        minimization: _description_. Defaults to True.
        cutoff_percentile_start: _description_. Defaults to 0.01.
        confidence_level: _description_. Defaults to 0.95.
        time_resolution: _description_. Defaults to 1e4.
        use_strategy_as_baseline: _description_. Defaults to None.

    Returns:
        The aggregation data in a dictionary, with `get_aggregation_data_key` as key and a tuple as value.
    """
    if int(time_resolution) != time_resolution:
        raise ValueError(f"The resolution must be an integer, yet is {time_resolution}.")
    time_resolution = int(time_resolution)

    aggregation_data: dict[str, tuple[Baseline, list[Curve], SearchspaceStatistics, np.ndarray]] = dict()
    for gpu_name in experiment["GPUs"]:
        for kernel_name in experiment["kernels"]:
            # get the statistics
            searchspace_stats = SearchspaceStatistics(
                kernel_name=kernel_name,
                device_name=gpu_name,
                minimization=minimization,
                objective_time_keys=experiment["objective_time_keys"],
                objective_performance_keys=experiment["objective_performance_keys"],
                bruteforced_caches_path=experiment_folderpath / experiment["bruteforced_caches_path"],
            )

            # get the cached strategy results as curves
            strategies_curves: list[Curve] = list()
            baseline_executed_strategy = None
            for strategy in strategies:
                results_description = results_descriptions[gpu_name][kernel_name][strategy["name"]]
                if results_description is None:
                    raise ValueError(
                        f"""Strategy {strategy['display_name']} not in results_description,
                            make sure execute_experiment() has ran first"""
                    )
                curve = StochasticOptimizationAlgorithm(results_description)
                strategies_curves.append(curve)
                if use_strategy_as_baseline is not None and strategy["name"] == use_strategy_as_baseline:
                    baseline_executed_strategy = curve
            if use_strategy_as_baseline is not None and baseline_executed_strategy is None:
                raise ValueError(f"Could not find '{use_strategy_as_baseline}' in executed strategies")

            # set the x-axis range
            _, cutoff_point_fevals, cutoff_point_time = searchspace_stats.cutoff_point_fevals_time(cutoff_percentile)
            _, cutoff_point_fevals_start, cutoff_point_time_start = searchspace_stats.cutoff_point_fevals_time(
                cutoff_percentile_start
            )
            fevals_range = np.arange(start=cutoff_point_fevals_start, stop=cutoff_point_fevals)
            time_range = np.linspace(start=cutoff_point_time_start, stop=cutoff_point_time, num=time_resolution)

            # get the random baseline
            random_baseline = (
                RandomSearchCalculatedBaseline(searchspace_stats)
                if baseline_executed_strategy is None
                else ExecutedStrategyBaseline(
                    searchspace_stats, strategy=baseline_executed_strategy, confidence_level=confidence_level
                )
            )

            # collect aggregatable data
            aggregation_data[get_aggregation_data_key(gpu_name, kernel_name)] = tuple(
                [random_baseline, strategies_curves, searchspace_stats, time_range, fevals_range]
            )

    return aggregation_data


def get_strategies_aggregated_performance(
    aggregation_data: list[tuple[Baseline, list[Curve], SearchspaceStatistics, np.ndarray]],
    confidence_level: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
    """Combines the performances across searchspaces into a single metric.

    Args:
        aggregation_data: the aggregated data from the various searchspaces.
        confidence_level: the confidence interval used for the confidence / prediction interval.

    Returns:
        The aggregated relative performances of each strategy.
        Tuple of [performance, lower error, upper error, stopping point fraction].
    """
    # for each strategy, collect the relative performance in each search space
    strategies_performance = [list() for _ in aggregation_data[0][1]]
    strategies_performance_lower_err = [list() for _ in aggregation_data[0][1]]
    strategies_performance_upper_err = [list() for _ in aggregation_data[0][1]]
    strategies_performance_real_stopping_point_fraction = [list() for _ in range(len(aggregation_data[0][1]))]
    for random_baseline, strategies_curves, searchspace_stats, time_range, _ in aggregation_data:
        dist = searchspace_stats.objective_performances_total_sorted
        for strategy_index, strategy_curve in enumerate(strategies_curves):
            # get the real and fictional performance curves
            (
                real_stopping_point_index,
                x_axis_range_real,
                curve_real,
                curve_lower_err_real,
                curve_upper_err_real,
                x_axis_range_fictional,
                curve_fictional,
                curve_lower_err_fictional,
                curve_upper_err_fictional,
            ) = strategy_curve.get_curve_over_time(time_range, dist=dist, confidence_level=confidence_level)
            # combine the real and fictional parts to get the full curve
            combine = x_axis_range_fictional.ndim > 0
            x_axis_range = np.concatenate([x_axis_range_real, x_axis_range_fictional]) if combine else x_axis_range_real
            assert np.array_equal(time_range, x_axis_range, equal_nan=True), "time_range != x_axis_range"
            curve = np.concatenate([curve_real, curve_fictional]) if combine else curve_real
            curve_lower_err = (
                np.concatenate([curve_lower_err_real, curve_lower_err_fictional]) if combine else curve_lower_err_real
            )
            curve_upper_err = (
                np.concatenate([curve_upper_err_real, curve_upper_err_fictional]) if combine else curve_upper_err_real
            )
            # get the standardised curves and write them to the collector
            curve, curve_lower_err, curve_upper_err = random_baseline.get_standardised_curves(
                time_range, [curve, curve_lower_err, curve_upper_err], x_type="time"
            )
            strategies_performance[strategy_index].append(curve)
            strategies_performance_lower_err[strategy_index].append(curve_lower_err)
            strategies_performance_upper_err[strategy_index].append(curve_upper_err)
            strategies_performance_real_stopping_point_fraction[strategy_index].append(
                real_stopping_point_index / x_axis_range.shape[0]
            )

    # for each strategy, get the mean performance per step in time_range
    strategies_aggregated_performance: list[np.ndarray] = list()
    strategies_aggregated_lower_err: list[np.ndarray] = list()
    strategies_aggregated_upper_err: list[np.ndarray] = list()
    strategies_aggregated_real_stopping_point_fraction: list[float] = list()
    for index, value in enumerate(strategies_performance):
        strategies_aggregated_performance.append(np.mean(np.array(value), axis=0))
        strategies_aggregated_lower_err.append(np.mean(np.array(strategies_performance_lower_err[index]), axis=0))
        strategies_aggregated_upper_err.append(np.mean(np.array(strategies_performance_upper_err[index]), axis=0))
        strategies_aggregated_real_stopping_point_fraction.append(
            np.median(strategies_performance_real_stopping_point_fraction[index])
        )

    return (
        strategies_aggregated_performance,
        strategies_aggregated_lower_err,
        strategies_aggregated_upper_err,
        strategies_aggregated_real_stopping_point_fraction,
    )
