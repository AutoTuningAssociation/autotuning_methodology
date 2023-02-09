""" Visualize the results of the experiments """
import argparse
import numpy as np
from typing import Tuple, Any
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import auc

from experiments import execute_experiment
from curves import Curve, StochasticOptimizationAlgorithm
from baseline import Baseline, RandomSearchBaseline, StochasticCurveBasedBaseline
from searchspace_statistics import SearchspaceStatistics

import sys

sys.path.append("..")

# The kernel information per device and device information for visualization purposes
marker_variatons = ["v", "s", "*", "1", "2", "d", "P", "X"]


def calculate_lower_upper_error(observations: list) -> Tuple[float, float]:
    """Calculate the lower and upper error by the mean of the values below and above the median respectively"""
    observations.sort()
    middle_index = len(observations) // 2
    middle_index_upper = (middle_index + 1 if len(observations) % 2 != 0 else middle_index)
    lower_values = observations[:middle_index]
    upper_values = observations[middle_index_upper:]
    lower_error = np.mean(lower_values)
    upper_error = np.mean(upper_values)
    return lower_error, upper_error


def smoothing_filter(array: np.ndarray, window_length: int, a_min=None, a_max=None) -> np.ndarray:
    """Create a rolling average where the kernel size is the smoothing factor"""
    window_length = int(window_length)
    # import pandas as pd
    # d = pd.Series(array)
    # return d.rolling(window_length).mean()
    from scipy.signal import savgol_filter

    if window_length % 2 == 0:
        window_length += 1
    smoothed = savgol_filter(array, window_length, 3)
    if a_min is not None or a_max is not None:
        smoothed = np.clip(smoothed, a_min, a_max)
    return smoothed


class Visualize:
    """ Class for visualization of experiments """

    x_metric_displayname = dict({
        "num_evals": "Number of function evaluations used",
        "strategy_time": "Average time taken by strategy in miliseconds",
        "compile_time": "Average compile time in miliseconds",
        "execution_time": "Evaluation execution time taken in miliseconds",
        "total_time": "Average total time taken in miliseconds",
        "kerneltime": "Total kernel compilation and runtime in seconds",
        "aggregate_time": "Relative time to cutoff point",
    })

    y_metric_displayname = dict({
        "objective": "Best found objective function value",
        "objective_relative_median": "Fraction of absolute optimum relative to median",
        "objective_baseline": "Best found objective function value relative to baseline",
        "objective_baseline_max": "Improvement over random sampling",
        "aggregate_objective": "Aggregate best found objective function value relative to baseline",
        "aggregate_objective_max": "Aggregate improvement over random sampling",
        "time": "Best found kernel time in miliseconds",
        "GFLOP/s": "GFLOP/s",
    })

    def __init__(self, experiment_filename: str) -> None:
        # silently execute the experiment
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.experiment, self.strategies, self.results_descriptions = execute_experiment(experiment_filename, profiling=False)
        print("\n")
        print("Visualizing")

        # settings
        minimization: bool = self.experiment.get("minimization", True)
        cutoff_percentile: float = self.experiment.get("cutoff_percentile", 1)
        cutoff_percentile_start: float = self.experiment.get("cutoff_percentile_start", 0.01)
        cutoff_type: str = self.experiment.get('cutoff_type', "fevals")
        assert cutoff_type == 'fevals' or cutoff_type == 'time'
        time_resolution: float = self.experiment.get('resolution', 1e4)
        if int(time_resolution) != time_resolution:
            raise ValueError(f"The resolution must be an integer, yet is {time_resolution}.")
        time_resolution = int(time_resolution)

        # plot settings
        plot_settings: dict = self.experiment.get("plot")
        plot_fevals: bool = plot_settings.get("plot_fevals", True)
        plot_time: bool = plot_settings.get("plot_time", True)
        plot_aggregated: bool = plot_settings.get("plot_aggregated")
        num_cols: int = 2 if plot_fevals and plot_time else 1
        if not plot_fevals and not plot_time:
            raise ValueError(f"At least one of 'plot_fevals' and 'plot_time' must be True")

        # visualize
        aggregation_data: list[tuple[Baseline, list[Curve], SearchspaceStatistics, np.ndarray]] = list()
        for gpu_name in self.experiment["GPUs"]:
            for kernel_name in self.experiment["kernels"]:
                print(f" | visualizing optimization of {kernel_name} for {gpu_name}")

                # create the figure and plots
                fig, axs = plt.subplots(nrows=1, ncols=num_cols, figsize=(9, 6))    # if multiple subplots, pass the axis to the plot function with axs[0] etc.
                if not hasattr(axs, "__len__"):
                    axs = [axs]
                title = f"{kernel_name} on {gpu_name}"
                fig.canvas.manager.set_window_title(title)
                fig.suptitle(title)

                # get the statistics
                searchspace_stats = SearchspaceStatistics(kernel_name=kernel_name, device_name=gpu_name, minimization=minimization)

                # get the cached strategy results as curves
                strategies_curves: list[Curve] = list()
                for strategy in self.strategies:
                    results_description = self.results_descriptions[gpu_name][kernel_name][strategy["name"]]
                    if results_description is None:
                        raise ValueError(f"Strategy {strategy['display_name']} not in results_description, make sure execute_experiment() has ran first")
                    strategies_curves.append(StochasticOptimizationAlgorithm(results_description))

                # set the x-axis range
                _, cutoff_point_fevals, cutoff_point_time = searchspace_stats.cutoff_point_fevals_time(cutoff_percentile)
                _, cutoff_point_fevals_start, cutoff_point_time_start = searchspace_stats.cutoff_point_fevals_time(cutoff_percentile_start)
                fevals_range = np.arange(start=cutoff_point_fevals_start, stop=cutoff_point_fevals)
                time_range = np.linspace(start=cutoff_point_time_start, stop=cutoff_point_time, num=time_resolution)
                # baseline_time_interpolated = np.linspace(mean_feval_time, cutoff_point_time, time_resolution)
                # baseline = get_random_curve(cutoff_point_fevals, sorted_times, time_resolution)

                # get the random baseline
                random_baseline = RandomSearchBaseline(searchspace_stats)

                # collect aggregatable data
                if plot_aggregated:
                    aggregation_data.append(tuple([random_baseline, strategies_curves, searchspace_stats, time_range]))

                # visualize the results
                if plot_time:
                    self.plot_strategies_curves(axs[0], searchspace_stats, strategies_curves, time_range, plot_settings, random_baseline)
                if plot_fevals:
                    self.plot_strategies_fevals(axs[-1], searchspace_stats, strategies_curves, fevals_range, plot_settings, random_baseline)

                # finalize the figure and display it
                if plot_time or plot_fevals:
                    fig.tight_layout()
                    plt.show()

        # plot the aggregated data
        if plot_aggregated:
            fig, axs = plt.subplots(ncols=1, figsize=(9, 6))    # if multiple subplots, pass the axis to the plot function with axs[0] etc.
            if not hasattr(axs, "__len__"):
                axs = [axs]
            title = f"Aggregated Data\nkernels: {', '.join(self.experiment['kernels'])}\nGPUs: {', '.join(self.experiment['GPUs'])}"
            fig.canvas.manager.set_window_title(title)
            fig.suptitle(title)

            # finalize the figure and display it
            self.plot_aggregated_curves(axs[0], aggregation_data)
            fig.tight_layout()
            plt.show()

    def plot_strategies_fevals(self, ax: plt.Axes, searchspace_stats: SearchspaceStatistics, strategies_curves: list[Curve], fevals_range: np.ndarray,
                               plot_settings: dict, baseline_curve: Baseline = None, plot_errors=True):
        """ Plots all optimization strategies with number of function evaluations on the x-axis """
        relative_to_baseline: bool = plot_settings.get("plot_relative_to_baseline", True)
        confidence_level: float = plot_settings.get("confidence_level", 0.95)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        absolute_optimum = searchspace_stats.total_performances_absolute_optimum()
        median: float = searchspace_stats.total_performance_median()

        # plot the absolute optimum
        absolute_optimum_y_value = 1 if relative_to_baseline else absolute_optimum
        ax.axhline(absolute_optimum_y_value, c='black', ls='-.', label='Absolute optimum {}'.format(round(absolute_optimum, 3)))

        # plot baseline
        sorted_times = searchspace_stats.objective_performances_total_sorted
        if baseline_curve is not None:
            if relative_to_baseline is True:
                ax.axhline(0, label="baseline trajectory", color="black", ls="--")
            else:
                ax.plot(fevals_range, baseline_curve.get_curve_over_fevals(fevals_range), label="baseline curve", color="black", ls="--")

        # plot each strategy
        for strategy_index, strategy in enumerate(self.strategies):
            if "hide" in strategy.keys() and strategy["hide"]:
                continue

            # get the data
            color = colors[strategy_index]
            strategy_curve = strategies_curves[strategy_index]

            # obtain the curves
            curve, curve_lower_err, curve_upper_err = strategy_curve.get_curve_over_fevals(fevals_range, dist=sorted_times, confidence_level=confidence_level)
            if relative_to_baseline:
                # sanity check: see if the calculated random curve is equal to itself
                # assert np.allclose(baseline_curve.get_curve_over_fevals(fevals_range), baseline_curve.get_curve_over_fevals(fevals_range))
                curve = baseline_curve.get_standardised_curve_over_fevals(fevals_range, curve, absolute_optimum)
                curve_lower_err = baseline_curve.get_standardised_curve_over_fevals(fevals_range, curve_lower_err, absolute_optimum)
                curve_upper_err = baseline_curve.get_standardised_curve_over_fevals(fevals_range, curve_upper_err, absolute_optimum)

            # visualize
            if plot_errors:
                ax.fill_between(fevals_range, curve_lower_err, curve_upper_err, alpha=0.2, antialiased=True, color=color)
            ax.plot(fevals_range, curve, label=f"{strategy['display_name']}", color=color)

        # # plot cutoff point
        # def plot_cutoff_point(cutoff_percentile):
        #     cutoff_point_value, cutoff_point_fevals = searchspace_stats.cutoff_point(cutoff_percentile)
        #     print("")
        #     # print(f"percentage of searchspace to get to {cutoff_percentile*100}%: {round((cutoff_point_fevals/len(sorted_times))*100, 3)}%")
        #     # print(f"cutoff_point_fevals: {cutoff_point_fevals}")
        #     # print(f"objective_value_at_cutoff_point: {objective_value_at_cutoff_point}")
        #     ax.plot([cutoff_point_fevals], [cutoff_percentile], marker='o', color='red', label=f"cutoff point {cutoff_percentile}")

        # if baseline_curve is not None:
        #     plot_cutoff_point(0.980)

        ax.set_xlim(tuple([fevals_range[0], fevals_range[-1]]))
        ax.set_xlabel(self.x_metric_displayname["num_evals"])
        ax.set_ylabel(self.y_metric_displayname["objective_baseline_max"] if relative_to_baseline else self.y_metric_displayname["objective"])
        ax.legend()

    def plot_strategies_curves(self, ax: plt.Axes, searchspace_stats: SearchspaceStatistics, strategies_curves: list[Curve], time_range: np.ndarray,
                               plot_settings: dict, baseline_curve: Baseline = None, plot_errors=False):
        """Plots all optimization strategy curves"""
        relative_to_baseline: bool = plot_settings.get("plot_relative_to_baseline", True)
        confidence_level: float = plot_settings.get("confidence_level", 0.95)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        absolute_optimum = searchspace_stats.total_performances_absolute_optimum()
        median: float = searchspace_stats.total_performance_median()

        # plot the absolute optimum
        absolute_optimum_y_value = 1 if relative_to_baseline else absolute_optimum
        ax.axhline(absolute_optimum_y_value, c='black', ls='-.', label='Absolute optimum {}'.format(round(absolute_optimum, 3)))

        # plot baseline
        if baseline_curve is not None:
            if relative_to_baseline is True:
                ax.axhline(0, label="baseline trajectory", color="black", ls="--")
            else:
                ax.plot(time_range, baseline_curve.get_curve_over_time(time_range), label="baseline curve", color="black", ls="--")

        # plot each strategy
        sorted_times = searchspace_stats.objective_performances_total_sorted
        for strategy_index, strategy in enumerate(self.strategies):
            if "hide" in strategy.keys() and strategy["hide"]:
                continue

            # get the data
            color = colors[strategy_index]
            strategy_curve = strategies_curves[strategy_index]

            # obtain the curves
            curve, curve_lower_err, curve_upper_err = strategy_curve.get_curve_over_time(time_range, dist=sorted_times, confidence_level=confidence_level)
            if relative_to_baseline:
                # sanity check: see if the calculated random curve is equal to itself
                # assert np.allclose(baseline_curve.get_curve_over_fevals(fevals_range), baseline_curve.get_curve_over_fevals(fevals_range))
                curve = baseline_curve.get_standardised_curve_over_time(time_range, curve)
                curve_lower_err = baseline_curve.get_standardised_curve_over_time(time_range, curve_lower_err)
                curve_upper_err = baseline_curve.get_standardised_curve_over_time(time_range, curve_upper_err)

            # visualize
            if plot_errors:
                ax.fill_between(time_range, curve_lower_err, curve_upper_err, alpha=0.2, antialiased=True, color=color)
            ax.plot(time_range, curve, label=f"{strategy['display_name']}", color=color)

        # # plot cutoff point
        # def plot_cutoff_point(cutoff_percentile):
        #     cutoff_point_value, cutoff_point_fevals = searchspace_stats.cutoff_point(cutoff_percentile)
        #     print("")
        #     # print(f"percentage of searchspace to get to {cutoff_percentile*100}%: {round((cutoff_point_fevals/len(sorted_times))*100, 3)}%")
        #     # print(f"cutoff_point_fevals: {cutoff_point_fevals}")
        #     # print(f"objective_value_at_cutoff_point: {objective_value_at_cutoff_point}")
        #     ax.plot([cutoff_point_fevals], [cutoff_percentile], marker='o', color='red', label=f"cutoff point {cutoff_percentile}")

        # if baseline_curve is not None:
        #     plot_cutoff_point(0.980)

        ax.set_xlim(tuple([time_range[0], time_range[-1]]))
        ax.set_xlabel(self.x_metric_displayname["total_time"])
        ax.set_ylabel(self.y_metric_displayname["objective_baseline_max"] if relative_to_baseline else self.y_metric_displayname["objective"])
        ax.legend()

    def plot_aggregated_curves(self, ax: plt.Axes, aggregation_data: list[tuple[Baseline, list[Curve], SearchspaceStatistics, np.ndarray]]):
        # plot the random baseline and absolute optimum
        ax.axhline(0, label="Random search", c='black', ls=':')
        ax.axhline(1, label="Absolute optimum", c='black', ls='-.')

        # get the relative performance for each strategy
        strategies_performance = [list() for _ in aggregation_data[0][1]]
        for random_baseline, strategies_curves, searchspace_stats, time_range in aggregation_data:
            for strategy_index, strategy_curve in enumerate(strategies_curves):
                curve, _, _ = strategy_curve.get_curve_over_time(time_range)
                relative_performance = random_baseline.get_standardised_curve_over_time(time_range, curve)
                strategies_performance[strategy_index].append(relative_performance)

        # plot each strategy
        for strategy_index, strategy_performances in enumerate(strategies_performance):
            strategy_performances = np.array([p for p in strategy_performances])
            strategy_performance: np.ndarray = np.mean(strategy_performances, axis=0)
            ax.plot(strategy_performance, label=self.strategies[strategy_index]["display_name"])

        # set the axis
        y_axis_size = strategy_performance.size
        cutoff_percentile: float = self.experiment.get("cutoff_percentile", 1)
        cutoff_percentile_start: float = self.experiment.get("cutoff_percentile_start", 0.01)
        ax.set_xlabel(f"{self.x_metric_displayname['aggregate_time']} ({cutoff_percentile_start*100}% to {cutoff_percentile*100}%)")
        ax.set_ylabel(self.y_metric_displayname["aggregate_objective"])
        num_ticks = 11
        ax.set_xticks(
            np.linspace(0, y_axis_size, num_ticks),
            np.round(np.linspace(0, 1, num_ticks), 2),
        )
        ax.set_ylim(top=1.0)
        ax.set_xlim((0, y_axis_size))
        ax.legend()


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "experiment",
        type=str,
        help="The experiment.json to execute, see experiments/template.json",
    )
    args = CLI.parse_args()
    filepath = args.experiment
    if filepath is None:
        raise ValueError("Invalid '-experiment' option. Run 'visualize_experiments.py -h' to read more about the options.")

    Visualize(filepath)
