""" Visualize the results of the experiments """
import argparse
import numpy as np
from typing import Tuple, Any
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import auc
from math import ceil

from experiments import execute_experiment, get_searchspaces_info_stats, calc_cutoff_point_fevals_time
from curves import Curve, StochasticOptimizationAlgorithm
from baseline import Baseline, RandomSearchBaseline

import sys

sys.path.append("..")
# TODO from cached_data_used.kernel_info_generator import searchspaces_info_stats # check whether this is necessary

searchspaces_info_stats = get_searchspaces_info_stats()

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
            self.experiment, self.strategies, self.results_descriptions = execute_experiment(
                experiment_filename,
                profiling=False,
                searchspaces_info_stats=searchspaces_info_stats,
            )
        print("\n")
        print("Visualizing")

        # settings
        minimization: bool = self.experiment.get("minimization", True)
        cutoff_percentile: float = self.experiment.get("cutoff_percentile", 1)
        cutoff_percentile_start: float = self.experiment.get("cutoff_percentile_start", 0)
        cutoff_type: str = self.experiment.get('cutoff_type', "fevals")
        assert cutoff_type == 'fevals' or cutoff_type == 'time'
        time_resolution: float = self.experiment.get('resolution', 1e4)
        if int(time_resolution) != time_resolution:
            raise ValueError(f"The resolution must be an integer, yet is {time_resolution}.")
        time_resolution = int(time_resolution)

        # plot settings
        plot_settings: dict = self.experiment.get("plot")
        plot_relative_to_baseline: bool = plot_settings.get("plot_relative_to_baseline", True)
        plot_fevals: bool = plot_settings.get("plot_fevals", True)
        plot_time: bool = plot_settings.get("plot_time", True)
        plot_aggregated: bool = plot_settings.get("plot_aggregated")

        # visualize
        all_strategies_curves = list()
        for gpu_name in self.experiment["GPUs"]:
            for kernel_name in self.experiment["kernels"]:
                print(f" | visualizing optimization of {kernel_name} for {gpu_name}")

                # create the figure and plots
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 6))    # if multiple subplots, pass the axis to the plot function with axs[0] etc.
                if not hasattr(axs, "__len__"):
                    axs = [axs]
                title = f"{kernel_name} on {gpu_name}"
                fig.canvas.manager.set_window_title(title)
                fig.suptitle(title)

                # get the cached strategy results as curves
                strategies_curves: list[Curve] = list()
                for strategy in self.strategies:
                    results_description = self.results_descriptions[gpu_name][kernel_name][strategy["name"]]
                    if results_description is None:
                        raise ValueError(f"Strategy {strategy['display_name']} not in results_description, make sure execute_experiment() has ran first")
                    strategies_curves.append(StochasticOptimizationAlgorithm(results_description))

                # set the x-axis range
                info: dict = searchspaces_info_stats[gpu_name]["kernels"][kernel_name]
                _, cutoff_point_fevals, cutoff_point_time = calc_cutoff_point_fevals_time(cutoff_percentile, info)
                _, cutoff_point_fevals_start, cutoff_point_time_start = calc_cutoff_point_fevals_time(cutoff_percentile_start, info)
                fevals_range = np.arange(start=cutoff_point_fevals_start, stop=cutoff_point_fevals)
                time_range = np.linspace(start=cutoff_point_time_start, stop=cutoff_point_time, num=time_resolution)
                # baseline_time_interpolated = np.linspace(mean_feval_time, cutoff_point_time, time_resolution)
                # baseline = get_random_curve(cutoff_point_fevals, sorted_times, time_resolution)

                # get the random baseline
                sorted_times = np.sort(info['sorted_times'])
                random_baseline = RandomSearchBaseline(minimization, sorted_times) if plot_relative_to_baseline else None

                # visualize the results
                if plot_time:
                    self.plot_strategies_curves(axs[0], info, strategies_curves, time_range, random_baseline, plot_relative_to_baseline)
                if plot_fevals:
                    self.plot_strategies_fevals(axs[1], info, strategies_curves, fevals_range, random_baseline, plot_relative_to_baseline)
                all_strategies_curves.append(strategies_curves)

                # finalize the figure and display it
                if plot_time or plot_fevals:
                    fig.tight_layout()
                    plt.show()

        # # plot the aggregated data
        # if plot_aggregated:
        #     fig, axs = plt.subplots(ncols=1, figsize=(15, 8))    # if multiple subplots, pass the axis to the plot function with axs[0] etc.
        #     if not hasattr(axs, "__len__"):
        #         axs = [axs]
        #     title = f"Aggregated Data\nkernels: {', '.join(self.experiment['kernels'])}\nGPUs: {', '.join(self.experiment['GPUs'])}"
        #     fig.canvas.manager.set_window_title(title)
        #     fig.suptitle(title)

        # # gather the aggregate y axis for each strategy
        # print("\n")
        # strategies_aggregated = list()
        # for strategy_index, strategy in enumerate(self.strategies):
        #     perf = list()
        #     y_axis_temp = list()
        #     for strategies_curves in all_strategies_curves:
        #         for strategy_curve in strategies_curves["strategies"]:
        #             if strategy_curve["strategy_index"] == strategy_index:
        #                 perf.append(strategy_curve["performance"])
        #                 y_axis_temp.append(strategy_curve["y_axis"])
        #     print(f"{strategy['display_name']} performance across kernels: {np.mean(perf)}")
        #     y_axis = np.array(y_axis_temp)
        #     strategies_aggregated.append(np.mean(y_axis, axis=0))

        # # finalize the figure and display it
        # self.plot_aggregated_curves(axs[0], strategies_aggregated)
        # fig.tight_layout()
        # plt.show()

    def get_strategies_curves(
        self,
        cache,
        strategies_data: list,
        info: dict,
        subtract_baseline=True,
        smoothing=False,
        minimization=True,
        smoothing_factor=100,
    ) -> dict:
        """Extract the strategies results"""
        # get the baseline
        x_axis, y_axis_baseline = cache.get_baseline()
        absolute_optimum = info["absolute_optimum"]
        y_max = absolute_optimum

        # y_min = absolute_optimum if minimization else info["median"]
        # y_max = info["median"] if minimization else absolute_optimum

        # normalize
        if subtract_baseline:
            # y_axis_baseline = (y_axis_baseline - y_min) / (y_max - y_min)
            y_min = y_axis_baseline
            # y_axis_baseline = (y_axis_baseline - y_axis_baseline) / (y_max - y_min)

        if smoothing:
            y_axis_baseline = smoothing_filter(y_axis_baseline, y_axis_baseline.size / smoothing_factor)

        # create resulting dict
        strategies_curves: dict[str, Any] = dict({
            "baseline": {
                "x_axis": x_axis,
                "y_axis": y_axis_baseline
            },
            "strategies": list(),
        })

        # clipstart = 10
        # random_curve = y_axis_baseline[clipstart:]
        # sorted_times = info["sorted_times"]

        performances = list()
        for strategy_index, strategy in enumerate(self.strategies):
            if "hide" in strategy.keys() and strategy["hide"]:
                continue

            # get the data
            strategy = strategies_data[strategy_index]
            results = strategy["results"]
            y_axis = np.array(results["interpolated_objective"])
            y_axis_std = np.array(results["interpolated_objective_std"])
            y_axis_std_lower = np.array(results["interpolated_objective_error_lower"])
            y_axis_std_upper = np.array(results["interpolated_objective_error_upper"])
            window_length = min(max(int(len(y_axis) * 0.5), 100), len(y_axis))
            y_axis_std_lower = smoothing_filter(y_axis_std_lower, window_length, a_max=y_axis)
            y_axis_std_upper = smoothing_filter(y_axis_std_upper, window_length, a_min=y_axis)

            # normalize
            if subtract_baseline:
                y_axis = (y_axis - y_axis_baseline) / (y_max - y_axis_baseline)
                # y_axis_std_lower = (y_axis_std_lower - y_min) / (y_max - y_min)
                # y_axis_std_upper = (y_axis_std_upper - y_min) / (y_max - y_min)

            # apply smoothing
            if smoothing:
                y_axis = smoothing_filter(y_axis, y_axis.size / smoothing_factor)

            # find out where the global optimum is found and substract the baseline
            # found_opt = np.argwhere(y_axis == absolute_optimum)
            # if subtract_baseline:
            # # y_axis =  y_axis - y_axis_baseline
            # y_axis = y_axis_baseline / y_axis
            # y_axis_std_lower = y_axis_baseline / y_axis_std_lower
            # y_axis_std_upper = y_axis_baseline / y_axis_std_upper
            # y_axis = (y_axis - y_min) / (y_max - y_min)

            # quantify the performance of this strategy
            if subtract_baseline:
                # use mean distance
                performance = np.mean(y_axis)
            else:
                # use area under curve approach
                performance = auc(x_axis, y_axis)
            performances.append(performance)
            print(f"Performance of {strategy['display_name']}: {performance}")

            # write to resulting dict
            result_dict = dict({
                "strategy_index": strategy_index,
                "y_axis": y_axis,
                "y_axis_std": y_axis_std,
                "y_axis_std_lower": y_axis_std_lower,
                "y_axis_std_upper": y_axis_std_upper,
                "performance": performance,
            })
            strategies_curves["strategies"].append(result_dict)

        print(f"Mean performance across strategies: {np.mean(performances)}")    # the higher the mean, the easier a search space is for the baseline
        return strategies_curves

    def plot_strategies_fevals(self, ax: plt.Axes, info: dict, strategies_curves: list[Curve], fevals_range: np.ndarray, baseline_curve: Baseline = None,
                               relative_to_baseline=True):
        """ Plots all optimization strategies with number of function evaluations on the x-axis """
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        absolute_optimum: float = info["absolute_optimum"]
        absolute_difference: float = info['absolute_difference']
        median: float = info['median']
        median_optimum_distance = median - absolute_optimum

        # plot the absolute optimum
        absolute_optimum_y_value = 1 if relative_to_baseline else absolute_optimum
        ax.axhline(absolute_optimum_y_value, c='black', ls='-.', label='Absolute optimum {}'.format(round(absolute_optimum, 3)))

        # plot baseline
        # sorted_times = np.array(info['sorted_times'])
        # # sorted_times = 1 - ((np.array(info['sorted_times']) - absolute_optimum) / absolute_difference)    # to fraction of optimum
        # # sorted_times = 1 - ((np.array(info['sorted_times']) - absolute_optimum) / median_optimum_distance)    # to fraction of median-optimum difference
        # cutoff_point_value, cutoff_point_fevals = calc_cutoff_point(0.98, info)
        # random_curve = get_random_curve(cutoff_point_fevals, sorted_times)
        # # ax.plot(random_curve, label="random trajectory", color='black', ls='--')
        if baseline_curve is not None:
            if relative_to_baseline is True:
                ax.axhline(0, label="baseline trajectory", color="black", ls="--")
            else:
                ax.plot(baseline_curve.get_curve_over_fevals(fevals_range), label="baseline curve", color="black", ls="--")

        # plot each strategy
        for strategy_index, strategy in enumerate(self.strategies):
            if "hide" in strategy.keys() and strategy["hide"]:
                continue

            # get the data
            color = colors[strategy_index]
            strategy_curve = strategies_curves[strategy_index]

            # TODO visualize error
            # ax.fill_between(
            #     range(len(results_obj_mean)),
            #     results_obj_mean - results_obj_std,
            #     results_obj_mean + results_obj_std,
            #     alpha=0.2,
            #     antialiased=True,
            #     color=color,
            # )
            # results_obj_mean = ((results_obj_mean - random_curve_strategy) / (absolute_optimum - random_curve_strategy))
            # y_min = min(min(results_obj_mean), y_min)
            curve = strategy_curve.get_curve_over_fevals(fevals_range)
            if relative_to_baseline:
                curve = baseline_curve.get_standardised_curve_over_fevals(fevals_range, curve, absolute_optimum)
            ax.plot(fevals_range, curve, label=f"{strategy['display_name']}", color=color)

        # # plot cutoff point
        # def plot_cutoff_point(cutoff_percentile):
        #     cutoff_point_value, cutoff_point_fevals = calc_cutoff_point(cutoff_percentile, info)
        #     print("")
        #     # print(f"percentage of searchspace to get to {cutoff_percentile*100}%: {round((cutoff_point_fevals/len(sorted_times))*100, 3)}%")
        #     # print(f"cutoff_point_fevals: {cutoff_point_fevals}")
        #     # print(f"objective_value_at_cutoff_point: {objective_value_at_cutoff_point}")
        #     ax.plot([cutoff_point_fevals], [cutoff_percentile], marker='o', color='red', label=f"cutoff point {cutoff_percentile}")

        # if baseline_curve is not None:
        #     plot_cutoff_point(0.980)

        ax.set_xlabel(self.x_metric_displayname["num_evals"])
        ax.set_ylabel(self.y_metric_displayname["objective_baseline_max"])
        ax.legend()

    def plot_strategies_curves(self, ax: plt.Axes, info: dict, strategies_curves: list[Curve], time_range: np.ndarray, baseline_curve=None,
                               relative_to_baseline=True, shaded=True, plot_errors=False):
        """Plots all optimization strategy curves"""
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        bar_groups_markers = {
            "reference": ".",
            "old": "+",
            "new": "d"
        }

        baseline = strategies_curves["baseline"]
        x_axis = baseline["x_axis"]

        # plot the absolute optimum
        absolute_optimum = info["absolute_optimum"]
        if subtract_baseline is False and absolute_optimum is not None:
            ax.axhline(absolute_optimum, label="Absolute optimum {}".format(round(absolute_optimum, 3)), c='black', ls='-')
        else:
            ax.axhline(0, label="Random search", c='black', ls=':')
            ax.axhline(1, label="Absolute optimum {}".format(round(absolute_optimum, 3)), c='black', ls='-.')

        color_index = 0
        marker = ","
        y_min = np.PINF
        y_max = np.NINF
        for strategy_curves in strategies_curves["strategies"]:
            # get the data
            strategy = strategies_data[strategy_curves["strategy_index"]]
            y_axis = strategy_curves["y_axis"]
            y_axis_std = strategy_curves["y_axis_std"]
            y_axis_std_lower = strategy_curves["y_axis_std_lower"]
            y_axis_std_upper = strategy_curves["y_axis_std_upper"]
            y_min = min(y_min, np.min(y_axis))
            y_max = max(y_max, np.max(y_axis))

            # set colors, transparencies and markers
            color = colors[color_index]
            color_index += 1
            alpha = 1.0
            fill_alpha = 0.2
            if "bar_group" in strategy:
                bar_group = strategy["bar_group"]
                marker = bar_groups_markers[bar_group]

            # plot the data
            if shaded is True:
                if plot_errors:
                    ax.fill_between(
                        x_axis,
                        y_axis_std_lower,
                        y_axis_std_upper,
                        alpha=fill_alpha,
                        antialiased=True,
                        color=color,
                    )
                ax.plot(
                    x_axis,
                    y_axis,
                    marker=marker,
                    alpha=alpha,
                    linestyle="-",
                    label=f"{strategy['display_name']}",
                    color=color,
                )
            else:
                if plot_errors:
                    ax.errorbar(
                        x_axis,
                        y_axis,
                        y_axis_std,
                        marker=marker,
                        alpha=alpha,
                        linestyle="--",
                        label=strategy["display_name"],
                    )
                else:
                    ax.plot(
                        x_axis,
                        y_axis,
                        marker=marker,
                        linestyle="-",
                        label=f"{strategy['display_name']}",
                        color=color,
                    )

        # finalize plot
        ax.axis([np.min(x_axis), np.max(x_axis), y_min * 0.9, y_max * 1.1])
        ax.set_xlabel(self.x_metric_displayname["kerneltime"])
        ax.set_ylabel(self.y_metric_displayname["objective_baseline_max" if subtract_baseline else "objective"])
        ax.set_ylim(bottom=y_min, top=1)
        ax.legend()
        if plot_errors is False:
            ax.grid(axis="y", zorder=0, alpha=0.7)

    def plot_aggregated_curves(self, ax: plt.Axes, strategies_aggregated: list):
        ax.axhline(0, label="Random search", c='black', ls=':')
        ax.axhline(1, label="Absolute optimum", c='black', ls='-.')
        overall_ymin = min(min(y_axis) for y_axis in strategies_aggregated)
        for strategy_index, y_axis in enumerate(strategies_aggregated):
            ax.plot(y_axis, label=self.strategies[strategy_index]["display_name"])

        ax.set_xlabel(self.x_metric_displayname["aggregate_time"])
        ax.set_ylabel(self.y_metric_displayname["aggregate_objective_max"])
        num_ticks = 11
        ax.set_xticks(
            np.linspace(0, y_axis.size, num_ticks),
            np.round(np.linspace(0, 1, num_ticks), 2),
        )
        ax.set_ylim(bottom=overall_ymin, top=1.0)
        ax.set_xlim((0, y_axis.size))
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
