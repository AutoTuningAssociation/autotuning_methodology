""" Visualize the results of the experiments """
import argparse
import numpy as np
from typing import Tuple, Any
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import auc
from math import ceil

from experiments import (
    execute_experiment,
    create_expected_results,
    get_searchspaces_info_stats,
)

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
    """Class for visualization of experiments"""

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.experiment, self.strategies, self.caches = execute_experiment(
                experiment_filename,
                profiling=False,
                searchspaces_info_stats=searchspaces_info_stats,
            )
        print("\n\n")

        # visualize
        all_strategies_curves = list()
        for gpu_name in self.experiment["GPUs"]:
            for kernel_name in self.experiment["kernels"]:
                print(f"  visualizing {kernel_name} on {gpu_name}")

                # create the figure and plots
                fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))    # if multiple subplots, pass the axis to the plot function with axs[0] etc.
                if not hasattr(axs, "__len__"):
                    axs = [axs]
                title = f"{kernel_name} on {gpu_name}"
                fig.canvas.manager.set_window_title(title)
                fig.suptitle(title)

                # prefetch the cached strategy results
                cache = self.caches[gpu_name][kernel_name]
                strategies_data = list()
                for strategy in self.strategies:
                    expected_results = create_expected_results()
                    cached_data = cache.get_strategy_results(
                        strategy["name"],
                        strategy["repeats"],
                        expected_results,
                    )
                    if cached_data is None:
                        raise ValueError(f"Strategy {strategy['display_name']} not in cache, make sure execute_experiment() has ran first")
                    strategies_data.append(cached_data)

                # visualize the results
                info = searchspaces_info_stats[gpu_name]["kernels"][kernel_name]
                # self.plot_strategies_fevals(axs[0], strategies_data, info)
                subtract_baseline = self.experiment["relative_to_baseline"]
                strategies_curves = self.get_strategies_curves(cache, strategies_data, info, subtract_baseline=subtract_baseline)
                self.plot_strategies_curves(
                    axs[0],
                    strategies_data,
                    strategies_curves,
                    info,
                    subtract_baseline=subtract_baseline,
                )
                all_strategies_curves.append(strategies_curves)

                # finalize the figure and display it
                fig.tight_layout()
                plt.show()

        # plot the aggregated data
        fig, axs = plt.subplots(ncols=1, figsize=(15, 8))    # if multiple subplots, pass the axis to the plot function with axs[0] etc.
        if not hasattr(axs, "__len__"):
            axs = [axs]
        title = f"Aggregated Data\nkernels: {', '.join(self.experiment['kernels'])}\nGPUs: {', '.join(self.experiment['GPUs'])}"
        fig.canvas.manager.set_window_title(title)
        fig.suptitle(title)

        # gather the aggregate y axis for each strategy
        print("\n")
        strategies_aggregated = list()
        for strategy_index, strategy in enumerate(self.strategies):
            perf = list()
            y_axis_temp = list()
            for strategies_curves in all_strategies_curves:
                for strategy_curve in strategies_curves["strategies"]:
                    if strategy_curve["strategy_index"] == strategy_index:
                        perf.append(strategy_curve["performance"])
                        y_axis_temp.append(strategy_curve["y_axis"])
            print(f"{strategy['display_name']} performance across kernels: {np.mean(perf)}")
            y_axis = np.array(y_axis_temp)
            strategies_aggregated.append(np.mean(y_axis, axis=0))

        # finalize the figure and display it
        self.plot_aggregated_curves(axs[0], strategies_aggregated)
        fig.tight_layout()
        plt.show()

    def get_random_curve(self, x_axis_length: int, absolute_optimum: float, sorted_times: list) -> np.ndarray:
        """ Returns the values of the random curve at each function value """
        # dist = np.random.rayleigh(size=100000)
        dist = sorted_times
        ks = range(x_axis_length)

        def redwhite_index(dist, M):
            N = len(dist)
            print("Running for subset size", M, end="\r", flush=True)
            #index = (N+1)*(N+1-M)*math.comb(N, M-1) / math.comb(N, M) / (M+1)
            index = M * (N + 1) / (M + 1)
            index = round(index)
            return dist[N - 1 - index]

        draws = np.array([redwhite_index(dist, k) for k in ks])
        # draws *= -1
        # draws += absolute_optimum
        # draws += np.max(draws) - np.min(draws)
        return draws

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
                y_axis = (y_axis - y_min) / (y_max - y_min)
                y_axis_std_lower = (y_axis_std_lower - y_min) / (y_max - y_min)
                y_axis_std_upper = (y_axis_std_upper - y_min) / (y_max - y_min)

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

    def plot_strategies_fevals(self, ax: plt.Axes, strategies_data: list, info: dict):
        """ Plots all optimization strategies with number of function evaluations on the x-axis """
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        absolute_optimum = info["absolute_optimum"]
        absolute_difference = info['absolute_difference']
        median = info['median']
        median_optimum_distance = median - absolute_optimum
        inverted_sorted_times_arr = np.array(info['sorted_times'])
        inverted_sorted_times_arr = inverted_sorted_times_arr[::-1]
        N = inverted_sorted_times_arr.shape[0]

        # plot each strategy
        x_axis_max = 0
        y_axis_max = np.NINF
        for strategy_index, strategy in enumerate(self.strategies):
            if "hide" in strategy.keys() and strategy["hide"]:
                continue

            # get the data
            color = colors[strategy_index]
            strategy = strategies_data[strategy_index]
            results = np.array(strategy["results"]["num_function_evaluations_repeated_results"])
            print(results)
            print(type(results))
            # results = 1 - ((results - absolute_optimum) / absolute_difference)    # to fraction of optimum
            results = 1 - ((results - absolute_optimum) / median_optimum_distance)    # to fraction of median-optimum difference
            results_obj_mean = list()
            results_obj_std = list()
            for num_func_evals in results:
                results_obj_mean.append(np.mean(num_func_evals))
                results_obj_std.append(np.std(num_func_evals))
            results_obj_mean = np.array(results_obj_mean)
            results_obj_std = np.array(results_obj_std)
            x_axis_max = max(len(results_obj_mean), x_axis_max)
            y_axis_max = max(max(results_obj_mean), y_axis_max)
            print(f"{y_axis_max=}")
            ax.fill_between(
                range(len(results_obj_mean)),
                results_obj_mean - results_obj_std,
                results_obj_mean + results_obj_std,
                alpha=0.2,
                antialiased=True,
                color=color,
            )
            ax.plot(results_obj_mean, label=f"{strategy['display_name']}", color=color)

        def calc_cutoff_point(cutoff_percentile):
            objective_value_at_cutoff_point = absolute_optimum + (median_optimum_distance * (1 - cutoff_percentile))
            # fevals_to_cutoff_point = ceil((cutoff_percentile * N) / (1 + (1 - cutoff_percentile) * N))

            # i = next(x[0] for x in enumerate(inverted_sorted_times_arr) if x[1] > cutoff_percentile * arr[-1])
            i = next(x[0] for x in enumerate(inverted_sorted_times_arr) if x[1] <= objective_value_at_cutoff_point)
            # In case of x <= (1+p) * f_opt
            # i = next(x[0] for x in enumerate(inverted_sorted_times_arr) if x[1] <= (1 + (1 - cutoff_percentile)) * arr[-1])
            # In case of p*x <= f_opt
            # i = next(x[0] for x in enumerate(inverted_sorted_times_arr) if cutoff_percentile * x[1] <= arr[-1])
            fevals_to_cutoff_point = ceil(i / (N + 1 - i))
            return fevals_to_cutoff_point

        # plot calculated random trajectory
        # sorted_times = 1 - ((np.array(info['sorted_times']) - absolute_optimum) / absolute_difference)    # to fraction of optimum
        sorted_times = 1 - ((np.array(info['sorted_times']) - absolute_optimum) / median_optimum_distance)    # to fraction of median-optimum difference
        cutoff_point = calc_cutoff_point(0.999)
        random_curve = self.get_random_curve(cutoff_point, absolute_optimum, sorted_times)
        ax.plot(random_curve, label="random trajectory", color='green')

        # plot cutoff point
        def plot_cutoff_point(cutoff_percentile):
            fevals_to_cutoff_point = calc_cutoff_point(cutoff_percentile)
            print("")
            print(f"percentage of searchspace to get to {cutoff_percentile*100}%: {round((fevals_to_cutoff_point/N)*100, 3)}%")
            # print(f"fevals_to_cutoff_point: {fevals_to_cutoff_point}")
            # print(f"objective_value_at_cutoff_point: {objective_value_at_cutoff_point}")
            ax.plot([fevals_to_cutoff_point], [cutoff_percentile], marker='o', color='red', label=f"cutoff point {cutoff_percentile}")

        plot_cutoff_point(0.950)
        plot_cutoff_point(0.960)
        plot_cutoff_point(0.970)
        plot_cutoff_point(0.975)
        plot_cutoff_point(0.980)
        plot_cutoff_point(0.990)
        plot_cutoff_point(0.995)
        plot_cutoff_point(0.996)
        plot_cutoff_point(0.997)
        plot_cutoff_point(0.998)
        plot_cutoff_point(0.999)

        # # plot the absolute optimum
        # if absolute_optimum is not None:
        #     ax.plot(
        #         [0, x_axis_max],
        #         [absolute_optimum, absolute_optimum],
        #         linestyle="-",
        #         label="True optimum {}".format(round(absolute_optimum, 3)),
        #         color="black",
        #     )

        # ax.set_ylim(bottom=absolute_optimum * 0.9, top=y_axis_max * 1.1)
        ax.set_ylim(bottom=0.8, top=1)
        ax.set_xlabel(self.x_metric_displayname["num_evals"])
        ax.set_ylabel(self.y_metric_displayname["objective_relative_median"])
        ax.legend()

    def plot_strategies_curves(
        self,
        ax: plt.Axes,
        strategies_data: list,
        strategies_curves: dict,
        info: dict,
        shaded=True,
        plot_errors=False,
        subtract_baseline=True,
    ):
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
        overall_ymin = min(min(strategy_curves["y_axis"]) for strategy_curves in strategies_curves["strategies"])
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
        ax.set_ylim(bottom=overall_ymin, top=1)
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
