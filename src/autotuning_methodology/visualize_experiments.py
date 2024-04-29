"""Visualize the results of the experiments."""

from __future__ import annotations  # for correct nested type hints e.g. list[str], tuple[dict, str]

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autotuning_methodology.baseline import Baseline, RandomSearchCalculatedBaseline, RandomSearchSimulatedBaseline
from autotuning_methodology.curves import Curve, CurveBasis, StochasticOptimizationAlgorithm
from autotuning_methodology.experiments import execute_experiment, get_args_from_cli
from autotuning_methodology.searchspace_statistics import SearchspaceStatistics

# The kernel information per device and device information for visualization purposes
marker_variatons = ["v", "s", "*", "1", "2", "d", "P", "X"]

# total set of objective time keys
objective_time_keys_values = ["compilation", "benchmark", "framework", "search_algorithm", "validation"]


class Visualize:
    """Class for visualization of experiments."""

    x_metric_displayname = dict(
        {
            "fevals": "Number of function evaluations used",
            "time_total": "Total time in seconds",
            "aggregate_time": "Relative time to cutoff point",
            "time_partial_framework_time": "framework time",
            "time_partial_strategy_time": "strategy time",
            "time_partial_compile_time": "compile time",
            "time_partial_benchmark_time": "kernel runtime",
            "time_partial_times": "kernel runtime",
            "time_partial_verification_time": "verification time",
        }
    )

    y_metric_displayname = dict(
        {
            "objective_absolute": "Best-found objective value",
            "objective_scatter": "Best-found objective value",
            "objective_relative_median": "Fraction of absolute optimum relative to median",
            "objective_normalized": "Best-found objective value\n(normalized from median to optimum)",
            "objective_baseline": "Best-found objective value\n(relative to baseline)",
            "objective_baseline_max": "Improvement over random sampling",
            "aggregate_objective": "Aggregate best-found objective value relative to baseline",
            "aggregate_objective_max": "Aggregate improvement over random sampling",
            "time": "Best-found kernel time in miliseconds",
            "GFLOP/s": "GFLOP/s",
        }
    )

    plot_x_value_types = ["fevals", "time", "aggregated"]  # number of function evaluations, time, aggregation
    plot_y_value_types = [
        "absolute",
        "scatter",
        "normalized",
        "baseline",
    ]  # absolute values, scatterplot, median-absolute normalized, improvement over baseline

    plot_filename_prefix_parent = "generated_plots"

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def __init__(
        self,
        experiment_filepath: str,
        save_figs=True,
        save_extra_figs=False,
        continue_after_comparison=False,
        compare_extra_baselines=False,
    ) -> None:
        """Initialization method for the Visualize class.

        Args:
            experiment_filepath: the path to the experiment-filename.json to run.
            save_figs: whether to save the figures to file, if not, displays in a window. Defaults to True.
            save_extra_figs: whether to save split times and baseline comparisons figures to file. Defaults to False.
            continue_after_comparison: whether to continue plotting after processing comparisons. Defaults to False.
            compare_extra_baselines: whether to include additional baselines for comparison. Defaults to False.

        Raises:
            ValueError: on various invalid inputs.
        """
        # # silently execute the experiment
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        self.experiment, self.strategies, self.results_descriptions = execute_experiment(
            experiment_filepath, profiling=False
        )
        experiment_folderpath = Path(experiment_filepath).parent
        experiment_folder_id: str = self.experiment["folder_id"]
        assert isinstance(experiment_folder_id, str) and len(experiment_folder_id) > 0
        self.plot_filename_prefix = f"{self.plot_filename_prefix_parent}/{experiment_folder_id}/"
        print("\n")
        print("Visualizing")

        # preparing filesystem
        if save_figs or save_extra_figs:
            Path(self.plot_filename_prefix_parent).mkdir(exist_ok=True)
            Path(self.plot_filename_prefix).mkdir(exist_ok=True)

        # settings
        self.minimization: bool = self.experiment.get("minimization", True)
        cutoff_percentile: float = self.experiment["cutoff_percentile"]
        cutoff_percentile_start: float = self.experiment.get("cutoff_percentile_start", 0.01)
        cutoff_type: str = self.experiment.get("cutoff_type", "fevals")
        assert cutoff_type == "fevals" or cutoff_type == "time", f"cutoff_type != 'fevals' or 'time', is {cutoff_type}"
        time_resolution: float = self.experiment.get("resolution", 1e4)
        if int(time_resolution) != time_resolution:
            raise ValueError(f"The resolution must be an integer, yet is {time_resolution}.")
        time_resolution = int(time_resolution)
        objective_time_keys: list[str] = self.experiment["objective_time_keys"]
        objective_performance_keys: list[str] = self.experiment["objective_performance_keys"]

        # plot settings
        plot_settings: dict = self.experiment["plot"]
        plot_x_value_types: list[str] = plot_settings["plot_x_value_types"]
        plot_y_value_types: list[str] = plot_settings["plot_y_value_types"]
        compare_baselines: bool = plot_settings.get("compare_baselines", False)
        compare_split_times: bool = plot_settings.get("compare_split_times", False)
        confidence_level: float = plot_settings.get("confidence_level", 0.95)

        # visualize
        aggregation_data: list[tuple[Baseline, list[Curve], SearchspaceStatistics, np.ndarray]] = list()
        for gpu_name in self.experiment["GPUs"]:
            for kernel_name in self.experiment["kernels"]:
                print(f" | visualizing optimization of {kernel_name} for {gpu_name}")
                title = f"{kernel_name} on {gpu_name}"
                title = title.replace("_", " ")

                # get the statistics
                searchspace_stats = SearchspaceStatistics(
                    kernel_name=kernel_name,
                    device_name=gpu_name,
                    minimization=self.minimization,
                    objective_time_keys=objective_time_keys,
                    objective_performance_keys=objective_performance_keys,
                    bruteforced_caches_path=experiment_folderpath / self.experiment["bruteforced_caches_path"],
                )

                # get the cached strategy results as curves
                strategies_curves: list[Curve] = list()
                for strategy in self.strategies:
                    results_description = self.results_descriptions[gpu_name][kernel_name][strategy["name"]]
                    if results_description is None:
                        raise ValueError(
                            f"""Strategy {strategy['display_name']} not in results_description,
                                make sure execute_experiment() has ran first"""
                        )
                    strategies_curves.append(StochasticOptimizationAlgorithm(results_description))

                # set the x-axis range
                _, cutoff_point_fevals, cutoff_point_time = searchspace_stats.cutoff_point_fevals_time(
                    cutoff_percentile
                )
                _, cutoff_point_fevals_start, cutoff_point_time_start = searchspace_stats.cutoff_point_fevals_time(
                    cutoff_percentile_start
                )
                fevals_range = np.arange(start=cutoff_point_fevals_start, stop=cutoff_point_fevals)
                time_range = np.linspace(start=cutoff_point_time_start, stop=cutoff_point_time, num=time_resolution)
                # baseline_time_interpolated = np.linspace(mean_feval_time, cutoff_point_time, time_resolution)
                # baseline = get_random_curve(cutoff_point_fevals, sorted_times, time_resolution)

                # compare baselines
                if compare_baselines is True:
                    self.plot_baselines_comparison(
                        time_range,
                        searchspace_stats,
                        objective_time_keys,
                        confidence_level=confidence_level,
                        title=title,
                        strategies_curves=strategies_curves,
                        save_fig=save_extra_figs,
                    )
                if compare_split_times is True:
                    self.plot_split_times_comparison(
                        "fevals",
                        fevals_range,
                        searchspace_stats,
                        objective_time_keys,
                        title=title,
                        strategies_curves=strategies_curves,
                        save_fig=save_extra_figs,
                    )
                    self.plot_split_times_comparison(
                        "time",
                        time_range,
                        searchspace_stats,
                        objective_time_keys,
                        title=title,
                        strategies_curves=strategies_curves,
                        save_fig=save_extra_figs,
                    )
                    self.plot_split_times_bar_comparison(
                        "time",
                        time_range,
                        searchspace_stats,
                        objective_time_keys,
                        title=title,
                        strategies_curves=strategies_curves,
                        save_fig=save_extra_figs,
                    )
                if not continue_after_comparison and (compare_baselines is True or compare_split_times is True):
                    continue

                # get the random baseline
                random_baseline = RandomSearchCalculatedBaseline(searchspace_stats)

                # set additional baselines for comparison
                baselines_extra: list[Baseline] = []
                if compare_extra_baselines is True:
                    baselines_extra.append(RandomSearchSimulatedBaseline(searchspace_stats, repeats=1000))
                    # baselines_extra.append(RandomSearchCalculatedBaseline(searchspace_stats, include_nan=True))

                # collect aggregatable data
                aggregation_data.append(tuple([random_baseline, strategies_curves, searchspace_stats, time_range]))

                # visualize the results
                for x_type in plot_x_value_types:
                    if x_type == "aggregated":
                        continue
                    elif x_type == "fevals":
                        x_axis_range = fevals_range
                    elif x_type == "time":
                        x_axis_range = time_range
                    else:
                        raise ValueError(f"Invalid {x_type=}")

                    # create the figure and plots
                    fig, axs = plt.subplots(
                        nrows=len(plot_y_value_types),
                        ncols=1,
                        figsize=(8, 3.4 * len(plot_y_value_types)),
                        sharex=True,
                        dpi=300,
                    )
                    if not hasattr(
                        axs, "__len__"
                    ):  # if there is just one subplot, wrap it in a list so it can be passed to the plot functions
                        axs = [axs]
                    fig.canvas.manager.set_window_title(title)
                    if not save_figs:
                        fig.suptitle(title)

                    # plot the subplots of individual searchspaces
                    for index, y_type in enumerate(plot_y_value_types):
                        self.plot_strategies(
                            x_type,
                            y_type,
                            axs[index],
                            searchspace_stats,
                            strategies_curves,
                            x_axis_range,
                            plot_settings,
                            random_baseline,
                            baselines_extra=baselines_extra,
                        )
                        if index == 0:
                            loc = "lower right" if y_type == "normalized" else "best"
                            axs[index].legend(loc=loc)

                    # finalize the figure and save or display it
                    fig.supxlabel(self.get_x_axis_label(x_type, objective_time_keys))
                    fig.tight_layout()
                    if save_figs:
                        filename = f"{self.plot_filename_prefix}{title}_{x_type}"
                        filename = filename.replace(" ", "_")
                        fig.savefig(filename, dpi=300)
                        print(f"Figure saved to {filename}")
                    else:
                        plt.show()

        # plot the aggregated searchspaces
        if (
            "aggregated" in plot_x_value_types
            and continue_after_comparison
            or not (compare_baselines or compare_split_times)
        ):
            fig, axs = plt.subplots(
                ncols=1, figsize=(9, 6), dpi=300
            )  # if multiple subplots, pass the axis to the plot function with axs[0] etc.
            if not hasattr(axs, "__len__"):
                axs = [axs]
            title = f"""Aggregated Data\nkernels:
                    {', '.join(self.experiment['kernels'])}\nGPUs: {', '.join(self.experiment['GPUs'])}"""
            fig.canvas.manager.set_window_title(title)
            if not save_figs:
                fig.suptitle(title)

            # finalize the figure and save or display it
            self.plot_strategies_aggregated(axs[0], aggregation_data, plot_settings=plot_settings)
            fig.tight_layout()
            if save_figs:
                filename = f"{self.plot_filename_prefix}aggregated"
                filename = filename.replace(" ", "_")
                fig.savefig(filename, dpi=300)
                print(f"Figure saved to {filename}")
            else:
                plt.show()

    def plot_baselines_comparison(
        self,
        time_range: np.ndarray,
        searchspace_stats: SearchspaceStatistics,
        objective_time_keys: list,
        confidence_level: float,
        title: str = None,
        strategies_curves: list[Curve] = list(),
        save_fig=False,
    ):
        """Plots a comparison of baselines on a time range.

        Optionally also compares against strategies listed in strategies_curves.

        Args:
            time_range: range of time to plot on.
            searchspace_stats: Searchspace statistics object.
            objective_time_keys: objective time keys.
            confidence_level: the confidence interval used for the confidence / prediction interval.
            title: the title for this plot, if not given, a title is generated. Defaults to None.
            strategies_curves: the strategy curves to draw in the plot. Defaults to list().
            save_fig: whether to save the resulting figure to file. Defaults to False.
        """
        dist = searchspace_stats.objective_performances_total_sorted
        plt.figure(figsize=(8, 5), dpi=300)

        # list the baselines to test
        baselines: list[Baseline] = list()
        # baselines.append(
        #     RandomSearchCalculatedBaseline(searchspace_stats, include_nan=False, time_per_feval_operator="median")
        # )
        baselines.append(RandomSearchCalculatedBaseline(searchspace_stats, time_per_feval_operator="mean"))
        baselines.append(
            RandomSearchCalculatedBaseline(
                searchspace_stats, include_nan=True, time_per_feval_operator="median_per_feval"
            )
        )

        # plot random baseline implementations
        for baseline in baselines:
            plt.plot(time_range, baseline.get_curve_over_time(time_range), label=baseline.label)

        # plot normal strategies
        for strategy_curve in strategies_curves:
            (
                _,
                x_axis_range_real,
                curve_real,
                curve_lower_err_real,
                curve_upper_err_real,
                x_axis_range_fictional,
                curve_fictional,
                curve_lower_err_fictional,
                curve_upper_err_fictional,
            ) = strategy_curve.get_curve_over_time(time_range, dist=dist, confidence_level=confidence_level)
            # when adding error shades to visualization, don't forget to pass confidence interval to get_curve_over_time
            plt.plot(x_axis_range_real, curve_real, label=strategy_curve.display_name, linestyle="dashed")
            if x_axis_range_fictional.ndim > 0:
                plt.plot(x_axis_range_fictional, curve_fictional, linestyle="dotted")

        # finalize the plot
        if title is not None:
            plt.title(title)
        plt.xlabel(self.get_x_axis_label("time", objective_time_keys))
        plt.ylabel(self.y_metric_displayname["objective_absolute"])
        plt.xlim(time_range[0], time_range[-1])
        plt.legend()
        plt.tight_layout()

        # write to file or show
        if save_fig:
            filename = f"{self.plot_filename_prefix}{title}_baselines"
            filename = filename.replace(" ", "_")
            plt.savefig(filename, dpi=300)
            print(f"Figure saved to {filename}")
        else:
            plt.show()

    def plot_split_times_comparison(
        self,
        x_type: str,
        fevals_or_time_range: np.ndarray,
        searchspace_stats: SearchspaceStatistics,
        objective_time_keys: list,
        title: str = None,
        strategies_curves: list[Curve] = list(),
        save_fig=False,
    ):
        """Plots a comparison of split times for strategies and baselines over the given range.

        Args:
            x_type: the type of ``fevals_or_time_range``.
            fevals_or_time_range: the time or function evaluations range to plot on.
            searchspace_stats: the Searchspace statistics object.
            objective_time_keys: the objective time keys.
            title: the title for this plot, if not given, a title is generated. Defaults to None.
            strategies_curves: the strategy curves to draw in the plot. Defaults to list().
            save_fig: whether to save the resulting figure to file. Defaults to False.

        Raises:
            ValueError: on unexpected strategies curve instance.
        """
        # list the baselines to test
        baselines: list[Baseline] = list()
        # baselines.append(
        #     RandomSearchCalculatedBaseline(
        #         searchspace_stats, include_nan=True, time_per_feval_operator="median_per_feval"
        #     )
        # )
        lines: list[CurveBasis] = strategies_curves + baselines

        # setup the subplots
        num_rows = len(lines)
        fig, axs = plt.subplots(nrows=num_rows, ncols=1, figsize=(9, 3 * num_rows), sharex=True)
        if not hasattr(
            axs, "__len__"
        ):  # if there is just one subplot, wrap it in a list so it can be passed to the plot functions
            axs = [axs]
        if title is not None:
            fig.canvas.manager.set_window_title(title)
            fig.suptitle(title)
        labels = list(key for key in objective_time_keys)

        # plot the baselines and strategies
        for ax_index, line in enumerate(lines):
            ax = axs[ax_index]
            if isinstance(line, Curve):
                curvetitle = line.display_name
            elif isinstance(line, Baseline):
                curvetitle = line.label
            else:
                raise ValueError(f"Expected Curve or Baseline instance, but line is {type(line)}")
            split_times = line.get_split_times(fevals_or_time_range, x_type, searchspace_stats)
            ax.set_title(curvetitle)
            ax.stackplot(fevals_or_time_range, split_times, labels=labels)
            ax.set_ylabel(self.get_x_axis_label("time", objective_time_keys))
            ax.set_xlim(fevals_or_time_range[0], fevals_or_time_range[-1])
            # plot the mean
            mean = np.mean(np.sum(split_times, axis=0))
            ax.axhline(y=mean, label="Mean sum")
            if isinstance(line, Baseline):
                average_time_per_feval_used = searchspace_stats.get_time_per_feval(line.time_per_feval_operator)
                ax.axhline(y=average_time_per_feval_used, label="Average used")
                print(f"{curvetitle} mean: {round(mean, 3)}, average used: {round(average_time_per_feval_used, 3)}")
            else:
                print(f"{curvetitle} mean: {round(mean, 3)}")

        # finalize the plot
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels)
        fig.supxlabel(self.get_x_axis_label(x_type, objective_time_keys))
        fig.tight_layout()

        # write to file or show
        if save_fig:
            filename = f"{self.plot_filename_prefix}{title}_split_times_{x_type}"
            filename = filename.replace(" ", "_")
            plt.savefig(filename, dpi=300)
            print(f"Figure saved to {filename}")
        else:
            plt.show()

    def plot_split_times_bar_comparison(
        self,
        x_type: str,
        fevals_or_time_range: np.ndarray,
        searchspace_stats: SearchspaceStatistics,
        objective_time_keys: list[str],
        title: str = None,
        strategies_curves: list[Curve] = list(),
        print_table_format=True,
        print_skip=["validation"],
        save_fig=False,
    ):
        """Plots a bar chart comparison of the average split times for strategies over the given range.

        Args:
            x_type: the type of ``fevals_or_time_range``.
            fevals_or_time_range: the time or function evaluations range to plot on.
            searchspace_stats: the Searchspace statistics object.
            objective_time_keys: the objective time keys.
            title: the title for this plot, if not given, a title is generated. Defaults to None.
            strategies_curves: the strategy curves to draw in the plot. Defaults to list().
            print_table_format: print a LaTeX-formatted table. Defaults to True.
            print_skip: list of ``time_keys`` to be skipped in the printed table. Defaults to ["verification_time"].
            save_fig: whether to save the resulting figure to file. Defaults to False.
        """
        fig, ax = plt.subplots(dpi=200)
        width = 0.5
        strategy_labels = list()

        for print_skip_key in print_skip:
            assert (
                print_skip_key in objective_time_keys
            ), f"Each key in print_skip must be in objective_time_keys, {print_skip_key} is not ({objective_time_keys})"

        # get a dictionary of {time_key: [array_average_time_per_strategy]}
        data_dict = dict.fromkeys(objective_time_keys)
        data_table = list(
            list(list() for _ in range(len(objective_time_keys) - len(print_skip)))
            for _ in range(len(strategies_curves) + 1)
        )
        for objective_time_key in objective_time_keys:
            data_dict[objective_time_key] = np.full((len(strategies_curves)), np.NaN)
        for strategy_index, strategy_curve in enumerate(strategies_curves):
            print_skip_counter = 0
            strategy_labels.append(strategy_curve.display_name)
            strategy_split_times = strategy_curve.get_split_times(fevals_or_time_range, x_type, searchspace_stats)
            # print(f"{strategy_curve.display_name}: ({strategy_split_times.shape})")
            for objective_time_key_index, objective_time_key in enumerate(objective_time_keys):
                key_split_times = strategy_split_times[objective_time_key_index]
                key_split_times = key_split_times[key_split_times > 0]
                split_time = max(np.median(key_split_times), 0.0)
                split_time = 0.0 if np.isnan(split_time) else split_time
                data_dict[objective_time_key][strategy_index] = split_time
                # print(
                #     f"""    {objective_time_key}: {key_split_times[key_split_times > 0].shape},
                #             {np.mean(key_split_times)}, {np.median(key_split_times)}"""
                # )
                if objective_time_key not in print_skip:
                    if strategy_index == 0:
                        data_table[0][objective_time_key_index - print_skip_counter] = objective_time_key.replace(
                            "_", " "
                        )
                    data_table[strategy_index + 1][objective_time_key_index - print_skip_counter] = str(
                        "%.3g" % split_time
                    )
                else:
                    print_skip_counter += 1

        # print in table format
        if print_table_format:
            print("")
            num_times = len(data_table[0])
            print("\\begin{tabularx}{\linewidth}{l" + "|X" * num_times + "}")
            print("\hline")
            header = "} & \\textbf{".join(data_table[0])
            print("\\textbf{Algorithm} & \\textbf{" + header + "} \\" + "\\")
            print("\hline")
            for strategy_index in range(len(strategy_labels)):
                print(
                    f"    {strategy_labels[strategy_index]} & {' & '.join(data_table[strategy_index + 1])} \\\\\hline"
                )
            print("\end{tabularx}")

        # plot the bars
        bottom = np.zeros(len(strategies_curves))
        for objective_time_key, objective_times in data_dict.items():
            objective_times = np.array(objective_times)
            ax.bar(strategy_labels, objective_times, width, label=objective_time_key, bottom=bottom)
            bottom += objective_times

        # finalize the plot
        ax.set_ylabel(self.get_x_axis_label(x_type, objective_time_keys))
        ax.legend()
        # ax.set_title(title)
        # fig.supxlabel("Median split times per optimization algorithm")
        fig.tight_layout()

        # write to file or show
        if save_fig:
            filename = f"{self.plot_filename_prefix}{title}_split_times_bar"
            filename = filename.replace(" ", "_")
            plt.savefig(filename, dpi=300)
            print(f"Figure saved to {filename}")
        else:
            plt.show()

    def plot_strategies(
        self,
        x_type: str,
        y_type: str,
        ax: plt.Axes,
        searchspace_stats: SearchspaceStatistics,
        strategies_curves: list[Curve],
        x_axis_range: np.ndarray,
        plot_settings: dict,
        baseline_curve: Baseline = None,
        baselines_extra: list[Baseline] = list(),
        plot_errors=True,
        plot_cutoffs=False,
    ):
        """Plots all optimization strategies for individual search spaces.

        Args:
            x_type: the type of ``x_axis_range``.
            y_type: the type of plot on the y-axis.
            ax: the axis to plot on.
            searchspace_stats: the Searchspace statistics object.
            strategies_curves: the strategy curves to draw in the plot. Defaults to list().
            x_axis_range: the time or function evaluations range to plot on.
            plot_settings: dictionary of additional plot settings.
            baseline_curve: the ``Baseline`` to be used as a baseline in the plot. Defaults to None.
            baselines_extra: additional ``Baseline`` curves to compare against. Defaults to list().
            plot_errors: whether errors (confidence / prediction intervals) are visualized. Defaults to True.
            plot_cutoffs: whether the cutoff points for early stopping algorithms are visualized. Defaults to False.

        """
        confidence_level: float = plot_settings.get("confidence_level", 0.95)
        absolute_optimum = searchspace_stats.total_performance_absolute_optimum()
        median = searchspace_stats.total_performance_median()
        optimum_median_difference = absolute_optimum - median

        def normalize(curve):
            """Min-max normalization with median as min and absolute optimum as max."""
            if curve is None:
                return None
            return (curve - median) / optimum_median_difference

        def normalize_multiple(curves: list) -> tuple:
            """Normalize multiple curves at once."""
            return tuple(normalize(curve) for curve in curves)

        # plot the absolute optimum
        absolute_optimum_y_value = absolute_optimum if y_type == "absolute" or y_type == "scatter" else 1
        absolute_optimum_label = (
            "Absolute optimum ({})".format(round(absolute_optimum, 3)) if y_type == "absolute" else "Absolute optimum"
        )
        ax.axhline(absolute_optimum_y_value, c="black", ls="-.", label=absolute_optimum_label)

        # plot baseline
        if baseline_curve is not None:
            if y_type == "baseline":
                ax.axhline(0, label="baseline trajectory", color="black", ls="--")
            elif y_type == "normalized" or y_type == "baseline":
                baseline = baseline_curve.get_curve(x_axis_range, x_type)
                if y_type == "normalized":
                    baseline = normalize(baseline)
                ax.plot(x_axis_range, baseline, label="Calculated baseline", color="black", ls="--")

        # plot additional baselines if provided
        baselines_extra_curves = list()
        for baseline_extra in baselines_extra:
            curve = baseline_extra.get_curve(x_axis_range, x_type)
            if y_type == "normalized":
                curve = normalize(curve)
            elif y_type == "baseline":
                curve = baseline_curve.get_standardised_curve(x_axis_range, curve, x_type=x_type)
            ax.plot(x_axis_range, curve, label=baseline_extra.label, ls=":")
            baselines_extra_curves.append(curve)
        if len(baselines_extra) >= 2:
            ax.plot(x_axis_range, np.mean(baselines_extra_curves, axis=0), label="Mean of extra baselines", ls=":")

        # plot each strategy
        dist = searchspace_stats.objective_performances_total_sorted
        ylim_min = 0
        for strategy_index, strategy in enumerate(self.strategies):
            if "hide" in strategy.keys() and strategy["hide"]:
                continue

            # get the data
            color = self.colors[strategy_index]
            label = f"{strategy['display_name']}"
            strategy_curve = strategies_curves[strategy_index]

            # get the plot data
            if y_type == "scatter":
                x_axis, y_axis = strategy_curve.get_scatter_data(x_type)
                ax.scatter(x_axis, y_axis, label=label, color=color)
                continue
            else:
                (
                    real_stopping_point,
                    x_axis_range_real,
                    curve_real,
                    curve_lower_err_real,
                    curve_upper_err_real,
                    x_axis_range_fictional,
                    curve_fictional,
                    curve_lower_err_fictional,
                    curve_upper_err_fictional,
                ) = strategy_curve.get_curve(x_axis_range, x_type, dist=dist, confidence_level=confidence_level)

            # transform the curves as necessary and set ylims
            if y_type == "normalized":
                curve_real, curve_lower_err_real, curve_upper_err_real = normalize_multiple(
                    [curve_real, curve_lower_err_real, curve_upper_err_real]
                )
                curve_fictional, curve_lower_err_fictional, curve_upper_err_fictional = normalize_multiple(
                    [curve_fictional, curve_lower_err_fictional, curve_upper_err_fictional]
                )
            elif y_type == "baseline":
                curve_real, curve_lower_err_real, curve_upper_err_real = baseline_curve.get_standardised_curves(
                    x_axis_range_real, [curve_real, curve_lower_err_real, curve_upper_err_real], x_type
                )
                if x_axis_range_fictional.ndim > 0:
                    (
                        curve_fictional,
                        curve_lower_err_fictional,
                        curve_upper_err_fictional,
                    ) = baseline_curve.get_standardised_curves(
                        x_axis_range_fictional,
                        [curve_fictional, curve_lower_err_fictional, curve_upper_err_fictional],
                        x_type,
                    )
                ylim_min = min(np.min(curve_real), ylim_min)

            # visualize
            if plot_errors:
                ax.fill_between(
                    x_axis_range_real,
                    curve_lower_err_real,
                    curve_upper_err_real,
                    alpha=0.15,
                    antialiased=True,
                    color=color,
                )
            ax.plot(x_axis_range_real, curve_real, label=label, color=color)

            # select the parts of the data that are fictional
            if real_stopping_point < x_axis_range.shape[-1]:
                # visualize fictional part
                if plot_errors:
                    ax.fill_between(
                        x_axis_range_fictional,
                        curve_lower_err_fictional,
                        curve_upper_err_fictional,
                        alpha=0.15,
                        antialiased=True,
                        color=color,
                        ls="dashed",
                    )
                ax.plot(x_axis_range_fictional, curve_fictional, color=color, ls="dashed")

        # # plot cutoff point
        # def plot_cutoff_point(cutoff_percentiles: np.ndarray, show_label=True):
        #     """ plot the cutoff point """
        #     cutoff_point_values = list()
        #     cutoff_point_fevals = list()
        #     for cutoff_percentile in cutoff_percentiles:
        #         cutoff_point_value, cutoff_point_feval = searchspace_stats.cutoff_point(cutoff_percentile)
        #         cutoff_point_values.append(cutoff_point_value)
        #         cutoff_point_fevals.append(cutoff_point_feval)

        #     # get the correct value depending on the plot type
        #     if y_type == 'absolute':
        #         y_values = cutoff_point_values
        #     elif y_type == 'normalized':
        #         y_values = normalize(cutoff_point_values)
        #     elif y_type == 'baseline':
        #         y_values = baseline_curve.get_standardised_curve_over_fevals(fevals_range, cutoff_percentiles)

        #     # plot
        #     label = f"cutoff point" if show_label else None
        #     ax.plot(cutoff_point_fevals, y_values, marker='o', color='red', label=label)

        # # test a range of cutoff percentiles to see if they match with random search
        # if y_type == 'absolute' or y_type == 'normalized':
        #     cutoff_percentile_start = self.experiment.get("cutoff_percentile_start", 0.01)
        #     cutoff_percentile_end = self.experiment.get("cutoff_percentile")
        #     cutoff_percentiles_low_precision = np.arange(cutoff_percentile_start, 0.925, step=0.05)
        #     cutoff_percentiles_high_precision = np.arange(0.925, cutoff_percentile_end, step=0.001)
        #     plot_cutoff_point(np.concatenate([cutoff_percentiles_low_precision, cutoff_percentiles_high_precision]))

        # finalize the plot
        ax.set_xlim(tuple([x_axis_range[0], x_axis_range[-1]]))
        ax.set_ylabel(self.y_metric_displayname[f"objective_{y_type}"], fontsize="large")
        normalized_ylim_margin = 0.02
        if y_type == "absolute":
            multiplier = 0.99 if self.minimization else 1.01
            ax.set_ylim(absolute_optimum * multiplier, median)
        # elif y_type == 'normalized':
        #     ax.set_ylim((0.0, 1 + normalized_ylim_margin))
        elif y_type == "baseline":
            ax.set_ylim((min(-normalized_ylim_margin, ylim_min - normalized_ylim_margin), 1 + normalized_ylim_margin))

    def get_strategies_aggregated_performance(
        self,
        aggregation_data: list[tuple[Baseline, list[Curve], SearchspaceStatistics, np.ndarray]],
        confidence_level: float,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
        """Combines the performances across searchspaces.

        Args:
            aggregation_data: the aggregated data from the various searchspaces.
            confidence_level: the confidence interval used for the confidence / prediction interval.

        Returns:
            The aggregated relative performances of each strategy.
        """
        # for each strategy, collect the relative performance in each search space
        strategies_performance = [list() for _ in aggregation_data[0][1]]
        strategies_performance_lower_err = [list() for _ in aggregation_data[0][1]]
        strategies_performance_upper_err = [list() for _ in aggregation_data[0][1]]
        strategies_performance_real_stopping_point_fraction = [list() for _ in range(len(aggregation_data[0][1]))]
        for random_baseline, strategies_curves, searchspace_stats, time_range in aggregation_data:
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
                x_axis_range = (
                    np.concatenate([x_axis_range_real, x_axis_range_fictional]) if combine else x_axis_range_real
                )
                assert np.array_equal(time_range, x_axis_range, equal_nan=True), "time_range != x_axis_range"
                curve = np.concatenate([curve_real, curve_fictional]) if combine else curve_real
                curve_lower_err = (
                    np.concatenate([curve_lower_err_real, curve_lower_err_fictional])
                    if combine
                    else curve_lower_err_real
                )
                curve_upper_err = (
                    np.concatenate([curve_upper_err_real, curve_upper_err_fictional])
                    if combine
                    else curve_upper_err_real
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
        for index in range(len(strategies_performance)):
            strategies_aggregated_performance.append(np.mean(np.array(strategies_performance[index]), axis=0))
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

    def plot_strategies_aggregated(
        self,
        ax: plt.Axes,
        aggregation_data: list[tuple[Baseline, list[Curve], SearchspaceStatistics, np.ndarray]],
        plot_settings: dict,
    ):
        """Plots all optimization strategies combined accross search spaces.

        Args:
            ax: the axis to plot on.
            aggregation_data: the aggregated data from the various searchspaces.
            plot_settings: dictionary of additional plot settings.
        """
        # plot the random baseline and absolute optimum
        ax.axhline(0, label="Calculated baseline", c="black", ls=":")
        ax.axhline(1, label="Absolute optimum", c="black", ls="-.")

        # get the relative aggregated performance for each strategy
        confidence_level: float = plot_settings.get("confidence_level", 0.95)
        (
            strategies_performance,
            strategies_lower_err,
            strategies_upper_err,
            strategies_real_stopping_point_fraction,
        ) = self.get_strategies_aggregated_performance(aggregation_data, confidence_level)

        # plot each strategy
        y_axis_size = strategies_performance[0].shape[0]
        time_range = np.arange(y_axis_size)
        plot_errors = True
        print("\n-------")
        print("Quantification of aggregate performance across all search spaces:")
        for strategy_index, strategy_performance in enumerate(strategies_performance):
            displayname = self.strategies[strategy_index]["display_name"]
            color = self.colors[strategy_index]
            real_stopping_point_fraction = strategies_real_stopping_point_fraction[strategy_index]
            real_stopping_point_index = round(real_stopping_point_fraction * time_range.shape[0])
            if real_stopping_point_index <= 0:
                warnings.warn(f"Stopping point index for {displayname} is at {real_stopping_point_index}")
                continue

            # plot the errors
            if plot_errors:
                strategy_lower_err = strategies_lower_err[strategy_index]
                strategy_upper_err = strategies_upper_err[strategy_index]
                ax.fill_between(
                    time_range[:real_stopping_point_index],
                    strategy_lower_err[:real_stopping_point_index],
                    strategy_upper_err[:real_stopping_point_index],
                    alpha=0.15,
                    antialiased=True,
                    color=color,
                )
                if (
                    real_stopping_point_index < time_range.shape[0]
                    and real_stopping_point_index < len(strategy_lower_err) - 1
                ):
                    ax.fill_between(
                        time_range[real_stopping_point_index:],
                        strategy_lower_err[real_stopping_point_index:],
                        strategy_upper_err[real_stopping_point_index:],
                        alpha=0.15,
                        antialiased=True,
                        color=color,
                        ls="dashed",
                    )

            # plot the curve
            ax.plot(
                time_range[:real_stopping_point_index],
                strategy_performance[:real_stopping_point_index],
                color=color,
                label=displayname,
            )
            if (
                real_stopping_point_index < time_range.shape[0]
                and real_stopping_point_index < len(strategy_performance) - 1
            ):
                ax.plot(
                    time_range[real_stopping_point_index:],
                    strategy_performance[real_stopping_point_index:],
                    color=color,
                    ls="dashed",
                )
            performance_score = round(np.mean(strategy_performance), 3)
            performance_score_std = round(np.std(strategy_performance), 3)
            print(f" | performance of {displayname}: {performance_score} (±{performance_score_std})")

        # set the axis
        cutoff_percentile: float = self.experiment.get("cutoff_percentile", 1)
        cutoff_percentile_start: float = self.experiment.get("cutoff_percentile_start", 0.01)
        ax.set_xlabel(
            f"{self.x_metric_displayname['aggregate_time']} ({cutoff_percentile_start*100}% to {cutoff_percentile*100}%)",  # noqa: E501
            fontsize="large",
        )
        ax.set_ylabel(self.y_metric_displayname["aggregate_objective"], fontsize="large")
        num_ticks = 11
        ax.set_xticks(
            np.linspace(0, y_axis_size, num_ticks),
            np.round(np.linspace(0, 1, num_ticks), 2),
        )
        ax.set_ylim(top=1.02)
        ax.set_xlim((0, y_axis_size))
        ax.legend()

    def get_x_axis_label(self, x_type: str, objective_time_keys: list):
        """Formatter to get the appropriate x-axis label depending on the x-axis type.

        Args:
            x_type: the type of a range, either time or function evaluations.
            objective_time_keys: the objective time keys used.

        Raises:
            ValueError: when an invalid ``x_type`` is given.

        Returns:
            The formatted x-axis label.
        """
        if x_type == "fevals":
            x_label = self.x_metric_displayname[x_type]
        elif x_type == "time" and len(objective_time_keys) == len(objective_time_keys_values):
            x_label = self.x_metric_displayname["time_total"]
        elif x_type == "time":
            partials = list(f"{self.x_metric_displayname[f'time_partial_{key}']}" for key in objective_time_keys)
            concatenated = ", ".join(partials)
            if len(objective_time_keys) > 2:
                concatenated = f"\n{concatenated}"
            x_label = f"Cumulative time in seconds of {concatenated}"
        else:
            raise ValueError(f"Invalid {x_type=}")
        return x_label


def is_ran_as_notebook() -> bool:  # pragma: no cover
    """Function to determine if this file is ran from an interactive notebook."""
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except ModuleNotFoundError or NameError:
        return False  # Probably standard Python interpreter


def entry_point():  #  pragma: no cover
    """Entry point function for Visualization."""
    is_notebook = is_ran_as_notebook()
    if is_notebook:
        # take the CWD one level up
        import os

        os.chdir("../")
        print(os.getcwd())
        experiment_filepath = "test_random_calculated"
        # experiment_filepath = "methodology_paper_example"
        # %matplotlib widget    # IPython magic line that sets matplotlib to widget backend for interactive
    else:
        experiment_filepath = get_args_from_cli()

    Visualize(experiment_filepath, save_figs=not is_notebook)


if __name__ == "__main__":
    entry_point()
