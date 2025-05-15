"""Visualize the results of the experiments."""

from __future__ import annotations  # for correct nested type hints e.g. list[str], tuple[dict, str]

import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap, rgb2hex

from autotuning_methodology.baseline import (
    Baseline,
    ExecutedStrategyBaseline,
    RandomSearchCalculatedBaseline,
    RandomSearchSimulatedBaseline,
)
from autotuning_methodology.curves import Curve, CurveBasis
from autotuning_methodology.experiments import execute_experiment, get_args_from_cli
from autotuning_methodology.report_experiments import (
    get_aggregation_data,
    get_aggregation_data_key,
    get_strategies_aggregated_performance,
)
from autotuning_methodology.searchspace_statistics import SearchspaceStatistics

# The kernel information per device and device information for visualization purposes
marker_variatons = ["v", "s", "*", "1", "2", "d", "P", "X"]

remove_from_gpus_label = ""
remove_from_applications_label = " milo"
remove_from_searchspace_label = " milo"

# total set of objective time keys
objective_time_keys_values = ["compilation", "benchmark", "framework", "search_algorithm", "validation"]


def get_colors(strategies: list[dict], scale_margin_left=0.4, scale_margin_right=0.15):
    """Function to get the colors for each of the strategies."""
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    main_colors = ["Blues", "Greens", "Reds", "Purples", "Greys"]
    main_color_counter = 0
    strategy_parents = defaultdict(list)

    # get the dictionary of parents with the index of their child strategies
    for strategy_index, strategy in enumerate(strategies):
        if "color_parent" in strategy:
            parent = strategy["color_parent"]
            strategy_parents[parent].append(strategy_index)
    if len(strategy_parents) == 0:
        return default_colors
    if len(strategy_parents) > len(main_colors):
        raise ValueError(f"Can't use parent colors with more than {len(main_colors)} strategies")

    def get_next_single_color_list(main_color_counter: int, num_colors: int):
        colorname = main_colors[main_color_counter]
        cmap = get_cmap(colorname)
        spacing = np.linspace(scale_margin_left, 1 - scale_margin_right, num=num_colors) if num_colors > 1 else [0.5]
        colormap = cmap(spacing)
        color_list = [rgb2hex(c) for c in colormap]
        return color_list

    parented_colors = dict()
    colors = list()
    for strategy_index, strategy in enumerate(strategies):
        name = strategy["name"]
        if name in strategy_parents:
            children_index = strategy_parents[name]
            if len(children_index) == 3:
                warnings.warn(f"Color parent '{name}' has three children, check if lines in plot are visually distinct")
            if len(children_index) > 3:
                raise ValueError(
                    f"Color parent '{name}' should not have more than three children to maintain visual distinction"
                )
            color_scale = get_next_single_color_list(main_color_counter, len(children_index) + 1)
            main_color_counter += 1
            parented_colors[name] = dict()
            for index, child_index in enumerate(children_index):
                parented_colors[name][child_index] = color_scale[(len(children_index) - 1) - index]
            color = color_scale[len(children_index)]
        else:
            if "color_parent" in strategy:
                parent = strategy["color_parent"]
                color = parented_colors[parent][strategy_index]
            else:
                color = get_next_single_color_list(main_color_counter, 1)[0]
                main_color_counter += 1
        colors.append(color)
    return colors


class Visualize:
    """Class for visualization of experiments."""

    x_metric_displayname = dict(
        {
            "fevals": "Number of function evaluations used",
            "time_total": "Total time in seconds",
            "aggregate_time": "Relative time to cutoff point",
            "time_partial_framework_time": "framework time",
            "time_partial_framework": "framework time",
            "time_partial_strategy_time": "strategy time",
            "time_partial_search_algorithm": "strategy time",
            "time_partial_compile_time": "compile time",
            "time_partial_compilation": "compile time",
            "time_partial_benchmark_time": "kernel runtime",
            "time_partial_times": "kernel runtime",
            "time_partial_runtimes": "kernel runtime",
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
            "time": "Best-found kernel time in milliseconds",
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

    def __init__(
        self,
        experiment_filepath: str,
        save_figs=True,
        save_extra_figs=False,
        continue_after_comparison=False,
        compare_extra_baselines=False,
        use_strategy_as_baseline=None,
    ) -> None:
        """Initialization method for the Visualize class.

        Args:
            experiment_filepath: the path to the experiment-filename.json to run.
            save_figs: whether to save the figures to file, if not, displays in a window. Defaults to True.
            save_extra_figs: whether to save split times and baseline comparisons figures to file. Defaults to False.
            continue_after_comparison: whether to continue plotting after processing comparisons. Defaults to False.
            compare_extra_baselines: whether to include additional baselines for comparison. Defaults to False.
            use_strategy_as_baseline: whether to use an executed strategy as the baseline. WARNING: likely destroys comparability. Defaults to None.

        Raises:
            ValueError: on various invalid inputs.
        """
        # # silently execute the experiment
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        self.experiment, self.all_experimental_groups, self.searchspace_statistics, self.results_descriptions = (
            execute_experiment(experiment_filepath, profiling=False)
        )
        experiment_folder: Path = self.experiment["parent_folder_absolute_path"]
        assert isinstance(experiment_folder, Path)
        self.plot_filename_prefix = experiment_folder.joinpath("run", "generated_graphs")
        print("\n")
        print("Visualizing")

        # preparing filesystem
        if save_figs or save_extra_figs:
            Path(self.plot_filename_prefix).mkdir(exist_ok=True)

        # search strategies are search methods defined in experiments setup file
        # self.all_experimental_groups are all combinations of gpu+application+search method that got executed
        self.strategies = self.experiment["search_strategies"]
        # settings
        cutoff_percentile: float = self.experiment["statistics_settings"]["cutoff_percentile"]
        cutoff_percentile_start: float = self.experiment["statistics_settings"]["cutoff_percentile_start"]
        cutoff_type: str = self.experiment["statistics_settings"]["cutoff_type"]
        assert cutoff_type == "fevals" or cutoff_type == "time", f"cutoff_type != 'fevals' or 'time', is {cutoff_type}"
        time_resolution: float = self.experiment["visualization_settings"]["resolution"]
        if int(time_resolution) != time_resolution:
            raise ValueError(f"The resolution must be an integer, yet is {time_resolution}.")
        time_resolution = int(time_resolution)
        objective_time_keys: list[str] = self.experiment["statistics_settings"]["objective_time_keys"]

        # plot settings
        plots: list[dict] = self.experiment["visualization_settings"]["plots"]
        compare_baselines: bool = self.experiment["visualization_settings"]["compare_baselines"]
        compare_split_times: bool = self.experiment["visualization_settings"]["compare_split_times"]
        confidence_level: float = self.experiment["visualization_settings"]["confidence_level"]
        self.colors = get_colors(
            self.strategies,
            scale_margin_left=self.experiment["visualization_settings"].get("color_parent_scale_margin_left", 0.4),
            scale_margin_right=self.experiment["visualization_settings"].get("color_parent_scale_margin_right", 0.1),
        )
        self.plot_skip_strategies: list[str] = list()
        if use_strategy_as_baseline is not None:
            self.plot_skip_strategies.append(use_strategy_as_baseline)

        # visualize
        aggregation_data = get_aggregation_data(
            experiment_folder,
            self.experiment,
            self.searchspace_statistics,
            self.strategies,
            self.results_descriptions,
            cutoff_percentile,
            cutoff_percentile_start,
            confidence_level,
            time_resolution,
            use_strategy_as_baseline,
        )

        # plot per searchspace
        for gpu_name in self.experiment["experimental_groups_defaults"]["gpus"]:
            for application_name in self.experiment["experimental_groups_defaults"]["applications_names"]:
                print(f" | visualizing optimization of {application_name} for {gpu_name}")
                title = f"{application_name} on {gpu_name}"
                title = title.replace("_", " ")

                # unpack the aggregation data
                random_baseline, strategies_curves, searchspace_stats, time_range, fevals_range = aggregation_data[
                    get_aggregation_data_key(gpu_name=gpu_name, application_name=application_name)
                ]

                # baseline_time_interpolated = np.linspace(mean_feval_time, cutoff_point_time, time_resolution)
                # baseline = get_random_curve(cutoff_point_fevals, sorted_times, time_resolution)

                # compare baselines
                if compare_baselines is True:
                    self.plot_baselines_comparison(
                        time_range,
                        searchspace_stats,
                        objective_time_keys,
                        strategies_curves=strategies_curves,
                        confidence_level=confidence_level,
                        title=title,
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

                # set additional baselines for comparison
                baselines_extra: list[Baseline] = []
                if compare_extra_baselines is True:
                    baselines_extra.append(RandomSearchSimulatedBaseline(searchspace_stats, repeats=1000))
                    baselines_extra.append(RandomSearchCalculatedBaseline(searchspace_stats, include_nan=True))
                    # baselines_extra.append(
                    #     ExecutedStrategyBaseline(
                    #         searchspace_stats,
                    #         strategy=(
                    #             baseline_executed_strategy
                    #             if baseline_executed_strategy is not None
                    #             else strategies_curves[0]
                    #         ),
                    #         confidence_level=confidence_level,
                    #     )
                    # )

                for plot in plots:
                    # get settings
                    scope: str = plot["scope"]
                    if scope != "searchspace":
                        continue
                    style: str = plot["style"]
                    plot_x_value_types: list[str] = plot["x_axis_value_types"]
                    plot_y_value_types: list[str] = plot["y_axis_value_types"]

                    # visualize the results
                    for x_type in plot_x_value_types:
                        if x_type == "fevals":
                            x_axis_range = fevals_range
                        elif x_type == "time":
                            x_axis_range = time_range
                        else:
                            raise NotImplementedError(f"X-axis type '{x_type}' not supported for scope '{plot}'")

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
                                style,
                                x_type,
                                y_type,
                                axs[index],
                                searchspace_stats,
                                strategies_curves,
                                x_axis_range,
                                self.experiment["visualization_settings"],
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
                            filename_path = Path(self.plot_filename_prefix) / f"{title}_{x_type}".replace(" ", "_")
                            fig.savefig(filename_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
                            print(f"Figure saved to {filename_path}")
                        else:
                            plt.show()

        # plot per searchstrategy
        for plot in plots:
            # get settings
            scope: str = plot["scope"]
            style: str = plot["style"]
            if scope != "search_strategy":
                continue
            if style != "heatmap" and style != "compare_heatmaps":
                raise NotImplementedError(
                    f"Scope {scope} currently only supports 'heatmap' or 'compare_heatmaps' as a style, not {style}"
                )
            plot_x_value_types: list[str] = plot["x_axis_value_types"]
            plot_y_value_types: list[str] = plot["y_axis_value_types"]
            assert len(plot_x_value_types) == 1
            assert len(plot_y_value_types) == 1
            x_type = plot_x_value_types[0]
            y_type = plot_y_value_types[0]
            bins = plot.get("bins", 10)
            vmin = plot.get("vmin", -15.0)  # color range lower limit
            vmax = plot.get("vmax", 1.0)  # color range upper limit
            cmin = plot.get("cmin", vmin)  # colorbar lower limit
            cmax = plot.get("cmax", vmax)  # colorbar upper limit
            cnum = plot.get("cnum", 5)  # number of ticks on the colorbar
            divide_train_test_axis = plot.get(
                "divide_train_test_axis", False
            )  # whether to add visual indication for train/test split
            divide_train_test_after_num = plot.get(
                "divide_train_test_after_num", False
            )  # where to add the visual indication for train/test split
            include_y_labels = plot.get("include_y_labels", None)
            include_colorbar = plot.get("include_colorbar", True)
            if vmin != -15.0:
                warnings.warn(
                    f"Careful: VMin has been changed from -15.0 to {vmin}. This breaks visual comparison compatiblity with plots that do not have the same VMin. Maybe use cmin instead?."
                )
            if vmax != 1.0:
                warnings.warn(
                    f"Careful: VMax has been changed from 1.0 to {vmax}. This breaks visual comparison compatiblity with plots that do not have the same VMax. Maybe use cmax instead?"
                )
            if cmin < vmin:
                raise ValueError(
                    f"Colorbar minimum can't be lower than the minimum value of the heatmap: {cmin} < {vmin}"
                )
            if cmax > vmax:
                raise ValueError(
                    f"Colorbar maximum can't be higher than the maximum value of the heatmap: {cmax} > {vmax}"
                )

            # set the colormap
            def norm_color_val(v):
                """Normalize a color value to fit in the 0-1 range."""
                return (v - vmin) / (vmax - vmin)

            cmap = LinearSegmentedColormap.from_list(
                "my_colormap",
                [
                    (norm_color_val(-15.0), "black"),
                    (norm_color_val(-4.0), "red"),
                    (norm_color_val(-1.0), "orange"),
                    (norm_color_val(0.0), "yellow"),
                    (norm_color_val(1.0), "green"),
                ],
            )

            # collect and plot the data for each search strategy
            data_collected: dict[str, list[tuple]] = defaultdict(list)
            for strategy in self.strategies:
                strategy_name = strategy["name"]
                strategy_displayname = strategy["display_name"]
                assert (
                    sum([1 for s in self.strategies if s["name"] == strategy_name]) == 1
                ), f"Strategy name '{strategy_name}' is not unqiue"

                # get the data from the collected aggregated data
                for gpu_name in self.experiment["experimental_groups_defaults"]["gpus"]:
                    for application_name in self.experiment["experimental_groups_defaults"]["applications_names"]:
                        # unpack the aggregation data
                        random_baseline, strategies_curves, searchspace_stats, time_range, fevals_range = (
                            aggregation_data[
                                get_aggregation_data_key(gpu_name=gpu_name, application_name=application_name)
                            ]
                        )

                        # get the data
                        dist = searchspace_stats.objective_performances_total_sorted
                        for _, strategy_curve in enumerate(strategies_curves):
                            if strategy_name != strategy_curve.name:
                                continue
                            # get the real and fictional performance curves
                            (
                                _,
                                x_axis_range_real,
                                curve_real,
                                _,
                                _,
                                x_axis_range_fictional,
                                curve_fictional,
                                _,
                                _,
                            ) = strategy_curve.get_curve_over_time(
                                time_range, dist=dist, confidence_level=confidence_level
                            )
                            # combine the real and fictional parts to get the full curve
                            combine = x_axis_range_fictional.ndim > 0
                            x_axis_range = (
                                np.concatenate([x_axis_range_real, x_axis_range_fictional])
                                if combine
                                else x_axis_range_real
                            )
                            assert np.array_equal(
                                time_range, x_axis_range, equal_nan=True
                            ), "time_range != x_axis_range"
                            curve = np.concatenate([curve_real, curve_fictional]) if combine else curve_real
                            # get the standardised curves and write them to the collector
                            curve: np.ndarray = random_baseline.get_standardised_curves(
                                time_range, [curve], x_type="time"
                            )[0]
                            score = np.mean(curve, axis=0)
                            curve_binned = np.array_split(curve, bins)
                            score_binned = [np.mean(c, axis=0) for c in curve_binned]

                            # set the data
                            gpu_display_name = str(gpu_name).replace("_", " ")
                            application_display_name = str(application_name).replace("_", " ").capitalize()
                            data_collected[strategy_name].append(
                                tuple([gpu_display_name, application_display_name, score, score_binned])
                            )
            if style == "heatmap":
                for strategy in self.strategies:
                    strategy_name = strategy["name"]
                    strategy_displayname = strategy["display_name"]
                    strategy_data = data_collected[strategy_name]

                    # get the performance per selected type in an array
                    plot_data = np.stack(np.array([t[2] for t in strategy_data]))
                    cutoff_percentile: float = self.experiment["statistics_settings"].get("cutoff_percentile", 1.0)
                    cutoff_percentile_start: float = self.experiment["statistics_settings"].get(
                        "cutoff_percentile_start", 0.01
                    )
                    label_data = {
                        "gpus": (
                            list(dict.fromkeys([t[0].replace(remove_from_gpus_label, "") for t in strategy_data])),
                            "GPUs",
                        ),
                        "applications": (
                            list(
                                dict.fromkeys([t[1].replace(remove_from_applications_label, "") for t in strategy_data])
                            ),
                            "Applications",
                        ),
                        "searchspaces": (
                            list(
                                dict.fromkeys(
                                    [
                                        f"{t[1]} on\n{t[0]}".replace(remove_from_searchspace_label, "")
                                        for t in strategy_data
                                    ]
                                )
                            ),
                            "Searchspaces",
                        ),
                        "time": (
                            np.round(np.linspace(0.0, 1.0, bins), 2),
                            f"Fraction of time between {cutoff_percentile_start*100}% and {cutoff_percentile*100}%",
                        ),
                    }
                    x_ticks = label_data[x_type][0]
                    y_ticks = label_data[y_type][0]
                    figsize = None
                    if (x_type == "time" and y_type == "searchspaces") or (
                        x_type == "searchspaces" and y_type == "time"
                    ):
                        plot_data: np.ndarray = np.stack(np.array([t[3] for t in strategy_data]))
                        if x_type == "searchspaces":
                            plot_data = plot_data.transpose()
                        figsize = (9, 5)
                    elif (x_type == "gpus" and y_type == "applications") or (
                        y_type == "gpus" and x_type == "applications"
                    ):
                        plot_data = np.reshape(
                            plot_data, (len(label_data["gpus"][0]), len(label_data["applications"][0]))
                        )
                        if x_type == "gpus":
                            plot_data = np.transpose(plot_data)
                        figsize = (5, 3.5)
                    else:
                        raise NotImplementedError(
                            f"Heatmap has not yet been implemented for {x_type}, {y_type}. Submit an issue to request it."
                        )

                    # validate the data is within the vmin-vmax range and visible colorbar range
                    outside_range = np.where(np.logical_or(plot_data < vmin, plot_data > vmax))
                    assert (
                        len(outside_range[0]) == 0 and len(outside_range[1]) == 0
                    ), f"There are values outside of the range ({vmin}, {vmax}): {plot_data[outside_range]} ({outside_range} for strategy {strategy_displayname})"
                    outside_visible_range = np.where(np.logical_or(plot_data < cmin, plot_data > cmax))
                    if not (len(outside_visible_range[0]) == 0 and len(outside_visible_range[1]) == 0):
                        warnings.warn(
                            f"There are values outside of the visible colorbar range ({cmin}, {cmax}): {plot_data[outside_visible_range]} ({outside_visible_range})"
                        )

                    # set up the plot
                    fig, axs = plt.subplots(
                        ncols=1, figsize=figsize, dpi=300
                    )  # if multiple subplots, pass the axis to the plot function with axs[0] etc.
                    if not hasattr(axs, "__len__"):
                        axs = [axs]
                    title = f"Performance of {strategy_displayname} over {'+'.join(plot_x_value_types)},{'+'.join(plot_y_value_types)}"
                    fig.canvas.manager.set_window_title(title)
                    if not save_figs:
                        fig.suptitle(title)

                    # plot the heatmap
                    axs[0].set_xlabel(plot.get("xlabel", label_data[x_type][1]))
                    axs[0].set_xticks(ticks=np.arange(len(x_ticks)), labels=x_ticks, rotation=0)
                    if include_y_labels is True or None:
                        axs[0].set_ylabel(plot.get("ylabel", label_data[y_type][1]))
                        axs[0].set_yticks(ticks=np.arange(len(y_ticks)), labels=y_ticks)
                    if include_y_labels is True:
                        # axs[0].yaxis.set_label_position("right")
                        axs[0].yaxis.tick_right()
                    elif include_y_labels is False:
                        axs[0].set_yticks(ticks=np.arange(len(y_ticks)))
                        axs[0].tick_params(labelleft=False)
                    hm = axs[0].imshow(
                        plot_data,
                        vmin=vmin,
                        vmax=vmax,
                        cmap=cmap,
                        interpolation="nearest",
                        aspect="auto",
                        # extent=[-0.5, plot_data.shape[1] + 0.5, -0.5, plot_data.shape[0] + 0.5],
                    )
                    if divide_train_test_axis is not False:
                        # axs[0].set_ylim(plot_data.shape[0] - 0.5, -0.5)  # Ensure correct y-axis limits
                        if x_type == divide_train_test_axis.lower():
                            # add the vertical line to the x-axis
                            axs[0].axvline(
                                x=divide_train_test_after_num - 0.5, color="black", linestyle="--", linewidth=0.8
                            )
                            # add train and test texts to either side of the x-label
                            axs[0].text(
                                x=divide_train_test_after_num - 0.5,
                                y=-0.5,
                                s="train",
                                ha="center",
                                va="top",
                                fontsize=10,
                            )
                            axs[0].text(
                                x=divide_train_test_after_num - 0.5,
                                y=plot_data.shape[0] - 0.5,
                                s="test",
                                ha="center",
                                va="bottom",
                                fontsize=10,
                            )
                        elif y_type == divide_train_test_axis.lower():
                            # add the horizontal line to the y-axis
                            axs[0].axhline(
                                y=divide_train_test_after_num - 0.5, color="black", linestyle="--", linewidth=0.8
                            )
                            if include_y_labels is not False:
                                # add train and test texts to either side of the y-label
                                x_loc = -0.02
                                y_center = 0.5
                                text = "train"
                                axs[0].text(
                                    x=x_loc,
                                    y=y_center + 0.25 + (len(text) * 0.01),
                                    s=text,
                                    color="grey",
                                    fontsize=8.5,
                                    ha="center",
                                    va="center",
                                    rotation=90,
                                    transform=axs[0].transAxes,
                                )
                                text = "test"
                                axs[0].text(
                                    x=x_loc,
                                    y=y_center - 0.25 - (len(text) * 0.01),
                                    s=text,
                                    color="grey",
                                    fontsize=8.5,
                                    ha="center",
                                    va="center",
                                    rotation=90,
                                    transform=axs[0].transAxes,
                                )
                        else:
                            raise ValueError(f"{divide_train_test_axis=} not in x ({x_type}) or y ({y_type}) axis")

                    # plot the colorbar
                    if include_colorbar is True:
                        cbar = fig.colorbar(hm)
                        if cmin != vmin or cmax != vmax:
                            cbar.set_ticks(np.linspace(cmin, cmax, num=cnum))  # set colorbar limits
                            cbar.ax.set_ylim(cmin, cmax)  # adjust visible colorbar limits
                        # cbar.set_label("Performance relative to baseline (0.0) and optimum (1.0)")
                        cbar.set_label("Performance score")

                    # keep only non-overlapping ticks
                    max_ticks = 15
                    if len(x_ticks) > max_ticks:
                        indices = np.linspace(0, len(x_ticks) - 1, max_ticks).round()
                        hide_tick = np.isin(np.arange(len(x_ticks)), indices, invert=True, assume_unique=True)
                        for i, t in enumerate(axs[0].xaxis.get_ticklabels()):
                            if hide_tick[i]:
                                t.set_visible(False)
                    if len(y_ticks) > max_ticks:
                        indices = np.linspace(0, len(y_ticks) - 1, max_ticks).round()
                        hide_tick = np.isin(np.arange(len(y_ticks)), indices, invert=True, assume_unique=True)
                        for i, t in enumerate(axs[0].yaxis.get_ticklabels()):
                            if hide_tick[i]:
                                t.set_visible(False)

                    # finalize the figure and save or display it
                    fig.tight_layout()
                    if save_figs:
                        filename_path = (
                            Path(self.plot_filename_prefix)
                            / f"{strategy_name}_heatmap_{'_'.join(plot_x_value_types)}_{'_'.join(plot_y_value_types)}"
                        )
                        fig.savefig(filename_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
                        print(f"Figure saved to {filename_path}")
                    else:
                        plt.show()
            elif style == "compare_heatmaps":
                comparisons = plot["comparison"]

                raise NotImplementedError("Still a work in progress")

                # set up the plot
                fig, axs = plt.subplots(
                    ncols=1, figsize=(9, 6), dpi=300
                )  # if multiple subplots, pass the axis to the plot function with axs[0] etc.
                if not hasattr(axs, "__len__"):
                    axs = [axs]
                # title = f"Performance of {strategy_displayname} over {'+'.join(plot_x_value_types)},{'+'.join(plot_y_value_types)}"
                # fig.canvas.manager.set_window_title(title)
                # if not save_figs:
                # fig.suptitle(title)

                for comparison in comparisons:
                    strategy_names = comparisons["strategies"]
                    strategies = [s for s in self.strategies if s["name"]]
                    # for strategy in strategies:
                    strategy_displayname = strategy["display_name"]
                    strategy_data = data_collected[strategy_name]

                    # get the performance per selected type in an array
                    plot_data = np.stack(np.array([t[2] for t in strategy_data]))
                    cutoff_percentile: float = self.experiment["statistics_settings"].get("cutoff_percentile", 1)
                    cutoff_percentile_start: float = self.experiment["statistics_settings"].get(
                        "cutoff_percentile_start", 0.01
                    )
                    label_data = {
                        "gpus": (
                            list(dict.fromkeys([t[0].replace(remove_from_gpus_label, "") for t in strategy_data])),
                            "GPUs",
                        ),
                        "applications": (
                            list(
                                dict.fromkeys([t[1].replace(remove_from_applications_label, "") for t in strategy_data])
                            ),
                            "Applications",
                        ),
                        "searchspaces": (
                            list(
                                dict.fromkeys(
                                    [
                                        f"{t[1]} on\n{t[0]}".replace(remove_from_searchspace_label, "")
                                        for t in strategy_data
                                    ]
                                )
                            ),
                            "Searchspaces",
                        ),
                        "time": (
                            np.round(np.linspace(0.0, 1.0, bins), 2),
                            f"Fraction of time between {cutoff_percentile_start*100}% and {cutoff_percentile*100}%",
                        ),
                    }
                    x_ticks = label_data[x_type][0]
                    y_ticks = label_data[y_type][0]
                    if (x_type == "time" and y_type == "searchspaces") or (
                        x_type == "searchspaces" and y_type == "time"
                    ):
                        plot_data: np.ndarray = np.stack(np.array([t[3] for t in strategy_data]))
                        if x_type == "searchspaces":
                            plot_data = plot_data.transpose()
                    elif (x_type == "gpus" and y_type == "applications") or (
                        y_type == "gpus" and x_type == "applications"
                    ):
                        plot_data = np.reshape(
                            plot_data, (len(label_data["gpus"][0]), len(label_data["applications"][0]))
                        )
                        if x_type == "gpus":
                            plot_data = np.transpose(plot_data)
                    else:
                        raise NotImplementedError(
                            f"Heatmap has not yet been implemented for {x_type}, {y_type}. Submit an issue to request it."
                        )

                    # validate the data
                    outside_range = np.where(np.logical_or(plot_data < vmin, plot_data > vmax))
                    assert (
                        len(outside_range[0]) == 0 and len(outside_range[1]) == 0
                    ), f"There are values outside of the range ({vmin}, {vmax}): {plot_data[outside_range]} ({outside_range} for strategy {strategy_displayname})"
            else:
                raise NotImplementedError(f"Invalid {style=}")

        # plot the aggregated searchspaces
        for plot in plots:
            # get settings
            scope: str = plot["scope"]
            style: str = plot["style"]
            vmin: float = plot.get("vmin", None)  # visual range lower limit
            if scope != "aggregate":
                continue
            if style != "line":
                raise NotImplementedError(f"{scope} currently only supports 'line' as a style, not {style}")
            # plot the aggregation
            if continue_after_comparison or not (compare_baselines or compare_split_times):
                fig, axs = plt.subplots(
                    ncols=1, figsize=(7.5, 5), dpi=300
                )  # if multiple subplots, pass the axis to the plot function with axs[0] etc.
                if not hasattr(axs, "__len__"):
                    axs = [axs]
                title = f"""Aggregated Data\napplications:
                        {', '.join(self.experiment['experimental_groups_defaults']['applications_names'])}\nGPUs: {', '.join(self.experiment['experimental_groups_defaults']['gpus'])}"""
                fig.canvas.manager.set_window_title(title)
                if not save_figs:
                    fig.suptitle(title)

                # finalize the figure and save or display it
                lowest_real_y_value = self.plot_strategies_aggregated(
                    axs[0], aggregation_data, visualization_settings=self.experiment["visualization_settings"], plot_settings=plot
                )
                if vmin is not None:
                    if isinstance(vmin, (int, float)):
                        axs[0].set_ylim(bottom=vmin)
                    elif vmin == "real":
                        axs[0].set_ylim(bottom=lowest_real_y_value - (abs(lowest_real_y_value)+1.0) * 0.02)
                    else:
                        raise NotImplementedError(f"{vmin=} not implemented")
                fig.tight_layout()
                if save_figs:
                    filename_path = Path(self.plot_filename_prefix) / "aggregated"
                    fig.savefig(filename_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
                    print(f"Figure saved to {filename_path}")
                else:
                    plt.show()

    def plot_baselines_comparison(
        self,
        time_range: np.ndarray,
        searchspace_stats: SearchspaceStatistics,
        objective_time_keys: list,
        strategies_curves: list[Curve],
        confidence_level: float,
        title: str = None,
        save_fig=False,
    ):
        """Plots a comparison of baselines on a time range.

        Optionally also compares against strategies listed in strategies_curves.

        Args:
            time_range: range of time to plot on.
            searchspace_stats: Searchspace statistics object.
            objective_time_keys: objective time keys.
            strategies_curves: the strategy curves to draw in the plot.
            confidence_level: the confidence interval used for the confidence / prediction interval.
            title: the title for this plot, if not given, a title is generated. Defaults to None.
            save_fig: whether to save the resulting figure to file. Defaults to False.
        """
        dist = searchspace_stats.objective_performances_total_sorted
        plt.figure(figsize=(9, 7), dpi=300)

        # list the baselines to test
        baselines: list[Baseline] = list()
        # baselines.append(
        #     RandomSearchCalculatedBaseline(searchspace_stats, include_nan=False, time_per_feval_operator="median")
        # )
        # baselines.append(RandomSearchCalculatedBaseline(searchspace_stats, time_per_feval_operator="mean"))
        # baselines.append(
        #     RandomSearchCalculatedBaseline(
        #         searchspace_stats, include_nan=True, time_per_feval_operator="median_per_feval"
        #     )
        # )
        baselines.append(
            ExecutedStrategyBaseline(
                searchspace_stats, strategy=strategies_curves[0], confidence_level=confidence_level
            )
        )

        # plot random baseline implementations
        for baseline in baselines:
            timecurve = baseline.get_curve_over_time(time_range)
            print(f"{baseline.label}: {timecurve[-1]}")
            plt.plot(time_range, timecurve, label=baseline.label)

        # plot normal strategies
        for strategy_curve in strategies_curves:
            if strategy_curve.name in self.plot_skip_strategies:
                continue
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
            filename_path = Path(self.plot_filename_prefix) / f"{title}_baselines".replace(" ", "_")
            plt.savefig(filename_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
            print(f"Figure saved to {filename_path}")
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
        for line in lines:
            if isinstance(line, Curve) and line.name in self.plot_skip_strategies:
                lines.remove(line)

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
            filename_path = Path(self.plot_filename_prefix) / f"{title}_split_times_{x_type}".replace(" ", "_")
            plt.savefig(filename_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
            print(f"Figure saved to {filename_path}")
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
            for _ in range((len(strategies_curves) - len(self.plot_skip_strategies)) + 1)
        )
        for objective_time_key in objective_time_keys:
            data_dict[objective_time_key] = np.full((len(strategies_curves)), np.NaN)
        for strategy_index, strategy_curve in enumerate(strategies_curves):
            if strategy_curve.name in self.plot_skip_strategies:
                continue
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
            filename_path = Path(self.plot_filename_prefix) / f"{title}_split_times_bar".replace(" ", "_")
            plt.savefig(filename_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
            print(f"Figure saved to {filename_path}")
        else:
            plt.show()

    def plot_strategies(
        self,
        style: str,
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
            style: the style of plot, either 'line' or 'scatter'.
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

        def normalize(curve):
            """Min-max normalization with median as min and absolute optimum as max."""
            if curve is None:
                return None
            return (curve - median) / (absolute_optimum - median)

        def normalize_multiple(curves: list) -> tuple:
            """Normalize multiple curves at once."""
            return tuple(normalize(curve) for curve in curves)

        # plot the absolute optimum
        absolute_optimum_y_value = absolute_optimum if y_type == "absolute" or style == "scatter" else 1
        absolute_optimum_label = (
            "Absolute optimum ({})".format(round(absolute_optimum, 3)) if y_type == "absolute" else "Absolute optimum"
        )
        ax.axhline(absolute_optimum_y_value, c="black", ls="-.", label=absolute_optimum_label)

        # plot baseline
        if baseline_curve is not None:
            if y_type == "baseline":
                ax.axhline(0, label="baseline trajectory", color="black", ls="--")
            elif y_type == "normalized" or y_type == "baseline" or y_type == "absolute":
                baseline = baseline_curve.get_curve(x_axis_range, x_type)
                if absolute_optimum in baseline:
                    raise ValueError(
                        f"The optimum {absolute_optimum} is in the baseline, this will cause zero division problems"
                    )
                    # cut_at_index = np.argmax(baseline == absolute_optimum)
                    # baseline = baseline[:cut_at_index]
                    # x_axis_range = x_axis_range[:cut_at_index]
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
                if strategy["name"] not in self.plot_skip_strategies:
                    self.plot_skip_strategies.append(strategy["name"])
                continue

            # get the data
            color = self.colors[strategy_index]
            label = f"{strategy['display_name']}"
            strategy_curve = strategies_curves[strategy_index]
            if strategy_curve.name in self.plot_skip_strategies:
                continue

            # get the plot data
            if style == "scatter":
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
                if x_axis_range_fictional.ndim > 0 and x_axis_range_fictional.shape[0] > 0:
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
            # multiplier = 0.99 if self.minimization else 1.01
            # ax.set_ylim(absolute_optimum * multiplier, median)
            # ax.set_ylim(1.0)
            pass
        # elif y_type == 'normalized':
        #     ax.set_ylim((0.0, 1 + normalized_ylim_margin))
        elif y_type == "baseline":
            ax.set_ylim((min(-normalized_ylim_margin, ylim_min - normalized_ylim_margin), 1 + normalized_ylim_margin))

    def plot_strategies_aggregated(
        self,
        ax: plt.Axes,
        aggregation_data,
        visualization_settings: dict = {},
        plot_settings: dict = {},
    ) -> float:
        """Plots all optimization strategies combined accross search spaces.

        Args:
            ax: the axis to plot on.
            aggregation_data: the aggregated data from the various searchspaces.
            visualization_settings: dictionary of additional visualization settings.
            plot_settings: dictionary of additional visualization settings related to this particular plot.

        Returns:
            The lowest performance value of the real stopping point for all strategies.
        """
        # plot the random baseline and absolute optimum
        ax.axhline(0, label="Calculated baseline", c="black", ls=":")
        ax.axhline(1, label="Absolute optimum", c="black", ls="-.")

        # get the relative aggregated performance for each strategy
        confidence_level: float = visualization_settings.get("confidence_level", 0.95)
        (
            strategies_performance,
            strategies_lower_err,
            strategies_upper_err,
            strategies_real_stopping_point_fraction,
        ) = get_strategies_aggregated_performance(list(aggregation_data.values()), confidence_level)

        # get the relevant plot settings
        cutoff_percentile: float = self.experiment["statistics_settings"].get("cutoff_percentile", 1)
        cutoff_percentile_start: float = self.experiment["statistics_settings"].get("cutoff_percentile_start", 0.01)
        xlabel = plot_settings.get("xlabel", f"{self.x_metric_displayname['aggregate_time']} ({cutoff_percentile_start*100}% to {cutoff_percentile*100}%)") # noqa: E501
        ylabel = plot_settings.get("ylabel", self.y_metric_displayname["aggregate_objective"])
        tmin = plot_settings.get("tmin", 1.0)

        # setup the plot
        y_axis_size = strategies_performance[0].shape[0]
        time_range = np.arange(y_axis_size)
        plot_errors = True
        lowest_real_y_value = 0.0
        print("\n-------")
        print("Quantification of aggregate performance across all search spaces:")

        # get the highest real_stopping_point_index, adjust y_axis_size and time_range if necessary
        real_stopping_point_indices = [min(round(strategies_real_stopping_point_fraction[strategy_index] * time_range.shape[0]) + 1, time_range.shape[0]) for strategy_index in range(len(strategies_performance))]  # noqa: E501
        real_stopping_point_index_max = max(real_stopping_point_indices)
        if tmin == "real":
            # stop the time at the largest real stopping point
            y_axis_size = min(real_stopping_point_index_max, y_axis_size)
            time_range = np.arange(y_axis_size)
        elif tmin < 1.0:
            # stop the time at the given tmin
            y_axis_size = y_axis_size * tmin
            time_range = np.arange(y_axis_size)
        elif tmin > 1.0:
            raise ValueError(f"Invalid {tmin=}, must be between 0.0 and 1.0 or 'real'")

        # adjust the xlabel if necessary
        if tmin == "real" and not "xlabel" in plot_settings:
            xlabel = "Relative time until the last strategy stopped"

        # plot each strategy
        for strategy_index, strategy_performance in enumerate(strategies_performance):
            if self.strategies[strategy_index]["name"] in self.plot_skip_strategies:
                continue
            displayname = self.strategies[strategy_index]["display_name"]
            color = self.colors[strategy_index]
            real_stopping_point_index = real_stopping_point_indices[strategy_index]
            if real_stopping_point_index <= 1:
                warnings.warn(f"Stopping point index for {displayname} is at {real_stopping_point_index}")
                continue

            # calculate the lowest real_y_value
            lowest_real_y_value = min(
                lowest_real_y_value,
                (
                    strategy_performance[real_stopping_point_index]
                    if real_stopping_point_index < time_range.shape[0]
                    else strategy_performance[time_range.shape[0] - 1]
                ),
            )
            assert isinstance(lowest_real_y_value, (int, float)), f"Invalid {lowest_real_y_value=}"

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
                        time_range[real_stopping_point_index-1:y_axis_size],
                        strategy_lower_err[real_stopping_point_index-1:y_axis_size],
                        strategy_upper_err[real_stopping_point_index-1:y_axis_size],
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
                    time_range[real_stopping_point_index-1:y_axis_size],
                    strategy_performance[real_stopping_point_index-1:y_axis_size],
                    color=color,
                    ls="dashed",
                )
            performance_score = round(np.mean(strategy_performance), 3)
            performance_score_std = round(np.std(strategy_performance), 3)
            print(f" | performance of {displayname}: {performance_score} (±{performance_score_std})")

        # set the axis labels
        ax.set_xlabel(xlabel, fontsize="large")
        ax.set_ylabel(ylabel, fontsize="large")

        # set the ticks
        if tmin == "real":
            ax.set_xticks([], [])
        else:
            num_ticks = 11
            ax.set_xticks(
                np.linspace(0, y_axis_size, num_ticks),
                np.round(np.linspace(0, tmin, num_ticks), 2),
            )

        # set the limits and legend
        ax.set_ylim(top=1.02)
        ax.set_xlim((0, y_axis_size-1))
        ax.legend()
        return lowest_real_y_value

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
