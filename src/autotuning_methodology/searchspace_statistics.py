"""Code for obtaining search space statistics."""

from __future__ import annotations  # for correct nested type hints e.g. list[str], tuple[dict, str]

import json
from math import ceil, floor
from pathlib import Path

import numpy as np

from autotuning_methodology.validators import is_invalid_objective_performance, is_invalid_objective_time


def nansumwrapper(array: np.ndarray, **kwargs) -> np.ndarray:
    """Wrapper around np.nansum. Ensures partials that contain only NaN are returned as NaN instead of 0.

    Args:
        array: values to sum.
        **kwargs: arbitrary keyword arguments.

    Returns:
        Summed NumPy array.
    """
    where_all_nan = np.isnan(array).all(**kwargs)  # get the locations where all partials are NaN
    summed_array = np.nansum(array, **kwargs)  # sum as usual
    summed_array[where_all_nan] = np.nan  # overwrite the sums where necessary
    return summed_array


class SearchspaceStatistics:
    """Object for obtaining information from a raw, brute-forced cache file."""

    size: int
    repeats: int
    objective_times: dict
    objective_performances: dict
    objective_times_array: np.ndarray
    objective_performances_array: np.ndarray
    objective_times_total: np.ndarray
    objective_performances_total: np.ndarray
    objective_times_total_sorted: np.ndarray
    objective_performances_total_sorted: np.ndarray
    objective_performances_total_sorted_nan: np.ndarray

    T4_time_keys_to_kernel_tuner_time_keys_mapping = {
        "compilation": "compile_time",
        "benchmark": "benchmark_time",
        "framework": "framework_time",
        "search_algorithm": "strategy_time",
        "validation": "verification_time",
    }
    kernel_tuner_time_keys_to_T4_time_keys_mapping = {
        v: k for k, v in T4_time_keys_to_kernel_tuner_time_keys_mapping.items()
    }

    def __init__(
        self,
        kernel_name: str,
        device_name: str,
        minimization: bool,
        objective_time_keys: list[str],
        objective_performance_keys: list[str],
        bruteforced_caches_path=Path("cached_data_used/cachefiles"),
    ) -> None:
        """Initialization method for a Searchspace statistics object.

        Args:
            kernel_name: the name of the kernel.
            device_name: the name of the device (GPU) used.
            minimization: whether the optimization algorithm was minimizing.
            objective_time_keys: the objective time keys used.
            objective_performance_keys: the objective performance keys used.
            bruteforced_caches_path: the path to the bruteforced caches.
        """
        self.loaded = False
        self.kernel_name = kernel_name
        self.device_name = device_name
        self.minimization = minimization
        self.objective_time_keys = self.T4_time_keys_to_kernel_tuner_time_keys(objective_time_keys)
        self.objective_performance_keys = objective_performance_keys
        self.bruteforced_caches_path = bruteforced_caches_path

        # load the data into the arrays
        self.loaded = self._load()

    def T4_time_keys_to_kernel_tuner_time_keys(self, time_keys: list[str]) -> list[str]:
        """Temporary utility function to use the kernel tuner search space files with the T4 output format.

        Args:
            time_keys: list of T4-style time keys.

        Returns:
            List of Kernel Tuner-style time keys.
        """
        return list(
            (
                self.T4_time_keys_to_kernel_tuner_time_keys_mapping[key]
                if key in self.T4_time_keys_to_kernel_tuner_time_keys_mapping
                else key
            )
            for key in time_keys
        )

    def total_performance_absolute_optimum(self) -> float:
        """Calculate the absolute optimum of the total performances.

        Returns:
            Absolute optimum of the total performances.
        """
        return self.total_performance_minimum() if self.minimization else self.total_performance_maximum()

    def objective_performance_at_cutoff_point(self, cutoff_percentile: float) -> float:
        """Calculate the objective performance value at which to stop for a given cutoff percentile.

        Args:
            cutoff_percentile: the desired cutoff percentile to reach before stopping.

        Returns:
            The objective performance value at which to stop.
        """
        absolute_optimum = self.total_performance_absolute_optimum()
        median = self.total_performance_median()
        objective_performance_target = absolute_optimum + ((median - absolute_optimum) * (1 - cutoff_percentile))
        # print(f"{cutoff_percentile=}, {median=}, {absolute_optimum=}; results in {objective_performance_target=}")
        return objective_performance_target

    def plot_histogram(self, cutoff_percentile: float):
        """Plots a histogram of the distribution.

        Args:
            cutoff_percentile: the desired cutoff percentile to reach before stopping.
        """
        # prepare plot
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
        if not isinstance(axs, list):
            axs = [axs]

        # prepare data
        performances = self.objective_performances_total_sorted
        mean = self.total_performance_mean()
        median = self.total_performance_median()
        cutoff_performance = self.objective_performance_at_cutoff_point(cutoff_percentile)

        # plot the data
        n_bins = 200
        axs[0].hist(performances, bins=n_bins)
        axs[0].set_ylabel("Number of configurations in bin")
        axs[0].set_xlabel("Performance in miliseconds")
        axs[0].axvline(x=[mean], label="Mean", c="red")
        axs[0].axvline(x=[median], label="Median", c="orange")
        axs[0].axvline(x=[cutoff_performance], label="Cutoff point", c="green")

        # finalize and show plot
        plt.legend()
        plt.show()

    def cutoff_point(self, cutoff_percentile: float) -> tuple[float, int]:
        """Calculates the cutoff point.

        Args:
            cutoff_percentile: the desired cutoff percentile to reach before stopping.

        Returns:
            A tuple of the objective value at the cutoff point and the fevals to the cutoff point.
        """
        inverted_sorted_performance_arr = self.objective_performances_total_sorted[::-1]
        N = inverted_sorted_performance_arr.shape[0]

        # get the objective performance at the cutoff point
        objective_performance_at_cutoff_point = self.objective_performance_at_cutoff_point(cutoff_percentile)

        # # top 10 duplicate values
        # uniques, counts = np.unique(inverted_sorted_performance_arr, return_counts=True)
        # top_ten = sorted(zip(uniques, counts), key=lambda x: x[1])[-10:]
        # print(f"Searchspace size: {N}, of which unique: {len(uniques)}")
        # print("Top ten duplicate values (value, occurance):")
        # print(top_ten)

        # # for each number of function evaluations from 0 to N
        # print("Calculating r_i (eq. 6) for each feval until the performance cutoff is reached:")
        # for feval in range(0, N):
        #     # calculate the random search index
        #     ri = round((feval * (N + 1)) / (feval + 1))
        #     print(f"  {feval}: {ri} ({round(ri / N * 100, 3)}%)")
        #     # lookup if random search has reached the performance at cutoff
        #     if inverted_sorted_performance_arr[ri] <= objective_performance_at_cutoff_point:
        #         i = ri
        #         fevals_to_cutoff_point = ceil(i / (N + 1 - i))
        #         break

        # fevals_to_cutoff_point_alt = ceil((cutoff_percentile * N) / (1 + (1 - cutoff_percentile) * N))

        # iterate over the inverted_sorted_performance_arr until we have
        # i = next(x[0] for x in enumerate(inverted_sorted_performance_arr) if x[1] > cutoff_percentile * arr[-1])
        i = next(
            x[0] for x in enumerate(inverted_sorted_performance_arr) if x[1] <= objective_performance_at_cutoff_point
        )
        # In case of x <= (1+p) * f_opt
        # i = next(x[0] for x in enumerate(inverted_sorted_performance_arr) if x[1] <= (1 + (1 - cutoff_percentile)) * arr[-1])  # noqa: E501
        # In case of p*x <= f_opt
        # i = next(x[0] for x in enumerate(inverted_sorted_performance_arr) if cutoff_percentile * x[1] <= arr[-1])
        fevals_to_cutoff_point = ceil(i / (N + 1 - i))

        # print(f"{fevals_to_cutoff_point=} ({i=})")
        # exit(0)
        return objective_performance_at_cutoff_point, fevals_to_cutoff_point

    def cutoff_point_fevals_time(self, cutoff_percentile: float) -> tuple[float, int, float]:
        """Calculates the cutoff point.

        Args:
            cutoff_percentile: the desired cutoff percentile to reach before stopping.

        Returns:
            A tuple of the objective value at cutoff point, fevals to cutoff point, and the mean time to cutoff point.
        """
        cutoff_point_value, cutoff_point_fevals = self.cutoff_point(cutoff_percentile)
        cutoff_point_time = cutoff_point_fevals * self.total_time_median()
        return cutoff_point_value, cutoff_point_fevals, cutoff_point_time

    def _get_filepath(self, lowercase=True) -> Path:
        """Returns the filepath."""
        kernel_directory = self.kernel_name
        if lowercase:
            kernel_directory = kernel_directory.lower()
        filename = f"{self.device_name}.json"
        if lowercase:
            filename = filename.lower()
        return self.bruteforced_caches_path / kernel_directory / filename

    def get_valid_filepath(self) -> Path:
        """Returns the filepath to the Searchspace statistics .json file if it exists.

        Raises:
            FileNotFoundError: if filepath does not exist.

        Returns:
            Filepath to the Searchspace statistics .json file.
        """
        filepath = self._get_filepath()
        if not filepath.exists():
            filepath = self._get_filepath(lowercase=False)
        if not filepath.exists():
            # if the file is not found, raise an error
            from os import getcwd

            raise FileNotFoundError(
                f"{filepath.resolve()} does not exist relative to current working directory {getcwd()}"
            )
        return filepath

    def _is_not_invalid_value(self, value, performance: bool) -> bool:
        """Checks if a cache performance or time value is an array or is not invalid."""
        if isinstance(value, str):
            return False
        if isinstance(value, (list, tuple, np.ndarray)):
            return True
        invalid_check_function = is_invalid_objective_performance if performance else is_invalid_objective_time
        return not invalid_check_function(value)

    def _to_valid_array(self, cache_values: list[dict], key: str, performance: bool) -> np.ndarray:
        """Convert valid cache performance or time values to a numpy array, sum if the input is a list of arrays."""
        # make a list of all valid values
        values = list(
            v[key] if key in v and self._is_not_invalid_value(v[key], performance) else np.nan for v in cache_values
        )
        # check if there are values that are arrays
        for value_index, value in enumerate(values):
            if isinstance(value, (list, tuple, np.ndarray)):
                # if the cache value is an array, sum the valid values
                array = value
                list_to_sum = list(v for v in array if self._is_not_invalid_value(v, performance))
                values[value_index] = (
                    sum(list_to_sum)
                    if len(list_to_sum) > 0 and self._is_not_invalid_value(sum(list_to_sum), performance)
                    else np.nan
                )
        assert all(isinstance(v, (int, float)) for v in values)
        return np.array(values)

    def _load(self) -> bool:
        """Load the contents of the cache file."""
        filepath = self.get_valid_filepath()
        with open(filepath, "r") as fh:
            print(f"Loading statistics for {filepath}...")
            # get the cache from the .json file
            orig_contents = fh.read()
            try:
                data = json.loads(orig_contents)
            except json.decoder.JSONDecodeError:
                contents = orig_contents[:-1] + "}\n}"
                try:
                    data = json.loads(contents)
                except json.decoder.JSONDecodeError:
                    contents = orig_contents[:-2] + "}\n}"
                    data = json.loads(contents)
            cache: dict = data["cache"]
            self.cache = cache

            # get the time values per configuration
            cache_values = list(cache.values())
            self.size = len(cache_values)
            self.objective_times = dict()
            for key in self.objective_time_keys:
                self.objective_times[key] = self._to_valid_array(cache_values, key, performance=False)
                self.objective_times[key] = (
                    self.objective_times[key] / 1000
                )  # TODO Kernel Tuner specific miliseconds to seconds conversion
                assert (
                    self.objective_times[key].ndim == 1
                ), f"Should have one dimension, has {self.objective_times[key].ndim}"
                assert self.objective_times[key].shape[0] == len(
                    cache_values
                ), f"Should have the same size as cache_values ({self.size}), has {self.objective_times[key].shape[0]}"
                assert not np.all(
                    np.isnan(self.objective_times[key])
                ), f"""All values for {key=} are NaN.
                        Likely the experiment did not collect time values for objective_time_keys '{key}'."""

            # get the performance values per configuration
            self.objective_performances = dict()
            for key in self.objective_performance_keys:
                self.objective_performances[key] = self._to_valid_array(cache_values, key, performance=True)
                assert (
                    self.objective_performances[key].ndim == 1
                ), f"Should have one dimension, has {self.objective_performances[key].ndim}"
                assert self.objective_performances[key].shape[0] == len(
                    cache_values
                ), f"""Should have the same size as cache_values ({self.size}),
                        has {self.objective_performances[key].shape[0]}"""
                assert not np.all(
                    np.isnan(self.objective_performances[key])
                ), f"""All values for {key=} are NaN.
                    Likely the experiment did not collect performance values for objective_performance_key '{key}'."""

            # get the number of repeats
            valid_cache_index: int = 0
            while "times" not in cache_values[valid_cache_index]:
                valid_cache_index += 1
            self.repeats = len(cache_values[valid_cache_index]["times"])

            # combine the arrays to the shape [len(objective_keys), self.size]
            self.objective_times_array = np.array(list(self.objective_times[key] for key in self.objective_time_keys))
            assert self.objective_times_array.shape == tuple([len(self.objective_time_keys), self.size])
            self.objective_performances_array = np.array(
                list(self.objective_performances[key] for key in self.objective_performance_keys)
            )
            assert self.objective_performances_array.shape == tuple([len(self.objective_performance_keys), self.size])

            # get the totals
            self.objective_times_total = nansumwrapper(self.objective_times_array, axis=0)
            assert self.objective_times_total.shape == tuple([self.size])
            # more of a test than a necessary assert
            assert (
                np.nansum(self.objective_times_array[:, 0]) == self.objective_times_total[0]
            ), f"""Sums of objective performances do not match:
                {np.nansum(self.objective_times_array[:, 0])} vs. {self.objective_times_total[0]}"""
            self.objective_performances_total = nansumwrapper(self.objective_performances_array, axis=0)
            assert self.objective_performances_total.shape == tuple([self.size])
            # more of a test than a necessary assert
            assert (
                np.nansum(self.objective_performances_array[:, 0]) == self.objective_performances_total[0]
            ), f"""Sums of objective performances do not match:
                {np.nansum(self.objective_performances_array[:, 0])} vs. {self.objective_performances_total[0]}"""

            # sort
            self.objective_times_total_sorted = np.sort(
                self.objective_times_total[~np.isnan(self.objective_times_total)]
            )
            self.objective_times_number_of_nan = (
                self.objective_times_total.shape[0] - self.objective_times_total_sorted.shape[0]
            )
            objective_performances_nan_mask = np.isnan(self.objective_performances_total)
            self.objective_performances_number_of_nan = np.count_nonzero(objective_performances_nan_mask)
            self.objective_performances_total_sorted = np.sort(
                self.objective_performances_total[~objective_performances_nan_mask]
            )
            # make sure the best values are at the start, because NaNs are appended to the end
            sorted_best_first = (
                self.objective_performances_total_sorted
                if self.minimization
                else self.objective_performances_total_sorted[::-1]
            )
            self.objective_performances_total_sorted_nan = np.concatenate(
                (sorted_best_first, [np.nan] * self.objective_performances_number_of_nan)
            )

        return True

    def get_value_in_config(self, config: str, key: str):
        """Get the value for a key given a configuration."""
        return self.cache[config][key]

    def get_num_duplicate_values(self, value: float) -> int:
        """Get the number of duplicate values in the searchspace."""
        duplicates = np.count_nonzero(np.where(self.objective_performances_total == value, 1, 0)) - 1
        if duplicates < 0:
            raise ValueError(f"Value {value} not in distribution")
        return duplicates

    def mean_strategy_time_per_feval(self) -> float:
        """Gets the average time spent on the strategy per function evaluation."""
        if "strategy" in self.objective_times:
            strategy_times = self.objective_times
            invalid_mask = np.isnan(self.objective_performances_total)
            if not all(invalid_mask):
                return np.mean(strategy_times)
        return 0

    def get_time_per_feval(self, time_per_feval_operator: str, strategy_offset=False) -> float:
        """Gets the average time per function evaluation. Several methods available.

        Args:
            time_per_feval_operator: method to use.
            strategy_offset: whether to include a small offset to adjust for missing strategy time.

        Raises:
            ValueError: if invalid ``time_per_feval_operator`` is passed.

        Returns:
            The average time per function evaluation.
        """
        if time_per_feval_operator == "mean":
            time_per_feval = self.total_time_mean()
        elif time_per_feval_operator == "median":
            time_per_feval = self.total_time_median()
        elif time_per_feval_operator == "median_nan":
            time_per_feval = self.total_time_median_nan()
        elif time_per_feval_operator == "mean_per_feval":
            time_per_feval = self.total_time_mean_per_feval()
        elif time_per_feval_operator == "median_per_feval":
            time_per_feval = self.total_time_median_per_feval()
        else:
            raise ValueError(f"Invalid {time_per_feval_operator=}")
        assert not np.isnan(time_per_feval) and time_per_feval > 0, f"Invalid {time_per_feval=}"
        # add a small amount of strategy time if necessary
        offset = 0
        if strategy_offset and self.mean_strategy_time_per_feval() == 0:
            # offset should not be more than 10% of actual time per feval
            offset = min(time_per_feval * 0.1, 0.001)
        return time_per_feval + offset

    def total_time_minimum(self) -> float:
        """Get the minimum value of total time."""
        return self.objective_times_total_sorted[0]

    def total_time_maximum(self) -> float:
        """Get the maximum value of total time."""
        return self.objective_times_total_sorted[-1]

    def total_time_mean(self) -> float:
        """Get the mean of total time."""
        return np.mean(self.objective_times_total_sorted)

    def total_time_median(self) -> float:
        """Get the median of total time."""
        return np.median(self.objective_times_total_sorted)

    def total_time_median_nan(self) -> float:
        """Get the median of total time, including NaN."""
        sorted = np.sort(self.objective_times_total)
        size = sorted.shape[0]
        median_index = (size - 1) / 2
        if size % 2 == 0:
            median = np.mean(sorted[floor(median_index) : ceil(median_index)])
        else:
            median = sorted[int(median_index)]
        return median

    def total_time_mean_per_feval(self) -> float:
        """Get the true mean per function evaluation by adding the chance of an invalid."""
        invalid_mask = np.isnan(self.objective_performances_total)
        if all(~invalid_mask):  # if there are no invalid values, this is the same as the normal mean
            return self.total_time_mean()
        mean_time_per_invalid_feval = np.mean(self.objective_times_total[invalid_mask])
        mean_time_per_valid_feval = np.mean(self.objective_times_total[~invalid_mask])
        fraction_invalid = np.count_nonzero(invalid_mask) / self.size
        return mean_time_per_valid_feval + (fraction_invalid * mean_time_per_invalid_feval)

    def total_time_median_per_feval(self) -> float:
        """Get the true median per function evaluation by adding the chance of an invalid."""
        invalid_mask = np.isnan(self.objective_performances_total)
        if all(~invalid_mask):  # if there are no invalid values, this is the same as the normal median
            return self.total_time_median()
        mean_time_per_invalid_feval = np.median(self.objective_times_total[invalid_mask])
        mean_time_per_valid_feval = np.median(self.objective_times_total[~invalid_mask])
        fraction_invalid = np.count_nonzero(invalid_mask) / self.size
        return mean_time_per_valid_feval + (fraction_invalid * mean_time_per_invalid_feval)

    def total_time_std(self) -> float:
        """Get the standard deviation of total time."""
        return np.std(self.objective_times_total_sorted)

    def total_time_quartiles(self) -> tuple[float, float]:
        """Get the quartiles (25th and 75th percentiles) of total time."""
        q25, q75 = np.percentile(self.objective_times_total_sorted, [25, 75])
        return tuple([q25, q75])

    def total_time_interquartile_range(self) -> float:
        """Get the interquartile range of total time."""
        q25, q75 = self.total_time_quartiles()
        return q75 - q25

    def total_performance_minimum(self) -> float:
        """Get the minimum value for total performance."""
        return self.objective_performances_total_sorted[0]

    def total_performance_maximum(self) -> float:
        """Get the maximum value for total performance."""
        return self.objective_performances_total_sorted[-1]

    def total_performance_mean(self) -> float:
        """Get the mean of total performance."""
        return np.mean(self.objective_performances_total_sorted)

    def total_performance_median(self) -> float:
        """Get the median of total performance."""
        return np.median(self.objective_performances_total_sorted)

    def total_performance_std(self) -> float:
        """Get the standard deviation of total performance."""
        return np.std(self.objective_performances_total_sorted)

    def total_performance_quartiles(self) -> tuple[float, float]:
        """Get the quartiles (25th and 75th percentiles) of total performance."""
        q25, q75 = np.percentile(self.objective_performances_total_sorted, [25, 75])
        return tuple([q25, q75])

    def total_performance_interquartile_range(self) -> float:
        """Get the interquartile range of total performance."""
        q25, q75 = self.total_performance_quartiles()
        return q75 - q25


def test():  # pragma: no cover
    """Test the SearchspaceStatistics object class."""
    ss_stats = SearchspaceStatistics("gemm", "RTX_2080_Ti")

    print(f"{ss_stats.total_time_minimum()=}")
    print(f"{ss_stats.total_time_maximum()=}")
    print(f"{ss_stats.total_time_mean()=}")
    print(f"{ss_stats.total_time_median()=}")
    print(f"{ss_stats.total_time_std()=}")
    print(f"{ss_stats.total_time_quartiles()=}")
    print(f"{ss_stats.total_time_interquartile_range()=}")

    print(f"{ss_stats.total_performance_minimum()=}")
    print(f"{ss_stats.total_performance_maximum()=}")
    print(f"{ss_stats.total_performance_mean()=}")
    print(f"{ss_stats.total_performance_median()=}")
    print(f"{ss_stats.total_performance_std()=}")
    print(f"{ss_stats.total_performance_quartiles()=}")
    print(f"{ss_stats.total_performance_interquartile_range()=}")


if __name__ == "__main__":
    test()
