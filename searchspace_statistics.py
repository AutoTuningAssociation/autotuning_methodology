from pathlib import Path
import json
import numpy as np


class SearchspaceStatistics():
    """ Object for obtaining information from a raw, brute-forced cache file """

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

    def __init__(self, kernel_name: str, device_name: str, objective_performance_keys=['time'],
                 objective_time_keys=['times', 'compile_time', 'verification_time', 'benchmark_time', 'strategy_time', 'framework_time']) -> None:
        self.loaded = False
        self.error_value = 1e20
        self.kernel_name = kernel_name
        self.device_name = device_name
        self.objective_performance_keys: list[str] = objective_performance_keys
        self.objective_time_keys: list[str] = objective_time_keys

        # load the data into the arrays
        self.loaded = self._load()

        print(f"{self.total_time_mean()=}")
        print(f"{self.total_performance_mean()=}")

    def _get_filepath(self) -> Path:
        """ Returns the filepath """
        basepath = Path("../cached_data_used/cachefiles")
        kernel_directory = self.kernel_name.lower()
        filename = f"{self.device_name.lower()}.json"
        return basepath / kernel_directory / filename

    def get_valid_filepath(self) -> Path:
        """ Returns the filepath if it exists """
        filepath = self._get_filepath()
        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} does not exist")
        return filepath

    def _is_not_invalid_value(self, value) -> bool:
        """ Checks if a cache value is an array or is not invalid """
        if isinstance(value, (list, tuple, np.ndarray)):
            return True
        return value != self.error_value and not value == 'RuntimeFailedConfig' and not np.isnan(value)

    def _to_valid_array(self, cache_values: list[dict], key: str) -> np.ndarray:
        """ Convert valid cache values to a numpy array, sum if the input is a list of arrays """
        # make a list of all valid values
        values = list(v[key] if key in v and self._is_not_invalid_value(v[key]) else np.nan for v in cache_values)
        # check if there are values that are arrays
        for value_index, value in enumerate(values):
            if isinstance(value, (list, tuple, np.ndarray)):
                # if the cache value is an array, sum the valid values
                array = value
                summed_value = sum(list(v for v in array if self._is_not_invalid_value(v)))
                values[value_index] = summed_value if summed_value != 0 and self._is_not_invalid_value(summed_value) else np.nan
        return np.array(values)

    def _load(self) -> bool:
        """ Load the contents of the cache file """
        filepath = self.get_valid_filepath()
        with open(filepath, 'r') as fh:
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
            cache: dict = data['cache']

            # get the performance and time values per configuration
            cache_values = list(cache.values())
            self.size = len(cache_values)
            self.objective_times = dict()
            for key in self.objective_time_keys:
                self.objective_times[key] = self._to_valid_array(cache_values, key)
                assert self.objective_times[key].ndim == 1, f"Should have one dimension, has {self.objective_times[key].ndim}"
                assert self.objective_times[key].shape[0] == len(
                    cache_values), f"Should have the same size as cache_values ({self.size}), has {self.objective_times[key].shape[0]}"
            self.objective_performances = dict()
            for key in self.objective_performance_keys:
                self.objective_performances[key] = self._to_valid_array(cache_values, key)
                assert self.objective_performances[key].ndim == 1, f"Should have one dimension, has {self.objective_performances[key].ndim}"
                assert self.objective_performances[key].shape[0] == len(
                    cache_values), f"Should have the same size as cache_values ({self.size}), has {self.objective_performances[key].shape[0]}"

            # get the number of repeats
            valid_cache_index: int = 0
            while 'times' not in cache_values[valid_cache_index]:
                valid_cache_index += 1
            self.repeats = len(cache_values[valid_cache_index]['times'])

            # combine the arrays to the shape [len(objective_keys), self.size]
            self.objective_times_array = np.array(list(self.objective_times[key] for key in self.objective_time_keys))
            assert self.objective_times_array.shape == tuple([len(self.objective_time_keys), self.size])
            self.objective_performances_array = np.array(list(self.objective_performances[key] for key in self.objective_performance_keys))
            assert self.objective_performances_array.shape == tuple([len(self.objective_performance_keys), self.size])

            # get the totals
            self.objective_times_total = np.nansum(self.objective_times_array, axis=0)
            assert self.objective_times_total.shape == tuple([self.size])
            assert np.sum(self.objective_times_array[:, 0]) == self.objective_times_total[0]    # more of a test
            self.objective_performances_total = np.nansum(self.objective_performances_array, axis=0)
            assert self.objective_performances_total.shape == tuple([self.size])
            assert np.sum(self.objective_performances_array[:, 0]) == self.objective_performances_total[0]    # more of a test

            # sort
            self.objective_times_total_sorted = np.sort(self.objective_times_total[~np.isnan(self.objective_times_total)])
            self.objective_performances_total_sorted = np.sort(self.objective_performances_total[~np.isnan(self.objective_performances_total)])

        return True

    def total_time_minimum(self) -> float:
        """ Get the minimum value of total time """
        return self.objective_times_total_sorted[0]

    def total_time_maximum(self) -> float:
        """ Get the maximum value of total time """
        return self.objective_times_total_sorted[-1]

    def total_time_mean(self) -> float:
        """ Get the mean of total time """
        return np.mean(self.objective_times_total_sorted)

    def total_time_median(self) -> float:
        """ Get the median of total time """
        return np.median(self.objective_times_total_sorted)

    def total_time_std(self) -> float:
        """ Get the standard deviation of total time """
        return np.std(self.objective_times_total_sorted)

    def total_time_quartiles(self) -> tuple[float, float]:
        """ Get the quartiles (25th and 75th percentiles) of total time """
        q25, q75 = np.percentile(self.objective_times_total_sorted, [25, 75])
        return tuple([q25, q75])

    def total_time_interquartile_range(self) -> float:
        """ Get the interquartile range of total time """
        q25, q75 = self.total_time_quartiles()
        return q75 - q25

    def total_performance_minimum(self) -> float:
        """ Get the minimum value for total performance """
        return self.objective_performances_total_sorted[0]

    def total_performance_maximum(self) -> float:
        """ Get the maximum value for total performance """
        return self.objective_performances_total_sorted[-1]

    def total_performance_mean(self) -> float:
        """ Get the mean of total performance """
        return np.mean(self.objective_performances_total)

    def total_performance_median(self) -> float:
        """ Get the median of total performance """
        return np.median(self.objective_performances_total_sorted)

    def total_performance_std(self) -> float:
        """ Get the standard deviation of total performance """
        return np.std(self.objective_performances_total_sorted)

    def total_performance_quartiles(self) -> tuple[float, float]:
        """ Get the quartiles (25th and 75th percentiles) of total performance """
        q25, q75 = np.percentile(self.objective_performances_total_sorted, [25, 75])
        return tuple([q25, q75])

    def total_performance_interquartile_range(self) -> float:
        """ Get the interquartile range of total performance """
        q25, q75 = self.total_performance_quartiles()
        return q75 - q25


if __name__ == "__main__":
    SearchspaceStatistics('pnpoly', 'RTX_2080_Ti')
