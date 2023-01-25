from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from caching import ResultsDescription


class Curve(ABC):
    """ The Curve object can produce NumPy arrays directly suitable for plotting """

    def __init__(self, results_description: ResultsDescription) -> None:
        """ Initialize using a ResultsDescription """

        # inputs
        self.name = results_description.strategy_name
        self.display_name = results_description.strategy_display_name
        self.device_name = results_description.device_name
        self.kernel_name = results_description.kernel_name
        self.stochastic = results_description.stochastic

        # result data
        results = results_description.get_results()
        self._x_fevals = results.fevals_results    # the time per objective value in number of function evaluations since start (1d if deterministic, 2d if stochastic)
        self._x_time = results.time_results    # the time per objective value in seconds since start the raw x-axis (1d if deterministic, 2d if stochastic)
        self._y = results.objective_value_best_results    # the objective values (1d if deterministic, 2d if stochastic)

        # complete initialisation
        self.check_attributes()
        super().__init__()

    def check_attributes(self) -> None:
        """ Asserts the types and values of attributes upon initialisation """

        # assert types
        assert isinstance(self.name, str)
        assert isinstance(self.display_name, str)
        assert isinstance(self.device_name, str)
        assert isinstance(self.kernel_name, str)
        assert isinstance(self.stochastic, bool)
        assert isinstance(self._x_fevals, np.ndarray)
        assert isinstance(self._x_time, np.ndarray)
        assert isinstance(self._y, np.ndarray)

        # assert values
        if self.stochastic is False:
            assert self._x_fevals.ndim == 1
            assert self._x_fevals[0] == 1    # the first function evaluation must be 1
            assert self._x_time.ndim == 1
            assert self._y.ndim == 1
        else:
            assert self._x_fevals.ndim == 2
            assert all(self._x_fevals[0] == 1)    # the first function evaluation must be 1
            assert self._x_time.ndim == 2
            assert self._y.ndim == 2
        assert self._x_fevals.shape == self._x_time.shape == self._y.shape

    @abstractmethod
    def get_curve_over_fevals(self, fevals_range: np.ndarray) -> np.ndarray:
        """ Get the curve over the specified range of function evaluations, returns NaN beyond limits. """
        return fevals_range

    @abstractmethod
    def get_curve_over_time(self, time_range: np.ndarray) -> np.ndarray:
        """ Get the curve at the specified times using isotonic regression, returns NaN beyond limits. """
        return time_range

    def fevals_find_pad_width(self, array: np.ndarray, target_array: np.ndarray) -> tuple[int, int]:
        """ Find the amount of padding required on both sides of array to match target_array """
        if array.ndim != 1 or target_array.ndim != 1:
            raise ValueError("Both arrays must be one-dimensional")

        # get the indices where elements of target_array are in array
        indices = np.nonzero(np.isin(target_array, array, assume_unique=True))[0]
        if len(indices) == len(target_array):
            return (0, 0)
        if len(indices) > len(target_array):
            raise ValueError(f"Length of indices ({len(indices)}) should be the less then or equal to length of target_array ({len(target_array)})")
        # check whether array is consecutively in target_array
        assert (array[~np.isnan(array)] == target_array[indices]).all()
        padding_start = indices[0]
        padding_end = len(target_array) - 1 - indices[-1]
        padding = (padding_start, padding_end)
        return padding


class DeterministicOptimizationAlgorithm(Curve):

    def get_curve_over_fevals(self, fevals_range: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
        return super().get_curve_over_fevals(curve)

    def get_curve_over_time(self, time_range: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
        return super().get_curve_over_time(time)


class StochasticOptimizationAlgorithm(Curve):

    def _get_indices(self, draws: np.ndarray, dist: np.ndarray) -> np.ndarray:
        """ For each draw, get the index (position) in the distribution """
        # TODO make this more efficient using np.argwhere(np.isin()) or list.index()
        # dist = dist[::-1]
        indices_found = list()
        if draws.ndim == 1:
            for x in draws:
                indices_per_trial = list()
                if not np.isnan(x):
                    indices = np.where(x == dist)[0]
                    indices = round(np.mean(indices))
                    indices_found.append(indices)
                else:
                    indices_per_trial.append(np.NaN)
            #indices = np.concatenate([np.where(x == dist) for x in draws]).flatten()
        elif draws.ndim == 2:
            for y in draws:
                indices_per_trial = list()
                for x in y:
                    if not np.isnan(x):
                        indices = np.where(x == dist)[0]
                        indices = round(np.mean(indices))
                        indices_per_trial.append(indices)
                    else:
                        indices_per_trial.append(np.NaN)
                indices_found.append(indices_per_trial)
            #indices = [np.concatenate([np.where(x == dist) for x in y]).flatten() for y in draws]
        else:
            raise Exception("Expected draws to be 1D or 2D")
        indices_found = np.array(indices_found)
        return indices_found

    def get_curve_over_fevals(self, fevals_range: np.ndarray, dist: np.ndarray = None,
                              confidence_level: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # first filter to only get the fevals range
        matching_indices_mask = np.array([np.isin(x_column, fevals_range, assume_unique=True)
                                          for x_column in self._x_fevals.T]).transpose()    # get the indices of the matching feval range per repeat (column)
        masked_values = np.where(matching_indices_mask, self._y, np.nan)    # apply the mask to the values, filling NaN for False
        masked_fevals = np.where(matching_indices_mask, self._x_fevals, np.nan).transpose()    # apply the mask to the fevals, filling NaN for False
        # check that the filtered fevals are consistent
        if not np.allclose(masked_fevals, masked_fevals[0], equal_nan=True):
            raise ValueError("The array of function evaluations differs per repeat")
        fevals = masked_fevals[0]    # safe to use as every repeat has the same array of fevals
        masked_fevals = masked_fevals.transpose()    # transpose back to original shape
        # remove fevals where every repeat has NaN
        num_repeats = masked_values.shape[1]
        masked_fevals = masked_fevals[~np.isnan(masked_fevals).all(axis=1)].reshape(-1, num_repeats)
        masked_values = masked_values[~np.isnan(masked_values).all(axis=1)].reshape(-1, num_repeats)
        assert fevals_range.shape[0] == masked_values.shape[0] == masked_fevals.shape[0]
        # if a distribution is included
        if dist is not None:
            # for each value, get the index in the distribution
            indices = self._get_indices(masked_values, dist)
            # get the mean index per feval
            indices_mean = np.array(np.round(np.nanmedian(indices, axis=1)), dtype=int)
            if confidence_level is None:
                # get the standard error on the indices per feval
                indices_std = np.array(np.round(np.nanstd(indices, axis=1)), dtype=int)
                indices_lower_err = np.clip(indices_mean - indices_std, a_min=0, a_max=dist.shape[0] - 1)
                indices_upper_err = np.clip(indices_mean + indices_std, a_min=0, a_max=dist.shape[0] - 1)
            else:
                indices_lower_err, indices_upper_err = self.get_confidence_interval(indices, confidence_level)
                indices_lower_err, indices_upper_err = indices_lower_err.astype(int), indices_upper_err.astype(int)
            # obtain the curves by looking up the associated values
            curve = dist[indices_mean]
            curve_lower_err = dist[indices_lower_err]
            curve_upper_err = dist[indices_upper_err]
        else:
            # obtain the curves
            curve: np.ndarray = np.nanmedian(masked_values, axis=1)    # get the curve by taking the mean
            if confidence_level is None:
                # get the standard error
                curve_std: np.ndarray = np.nanstd(masked_values, axis=1)
                curve_lower_err = curve - curve_std
                curve_upper_err = curve + curve_std
            else:
                curve_lower_err, curve_upper_err = self.get_confidence_interval(masked_values, confidence_level)

        # # remove remaining NaN, yielding an array which <= fevals.shape
        # curve = curve[~np.isnan(curve)]
        # curve_lower_err = curve_lower_err[~np.isnan(curve_lower_err)]
        # curve_upper_err = curve_upper_err[~np.isnan(curve_upper_err)]

        # pad with NaN where outside the range, yielding an array which == fevals.shape
        if curve.shape != fevals_range.shape:
            pad_width = self.fevals_find_pad_width(fevals, fevals_range)
            curve = np.pad(curve, pad_width=pad_width, constant_values=np.nan)
            curve_lower_err = np.pad(curve_lower_err, pad_width=pad_width, constant_values=np.nan)
            curve_upper_err = np.pad(curve_upper_err, pad_width=pad_width, constant_values=np.nan)
        assert curve.shape == fevals_range.shape
        return curve, curve_lower_err, curve_upper_err

    def get_curve_over_time(self, time_range: np.ndarray) -> np.ndarray:
        return super().get_curve_over_time(time_range)

    def get_confidence_interval(self, values: np.ndarray, confidence_level: float) -> Tuple[np.ndarray, np.ndarray]:
        """ Calculates the non-parametric confidence interval for repeated individual configurations, assumed to be IID """
        assert values.ndim == 2    # should be two-dimensional (iterations, repeats)
        from math import floor, ceil, sqrt
        n = values.shape[1]
        alpha = 1 - confidence_level
        area_per_tail = alpha / 2
        q = 0.5
        z = 1.96
        base = z * sqrt(n * q * (1 - q))
        lower_rank = max(floor(n * q - base), 0)
        upper_rank = min(ceil(n * q + base), n - 1)
        confidence_interval_lower = np.full(values.shape[0], np.nan)
        confidence_interval_upper = np.full(values.shape[0], np.nan)
        print(f"  {lower_rank}, {upper_rank}")

        # for each function evaluation, calculate the confidence interval
        for feval_index, feval_repeats in enumerate(values):
            feval_repeats_sorted = np.sort(feval_repeats)
            confidence_interval_lower[feval_index] = feval_repeats_sorted[lower_rank]
            confidence_interval_upper[feval_index] = feval_repeats_sorted[upper_rank]

        return confidence_interval_lower, confidence_interval_upper

    def get_times_confidence_interval(times: list, confidence_level=0.95) -> Tuple[float, float]:
        """ Calculate the non-parametric confidence interval for repeated configurations, assumed to be IID """
        alpha = 1 - confidence_level
        area_per_tail = alpha / 2
        n = len(times)
        times.sort()
        np_times = np.array(times)
        lower_rank = floor((n - 1.96 * sqrt(n)) / 2)
        upper_rank = ceil(1 + ((n + 1.96 * sqrt(n)) / 2))
        print(f"  {lower_rank}, {upper_rank}")
        # TODO how much does using the binomial distribution here change the outcome, and is it possible?
        return np_times[max(lower_rank, 0)], np_times[min(upper_rank, n - 1)]
