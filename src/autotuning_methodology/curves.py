"""Code for curve generation."""

from __future__ import annotations  # for correct nested type hints e.g. list[str], tuple[dict, str]

from abc import ABC, abstractmethod
from math import ceil, floor, sqrt
from warnings import warn

import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.isotonic import IsotonicRegression

from autotuning_methodology.caching import ResultsDescription
from autotuning_methodology.searchspace_statistics import SearchspaceStatistics


def get_indices_in_distribution(
    draws: np.ndarray, dist: np.ndarray, sorter=None, skip_draws_check: bool = False, skip_dist_check: bool = False
) -> np.ndarray:
    """Function to get the indices in a distribution in an efficient manner.

    For each draw, get the index (position) in the ascendingly sorted distribution.
    For unsorted dists, use `get_indices_in_array()`.

    Args:
        draws: the values to get the indices of.
        dist: the distribution, assumed to be ascendingly sorted unless `sorter` is provided.
        sorter: NumPy array of indices that sort the distribution. Defaults to None.
        skip_draws_check: skips checking that each value in `draws` is in the `dist`. Defaults to False.
        skip_dist_check: skips checking that the distribution is correctly ordered. Defaults to False.

    Returns:
        A NumPy array of type float of the same shape as `draws`, with NaN where not found in `dist`.
    """
    assert dist.ndim == 1, f"distribution can not have more than one dimension, has {dist.ndim}"

    # check whether the distribution is correctly ordered
    if not skip_dist_check:
        strictly_ascending_sort = dist[:-1] <= dist[1:]
        assert np.all(
            strictly_ascending_sort
        ), f"""Distribution is not sorted ascendingly,
            {np.count_nonzero(~strictly_ascending_sort)} violations in {len(dist)} values: {dist}"""

    # check whether each value of draws (excluding NaN) is in dist
    if not skip_draws_check:
        assert np.all(
            np.in1d(draws[~np.isnan(draws)], dist)
        ), f"""
            Each value in draws should be in dist,
            but {np.size(draws[~np.isnan(draws)][~np.in1d(draws[~np.isnan(draws)], dist)])} values
            of the {np.size(draws)} are missing: {draws[~np.isnan(draws)][~np.in1d(draws[~np.isnan(draws)], dist)]}"""

    # check the sorter
    if sorter is not None:
        assert sorter.shape == dist.shape, "The shape of the sorter must be the same as the distribution"

    # find the index of each draw in the distribution
    indices_found = np.searchsorted(dist, draws, side="left", sorter=sorter).astype(float)
    assert indices_found.shape == draws.shape, "The shape of the indices must match the shape of the draws"

    # if indices found are outside the array, make them NaN
    indices_found[indices_found < 0] = np.NaN
    indices_found[indices_found >= len(dist)] = np.NaN

    return indices_found


def get_indices_in_array(values: np.ndarray, array: np.ndarray) -> np.ndarray:
    """Function to get the indices in an array in an efficient manner.

    For each value, get the index (position) in the 1D array.
    More general version of ``get_indices_in_distribution()``, first sorts array and reverses the sort on the result.

    Args:
        values: the values to get the indices of.
        array: the array to look up the indices in.

    Returns:
        A NumPy integer array with the same shape as ``values``, containing the indices.
    """
    # get the order of indices that would sort the array
    array_sorter = np.argsort(array)

    # get the index in array of the values
    indices_found = get_indices_in_distribution(values, dist=array, sorter=array_sorter, skip_dist_check=True)

    # replace the indices found with the original, unsorted indices of array
    nan_mask = ~np.isnan(indices_found)
    indices_found_unsorted = np.full_like(indices_found, fill_value=np.NaN)
    indices_found_unsorted[nan_mask] = array_sorter[indices_found[nan_mask].astype(int)]

    return indices_found_unsorted


def moving_average(y: np.ndarray, window_size=3) -> np.ndarray:
    """Function to calculate the moving average over an array.

    Output array is the same size as input array.
    After https://stackoverflow.com/a/47490020.

    Args:
        y: the 1-dimensional array to calculate the moving average over.
        window_size: the moving average window size. Defaults to 3.

    Returns:
        the smoothed 1-dimensional array with the same size as input array.
    """
    assert y.ndim == 1
    y_padded = np.pad(y, (window_size // 2, window_size - 1 - window_size // 2), mode="edge")
    y_smooth = np.convolve(y_padded, np.ones((window_size,)) / window_size, mode="valid")
    return y_smooth


class CurveBasis(ABC):
    """Abstract object providing minimals for visualization and analysis. Implemented by ``Curve`` and ``Baseline``."""

    @abstractmethod
    def get_curve(self, range: np.ndarray, x_type: str, dist: np.ndarray = None, confidence_level: float = None):
        """Get the curve over the specified range of time or function evaluations.

        Args:
            range: the range of time or function evaluations.
            x_type: the type of the x-axis range (either time or function evaluations).
            dist: the distribution, used for looking up indices. Ignored in ``Baseline``. Defaults to None.
            confidence_level: confidence level for the confidence interval. Ignored in ``Baseline``. Defaults to None.

        Raises:
            ValueError: on invalid ``x_type`` argument.

        Returns:
            A tuple of NDArrays with NaN beyond limits.
            See ``get_curve_over_fevals()`` and ``get_curve_over_time()`` for more precise return values.
        """
        if x_type == "fevals":
            return self.get_curve_over_fevals(range, dist, confidence_level)
        elif x_type == "time":
            return self.get_curve_over_time(range, dist, confidence_level)
        raise ValueError(f"x_type must be 'fevals' or 'time', is {x_type}")

    @abstractmethod
    def get_curve_over_fevals(self, fevals_range: np.ndarray, dist: np.ndarray = None, confidence_level: float = None):
        """Get the curve over function evaluations.

        Args:
            fevals_range: the range of function evaluations.
            dist: the distribution, used for looking up indices. Ignored in ``Baseline``. Defaults to None.
            confidence_level: confidence level for the confidence interval. Ignored in ``Baseline``. Defaults to None.

        Returns:
            Two possible returns, for ``Baseline`` and ``Curve`` respectively:
            - NumPy array of the baseline trajectory over the specified ``fevals_range``.
            - The real_stopping_point_index and the real, fictional curve, errors over the specified ``fevals_range``.
        """
        raise NotImplementedError

    @abstractmethod
    def get_curve_over_time(self, time_range: np.ndarray, dist: np.ndarray = None, confidence_level: float = None):
        """Get the curve over time.

        Args:
            time_range: the range of time.
            dist: the distribution, used for looking up indices. Ignored in ``Baseline``. Defaults to None.
            confidence_level: confidence level for the confidence interval. Ignored in ``Baseline``. Defaults to None.

        Returns:
            Two possible returns, for ``Baseline`` and ``Curve`` respectively:
            - NumPy array of the baseline trajectory over the specified ``time_range``.
            - The real_stopping_point_index and the real, fictional curve, errors over the specified ``time_range``.
        """
        raise NotImplementedError

    @abstractmethod
    def get_split_times(self, range: np.ndarray, x_type: str, searchspace_stats: SearchspaceStatistics) -> np.ndarray:
        """Get the times at each point in range split into objective_time_keys.

        Args:
            range: the range of time or function evaluations.
            x_type: the type of range (either time or function evaluations).
            searchspace_stats: Searchspace statistics object.

        Raises:
            ValueError: on wrong ``x_type``.

        Returns:
            A NumPy array of size (len(objective_time_keys), len(range)).
        """
        if x_type == "fevals":
            return self.get_split_times_at_feval(range, searchspace_stats)
        elif x_type == "time":
            return self.get_split_times_at_time(range, searchspace_stats)
        raise ValueError(f"x_type must be 'fevals' or 'time', is {x_type}")

    @abstractmethod
    def get_split_times_at_feval(
        self, fevals_range: np.ndarray, searchspace_stats: SearchspaceStatistics
    ) -> np.ndarray:
        """Get the times at each function eval in the range split into objective_time_keys.

        Args:
            fevals_range: the range of function evaluations.
            searchspace_stats: Searchspace statistics object.

        Returns:
            A NumPy array of size (len(objective_time_keys), len(range)).
        """
        raise NotImplementedError()

    @abstractmethod
    def get_split_times_at_time(self, time_range: np.ndarray, searchspace_stats: SearchspaceStatistics) -> np.ndarray:
        """Get the times at each time point in the range split into objective_time_keys.

        Args:
            time_range: the range of time.
            searchspace_stats: Searchspace statistics object.

        Returns:
            A NumPy array of size (len(objective_time_keys), len(range)).
        """
        raise NotImplementedError()


class Curve(CurveBasis):
    """The Curve object can produce NumPy arrays directly suitable for plotting from a ResultsDescription."""

    def __init__(self, results_description: ResultsDescription) -> None:
        """Initialize using a ResultsDescription.

        Args:
            results_description: the ResultsDescription object containing the data for the Curve.
        """
        # inputs
        self.name = results_description.strategy_name
        self.display_name = results_description.strategy_display_name
        self.device_name = results_description.device_name
        self.kernel_name = results_description.kernel_name
        self.stochastic = results_description.stochastic
        self.minimization = results_description.minimization

        # result data
        results = results_description.get_results()
        self._x_fevals = (
            results.fevals_results
        )  # the time per objective value in fevals since start (1d if deterministic, 2d if stochastic)
        self._x_time = (
            results.objective_time_results
        )  # the time per objective value in seconds since start the raw x-axis (1d if deterministic, 2d if stochastic)
        self._x_time_per_key = (
            results.objective_time_results_per_key
        )  # the time per objective time key (2d if deterministic, 3d if stochastic)
        self._y = (
            results.objective_performance_best_results
        )  # the objective performances (1d if deterministic, 2d if stochastic)
        self._y_per_key = (
            results.objective_performance_results_per_key
        )  # the performance per objective time key (2d if deterministic, 3d if stochastic)

        # complete initialisation
        self.check_attributes()
        super().__init__()

    def check_attributes(self) -> None:
        """Asserts the types and values of attributes upon initialisation."""
        # assert types
        assert isinstance(self.name, str)
        assert isinstance(self.display_name, str)
        assert isinstance(self.device_name, str)
        assert isinstance(self.kernel_name, str)
        assert isinstance(self.stochastic, bool)
        assert isinstance(self._x_fevals, np.ndarray)
        assert isinstance(self._x_time, np.ndarray)
        assert isinstance(self._x_time_per_key, np.ndarray)
        assert isinstance(self._y, np.ndarray)
        assert isinstance(self._y_per_key, np.ndarray)

        # assert values
        if self.stochastic is False:
            assert self._x_fevals.ndim == 1
            assert self._x_fevals[0] == 1  # the first function evaluation must be 1
            assert self._x_time.ndim == 1
            assert self._x_time_per_key.ndim == 2
            assert self._y.ndim == 1
            assert self._y_per_key.ndim == 2
        else:
            assert self._x_fevals.ndim == 2
            assert all(self._x_fevals[0] == 1)  # the first function evaluation must be 1
            assert self._x_time.ndim == 2
            assert self._x_time_per_key.ndim == 3
            assert self._y.ndim == 2
            assert self._y_per_key.ndim == 3
        assert (
            self._x_fevals.shape
            == self._x_time.shape
            == self._x_time_per_key.shape[1:]
            == self._y.shape
            == self._y_per_key.shape[1:]
        )

    def fevals_find_pad_width(self, array: np.ndarray, target_array: np.ndarray) -> tuple[int, int]:
        """Finds the amount of padding required on both sides of ``array`` to match ``target_array``.

        Args:
            array: input array to be padded.
            target_array: target array shape.

        Raises:
            ValueError: when arrays are not one-dimensional or there is a mismatch in the number of values.

        Returns:
            The padded ``array`` with the same shape as ``target_array``.
        """
        if array.ndim != 1 or target_array.ndim != 1:
            raise ValueError("Both arrays must be one-dimensional")

        # get the indices where elements of target_array are in array
        indices = np.nonzero(np.isin(target_array, array, assume_unique=True))[0]
        if len(indices) == len(target_array):
            return (0, 0)
        if len(indices) > len(target_array):
            raise ValueError(
                f"""Length of indices ({len(indices)}) should be
                 less then or equal to length of target_array ({len(target_array)})"""
            )
        # check whether array is consecutively in target_array
        assert (array[~np.isnan(array)] == target_array[indices]).all(), "Array must be consecutively in target array"
        padding_start = indices[0]
        padding_end = len(target_array) - 1 - indices[-1]
        padding = (padding_start, padding_end)
        return padding

    def get_scatter_data(self, x_type: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the raw data required for a scatter plot.

        Args:
            x_type: the type of the x-axis range (either time or function evaluations).

        Raises:
            ValueError: on invalid ``x_type`` argument.

        Returns:
            A tuple of two NumPy arrays for the range and data respectively.
        """
        if x_type == "fevals":
            return self._x_fevals, self._y
        elif x_type == "time":
            return self._x_time, self._y
        raise ValueError(f"x_type must be 'fevals' or 'time', is {x_type}")

    def get_isotonic_regressor(self, y_min: float, y_max: float, out_of_bounds: str = "clip") -> IsotonicRegression:
        """Wrapper function to get the isotonic regressor.

        Args:
            y_min: lower bound on the lowest predicted value. Defaults to -inf.
            y_max: upper bound on the highest predicted value. Defaults to +inf.
            out_of_bounds: handles how values outside the training domain are handled in prediction. Defaults to "clip".

        Returns:
            _description_
        """
        return IsotonicRegression(
            increasing=not self.minimization, y_min=y_min, y_max=y_max, out_of_bounds=out_of_bounds
        )

    def get_isotonic_curve(self, x: np.ndarray, y: np.ndarray, x_new: np.ndarray, ymin=None, ymax=None) -> np.ndarray:
        """Get the isotonic regression curve fitted to x_new using package 'sklearn' or 'isotonic'.

        Args:
            x: x-dimension training data.
            y: y-dimension training data.
            x_new: x-dimension prediction data.
            ymin: lower bound on the lowest predicted value. Defaults to None.
            ymax: upper bound on the highest predicted value. Defaults to None.

        Returns:
            The predicted monotonic piecewise curve for ``x_new``.
        """
        # check if the assumptions that the input arrays are numpy arrays holds
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(x_new, np.ndarray)
        assert x_new.ndim == 1

        # flatten the input arrays if necessary
        if x.ndim > 1:
            x = x.flatten()
        if y.ndim > 1:
            y = y.flatten()

        # apply isotonic regression
        ir = self.get_isotonic_regressor(y_min=ymin, y_max=ymax)
        ir.fit(x, y)
        isotonic_curve = ir.predict(x_new)
        return isotonic_curve


class StochasticOptimizationAlgorithm(Curve):
    """Class for producing a curve for stochastic optimization algorithms."""

    def _get_curve_split_real_fictional_parts(
        self,
        real_stopping_point_index: int,
        x_axis_range: np.ndarray,
        curve: np.ndarray,
        curve_lower_err: np.ndarray,
        curve_upper_err: np.ndarray,
        smoothing_factor=0.0,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split the provided curves based on the real_stopping_point_index.

        Return real_stopping_point_index and the real and fictional part for each curve.
        """
        # select the parts of the data that are real
        x_axis_range_real = x_axis_range[:real_stopping_point_index]
        curve_real = curve[:real_stopping_point_index]
        curve_lower_err_real = curve_lower_err[:real_stopping_point_index]
        curve_upper_err_real = curve_upper_err[:real_stopping_point_index]

        # select the parts of the data that are fictional
        x_axis_range_fictional = np.empty(0)
        curve_fictional = np.empty(0)
        curve_lower_err_fictional = np.empty(0)
        curve_upper_err_fictional = np.empty(0)
        target_index = x_axis_range.shape[0] - 1
        if real_stopping_point_index <= target_index:
            x_axis_range_fictional = x_axis_range[real_stopping_point_index:]
            curve_fictional = curve[real_stopping_point_index:]
            curve_lower_err_fictional = curve_lower_err[real_stopping_point_index:]
            curve_upper_err_fictional = curve_upper_err[real_stopping_point_index:]

        # check the consistency
        self._check_curve_real_fictional_consistency(
            x_axis_range,
            curve,
            curve_lower_err,
            curve_upper_err,
            x_axis_range_real,
            curve_real,
            curve_lower_err_real,
            curve_upper_err_real,
            x_axis_range_fictional,
            curve_fictional,
            curve_lower_err_fictional,
            curve_upper_err_fictional,
        )

        # smooth the curve if applicable
        if smoothing_factor > 0.0:
            if smoothing_factor >= 1:
                raise ValueError(f"Smoothing factor can't be >= 1, is {smoothing_factor}")
            elif smoothing_factor >= 0.05:
                warn(
                    f"Smoothing factor {smoothing_factor} >= 0.05, likely making curves too smooth. 0.005 recommended",
                    category=UserWarning,
                )
            window_size = min(x_axis_range.size, ceil(x_axis_range.size * smoothing_factor))
            if window_size > 1:
                curve_real = moving_average(curve_real, window_size)
                curve_lower_err_real = moving_average(curve_lower_err_real, window_size)
                curve_upper_err_real = moving_average(curve_upper_err_real, window_size)

        return (
            real_stopping_point_index,
            x_axis_range_real,
            curve_real,
            curve_lower_err_real,
            curve_upper_err_real,
            x_axis_range_fictional,
            curve_fictional,
            curve_lower_err_fictional,
            curve_upper_err_fictional,
        )

    def _check_curve_real_fictional_consistency(
        self,
        x_axis_range,
        curve,
        curve_lower_err,
        curve_upper_err,
        x_axis_range_real,
        curve_real,
        curve_lower_err_real,
        curve_upper_err_real,
        x_axis_range_fictional,
        curve_fictional,
        curve_lower_err_fictional,
        curve_upper_err_fictional,
    ):
        """Asserts that the real and fictional results add up correctly."""
        assert (
            x_axis_range.shape == curve.shape == curve_lower_err.shape == curve_upper_err.shape
        ), f"""Shapes must be equal: {x_axis_range.shape=}, {curve.shape=},
                        {curve_lower_err.shape=}, {curve_upper_err.shape=}"""
        assert (
            x_axis_range_real.shape == curve_real.shape == curve_lower_err_real.shape == curve_upper_err_real.shape
        ), f"""Shapes must be equal: {x_axis_range_real.shape=}, {curve_real.shape=},
                        {curve_lower_err_real.shape=}, {curve_upper_err_real.shape=}"""
        assert (
            x_axis_range_fictional.shape
            == curve_fictional.shape
            == curve_lower_err_fictional.shape
            == curve_upper_err_fictional.shape
        ), f"""Shapes must be equal: {x_axis_range_fictional.shape=}, {curve_fictional.shape=},
                        {curve_lower_err_fictional.shape=} {curve_upper_err_fictional.shape=}"""
        if x_axis_range_fictional.ndim > 0:
            # if there's a fictional part, ensure that all the expected data is in the combined real and fictional parts
            x_axis_range_combined = np.concatenate([x_axis_range_real, x_axis_range_fictional])
            assert (
                x_axis_range.shape == x_axis_range_combined.shape
            ), f"The shapes of {x_axis_range.shape=} and {x_axis_range_combined.shape=} do not match"
            assert np.array_equal(
                x_axis_range, np.concatenate([x_axis_range_real, x_axis_range_fictional]), equal_nan=True
            )
            assert np.array_equal(curve, np.concatenate([curve_real, curve_fictional]), equal_nan=True)
            assert np.array_equal(
                curve_lower_err, np.concatenate([curve_lower_err_real, curve_lower_err_fictional]), equal_nan=True
            )
            assert np.array_equal(
                curve_upper_err, np.concatenate([curve_upper_err_real, curve_upper_err_fictional]), equal_nan=True
            )
        else:
            # if there is no fictional part, ensure that all the expected data is in the real part
            assert (
                x_axis_range.shape == x_axis_range_real.shape
            ), f"The shapes of {x_axis_range.shape=} and {x_axis_range_real.shape=} do not match"
            assert np.array_equal(
                x_axis_range, x_axis_range_real, equal_nan=True
            ), f"Unequal arrays: {x_axis_range}, {x_axis_range_real}"
            assert np.array_equal(curve, curve_real, equal_nan=True), f"Unequal arrays: {curve}, {curve_real}"
            assert np.array_equal(
                curve_lower_err, curve_lower_err_real, equal_nan=True
            ), f"Unequal arrays: {curve_lower_err}, {curve_lower_err_real}"
            assert np.array_equal(
                curve_upper_err, curve_upper_err_real, equal_nan=True
            ), f"Unequal arrays: {curve_upper_err}, {curve_upper_err_real}"

    def get_curve(  # noqa: D102
        self, range: np.ndarray, x_type: str, dist: np.ndarray = None, confidence_level: float = None
    ):
        return super().get_curve(range, x_type, dist, confidence_level)

    def _get_matching_feval_indices_in_range(self, fevals_range: np.ndarray) -> np.ndarray:
        """Get a mask of where the fevals range matches with the data."""
        assert fevals_range.ndim == 1
        assert np.all(np.isfinite(fevals_range))
        matching_indices_mask = np.array(
            [np.isin(x_column, fevals_range, assume_unique=True) for x_column in self._x_fevals.T]
        ).transpose()  # get the indices of the matching feval range per repeat (column)
        if np.all(~matching_indices_mask):
            raise ValueError(f"No overlap in data and given {fevals_range=}")
        return matching_indices_mask

    def _get_curve_over_fevals_values_in_range(self, fevals_range: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get the valid fevals and values that are in the given range."""
        target_index: int = fevals_range[-1] - 1

        # filter to only get data in the fevals range
        matching_indices_mask = self._get_matching_feval_indices_in_range(
            fevals_range
        )  # get the indices of the matching feval range per repeat (column)
        masked_values = np.where(
            matching_indices_mask, self._y, np.nan
        )  # apply the mask to the values, filling NaN for False
        masked_fevals = np.where(
            matching_indices_mask, self._x_fevals, np.nan
        ).transpose()  # apply the mask to the fevals, filling NaN for False

        # make sure that the filtered fevals are consistent (every repeat has the same array of fevals)
        if not np.allclose(masked_fevals, masked_fevals[0], equal_nan=True):
            indices = np.nanargmax(masked_fevals, axis=1)  # get the index of the last non-nan value of each repeat
            num_repeats = len(indices)
            # find the repeats that end before the index
            early_ending_repeats = np.where(indices < target_index)
            greatest_common_non_NaN_index = min(floor(np.median(indices)), target_index)
            if np.count_nonzero(early_ending_repeats) > 0:
                warn(
                    f"""For optimization algorithm {self.display_name},
                    {np.count_nonzero(early_ending_repeats)} of the {num_repeats} runs ended before
                     the end of fevals_range ({target_index + 1}).
                     Only data up to {greatest_common_non_NaN_index + 1} fevals will be used.
                     Perhaps increase the allotted auto-tuning time for this optimization algorithm?"""
                )

            # drop the repeats where the highest index is less than greatest_common_non_NaN_index
            keep_repeats = np.where(indices >= greatest_common_non_NaN_index)
            masked_fevals = masked_fevals[keep_repeats]
            masked_values = masked_values.transpose()
            masked_values = masked_values[keep_repeats]

            # set all values beyond the greatest common non-NaN index to NaN
            masked_values[:, greatest_common_non_NaN_index + 1 :] = np.nan
            masked_values = masked_values.transpose()  # transpose back to original shape
            masked_fevals[:, greatest_common_non_NaN_index + 1 :] = np.nan

            # check that the filtered fevals are consistent
            assert np.allclose(
                masked_fevals, masked_fevals[0], equal_nan=True
            ), "Every repeat must have the same array of function evaluations"

        # as every repeat has the same array of fevals, check whether they match the range
        fevals = masked_fevals[
            0
        ]  # safe to assume as every repeat has the same array of fevals, set it before removing NaN to pad the curve
        masked_fevals = masked_fevals.transpose()  # transpose back to original shape

        # remove fevals where every repeat has NaN
        num_repeats = masked_values.shape[1]
        nan_mask = ~np.isnan(masked_values).all(axis=1)
        masked_fevals = masked_fevals[nan_mask].reshape(-1, num_repeats)
        masked_values = masked_values[nan_mask].reshape(-1, num_repeats)
        return fevals, masked_values

    def get_curve_over_fevals(  # noqa: D102
        self, fevals_range: np.ndarray, dist: np.ndarray = None, confidence_level: float = None
    ):
        fevals, masked_values = self._get_curve_over_fevals_values_in_range(fevals_range)

        # if a distribution is included
        curve: np.ndarray
        if dist is not None:
            assert dist.ndim == 1, "The distribution must one-dimensional"
            # for each value, get the index in the distribution
            indices = get_indices_in_distribution(masked_values, dist)
            # get the mean index per feval
            indices_mean = np.array(np.round(np.nanmedian(indices, axis=1)), dtype=int)
            if confidence_level is None:
                # get the standard error on the indices per feval
                indices_std = np.array(np.round(np.nanstd(indices, axis=1)), dtype=int)
                indices_lower_err = np.clip(indices_mean - indices_std, a_min=0, a_max=dist.shape[0] - 1)
                indices_upper_err = np.clip(indices_mean + indices_std, a_min=0, a_max=dist.shape[0] - 1)
            else:
                # get the confidence interval
                indices_lower_err, indices_upper_err = self.get_confidence_interval(indices, confidence_level)
                indices_lower_err, indices_upper_err = indices_lower_err.astype(int), indices_upper_err.astype(int)
            # obtain the curves by looking up the associated values
            curve = dist[indices_mean]
            curve_lower_err = dist[indices_lower_err]
            curve_upper_err = dist[indices_upper_err]
        else:
            # obtain the curves
            curve = np.nanmedian(masked_values, axis=1)  # get the curve by taking the mean
            if confidence_level is None:
                # get the standard error
                curve_std: np.ndarray = np.nanstd(masked_values, axis=1)
                curve_lower_err = curve - curve_std
                curve_upper_err = curve + curve_std
            else:
                # get the confidence interval
                curve_lower_err, curve_upper_err = self.get_confidence_interval(masked_values, confidence_level)

        # # remove remaining NaN, yielding an array which <= fevals.shape
        # curve = curve[~np.isnan(curve)]
        # curve_lower_err = curve_lower_err[~np.isnan(curve_lower_err)]
        # curve_upper_err = curve_upper_err[~np.isnan(curve_upper_err)]

        # pad with NaN where outside the range, yielding an array.shape == fevals.shape
        real_stopping_point_fevals = curve.shape[0]
        real_stopping_point_index = real_stopping_point_fevals - 1
        if curve.shape != fevals_range.shape:
            pad_width = self.fevals_find_pad_width(fevals, fevals_range)
            curve = np.pad(curve, pad_width=pad_width, constant_values=np.nan)
            curve_lower_err = np.pad(curve_lower_err, pad_width=pad_width, constant_values=np.nan)
            curve_upper_err = np.pad(curve_upper_err, pad_width=pad_width, constant_values=np.nan)
        assert curve.shape == fevals_range.shape, "The curve shape must match the input fevals_range shape"

        # if necessary, extend the curves up to target_index
        target_index: int = fevals_range.shape[0] - 1
        if real_stopping_point_index < target_index:
            warn(
                f"""For optimization algorithm {self.display_name},
                all runs end at {real_stopping_point_index + 1} fevals,
                which is before the target number of function evals of {target_index + 1}.""",
                category=UserWarning,
            )
            # take the last non-NaN value and overwrite the curves up to the target index with it
            curve[real_stopping_point_index : target_index + 1] = curve[real_stopping_point_index]
            curve_lower_err[real_stopping_point_index : target_index + 1] = curve_lower_err[real_stopping_point_index]
            curve_upper_err[real_stopping_point_index : target_index + 1] = curve_upper_err[real_stopping_point_index]

        # check whether there are no NaNs left
        assert curve.shape == fevals_range.shape, "The curve shape must match the input fevals_range shape"
        assert np.all(~np.isnan(curve)), f"NaNs at {np.nonzero(np.isnan(curve))[0]}"
        assert np.all(~np.isnan(curve_lower_err)), f"NaNs at {np.nonzero(np.isnan(curve_lower_err))[0]}"
        assert np.all(~np.isnan(curve_upper_err)), f"NaNs at {np.nonzero(np.isnan(curve_upper_err))[0]}"

        # return the curves split in real and fictional
        return self._get_curve_split_real_fictional_parts(
            real_stopping_point_index + 1, fevals_range, curve, curve_lower_err, curve_upper_err
        )

    def _get_curve_over_time_values_in_range(
        self, time_range: np.ndarray, return_1d=True
    ) -> tuple[np.ndarray, np.ndarray, float, int, int]:
        """Get the valid times and values that are in the given range."""
        # check and get the variables
        assert time_range.ndim == 1
        assert np.all(np.isfinite(time_range))
        _times = self._x_time
        _values = self._y

        # remove iterations where every repeat has NaN
        num_repeats = _values.shape[1]
        nan_mask = ~np.isnan(_values).all(axis=1)
        times: np.ndarray = _times[nan_mask].reshape(-1, num_repeats)
        values: np.ndarray = _values[nan_mask].reshape(-1, num_repeats)

        # get the highest time of each run of the algorithm, take the median
        times_no_nan = times
        times_no_nan[np.isnan(values)] = np.nan  # to count only valid configurations towards highest time
        highest_time_per_repeat = np.nanmax(times_no_nan, axis=0)
        assert highest_time_per_repeat.shape[0] == num_repeats
        real_stopping_point_time: float = np.nanmedian(highest_time_per_repeat)

        # filter to get the time range with a margin on both ends for the isotonic regression
        time_range_margin = 0.1
        range_mask_margin = (time_range[0] * (1 - time_range_margin) <= times) & (
            times <= time_range[-1] * (1 + time_range_margin)
        )
        assert np.all(
            np.count_nonzero(range_mask_margin, axis=0) > 1
        ), "Not enough overlap in time range and time values"
        times = np.where(range_mask_margin, times, np.nan)
        values = np.where(range_mask_margin, values, np.nan)
        num_repeats = values.shape[1]

        # remove columns that are completely NaN
        no_nan_column_mask = ~np.all(np.isnan(values), axis=1)
        times = times[no_nan_column_mask, :]
        values = values[no_nan_column_mask, :]

        # get the shape
        num_fevals = values.shape[0]
        num_repeats = values.shape[1]

        # return the correct values
        if return_1d:
            # remove all NaNs, yielding a 1D array
            #   (because iterations has no meaning for over time anyway, and isotonic regression requires a 1D array)
            no_nan_mask = ~np.isnan(times) & ~np.isnan(
                values
            )  # only keep indices where both the times and values are not NaN
            times_1D = times[no_nan_mask]
            values_1D = values[no_nan_mask]
            assert times_1D.ndim == 1
            assert times_1D.shape == values_1D.shape
            assert np.all(~np.isnan(times_1D))
            assert np.all(~np.isnan(values_1D))
            return times_1D, values_1D, real_stopping_point_time, num_fevals, num_repeats
        else:
            return times, values, real_stopping_point_time, num_fevals, num_repeats

    def get_curve_over_time(  # noqa: D102
        self, time_range: np.ndarray, dist: np.ndarray = None, confidence_level: float = None, use_bagging=True
    ):
        # check the distribution
        if dist is None:
            raise NotImplementedError()
        assert dist.ndim == 1, "Distribution must be one-dimensional"
        dist_size = dist.shape[0]
        assert confidence_level is not None, "confidence_level must not be None"

        # use a bagging prediction / interval method or the seperated prediction / interval method
        if use_bagging:
            # get the curve within the time range
            (
                times_1D,
                values_1D,
                real_stopping_point_time,
                num_fevals,
                num_repeats,
            ) = self._get_curve_over_time_values_in_range(time_range, return_1d=True)

            # for each value, get the index in the distribution
            indices = get_indices_in_distribution(values_1D, dist)
            indices_min: int = np.min(indices)
            indices_max: int = np.max(indices)
            indices_curve = self.get_isotonic_curve(times_1D, indices, time_range, ymin=indices_min, ymax=indices_max)
            indices_curve = np.clip(np.array(np.round(indices_curve), dtype=int), a_min=0, a_max=dist_size - 1)
            curve = dist[indices_curve]

            # set data to correct shape and remove any NaN
            no_nan_mask = ~(np.isnan(times_1D) | np.isnan(indices))
            x: np.ndarray = times_1D[no_nan_mask]
            y: np.ndarray = indices[no_nan_mask]
            assert x.shape == y.shape, f"Shapes do not match: {x.shape} != {y.shape}"

            # get the lower and upper error curves
            # prediction_interval = self._get_prediction_interval_conformal(
            #     x, y, time_range, confidence_level=confidence_level, method="conformal"
            # )
            prediction_interval = self._get_prediction_interval_bagging(
                x, y, time_range, confidence_level=confidence_level, num_repeats=num_repeats
            )
        else:
            times, values, real_stopping_point_time, _, _ = self._get_curve_over_time_values_in_range(
                time_range, return_1d=False
            )
            indices = get_indices_in_distribution(values, dist)
            prediction_interval = self._get_prediction_interval_separated(
                times, indices, time_range, confidence_level=confidence_level
            )

        # round off the intervals and prediction in the correct way
        prediction_interval[:, 0] = np.floor(prediction_interval[:, 0])
        prediction_interval[:, 1] = np.ceil(prediction_interval[:, 1])
        prediction_interval = np.clip(np.array(np.round(prediction_interval), dtype=int), a_min=0, a_max=dist_size - 1)

        # get the curves
        if prediction_interval.shape[1] >= 3:
            indices_curve = prediction_interval[:, 2]
            curve = dist[indices_curve]
        curve_lower_err, curve_upper_err = dist[prediction_interval[:, 0]], dist[prediction_interval[:, 1]]
        assert (
            curve_lower_err.shape == curve_upper_err.shape == curve.shape
        ), f"{curve_lower_err.shape=} != {curve_upper_err.shape=} != {curve.shape=}"

        # print(f"{self.display_name}: {np.median(curve - curve_lower_err)}, {np.median(curve_upper_err - curve)}")
        # for t, e, i in zip(time_range, curve_lower_err, prediction[:, 0]):
        #     print(f"{t}: {e} ({i})")
        # exit(0)

        # from the real_stopping_point_time until the end of the time range,
        #   clip the values because with fewer than 50% of the repeats, as those the results can not be trusted
        assert curve.shape == time_range.shape, "Curve and time shapes must match"
        real_stopping_point_index = time_range.shape[0]
        if real_stopping_point_time < time_range[-1]:
            if real_stopping_point_time <= time_range[0]:
                raise ValueError(f"{real_stopping_point_time=} is before first time in time_range ({time_range[0]})")
            real_stopping_point_index = min(
                int(np.nonzero(time_range > real_stopping_point_time)[0][0]), real_stopping_point_index
            )  # look up the index of the stopping point
            # set everything after the index to the last value
            curve[real_stopping_point_index:] = curve[real_stopping_point_index]
            curve_lower_err[real_stopping_point_index:] = curve_lower_err[real_stopping_point_index]
            curve_upper_err[real_stopping_point_index:] = curve_upper_err[real_stopping_point_index]

        return self._get_curve_split_real_fictional_parts(
            real_stopping_point_index, time_range, curve, curve_lower_err, curve_upper_err
        )

    def get_split_times(  # noqa: D102
        self, range: np.ndarray, x_type: str, searchspace_stats: SearchspaceStatistics
    ) -> np.ndarray:
        return super().get_split_times(range, x_type, searchspace_stats)

    def get_split_times_at_feval(  # noqa: D102
        self, fevals_range: np.ndarray, searchspace_stats: SearchspaceStatistics
    ) -> np.ndarray:
        # get the indices in the range
        matching_indices_mask = self._get_matching_feval_indices_in_range(fevals_range)
        objective_time_keys = searchspace_stats.objective_time_keys
        num_keys = len(objective_time_keys)
        num_repeats = matching_indices_mask.shape[1]
        masked_time_per_key = np.full((num_keys, matching_indices_mask.shape[0], num_repeats), np.NaN)

        # for each key, apply the boolean mask
        for key_index in range(num_keys):
            masked_time_per_key[key_index, matching_indices_mask] = self._x_time_per_key[
                key_index, matching_indices_mask
            ]

        # remove where every repeat has NaN
        time_in_range_per_key = np.full((num_keys, fevals_range.shape[0], num_repeats), np.NaN)
        for key_index in range(num_keys):
            all_nan_mask = ~np.all(np.isnan(masked_time_per_key[key_index]), axis=1)
            time_in_range_per_key[key_index] = masked_time_per_key[key_index][all_nan_mask]

        # get the median time per key at each repeat
        split_time_per_feval = np.full((num_keys, fevals_range.shape[0]), np.NaN)
        for key_index in range(num_keys):
            split_time_per_feval[key_index] = np.mean(time_in_range_per_key[key_index], axis=1)
        assert split_time_per_feval.shape == (
            num_keys,
            fevals_range.shape[0],
        ), f"{split_time_per_feval.shape} != {(num_keys, fevals_range.shape[0])}"
        return split_time_per_feval

    def get_split_times_at_time(  # noqa: D102
        self, time_range: np.ndarray, searchspace_stats: SearchspaceStatistics
    ) -> np.ndarray:
        # get the raw times
        nan_mask = ~np.isnan(self._x_time)
        times_total = self._x_time[nan_mask]
        times_split = self._x_time_per_key[:, nan_mask]

        # for each key, interpolate the split times to the time range
        num_keys = len(searchspace_stats.objective_time_keys)
        split_time_per_timestamp = np.full((num_keys, time_range.shape[0]), np.NaN)
        for key_index in range(num_keys):
            # remove NaN
            times_split_key = times_split[key_index]
            nan_mask = ~np.isnan(times_split_key)
            times_total_key = times_total[nan_mask]
            times_split_key = times_split_key[nan_mask]
            assert times_total_key.shape == times_split_key.shape, "The shapes of split times must match the original."
            # interpolate the times to the time range
            split_time_per_timestamp[key_index] = np.interp(time_range, times_total_key, times_split_key)

        return split_time_per_timestamp

    def get_confidence_interval(self, values: np.ndarray, confidence_level: float) -> tuple[np.ndarray, np.ndarray]:
        """Calculates the non-parametric confidence interval at each function evaluation across repeats.

        Observations are assumed to be IID.

        Args:
            values: the two-dimensional values to calculate the confidence interval on.
            confidence_level: the confidence level for the confidence interval.

        Raises:
            NotImplementedError: when weights is passed.

        Returns:
            A tuple of two NumPy arrays with lower and upper error, respectively.
        """
        assert values.ndim == 2, "The values need to be two-dimensional (iterations, repeats)"

        # confidence interval using normal distribution assumption
        from statistics import NormalDist

        distribution = NormalDist()
        z = distribution.inv_cdf((1 + confidence_level) / 2.0)
        n = values.shape[1]
        q = 0.5
        nq = n * q
        base = z * sqrt(nq * (1 - q))
        lower_rank = max(floor(nq - base), 0)
        upper_rank = min(ceil(nq + base) + 1, n - 1)
        confidence_interval_lower = np.full(values.shape[0], np.nan)
        confidence_interval_upper = np.full(values.shape[0], np.nan)

        # # confidence interval according to Hoefler 2015 (student-t)
        # alpha = 1 - confidence_level
        # mean = values.mean()
        # t = t(n - 1, alpha / 2)

        # for each function evaluation, look up the confidence interval
        fevals_repeats_sorted = np.sort(values, axis=1)
        for feval_index, feval_repeats_sorted in enumerate(fevals_repeats_sorted):
            confidence_interval_lower[feval_index] = feval_repeats_sorted[lower_rank]
            confidence_interval_upper[feval_index] = feval_repeats_sorted[upper_rank]

        return confidence_interval_lower, confidence_interval_upper

    def _get_prediction_interval_separated(
        self, times: np.ndarray, values: np.ndarray, time_range: np.ndarray, confidence_level: float
    ) -> np.ndarray:
        """Calculates the prediction interval and isotonic regression mean by separating the runs."""
        assert times.shape == values.shape
        assert values.ndim == 2
        num_fevals = values.shape[0]
        num_repeats = values.shape[1]

        # predict an isotonic curve for the time range for each run
        predictions = np.full((num_repeats, time_range.shape[0]), fill_value=np.NaN)
        for run in range(num_repeats):
            # get the data of this run
            _x = times[:, run]
            _y = values[:, run]
            assert _x.shape[0] == _y.shape[0] == num_fevals

            # filter NaN
            no_nan_mask = ~np.isnan(_x) & ~np.isnan(_y)  # only keep indices where both the times and values are not NaN
            _x = _x[no_nan_mask]
            _y = _y[no_nan_mask]
            assert _x.ndim == _y.ndim == 1
            assert np.all(~np.isnan(_x))
            assert np.all(~np.isnan(_y))
            fraction_valid = _y.shape[0] / num_fevals
            if fraction_valid < 0.05:
                warn(
                    f"{round(fraction_valid * 100, 1)}% data left after removing NaN ({_y.shape[0]} / {num_fevals})",
                    category=UserWarning,
                )

            # get the prediction
            predictions[run] = self.get_isotonic_curve(_x, _y, time_range)

        # extract the mean and prediction intervals
        assert np.all(~np.isnan(predictions))
        predictions = predictions.transpose()  # set to (time_range, num_repeats)
        y_lower_err, y_upper_err = self.get_confidence_interval(predictions, confidence_level=confidence_level)
        mean_prediction = np.median(predictions, axis=1)
        assert (
            y_lower_err.shape == y_upper_err.shape == mean_prediction.shape == time_range.shape
        ), f"{y_lower_err.shape=} != {y_upper_err.shape=} != {mean_prediction.shape=} != {time_range.shape=}"

        # combine the data and return as a prediction interval
        prediction_interval = np.concatenate([y_lower_err, y_upper_err, mean_prediction]).reshape((3, -1)).transpose()
        return prediction_interval

    def _get_prediction_interval_bagging(
        self, x_1d: np.ndarray, y_1d: np.ndarray, x_test_1d: np.ndarray, confidence_level: float, num_repeats: int
    ) -> np.ndarray:
        """Calculates the prediction interval and isotonic regression mean using a bootstrap bagging method."""
        # prepare the data
        x = x_1d.reshape(-1, 1)
        x_test = x_test_1d.reshape(-1, 1)
        y = y_1d

        # set the parameters
        # based on the number of repeats,
        #   where the number of estimators is equal to the number of repeats and the fraction of samples is
        #   inversely proportional to the square root of the number of estimators.
        #   This way, the reuse of data when bootstrapping is limited.
        n_estimators = max(num_repeats, 3)
        max_samples = 1 / np.sqrt(n_estimators)
        # alternative parameters (with max_samples this way, on average each datum is used only once).
        # n_estimators = max(round(np.sqrt(num_repeats)), 3)
        # n_estimators = max(round(np.log2(num_repeats**2)), 3)
        # max_samples = 1 / n_estimators

        # do the bootstrap bagging
        regression_model = self.get_isotonic_regressor(y_min=y.min(), y_max=y.max())
        bagging_regressor = BaggingRegressor(
            regression_model, n_estimators=n_estimators, max_samples=max_samples, bootstrap=True
        )
        br = bagging_regressor.fit(x, y)  # fit the data to the estimators
        br_prediction = br.predict(x_test)
        br_collection = np.array(
            [est.predict(x_test) for est in br.estimators_]
        )  # predict for each estimator; yields array with shape (run, x_test)

        # get the prediction interval in the correct shape
        y_lower_err, y_upper_err = self.get_confidence_interval(
            br_collection.transpose(), confidence_level=confidence_level
        )
        prediction_interval = np.concatenate([y_lower_err, y_upper_err, br_prediction]).reshape((3, -1)).transpose()
        assert prediction_interval.shape == (x_test.shape[0], 3), f"{prediction_interval.shape}"
        return prediction_interval

    def _get_prediction_interval_conformal(
        self,
        x_1d: np.ndarray,
        y_1d: np.ndarray,
        x_test_1d: np.ndarray,
        confidence_level: float,
        method="inductive_conformal",
        train_fraction: float = 0.75,
    ) -> np.ndarray:
        """Calculates the prediction interval using various conformal methods."""
        methods = ["inductive_conformal"]
        assert method in methods
        assert 0 < train_fraction < 1

        # divide data into training and calibration set (75%-25%)
        N = x_1d.shape[0]
        calibration_cutoff = round(N * train_fraction)
        random_indices = np.random.permutation(N)
        indices_train, indices_calibrate = random_indices[:calibration_cutoff], random_indices[calibration_cutoff:]
        assert len(indices_train) + len(indices_calibrate) == N

        # set the correct shape of the data
        y = y_1d
        y_train = y[indices_train]
        y_cal = y[indices_calibrate]
        to_2d = True
        if to_2d:
            x = x_1d.reshape(-1, 1)
            x_test = x_test_1d.reshape(-1, 1)
            x_train = x[indices_train, :]
            x_cal = x[indices_calibrate, :]
        else:
            x = x_1d
            x_test = x_test_1d
            x_train = x[indices_train]
            x_cal = x[indices_calibrate]

        # create the regression model
        regression_model = self.get_isotonic_regressor(y_min=y.min(), y_max=y.max())

        # Inductive Conformal Point Prediction (based on https://arxiv.org/pdf/2107.00363.pdf, page 16 & 17)
        from nonconformist.cp import IcpRegressor
        from nonconformist.nc import NcFactory, SignErrorErrFunc

        # create the nonconformity function and inductive conformal regressor
        nonconformity_function = NcFactory.create_nc(regression_model, err_func=SignErrorErrFunc())
        inductive_conformal_regressor = IcpRegressor(nonconformity_function)

        # fit and calibrate the ICP
        inductive_conformal_regressor.fit(x_train, y_train)
        inductive_conformal_regressor.calibrate(x_cal, y_cal)

        # get the interval on the prediction
        prediction_interval = inductive_conformal_regressor.predict(x_test, significance=1 - confidence_level)

        return prediction_interval
