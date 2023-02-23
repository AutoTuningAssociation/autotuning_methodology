from abc import ABC, abstractmethod
from typing import Tuple, Union
import numpy as np
from caching import ResultsDescription
from searchspace_statistics import SearchspaceStatistics
from math import floor, ceil, sqrt
import warnings


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
        self.minimization = results_description.minimization

        # result data
        results = results_description.get_results()
        self._x_fevals = results.fevals_results    # the time per objective value in number of function evaluations since start (1d if deterministic, 2d if stochastic)
        self._x_time = results.objective_time_results    # the time per objective value in seconds since start the raw x-axis (1d if deterministic, 2d if stochastic)
        self._y = results.objective_performance_best_results    # the objective performances (1d if deterministic, 2d if stochastic)

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
    def get_curve(self, range: np.ndarray, x_type: str, dist: np.ndarray = None, confidence_level: float = None):
        """ Get the curve over the specified range of time or function evaluations, returns a tuple of NDArrays with NaN beyond limits. """
        if x_type == 'fevals':
            return self.get_curve_over_fevals(range, dist, confidence_level)
        elif x_type == 'time':
            return self.get_curve_over_time(range, dist, confidence_level)
        raise ValueError(f"x_type must be 'fevals' or 'time', is {x_type}")

    @abstractmethod
    def get_curve_over_fevals(self, fevals_range: np.ndarray, dist: np.ndarray = None, confidence_level: float = None):
        """ Get the real_stopping_point_fevals and the real and fictional curve, errors over the specified range of function evaluations """
        raise NotImplementedError

    @abstractmethod
    def get_curve_over_time(self, time_range: np.ndarray, dist: np.ndarray = None, confidence_level: float = None):
        """ Get the real_stopping_point_time and the real and fictional curve, errors at the specified times using isotonic regression """
        raise NotImplementedError

    @abstractmethod
    def get_split_times_at_feval(self, fevals_range: np.ndarray, searchspace_stats: SearchspaceStatistics) -> np.ndarray:
        """ Get the times at each function eval in the range split into objective_time_keys """
        raise NotImplementedError()

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

    def get_scatter_data(self, x_type: str) -> Tuple[np.ndarray, np.ndarray]:
        if x_type == 'fevals':
            return self._x_fevals, self._y
        elif x_type == 'time':
            return self._x_time, self._y
        raise ValueError(f"x_type must be 'fevals' or 'time', is {x_type}")

    def get_isotonic_curve(self, x: np.ndarray, y: np.ndarray, x_new: np.ndarray, package='isotonic', npoints=1000, power=2, ymin=None,
                           ymax=None) -> np.ndarray:
        """ Get the isotonic regression curve fitted to x_new using package 'sklearn' or 'isotonic' """
        # check if the assumptions that the input arrays are numpy arrays holds
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(x_new, np.ndarray)
        assert x_new.ndim == 1

        if x.ndim > 1:
            x = x.flatten()
        if y.ndim > 1:
            y = y.flatten()

        increasing = not self.minimization
        if package == 'sklearn':
            from sklearn.isotonic import IsotonicRegression
            import warnings
            if npoints != 1000:
                warnings.warn("npoints argument is impotent for sklearn package")
            if power != 2:
                warnings.warn("power argument is impotent for sklearn package")
            ir = IsotonicRegression(increasing=increasing, y_min=ymin, y_max=ymax, out_of_bounds='clip')
            ir.fit(x, y)
            return ir.predict(x_new)
        elif package == 'isotonic':
            from isotonic.isotonic import LpIsotonicRegression
            ir = LpIsotonicRegression(npoints, increasing=increasing, power=power).fit(x, y)
            y_isotonic_regression = ir.predict_proba(x_new)
            # TODO check if you are not indadvertedly clipping too much here
            if ymin is not None or ymax is not None:
                y_isotonic_regression = np.clip(y_isotonic_regression, ymin, ymax)
            return y_isotonic_regression
        raise ValueError(f"Package name {package} is not a valid package name")


class DeterministicOptimizationAlgorithm(Curve):

    def get_curve(self, range: np.ndarray, x_type: str, dist: np.ndarray = None, confidence_level: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple([super().get_curve(range, x_type, dist, confidence_level), None, None])

    def get_curve_over_fevals(self, fevals_range: np.ndarray, dist: np.ndarray = None, confidence_level: float = None) -> np.ndarray:
        return super().get_curve_over_fevals(fevals_range, dist, confidence_level)

    def get_curve_over_time(self, time_range: np.ndarray, dist: np.ndarray = None, confidence_level: float = None) -> np.ndarray:
        return super().get_curve_over_time(time_range, dist, confidence_level)

    def get_split_times_at_feval(self, fevals_range: np.ndarray, searchspace_stats: SearchspaceStatistics) -> np.ndarray:
        return super().get_split_times_at_feval(fevals_range, searchspace_stats)


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

    def check_curve_real_fictional_consistency(self, x_axis_range, curve, curve_lower_err, curve_upper_err, x_axis_range_real, curve_real, curve_lower_err_real,
                                               curve_upper_err_real, x_axis_range_fictional, curve_fictional, curve_lower_err_fictional,
                                               curve_upper_err_fictional):
        """ Asserts that the real and fictional results add up correctly """
        assert x_axis_range.shape == curve.shape == curve_lower_err.shape == curve_upper_err.shape, f"Shapes must be equal: {x_axis_range.shape=}, {curve.shape=}, {curve_lower_err.shape=}, {curve_upper_err.shape=}"
        assert x_axis_range_real.shape == curve_real.shape == curve_lower_err_real.shape == curve_upper_err_real.shape, f"Shapes must be equal: {x_axis_range_real.shape=}, {curve_real.shape=}, {curve_lower_err_real.shape=}, {curve_upper_err_real.shape=}"
        assert x_axis_range_fictional.shape == curve_fictional.shape == curve_lower_err_fictional.shape == curve_upper_err_fictional.shape, f"Shapes must be equal: {x_axis_range_fictional.shape=}, {curve_fictional.shape=}, {curve_lower_err_fictional.shape=} {curve_upper_err_fictional.shape=}"
        if x_axis_range_fictional.ndim > 0:
            assert np.array_equal(x_axis_range, np.concatenate([x_axis_range_real, x_axis_range_fictional]), equal_nan=True)
            assert np.array_equal(curve, np.concatenate([curve_real, curve_fictional]), equal_nan=True)
            assert np.array_equal(curve_lower_err, np.concatenate([curve_lower_err_real, curve_lower_err_fictional]), equal_nan=True)
            assert np.array_equal(curve_upper_err, np.concatenate([curve_upper_err_real, curve_upper_err_fictional]), equal_nan=True)
        else:
            assert np.array_equal(x_axis_range, x_axis_range_real, equal_nan=True), f"Unequal arrays: {x_axis_range}, {x_axis_range_real}"
            assert np.array_equal(curve, curve_real, equal_nan=True), f"Unequal arrays: {curve}, {curve_real}"
            assert np.array_equal(curve_lower_err, curve_lower_err_real, equal_nan=True), f"Unequal arrays: {curve_lower_err}, {curve_lower_err_real}"
            assert np.array_equal(curve_upper_err, curve_upper_err_real, equal_nan=True), f"Unequal arrays: {curve_upper_err}, {curve_upper_err_real}"

    def get_curve(self, range: np.ndarray, x_type: str, dist: np.ndarray = None, confidence_level: float = None):
        return super().get_curve(range, x_type, dist, confidence_level)

    def _get_curve_over_fevals_values_in_range(self, fevals_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Get the valid fevals and values that are in the given range """
        assert fevals_range.ndim == 1
        assert np.all(np.isfinite(fevals_range))
        target_index: int = fevals_range[-1] - 1

        # filter to only get data in the fevals range
        matching_indices_mask = np.array([np.isin(x_column, fevals_range, assume_unique=True)
                                          for x_column in self._x_fevals.T]).transpose()    # get the indices of the matching feval range per repeat (column)
        masked_values = np.where(matching_indices_mask, self._y, np.nan)    # apply the mask to the values, filling NaN for False
        masked_fevals = np.where(matching_indices_mask, self._x_fevals, np.nan).transpose()    # apply the mask to the fevals, filling NaN for False

        # make sure that the filtered fevals are consistent (every repeat has the same array of fevals)
        if not np.allclose(masked_fevals, masked_fevals[0], equal_nan=True):
            indices = np.nanargmax(masked_fevals, axis=1)    # get the index of the last non-nan value of each repeat
            # drop the data which ends before the index
            keep_data = np.where(indices >= target_index)
            greatest_common_non_NaN_index = min(floor(np.median(indices)), target_index)
            if np.count_nonzero(keep_data) < len(indices) * 0.5:
                warnings.warn(
                    f"For optimization algorithm {self.display_name}, more than 50% of the runs ended before the end of fevals_range ({target_index + 1}). Only data up to {greatest_common_non_NaN_index + 1} fevals will be used. Perhaps increase the allotted auto-tuning time for this optimization algorithm?"
                )
            else:
                warnings.warn(
                    f"Dropped {len(indices) - np.count_nonzero(keep_data)} repeats of {self.display_name} runs that ended before the end of fevals_range ({target_index + 1}), perhaps increase the allotted auto-tuning time for this optimization algorithm?",
                    UserWarning)
            masked_fevals = masked_fevals[keep_data]
            masked_values = masked_values.transpose()
            masked_values = masked_values[keep_data]

            # set all values beyond the greatest common non-NaN index to NaN
            masked_values[:, greatest_common_non_NaN_index + 1:] = np.nan
            masked_values = masked_values.transpose()    # transpose back to original shape
            masked_fevals[:, greatest_common_non_NaN_index + 1:] = np.nan
            print(f"{greatest_common_non_NaN_index=}")

            # check that the filtered fevals are consistent
            assert np.allclose(masked_fevals, masked_fevals[0], equal_nan=True), "Every repeat must have the same array of function evaluations"

        # as every repeat has the same array of fevals, check whether they match the range
        fevals = masked_fevals[0]    # safe to assume as every repeat has the same array of fevals, set it before removing NaN to pad the curve
        masked_fevals = masked_fevals.transpose()    # transpose back to original shape

        # remove fevals where every repeat has NaN
        num_repeats = masked_values.shape[1]
        nan_mask = ~np.isnan(masked_values).all(axis=1)
        masked_fevals = masked_fevals[nan_mask].reshape(-1, num_repeats)
        masked_values = masked_values[nan_mask].reshape(-1, num_repeats)

        # this no longer holds as it is possible for optimization algorithms to end early
        # assert fevals_range.shape[0] == masked_values.shape[0] == masked_fevals.shape[0], f"The masked fevals and values should have the same first dimension as fevals_range, but {fevals_range.shape[0]=}, {masked_fevals.shape[0]=}, {masked_values.shape[0]=}"
        return fevals, masked_values

    def get_curve_over_fevals(
            self, fevals_range: np.ndarray, dist: np.ndarray = None,
            confidence_level: float = None) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fevals, masked_values = self._get_curve_over_fevals_values_in_range(fevals_range)

        # if a distribution is included
        if dist is not None:
            assert dist.ndim == 1
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
                # get the confidence interval
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
                # get the confidence interval
                curve_lower_err, curve_upper_err = self.get_confidence_interval(masked_values, confidence_level)

        # # remove remaining NaN, yielding an array which <= fevals.shape
        # curve = curve[~np.isnan(curve)]
        # curve_lower_err = curve_lower_err[~np.isnan(curve_lower_err)]
        # curve_upper_err = curve_upper_err[~np.isnan(curve_upper_err)]

        # pad with NaN where outside the range, yielding an array.shape == fevals.shape
        real_stopping_point_fevals = curve.shape[0]
        real_stopping_point_index = real_stopping_point_fevals
        if curve.shape != fevals_range.shape:
            pad_width = self.fevals_find_pad_width(fevals, fevals_range)
            curve = np.pad(curve, pad_width=pad_width, constant_values=np.nan)
            curve_lower_err = np.pad(curve_lower_err, pad_width=pad_width, constant_values=np.nan)
            curve_upper_err = np.pad(curve_upper_err, pad_width=pad_width, constant_values=np.nan)
        assert curve.shape == fevals_range.shape

        # if necessary, extend the curves up to target_index
        target_index: int = fevals_range[-1] - 1
        if real_stopping_point_index < target_index:
            # warnings.warn(
            #     f"For optimization algorithm {self.display_name}, all runs end at {real_stopping_point_index + 1} fevals, which is before the target number of function evals of {target_index + 1}."
            # )
            # take the last non-NaN value and overwrite the curves up to the target index with it
            curve[real_stopping_point_index:target_index] = curve[real_stopping_point_index]
            curve_lower_err[real_stopping_point_index:target_index] = curve_lower_err[real_stopping_point_index]
            curve_upper_err[real_stopping_point_index:target_index] = curve_upper_err[real_stopping_point_index]

        # select the parts of the data that are real
        fevals_range_real = fevals_range[:real_stopping_point_index]
        curve_real = curve[:real_stopping_point_index]
        curve_lower_err_real = curve_lower_err[:real_stopping_point_index]
        curve_upper_err_real = curve_upper_err[:real_stopping_point_index]

        # select the parts of the data that are fictional
        fevals_range_fictional, curve_fictional, curve_lower_err_fictional, curve_upper_err_fictional = np.ndarray([]), np.ndarray([]), np.ndarray(
            []), np.ndarray([])
        if real_stopping_point_index < target_index:
            fevals_range_fictional = fevals_range[real_stopping_point_index:]
            curve_fictional = curve[real_stopping_point_index:]
            curve_lower_err_fictional = curve_lower_err[real_stopping_point_index:]
            curve_upper_err_fictional = curve_upper_err[real_stopping_point_index:]

        # check and return
        self.check_curve_real_fictional_consistency(fevals_range, curve, curve_lower_err, curve_upper_err, fevals_range_real, curve_real, curve_lower_err_real,
                                                    curve_upper_err_real, fevals_range_fictional, curve_fictional, curve_lower_err_fictional,
                                                    curve_upper_err_fictional)
        return real_stopping_point_fevals, fevals_range_real, curve_real, curve_lower_err_real, curve_upper_err_real, fevals_range_fictional, curve_fictional, curve_lower_err_fictional, curve_upper_err_fictional

    def get_curve_over_time(
            self, time_range: np.ndarray, dist: np.ndarray = None,
            confidence_level: float = None) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert time_range.ndim == 1
        assert np.all(np.isfinite(time_range))
        if dist is not None:
            assert dist.ndim == 1

        times = self._x_time
        values = self._y

        # remove iterations where every repeat has NaN
        num_repeats = values.shape[1]
        nan_mask = ~np.isnan(values).all(axis=1)
        times: np.ndarray = times[nan_mask].reshape(-1, num_repeats)
        values: np.ndarray = values[nan_mask].reshape(-1, num_repeats)

        # get the highest time of each repeat, take the median
        times_no_nan = times
        times_no_nan[np.isnan(values)] = np.nan    # to count only valid configurations towards highest time
        highest_time_per_repeat = np.nanmax(times_no_nan, axis=0)
        assert highest_time_per_repeat.shape[0] == num_repeats
        real_stopping_point_time = np.nanmedian(highest_time_per_repeat)

        # filter to get the time range with a margin on both ends for the isotonic regression
        time_range_margin = 0.1
        range_mask_margin = (time_range[0] * (1 - time_range_margin) <= times) & (times <= time_range[-1] * (1 + time_range_margin))
        assert np.all(np.count_nonzero(range_mask_margin, axis=0) > 1), "Not enough overlap in time range and time values"
        times = np.where(range_mask_margin, times, np.nan)
        values = np.where(range_mask_margin, values, np.nan)
        num_fevals, num_repeats = values.shape

        # remove all NaNs, yielding a 1D array (because iterations has no meaning for over time anyway, and isotonic regression requires a 1D array)
        no_nan_mask = ~np.isnan(times) & ~np.isnan(values)    # only keep indices where both the times and values are not NaN
        times_1D = times[no_nan_mask]
        values_1D = values[no_nan_mask]
        assert times_1D.ndim == 1
        assert times_1D.shape == values_1D.shape
        assert np.all(~np.isnan(times_1D))
        assert np.all(~np.isnan(values_1D))

        # if a distribution is included
        if dist is not None:
            # for each value, get the index in the distribution
            indices = self._get_indices(values_1D, dist)
            indices_curve = self.get_isotonic_curve(times_1D, indices, time_range, npoints=num_fevals, package='sklearn')
            indices_curve = np.array(np.round(indices_curve), dtype=int)
            curve = dist[indices_curve]
        else:
            # obtain the curves
            raise NotImplementedError()

        # filter to only get the time range (for the binned error calculation)
        range_mask = (time_range[0] <= times) & (times <= time_range[-1])
        assert np.all(np.count_nonzero(range_mask, axis=0) > 1), "Not enough overlap in time range and time values"
        masked_times = np.where(range_mask, times, np.nan)
        masked_values = np.where(range_mask, values, np.nan)
        assert masked_times.ndim == 2
        assert masked_times.shape == masked_values.shape

        # bin the values to their closest point in low resolution time_range
        time_range_low_res = np.linspace(time_range[0], time_range[-1],
                                         num=round(num_fevals / 10))    # should result in on average num_repeat observations per bin
        bins = [[] for _ in range(len(time_range_low_res))]
        for multi_index, value in np.ndenumerate(masked_values):
            # for each element look up the index of the closest point in time_range, write the value to this bin
            if not np.isnan(value):
                index = (np.abs(time_range_low_res - masked_times[multi_index])).argmin()
                bins[index].append(value)

        # calculate the confidence interval for each bin
        bins = list([np.array(bin) for bin in bins])
        if confidence_level is None:
            # get the standard error, interpolate missing bins
            curve_std: np.ndarray = np.full_like(time_range_low_res, np.nan)
            for bin_index, bin in enumerate(bins):
                # at least three non-zero values must be present to calculate the standard error
                if np.count_nonzero(~np.isnan(bin)) >= 3:
                    curve_std[bin_index] = np.nanstd(bin)
            # filter out where NaN
            nan_mask = ~np.isnan(curve_std)
            curve_std = curve_std[nan_mask]
            time_range_low_res = time_range_low_res[nan_mask]
            # interpolate missing bins
            curve_std = np.interp(time_range, time_range_low_res, curve_std)
            curve_lower_err = curve - curve_std
            curve_upper_err = curve + curve_std
        else:
            # calculate in bins, interpolate missing bins
            # TODO make sure this is appropriate, calculating the confidence interval of the isotonic curve instead of the median
            curve_lower_err, curve_upper_err = self.get_confidence_interval_jagged(bins, confidence_level)
            curve_lower_err, curve_upper_err = np.interp(time_range, time_range_low_res,
                                                         curve_lower_err), np.interp(time_range, time_range_low_res, curve_upper_err)
            # alternative: calculate using get_confidence_interval, interpolate to time_range afterwards (cons: naive assumption that the times roughly match per function evaluation)
            # curve_lower_err, curve_upper_err = self.get_confidence_interval(values, confidence_level)

        # pad with NaN where outside the range, yielding an array.shape == fevals.shape
        assert curve.shape == time_range.shape

        # from the real_stopping_point_time until the end of the time range, clip the values because with fewer than 50% of the repeats, as those the results can not be trusted
        real_stopping_point_index = time_range.shape[0]
        if real_stopping_point_time < time_range[-1]:
            if real_stopping_point_time <= time_range[0]:
                raise ValueError(f"{real_stopping_point_time=} is before first time in time_range ({time_range[0]})")
            real_stopping_point_index = min(int(np.nonzero(time_range > real_stopping_point_time)[0][0]),
                                            real_stopping_point_index)    # look up the index of the stopping point
            # set everything after the index to the last value
            curve[real_stopping_point_index:] = curve[real_stopping_point_index]
            curve_lower_err[real_stopping_point_index:] = curve_lower_err[real_stopping_point_index]
            curve_upper_err[real_stopping_point_index:] = curve_upper_err[real_stopping_point_index]

        # select the parts of the data that are real
        time_range_real = time_range[:real_stopping_point_index]
        curve_real = curve[:real_stopping_point_index]
        curve_lower_err_real = curve_lower_err[:real_stopping_point_index]
        curve_upper_err_real = curve_upper_err[:real_stopping_point_index]

        # select the parts of the data that are fictional
        time_range_fictional, curve_fictional, curve_lower_err_fictional, curve_upper_err_fictional = np.ndarray([]), np.ndarray([]), np.ndarray(
            []), np.ndarray([])
        if real_stopping_point_time < time_range[-1]:
            time_range_fictional = time_range[real_stopping_point_index:]
            curve_fictional = curve[real_stopping_point_index:]
            curve_lower_err_fictional = curve_lower_err[real_stopping_point_index:]
            curve_upper_err_fictional = curve_upper_err[real_stopping_point_index:]

        # check and return
        self.check_curve_real_fictional_consistency(time_range, curve, curve_lower_err, curve_upper_err, time_range_real, curve_real, curve_lower_err_real,
                                                    curve_upper_err_real, time_range_fictional, curve_fictional, curve_lower_err_fictional,
                                                    curve_upper_err_fictional)
        return real_stopping_point_time, time_range_real, curve_real, curve_lower_err_real, curve_upper_err_real, time_range_fictional, curve_fictional, curve_lower_err_fictional, curve_upper_err_fictional

    def get_split_times_at_feval(self, fevals_range: np.ndarray, searchspace_stats: SearchspaceStatistics) -> np.ndarray:
        fevals, masked_values = self._get_curve_over_fevals_values_in_range(fevals_range)
        indices = self._get_indices(masked_values, searchspace_stats.objective_performances_total)
        average_index_at_feval = np.array(np.round(np.nanmedian(indices, axis=1)), dtype=int)

        # for each key, obtain the time at a feval
        objective_time_keys = searchspace_stats.objective_time_keys
        split_time_per_feval = np.empty((len(objective_time_keys), average_index_at_feval.shape[0]))
        for key_index, key in enumerate(objective_time_keys):
            split_time_per_feval[key_index] = searchspace_stats.objective_times_array[key_index, average_index_at_feval]

        return split_time_per_feval

    def get_confidence_interval(self, values: np.ndarray, confidence_level: float) -> Tuple[np.ndarray, np.ndarray]:
        """ Calculates the non-parametric confidence interval for repeated function evaluations, assumed to be IID """
        assert values.ndim == 2    # should be two-dimensional (iterations, repeats)

        # confidence interval using normal distribution assumption
        from statistics import NormalDist
        distribution = NormalDist()    # TODO check if binomial is more appropriate (calculate according to book)
        z = distribution.inv_cdf((1 + confidence_level) / 2.)
        n = values.shape[1]
        q = 0.5
        nq = n * q
        base = z * sqrt(nq * (1 - q))
        lower_rank = max(floor(nq - base), 0)
        upper_rank = min(ceil(nq + base), n - 1)
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

    def get_confidence_interval_jagged(self, bins: list[np.ndarray], confidence_level: float) -> Tuple[np.ndarray, np.ndarray]:
        """ Calculates the non-parametric confidence interval for jagged bins, assumed to be IID, slower than get_confidence_interval() """
        confidence_interval_lower = np.full(len(bins), np.nan)
        confidence_interval_upper = np.full(len(bins), np.nan)

        # confidence interval using normal distribution assumption
        from statistics import NormalDist
        distribution = NormalDist()    # TODO check if binomial is more appropriate (calculate according to book)
        z = distribution.inv_cdf((1 + confidence_level) / 2.)
        q = 0.5

        # for each bin, look up the confidence interval
        for feval_index, bin in enumerate(bins):
            n = len(bin)
            if n <= 0:
                continue
            bin = np.sort(bin)
            base = z * sqrt(n * q * (1 - q))
            lower_rank = max(floor(n * q - base), 0)
            upper_rank = min(ceil(n * q + base), n - 1)
            confidence_interval_lower[feval_index] = bin[lower_rank]
            confidence_interval_upper[feval_index] = bin[upper_rank]

        # interpolate missing values
        nan_mask = np.isnan(confidence_interval_lower)    # is the same is np.isnan(confidence_interval_upper)
        confidence_interval_lower[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), confidence_interval_lower[~nan_mask])
        confidence_interval_upper[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), confidence_interval_upper[~nan_mask])

        # check before returning
        assert not np.isnan(confidence_interval_lower).any()
        assert not np.isnan(confidence_interval_upper).any()
        return confidence_interval_lower, confidence_interval_upper
