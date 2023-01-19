from abc import ABC, abstractmethod
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


class RandomBaseline(Curve):

    def __init__(self, device_name: str, kernel_name: str) -> None:
        self.name = "randomsearch_baseline"
        self.display_name = "Random Search baseline"
        self.device_name = device_name
        self.kernel_name = kernel_name
        self.stochastic = False

        self._x_fevals = None
        self._x_time = None
        self._y = None

        self.check_attributes()

    def get_curve_over_fevals(self, fevals_range: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
        return super().get_curve_over_fevals(curve)

    def get_curve_over_time(self, time_range: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
        return super().get_curve_over_time(time)


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

    def get_curve_over_fevals(self, fevals_range: np.ndarray, dist: np.ndarray = None) -> np.ndarray:
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
            indices_mean = np.mean(indices, axis=1)
            indices_mean_rounded = np.array(np.round(indices_mean), dtype=int)
            # obtain the curve by looking up the associated values
            curve = dist[indices_mean_rounded]
        else:
            # obtain the curve
            curve = np.nanmean(masked_values, axis=1)    # get the curve by taking the mean
        curve = curve[~np.isnan(curve)]    # remove remaining NaN, yielding an array which <= fevals.shape
        pad_width = self.fevals_find_pad_width(fevals, fevals_range)
        curve = np.pad(curve, pad_width=pad_width, constant_values=np.nan)    # pad with NaN where outside the range, yielding an array which == fevals.shape
        assert curve.shape == fevals_range.shape
        return curve

    def get_curve_over_time(self, time_range: np.ndarray) -> np.ndarray:
        return super().get_curve_over_time(time_range)
