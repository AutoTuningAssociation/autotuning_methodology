from abc import ABC, abstractmethod
import numpy as np
from caching import ResultsDescription


class Curve(ABC):

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
            assert self._x_time.ndim == 1
            assert self._y.ndim == 1
        else:
            assert self._x_fevals.ndim == 2
            assert self._x_time.ndim == 2
            assert self._y.ndim == 2
        assert self._x_fevals.shape == self._x_time.shape == self._y.shape

    @abstractmethod
    def get_curve_over_fevals(self, fevals: np.ndarray) -> np.ndarray:
        """ Get the curve over the specified range of function evaluations, returns NaN beyond limits. """
        return fevals

    @abstractmethod
    def get_curve_over_time(self, time: np.ndarray) -> np.ndarray:
        """ Get the curve at the specified times using isotonic regression, returns NaN beyond limits. """
        return time

    def pad_array_fevals(self, array: np.ndarray, target_array: np.ndarray) -> np.ndarray:
        """ Pad an array of fevals with NaNs based on the target array, assumes both arrays are incremented by one feval per element """
        # TODO remove assumption of increment by one by rewriting to checking whether array is consecutively in target_array, taking the difference in starts and ends of target_array
        if array.ndim == 1:
            limit_difference_start = int(np.max([array[0] - target_array[0], 0]))    # find the number of fevals missing before the start
            limit_difference_end = int(np.max([target_array[-1] - array[-1], 0]))    # find the number of fevals missing after the end
        elif array.ndim == 2:
            limit_difference_start = int(np.max([array[0][0] - target_array[0], 0]))    # find the number of fevals missing before the start
            limit_difference_end = int(np.max([target_array[-1] - array[-1][0], 0]))    # find the number of fevals missing after the end
        else:
            raise NotImplementedError("Padding arrays beyond two dimensions is not implemented")
        array = np.pad(array, pad_width=((limit_difference_start, limit_difference_end)), constant_values=np.nan)
        assert array.shape == target_array.shape    # verify that the curve is now in the requested shape
        return array


class RandomBaseline(Curve):
    pass


class DeterministicOptimizationAlgorithm(Curve):

    def get_curve_over_fevals(self, fevals: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
        return super().get_curve_over_fevals(curve)

    def get_curve_over_time(self, time: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
        return super().get_curve_over_time(time)


class StochasticOptimizationAlgorithm(Curve):

    def get_curve_over_fevals(self, fevals: np.ndarray) -> np.ndarray:
        matching_indices_mask = np.array([np.isin(x_column, fevals, assume_unique=True)
                                          for x_column in self._x_fevals.T]).transpose()    # get the indices of the matching feval range per repeat (column)
        masked = np.where(matching_indices_mask, self._y, np.nan)    # apply the mask, filling NaN for False
        masked_fevals = np.where(matching_indices_mask, self._x_fevals, np.nan)
        print(masked)
        print(masked_fevals)
        curve = np.nanmean(masked, axis=1)    # get the curve by taking the mean
        curve = curve[~np.isnan(curve)]    # remove the NaN, yielding an array which <= fevals.shape
        curve = self.pad_array_fevals(curve, fevals)    # pad with NaN where outside the range, yielding an array which == fevals.shape
        return super().get_curve_over_fevals(curve)

    def get_curve_over_time(self, time: np.ndarray) -> np.ndarray:
        return super().get_curve_over_time(time)
