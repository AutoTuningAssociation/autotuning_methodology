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
        self.__x_fevals = results.fevals_results    # the time per objective value in number of function evaluations since start (1d if deterministic, 2d if stochastic)
        self.__x_time = results.time_results    # the time per objective value in seconds since start the raw x-axis (1d if deterministic, 2d if stochastic)
        self.__y = results.objective_value_best_results    # the objective values (1d if deterministic, 2d if stochastic)

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
        assert isinstance(self.__x_fevals, np.ndarray)
        assert isinstance(self.__x_time, np.ndarray)
        assert isinstance(self.__y, np.ndarray)

        # assert values
        if self.stochastic is False:
            assert self.__x_fevals.ndim == 1
            assert self.__x_time.ndim == 1
            assert self.__y.ndim == 1
        else:
            assert self.__x_fevals.ndim == 2
            assert self.__x_time.ndim == 2
            assert self.__y.ndim == 2

    # @abstractmethod
    def get_curve_over_fevals(self, fevals: np.ndarray) -> np.ndarray:
        """ Get the curve over function evaluations, returns NaN beyond limits. """
        pass

    # @abstractmethod
    def get_curve_over_time(self, time: np.ndarray) -> np.ndarray:
        """ Get the curve over time using isotonic regression, returns NaN beyond limits. """
        pass


class RandomBaseline(Curve):
    pass


class DeterministicOptimizationAlgorithm(Curve):
    pass


class StochasticOptimizationAlgorithm(Curve):
    pass
