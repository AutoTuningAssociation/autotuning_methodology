from abc import ABC, abstractmethod
import numpy as np
from caching import ResultsDescription


class Curve(ABC):

    def __init__(self, name: str, displayname: str, gpu_name: str, kernel_name: str, x_fevals: np.ndarray, x_time: np.ndarray, y: np.ndarray,
                 deterministic: bool) -> None:
        # inputs
        self.name = name
        self.displayname = displayname
        self.gpu_name = gpu_name
        self.kernel_name = kernel_name
        self.__x_fevals = x_fevals    # the time per objective value in number of function evaluations since start (1d if deterministic, 2d if stochastic)
        self.__x_time = x_time    # the time per objective value in seconds since start the raw x-axis (1d if deterministic, 2d if stochastic)
        self.__y = y    # the objective values (1d if deterministic, 2d if stochastic)
        self.deterministic = deterministic

        # checks
        if self.deterministic is True:
            assert self.__x_fevals.ndim == 1
            assert self.__x_time.ndim == 1
            assert self.__y.ndim == 1
        elif self.deterministic is False:
            assert self.__x_fevals.ndim == 2
            assert self.__x_time.ndim == 2
            assert self.__y.ndim == 2
        else:
            raise ValueError(f"'deterministic' argument must be boolean, is {deterministic}")
        super().__init__()

    def __init__(self, results_description: ResultsDescription) -> None:
        """ Initialize using a ResultsDescription """
        super().__init__()
        # TODO
        raise NotImplementedError()

    @abstractmethod
    def get_curve_over_fevals(self, fevals: np.ndarray) -> np.ndarray:
        """ Get the curve over function evaluations, returns NaN beyond limits. """
        pass

    @abstractmethod
    def get_curve_over_time(self, time: np.ndarray) -> np.ndarray:
        """ Get the curve over time using isotonic regression, returns NaN beyond limits. """
        pass


class RandomBaseline(Curve):
    pass


class DeterministicOptimizationAlgorithm(Curve):
    pass


class StochasticOptimizationAlgorithm(Curve):
    pass
