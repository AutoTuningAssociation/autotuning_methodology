from abc import ABC, abstractmethod
import numpy as np


class Baseline(ABC):
    """ Class to use as a baseline for Curves in plots """

    def __init__(self, minimization: bool) -> None:
        self.minimization = minimization
        super().__init__()

    @abstractmethod
    def get_curve_over_fevals(self, fevals_range: np.ndarray) -> np.ndarray:
        """ Get the curve over the specified range of function evaluations, returns NaN beyond limits. """
        return fevals_range

    @abstractmethod
    def get_curve_over_time(self, time_range: np.ndarray) -> np.ndarray:
        """ Get the curve at the specified times using isotonic regression, returns NaN beyond limits. """
        return time_range

    @abstractmethod
    def get_standardised_curve_over_fevals(self, strategy_curve: np.ndarray) -> np.ndarray:
        """ Substract the baseline curve from the provided strategy curve, yielding a standardised strategy curve """
        return strategy_curve

    @abstractmethod
    def get_standardised_curve_over_time(self, strategy_curve: np.ndarray) -> np.ndarray:
        """ Substract the baseline curve from the provided strategy curve, yielding a standardised strategy curve """
        return strategy_curve


class RandomSearchBaseline(Baseline):
    """ Baseline class using calculated random search without replacement """

    def __init__(self, minimization: bool) -> None:
        self.minimization = minimization
        super().__init__()

    def _draw_random(xs, k):
        """ Monte Carlo simulation over cache """
        return np.random.choice(xs, size=k, replace=False)

    def _get_indices(self, dist: np.ndarray, draws: np.ndarray) -> np.ndarray:
        """ For each draw, get the index (position) in the distribution """
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

    def get_curve_over_fevals(self, fevals_range: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        return super().get_curve_over_fevals(curve)

    def get_curve_over_time(self, time_range: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
        return super().get_curve_over_time(time)

    def get_standardised_curve_over_fevals(self, strategy_curve: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
        return super().get_standardised_curve_over_fevals(strategy_curve)

    def get_standardised_curve_over_time(self, strategy_curve: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
        return super().get_standardised_curve_over_fevals(strategy_curve)
