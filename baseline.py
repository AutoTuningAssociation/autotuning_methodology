from abc import ABC, abstractmethod
import numpy as np
from experiments import median_time_per_feval


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

    def __init__(self, minimization: bool, sorted_times: np.ndarray) -> None:
        self.minimization = minimization
        self.N = len(sorted_times)
        self.dist_ascending = sorted_times
        self.dist_descending = sorted_times[::-1]
        assert np.all(self.dist_ascending[:-1] <= self.dist_ascending[1:])
        assert np.all(self.dist_descending[:-1] >= self.dist_descending[1:])
        self._redwhite_index_dist = self.dist_descending if minimization else self.dist_descending
        super().__init__(minimization)

    def time_to_fevals(self, time_range: np.ndarray) -> np.ndarray:
        """ Convert a time range to a number of function evaluations range """

    def _draw_random(xs, k):
        """ Monte Carlo simulation over cache """
        return np.random.choice(xs, size=k, replace=False)

    def _get_indices(self, draws: np.ndarray) -> np.ndarray:
        """ For each draw, get the index (position) in the distribution """
        dist = self.dist_descending
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

    def _redwhite_index(self, M: int) -> float:
        """ Get the expected value in the distribution for a single budget """
        N = self.N
        index = M * (N + 1) / (M + 1)
        index = round(index)
        dist = self._redwhite_index_dist
        index = min(dist.shape[0] - 1, index)
        return dist[index]

    def _get_random_curve(self, fevals_range: np.ndarray) -> np.ndarray:
        """ Returns the drawn values of the random curve at each number of function evaluation """
        ks = fevals_range - 1    # because ranges of number of function evaluations start at 1, we need to subtract 1 to use the index version
        draws = np.array([self._redwhite_index(k) for k in ks])
        return draws

    def _draw_random(self, xs: np.ndarray, k: int):
        return np.random.choice(xs, size=k, replace=False)

    def _stats_max(self, xs: np.ndarray, k: int, trials: int, opt_func: callable) -> np.ndarray:
        # print("Running for stats max size", k, end="\r", flush=True)
        return np.array([opt_func(self._draw_random(xs, k)) for trial in range(trials)])

    def _get_random_curve_means(self, fevals_range: np.ndarray) -> np.ndarray:
        """ Returns the mean drawn values of the random curve at each function evaluation """
        trials = 500
        dist = self.dist_descending
        opt_func = np.min if self.minimization else np.max
        results = np.array([self._stats_max(dist, budget, trials, opt_func) for budget in fevals_range])
        val_indices = self._get_indices(results)
        # Find the mean index per list of trial runs per function evaluation.
        mean_indices = [round(x) for x in val_indices.mean(axis=1)]
        val_results_index_mean = dist[mean_indices]
        return val_results_index_mean

    def get_curve_over_fevals(self, fevals_range: np.ndarray) -> np.ndarray:
        curve = self._get_random_curve(fevals_range)
        return super().get_curve_over_fevals(curve)

    def get_curve_over_time(self, time_range: np.ndarray, stats_info: dict) -> np.ndarray:
        # TODO map from time range to fevals
        median_feval_time = median_time_per_feval(stats_info)
        fevals_range = np.maximum(time_range / median_feval_time, 0)
        curve = self._get_random_curve(time_range)
        indices = self._get_indices(curve)
        # indices_interpolated = np.interp(, fevals_range, indices)
        assert indices.shape == time_range.shape
        return self.dist_descending[indices]
        # TODO map from fevals to time, use indices to get this curve

    def get_standardised_curve_over_fevals(self, fevals_range: np.ndarray, strategy_curve: np.ndarray, absolute_optimum: float) -> np.ndarray:
        random_curve = self.get_curve_over_fevals(fevals_range)
        assert strategy_curve.shape == random_curve.shape
        if not self.minimization:
            raise NotImplementedError()    # make sure this works when maximizing
        standardised_curve = (strategy_curve - random_curve) / (absolute_optimum - random_curve)
        return standardised_curve

    def get_standardised_curve_over_time(self, time_range: np.ndarray, strategy_curve: np.ndarray, absolute_optimum: float) -> np.ndarray:
        raise NotImplementedError()
        return super().get_standardised_curve_over_fevals(strategy_curve)
