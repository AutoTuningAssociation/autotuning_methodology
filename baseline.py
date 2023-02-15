from abc import ABC, abstractmethod
import numpy as np
from curves import Curve
from searchspace_statistics import SearchspaceStatistics


class Baseline(ABC):
    """ Class to use as a baseline for Curves in plots """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_curve(self, range: np.ndarray, x_type: str) -> np.ndarray:
        """ Get the curve over the specified range of time or function evaluations, returns NaN beyond limits. """
        if x_type == 'fevals':
            return self.get_curve_over_fevals(range)
        elif x_type == 'time':
            return self.get_curve_over_time(range)
        raise ValueError(f"x_type must be 'fevals' or 'time', is {x_type}")

    @abstractmethod
    def get_curve_over_fevals(self, fevals_range: np.ndarray) -> np.ndarray:
        """ Get the curve over the specified range of function evaluations, returns NaN beyond limits. """
        raise NotImplementedError

    @abstractmethod
    def get_curve_over_time(self, time_range: np.ndarray) -> np.ndarray:
        """ Get the curve at the specified times using isotonic regression, returns NaN beyond limits. """
        raise NotImplementedError

    @abstractmethod
    def get_standardised_curve(self, range: np.ndarray, strategy_curve: np.ndarray, x_type: str) -> np.ndarray:
        """ Substract the baseline curve from the provided strategy curve, yielding a standardised strategy curve """
        absolute_optimum = self.searchspace_stats.total_performance_absolute_optimum()
        random_curve = self.get_curve(range, x_type)
        assert strategy_curve.shape == random_curve.shape
        if not self.searchspace_stats.minimization:
            raise NotImplementedError()    # make sure this works when maximizing
        standardised_curve = (strategy_curve - random_curve) / (absolute_optimum - random_curve)
        return standardised_curve


class StochasticCurveBasedBaseline(Baseline):
    """ Baseline object using a stochastic curve as input """

    def __init__(self, curve: Curve) -> None:
        self.curve = curve
        self.minimization = curve.minimization
        super().__init__()

    def get_curve(self, range: np.ndarray, x_type: str) -> np.ndarray:
        return super().get_curve(range, x_type)

    def get_curve_over_fevals(self, fevals_range: np.ndarray) -> np.ndarray:
        return self.curve.get_curve_over_fevals(fevals_range)

    def get_curve_over_time(self, time_range: np.ndarray) -> np.ndarray:
        return self.curve.get_curve_over_time(time_range)

    def get_standardised_curve(self, range: np.ndarray, strategy_curve: np.ndarray, x_type: str) -> np.ndarray:
        return super().get_standardised_curve(range, strategy_curve, x_type)


class RandomSearchCalculatedBaseline(Baseline):
    """ Baseline object using calculated random search without replacement """

    def __init__(self, searchspace_stats: SearchspaceStatistics) -> None:
        self.searchspace_stats = searchspace_stats
        self.dist_ascending = searchspace_stats.objective_performances_total_sorted
        self.dist_descending = self.dist_ascending[::-1]
        assert np.all(self.dist_ascending[:-1] <= self.dist_ascending[1:])
        assert np.all(self.dist_descending[:-1] >= self.dist_descending[1:])
        self._redwhite_index_dist = self.dist_descending if searchspace_stats.minimization else self.dist_ascending
        super().__init__()

    def time_to_fevals(self, time_range: np.ndarray) -> np.ndarray:
        """ Convert a time range to a number of function evaluations range """
        # TODO more accurate mapping from fevals to time, using interpolated indices, preferably without median_time_per_feval
        median_feval_time = self.searchspace_stats.total_time_median()
        # assert all(a <= b for a, b in zip(time_range, time_range[1:])), "Time range is not monotonically non-decreasing"
        fevals_range = np.maximum(time_range / median_feval_time, 0)
        # assert all(a <= b for a, b in zip(fevals_range, fevals_range[1:])), "Fevals range is not monotonically non-decreasing"
        fevals_range = np.array(np.round(fevals_range), dtype=int)
        return fevals_range
        # curve = self._get_random_curve(fevals_range)
        # indices = self._get_indices(curve)
        # indices_interpolated = np.interp(, fevals_range, indices)
        # assert indices.shape == time_range.shape
        # return self.dist_descending[indices]

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
        """ Get the expected value in the distribution for a budget in number of function evaluations """
        # N = self.searchspace_stats.size
        dist = self._redwhite_index_dist
        N = dist.shape[0]
        index = round(M * (N + 1) / (M + 1))
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
        opt_func = np.min if self.searchspace_stats.minimization else np.max
        results = np.array([self._stats_max(dist, budget, trials, opt_func) for budget in fevals_range])
        val_indices = self._get_indices(results)
        # Find the mean index per list of trial runs per function evaluation.
        mean_indices = [round(x) for x in val_indices.mean(axis=1)]
        val_results_index_mean = dist[mean_indices]
        return val_results_index_mean

    def get_curve(self, range: np.ndarray, x_type: str) -> np.ndarray:
        return super().get_curve(range, x_type)

    def get_curve_over_fevals(self, fevals_range: np.ndarray) -> np.ndarray:
        return self._get_random_curve(fevals_range)

    def get_curve_over_time(self, time_range: np.ndarray) -> np.ndarray:
        return self._get_random_curve(self.time_to_fevals(time_range))

    def get_standardised_curve(self, range: np.ndarray, strategy_curve: np.ndarray, x_type: str) -> np.ndarray:
        return super().get_standardised_curve(range, strategy_curve, x_type)


class RandomSearchSimulatedBaseline(Baseline):
    """ Baseline object using simulated random search"""

    def __init__(self, searchspace_stats: SearchspaceStatistics, repeats: int = 500) -> None:
        self.searchspace_stats = searchspace_stats
        self._simulate(repeats=repeats)

    def _simulate(self, repeats: int, limit_fevals: int = None):
        """ Simulate running random search over the entire search space or limit_fevals [repeats] times """
        opt_func = np.fmin if self.searchspace_stats.minimization else np.fmax
        time_array = self.searchspace_stats.objective_times_total
        performance_array = self.searchspace_stats.objective_performances_total
        size = min(time_array.shape[0], limit_fevals) if limit_fevals is not None else time_array.shape[0]
        times_at_feval = np.empty((repeats, size))
        performances_at_feval = np.empty((repeats, size))
        indices_chosen = np.arange(size)
        for repeat_index in range(repeats):
            if limit_fevals is None:
                np.random.shuffle(indices_chosen)
            else:
                indices_chosen = np.random.choice(indices_chosen, size=size, replace=False)

            times_at_feval[repeat_index] = np.nancumsum(time_array[indices_chosen])
            performances_at_feval[repeat_index] = opt_func.accumulate(performance_array[indices_chosen])
        assert times_at_feval.shape == (repeats, size)
        self.time_at_feval: np.ndarray = np.nanmean(times_at_feval, axis=0)
        self.performance_at_feval: np.ndarray = np.nanmean(performances_at_feval, axis=0)

        # prepare isotonic regression
        from sklearn.isotonic import IsotonicRegression
        increasing = not self.searchspace_stats.minimization
        self._ir = IsotonicRegression(increasing=increasing, out_of_bounds='clip')
        self._ir.fit(self.time_at_feval, self.performance_at_feval)

    def get_curve(self, range: np.ndarray, x_type: str) -> np.ndarray:
        return super().get_curve(range, x_type)

    def get_curve_over_fevals(self, fevals_range: np.ndarray) -> np.ndarray:
        return self.performance_at_feval[fevals_range]

    def get_curve_over_time(self, time_range: np.ndarray) -> np.ndarray:
        return self._ir.predict(time_range)

    def get_standardised_curve(self, range: np.ndarray, strategy_curve: np.ndarray, x_type: str) -> np.ndarray:
        return super().get_standardised_curve(range, strategy_curve, x_type)
