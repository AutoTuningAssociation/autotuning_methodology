from abc import ABC, abstractmethod
from math import ceil
import numpy as np
from typing import Callable

from autotuning_methodology.curves import Curve, get_indices_in_array
from autotuning_methodology.searchspace_statistics import SearchspaceStatistics


class Baseline(ABC):
    """Class to use as a baseline for Curves in plots"""

    label: str

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_curve(self, range: np.ndarray, x_type: str) -> np.ndarray:
        """Get the curve over the specified range of time or function evaluations, returns NaN beyond limits."""
        if x_type == "fevals":
            return self.get_curve_over_fevals(range)
        elif x_type == "time":
            return self.get_curve_over_time(range)
        raise ValueError(f"x_type must be 'fevals' or 'time', is {x_type}")

    @abstractmethod
    def get_curve_over_fevals(self, fevals_range: np.ndarray) -> np.ndarray:
        """Get the curve over the specified range of function evaluations, returns NaN beyond limits."""
        raise NotImplementedError

    @abstractmethod
    def get_curve_over_time(self, time_range: np.ndarray) -> np.ndarray:
        """Get the curve at the specified times using isotonic regression, returns NaN beyond limits."""
        raise NotImplementedError

    @abstractmethod
    def get_split_times(self, range: np.ndarray, x_type: str, searchspace_stats: SearchspaceStatistics) -> np.ndarray:
        """Get the times at each point in range split into objective_time_keys"""
        if x_type == "fevals":
            return self.get_split_times_at_feval(range, searchspace_stats)
        elif x_type == "time":
            return self.get_split_times_at_time(range, searchspace_stats)
        raise ValueError(f"x_type must be 'fevals' or 'time', is {x_type}")

    @abstractmethod
    def get_split_times_at_feval(
        self, fevals_range: np.ndarray, searchspace_stats: SearchspaceStatistics
    ) -> np.ndarray:
        """Get the times at each function eval in the range split into objective_time_keys"""
        raise NotImplementedError()

    @abstractmethod
    def get_split_times_at_time(self, fevals_range: np.ndarray, searchspace_stats: SearchspaceStatistics) -> np.ndarray:
        """Get the times at each time point in the range split into objective_time_keys"""
        raise NotImplementedError()

    @abstractmethod
    def get_standardised_curves(
        self, range: np.ndarray, strategy_curves: list[np.ndarray], x_type: str
    ) -> tuple[np.ndarray]:
        """Substract the baseline curve from the provided strategy curves, yielding standardised strategy curves"""
        absolute_optimum = self.searchspace_stats.total_performance_absolute_optimum()
        random_curve = self.get_curve(range, x_type)
        standardised_curves: list[np.ndarray] = list()
        for strategy_curve in strategy_curves:
            if strategy_curve is None:
                standardised_curves.append(None)
                continue
            assert strategy_curve.shape == random_curve.shape
            standardised_curve = (strategy_curve - random_curve) / (absolute_optimum - random_curve)
            standardised_curves.append(standardised_curve)
        return tuple(standardised_curves)

    @abstractmethod
    def get_standardised_curve(self, range: np.ndarray, strategy_curve: np.ndarray, x_type: str) -> np.ndarray:
        """Substract the baseline curve from the provided strategy curve, yielding a standardised strategy curve"""
        return self.get_standardised_curves(range, [strategy_curve], x_type)[0]

    @abstractmethod
    def get_split_times_at_feval(
        self, fevals_range: np.ndarray, searchspace_stats: SearchspaceStatistics
    ) -> np.ndarray:
        """Get the times at each function eval in the range split into objective_time_keys"""
        raise NotImplementedError()


class StochasticCurveBasedBaseline(Baseline):
    """Baseline object using a stochastic curve as input"""

    def __init__(self, curve: Curve) -> None:
        self.curve = curve
        self.label = "StochasticCurveBasedBaseline"
        self.minimization = curve.minimization
        super().__init__()

    def get_curve(self, range: np.ndarray, x_type: str) -> np.ndarray:
        return super().get_curve(range, x_type)

    def get_curve_over_fevals(self, fevals_range: np.ndarray) -> np.ndarray:
        return self.curve.get_curve_over_fevals(fevals_range)

    def get_curve_over_time(self, time_range: np.ndarray) -> np.ndarray:
        return self.curve.get_curve_over_time(time_range)

    def get_split_times(self, range: np.ndarray, x_type: str, searchspace_stats: SearchspaceStatistics):
        return super().get_split_times(range, x_type, searchspace_stats)

    def get_split_times_at_feval(
        self, fevals_range: np.ndarray, searchspace_stats: SearchspaceStatistics
    ) -> np.ndarray:
        return super().get_split_times_at_feval(fevals_range, searchspace_stats)

    def get_split_times_at_time(self, fevals_range: np.ndarray, searchspace_stats: SearchspaceStatistics) -> np.ndarray:
        return super().get_split_times_at_time(fevals_range, searchspace_stats)

    def get_standardised_curve(self, range: np.ndarray, strategy_curve: np.ndarray, x_type: str) -> np.ndarray:
        return super().get_standardised_curve(range, strategy_curve, x_type)

    def get_standardised_curves(
        self, range: np.ndarray, strategy_curves: list[np.ndarray], x_type: str
    ) -> tuple[np.ndarray]:
        return super().get_standardised_curves(range, strategy_curves, x_type)


class RandomSearchCalculatedBaseline(Baseline):
    """Baseline object using calculated random search without replacement"""

    def __init__(
        self,
        searchspace_stats: SearchspaceStatistics,
        include_nan=False,
        time_per_feval_operator: str = "mean_per_feval",
        simulate: bool = False,
    ) -> None:
        self.searchspace_stats = searchspace_stats
        self.time_per_feval_operator = time_per_feval_operator
        self.simulate = simulate
        self.label = f"Calculated baseline, {include_nan=}, {time_per_feval_operator=}, {simulate=}"
        self.dist_best_first = searchspace_stats.objective_performances_total_sorted_nan
        self.dist_ascending = searchspace_stats.objective_performances_total_sorted
        self.dist_descending = self.dist_ascending[::-1]
        assert np.all(self.dist_ascending[:-1] <= self.dist_ascending[1:])
        assert np.all(self.dist_descending[:-1] >= self.dist_descending[1:])
        if include_nan:
            self._redwhite_index_dist = self.dist_best_first[::-1]
        else:
            self._redwhite_index_dist = self.dist_descending if searchspace_stats.minimization else self.dist_ascending
        super().__init__()

    def time_to_fevals(self, time_range: np.ndarray) -> np.ndarray:
        """Convert a time range to a number of function evaluations range"""
        # TODO more accurate mapping from fevals to time, using interpolated indices, preferably without median_time_per_feval
        time_per_feval = self.searchspace_stats.get_time_per_feval(self.time_per_feval_operator)
        # assert all(a <= b for a, b in zip(time_range, time_range[1:])), "Time range is not monotonically non-decreasing"
        fevals_range = np.maximum(time_range / time_per_feval, 1)
        # assert all(a <= b for a, b in zip(fevals_range, fevals_range[1:])), "Fevals range is not monotonically non-decreasing"
        fevals_range = np.array(np.round(fevals_range), dtype=int)
        assert all(fevals_range >= 1), f"Fevals range must have minimum of 1, has {fevals_range[fevals_range < 1]}"
        return fevals_range
        # curve = self._get_random_curve(fevals_range)
        # indices = get_indices_in_distribution(curve)
        # indices_interpolated = np.interp(, fevals_range, indices)
        # assert indices.shape == time_range.shape
        # return self.dist_descending[indices]

    def _draw_random(xs, k):
        """Monte Carlo simulation over cache"""
        return np.random.choice(xs, size=k, replace=False)

    def _redwhite_index(self, M: int, N: int) -> float:
        """Get the expected index in the distribution for a budget in number of function evaluations M"""
        assert M >= 0, f"M must be >= 0, is {M}"
        index = round(M * (N + 1) / (M + 1))
        index = min(N - 1, index)
        return index

    def _redwhite_index_value(self, M: int, N: int, dist: np.ndarray) -> float:
        """Get the expected value in the distribution for a budget in number of function evaluations M"""
        return dist[self._redwhite_index(M, N)]

    def _get_random_curve(self, fevals_range: np.ndarray) -> np.ndarray:
        """Returns the drawn values of the random curve at each number of function evaluations"""
        ks = fevals_range
        dist = self._redwhite_index_dist
        N = dist.shape[0]
        draws = np.array([self._redwhite_index_value(k, N, dist) for k in ks])
        return draws

    def _draw_random(self, xs: np.ndarray, k: int):
        return np.random.choice(xs, size=k, replace=False)

    def _stats_max(self, xs: np.ndarray, k: int, trials: int, opt_func: callable) -> np.ndarray:
        # print("Running for stats max size", k, end="\r", flush=True)
        return np.array([opt_func(self._draw_random(xs, k)) for trial in range(trials)])

    def _get_random_curve_means(self, fevals_range: np.ndarray) -> np.ndarray:
        """Returns the mean drawn values of the random curve at each function evaluation"""
        trials = 500
        dist = self.dist_descending
        opt_func = np.min if self.searchspace_stats.minimization else np.max
        results = np.array([self._stats_max(dist, budget, trials, opt_func) for budget in fevals_range])
        val_indices = get_indices_in_array(results, dist)
        # Find the mean index per list of trial runs per function evaluation.
        mean_indices = [round(x) for x in val_indices.mean(axis=1)]
        val_results_index_mean = dist[mean_indices]
        return val_results_index_mean

    def get_curve(self, range: np.ndarray, x_type: str) -> np.ndarray:
        return super().get_curve(range, x_type)

    def get_curve_over_fevals(self, fevals_range: np.ndarray) -> np.ndarray:
        if self.simulate:
            return self._get_random_curve_means(fevals_range)
        return self._get_random_curve(fevals_range)

    def get_curve_over_time(self, time_range: np.ndarray) -> np.ndarray:
        fevals_range = self.time_to_fevals(time_range)
        return self.get_curve_over_fevals(fevals_range)

    def get_standardised_curve(self, range: np.ndarray, strategy_curve: np.ndarray, x_type: str) -> np.ndarray:
        return super().get_standardised_curve(range, strategy_curve, x_type)

    def get_standardised_curves(
        self, range: np.ndarray, strategy_curves: list[np.ndarray], x_type: str
    ) -> tuple[np.ndarray]:
        return super().get_standardised_curves(range, strategy_curves, x_type)

    def get_split_times(self, range: np.ndarray, x_type: str, searchspace_stats: SearchspaceStatistics) -> np.ndarray:
        return super().get_split_times(range, x_type, searchspace_stats)

    def get_split_times_at_feval(
        self, fevals_range: np.ndarray, searchspace_stats: SearchspaceStatistics
    ) -> np.ndarray:
        random_curve = self.get_curve_over_fevals(fevals_range)
        index_at_feval = get_indices_in_array(random_curve, searchspace_stats.objective_performances_total)
        assert not np.all(np.isnan(index_at_feval))
        index_at_feval = index_at_feval.astype(int)

        # for each key, obtain the time at a feval
        objective_time_keys = searchspace_stats.objective_time_keys
        split_time_per_feval = np.full((len(objective_time_keys), index_at_feval.shape[0]), np.NaN)
        for key_index, key in enumerate(objective_time_keys):
            split_time_per_feval[key_index] = searchspace_stats.objective_times_array[key_index, index_at_feval]

        return split_time_per_feval

    def get_split_times_at_time(self, fevals_range: np.ndarray, searchspace_stats: SearchspaceStatistics) -> np.ndarray:
        return super().get_split_times_at_time(fevals_range, searchspace_stats)


class RandomSearchSimulatedBaseline(Baseline):
    """Baseline object using simulated random search"""

    def __init__(
        self,
        searchspace_stats: SearchspaceStatistics,
        repeats: int = 500,
        limit_fevals: int = None,
        index=True,
        flatten=True,
        index_avg="mean",
        performance_avg="mean",
    ) -> None:
        self.searchspace_stats = searchspace_stats
        self.label = f"Simulated baseline, {repeats} repeats ({'index' if index else 'performance'}, {'flattened' if flatten else 'accumulated'}{'' if index_avg == 'mean' else ', index-median'}{'' if performance_avg == 'mean' else ', performance-median'})"
        self.use_index = index
        index_average_func = np.nanmean if index_avg == "mean" else np.nanmedian
        performance_average_func = np.nanmean if performance_avg == "mean" else np.nanmedian
        self._simulate(repeats, limit_fevals, index, flatten, index_average_func, performance_average_func)

    def _simulate(
        self,
        repeats: int,
        limit_fevals: int,
        index: bool,
        flatten: bool,
        index_average_func: Callable,
        performance_average_func: Callable,
    ):
        """Simulate running random search over half of the search space or limit_fevals [repeats] times"""
        opt_func = np.fmin if self.searchspace_stats.minimization else np.fmax
        time_array = self.searchspace_stats.objective_times_total
        performance_array = self.searchspace_stats.objective_performances_total_sorted_nan
        size = min(time_array.shape[0], limit_fevals) if limit_fevals is not None else ceil(time_array.shape[0] / 2)
        indices = np.arange(size)

        # set target arrays
        target_shape = (repeats, size)
        times_at_feval = np.empty(target_shape)
        best_performances_at_feval = np.empty(target_shape)
        best_indices_at_feval = np.empty(target_shape)

        # simulate repeats
        for repeat_index in range(repeats):
            indices_chosen = np.random.choice(indices, size=size, replace=False)
            times_at_feval[repeat_index] = np.nancumsum(time_array[indices_chosen])
            best_indices_at_feval[repeat_index] = np.fmin.accumulate(
                indices_chosen
            )  # NaN sorted to end, so lower index is better
            best_performances_at_feval[repeat_index] = opt_func.accumulate(performance_array[indices_chosen])
        assert times_at_feval.shape == best_performances_at_feval.shape == best_indices_at_feval.shape == target_shape

        # accumulate if necessary
        self.index_at_feval: np.ndarray = index_average_func(best_indices_at_feval, axis=0)
        self.performance_at_feval: np.ndarray = performance_average_func(best_performances_at_feval, axis=0)
        if not flatten:
            self.time_at_feval: np.ndarray = np.nanmean(times_at_feval, axis=0)
            assert self.time_at_feval.shape == self.index_at_feval.shape == self.performance_at_feval.shape == (size,)

        # prepare isotonic regression
        from sklearn.isotonic import IsotonicRegression

        increasing = not self.searchspace_stats.minimization
        self._ir = IsotonicRegression(increasing=increasing, out_of_bounds="clip")

        # fit the data
        x_array = times_at_feval.flatten() if flatten else self.time_at_feval
        y_array = best_indices_at_feval if index else best_performances_at_feval
        self.y_array = y_array
        if flatten:
            y_array = y_array.flatten()
        else:
            y_array = self.index_at_feval if index else self.performance_at_feval
            self.y_array = y_array
        self._ir.fit(x_array, y_array)

    def get_curve(self, range: np.ndarray, x_type: str) -> np.ndarray:
        return super().get_curve(range, x_type)

    def get_curve_over_fevals(self, fevals_range: np.ndarray) -> np.ndarray:
        if self.use_index:
            return self.searchspace_stats.objective_performances_total_sorted_nan[
                np.array(np.round(self.index_at_feval[fevals_range]), dtype=int)
            ]
        else:
            assert self.y_array.ndim == 1
            return self.y_array[fevals_range]

    def get_curve_over_time(self, time_range: np.ndarray) -> np.ndarray:
        predicted_y_values = self._ir.predict(time_range)
        if not self.use_index:
            return predicted_y_values
        return self.searchspace_stats.objective_performances_total_sorted_nan[
            np.array(np.round(predicted_y_values), dtype=int)
        ]

    def get_standardised_curve(self, range: np.ndarray, strategy_curve: np.ndarray, x_type: str) -> np.ndarray:
        return super().get_standardised_curve(range, strategy_curve, x_type)

    def get_standardised_curves(
        self, range: np.ndarray, strategy_curves: list[np.ndarray], x_type: str
    ) -> tuple[np.ndarray]:
        return super().get_standardised_curves(range, strategy_curves, x_type)

    def get_split_times(self, range: np.ndarray, x_type: str, searchspace_stats: SearchspaceStatistics) -> np.ndarray:
        return super().get_split_times(range, x_type, searchspace_stats)

    def get_split_times_at_time(self, fevals_range: np.ndarray, searchspace_stats: SearchspaceStatistics) -> np.ndarray:
        return super().get_split_times_at_time(fevals_range, searchspace_stats)

    def get_split_times_at_feval(
        self, fevals_range: np.ndarray, searchspace_stats: SearchspaceStatistics
    ) -> np.ndarray:
        return super().get_split_times_at_feval(fevals_range, searchspace_stats)
