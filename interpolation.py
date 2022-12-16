import numpy as np
from sklearn.isotonic import IsotonicRegression
from isotonic.isotonic import LpIsotonicRegression


def get_isotonic_curve(x: np.ndarray, y: np.ndarray, x_new: np.ndarray, package='isotonic', increasing=False, npoints=1000, power=2, ymin=None,
                       ymax=None) -> np.ndarray:
    """ Get the isotonic regression curve fitted to x_new using package 'sklearn' or 'isotonic' """
    # check if the assumptions that the input arrays are numpy arrays holds
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(x_new, np.ndarray)
    if package == 'sklearn':
        if npoints != 1000:
            warnings.warn("npoints argument is impotent for sklearn package")
        if power != 2:
            warnings.warn("power argument is impotent for sklearn package")
        ir = IsotonicRegression(increasing=increasing, y_min=ymin, y_max=ymax, out_of_bounds='clip')
        ir.fit(x, y)
        return ir.predict(x_new)
    elif package == 'isotonic':
        ir = LpIsotonicRegression(npoints, increasing=increasing, power=power).fit(x, y)
        y_isotonic_regression = ir.predict_proba(x_new)
        # TODO check if you are not indadvertedly clipping too much here
        if ymin is not None or ymax is not None:
            y_isotonic_regression = np.clip(y_isotonic_regression, ymin, ymax)
        return y_isotonic_regression
    raise ValueError(f"Package name {package} is not a valid package name")


def create_interpolated_results(repeated_results: list, expected_results: dict, optimization_objective: str, cutoff_point_fevals: int,
                                objective_value_at_cutoff_point: float, time_resolution: int, time_interpolated_axis: np.ndarray, y_min=None, y_median=None,
                                segment_factor=0.05) -> Dict[Any, Any]:
    """ Creates a monotonically non-increasing curve from the combined objective datapoints across repeats for a strategy, interpolated for [time_resolution] points, using [time_resolution * segment_factor] piecewise linear segments """
    results = deepcopy(expected_results)

    # find the maximum number of function evaluations
    max_num_evals = max(len(res) for res in repeated_results)

    # find the minimum objective value and time spent for each evaluation per repeat
    dtype = [('total_time', 'float64'), ('objective_value', 'float64'), ('objective_value_std', 'float64')]
    total_times = list()
    best_found_objective_values = list()
    num_function_evaluations = list()
    num_function_evaluations_repeated_results = np.empty((max_num_evals, len(repeated_results)))
    num_function_evaluations_repeated_results[:] = np.nan    # set all to nan so they are not counted as zeros inadvertedly
    for res_index, res in enumerate(repeated_results):
        # extract the objective and time spent per configuration
        repeated_results[res_index] = np.array(
            list(tuple([sum(r['times']) /
                        1000, r[optimization_objective], np.std(r[optimization_objective + 's'])]) for r in res if r['time'] != 1e20), dtype=dtype)
        # take the minimum of the objective and the sum of the time
        obj_minimum = 1e20
        total_time = 0
        for r_index, r in enumerate(repeated_results[res_index]):
            total_time += r[0]
            obj_minimum = min(r[1], obj_minimum)
            assert obj_minimum >= y_min
            obj_std = r[2]
            repeated_results[res_index][r_index] = np.array(tuple([total_time, obj_minimum, obj_std]), dtype=dtype)
            # also add results at the same number of function evaluations together
            try:
                num_function_evaluations_repeated_results[r_index][res_index] = obj_minimum
            except IndexError as e:
                raise e
                # in case of an index error, repeated_results has more evals than max_num_evals
                break
        total_times.append(total_time)
        best_found_objective_values.append(obj_minimum)
        num_function_evaluations.append(len(repeated_results[res_index]))

    # write to the results
    if 'total_times' in expected_results:
        results['total_times'] = total_times
    if 'best_found_objective_values' in expected_results:
        results['best_found_objective_values'] = best_found_objective_values
    if 'num_function_evaluations' in expected_results:
        results['num_function_evaluations'] = num_function_evaluations
    if 'num_function_evaluations_repeated_results' in expected_results:
        results['num_function_evaluations_repeated_results'] = num_function_evaluations_repeated_results

    # combine the results across repeats to be in time-order
    combined_results = np.concatenate(repeated_results)
    combined_results = np.sort(combined_results, order='total_time')    # sort objective is the total times increasing
    x: np.ndarray = combined_results['total_time']
    y: np.ndarray = combined_results['objective_value']
    y_std: np.ndarray = combined_results['objective_value_std']
    # assert that the total time is monotonically non-decreasing
    assert all(a <= b for a, b in zip(x, x[1:]))

    x_new = time_interpolated_axis
    npoints = int(len(x_new) * segment_factor)

    # # calculate polynomial fit
    # z = np.polyfit(x, y, 10)
    # f = np.poly1d(z)
    # y_polynomial = f(x_new)
    # # make it monotonically non-increasing (this is a very slow function due to O(n^2) complexity)
    # y_polynomial = list(min(y_polynomial[:i]) if i > 0 else y_polynomial[i] for i in range(len(y_polynomial)))

    # calculate Isotonic Regression
    # the median is used as the maximum because as number of samples approaches infinity, the probability that the found minimum is <= median approaches 1
    y_isotonic_regression = get_isotonic_curve(x, y, x_new, ymin=y_min, ymax=y_median, npoints=npoints)
    # # assert that monotonicity is satisfied in the isotonic regression
    # assert all(a>=b for a, b in zip(y_isotonic_regression, y_isotonic_regression[1:]))

    # # do linear interpolation for the errors
    # # get the distance between the isotonic curve and the actual datapoint for each datapoint
    # error: np.ndarray = y - get_isotonic_curve(x, y, x, ymin=y_min, ymax=y_median, npoints=npoints)
    # # f_error_interpolated = interp1d(x, np.abs(error), bounds_error=False, fill_value=tuple([error[0], error[-1]]))
    # # error_interpolated = f_error_interpolated(x_new)
    # error_lower_indices = np.where(error <= 0)
    # error_upper_indices = np.where(error >= 0)
    # x_lower = x[error_lower_indices]
    # error_lower = error[error_lower_indices]
    # x_upper = x[error_upper_indices]
    # error_upper = error[error_upper_indices]
    # # # interpolate to the baseline time axis, when extrapolating use the first or last value
    # # f_error_lower_interpolated = interp1d(x_lower, error_lower, bounds_error=False, fill_value=tuple([error_lower[0], error_lower[-1]]))
    # # f_error_upper_interpolated = interp1d(x_upper, error_upper, bounds_error=False, fill_value=tuple([error_upper[0], error_upper[-1]]))
    # # error_lower_interpolated: np.ndarray = smoothing_filter(f_error_lower_interpolated(x_new), 100)
    # # error_upper_interpolated: np.ndarray = smoothing_filter(f_error_upper_interpolated(x_new), 100)

    # # alternative: do isotonic regression for the upper and lower values seperately
    # error_lower_interpolated = get_isotonic_curve(x_lower, error_lower, x_new, npoints=npoints, ymax=0)
    # error_upper_interpolated = get_isotonic_curve(x_upper, error_upper, x_new, npoints=npoints, ymin=0)

    # do linear interpolation for the other attributes
    f_li_y_std = interp1d(x, y_std, fill_value='extrapolate')
    y_std_li: np.ndarray = f_li_y_std(x_new)

    # write to the results
    if 'interpolated_time' in expected_results:
        results['interpolated_time'] = time_interpolated_axis    # TODO maybe not write this for every strategy, but once
    if 'interpolated_objective' in expected_results:
        results['interpolated_objective'] = y_isotonic_regression
    if 'interpolated_objective_std' in expected_results:
        results['interpolated_objective_std'] = y_std_li
    if 'interpolated_objective_error_lower' in expected_results:
        results['interpolated_objective_error_lower'] = error_lower_interpolated
    if 'interpolated_objective_error_upper' in expected_results:
        results['interpolated_objective_error_upper'] = error_upper_interpolated

    return results
