""" Interface to run an experiment on Kernel Tuner """
from cProfile import label
from copy import deepcopy
import numpy as np
import progressbar
from typing import Any, Tuple, Dict
import time as python_time
import warnings
import yappi
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
from isotonic.isotonic import LpIsotonicRegression

record_data = ['mean_actual_num_evals']


def remove_duplicates(res: list, remove_duplicate_results: bool):
    """ Removes duplicate configurations from the results """
    if not remove_duplicate_results:
        return res
    unique_res = list()
    for result in res:
        if result not in unique_res:
            unique_res.append(result)
    return unique_res

def get_isotonic_curve(x: np.ndarray, y: np.ndarray, x_new: np.ndarray, package='isotonic', increasing=False, npoints=1000, power=2, ymin=None, ymax=None) -> np.ndarray:
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
        if ymin is not None or ymax is not None:
            y_isotonic_regression = np.clip(y_isotonic_regression, ymin, ymax)
        return y_isotonic_regression
    raise ValueError(f"Package name {package} is not a valid package name")



def tune(kernel, kernel_name: str, device_name: str, strategy: dict, tune_options: dict, profiling: bool) -> Tuple[list, int]:
    """ Execute a strategy, return the result, runtime and optional profiling statistics """

    def tune_with_kerneltuner():
        """ interface with kernel tuner to tune the kernel and return the results """
        if profiling:
            yappi.set_clock_type("cpu")
            yappi.start()
        res, env = kernel.tune(device_name=device_name, strategy=strategy['strategy'], strategy_options=strategy['options'], **tune_options)
        if profiling:
            yappi.stop()
        return res, env

    total_start_time = python_time.perf_counter()
    warnings.simplefilter("ignore", UserWarning)
    try:
        res, _ = tune_with_kerneltuner()
    except ValueError:
        print("Something went wrong, trying once more.")
        res, _ = tune_with_kerneltuner()
    warnings.simplefilter("default", UserWarning)
    total_end_time = python_time.perf_counter()
    total_time_ms = round((total_end_time - total_start_time) * 1000)
    # TODO when profiling, should the total_time_ms not be the time from profiling_stats? Otherwise we are timing the profiling code as well
    return res, total_time_ms


def collect_results(kernel, kernel_name: str, device_name: str, strategy: dict, expected_results: dict, profiling: bool, objective_value_at_cutoff_point: float,
                    optimization_objective='time', remove_duplicate_results=True, time_resolution = 1e4, time_interpolated_axis = None, y_min = None, y_median = None, segment_factor = 0.05) -> dict:
    """ Executes strategies to obtain (or retrieve from cache) the statistical data """
    print(f"Running {strategy['display_name']}")
    nums_of_evaluations = strategy['nums_of_evaluations']
    max_num_evals = min(strategy['nums_of_evaluations'])
    # TODO put the tune options in the .json in strategy_defaults?
    tune_options = {
        'verbose': False,
        'quiet': True,
        'simulation_mode': True
    }

    def report_multiple_attempts(rep: int, len_res: int, len_unique_res: int, strategy_repeats: int):
        """ If multiple attempts are necessary, report the reason """
        if len_res < 1:
            print(f"({rep+1}/{strategy_repeats}) No results found, trying once more...")
        elif len_unique_res < max_num_evals:
            print(f"Too few unique results found ({len_unique_res} in {len_res} evaluations), trying once more...")
        else:
            print(f"({rep+1}/{strategy_repeats}) Only invalid results found, trying once more...")

    # repeat the strategy as specified
    repeated_results = list()
    total_time_results = np.array([])
    for rep in progressbar.progressbar(range(strategy['repeats']), redirect_stdout=True):
        attempt = 0
        only_invalid = True
        while only_invalid or (remove_duplicate_results and len_unique_res < max_num_evals):
            if attempt > 0:
                report_multiple_attempts(rep, len_res, len_unique_res, strategy['repeats'])
            res, total_time_ms = tune(kernel, kernel_name, device_name, strategy, tune_options, profiling)
            len_res: int = len(res)
            # check if there are only invalid configs in the first 10 fevals, if so, try again
            only_invalid = len_res < 1 or min(res[:10], key=lambda x: x['time'])['time'] == 1e20
            unique_res = remove_duplicates(res, remove_duplicate_results)
            len_unique_res: int = len(unique_res)
            attempt += 1
        # register the results
        repeated_results.append(unique_res)
        total_time_results = np.append(total_time_results, total_time_ms)
        if len(strategy['nums_of_evaluations']) <= 0:
            nums_of_evaluations = np.append(nums_of_evaluations, len_unique_res)

    # gather profiling data and clear the profiler before the next round
    if profiling:
        stats = yappi.get_func_stats()
        # stats.print_all()
        path = "../experiments/profilings/random/profile-v2.prof"
        stats.save(path, type="pstat")    # pylint: disable=no-member
        yappi.clear_stats()

    # create the interpolated results from the repeated results
    results = create_interpolated_results(repeated_results, expected_results, optimization_objective, objective_value_at_cutoff_point, time_resolution, time_interpolated_axis, y_min, y_median, segment_factor)

    # check that all expected results are present
    for key in results.keys():
        if key == 'cutoff_quantile' or key == 'curve_segment_factor':
            continue
        if results[key] is None:
            raise ValueError(f"Expected result {key} was not filled in the results")
    return results

def create_interpolated_results(repeated_results: list, expected_results: dict, optimization_objective: str, objective_value_at_cutoff_point: float, time_resolution: int, time_interpolated_axis: np.ndarray, y_min = None, y_median = None, segment_factor=0.05) -> Dict[Any, Any]:
    """ Creates a monotonically non-increasing curve from the combined objective datapoints across repeats for a strategy, interpolated for [time_resolution] points, using [time_resolution * segment_factor] piecewise linear segments """
    results = deepcopy(expected_results)

    # find the minimum objective value and time spent for each evaluation per repeat
    dtype = [('total_time', 'float64'), ('objective_value', 'float64'), ('objective_value_std', 'float64')]
    total_times = list()
    best_found_objective_values = list()
    num_function_evaluations = list()
    for res_index, res in enumerate(repeated_results):
        # extract the objective and time spent per configuration
        repeated_results[res_index] = np.array(list(tuple([sum(r['times']) / 1000, r[optimization_objective], np.std(r[optimization_objective + 's'])]) for r in res if r['time'] != 1e20), dtype=dtype)
        # take the minimum of the objective and the sum of the time
        obj_minimum = 1e20
        total_time = 0
        for r_index, r in enumerate(repeated_results[res_index]):
            total_time += r[0]
            obj_minimum = min(r[1], obj_minimum)
            obj_std = r[2]
            repeated_results[res_index][r_index] = np.array(tuple([total_time, obj_minimum, obj_std]), dtype=dtype)
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

    # combine the results across repeats to be in time-order
    combined_results = np.concatenate(repeated_results)
    combined_results = np.sort(combined_results, order='total_time') # sort objective is the total times increasing
    x: np.ndarray = combined_results['total_time']
    y: np.ndarray = combined_results['objective_value']
    y_std: np.ndarray = combined_results['objective_value_std']
    # assert that the total time is monotonically non-decreasing
    assert all(a<=b for a, b in zip(x, x[1:]))

    # create the new x-axis array to interpolate
    if time_interpolated_axis is None:
        # first create a temporary interpolation using the absolute results, using sklearn because this has arbitrary number of segments
        _y_isotonic_regression = get_isotonic_curve(x, y, x, ymin=y_min, ymax=y_median, package='sklearn')

        # find the cutoff point using the temporary interpolation
        try:
            cutoff_index = np.argwhere(_y_isotonic_regression <= objective_value_at_cutoff_point)[0]
            assert cutoff_index == int(cutoff_index)    # ensure that it is an integer
            cutoff_index = int(cutoff_index)
            # print(f"Percentage of baseline search space used to get to cutoff quantile: {(cutoff_index / _y_isotonic_regression.size)*100}%")
        except IndexError:
            raise ValueError(f"The baseline has not reliably found the cutoff quantile, either decrease the cutoff or increase the allowed time.")

        # create the baseline time axis
        cutoff_time = x[cutoff_index]
        time_interpolated_axis = np.linspace(x[0], cutoff_time, time_resolution)
    else:
        assert len(time_interpolated_axis) == time_resolution
    x_new = time_interpolated_axis
    npoints = int(len(x_new)*segment_factor)

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

    # get the errors by snapping the original x-values to the closest x_new-values, assumes both x and x_new are sorted in increasing order
    curr_index = 0
    x_snapped_temp = list()
    for x_val in x:
        try:
            while abs(x_val - x_new[curr_index+1]) < abs(x_val - x_new[curr_index]):
                curr_index += 1
            x_snapped_temp.append(curr_index)
        except IndexError:
            x_snapped_temp.append(x_new.size - 1)
    snapped_indices = np.array(x_snapped_temp)  # an array of shape x with indices pointing to x_new
    error_snapped = y - y_isotonic_regression[snapped_indices]

    # seperate the lower and upper error
    error_lower_indices = np.where(error_snapped <= 0)
    error_upper_indices = np.where(error_snapped >= 0)
    x_lower = x[error_lower_indices]
    error_lower = error_snapped[error_lower_indices]
    x_upper = x[error_upper_indices]
    error_upper = error_snapped[error_upper_indices]

    # # get the snapped error
    # x_new_error_lower = snapped_indices[error_lower_indices]
    # x_new_error_upper = snapped_indices[error_upper_indices]
    # error_lower_x: np.ndarray = smoothing_filter(x_new[x_new_error_lower], 100)
    # error_upper_x: np.ndarray = smoothing_filter(x_new[x_new_error_upper], 100)

    # interpolate lower and upper error to x_new
    f_error_lower_interpolated = interp1d(x_lower, error_lower, bounds_error=False, fill_value=tuple([error_lower[0], error_lower[-1]]))
    f_error_upper_interpolated = interp1d(x_upper, error_upper, bounds_error=False, fill_value=tuple([error_upper[0], error_upper[-1]]))
    error_lower_interpolated: np.ndarray = y_isotonic_regression + f_error_lower_interpolated(x_new)
    error_upper_interpolated: np.ndarray = y_isotonic_regression + f_error_upper_interpolated(x_new)

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
        results['interpolated_time'] = time_interpolated_axis   # TODO maybe not write this for every strategy, but once
    if 'interpolated_objective' in expected_results:
        results['interpolated_objective'] = y_isotonic_regression
    if 'interpolated_objective_std' in expected_results:
        results['interpolated_objective_std'] = y_std_li
    if 'interpolated_objective_error_lower' in expected_results:
        results['interpolated_objective_error_lower'] = error_lower_interpolated
    if 'interpolated_objective_error_upper' in expected_results:
        results['interpolated_objective_error_upper'] = error_upper_interpolated

    return results

    # # TODO plot
    # y_isotonic_regression_1 = get_isotonic_curve(x, y, x_new, ymin=y_min, ymax=y_median, package='sklearn', npoints=npoints)
    # y_isotonic_regression_3 = get_isotonic_curve(x, y, x_new, ymin=y_min, ymax=y_median, npoints=npoints, power=1.1)
    # import matplotlib.pyplot as plt
    # plt.plot(x,y,',')
    # # plt.plot(x_new, y_polynomial)
    # plt.plot(x, _y_isotonic_regression, label="temp_cutoff")
    # # plt.plot(x_new, y_isotonic_regression_1, label="SKLearn")
    # plt.plot(x_new, y_isotonic_regression, label="Isotonic")
    # # plt.plot(x_new, y_isotonic_regression_3, label="Isotonic p=1.1")
    # plt.plot(x_new, error_lower_interpolated, label="lower error interpolated")
    # plt.plot(x_new, error_upper_interpolated, label="upper error interpolated")
    # # plt.plot(error_lower_x, y_isotonic_regression[x_new_error_lower] + error_lower, label='lower error')
    # # plt.plot(error_upper_x, y_isotonic_regression[x_new_error_upper] + error_upper, label='lower error')
    # plt.xlim([x_new[0]-1, x_new[-1] + 1 ])
    # plt.xlabel("Cumulative time in seconds")
    # plt.ylabel("Minimum value found")
    # plt.legend()
    # plt.show()
    # exit(0)
