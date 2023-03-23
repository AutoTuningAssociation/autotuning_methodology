import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_notebook, show
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from isotonic.isotonic import LpIsotonicRegression
from isotonic.isotonic.curves import PiecewiseLinearIsotonicCurve, PiecewiseConstantIsotonicCurve
from scipy.stats import norm, bernoulli  # for isotonic dataset only
import numpy as np
from time import perf_counter


# basic setup
alpha = 0.95
confidence_level = 0.95
dict_timings = dict()
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
dict_regressions = dict(
    {
        "sklearn_linear": {"use": True, "name": "SKLearn linear", "color": colors[0]},
        "sklearn_isotonic": {"use": True, "name": "SKLearn isotonic", "color": colors[1]},
        "isotonic_all": {"use": False, "name": "Isotonic (all)", "color": colors[2]},
        "isotonic_half": {"use": False, "name": "Isotonic (half)", "color": colors[3]},
        "sklearn_isotonic_bagging": {"use": True, "name": "SKLearn isotonic bagging", "color": colors[4]},
    }
)

# select errors
use_gradient_boosting_regressor_error = False
use_bagging_regressor_error = dict_regressions["sklearn_isotonic_bagging"]["use"] and True

# Sample dataset (from https://www.geeksforgeeks.org/isotonic-regression-in-scikit-learn/)
# start_perf_counter = perf_counter()
N = 200
x = np.arange(N)
x_2d = x.reshape(-1, 1)
# print('Input:\n', x)
y = np.random.randint(0, 20, size=N) + 10 * np.log1p(np.arange(N))
# print("Target :\n", y)
x_test = x
x_test_2d = x_test.reshape(-1, 1)
# dict_timings["loading dataset"] = perf_counter() - start_perf_counter

# # Sample dataset (from https://github.com/stucchio/isotonic)
# # start_perf_counter = perf_counter()
# N = 500
# x = norm(0, 50).rvs(N) - bernoulli(0.25).rvs(N) * 50
# y = -7 + np.sqrt(np.maximum(x, 0)) + norm(0, 0.5).rvs(N)
# x_test = np.arange(x.min(), x.max(), 0.01)
# # dict_timings["loading dataset"] = perf_counter() - start_perf_counter


# Regressions

# Fit linear regression model for comparison
# create an instance of the LinearRegression class
if dict_regressions["sklearn_linear"]["use"]:
    start_perf_counter = perf_counter()
    lr = LinearRegression().fit(x_2d, y)
    # make predictions using the fitted model
    y_lr = lr.predict(x_test_2d)
    dict_timings["sklearn linear"] = perf_counter() - start_perf_counter

# Fit isotonic regression model
# create an instance of the IsotonicRegression class
if dict_regressions["sklearn_isotonic"]["use"]:
    start_perf_counter = perf_counter()
    ir = IsotonicRegression().fit(x, y)
    # make predictions using the fitted model
    y_ir = ir.predict(x_test)
    dict_timings["sklearn isotonic"] = perf_counter() - start_perf_counter

# Isotonic regression with Isotonic package (all segments)
if dict_regressions["isotonic_all"]["use"]:
    start_perf_counter = perf_counter()
    ir2linear_all = LpIsotonicRegression(N, increasing=True, curve_algo=PiecewiseLinearIsotonicCurve)
    ir2linear_all_fit = ir2linear_all.fit(x, y)
    y_ir2linear_all = ir2linear_all_fit.predict_proba(x_test)
    dict_timings["isotonic all"] = perf_counter() - start_perf_counter

# Isotonic regression with Isotonic package (half segments)
if dict_regressions["isotonic_half"]["use"]:
    start_perf_counter = perf_counter()
    ir2linear_half = LpIsotonicRegression(round(N / 10), increasing=True, curve_algo=PiecewiseLinearIsotonicCurve)
    ir2linear_half_fit = ir2linear_half.fit(x, y)
    y_ir2linear_half = ir2linear_half_fit.predict_proba(x_test)
    dict_timings["isotonic half"] = perf_counter() - start_perf_counter

# Bagging Regressor
if dict_regressions["sklearn_isotonic_bagging"]["use"]:
    start_perf_counter = perf_counter()
    br = BaggingRegressor(IsotonicRegression(), n_estimators=round(N / 5), bootstrap=True).fit(x_2d, y)
    y_br = br.predict(x_test_2d)
    dict_timings["sklearn isotonic bagging"] = perf_counter() - start_perf_counter


# Error calculations


def calculate_confidence_interval(values: np.ndarray):
    from statistics import NormalDist
    from math import sqrt, floor, ceil

    assert values.ndim == 2

    distribution = NormalDist()  # TODO check if binomial is more appropriate (calculate according to book)
    z = distribution.inv_cdf((1 + confidence_level) / 2.0)
    n = values.shape[1]
    q = 0.5
    nq = n * q
    base = z * sqrt(nq * (1 - q))
    lower_rank = max(floor(nq - base), 0)
    upper_rank = min(ceil(nq + base), n - 1)
    confidence_interval_lower = np.full(values.shape[0], np.nan)
    confidence_interval_upper = np.full(values.shape[0], np.nan)

    # # confidence interval according to Hoefler 2015 (student-t)
    # alpha = 1 - confidence_level
    # mean = values.mean()
    # t = t(n - 1, alpha / 2)

    # for each step on x, look up the confidence interval
    values_sorted = np.sort(values, axis=1)
    for x_index, x_repeats_sorted in enumerate(values_sorted):
        confidence_interval_lower[x_index] = x_repeats_sorted[lower_rank]
        confidence_interval_upper[x_index] = x_repeats_sorted[upper_rank]

    return confidence_interval_lower, confidence_interval_upper


# Bagging Regressor (based on https://stats.stackexchange.com/questions/183230/bootstrapping-confidence-interval-from-a-regression-prediction)
if use_bagging_regressor_error:
    start_perf_counter = perf_counter()
    br_collection = np.array([m.predict(x_test_2d) for m in br.estimators_])  # yields 2D array with shape (run, x_test)
    y_br_lower, y_br_upper = calculate_confidence_interval(br_collection.transpose())
    dict_timings["sklearn isotonic bagging error"] = perf_counter() - start_perf_counter


# Gradient Boosting Regressor (based on https://towardsdatascience.com/how-to-generate-prediction-intervals-with-scikit-learn-and-python-ab3899f992ed)
if use_gradient_boosting_regressor_error:
    start_perf_counter = perf_counter()
    gbr_lower = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha).fit(x_2d, y)
    gbr_upper = GradientBoostingRegressor(loss="quantile", alpha=alpha).fit(x_2d, y)
    y_gbr_lower = gbr_lower.predict(x_test_2d)
    y_gbr_upper = gbr_upper.predict(x_test_2d)
    dict_timings["GBR error"] = perf_counter() - start_perf_counter

# # scipy isotonic regression error
# total_sum_of_squares = ((y - y.mean()) ** 2).sum()
# residual_sum_of_squares = total_sum_of_squares * (1 - ir.score(x, y))
# # TODO instead take ((y_true - y_pred)** 2) to get error at each point
# print(f"Scipy error: {np.sqrt(residual_sum_of_squares)}")

# why can we not have a confidence interval in isotonic regression?
# -> because we can not get multiple values (one for each repeat) at a single point in time like with index
# --> instead, we look up in each repeat the x-value closest to each x_test (x[i]) and its index (i)
# ---> for each repeat, we can now use the raw value y[i] this gives the error abs(y_test - y[i])
# ----> optionally, values closer to x can be given more importance by taking 1 - (abs(x_test - x[i]) / sum(abs(x_test - x[i]) for each repeat))

# # error calculation own method
# def index_of_nearest(array, value):
#     """Find the indices of the closest given values in array"""
#     idx = np.searchsorted(array, value, side="left")
#     idx = idx - (np.abs(value - array[idx - 1]) < np.abs(value - array[idx]))
#     return idx


# # find the closest x-value in the raw data for each test value
# indices_real = index_of_nearest(x, x_test)
# # for each element in the resolution
# for i_test, i_real in enumerate(indices_real):
#     # get the index on either side
#     i_l = max(i_real - 1, 0)
#     i_r = min(i_real + 1, len(x_test) - 1)
#     # get the distance to the real y-value on either side
#     y_l = np.abs(y_ir[i_test] - y[i_l]) if i_test != i_l else np.nan
#     y_r = np.abs(y_ir[i_test] - y[i_r]) if i_test != i_r else np.nan
#     # get the distance to the real x-value on either side
#     x_l = np.abs(x_test[i_test] - x[i_l])
#     x_r = np.abs(x_test[i_test] - x[i_l])
#     sum_dist = x_l + x_r
#     # get the importance by taking the distance on x and reversing it, because the closer to the real value it is, because the closer to x_test, the more importance they should have
#     imp_l = x_l / sum_dist
#     err_l = y_l * imp_l
#     err = np.mean([])

# # isotonic regression error
# alpha_arr = np.full(N, 1 - alpha)
# print(ir2linear_all._err_func(x_test, x, y)(alpha_arr))
# print(ir2linear_all._grad_err_func(x_test, x, y)(alpha_arr))
# print(ir2linear_half._err_func(x_test, x, y)(alpha_arr))
# print(ir2linear_half._grad_err_func(x_test, x, y)(alpha_arr))


# # Plot the results (from https://www.geeksforgeeks.org/isotonic-regression-in-scikit-learn/)
# plt.plot(x, y, 'o', label='data')    # plot the original data
# # plot the fitted linear regression model
# plt.plot(x, y_lr, label='linear regression')
# # plot the fitted isotonic regression model
# plt.plot(x, y_ir, label='isotonic regression')
# plt.plot(x, y_ir2linear_all, label='isotonic regression (all segments)')
# plt.plot(x, y_ir2linear_half, label='isotonic regression (half segments)')
# plt.legend()    # add a legend

# # Add labels and title
# plt.xlabel('X')    # add x-axis label
# plt.ylabel('Y')    # add y-axis label
# plt.title('Comparison of Regression Techniques')    # add title
# plt.show()    # show the plot

# Plot the results (from https://github.com/stucchio/isotonic)
output_notebook()
plot = figure(tools="pan,box_zoom,reset,save,", y_axis_label="y", title="Comparison of isotonic regression and interval methods", x_axis_label="x")

# plot raw data
plot.circle(x, y, color="black", alpha=0.2, legend_label="raw data")

# plot regressors
if dict_regressions["sklearn_linear"]["use"]:
    reginfo = dict_regressions["sklearn_linear"]
    plot.line(x_test, y_lr, color=reginfo["color"], legend_label=reginfo["name"])
if dict_regressions["sklearn_isotonic"]["use"]:
    reginfo = dict_regressions["sklearn_isotonic"]
    plot.line(x_test, y_ir, color=reginfo["color"], legend_label=reginfo["name"])
if dict_regressions["isotonic_all"]["use"]:
    reginfo = dict_regressions["isotonic_all"]
    plot.line(x_test, y_ir2linear_all, color=reginfo["color"], legend_label=reginfo["name"])
if dict_regressions["isotonic_half"]["use"]:
    reginfo = dict_regressions["isotonic_half"]
    plot.line(x_test, y_ir2linear_half, color=reginfo["color"], legend_label=reginfo["name"])
if dict_regressions["sklearn_isotonic_bagging"]["use"]:
    reginfo = dict_regressions["sklearn_isotonic_bagging"]
    plot.line(x_test, y_br, color=reginfo["color"], legend_label=reginfo["name"])

# # plot errors

# gradient boosting regressor error
if use_gradient_boosting_regressor_error:
    # plot.line(x_test, y_gbr_lower, alpha=0.5, color="orange", legend_label="bagging regressor error")
    # plot.line(x_test, y_gbr_upper, alpha=0.5, color="orange")
    plot.varea(x_test, y1=y_gbr_lower, y2=y_gbr_upper, alpha=0.3, color="orange")

# bagging regressor: plot each base estimator
if use_bagging_regressor_error:
    # for bre in br_collection:
    #     plot.line(x_test, bre, color="grey", alpha=0.2)
    # plot.line(x_test, y_br_lower)
    # plot.line(x_test, y_br_upper)
    plot.varea(x_test, y1=y_br_lower, y2=y_br_upper, alpha=0.3, color=dict_regressions["sklearn_isotonic_bagging"]["color"])

# plot setup
plot.legend.location = "bottom_right"
show(plot)


# plot the performance
fig = plt.figure(figsize=(12, 8))
plt.bar(dict_timings.keys(), dict_timings.values())
plt.yscale("log")
plt.ylabel("Time in seconds")
plt.title(f"Timing comparison with {N=}")
plt.show()
