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
confidence_level = 0.95
dict_timings = dict()
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
dict_regressions = dict(
    {
        "sklearn_linear": {"use": False, "use_interval": False, "name": "SKLearn linear", "color": colors[0]},
        "sklearn_isotonic": {"use": True, "use_interval": False, "name": "SKLearn isotonic", "color": colors[1]},
        "isotonic_all": {"use": False, "use_interval": False, "name": "Isotonic (all)", "color": colors[2]},
        "isotonic_half": {"use": False, "use_interval": False, "name": "Isotonic (half)", "color": colors[3]},
        "sklearn_gradient_boosting": {"use": False, "use_interval": False, "name": "SKLearn gradient boosting", "color": colors[4]},
        "sklearn_isotonic_bagging": {"use": True, "use_interval": True, "name": "SKLearn isotonic bagging", "color": colors[5]},
    }
)

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


def calculate_confidence_interval(values: np.ndarray):
    """Helper function for calculating a confidence interval on a 2D array"""
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


# Calculate regressions
for key, reginfo in dict_regressions.items():
    if reginfo["use"]:
        start_perf_counter = perf_counter()

        # check the type of regression and calculate accordingly
        if key == "sklearn_linear":
            lr = LinearRegression().fit(x_2d, y)
            y_pred = lr.predict(x_test_2d)
        elif key == "sklearn_isotonic":
            ir = IsotonicRegression().fit(x, y)
            y_pred = ir.predict(x_test)
        elif key == "isotonic_all":
            ir2linear_all = LpIsotonicRegression(N, increasing=True, curve_algo=PiecewiseLinearIsotonicCurve)
            ir2linear_all_fit = ir2linear_all.fit(x, y)
            y_pred = ir2linear_all_fit.predict_proba(x_test)
        elif key == "isotonic_half":
            ir2linear_half = LpIsotonicRegression(round(N / 10), increasing=True, curve_algo=PiecewiseLinearIsotonicCurve)
            ir2linear_half_fit = ir2linear_half.fit(x, y)
            y_pred = ir2linear_half_fit.predict_proba(x_test)
        elif key == "sklearn_gradient_boosting":
            gbr = GradientBoostingRegressor(loss="quantile", alpha=0.5).fit(x_2d, y)  # predicts median
            y_pred = gbr.predict(x_test_2d)
        elif key == "sklearn_isotonic_bagging":
            br = BaggingRegressor(IsotonicRegression(), n_estimators=round(N / 5), bootstrap=True).fit(x_2d, y)
            y_pred = br.predict(x_test_2d)
        else:
            raise KeyError(f"Regression method key '{key}' unkown")

        # write the prediction and timing to the dicts
        dict_regressions[key]["y_pred"] = y_pred
        dict_timings[reginfo["name"]] = perf_counter() - start_perf_counter


# Calculate intervals
for key, reginfo in dict_regressions.items():
    if reginfo["use_interval"]:
        start_perf_counter = perf_counter()

        # check the type of interval and calculate accordingly
        if key == "sklearn_gradient_boosting":
            # Gradient Boosting Regressor (based on https://towardsdatascience.com/how-to-generate-prediction-intervals-with-scikit-learn-and-python-ab3899f992ed)
            lower_alpha = 1 - confidence_level
            upper_alpha = confidence_level
            gbr_lower = GradientBoostingRegressor(loss="quantile", alpha=lower_alpha).fit(x_2d, y)
            gbr_upper = GradientBoostingRegressor(loss="quantile", alpha=upper_alpha).fit(x_2d, y)
            y_lower_err = gbr_lower.predict(x_test_2d)
            y_upper_err = gbr_upper.predict(x_test_2d)
        elif key == "sklearn_isotonic_bagging":
            # Bagging Regressor (based on https://stats.stackexchange.com/questions/183230/bootstrapping-confidence-interval-from-a-regression-prediction)
            br_collection = np.array([m.predict(x_test_2d) for m in br.estimators_])  # yields 2D array with shape (run, x_test)
            y_lower_err, y_upper_err = calculate_confidence_interval(br_collection.transpose())
        else:
            raise KeyError(f"Interval method key '{key}' unkown")

        # write the errors and timing to the dicts
        dict_regressions[key]["y_lower_err"] = y_lower_err
        dict_regressions[key]["y_upper_err"] = y_upper_err
        dict_timings[f"{reginfo['name']} error"] = perf_counter() - start_perf_counter


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
# alpha_arr = np.full(N, 1 - confidence_level)
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

# plot regressors and errors
for key, reginfo in dict_regressions.items():
    # plot the error first so the regression line comes on top of it
    if reginfo["use_interval"]:
        plot.varea(x_test, y1=reginfo["y_lower_err"], y2=reginfo["y_upper_err"], alpha=0.3, color=reginfo["color"])
    if reginfo["use"]:
        plot.line(x_test, reginfo["y_pred"], color=reginfo["color"], legend_label=reginfo["name"])

# plot setup
plot.legend.location = "bottom_right"
show(plot)


# plot the performance
fig = plt.figure(figsize=(20, 8))
plt.bar(dict_timings.keys(), dict_timings.values(), zorder=3)
plt.yscale("log")
plt.ylabel("Time in seconds")
plt.title(f"Timing comparison with {N=}")
plt.grid(axis="y", which="both", zorder=0)
plt.show()
