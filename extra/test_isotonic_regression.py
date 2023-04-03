# import utilities
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_notebook, show

# import regression and interval models
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor


# Parameters
N = 500
selected_dataset = 0
increasing = [True, True][selected_dataset]
confidence_level = 0.95
constant_colors = False  # if true, the colors are assigned in the order of dict_regressions. if false, only used algorithms are assigned colors

# Configuration of regression algorithms
dict_timings = dict()
dict_regressions = dict(
    {
        "sklearn_linear": {"use": False, "use_interval": False, "name": "SKLearn linear"},
        "sklearn_isotonic": {"use": True, "use_interval": False, "name": "SKLearn isotonic"},
        "isotonic_all": {"use": False, "use_interval": False, "name": "Isotonic (all)"},
        "isotonic_half": {"use": False, "use_interval": False, "name": "Isotonic (half)"},
        "isotonic_constant": {"use": False, "use_interval": False, "name": "Isotonic (half, constant)"},
        "sklearn_gradient_boosting": {"use": False, "use_interval": False, "name": "SKLearn gradient boosting"},
        "sklearn_isotonic_bagging": {"use": False, "use_interval": False, "name": "SKLearn isotonic bagging"},
        "inductive_conformal_prediction": {"use": False, "use_interval": True, "name": "Inductive Conformal Prediction"},
        "conformal_prediction_crepes": {"use": False, "use_interval": False, "name": "Conformal Prediction"},
        "conformal_prediction_crepes_normalized": {"use": False, "use_interval": True, "name": "Conformal Prediction Normalized"},
        "conformal_prediction_crepes_mondrian": {"use": False, "use_interval": True, "name": "Conformal Prediction Mondrian"},
    }
)

# Set the colors
color_index = 0
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for key in dict_regressions.keys():
    if constant_colors or (dict_regressions[key]["use"] or dict_regressions[key]["use_interval"]):
        dict_regressions[key]["color"] = colors[color_index]
        color_index += 1

# Conditional imports
if dict_regressions["isotonic_all"]["use"] or dict_regressions["isotonic_half"]["use"] or dict_regressions["isotonic_constant"]["use"]:
    # can be installed from https://github.com/stucchio/isotonic
    from isotonic.isotonic import LpIsotonicRegression
    from isotonic.isotonic.curves import PiecewiseLinearIsotonicCurve, PiecewiseConstantIsotonicCurve
if dict_regressions["inductive_conformal_prediction"]["use"] or dict_regressions["inductive_conformal_prediction"]["use_interval"]:
    # can be installed with 'pip install nonconformist'
    from nonconformist.cp import IcpRegressor
    from nonconformist.nc import NcFactory, SignErrorErrFunc
if dict_regressions["conformal_prediction_crepes"]["use_interval"] or dict_regressions["conformal_prediction_crepes_normalized"]["use_interval"]:
    from crepes import ConformalRegressor
    from crepes.fillings import sigma_knn, binning

# Data setup
start_perf_counter = perf_counter()
datasets = [0, 1]
if selected_dataset == 0:
    # Logarithmic sample data from https://www.geeksforgeeks.org/isotonic-regression-in-scikit-learn/
    np.random.seed(42)
    x = np.arange(N)
    y = np.random.randint(0, 20, size=N) + 10 * np.log1p(np.arange(N))
elif selected_dataset == 1:
    # Bernoulli sample data from https://github.com/stucchio/isotonic
    from scipy.stats import norm, bernoulli

    x = norm(0, 50).rvs(N) - bernoulli(0.25).rvs(N) * 50
    y = -7 + np.sqrt(np.maximum(x, 0)) + norm(0, 0.5).rvs(N)
else:
    raise KeyError(f"Selected dataset {selected_dataset} not in {datasets=}")
x_test = np.linspace(start=x.min(), stop=x.max(), num=round(N * 2.5))
x_2d = x.reshape(-1, 1)
x_test_2d = x_test.reshape(-1, 1)
dict_timings["loading dataset"] = perf_counter() - start_perf_counter


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
            ir = IsotonicRegression(y_min=y.min(), y_max=y.max(), increasing=increasing, out_of_bounds="clip").fit(x, y)
            y_pred = ir.predict(x_test)
        elif key == "isotonic_all":
            ir2linear_all = LpIsotonicRegression(N, increasing=increasing, curve_algo=PiecewiseLinearIsotonicCurve)
            ir2linear_all_fit = ir2linear_all.fit(x, y)
            y_pred = ir2linear_all_fit.predict_proba(x_test)
        elif key == "isotonic_half":
            ir2linear_half = LpIsotonicRegression(round(N / 5), increasing=increasing, curve_algo=PiecewiseLinearIsotonicCurve)
            ir2linear_half_fit = ir2linear_half.fit(x, y)
            y_pred = ir2linear_half_fit.predict_proba(x_test)
        elif key == "isotonic_constant":
            ir2constant = LpIsotonicRegression(round(N / 5), increasing=increasing, curve_algo=PiecewiseConstantIsotonicCurve)
            ir2constant_fit = ir2constant.fit(x, y)
            y_pred = ir2constant_fit.predict_proba(x_test)
        elif key == "sklearn_gradient_boosting":
            gbr = GradientBoostingRegressor(loss="quantile", alpha=0.5).fit(x_2d, y)  # predicts median
            y_pred = gbr.predict(x_test_2d)
        elif key == "sklearn_isotonic_bagging":
            br = BaggingRegressor(IsotonicRegression(), n_estimators=round(N / 5), bootstrap=True).fit(x_2d, y)
            y_pred = br.predict(x_test_2d)
        elif (
            key == "inductive_conformal_prediction"
            or key == "conformal_prediction_crepes"
            or key == "conformal_prediction_crepes_normalized"
            or key == "conformal_prediction_crepes_mondrian"
        ):
            raise NotImplementedError(f"{reginfo['name']} is not implemented as a regressor, only as an interval estimator")
        else:
            raise KeyError(f"Regression method key '{key}' unkown")

        # write the prediction and timing to the dicts
        dict_regressions[key]["y_pred"] = y_pred
        dict_timings[str(reginfo["name"])] = perf_counter() - start_perf_counter


# Calculate intervals
for key, reginfo in dict_regressions.items():
    if reginfo["use_interval"]:
        start_perf_counter = perf_counter()

        if key.startswith("conformal_prediction_crepes"):
            # Conformal Point Prediction (based on https://proceedings.mlr.press/v179/bostrom22a/bostrom22a.pdf)
            # divide data into training and calibration set (75%-25%)
            cutoff = round(N * 0.75)
            random_indices = np.random.permutation(N)
            indices_train, indices_calibrate = random_indices[:cutoff], random_indices[cutoff:]
            assert len(indices_train) + len(indices_calibrate) == N
            x_calibrate_2d = x_2d[indices_calibrate, :]

            # create the regression model, nonconformity function and inductive conformal regressor
            regression_model = IsotonicRegression(y_min=y.min(), y_max=y.max(), increasing=increasing, out_of_bounds="clip")
            regression_model.fit(x_2d[indices_train, :], y[indices_train])
            prediction = regression_model.predict(x_test_2d)

            # get the difference in prediction and true values on the calibration set
            prediction_calibrated = regression_model.predict(x_calibrate_2d)
            residuals_calibrated = y[indices_calibrate] - prediction_calibrated

        # check the type of interval and calculate accordingly
        if key == "sklearn_gradient_boosting":
            # Gradient Boosting Regressor (based on https://towardsdatascience.com/how-to-generate-prediction-intervals-with-scikit-learn-and-python-ab3899f992ed)
            lower_alpha = (1 - confidence_level) / 2
            upper_alpha = 1 - lower_alpha
            gbr_lower = GradientBoostingRegressor(loss="quantile", alpha=lower_alpha).fit(x_2d, y)
            gbr_upper = GradientBoostingRegressor(loss="quantile", alpha=upper_alpha).fit(x_2d, y)
            y_lower_err = gbr_lower.predict(x_test_2d)
            y_upper_err = gbr_upper.predict(x_test_2d)

        elif key == "sklearn_isotonic_bagging":
            # Bagging Regressor (based on https://stats.stackexchange.com/questions/183230/bootstrapping-confidence-interval-from-a-regression-prediction)
            br_collection = np.array([m.predict(x_test_2d) for m in br.estimators_])  # yields 2D array with shape (run, x_test)
            y_lower_err, y_upper_err = calculate_confidence_interval(br_collection.transpose())

        elif key == "inductive_conformal_prediction":
            # Inductive Conformal Point Prediction (based on https://arxiv.org/pdf/2107.00363.pdf, page 16 & 17)
            # divide data into training and calibration set (75%-25%)
            cutoff = round(N * 0.75)
            random_indices = np.random.permutation(N)
            indices_train, indices_calibrate = random_indices[:cutoff], random_indices[cutoff:]
            assert len(indices_train) + len(indices_calibrate) == N

            # create the regression model, nonconformity function and inductive conformal regressor
            regression_model = IsotonicRegression(y_min=y.min(), y_max=y.max(), increasing=increasing, out_of_bounds="clip")
            nonconformity_function = NcFactory.create_nc(regression_model, err_func=SignErrorErrFunc())
            inductive_conformal_regressor = IcpRegressor(nonconformity_function)

            # fit and calibrate the ICP
            inductive_conformal_regressor.fit(x_2d[indices_train, :], y[indices_train])
            inductive_conformal_regressor.calibrate(x_2d[indices_calibrate, :], y[indices_calibrate])

            # predict the interval
            prediction_interval = inductive_conformal_regressor.predict(x_test_2d, significance=1 - confidence_level)
            y_lower_err, y_upper_err = prediction_interval[:, 0], prediction_interval[:, 1]

        elif key == "conformal_prediction_crepes":
            # fit a conformal regressor
            cr = ConformalRegressor()
            cr.fit(residuals=residuals_calibrated)

            # get the prediction intervals for the test set
            prediction_interval = cr.predict(y_hat=prediction, confidence=confidence_level)
            y_lower_err, y_upper_err = prediction_interval[:, 0], prediction_interval[:, 1]

        elif key == "conformal_prediction_crepes_normalized":
            # fit a normalized conformal regressor
            sigmas_cal = sigma_knn(X=x_calibrate_2d, residuals=residuals_calibrated)
            cr_norm = ConformalRegressor()
            cr_norm.fit(residuals=residuals_calibrated, sigmas=sigmas_cal)

            # generate difficulty estimates for the test set
            sigmas_test = sigma_knn(X=x_calibrate_2d, residuals=residuals_calibrated, X_test=x_test_2d)

            # get the prediction intervals for the test set
            prediction_interval = cr_norm.predict(y_hat=prediction, confidence=confidence_level, sigmas=sigmas_test)
            y_lower_err, y_upper_err = prediction_interval[:, 0], prediction_interval[:, 1]
            print("hello")

        else:
            raise KeyError(f"Interval method key '{key}' unkown")

        # write the errors and timing to the dicts
        print(key)
        assert y_lower_err.shape == y_upper_err.shape
        dict_regressions[key]["y_lower_err"] = y_lower_err
        dict_regressions[key]["y_upper_err"] = y_upper_err
        dict_timings[f"{reginfo['name']} error"] = perf_counter() - start_perf_counter

# Plot the results (from https://github.com/stucchio/isotonic)
output_notebook()
plot = figure(tools="pan,box_zoom,reset,save,", y_axis_label="y", title="Comparison of isotonic regression and interval methods", x_axis_label="x")

# plot raw data
plot.circle(x, y, color="black", alpha=0.2, legend_label="raw data")

# plot the intervals
for key, reginfo in dict_regressions.items():
    # plot the intervals first so the regression line comes on top of it
    if reginfo["use_interval"]:
        if reginfo["use"]:
            plot.varea(x_test, y1=reginfo["y_lower_err"], y2=reginfo["y_upper_err"], alpha=0.3, color=reginfo["color"])
        else:
            # if only the interval shade is drawn, add it to the legend
            plot.varea(x_test, y1=reginfo["y_lower_err"], y2=reginfo["y_upper_err"], alpha=0.3, color=reginfo["color"], legend_label=reginfo["name"])

# plot the regressors
for key, reginfo in dict_regressions.items():
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
