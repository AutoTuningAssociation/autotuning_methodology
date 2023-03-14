from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from isotonic.isotonic import LpIsotonicRegression
from isotonic.isotonic.curves import PiecewiseLinearIsotonicCurve, PiecewiseConstantIsotonicCurve
from scipy.stats import norm, bernoulli    # for isotonic dataset only
import numpy as np
import matplotlib.pyplot as plt

# Sample dataset (from https://www.geeksforgeeks.org/isotonic-regression-in-scikit-learn/)
N = 20
x = np.arange(N)
# print('Input:\n', x)
y = np.random.randint(0, 20, size=N) + 10 * np.log1p(np.arange(N))
# print("Target :\n", y)
x_test = x

# # Sample dataset (from https://github.com/stucchio/isotonic)
# N = 500
# x = norm(0, 50).rvs(N) - bernoulli(0.25).rvs(N) * 50
# y = -7 + np.sqrt(np.maximum(x, 0)) + norm(0, 0.5).rvs(N)
# x_test = np.arange(x.min(), x.max(), 0.01)

# Fit isotonic regression model
# create an instance of the IsotonicRegression class
ir = IsotonicRegression().fit(x, y)

# fit the model and transform the data
y_ir = ir.predict(x_test)

# Fit linear regression model

# create an instance of the LinearRegression class
lr = LinearRegression()

# fit the model to the data
lr.fit(x.reshape(-1, 1), y)

# make predictions using the fitted model
y_lr = lr.predict(x.reshape(-1, 1))

# Isotonic regression with Isotonic
ir2linear_all = LpIsotonicRegression(N, increasing=True, curve_algo=PiecewiseLinearIsotonicCurve)
ir2linear_all_fit = ir2linear_all.fit(x, y)
y_ir2linear_all = ir2linear_all_fit.predict_proba(x_test)
ir2linear_half = LpIsotonicRegression(N / 2, increasing=True, curve_algo=PiecewiseLinearIsotonicCurve)
ir2linear_half_fit = ir2linear_half.fit(x, y)
y_ir2linear_half = ir2linear_half_fit.predict_proba(x_test)

# Error calculations
# scipy isotonic regression error
total_sum_of_squares = ((y - y.mean())**2).sum()
residual_sum_of_squares = total_sum_of_squares * (1 - ir.score(x, y))
# TODO instead take ((y_true - y_pred)** 2) to get error at each point
print(f"Scipy error: {np.sqrt(residual_sum_of_squares)}")

# why can we not have a confidence interval in isotonic regression?
# -> because we can not get multiple values (one for each repeat) at a single point in time like with index
# --> instead, we look up in each repeat the x-value closest to each x_test (x[i]) and its index (i)
# ---> for each repeat, we can now use the raw value y[i] this gives the error abs(y_test - y[i])
# ----> optionally, values closer to x can be given more importance by taking 1 - (abs(x_test - x[i]) / sum(abs(x_test - x[i]) for each repeat))


# error calculation own method
def index_of_nearest(array, value):
    """ Find the indices of the closest given values in array """
    idx = np.searchsorted(array, value, side="left")
    idx = idx - (np.abs(value - array[idx - 1]) < np.abs(value - array[idx]))
    return idx


# find the closest x-value in the raw data for each test value
indices_real = index_of_nearest(x, x_test)
# for each element in the resolution
for i_test, i_real in enumerate(indices_real):
    # get the index on either side
    i_l = max(i_real - 1, 0)
    i_r = min(i_real + 1, len(x_test) - 1)
    # get the distance to the real y-value on either side
    y_l = np.abs(y_ir[i_test] - y[i_l]) if i_test != i_l else np.nan
    y_r = np.abs(y_ir[i_test] - y[i_r]) if i_test != i_r else np.nan
    # get the distance to the real x-value on either side
    x_l = np.abs(x_test[i_test] - x[i_l])
    x_r = np.abs(x_test[i_test] - x[i_l])
    sum_dist = x_l + x_r
    # get the importance by taking the distance on x and reversing it, because the closer to the real value it is, because the closer to x_test, the more importance they should have
    imp_l = x_l / sum_dist
    err_l = y_l * imp_l
    err = np.mean([])

# isotonic regression error
alpha = np.full(N, 0.05)
print(ir2linear_all._err_func(x_test, x, y)(alpha))
print(ir2linear_all._grad_err_func(x_test, x, y)(alpha))
print(ir2linear_half._err_func(x_test, x, y)(alpha))
print(ir2linear_half._grad_err_func(x_test, x, y)(alpha))

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
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import Span, LinearAxis, Range1d, ColumnDataSource

output_notebook()

plot = figure(tools="pan,box_zoom,reset,save,", y_axis_label="y", title="isotonic", x_axis_label='x')

plot.line(x_test, y_ir, color='green', legend_label='scikit isotonic regression')
plot.line(x_test, y_ir2linear_all, color='red', legend_label='isotonic regression (all segments)')
plot.line(x_test, y_ir2linear_half, color='blue', legend_label='isotonic regression (half segments)')

plot.circle(x, y, color='black', alpha=0.2, legend_label='raw data')

plot.legend.location = "bottom_right"
show(plot)
