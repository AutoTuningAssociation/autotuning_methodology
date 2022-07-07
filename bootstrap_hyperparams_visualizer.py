from imp import reload
import json
import time
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
from typing import Tuple, Any

skip_hyperparams_list = ['max_fevals', 'time', 'times', 'strategy_time']
hyperparam_plot_index: dict[str, Any] = dict()
hyperparam_nonconstant_names: list[str] = list()


def reload_cache():
    with open('../cached_data_used/cache_files/bootstrap_hyperparamtuning.json', 'r') as fh:
        contents = fh.read()
        try:
            data = json.loads(contents)
        except json.decoder.JSONDecodeError:
            try:
                contents = contents[:-1] + '}}'
                data = json.loads(contents)
            except json.decoder.JSONDecodeError:
                contents = contents[:-3] + '}}'
                data = json.loads(contents)
        return data


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ma = ret[n - 1:] / n
    prefix = np.full(n - 1, np.mean(a[:n - 1]))
    return np.concatenate([prefix, ma])


def preload_hyperparameterdata() -> Tuple[int, list, dict, list, list]:
    cache = reload_cache()
    search_space_size = np.prod(tuple(len(params) for params in cache['tune_params'].values()))
    hyperparam_names = list(hp for hp in cache['tune_params_keys'] if hp not in skip_hyperparams_list)
    isfullynumeric = list()
    onehotencoding = list()    # add the list of unique values for each hyperparameter to easily do onehot encoding

    # create the fully numeric mask and one-hot encoded values
    nonconstant_counter = 0
    for key in hyperparam_names:
        values = cache['tune_params'][key]
        # do not display hyperparams that are constant
        constant = len(values) <= 1
        if not constant:
            hyperparam_plot_index[key] = nonconstant_counter
            hyperparam_nonconstant_names.append(key)
            nonconstant_counter += 1
        else:
            continue
        isfullynumeric.append(all(isinstance(h, (int, float)) for h in values))
        onehotencoding.append(list(values))
        # remove the boilerplate around the explorationfactor
        if not isfullynumeric[-1] and 'explorationfactor' in onehotencoding[-1][0]:
            for i in range(len(onehotencoding[-1])):
                onehotencoding[-1][i] = str(onehotencoding[-1][i])[22:-1]

    return search_space_size, hyperparam_names, cache, isfullynumeric, onehotencoding


def setup_plot_per_hyperparameter(subfigure, hyperparam_names, isfullynumeric, onehotencoding):
    # setup the plot
    subfigure.suptitle("Individual hyperparameters")
    axes = subfigure.subplots(1, len(isfullynumeric))
    plots = list()
    for i in range(len(isfullynumeric)):
        axis = axes[i] if len(isfullynumeric) > 1 else axes
        plots.append(axis.scatter([], [], cmap='viridis'))
        # axes[i].set_xticks(onehotencoding[i] if isfullynumeric[i] else range(len(onehotencoding[i])))
        axis.set_xticks(range(len(onehotencoding[i])))
        axis.set_xticklabels(onehotencoding[i])
        axis.set_title(hyperparam_nonconstant_names[i], fontsize=9)

    return axes, plots


def setup_plot_per_warning(subfigure, hyperparams):
    subfigure.suptitle("Mean number of warnings")
    warning_names = list(hyperparams['warnings'][0].keys())
    axes = subfigure.subplots(1, len(warning_names))
    plots = list()
    for i in range(len(warning_names)):
        axis = axes[i] if len(warning_names) > 1 else axes
        plots.append(axis.scatter([], [], cmap='viridis'))
        axis.set_title(warning_names[i], fontsize=9)
    return axes, plots, warning_names


def setup_plot_parallel_coordinates(subfigure, isfullynumeric, onehotencoding):
    print("TODO")


def plot_per_hyperparameter(axes, plots, times, hyperparams: dict, isnumeric: list, onehotencoding: list, cmap):
    # update each plot
    for index, y in enumerate(hyperparams.values()):
        hyperparam_name = list(hyperparams.keys())[index]
        if hyperparam_name == 'warnings' or hyperparam_name in skip_hyperparams_list or hyperparam_name not in hyperparam_plot_index.keys():
            continue
        index = hyperparam_plot_index[hyperparam_name]
        # if not isnumeric[index]:
        for i, e in enumerate(y):
            if not isnumeric[index] and 'explorationfactor' in e:
                e = str(e)[22:-1]
            y[i] = onehotencoding[index].index(e if isnumeric[index] else str(e))

        x = times
        x_offset = np.mean(x) * 0.2
        y_offset = np.mean(y) * 0.2
        plots[index].set_offsets(list(zip(y, x)))
        plots[index].set_facecolor(cmap)
        axis = axes[index] if isinstance(axes, np.ndarray) else axes
        if min(x) != max(x):
            axis.set_ylim(min(x) - x_offset, max(x) + x_offset)
        if min(y) != max(y):
            axis.set_xlim(min(y) - y_offset, max(y) + y_offset)


def plot_per_warning(axes, plots, times, hyperparams, warning_names, cmap):
    warnings = dict(zip(warning_names, list(list() for _ in warning_names)))
    for dct in hyperparams['warnings']:
        for key, value in dct.items():
            warnings[key].append(value)
    for index, warning_name in enumerate(warning_names):
        x = times
        y = warnings[warning_name]
        x_offset = np.mean(x) * 0.2
        y_offset = np.mean(y) * 0.2
        plots[index].set_offsets(list(zip(y, x)))
        plots[index].set_facecolor(cmap)
        axis = axes[index] if len(warning_names) > 1 else axes
        if min(x) != max(x):
            axis.set_ylim(min(x) - x_offset, max(x) + x_offset)
        if min(y) != max(y):
            axis.set_xlim(min(y) - y_offset, max(y) + y_offset)


def plot_parallel_coordinates(subfigure, hyperparams: dict, isnumeric: list, onehotencoding: list, cmap):
    host = subfigure.subplots()

    # make sure the numeric hyperparameters are converted and the others are one-hot encoded
    ynames = list(hyperparams.keys())
    ys_temp = list(np.dstack(list(hyperparams.values()))[0].tolist())
    for y in ys_temp:
        for i, e in enumerate(y):
            if isnumeric[i]:
                if np.char.isnumeric(e):
                    y[i] = int(e)
                else:
                    y[i] = float(e)
            else:
                y[i] = onehotencoding[i].index(e)
    ys = np.array(ys_temp)

    # organize the data
    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys * 0.05    # add 5% padding below and above
    ymaxs += dys * 0.05
    dys = ymaxs - ymins

    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    axes = [host] + [host.twinx() for _ in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()
    host.set_title('Parallel Coordinates Plot')

    colors = plt.cm.tab10.colors
    for j in range(len(ys)):
        # to just draw straight lines between the axes:
        # host.plot(range(ys.shape[1]), zs[j,:], c=colors[(category[j] - 1) % len(colors) ])

        # create bezier curves
        # for each axis, there will a control vertex at the point itself, one at 1/3rd towards the previous and one
        #   at one third towards the next axis; the first and last axis have one less control vertex
        # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
        # y-coordinate: repeat every point three times, except the first and last only twice
        verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)], np.repeat(zs[j, :], 3)[1:-1]))
        # for x,y in verts: host.plot(x, y, 'go') # to show the control points of the beziers
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=1, edgecolor=cmap[j])
        host.add_patch(patch)


# preload the cache
search_space_size, hyperparam_names, preloaded_cache, isfullynumeric, onehotencoding = preload_hyperparameterdata()

# set the stage
plt.ion()
plt.tight_layout()
cmap = plt.get_cmap("viridis")
f = plt.figure(constrained_layout=True, figsize=(10, 8))
subfigures = f.subfigures(4, 1)
ax_1 = subfigures[0].subplots(1, 1)
ax_2 = subfigures[1].subplots(1, 1)
fig_3 = subfigures[2]
fig_4 = subfigures[3]
ax_1.set_title("Grandmean MNE versus strategy time")
ax_1.set_ylabel("Grandmean MNE")
ax_1.set_xlabel("Strategy time in seconds")
ax_2.set_ylabel("Grandmean MNE")
ax_2.set_xlabel("Order of attempt")
# setup_plot_parallel_coordinates(fig_3, isfullynumeric, onehotencoding)
fig_3_axes, fig_3_plots = setup_plot_per_hyperparameter(fig_3, hyperparam_names, isfullynumeric, onehotencoding)
fig_4_axes, fig_4_plots = None, None

# set starting values
invalid_value = 1e20
best_yet = invalid_value
last_num_samples = 0
iterations = 56

# add the old methods as reference points
# MAE
# ax_1.errorbar([0.5675140840000026], [0.44807610254902946], fmt='go', label='random sampling')    # mean: 0.4578314064149997
# ax_1.errorbar([49.31634912499976], [0.232196639255723], fmt='ko', label='BO EI old')    # mean: 0.3202119772091951
# ax_1.errorbar([12.145443499999942], [0.1796706015192193], fmt='mo', label='BO EI LHS')    # mean: 0.2472707967000206)

# MRE
# ax_1.errorbar([0.5718150829999971], [0.05817556337825], fmt='go', label='random sampling')    # median: 0.06045303005
# ax_1.errorbar([61.31634912499976], [0.036183987295925], fmt='ko', label='BO EI old')    # median: 0.02833364508
# ax_1.errorbar([12.145443499999942], [0.032670246874297], fmt='mo', label='BO EI LHS')    # median: 0.027983519090599

# MNE
# ax_1.errorbar([0.6784772168], [0.002305952911221402], fmt='go', label='random sampling')
# ax_1.errorbar([75.12219994880002], [0.001822892334664491], fmt='ko', label='BO EI old')
# ax_1.errorbar([30.777686772599996], [0.001611096968835707], fmt='mo', label='BO EI LHS')

# # MNE 2 (median)
# ax_1.scatter([0.6700462250857142], [0.025970953266679253], label='random sampling', edgecolors='k', facecolors='b')
# ax_1.scatter([74.26725870240001], [0.017804530501047442], label='BO EI old', edgecolors='k', facecolors='tab:orange')
# ax_1.scatter([29.296166623685714], [0.016086731033092855], label='BO EI LHS', edgecolors='k', facecolors='r')
# static_x_values = [0.6700462250857142, 74.26725870240001, 29.296166623685714]
# static_y_values = [0.025970953266679253, 0.017804530501047442, 0.016086731033092855]

# # MNE 3 (std)
# plot1 = ax_1.scatter([], [], label='BO hyperparamtuning', cmap='viridis')
# ax_1.scatter([0.6804847034285715], [0.01928227862714799], label='random sampling', edgecolors='k', facecolors='b')
# ax_1.scatter([75.90840997131431], [0.016827752345157633], label='BO EI old', edgecolors='k', facecolors='tab:orange')
# ax_1.scatter([31.012469295171428], [0.012762337277518026], label='BO EI LHS', edgecolors='k', facecolors='r')
# static_x_values = [0.6804847034285715, 75.90840997131431, 31.012469295171428]
# static_y_values = [0.01928227862714799, 0.016827752345157633, 0.012762337277518026]

# # MNE 3 (std, grandmedian)
# plot1 = ax_1.scatter([], [], label='BO hyperparamtuning', cmap='viridis')
# ax_1.scatter([0.6500610310571429], [0.019283924152746993], label='random sampling', edgecolors='k', facecolors='b')
# ax_1.scatter([79.79173324522856], [0.0127291622968705], label='BO EI old', edgecolors='k', facecolors='tab:orange')
# ax_1.scatter([58.08916327734285], [0.012189615000319929], label='BO EI LHS', edgecolors='k', facecolors='r')
# static_x_values = [0.6500610310571429, 79.79173324522856, 58.08916327734285]
# static_y_values = [0.019283924152746993, 0.0127291622968705, 0.012189615000319929]

# # MNE 4 (std, grandmean of median)
# plot1 = ax_1.scatter([], [], label='BO hyperparamtuning', cmap='viridis')
# ax_1.scatter([0.6074726796857142], [0.03903], label='random sampling', edgecolors='k', facecolors='b')
# # ax_1.scatter([61.07047364405715], [0.012434733581032989], label='BO EI old', edgecolors='k', facecolors='tab:orange')
# ax_1.scatter([30.459359194085717], [0.0017592476656916748], label='BO EI LHS Gaussian', edgecolors='k', facecolors='r')
# static_x_values = [0.6074726796857142, 30.459359194085717]
# static_y_values = [0.03903, 0.0017592476656916748]
# ax_1.legend()

# MWP (mean weighted position, inverse-variance weighted average)
plot1 = ax_1.scatter([], [], label='BO hyperparamtuning', cmap='viridis')
ax_1.scatter([0.6607871937857144], [0.03915019], label='random sampling', edgecolors='k', facecolors='b')
ax_1.scatter([57.22632902153571], [0.0016490878339355402], label='BO EI old', edgecolors='k', facecolors='tab:orange')
ax_1.scatter([17.61113134375], [0.0007871754141325752], label='BO hyperparamtuned', edgecolors='k', facecolors='r')
static_x_values = [0.6607871937857144, 57.22632902153571, 17.61113134375]
static_y_values = [0.03915019, 0.0016490878339355402, 0.0007871754141325752]
ax_1.legend()

barchart = ax_2.bar([], [])
plot_moving_average, = ax_2.plot([], [], 'b')

while True:
    # load the data
    cache = reload_cache()['cache']
    keys, values = list(cache.keys()), cache.values()
    num_samples = len(keys)
    if last_num_samples == num_samples:
        plt.pause(2.0)
        continue    # nothing has changed, no need to update
    else:
        last_num_samples = num_samples

    # get the data
    times = np.array(list(v['times'] for v in values))
    times[times >= invalid_value] = np.nan
    x = np.array(list(v['strategy_time'] for v in values))
    y = np.array(list(v['time'] for v in values))
    y_err = np.array(list(np.std(v) for v in times))
    y_num_valid = np.array(list(np.count_nonzero(~np.isnan(v)) for v in times))

    # filter out invalids
    invalid_mask = (y < invalid_value) & (y_num_valid >= iterations)
    x_valid = x[invalid_mask]
    y_valid = y[invalid_mask]
    y_err_valid = y_err[invalid_mask]

    # scaled colormap
    y_valid_min = np.min(y_valid)
    y_valid_max = np.max(y_valid)
    if y_valid_min == y_valid_max:
        scaled_cmap = cmap(y_valid)
    else:
        diff = y_valid_max - y_valid_min
        rescale = lambda v: (v - y_valid_min) / diff
        scaled_cmap = cmap(rescale(y_valid))

    # obtain the hyperparameters for a parallel coordinates plot
    hyperparams: dict[str, Any] = dict()
    for hyperparam_name, hyperparam_values in list(values)[0].items():
        if hyperparam_name in skip_hyperparams_list:
            continue
        hyperparams[hyperparam_name] = list()
    for e in values:
        for hyperparam_name, hyperparam_value in e.items():
            if hyperparam_name in skip_hyperparams_list:
                continue
            hyperparams[hyperparam_name].append(hyperparam_value)
    # # if there are non-numeric types, cast to string to avoid mixed or complex types
    # isfullynumeric = list()
    # onehotencoding = list()    # add the list of unique values for each hyperparameter to easily do onehot encoding
    for hyperparam_name, hyperparam_values in hyperparams.items():
        hyperparams[hyperparam_name] = np.array(hyperparam_values)[invalid_mask].tolist()
        # if all(isinstance(h, (int, float)) for h in hyperparams[hyperparam_name]):
    #         isfullynumeric.append(True)
    #     else:
    #         isfullynumeric.append(False)
    #         hyperparams[hyperparam_name] = list(str(h) for h in hyperparams[hyperparam_name])
    #     onehotencoding.append(list(dict.fromkeys(hyperparams[hyperparam_name])))
    # plot_parallel_coordinates(fig_3, hyperparams, isfullynumeric, onehotencoding, scaled_cmap)
    plot_per_hyperparameter(fig_3_axes, fig_3_plots, y_valid, hyperparams, isfullynumeric, onehotencoding, scaled_cmap)

    # set the barcharts
    iterations_valid = range(len(x_valid) + 1)[1:]
    barchart = ax_2.bar(iterations_valid, y_valid, color=scaled_cmap, edgecolor="none")
    plot_moving_average.set_xdata(iterations_valid)
    plot_moving_average.set_ydata(moving_average(y_valid, round(np.sqrt(len(y_valid)) + 1)))
    ax_2.relim()
    ax_2.autoscale_view()
    ax_2.set_ylim(0, max(y_valid) + np.mean(y_valid) * 0.05)

    # find the optimum
    index = np.argmin(y)
    y_value = y[index]
    if y_value < best_yet:
        print(f"Found a new optimum out of {len(y_valid)} valid samples: {y_value} (strategy time {round(x[index], 3)} seconds).")
        best_yet = y_value

    # update the plot
    plot1.set_offsets(list(zip(x_valid, y_valid)))
    plot1.set_facecolor(scaled_cmap)

    # update the view and window
    x_offset = np.mean(x_valid) * 0.05
    y_offset = np.mean(y_valid) * 0.05
    x_lim = np.append(x_valid, static_x_values)
    y_lim = np.append(y_valid, static_y_values)
    ax_1.set_xlim(min(x_lim) - x_offset, max(x_lim) + x_offset)
    ax_1.set_ylim(min(y_lim) - y_offset, max(y_lim) + y_offset)
    f.canvas.manager.set_window_title(f"{num_samples} / {search_space_size}")
    title = f"Showing {len(y_valid)} valid samples. {len(y) - len(y_valid)} invalid. Total {num_samples}."
    f.suptitle(title)

    # set the warning plots
    if fig_4_axes is None:
        fig_4_axes, fig_4_plots, warning_names = setup_plot_per_warning(fig_4, hyperparams)
    plot_per_warning(fig_4_axes, fig_4_plots, y_valid, hyperparams, warning_names, scaled_cmap)

    # finally, update the canvas and wait for the next iteration
    f.canvas.draw()
    f.canvas.flush_events()
    plt.pause(10.0)


def plot_parallel(data, labels):

    data = np.array(data)
    x = list(range(len(data[0])))
    fig, axis = plt.subplots(1, len(data[0]) - 1, sharey=False)

    for d in data:
        for i, a in enumerate(axis):
            temp = d[i:i + 2].copy()
            temp[1] = (temp[1] - np.min(data[:, i + 1])) * (np.max(data[:, i]) - np.min(data[:, i])) / (np.max(data[:, i + 1]) -
                                                                                                        np.min(data[:, i + 1])) + np.min(data[:, i])
            a.plot(x[i:i + 2], temp)

    for i, a in enumerate(axis):
        a.set_xlim([x[i], x[i + 1]])
        a.set_xticks([x[i], x[i + 1]])
        a.set_xticklabels([labels[i], labels[i + 1]], minor=False, rotation=45)
        a.set_ylim([np.min(data[:, i]), np.max(data[:, i])])

    plt.subplots_adjust(wspace=0)

    plt.show()
