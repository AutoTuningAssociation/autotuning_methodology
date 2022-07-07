""" Generate a cachefile using a synthetic function
    usage: python generate_cachefile.py
"""

import itertools
import time
import json
from random import randint
from random import uniform as randuni
from collections import OrderedDict
import numpy as np
from math import sin, cos, pi, sqrt, ceil
from typing import Tuple

import progressbar
import multiprocessing

from kernel_tuner import util               # type: ignore
from kernel_tuner.interface import Options  # type: ignore


# generates noise to be added to the function evaluations
def noise() -> float:
    return 0.0
    # return randuni(-1, 1)


# Rosenbrock's function (constrained to a disk)
def rosenbrock_constrained(x, y) -> float:
    if (x**2 + y**2) > 2:
        return 1e20
    return ((1 - x)**2 + 100 * (y - x**2)**2) + noise()


# Mishra's bird function
def mishras_bird(x, y) -> float:
    return (sin(x) * np.exp((1 - cos(y))**2) + cos(y) * np.exp((1 - sin(x))**2) + (x - y)**2) + noise()


# Mishra's bird function (constrained), minimum set to 0.0
def mishras_bird_constrained(x, y) -> float:
    if (x + 5)**2 + (y + 5)**2 >= 25:
        return 1e20
    return (sin(x) * np.exp((1 - cos(y))**2) + cos(y) * np.exp((1 - sin(x))**2) + (x - y)**2) + noise() + 106.7645367


# Gomez and Levy function, minimum set to 0.0
def gomez_levy(x, y) -> float:
    if -np.sin(4 * np.pi * x) + 2 * np.sin(2 * np.pi * y)**2 > 1.5:
        return 1e20
    return (4 * x**2 - 2.1 * x**4 + (1 / 3) * x**6 + x * y - 4 * y**2 + 4 * y**4) + noise() + 1.031628453


# helper function for a linear array with integers
def param_space(start, end, num) -> list:
    lin = np.linspace(start, end, num=num)
    ints = np.arange(int(ceil(start)), int(end) + 1, dtype=int)
    combination = np.unique(np.concatenate((lin, ints), 0))
    return combination.tolist()


# evaluates the function with a parameter configuration, forwards arguments as unpacked kwargs
def eval_func(param_config):
    return function(**param_config)


# helper function to unroll the parameters into a searchspace
def cartesian_product(params_dict: dict) -> list:
    return list((dict(zip(params_dict.keys(), x)) for x in itertools.product(*params_dict.values())))


# helper function for dumping numpy objects to JSON, from store_cache() in util.py
def npconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj.__str__()


# helper function to create the unique keystring from a parameter config
def keystring(param_config: dict):
    # TODO change float(i) to int(i) if needed
    values = list(float(i) if i.is_integer() else i for i in param_config.values())
    return ",".join([str(i) for i in values])


# set the function to evaluate
function = mishras_bird_constrained
# set the number of times observations must be repeated (will be aggregated to the mean)
repeat_evals = 1

# set the parameters to explore
#    # Rosenbrock
# params_to_eval = {
#     'x': param_space(-1.5, 1.5, num=100),
#     'y': param_space(-1.5, 1.5, num=100),
# }
# Mishra's bird
params_to_eval = {
    'x': param_space(-10, 0, num=100),
    'y': param_space(-6.5, 0, num=100),
}
# # Gomez and Levy
# params_to_eval = {
#     'x': param_space(-1, 0.75, num=100),
#     'y': param_space(-1, 1, num=100),
# }

# set the required dummy variables
searchspace = cartesian_product(params_to_eval)
# name = 'gomez_levy'
name = 'mishras_bird_constrained'
devname = 'generator'
kernelname = name + '_kernel'
cache = name + '_' + devname
kernel_options = Options(kernel_name=kernelname)
tuning_options = Options(cache=cache, tune_params=Options(params_to_eval), simulation_mode=False)
runner = Options(dev=Options(name=devname), simulation_mode=False)
prune_searchspace = False
prune_searchspace_with_numpy = False
multi_threaded = True


# loop over the function to evaluate and store in cache
def eval_and_store(param_config: dict) -> Tuple[str, dict]:
    param_config_keystring = keystring(param_config)
    # evaluate the function
    if repeat_evals > 1:
        values = np.array([])
        for _ in range(repeat_evals):
            values = np.append(values, eval_func(param_config))
        value = np.mean(values)
    else:
        value = eval_func(param_config)
        values = np.array([value])
    # store the result in the param config
    param_config['times'] = values
    param_config['time'] = value
    return param_config_keystring, param_config


if __name__ == "__main__":
    # initialize the cache file
    time_start = time.perf_counter()
    print("Initializing cachefile", end='')
    if cache[-5:] != ".json":
        cache += ".json"
    util.process_cache(cache, kernel_options, tuning_options, runner)
    print(" ({} seconds)".format(round(time.perf_counter() - time_start, 3)))
    assert tuning_options.cache is not None, "Cache  has not been properly initialized"

    # remove already evaluated from the searchspace
    if prune_searchspace:
        time_start_prune = time.perf_counter()
        print("Skipping parameter configurations already in the cache")
        if prune_searchspace_with_numpy:
            searchspace_keystrings = list(keystring(x) for x in searchspace)
            cache_keystrings = list(tuning_options.cache.keys())
            searchspace = np.setdiff1d(searchspace_keystrings, cache_keystrings)
            print(" ({} seconds)".format(round(time.perf_counter() - time_start_prune, 3)))
        else:
            searchspace_pruned = list()
            for param_config in progressbar.progressbar(searchspace, redirect_stdout=True):
                param_config_keystring = keystring(param_config)
                if not param_config_keystring in tuning_options.cache:
                    searchspace_pruned.append(param_config)
            searchspace = searchspace_pruned

    # stop if no searchspace left
    searchspace_len = len(searchspace)
    if searchspace_len < 1:
        util.close_cache(cache)
        print("No evaluations to add to cache")
        exit(0)

    # evaluation of the searchspace
    time_start_eval = time.perf_counter()
    if multi_threaded:
        print("Multi-threaded evaluation of {} parameter configurations".format(searchspace_len), end='', flush=True)
        pool = multiprocessing.pool.ThreadPool()
        evals = pool.map(eval_and_store, searchspace)
        pool.close()
        pool.join()
        print(" ({} seconds)".format(round(time.perf_counter() - time_start_eval, 3)))
    else:
        print("Evaluation of {} parameter configurations".format(searchspace_len))
        evals = list()
        for param_config in progressbar.progressbar(searchspace, redirect_stdout=True):
            evals.append(eval_and_store(param_config))

    # combination into JSON string and dumping to cache
    print("Writing to cache")
    json_string = ""
    for param_config_keystring, param_config in progressbar.progressbar(evals, redirect_stdout=True):
        tuning_options.cache[param_config_keystring] = param_config
        json_string += "\n" + json.dumps({ param_config_keystring: param_config }, default=npconverter)[1:-1] + ","

    util.dump_cache(json_string, tuning_options)

    # close the cache file
    util.close_cache(cache)
    time_end = time.perf_counter()
    print("Generated cachefile '{}' with {} parameter configurations in {} seconds".format(str(cache), searchspace_len, round(time_end - time_start, 3)))
