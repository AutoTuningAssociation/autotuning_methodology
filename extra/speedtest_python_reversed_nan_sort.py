import numpy as np
from time import perf_counter


def performance(name, function, array):
    start = perf_counter()
    array = function(array)
    end = perf_counter()
    print(f"{name}: {round(end - start, 5)} seconds")
    num_nonzero = np.count_nonzero(np.isnan(array))
    assert np.all(np.isnan(array[-num_nonzero:]))  # make sure all NaNs are at the end
    assert all(a >= b or np.isnan(a) or np.isnan(b) for a, b in zip(array, array[1:]))  # make sure everything is sorted


def sort_append_reverse(x):
    sorted = np.sort(x)[::-1]
    isnan = np.isnan(sorted)
    return np.concatenate((sorted[~isnan], [np.nan] * np.count_nonzero(isnan)))


def sort_append_flip(x):
    sorted = np.flip(np.sort(x))
    isnan = np.isnan(sorted)
    return np.concatenate((sorted[~isnan], [np.nan] * np.count_nonzero(isnan)))


def sort_negative(x):
    return -np.sort(-x)


arr = np.random.randint(1, 100000, size=10000)
arr = np.where(arr % 7 == 0, np.nan, arr)
print(f"{np.count_nonzero(np.isnan(arr))} / {arr.shape[0]}")

performance("append reverse", sort_append_reverse, arr)
performance("append flip", sort_append_flip, arr)
performance("negative", sort_negative, arr)
