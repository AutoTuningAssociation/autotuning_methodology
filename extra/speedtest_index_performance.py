from random import randrange, choice
from time import perf_counter
import string
import numpy as np

alphanumeric = list(string.ascii_lowercase + string.ascii_uppercase + string.digits)

# create and fill the parameter space
size = 100000
start_creation = perf_counter()
param_space = list()
param_space_index = list(range(size))
for i in range(size):
    param_config = tuple([i % 9, choice(alphanumeric), i, i / 100])
    param_space.append(param_config)
num_params = len(param_space[0])

# convert to tuple and numpy array
param_space_dict = dict(zip(param_space, param_space_index))
param_space_tuple = tuple(param_space)
param_space_numpy = np.array(param_space)
size_unique = np.unique(param_space_numpy, axis=-1).size
if size_unique != param_space_numpy.size:
    raise ValueError(f"{param_space_numpy.size - size_unique} duplicate parameter configurations in the searchspace")
print(f"creation: {perf_counter() - start_creation} seconds")

# now test random value lookups for list, tuple and numpy array
start_index_lookup_list = perf_counter()
for _ in range(int(size / 10)):
    random_index = randrange(0, size)
    random_param_config = param_space[random_index]
    index = param_space.index(random_param_config)
    assert index == random_index
print(f"index lookup list: {perf_counter() - start_index_lookup_list} seconds")

start_index_lookup_dict = perf_counter()
for _ in range(int(size / 10)):
    random_index = randrange(0, size)
    random_param_config = param_space[random_index]
    index = param_space_dict.get(random_param_config)
    assert index == random_index
print(f"index lookup dict: {perf_counter() - start_index_lookup_dict} seconds")

start_index_lookup_tuple = perf_counter()
for _ in range(int(size / 10)):
    random_index = randrange(0, size)
    random_param_config = param_space[random_index]
    index = param_space_tuple.index(random_param_config)
    assert index == random_index
print(f"index lookup tuple: {perf_counter() - start_index_lookup_tuple} seconds")

start_index_lookup_numpy = perf_counter()
for _ in range(int(size / 10)):
    random_index = randrange(0, size)
    random_param_config = param_space[random_index]
    num_matching_params = np.count_nonzero(param_space_numpy == random_param_config, -1)
    index = (num_matching_params == num_params).nonzero()[0][0]
    assert index == random_index
print(f"index lookup numpy: {perf_counter() - start_index_lookup_numpy} seconds")

# test the performance of Numpy indices lookup vs map(get) indices lookup
num_random_indices = 10
start_multi_index_lookup_map = perf_counter()
for _ in range(int(size / 10)):
    random_indices = list(randrange(0, size) for _ in range(num_random_indices))
    result = list(map(param_space_tuple.__getitem__, random_indices))
    assert len(result) == len(random_indices)
print(f"multi-index lookup map: {perf_counter() - start_multi_index_lookup_map} seconds")

start_multi_index_lookup_numpy = perf_counter()
for _ in range(int(size / 10)):
    random_indices = list(randrange(0, size) for _ in range(num_random_indices))
    result = list(param_space_numpy[random_indices])
    assert len(result) == len(random_indices)
print(f"multi-index lookup numpy: {perf_counter() - start_multi_index_lookup_numpy} seconds")
