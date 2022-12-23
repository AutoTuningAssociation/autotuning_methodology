import os
import json
import numpy as np
from typing import Union, Optional, Dict, Any


class Results():
    """ Object containing the results for an optimization algorithm on a search space """

    def __init__(self, numpy_arrays: list) -> None:
        self.fevals_results = numpy_arrays[0]
        self.time_results = numpy_arrays[1]
        self.objective_time_results = numpy_arrays[2]
        self.objective_value_results = numpy_arrays[3]
        self.objective_value_best_results = numpy_arrays[4]
        self.objective_value_stds = numpy_arrays[5]


class ResultsDescription():
    """ Object to store a description of the results and retrieve results for an optimization algorithm on a search space """

    def __init__(self, kernel_name: str, device_name: str, strategy_name: str, objective_time_keys: str, objective_value_key: str,
                 objective_value_keys: str) -> None:
        self.__stored = False
        self.kernel_name = kernel_name
        self.device_name = device_name
        self.strategy_name = strategy_name
        self.objective_time_keys = objective_time_keys
        self.objective_value_key = objective_value_key
        self.objective_value_keys = objective_value_keys
        self.numpy_arrays_keys = [
            'fevals_results', 'time_results', 'objective_time_results', 'objective_value_results', 'objective_value_best_results', 'objective_value_stds'
        ]    # the order must not be changed here!

    def __get_cache_filename(self) -> str:
        return f"{self.kernel_name}_{self.device_name}_{self.strategy_name}"

    def __get_cache_filepath(self) -> str:
        return f"{self.__get_cache_filename()}"

    def __check_for_file(self) -> bool:
        """ Check whether the file exists """
        self.__stored = np.DataSource().exists(self.__get_cache_filepath())
        return self.__stored

    def __write_to_file(self, arrays: Dict):
        """ Write the resultsdescription and the accompanying numpy arrays to file """
        if self.__stored is True:
            raise ValueError(f"Do not overwrite a ResultsDescription")
        self.__stored = True
        np.savez(self.__get_cache_filepath(), resultsdescription=self, **arrays)

    def set_results(self, arrays: Dict):
        """ Set and cache the results """
        return self.__write_to_file(arrays)

    def __read_from_file(self) -> list(np.ndarray):
        """ Read and verify the accompanying numpy arrays from file """
        self.__check_for_file()
        filepath = self.__get_cache_filepath()
        if self.__stored is False:
            raise ValueError(f"File {filepath} does not exist")

        # load the data and verify the resultsdescription object is the same
        data = np.load(filepath, allow_pickle=True)
        assert data['resultsdescription'] == self

        # get the numpy arrays
        numpy_arrays = list()
        for numpy_array in self.numpy_arrays_keys:
            numpy_arrays.append(numpy_array)
        return numpy_arrays

    def get_results(self) -> Results:
        """ Get the Results object """
        return Results(self.__read_from_file())

    def has_results(self) -> bool:
        """ Checks whether there are results or the file exists """
        return self.__stored or self.__check_for_file()


class NumpyEncoder(json.JSONEncoder):
    """ JSON encoder for NumPy types, from https://www.programmersought.com/article/18271066028/ """

    def default(self, obj):    # pylint: disable=arguments-differ
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class CachedObject():
    """ Class for managing cached results """

    def __init__(self, kernel_name: str, device_name: str, baseline_time: np.ndarray, baseline_result: np.ndarray, strategies: dict):
        try:
            cache = CacheInterface.read(kernel_name, device_name)
            # print("Cache with type: ", type(cache), ":\n ", cache)
            self.kernel_name = cache['kernel_name']
            self.device_name = cache['device_name']
            self.obj = cache
        except (FileNotFoundError, json.decoder.JSONDecodeError) as _:
            print("No cached visualization found, creating new cache")
            # make the strategies a dict with the name as key for faster lookup
            strategies_dict = dict()
            for strategy in strategies:
                strategies_dict[strategy['name']] = strategy

            self.kernel_name = kernel_name
            self.device_name = device_name
            self.obj = {
                "kernel_name": kernel_name,
                "device_name": device_name,
                "baseline_time": baseline_time,
                "baseline_result": baseline_result,
                "strategies": strategies_dict
            }

    def read(self):
        return CacheInterface.read(self.kernel_name, self.device_name)

    def write(self):
        return CacheInterface.write(self.obj)

    def delete(self):
        return CacheInterface.delete(self.kernel_name, self.device_name)

    def has_strategy(self, strategy_name: str) -> bool:
        """ Checks whether the cache contains the strategy with matching parameter 'name' """
        return strategy_name in self.obj["strategies"].keys()

    def has_matching_strategy(self, strategy_name: str, repeats: int) -> bool:
        """ Checks whether the cache contains the strategy with matching parameters 'name', 'options' and 'repeats' """
        if self.has_strategy(strategy_name):
            strategy = self.obj['strategies'][strategy_name]
            return (strategy['name'] == strategy_name and strategy['repeats'] == repeats)
        return False

    def recursively_compare_dict_keys(self, dict_elem, compare_elem) -> bool:
        """ Recursively go trough a dict to check whether the keys match, returns true if they match """
        if compare_elem is None:
            return True
        if isinstance(dict_elem, list):
            for idx in range(min(len(dict_elem), len(compare_elem))):
                if self.recursively_compare_dict_keys(dict_elem[idx], compare_elem[idx]) is False:
                    return False
        elif isinstance(dict_elem, dict):
            if not isinstance(compare_elem, dict):
                return False
            return dict_elem.keys() == compare_elem.keys() and all(self.recursively_compare_dict_keys(dict_elem[key], compare_elem[key]) for key in dict_elem)
        return True

    def get_baseline(self):
        return np.array(self.obj["baseline_time"]), np.array(self.obj["baseline_result"])

    def get_strategy(self, strategy_name: str, repeats: int) -> Optional[dict]:
        """ Returns a strategy by matching the parameters, if it exists """
        if self.has_matching_strategy(strategy_name, repeats):
            return self.obj['strategies'][strategy_name]
        return None

    def get_strategy_results(self, strategy_name: str, repeats: int, expected_results: dict = None) -> Optional[dict]:
        """ Checks whether the cache contains the expected results for the strategy and returns it if true """
        cached_data = self.get_strategy(strategy_name, repeats)
        if cached_data is not None and 'results' in cached_data and (expected_results is None
                                                                     or self.recursively_compare_dict_keys(cached_data['results'], expected_results)):
            return cached_data
        return None

    def set_strategy(self, strategy: dict[str, Any], results: dict[str, Any]):
        """ Sets a strategy and its results """
        strategy_name = strategy['name']
        # delete old strategy if any
        if self.has_strategy(strategy['name']):
            del self.obj["strategies"][strategy_name]
        # set new strategy
        self.obj["strategies"][strategy_name] = strategy
        # set new values
        self.obj["strategies"][strategy_name]["options"] = strategy["options"]
        self.obj["strategies"][strategy_name]["results"] = results
        self.write()


class CacheInterface:
    """ Interface for cache filesystem interaction """

    @staticmethod
    def file_name(kernel_name: str, device_name: str) -> str:    # pylint: disable=no-self-argument
        """ Combine the variables into the target filename """
        return f"cached_plot_{kernel_name}_{device_name}.json"

    @staticmethod
    def file_path(file_name: str) -> str:
        """ Returns the absolute file path """
        # TODO fix this so it works more flexibly for nested folders
        return os.path.abspath(f"cached_visualizations/{file_name}")

    @staticmethod
    def read(kernel_name: str, device_name: str) -> Dict[str, Any]:    # pylint: disable=no-self-argument
        """ Read and parse a cachefile """
        filename = CacheInterface.file_name(kernel_name, device_name)
        with open(CacheInterface.file_path(filename)) as json_file:
            return json.load(json_file)

    @staticmethod
    def write(cached_object: Dict[str, Any]):    # pylint: disable=no-self-argument
        """ Serialize and write a cachefile """
        filename = CacheInterface.file_name(cached_object['kernel_name'], cached_object['device_name'])    # pylint: disable=unsubscriptable-object
        with open(CacheInterface.file_path(filename), 'w') as json_file:
            json.dump(cached_object, json_file, cls=NumpyEncoder)

    @staticmethod
    def delete(kernel_name: str, device_name: str) -> bool:    # pylint: disable=no-self-argument
        """ Delete a cachefile, returns True for completion and False if file can not be deleted """
        try:
            import os
            filename = CacheInterface.file_name(kernel_name, device_name)
            os.remove(CacheInterface.file_path(filename))
            return True
        except OSError:
            return False
