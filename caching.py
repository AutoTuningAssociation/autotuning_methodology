import os
import json
import numpy as np
from typing import Union, Optional, Dict, Any

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

    def __init__(self, kernel_name: str, device_name: str, strategies: dict):
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

    def has_matching_strategy(self, strategy_name: str, options: dict, repeats: int) -> bool:
        """ Checks whether the cache contains the strategy with matching parameters 'name', 'options' and 'repeats' """
        if self.has_strategy(strategy_name):
            strategy = self.obj['strategies'][strategy_name]
            if (strategy['name'] == strategy_name and strategy['options'] == options and strategy['repeats'] == repeats):
                return True
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

    def get_strategy(self, strategy_name: str, options: dict, repeats: int) -> Optional[dict]:
        """ Returns a strategy by matching the parameters, if it exists """
        if self.has_matching_strategy(strategy_name, options, repeats):
            return self.obj['strategies'][strategy_name]
        return None

    def get_strategy_results(self, strategy_name: str, options: dict, repeats: int, expected_results: dict = None) -> Optional[dict]:
        """ Checks whether the cache contains the expected results for the strategy and returns it if true """
        cached_data = self.get_strategy(strategy_name, options, repeats)
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
