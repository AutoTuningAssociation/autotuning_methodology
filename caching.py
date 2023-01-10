import numpy as np
from typing import Dict


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
                 objective_values_key: str) -> None:
        self.__stored = False
        self.kernel_name = kernel_name
        self.device_name = device_name
        self.strategy_name = strategy_name
        self.objective_time_keys = objective_time_keys
        self.objective_value_key = objective_value_key
        self.objective_values_key = objective_values_key
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

    def __read_from_file(self) -> list[np.ndarray]:
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
