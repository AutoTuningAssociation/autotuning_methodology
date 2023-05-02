""" Code regarding storage and retrieval of caches """

from __future__ import annotations  # for referring to class within own method
import numpy as np
from typing import Dict
from pathlib import Path


class Results:
    """Object containing the results for an optimization algorithm on a search space"""

    def __init__(self, numpy_arrays: list[np.ndarray]) -> None:
        self.fevals_results = numpy_arrays[0]
        self.objective_time_results = numpy_arrays[1]
        self.objective_performance_results = numpy_arrays[2]
        self.objective_performance_best_results = numpy_arrays[3]
        self.objective_performance_stds = numpy_arrays[4]
        self.objective_time_results_per_key = numpy_arrays[5]
        self.objective_performance_results_per_key = numpy_arrays[6]


class ResultsDescription:
    """Object to store a description of the results and retrieve results for an optimization algorithm on a search space"""

    def __init__(
        self,
        folder_id: str,
        kernel_name: str,
        device_name: str,
        strategy_name: str,
        strategy_display_name: str,
        stochastic: bool,
        objective_time_keys: list[str],
        objective_performance_keys: list[str],
        minimization: bool,
    ) -> None:
        # all attributes must be hashable for symetric difference checking
        self._version = "1.3.0"
        self.__stored = False
        self.__folder_id = folder_id
        self.kernel_name = kernel_name
        self.device_name = device_name
        self.strategy_name = strategy_name
        self.strategy_display_name = strategy_display_name
        self.stochastic = stochastic
        self.objective_time_keys = objective_time_keys
        self.objective_performance_keys = objective_performance_keys
        self.minimization = minimization
        self.numpy_arrays_keys = [
            "fevals_results",
            "objective_time_results",
            "objective_performance_results",
            "objective_performance_best_results",
            "objective_performance_stds",
            "objective_time_results_per_key",
            "objective_performance_results_per_key",
        ]  # the order must not be changed here! see 'numpy_arrays' in runner.py

    def is_same_as(self, other: ResultsDescription) -> bool:
        """Check for equality against another ResultsDescription object"""
        # check if same type
        if not isinstance(other, (ResultsDescription)):
            # additional check for legacy structure
            # if not str(type(other)) == "<class 'caching.ResultsDescription'>":
            raise NotImplementedError(f"Can not compare to object of type {type(other)}")

        # check if same version
        if not hasattr(other, "_version"):
            raise ValueError("ResultsDescription compared against has no version number")
        if self._version != other._version:
            raise ValueError(f"Incompatible versions: {self._version} (own), {other._version} (other)")

        # check if same keys
        symetric_difference_keys = self.__dict__.keys() ^ other.__dict__.keys()
        if len(symetric_difference_keys) != 0:
            raise KeyError(f"Difference in keys: {symetric_difference_keys}")

        # check if same value for each key
        for attribute_key, attribute_value in self.__dict__.items():
            if attribute_key == "strategy_display_name":
                continue
            else:
                assert (
                    attribute_value == other.__dict__[attribute_key]
                ), f"{attribute_key} has different values: {attribute_value}, other: {other.__dict__[attribute_key]}"

        return True

    def __get_cache_filename(self) -> str:
        return f"{self.device_name}_{self.strategy_name}.npz"

    def __get_cache_filepath(self) -> Path:
        """Get the filepath to this experiment"""
        return Path("cached_data_used/visualizations") / self.__folder_id / self.kernel_name

    def __get_cache_full_filepath(self) -> Path:
        """Get the filepath for this file, including the filename and extension"""
        return self.__get_cache_filepath() / self.__get_cache_filename()

    def __check_for_file(self) -> bool:
        """Check whether the file exists"""
        full_filepath = self.__get_cache_full_filepath()
        self.__stored = full_filepath.exists() and np.DataSource().exists(full_filepath)
        return self.__stored

    def __write_to_file(self, arrays: Dict):
        """Write the resultsdescription and the accompanying numpy arrays to file"""
        if self.__stored is True:
            raise ValueError(f"Do not overwrite a ResultsDescription")
        filepath = self.__get_cache_filepath()
        if not filepath.exists():
            filepath.mkdir(parents=True, exist_ok=False)
        self.__stored = True
        np.savez_compressed(self.__get_cache_full_filepath(), resultsdescription=self, **arrays)

    def set_results(self, arrays: Dict):
        """Set and cache the results"""
        return self.__write_to_file(arrays)

    def __read_from_file(self) -> list[np.ndarray]:
        """Read and verify the accompanying numpy arrays from file"""
        self.__check_for_file()
        full_filepath = self.__get_cache_full_filepath()
        if self.__stored is False:
            raise ValueError(f"File {full_filepath} does not exist")

        # load the data and verify the resultsdescription object is the same
        data = np.load(full_filepath, allow_pickle=True)
        data_results_description = data["resultsdescription"].item()
        assert self.is_same_as(data_results_description)

        # get the numpy arrays
        numpy_arrays = list()
        for numpy_array_key in self.numpy_arrays_keys:
            numpy_arrays.append(data[numpy_array_key])
        return numpy_arrays

    def get_results(self) -> Results:
        """Get the Results object"""
        args = self.__read_from_file()
        return Results(args)

    def has_results(self) -> bool:
        """Checks whether there are results or the file exists"""
        return self.__stored or self.__check_for_file()
