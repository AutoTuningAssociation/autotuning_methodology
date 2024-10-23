"""Module containing various checks for validity."""

from importlib.resources import files
from json import load

import numpy as np
from jsonschema import validate

error_types_strings = ["", "InvalidConfig", "CompilationFailedConfig", "RuntimeFailedConfig"]
kernel_tuner_error_value = 1e20


def get_experiment_schema_filepath():
    """Obtains and checks the filepath to the JSON schema.

    Returns:
        the filepath to the schema in Traversable format.
    """
    schemafile = files("autotuning_methodology").joinpath("schema.json")
    assert schemafile.is_file(), f"Path to schema.json does not exist, attempted path: {schemafile}"
    return schemafile


def validate_experimentsfile(instance: dict, encoding="utf-8") -> dict:
    """Validates the passed instance against the T4 schema. Returns schema or throws ValidationError."""
    schemafile_path = get_experiment_schema_filepath()
    with schemafile_path.open("r", encoding=encoding) as fp:
        schema = load(fp)
        validate(instance=instance, schema=schema)
        return schema


def is_invalid_objective_performance(objective_performance: float) -> bool:
    """Returns whether an objective value is invalid by checking against NaN and the error value.

    Args:
        objective_performance: the objective performance value to check.

    Raises:
        TypeError: if the ``objective_performance`` value has an unexpected type.

    Returns:
        True if the ``objective_performance`` value is invalid, False otherwise.
    """
    if any(str(objective_performance) == error_type_string for error_type_string in error_types_strings):
        return True
    if not isinstance(objective_performance, (int, float)):
        raise TypeError(
            f"""Objective value should be of type float,
                but is of type {type(objective_performance)} with value {objective_performance}"""
        )
    return np.isnan(objective_performance) or objective_performance == kernel_tuner_error_value


def is_invalid_objective_time(objective_time: float) -> bool:
    """Returns whether an objective time is invalid.

    Args:
        objective_time: the objective time value to check.

    Returns:
        True if the ``objective_time`` value is invalid, False otherwise.
    """
    return np.isnan(objective_time) or objective_time < 0


def is_valid_config_result(config: dict) -> bool:
    """Returns whether a given configuration is valid.

    Args:
        config: the configuration to check.

    Returns:
         True if the ``config`` is valid, False otherwise.
    """
    return "invalidity" in config and config["invalidity"] == "correct"
