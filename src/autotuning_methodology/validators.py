"""Module containing various checks for validity."""

import numpy as np

error_types_strings = ["", "InvalidConfig", "CompilationFailedConfig", "RuntimeFailedConfig"]
kernel_tuner_error_value = 1e20


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
