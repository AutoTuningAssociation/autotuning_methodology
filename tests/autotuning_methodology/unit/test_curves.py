"""Unit tests for testing the curves."""

import numpy as np
import pytest

from autotuning_methodology.curves import Curve, get_indices_in_array, get_indices_in_distribution


def test_get_indices_in_distribution():
    """Each draw should have the same value as the associated value in the distribution."""
    draws = np.array([[4, np.NaN, 5], [1, 2, 4.5]])
    dist = np.array([1, 2, 4, 4, 4.5, 5])
    expected_indices = np.array([[2, np.NaN, 5], [0, 1, 4]])

    indices_found = get_indices_in_distribution(draws=draws, dist=dist)

    # each draw should have the same value as the associated value in the distribution
    for index, draw in np.ndenumerate(draws):
        if np.isnan(draw):
            assert np.isnan(expected_indices[index])
            assert np.isnan(indices_found[index])
            continue
        try:
            assert indices_found[index] == expected_indices[index]
            assert (
                draw == dist[int(indices_found[index])]
            ), f"Is {draw}, but distribution value at index is {dist[int(indices_found[index])]}"
        except ValueError:
            raise ValueError(f"{draw=}, but {indices_found[index]=}")


def test_get_indices_in_distribution_check_dist():
    """Dist order should be checked by default and dist should not contain NaN."""
    draws = np.array([[4, np.NaN, 5], [1, 2, 4.5]])
    with pytest.raises(AssertionError, match="2 violations in 5 values"):
        get_indices_in_distribution(draws=draws, dist=np.array([1, 2, np.NaN, 4, 4.5]))
    with pytest.raises(AssertionError, match="1 violations in 4 values"):
        get_indices_in_distribution(draws=draws, dist=np.array([5, 4, 6, 7]))


def test_get_indices_in_distribution_check_draws():
    """Values in draw (with the exception of NaN) that are not in dist should throw an exception."""
    draws = np.array([[4, np.NaN, 3], [1, 2, 4.5]])
    dist = np.array([1, 2, 4, 4, 4.5, 5])
    with pytest.raises(AssertionError, match="Each value in draws should be in dist"):
        get_indices_in_distribution(draws=draws, dist=dist)


def test_get_indices_in_array():
    """Each value should have the same value as the associated value in the unsorted array."""
    draws = np.array([[4, np.NaN, 5], [1, 2, 4.5]])
    dist = np.array([4, 2, 1, 4, 5, 4.5])
    expected_indices = np.array([[0, np.NaN, 4], [2, 1, 5]])

    indices_found = get_indices_in_array(values=draws, array=dist)

    # each draw should have the same value as the associated value in the distribution
    for index, draw in np.ndenumerate(draws):
        if np.isnan(draw):
            assert np.isnan(expected_indices[index])
            assert np.isnan(indices_found[index])
            continue
        try:
            assert indices_found[index] == expected_indices[index]
            assert (
                draw == dist[int(indices_found[index])]
            ), f"Is {draw}, but distribution value at index is {dist[int(indices_found[index])]}"
        except ValueError:
            raise ValueError(f"{draw=}, but {indices_found[index]=}")


def test_fevals_find_pad_width():
    """The input array should be padded to result in the same shape as target_array."""
    array = np.array([1, 2])
    target_array = np.array([0, 1, 2, 3, 4])
    expected_padding = tuple([1, 2])
    result_padding = Curve.fevals_find_pad_width(None, array, target_array)
    assert array.shape != target_array.shape
    assert expected_padding == result_padding
