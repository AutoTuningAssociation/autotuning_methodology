import pytest
import numpy as np
from autotuning_methodology.curves import get_indices_in_distribution


def test_get_indices_in_distribution():
    """ each draw should have the same value as the associated value in the distribution """
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
            assert draw == dist[int(indices_found[index])], f"Is {draw}, but distribution value at index is {dist[int(indices_found[index])]}"
        except ValueError:
            raise ValueError(f"{draw=}, but {indices_found[index]=}")


def test_get_indices_in_distribution_check_dist():
    """ Dist order should be checked by default and dist should not contain NaN """
    draws = np.array([[4, np.NaN, 5], [1, 2, 4.5]])
    with pytest.raises(AssertionError, match="Distribution is not sorted ascendingly, 2 violations in"):
        get_indices_in_distribution(draws=draws, dist=np.array([1, 2, np.NaN, 4, 4.5]))
    with pytest.raises(AssertionError, match="Distribution is not sorted ascendingly, 1 violations in"):
        get_indices_in_distribution(draws=draws, dist=np.array([5, 4, 6, 7]))


def test_get_indices_in_distribution_check_draws():
    """ Values in draw (with the exception of NaN) that are not in dist should throw an exception """
    draws = np.array([[4, np.NaN, 3], [1, 2, 4.5]])
    dist = np.array([1, 2, 4, 4, 4.5, 5])
    with pytest.raises(AssertionError, match="Each value in draws should be in dist"):
        get_indices_in_distribution(draws=draws, dist=dist)