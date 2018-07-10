"""Test the `met` module."""

from hugs.calc import get_wind_dir, get_wind_speed, get_wind_components

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

import pytest


def test_speed():
    """Test calculating wind speed."""
    u = np.array([4., 2., 0., 0.])
    v = np.array([0., 2., 4., 0.])

    speed = get_wind_speed(u, v)

    s2 = np.sqrt(2.)
    true_speed = np.array([4., 2 * s2, 4., 0.])

    assert_array_almost_equal(true_speed, speed, 4)


def test_scalar_speed():
    """Test wind speed with scalars."""
    s = get_wind_speed(-3., -4.)
    assert_almost_equal(s, 5., 3)


def test_dir():
    """Test calculating wind direction."""
    u = np.array([4., 2., 0., 0.])
    v = np.array([0., 2., 4., 0.])

    direc = get_wind_dir(u, v)

    true_dir = np.array([270., 225., 180., 270.])

    assert_array_almost_equal(true_dir, direc, 4)


def test_scalar_wind_components():
    """Tests get_wind_components with scalar inputs."""
    speed = 20.
    wdir = 25.
    u, v = get_wind_components(speed, wdir)
    assert_almost_equal(u, -8.45236, 4)
    assert_almost_equal(v, -18.12615, 4)


def test_array_wind_components():
    """Tests get_wind_components with array inputs."""
    speed = np.array([10, 17, 40, 0])
    wdir = np.array([3, 92, 210, 297])
    true_u = np.array([-0.523359, -16.98964, 20.0, 0.0])
    true_v = np.array([-9.986295, 0.593291, 34.641016, 0.0])
    u, v = get_wind_components(speed, wdir)
    assert_array_almost_equal(u, true_u, 4)
    assert_array_almost_equal(v, true_v, 4)


def test_warning_direction():
    """Tests the warning is raised when wind direction > 360."""
    # Raises UserWarning, since that is the default of warnings.warn
    # For exceptions: pytest.raise(<Exception Type>)
    with pytest.warns(UserWarning):
        get_wind_components(3, 480)
