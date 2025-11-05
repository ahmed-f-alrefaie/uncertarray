import numpy as np

from uncertarray.core import uncertarray


def test_creation():
    x = np.array([1.0, 2.0, 3.0])
    u = np.array([0.1, 0.2, 0.3])
    ua = uncertarray(x, u)
    assert np.allclose(ua.data, x)
    assert np.allclose(ua.uncertainty, u)


def test_creation_from_python_list():
    x = [1.0, 2.0, 3.0]
    u = [0.1, 0.2, 0.3]
    ua = uncertarray(x, u)
    assert np.allclose(ua.data, np.array(x))
    assert np.allclose(ua.uncertainty, np.array(u))


def test_scalar_creation():
    x = 5.0
    u = 0.5
    ua = uncertarray(x, u)
    assert ua.data == x
    assert ua.uncertainty == u


def test_addition():
    x1 = uncertarray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
    x2 = uncertarray(np.array([3.0, 4.0]), np.array([0.3, 0.4]))
    result = x1 + x2
    expected_data = np.array([4.0, 6.0])
    expected_uncertainty = np.sqrt(np.array([0.1**2 + 0.3**2, 0.2**2 + 0.4**2]))
    assert np.allclose(result.data, expected_data)
    assert np.allclose(result.uncertainty, expected_uncertainty)

    result = x1 - x2
    expected_data = np.array([-2.0, -2.0])
    expected_uncertainty = np.sqrt(np.array([0.1**2 + 0.3**2, 0.2**2 + 0.4**2]))
    assert np.allclose(result.data, expected_data)
    assert np.allclose(result.uncertainty, expected_uncertainty)
