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


def test_operations():
    x1 = uncertarray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
    x2 = uncertarray(np.array([3.0, 4.0]), np.array([0.3, 0.4]))
    result = x1 + x2
    result = x1 * x2
    result = x1 / x2

    result = x1 - x2
    result = x1**x2
    assert result.shape == x1.shape


def test_works_as_scalar():
    x = uncertarray(10.0, 0.5)
    result = x + 5.0
    expected_data = 15.0
    expected_uncertainty = 0.5
    assert result.data == expected_data
    assert result.uncertainty == expected_uncertainty


def test_uscalar_op_array():
    x = uncertarray(10.0, 0.5)
    y = np.array([1.0, 2.0, 3.0])
    result = x + y
    expected_data = np.array([11.0, 12.0, 13.0])
    expected_uncertainty = np.array([0.5, 0.5, 0.5])
    assert result.shape == y.shape
    assert np.allclose(result.data, expected_data)
    assert np.allclose(result.uncertainty, expected_uncertainty)

    result = y + x
    assert result.shape == y.shape
    assert np.allclose(result.data, expected_data)
    assert np.allclose(result.uncertainty, expected_uncertainty)


def test_uarray_op_uscalar():
    x = uncertarray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
    y = uncertarray(1.0, 0.5)
    result = x * y
    assert result.shape == x.shape

    result = y * x
    assert result.shape == x.shape
