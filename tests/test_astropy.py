from astropy import units as u

from uncertarray import uncertarray


def test_array_creation():
    xd = [1.0, 2.0, 3.0]
    un = [0.1, 0.2, 0.3]
    ua = uncertarray(xd, un) << u.K

    assert ua.unit == u.K


def test_scalar_creation():
    xd = 5.0
    un = 0.5
    ua = uncertarray(xd, un) << u.m

    assert ua.unit == u.m
    assert ua.data.unit == u.m
    assert ua.uncertainty.unit == u.m


def test_operations():
    x1 = uncertarray([1.0, 2.0], [0.1, 0.2]) << u.s
    x2 = uncertarray([3.0, 4.0], [0.3, 0.4]) << u.s
    result = x1 + x2
    result = x1 * x2
    result = x1 / x2

    result = x1 - x2

    assert result.shape == x1.shape


def test_unary_ops():
    import numpy as np

    x = uncertarray([1.0, 4.0, 9.0], [0.1, 0.2, 0.3]) << u.m
    result = np.sqrt(x)
    assert result.unit == u.m**0.5
    assert result.shape == x.shape
