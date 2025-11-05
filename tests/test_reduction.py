import numpy as np

from uncertarray.core import uncertarray


def test_sum():
    x = uncertarray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
    result = np.sum(x)
    expected_data = 6.0
    expected_uncertainty = np.sqrt(0.1**2 + 0.2**2 + 0.3**2)
    assert np.isclose(result.data, expected_data)
    assert np.isclose(result.uncertainty, expected_uncertainty)
