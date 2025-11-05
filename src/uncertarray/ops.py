import numpy as np

from .core import implements, uncertarray


@implements(np.equal)
def equal(x: uncertarray, y, **kwargs):
    """Compare values (ignoring uncertainty for equality)."""
    x_data = x.data if isinstance(x, uncertarray) else x
    y_data = y.data if isinstance(y, uncertarray) else y
    return np.equal(x_data, y_data, **kwargs)


@implements(np.less)
def less(x: uncertarray, y, **kwargs):
    """Compare considering uncertainty overlap."""
    x_data = x.data if isinstance(x, uncertarray) else x
    y_data = y.data if isinstance(y, uncertarray) else y
    x_unc = x.uncertainty if isinstance(x, uncertarray) else 0
    y_unc = y.uncertainty if isinstance(y, uncertarray) else 0

    # Conservative: values don't overlap within uncertainties
    return (x_data + x_unc) < (y_data - y_unc)


@implements(np.cumsum)
def ucumsum(x: uncertarray, *args, **kwargs):
    """Compute cumulative sum of array."""
    data_cumsum = np.cumsum(x.data, *args, **kwargs)
    uncertainty_cumsum = np.sqrt(np.cumsum(np.square(x.uncertainty), *args, **kwargs))
    return uncertarray(data_cumsum, uncertainty_cumsum)
