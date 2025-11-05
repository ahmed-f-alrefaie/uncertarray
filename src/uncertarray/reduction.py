import numpy as np

from .core import implements, uncertarray


@implements(np.mean)
def umean(x: uncertarray, *args, **kwargs):
    """Compute mean of array."""
    axis = kwargs.pop("axis", None)
    size = x.size
    if axis is not None:
        size = x.shape[axis]

    return np.sum(x, *args, **kwargs) / size


@implements(np.sum)
def usum(x: uncertarray, *args, **kwargs):
    """Compute sum of array."""
    return uncertarray(
        np.sum(x.data, *args, **kwargs),
        np.sqrt(np.sum(np.square(x.uncertainty), *args, **kwargs)),
    )


@implements(np.prod)
def uprod(x: uncertarray, *args, **kwargs):
    """Compute product of array."""
    data_result = np.prod(x.data, *args, **kwargs)
    relative_uncert = x.uncertainty / x.data

    sum_squared = np.sum(np.square(relative_uncert), *args, **kwargs)

    uncertainty_result = np.sqrt(np.square(data_result) * sum_squared)

    return uncertarray(
        data_result,
        uncertainty_result,
    )


@implements(np.var)
def uvar(x: uncertarray, *args, **kwargs):
    """Compute variance of array."""

    mean_x = umean(x, *args, *args, **kwargs)
    mean_x_squared = mean_x * mean_x

    mean_x2 = umean(x * x, *args, **kwargs)

    return mean_x2 - mean_x_squared


@implements(np.std)
def std(x: uncertarray, *args, **kwargs):
    """Compute standard deviation of array."""
    return np.sqrt(uvar(x, *args, **kwargs))


@implements(np.min)
def umin(x: uncertarray, *args, **kwargs):
    """Compute minimum of array."""
    return uncertarray(
        np.min(x.data, *args, **kwargs),
        x.uncertainty[np.unravel_index(np.argmin(x.data, *args, **kwargs), x.data.shape)],
    )


@implements(np.max)
def umax(x: uncertarray, *args, **kwargs):
    """Compute maximum of array."""
    return uncertarray(
        np.max(x.data, *args, **kwargs),
        x.uncertainty[np.unravel_index(np.argmax(x.data, *args, **kwargs), x.data.shape)],
    )


# @implements(np.trapezoid)
# def utrapz(y: uncertarray, *args, **kwargs):
#     """Compute trapezoidal integral of array."""
#     data = np.trapezoid(y.data, *args, **kwargs)
