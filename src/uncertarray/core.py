"""Handle array that contains uncertainty information."""

import typing as t

import numpy as np
from scipy.special import psi

T = t.TypeVar("T", bound=np.ndarray)
Param = t.ParamSpec("Param")
RetType = t.TypeVar("RetType")

UncertOrGeneric = t.Union[T, "uncertarray[T]"]

AnyNumberType = t.Union[int, float, np.integer, np.floating, UncertOrGeneric]

HANDLED_FUNCTIONS = {}

BinaryCallableType = t.Callable[[T, T], AnyNumberType]
UnaryCallableType = t.Callable[[T], AnyNumberType]

# Derivatives of operators w.r.t. operands, first x, then y
operator_derivatives: dict[str, tuple[BinaryCallableType, BinaryCallableType]] = {
    "add": (lambda x, y: 1, lambda x, y: 1.0),
    "sub": (lambda x, y: 1, lambda x, y: -1.0),
    "mul": (lambda x, y: y, lambda x, y: x),
    "truediv": (lambda x, y: 1 / y, lambda x, y: -x / y**2),
    "divide": (lambda x, y: 1 / y, lambda x, y: -x / y**2),
    "pow": (lambda x, y: y * x ** (y - 1), lambda x, y: np.log(x) * x**y),
}

numpy_derivatives: dict[str, tuple[BinaryCallableType, BinaryCallableType]] = {
    "add": (lambda x, y: 1, lambda x, y: 1),
    "subtract": (lambda x, y: 1, lambda x, y: -1),
    "multiply": (lambda x, y: y, lambda x, y: x),
    "divide": (lambda x, y: 1 / y, lambda x, y: -x / y**2),
    "true_divide": (lambda x, y: 1 / y, lambda x, y: -x / y**2),
    "power": (lambda x, y: y * x ** (y - 1), lambda x, y: np.log(x) * x**y),
}

unary_derivatives: dict[str, UnaryCallableType] = {
    "exp": lambda x: np.exp(x),
    "log": lambda x: 1 / x,
    "log10": lambda x: 1 / (x * np.log(10)),
    "sin": lambda x: np.cos(x),
    "cos": lambda x: -np.sin(x),
    "tan": lambda x: 1 / np.cos(x) ** 2,
    "arcsin": lambda x: 1 / np.sqrt(1 - x**2),
    "arccos": lambda x: -1 / np.sqrt(1 - x**2),
    "arctan": lambda x: 1 / (1 + x**2),
    "sinh": lambda x: np.cosh(x),
    "cosh": lambda x: np.sinh(x),
    "tanh": lambda x: 1 / np.cosh(x) ** 2,
    "arcsinh": lambda x: 1 / np.sqrt(1 + x**2),
    "arccosh": lambda x: 1 / np.sqrt(x**2 - 1),
    "arctanh": lambda x: 1 / (1 - x**2),
    "sqrt": lambda x: 0.5 / np.sqrt(x),
    "abs": lambda x: np.sign(x),
    "erf": lambda x: 2 / np.sqrt(np.pi) * np.exp(-(x**2)),
    "erfc": lambda x: -2 / np.sqrt(np.pi) * np.exp(-(x**2)),
    "gamma": lambda x: psi(x),  # Using the digamma function
    "gammaln": lambda x: psi(x),
    "square": lambda x: 2 * x,
    "exp2": lambda x: np.log(2) * 2**x,
    "expm1": lambda x: np.exp(x),
    "log2": lambda x: 1 / (x * np.log(2)),
    "log1p": lambda x: 1 / (1 + x),
    "reciprocal": lambda x: -1 / x**2,
    # For these, in practical applications, you might treat their derivatives as zero.
    "ceil": lambda x: 0,
    "floor": lambda x: 0,
    "trunc": lambda x: 0,
    "negative": lambda x: -1,
    "sign": lambda x: 0,  # Keep in mind the approximation note above.
}


def implements(np_function: t.Any) -> t.Callable[[t.Callable[..., t.Any]], t.Callable[..., t.Any]]:
    """Register an __array_function__ implementation for uncertarray objects.

    np_function may be any numpy function or ufunc (or module object such as
    ``np.fft``). We return a decorator that registers the handler in
    ``HANDLED_FUNCTIONS``.
    """

    doc = getattr(np_function, "__doc__", None)

    def decorator(func: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
        """Decorator that registers `func` for `np_function`."""
        HANDLED_FUNCTIONS[np_function] = func
        return func

    decorator.__doc__ = doc

    return decorator


# @implements(np.equal)
# def equal(x: "uncertarray", *args: t.Any, **kwargs: t.Any) -> np.ndarray:
#     """Determine if array is equal to."""
#     return np.equal(x.data, *args, **kwargs)


@implements(np.fft)
def fft(x: "uncertarray", *args: t.Any, **kwargs: t.Any) -> "uncertarray":
    """Perform FFT with error propagation."""

    data_fft = np.fft.fft(x.data, *args, **kwargs)

    axis = kwargs.get("axis")

    n = x.shape[axis] if axis is not None else x.size

    uncertainty_fft = np.sqrt(n) * x.uncertainty

    return uncertarray(data_fft, uncertainty_fft)


@implements(np.pad)
def pad(x: "uncertarray", *args: t.Any, **kwargs: t.Any) -> "uncertarray":
    """Pad array with error propagation."""

    data_padded = np.pad(x.data, *args, **kwargs)

    uncertainty_padded = np.pad(x.uncertainty, *args, **kwargs)

    return uncertarray(data_padded, uncertainty_padded)


# @implements(np.trapz)
# def trapz(x: "uncertarray", *args, **kwargs):
#     """Compute product of array."""
#     data_result = np.trapz(x.data, *args, **kwargs)
#     relative_uncert = x.uncertainty / x.data

#     sum_squared = np.sum(np.square(relative_uncert), *args, **kwargs)

#     uncertainty_result = np.sqrt(np.square(data_result) * sum_squared)

#     return uncertarray(
#         data_result,
#         uncertainty_result,
#     )


def _determine_data(array: UncertOrGeneric) -> UncertOrGeneric:
    """Determine data of array."""
    if hasattr(array, "uncertainty"):
        return array.data
    else:
        return array


def _apply_binary_op_uncert(
    op_x: BinaryCallableType[T],
    op_y: BinaryCallableType[T],
    x: UncertOrGeneric[T],
    y: UncertOrGeneric[T],
) -> T:
    """Apply error propagation to binary operator."""
    sig_x = x.uncertainty if isinstance(x, uncertarray) else None
    sig_y = y.uncertainty if isinstance(y, uncertarray) else None

    x_data = x.data if isinstance(x, uncertarray) else x
    y_data = y.data if isinstance(y, uncertarray) else y

    deriv_x, deriv_y = op_x(x_data, y_data), op_y(x_data, y_data)

    std_x = np.square(deriv_x * sig_x) if sig_x is not None else 0.0
    std_y = np.square(deriv_y * sig_y) if sig_y is not None else 0.0

    return t.cast(T, np.sqrt(std_x + std_y))


def _apply_unary_op_uncert(op: UnaryCallableType, x: "uncertarray[T]") -> T:
    """Apply error propagation to unary operator."""
    sig_x = x.uncertainty

    x_data = x.data

    return t.cast(T, np.sqrt(np.square(op(x_data)) * np.square(sig_x)))


class _ucert_meta:
    """Meta class for uncertarray when ``asarray`` used."""

    def __init__(self, data: UncertOrGeneric, uncertainty: UncertOrGeneric) -> None:
        """Initialize uncertarray."""
        self.data = data
        self.uncertainty = uncertainty

    def __repr__(self) -> str:
        """Return representation of uncertarray."""
        return f"{self.data}+/-{self.uncertainty} "


class uncertarray(t.Generic[T]):
    """Array that contains uncertainty information."""

    data: T
    uncertainty: T

    def __init__(self, data: T, uncertainty: T) -> None:
        """Initialize uncertarray."""
        self.data = np.asanyarray(data)
        self.uncertainty = np.asanyarray(uncertainty)

        if self.data.shape != self.uncertainty.shape:
            raise ValueError("Data and uncertainty must have the same shape")

        # Ensure uncertainty is non-negative
        if np.any(self.uncertainty < 0):
            raise ValueError("Uncertainty must be non-negative")

    def __getitem__(self, key: t.Union[int, slice, tuple]) -> "uncertarray[T]":
        """Get item from array."""
        data: T = self.data[key]
        uncertainty: T = self.uncertainty[key]
        return uncertarray(data, uncertainty)

    def __setitem__(self, key: t.Union[int, slice, tuple], value: t.Union["uncertarray[T]", T, float, int]) -> None:
        """Set item in array."""
        if isinstance(value, uncertarray):
            self.data[key] = value.data
            self.uncertainty[key] = value.uncertainty
        else:
            # Print runtime warning
            self.data[key] = value
            raise RuntimeWarning(
                "Array has been set with a non uncertarray value.Uncertainty information may not be correct."
            )

    def __array__(self) -> np.ndarray:
        """Return array."""
        return np.array(
            [_ucert_meta(d, u) for d, u in zip(self.data.ravel(), self.uncertainty.ravel(), strict=True)],
            dtype=object,
        ).reshape(self.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape of array."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return self.data.ndim

    # def __array__(self, dtype=None):
    #     return self.data

    def __repr__(self) -> str:
        """Return representation of uncertarray."""
        return f"data: {self.data.__repr__()}\nuncertainty: {self.uncertainty.__repr__()}"

    # def set_unit(self, unit: u.Unit) -> "uncertarray":
    #     """Set unit of array."""
    #     self.data = self.data << unit
    #     self.uncertainty = self.uncertainty << unit
    #     return self

    @property
    def relative_uncertainty(self) -> T:
        """Return relative uncertainty (uncertainty / |value|)."""
        return self.uncertainty / np.abs(self.data)

    @property
    def signal_to_noise(self) -> T:
        """Return signal-to-noise ratio (|value| / uncertainty)."""
        return np.abs(self.data) / self.uncertainty

    def reshape(self, *args: t.Any, **kwargs: t.Any) -> "uncertarray[T]":
        """Reshape array."""
        return uncertarray(
            self.data.reshape(*args, **kwargs),
            self.uncertainty.reshape(*args, **kwargs),
        )

    def take(self, *args: t.Any, **kwargs: t.Any) -> "uncertarray[T]":
        """Take array."""
        return uncertarray(
            self.data.take(*args, **kwargs),
            self.uncertainty.take(*args, **kwargs),
        )

    def transpose(self, *args: t.Any, **kwargs: t.Any) -> "uncertarray[T]":
        """Transpose array."""
        return uncertarray(
            self.data.transpose(*args, **kwargs),
            self.uncertainty.transpose(*args, **kwargs),
        )

    @property
    def size(self) -> int:
        """Return size of array."""
        return self.data.size

    def __array_function__(
        self,
        func: t.Any,
        types: tuple[type, ...],
        args: tuple[t.Any, ...],
        kwargs: dict[str, t.Any],
    ) -> t.Any:
        """Handle numpy array functions."""
        if func not in HANDLED_FUNCTIONS:
            return NotImplementedError(f"Function {func} not implemented")

        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __array_ufunc__(
        self,
        ufunc: t.Any,
        method: str,
        *inputs: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        """Handle numpy ufuncs."""
        inputs_data = tuple(i.data if isinstance(i, uncertarray) else i for i in inputs)
        inputs_uncertainty = tuple(i.uncertainty if isinstance(i, uncertarray) else i * 0.0 for i in inputs)

        uncert_inputs = tuple(uncertarray(x, y) for x, y in zip(inputs_data, inputs_uncertainty, strict=True))
        if method == "__call__":
            if ufunc.__name__ in unary_derivatives:
                data_result = ufunc(*inputs_data, **kwargs)
                uncertainty_result = _apply_unary_op_uncert(unary_derivatives[ufunc.__name__], uncert_inputs[0])
                return uncertarray(data_result, uncertainty_result)
            elif ufunc.__name__ in numpy_derivatives:
                data_result = getattr(np, ufunc.__name__)(*inputs_data, **kwargs)
                uncertainty_result = _apply_binary_op_uncert(*numpy_derivatives[ufunc.__name__], *uncert_inputs)
                return uncertarray(data_result, uncertainty_result)
            elif ufunc in HANDLED_FUNCTIONS:
                return HANDLED_FUNCTIONS[ufunc](*inputs, **kwargs)
            else:
                return ufunc(*inputs_data, **kwargs)

        else:
            raise NotImplementedError(f"Uncertainty propagation not implemented for this method {method}")

    # @property
    # def unit(self) -> t.Union[u.Unit, None]:
    #     """Return unit of array."""
    #     if hasattr(self.data, "unit"):
    #         return self.data.unit
    #     return None

    # def __lshift__(self, other: t.Union[float, int, UncertOrGeneric, "uncertarray", u.Unit]):
    #     """Left shift."""
    #     if issubclass(type(other), u.UnitBase):
    #         return uncertarray(self.data << other, self.uncertainty << other)

    #     return self.__array_ufunc__(np.left_shift, "__call__", self, other)

    def __neg__(self) -> "uncertarray":
        """Negate uncertarray."""
        return uncertarray(-self.data, self.uncertainty)

    def __abs__(self) -> "uncertarray":
        """Absolute value of uncertarray."""
        return uncertarray(np.abs(self.data), self.uncertainty)


# TODO: embed docs
def _generate_binary_ops() -> None:
    """Generate binary operators."""
    for op, (op_x, op_y) in operator_derivatives.items():
        operator_name = f"__{op}__"

        def fop(
            self: "uncertarray",
            other: UncertOrGeneric,
            op_x: BinaryCallableType = op_x,
            op_y: BinaryCallableType = op_y,
            op: str = operator_name,
        ) -> "uncertarray":
            return uncertarray(
                getattr(self.data, op)(_determine_data(other)),
                _apply_binary_op_uncert(op_x, op_y, self, other),
            )

        setattr(
            uncertarray,
            operator_name,
            fop,
        )

        operator_name = f"__r{op}__"

        def rop(
            self: "uncertarray",
            other: UncertOrGeneric,
            op_x: BinaryCallableType = op_x,
            op_y: BinaryCallableType = op_y,
            op: str = operator_name,
        ) -> "uncertarray":
            return uncertarray(
                getattr(self.data, op)(_determine_data(other)),
                _apply_binary_op_uncert(op_x, op_y, other, self),
            )

        setattr(
            uncertarray,
            operator_name,
            rop,
        )

        operator_name = f"__i{op}__"

        def inplace_op(
            self: "uncertarray",
            other: UncertOrGeneric,
            op: str = operator_name,
            op_x: BinaryCallableType = op_x,
            op_y: BinaryCallableType = op_y,
        ) -> "uncertarray":
            self.data = getattr(self.data, op)(_determine_data(other))
            self.uncertainty = _apply_binary_op_uncert(op_x, op_y, self, other)
            return self

        # Inplace
        setattr(
            uncertarray,
            operator_name,
            inplace_op,
        )


def _generate_meta_ops() -> None:
    """Generate binary operators."""
    for op, (op_x, op_y) in operator_derivatives.items():
        operator_name = f"__{op}__"

        def fop(self: _ucert_meta, other: t.Any, op_x=op_x, op_y=op_y, op=operator_name) -> _ucert_meta:
            return _ucert_meta(
                getattr(self.data, op)(_determine_data(other)),
                _apply_binary_op_uncert(op_x, op_y, self, other),
            )

        setattr(
            _ucert_meta,
            operator_name,
            fop,
        )

        operator_name = f"__r{op}__"

        def rop(self: _ucert_meta, other: t.Any, op_x=op_x, op_y=op_y, op=operator_name) -> _ucert_meta:
            return _ucert_meta(
                getattr(self.data, op)(_determine_data(other)),
                _apply_binary_op_uncert(op_x, op_y, other, self),
            )

        setattr(
            _ucert_meta,
            operator_name,
            rop,
        )

        operator_name = f"__i{op}__"

        def inplace_op(self: _ucert_meta, other: t.Any, op=operator_name, op_x=op_x, op_y=op_y) -> None:
            self.data = getattr(self.data, op)(_determine_data(other))
            self.uncertainty = _apply_binary_op_uncert(op_x, op_y, self, other)

        # Inplace
        setattr(
            _ucert_meta,
            operator_name,
            inplace_op,
        )

    # Create properly annotated unary methods for _ucert_meta.
    def _make_meta_unary(op_name: str, op_func: UnaryCallableType) -> t.Callable[[_ucert_meta], _ucert_meta]:
        def _meta_unary(self: _ucert_meta) -> _ucert_meta:
            return _ucert_meta(
                getattr(np, op_name)(self.data),
                _apply_unary_op_uncert(op_func, self),
            )

        return _meta_unary

    for op, op_func in unary_derivatives.items():
        setattr(_ucert_meta, op, _make_meta_unary(op, op_func))


_generate_binary_ops()
_generate_meta_ops()
