try:
    from astropy import units as u

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False


def nop(*args, **kwargs):
    """No operation function."""
    pass


def only_if_astropy_available(func):
    """Decorator to run function only if Astropy is available."""

    def wrapper(*args, **kwargs):
        if not ASTROPY_AVAILABLE:
            return nop(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


@only_if_astropy_available
def add_astropy_unit_support():
    """Add Astropy unit support to uncertarray."""
    from .core import uncertarray

    def to_astropy_unit(self, unit: u.Unit, equivalences: u.Equivalency) -> uncertarray:
        return uncertarray(self.data.to(unit), self.uncertainty.to(unit))

    uncertarray.to = to_astropy_unit

    # Add a unit property to uncertarray
    @property
    def unit(self) -> u.Unit:
        return self.data.unit

    uncertarray.unit = unit


@only_if_astropy_available
def modify_lshift_operator():
    """Modify the left shift operator to support Astropy units."""
    from .core import uncertarray

    current_lshift = getattr(uncertarray, "__lshift__", None)

    def lshift(self, other):
        if isinstance(other, u.UnitBase):
            return uncertarray(self.data << other, self.uncertainty << other)
        else:
            return current_lshift(self, other)

    uncertarray.__lshift__ = lshift


def initialize_astropy_support():
    """Initialize Astropy support if available."""
    add_astropy_unit_support()
    modify_lshift_operator()
