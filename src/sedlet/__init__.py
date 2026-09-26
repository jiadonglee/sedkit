"""Download and fit stellar SEDs on an absolute flux scale."""

from .data import SED
from .model import StellarModel
from .fit import fit
from .plot import plot
from .fetch import download

__version__ = "0.1.0"
__all__ = ["SED", "StellarModel", "download", "fit", "plot"]
