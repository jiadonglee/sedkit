"""Download and fit stellar SEDs on an absolute flux scale."""

from .data import SED
from .model import StellarModel
from .fit import fit
from .plot import plot
from .fetch import download
from .spherex import download_spherex, load_spherex

__version__ = "0.1.2"
__all__ = ["SED", "StellarModel", "download", "download_spherex", "load_spherex", "fit", "plot"]
