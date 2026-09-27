"""Download and fit stellar SEDs on an absolute flux scale."""

from .data import SED
from .model import StellarModel
from .fit import fit
from .likelihood import loglike_sed
from .plot import plot
from .fetch import download
from .spherex import download_spherex, load_spherex

__version__ = "0.1.3"
__all__ = ["SED", "StellarModel", "download", "download_spherex", "load_spherex", "fit", "loglike_sed", "plot"]
