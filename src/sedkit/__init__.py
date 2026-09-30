"""Download and fit stellar SEDs on an absolute flux scale."""

from .data import SED
from .model import StellarModel
from .fit import fit
from .likelihood import loglike_sed
from .plot import plot
from .fetch import download
from .gaia import query_gaia, download_gaia
from .spherex import download_spherex, load_spherex

__version__ = "0.1.4"
__all__ = ["SED", "StellarModel", "download", "query_gaia", "download_gaia", "download_spherex", "load_spherex", "fit", "loglike_sed", "plot"]
