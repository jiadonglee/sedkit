"""Download and fit stellar SEDs on an absolute flux scale."""

from .data import SED
from .model import StellarModel
from .fit import fit
from .giant import GiantTemplate, fit_giant_companion
from .subdwarf import SubdwarfModel, fit_subdwarf_companion
from .likelihood import loglike_sed
from .extinction import EdenhoferPrior
from .plot import plot
from .fetch import download
from .gaia import query_gaia, download_gaia
from .spherex import download_spherex, load_spherex

__version__ = "0.2.0"
__all__ = ["SED", "StellarModel", "EdenhoferPrior", "download", "query_gaia", "download_gaia", "download_spherex", "load_spherex", "fit", "loglike_sed", "plot", "GiantTemplate", "fit_giant_companion", "SubdwarfModel", "fit_subdwarf_companion"]
