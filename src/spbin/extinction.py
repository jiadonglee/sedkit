"""ZGR23 attenuation and a distance-dependent Edenhofer dust prior."""

from pathlib import Path

import numpy as np
from scipy.special import log_ndtr


def extinction_curve(wavelength_um):
    """Optical depth per ZGR23 E; log interpolation of the published curve.

    Beyond W2, continue the W1--W2 power-law slope to the SPHEREx edge.
    Broadband coefficients are evaluated at their catalogue wavelengths.
    """
    table = np.loadtxt(Path(__file__).parent / "models" / "extinction_curve.txt")
    wave = np.asarray(wavelength_um, float) * 1000
    if np.any(~np.isfinite(wave)) or np.any(wave < table[0, 0] - 1e-4):
        raise ValueError("extinction curve requires wavelengths >= 0.392 micron")
    log_wave, log_curve = np.log(table[:, 0]), np.log(table[:, 1])
    result = np.interp(np.log(wave), log_wave, log_curve)
    slope = (log_curve[-1] - log_curve[-2]) / (log_wave[-1] - log_wave[-2])
    result = np.where(wave > table[-1, 0],
                      log_curve[-1] + slope * (np.log(wave) - log_wave[-1]), result)
    return np.exp(result)


def transmission(wavelength_um, extinction):
    """Shared foreground attenuation exp(-E * k_lambda), with E >= 0."""
    if not np.isfinite(extinction) or extinction < 0:
        raise ValueError("extinction must be a finite nonnegative ZGR23 E")
    return np.exp(-extinction * extinction_curve(wavelength_um))


class EdenhoferPrior:
    """Nonnegative Gaussian approximation to integrated map extinction E.

    Reuse one instance across targets. The default loads posterior samples
    from an existing dustmaps installation; no map is downloaded. sigma=
    explicitly replaces the map width, allowing a mean-only query object.
    """

    def __init__(self, query=None, *, sigma=None):
        if sigma is not None and (not np.isfinite(sigma) or sigma <= 0):
            raise ValueError("sigma must be a positive width in ZGR23 E")
        if query is None:
            from dustmaps.edenhofer2023 import Edenhofer2023Query

            query = Edenhofer2023Query(integrated=True, load_samples=sigma is None)
        if not query.integrated:
            raise ValueError("Edenhofer query must use integrated=True")
        if sigma is None and query.n_samples is None:
            raise ValueError("integrated map uncertainty requires load_samples=True or explicit sigma=")
        self.query = query
        self.sigma = sigma

    def moments(self, ra, dec, distance_pc):
        """Mean and width at ICRS degrees and the trial distance in pc."""
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg,
                          distance=distance_pc * u.pc, frame="icrs")
        mean = float(np.asarray(self.query(coords, mode="mean")).item())
        sigma = self.sigma if self.sigma is not None else float(
            np.asarray(self.query(coords, mode="std")).item())
        if not np.isfinite(mean) or mean < 0 or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Edenhofer prior has no support at this position/distance or no positive width")
        return mean, sigma

    @staticmethod
    def penalty(extinction, mean, sigma):
        """-2 log p(E|distance), omitting only the constant log(2pi).

        Retain width and E>=0 normalization when the fitted distance moves.
        """
        if not np.isfinite(extinction) or extinction < 0:
            return np.inf
        return ((extinction - mean) / sigma)**2 + 2*np.log(sigma) + 2*log_ndtr(mean / sigma)
