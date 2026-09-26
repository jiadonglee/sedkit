"""Observed channel fluxes, errors, masks and catalogue provenance."""

from dataclasses import dataclass, field, replace
import json
from pathlib import Path

import numpy as np

from .model import FLUX_UNIT, StellarModel


@dataclass
class SED:
    """An observed SED in 1e-18 W m^-2 nm^-1, at its actual distance.

    Channel order is XP61, J/H/Ks/W1/W2, SPHEREx102. Missing measurements
    are NaN and masked. Fluxes and errors are never adjusted by fitting.
    """

    flux: np.ndarray
    error: np.ndarray
    mask: np.ndarray
    parallax_mas: float
    parallax_error_mas: float = 0.0
    source_id: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.flux = np.asarray(self.flux, float).copy()
        self.error = np.asarray(self.error, float).copy()
        self.mask = np.asarray(self.mask, bool).copy()
        if any(value.shape != (168,) for value in (self.flux, self.error, self.mask)):
            raise ValueError("flux, error and mask must have 168 channels")
        self.mask &= np.isfinite(self.flux) & np.isfinite(self.error) & (self.error > 0)
        self.source_id = str(self.source_id)

    @property
    def wavelength_um(self):
        return StellarModel().wavelength_um

    def save(self, path):
        """Save measurements and provenance without pickled Python objects."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, flux=self.flux, error=self.error, mask=self.mask,
                                parallax_mas=self.parallax_mas,
                                parallax_error_mas=self.parallax_error_mas,
                                source_id=self.source_id,
                                metadata=json.dumps(self.metadata), flux_unit=FLUX_UNIT)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            return cls(archive["flux"], archive["error"], archive["mask"],
                       float(archive["parallax_mas"]), float(archive["parallax_error_mas"]),
                       str(archive["source_id"]), json.loads(str(archive["metadata"])))

    def with_spherex(self, wavelength_um, flux, error, valid=None):
        """Attach an extracted 102-channel spectrum on the bundled grid.

        Input flux and error use FLUX_UNIT at the source's actual distance.
        Extraction and resampling happen upstream; this method changes no
        observed values and rejects a mismatched wavelength grid.
        """
        wave = np.asarray(wavelength_um, float)
        expected = self.wavelength_um[66:]
        if wave.shape != expected.shape or not np.allclose(wave, expected, rtol=0, atol=1e-6):
            raise ValueError("SPHEREx wavelengths must match the bundled 102-channel grid")
        flux, error = np.asarray(flux, float), np.asarray(error, float)
        if flux.shape != (102,) or error.shape != (102,):
            raise ValueError("SPHEREx flux and error must have 102 channels")
        result = replace(self, metadata=dict(self.metadata))
        result.flux[66:], result.error[66:] = flux, error
        good = np.isfinite(flux) & np.isfinite(error) & (error > 0)
        if valid is not None:
            good &= np.asarray(valid, bool)
        result.mask[66:] = good
        return result

    def fit_mask(self, use_wise=False):
        """Quality mask shared by single and binary; W1/W2 held out by default.

        Attached SPHEREx data retain their supplied quality mask. For sources
        with Gaia G < 9, SPHEREx is excluded because of bright-source bias.
        """
        mask = self.mask.copy()
        if not use_wise:
            mask[64:66] = False
        if self.metadata.get("phot_g_mean_mag", np.inf) < 9:
            mask[66:] = False
        return mask
