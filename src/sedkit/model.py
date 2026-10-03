"""Coeval stellar spectra from PARSEC tracks and a compact empirical network."""

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "models"
FLUX_UNIT = "1e-18 W m^-2 nm^-1"
KS_ZERO_FLAMBDA = 666.7 * 299792458.0 * 10.0 / 2159.0**2
# Primary Teff range (K) over which the cold and warm model-error terms are blended.
ERROR_BLEND_K = (3800.0, 4200.0)


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _stellar_parameters(masses, age_gyr, tracks):
    """Interpolate logTe, logL, Ks and G in mass and log age."""
    ages = np.array(sorted(tracks))
    log_age = np.log10(age_gyr) + 9
    result = np.full((len(masses), 4), np.nan)
    if not ages[0] <= log_age <= ages[-1]:
        return result
    high = np.clip(np.searchsorted(ages, log_age, side="right"), 1, len(ages) - 1)
    lower, upper = tracks[ages[high - 1]], tracks[ages[high]]
    weight = (log_age - ages[high - 1]) / (ages[high] - ages[high - 1])
    valid = ((masses >= max(lower[0, 0], upper[0, 0]))
             & (masses <= min(lower[-1, 0], upper[-1, 0])))
    for column in range(4):
        result[valid, column] = (
            (1 - weight) * np.interp(masses[valid], lower[:, 0], lower[:, column + 1])
            + weight * np.interp(masses[valid], upper[:, 0], upper[:, column + 1]))
    return result


class StellarModel:
    """J-CAPS v2.1 stellar network with PARSEC masses, age and metallicity.

    Outputs 168 channels at 10 pc: XP61, J/H/Ks/W1/W2, SPHEREx102.
    Age is 0.5--10 Gyr and [M/H] is -1--0.5. Each component must be
    inside the network's training domain. No cold-atmosphere fallback.
    """

    def __init__(self):
        directory = DATA / "stellar"
        self.summary = json.loads((directory / "summary.json").read_text())
        self.norm = _load(directory / "normalization.npz")
        self.params = _load(directory / "model_params.npz")
        self.calibration = _load(directory / "nir_calibration.npz")
        self.tracks = {}
        for key, value in _load(DATA / "parsec_mh_tracks.npz").items():
            metallicity, age = map(float, key.split("/"))
            self.tracks.setdefault(metallicity, {})[age] = value
        self.metals = np.array(sorted(self.tracks))
        self.errors = _load(DATA / "model_error.npz")
        self.wavelength_um = np.r_[self.norm["xp_spectral_wavelength_nm"] / 1000,
                                   self.norm["photometry_wavelength_um"],
                                   self.norm["spherex_wavelength_um"]]
        self.channels = np.array([f"XP{i:02d}" for i in range(61)]
                                 + ["J", "H", "Ks", "W1", "W2"]
                                 + [f"SPHEREx{i:03d}" for i in range(102)])

    def labels(self, masses, age_gyr=5.0, feh=0.0):
        """Return [Teff, M_Ks, G-Ks, [M/H]] for masses in solar units."""
        masses = np.atleast_1d(np.asarray(masses, float))
        if (not np.isfinite(age_gyr) or not 0.5 <= age_gyr <= 10
                or not np.isfinite(feh) or not -1 <= feh <= 0.5):
            raise ValueError("age_gyr must be 0.5--10 and feh must be -1--0.5")
        k = np.clip(np.searchsorted(self.metals, feh, side="right"), 1, len(self.metals) - 1)
        low, high = self.metals[k - 1:k + 1]
        weight = (feh - low) / (high - low)
        lower = _stellar_parameters(masses, age_gyr, self.tracks[low])
        upper = _stellar_parameters(masses, age_gyr, self.tracks[high])
        pars = lower if weight < 1e-9 else upper if weight > 1 - 1e-9 else (
            (1 - weight) * lower + weight * upper)
        return np.c_[10**pars[:, 0], pars[:, 2], pars[:, 3] - pars[:, 2],
                     np.full(len(masses), feh)]

    def in_domain(self, labels):
        """Training coverage only; this is not a measured accuracy statement."""
        labels = np.atleast_2d(np.asarray(labels, float))
        inside = np.zeros(len(labels), bool)
        for box in self.summary["prediction_domain_boxes"]:
            bounds = np.asarray(box)
            inside |= np.all(np.isfinite(labels) & (labels >= bounds[:, 0])
                             & (labels <= bounds[:, 1]), axis=1)
        return inside

    def _network(self, name, values):
        count = sum(key.startswith(name + "_weight_") for key in self.params)
        for index in range(count):
            values = values @ self.params[f"{name}_weight_{index}"].T + self.params[f"{name}_bias_{index}"]
            if index < count - 1:
                values = 0.5 * values * (1 + np.tanh(
                    np.sqrt(2 / np.pi) * (values + 0.044715 * values**3)))
        return values

    def predict_labels(self, labels):
        """Absolute 10-pc channel fluxes for explicit four-coordinate labels."""
        labels = np.atleast_2d(np.asarray(labels, float))
        if labels.shape[1] != 4 or not self.in_domain(labels).all():
            raise ValueError("labels must be supported [Teff, M_Ks, G-Ks, [M/H]]")
        shape = labels[:, [0, 2, 3]]
        hidden = self._network("trunk", (shape - self.norm["x_center"]) / self.norm["x_scale"])
        values = np.concatenate([
            self._network(name + "_head", hidden) * self.norm[name + "_y_scale"]
            + self.norm[name + "_y_center"] for name in ("xp", "spherex")], axis=1)
        log_flux = (values - values[:, 63, None] + np.log10(KS_ZERO_FLAMBDA)
                    - 0.4 * labels[:, 1, None])
        temperature = (labels[:, 0] - 2600) / 500
        color = labels[:, 2] - 5.5
        u = np.clip((labels[:, 0] - 2800) / 600, 0, 1)
        gate = 1 - 10*u**3 + 15*u**4 - 6*u**5
        features = np.stack([np.ones_like(u), temperature, color,
                             temperature * color, temperature**2, color**2], axis=1)
        log_flux += ((gate[:, None] * features) @ self.calibration["coefficients"]
                     @ self.calibration["spectral_basis"].T)
        return 10**log_flux

    def evaluate(self, m1, q=0.0, age_gyr=5.0, feh=0.0):
        """One star (q=0) or a coeval pair; None if either is unsupported.

        beta_g is F_G,2/F_G,1, not the companion fraction of total light.
        Signed a_phot/a1 follows the primary-frame orbit convention.
        """
        if not np.isfinite(m1) or m1 <= 0 or not np.isfinite(q) or not 0 <= q <= 1:
            raise ValueError("m1 must be positive and q must be 0--1")
        masses = np.array([m1] if q == 0 else [m1, m1 * q])
        labels = self.labels(masses, age_gyr, feh)
        if not self.in_domain(labels).all():
            return None
        components = self.predict_labels(labels)
        mg = labels[:, 1] + labels[:, 2]
        beta = 0.0 if q == 0 else float(10**(-0.4 * (mg[1] - mg[0])))
        return dict(masses=masses, labels=labels, teff=labels[:, 0], M_G=mg,
                    components=components, flux_10pc=components.sum(axis=0),
                    beta_g=beta, a_phot_over_a1=None if q == 0 else (q - beta) / (q * (1 + beta)),
                    age_gyr=float(age_gyr), feh=float(feh), route="PARSEC+J-CAPS-v2.1")

    def error_factors(self, prediction, scale=1.0):
        """Shared fractional model error for all components of a system.

        The primary's Teff selects the error term for both components:
        v2.1_cold below 3800 K, v2.1_warm above 4200 K, and in between
        the covariance (1-w) C_cold + w C_warm with a smoothstep weight w.
        Returns low-rank columns and diagonal variance.
        """
        u = np.clip((prediction["teff"][0] - ERROR_BLEND_K[0])
                    / (ERROR_BLEND_K[1] - ERROR_BLEND_K[0]), 0, 1)
        warm = 10*u**3 - 15*u**4 + 6*u**5
        flux = prediction["flux_10pc"] * scale
        columns, variance = [], np.zeros_like(flux)
        for term, weight in (("v2.1_cold", 1 - warm), ("v2.1_warm", warm)):
            if weight > 0:
                columns.append(np.sqrt(weight) * flux[:, None] * self.errors[term + "/basis"])
                variance += weight * (flux * self.errors[term + "/diag"])**2
        return np.concatenate(columns, axis=1), variance
