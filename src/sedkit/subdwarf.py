"""Hot subdwarfs and their companions: TMAP NLTE atmospheres plus an optional cool star.

The subdwarf is a Tuebingen TMAP spectrum (TheoSSA) in one of three helium tiers, passed through the Gaia XP
forward model of the hot table and tabulated for R = 1 Rsun at 10 pc. Its radius is free; M = g R**2 / G
follows from log g. The companion is a sedkit FGKM dwarf (PARSEC mass, age and [M/H] through the J-CAPS
network), a subgiant from the empirical GiantTemplate (log g 3.2-3.8, free scale with its implied mass held to
0.7-3 Msun), or absent. The stars share the parallax and ZGR23 extinction and nothing else.
fit_subdwarf_companion compares a single FGK star, a single subdwarf and subdwarf + companion on one data vector.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .extinction import extinction_curve
from .fetch import ZERO_JY, TRAINING_SCALE
from .giant import (GiantTemplate, _dust_table, _parsec_prior, DUST_FLOOR, DUST_WALL, DUST_Z_GRID,
                    MAP_EDGE_PC, MBOL_SUN, TEFF_SUN)
from .model import StellarModel, _bilinear, HOT_CHANNELS

DATA = Path(__file__).parent / "models"
TIERS = ("H", "mid", "He")
N_XP, KS = 61, 63
G_CGS, MSUN_G, RSUN_CM = 6.674e-8, 1.989e33, 6.957e10
H_C_OVER_K_NM = 1.4387769e7      # hc/k in nm K
# Fractional model error of the subdwarf beyond the hot error term (XP and J/H/Ks): W1/W2 and SPHEREx,
# the six XP channels below 392 nm and the GALEX bands.
IR_ERROR, BLUE_ERROR, GALEX_ERROR = 0.03, 0.05, 0.05
# A companion's flux below 392 nm is a blackbody joined to its 392-412 nm XP flux; line blanketing makes it
# uncertain by this fraction.
COMPANION_UV_ERROR = 0.5
# GALEX: AB zero points (counts/s = 1) and the 10 per cent local roll-off, 114 and 303 counts/s (Morrissey et
# al. 2007, Table 1), and the calibration uncertainty added in quadrature.
GALEX_ZERO = {"FUV": 18.82, "NUV": 20.08}
GALEX_BRIGHT = {band: GALEX_ZERO[band] - 2.5 * np.log10(rate) for band, rate in (("FUV", 114.0), ("NUV", 303.0))}
GALEX_CALIBRATION = {"FUV": 0.05, "NUV": 0.03}
COMPANIONS = ("dwarf", "subgiant")
SUBGIANT_LOGG = (3.2, 3.8)
# A subgiant's implied mass g R**2 / G (R from its Ks flux, the parallax and BC_Ks) is held in SUBGIANT_MASS
# (solar masses) by Gaussian walls of SUBGIANT_WALL_DEX in log10 M.
SUBGIANT_MASS, SUBGIANT_WALL_DEX = (0.7, 3.0), 0.1
Z_MAX = 3.0
# Initial Nelder-Mead steps: Teff_sd/1000, log g_sd, ln R_sd, ln M, ln age, [M/H], Teff_g/100, log g_g,
# [M/H]_g, ln scale_g, E, z.
STEP = dict(t=0.5, g=0.1, lnr=0.05, lnm=0.05, lnage=0.3, feh=0.1, tg=1.0, gg=0.1, mg=0.1, lns=0.05,
            e=0.02, z=0.3)
PROFILE_COARSE_K, PROFILE_FINE_K, PROFILE_WINDOW_K = 2000.0, 500.0, 3000.0


def _planck(wave_nm, teff):
    x = np.minimum(H_C_OVER_K_NM / (wave_nm * teff), 700.0)
    return wave_nm**-5 / np.expm1(x)


class SubdwarfModel:
    """TMAP subdwarf channel tables with the XP operator correction of the hot table.

    predict(teff, logg, radius, tier) returns absolute 10-pc fluxes on the 168 sedkit channels, the six XP
    channels at 332-382 nm and a 2 nm spectrum over 130-1100 nm. Teff and log g outside a tier's table
    raise ValueError: the table is not extrapolated. correction="a" applies the per-channel XP operator
    offset of the hot table, "ab" adds its Balmer-index term, None neither. calibration="bundled" adds the
    subdwarf correction exp(a + W b) on XP, J/H/Ks and the blue channels, fitted to single subdwarfs at
    their spectroscopic labels (models/subdwarf/calibration.npz); a dict with arrays a, b, blue_a, blue_b
    supplies another one, None none.
    """

    def __init__(self, correction="ab", calibration="bundled", wavelength_um=None):
        if correction not in (None, "a", "ab"):
            raise ValueError("correction must be None, 'a' or 'ab'")
        directory = DATA / "subdwarf"
        self.summary = json.loads((directory / "summary.json").read_text())
        with np.load(directory / "subdwarf_table.npz", allow_pickle=False) as archive:
            table = {key: archive[key] for key in archive.files}
        self.tiers = {}
        for tier in TIERS:
            self.tiers[tier] = {key.split("/", 1)[1]: table[key].astype(float)
                                for key in table if key.startswith(tier + "/")}
        self.blue_wavelength_nm = table["blue_wavelength_nm"]
        self.coarse_wavelength_nm = table["coarse_wavelength_nm"]
        self.wavelength_um = StellarModel().wavelength_um if wavelength_um is None else wavelength_um
        self.correction = correction
        if isinstance(calibration, str):
            if calibration != "bundled":
                raise ValueError("calibration must be 'bundled', None or a dict of arrays")
            with np.load(directory / "calibration.npz") as archive:
                calibration = {key: archive[key] for key in archive.files}
        self.calibration = calibration
        with np.load(DATA / "hot" / "hot_table.npz") as hot:
            self.delta_a, self.delta_b = hot["delta_a"].astype(float), hot["delta_b"].astype(float)
        with np.load(DATA / "hot" / "model_error.npz") as error:
            self.error_basis = np.nan_to_num(error["hot_v5/basis"][:HOT_CHANNELS])
            self.error_diag = error["hot_v5/diag"][:HOT_CHANNELS]
        # extinction per ZGR23 E on the coarse grid and the blue channels: G23 below 392 nm, ZGR23 above
        uv_wave, uv = table["uv_curve_wavelength_nm"], table["uv_curve"]
        coarse = self.coarse_wavelength_nm
        self.coarse_curve = np.where(coarse < 392.0, np.interp(coarse, uv_wave, uv),
                                     extinction_curve(np.maximum(coarse, 392.0) / 1000))
        self.blue_curve = np.interp(self.blue_wavelength_nm, uv_wave, uv)
        # photon-weighted passband weights on the coarse grid
        self.passbands = {}
        for name in ("G", "BP", "RP", "FUV", "NUV"):
            lam, trans = table[f"filter/{name}_wave_nm"], table[f"filter/{name}_trans"]
            weight = np.interp(coarse, lam, trans, left=0.0, right=0.0) * coarse
            inside = np.flatnonzero(weight > 0)
            self.passbands[name] = (inside, weight[inside] / weight[inside].sum())
        lam = {name: table[f"filter/{name}_wave_nm"] for name in ("FUV", "NUV")}
        tr = {name: table[f"filter/{name}_trans"] for name in ("FUV", "NUV")}
        # pivot wavelengths convert AB magnitudes to photon-weighted F_lambda
        self.galex_pivot_nm = {name: float(np.sqrt(np.trapezoid(tr[name] * lam[name], lam[name])
                                                   / np.trapezoid(tr[name] / lam[name], lam[name])))
                               for name in lam}

    def support(self, tier):
        t = self.tiers[tier]
        return (float(t["teff_ax"][0]), float(t["teff_ax"][-1])), (float(t["logg_ax"][0]), float(t["logg_ax"][-1]))

    def in_support(self, teff, logg, tier):
        if tier not in self.tiers:
            return False
        (t0, t1), (g0, g1) = self.support(tier)
        return bool(np.isfinite(teff) and np.isfinite(logg) and t0 <= teff <= t1 and g0 <= logg <= g1)

    def predict(self, teff, logg, radius, tier):
        """10-pc fluxes of one subdwarf: dict with flux (168), blue (6), coarse and balmer_w."""
        if not self.in_support(teff, logg, tier):
            raise ValueError(f"Teff {teff} and log g {logg} lie outside the {tier} tier "
                             f"({self.support(tier) if tier in self.tiers else 'unknown tier'})")
        t = self.tiers[tier]
        teff, logg = np.atleast_1d(float(teff)), np.atleast_1d(float(logg))
        ln_flux = _bilinear(t["teff_ax"], t["logg_ax"], t["ln_flux"], teff, logg)[0]
        w = float(_bilinear(t["teff_ax"], t["logg_ax"], t["balmer_w"], teff, logg)[0])
        ln_blue = _bilinear(t["teff_ax"], t["logg_ax"], t["ln_blue"], teff, logg)[0]
        if self.correction:
            ln_flux[:HOT_CHANNELS] += self.delta_a + (w * self.delta_b if self.correction == "ab" else 0.0)
        if self.calibration is not None:
            c = self.calibration
            ln_flux[:HOT_CHANNELS] += c["a"] + w * c["b"]
            ln_blue = ln_blue + c["blue_a"] + w * c["blue_b"]
        r2 = float(radius)**2
        return dict(flux=np.exp(ln_flux) * r2,
                    blue=np.exp(ln_blue) * r2,
                    coarse=np.exp(_bilinear(t["teff_ax"], t["logg_ax"], t["ln_coarse"], teff, logg)[0]) * r2,
                    balmer_w=w)

    def fractional_error(self):
        """Fractional covariance of the subdwarf on the 168 channels: hot-term columns and diagonal."""
        basis = np.zeros((168, self.error_basis.shape[1]))
        basis[:HOT_CHANNELS] = self.error_basis
        diag = np.full(168, IR_ERROR)
        diag[:HOT_CHANNELS] = self.error_diag
        return basis, diag

    def passband(self, coarse, name):
        """Photon-weighted mean F_lambda of a coarse spectrum in a G/BP/RP/FUV/NUV passband."""
        inside, weight = self.passbands[name]
        return coarse[inside] @ weight


def physical(teff, logg, radius):
    """Mass (Msun) from g R**2 / G and luminosity (Lsun) from R**2 (Teff / Teff_sun)**4."""
    mass = 10**logg * (radius * RSUN_CM)**2 / G_CGS / MSUN_G
    return float(mass), float(radius**2 * (teff / TEFF_SUN)**4)


# ---------------------------------------------------------------- optional ultraviolet data

def blue_xp(sed, cache_dir="data"):
    """XP flux and error (model units) at 332-382 nm from the source's cached continuous spectrum.

    Calibrates cache_dir/source_id/xp.xml (written by `download`) with GaiaXPy on the six blue channels and
    caches blue_xp.ecsv beside it. Returns (flux, error) arrays of six values.
    """
    from astropy.table import Table

    directory = Path(cache_dir).expanduser() / str(sed.source_id)
    path = directory / "blue_xp.ecsv"
    if not path.exists():
        from gaiaxpy import calibrate

        raw = directory / "xp.xml"
        if not raw.exists():
            raise FileNotFoundError(f"{raw} is missing; run download(source_id, cache_dir=...) first")
        sampling = np.arange(332.0, 383.0, 10.0)
        calibrated, sampling = calibrate(str(raw), sampling=sampling, truncation=False, save_file=False)
        row = calibrated[calibrated["source_id"].astype(str) == str(sed.source_id)].iloc[0]
        Table([sampling, np.asarray(row["flux"], float), np.asarray(row["flux_error"], float)],
              names=["wavelength_nm", "flux_W_m2_nm", "error_W_m2_nm"]).write(path, format="ascii.ecsv")
    table = Table.read(path, format="ascii.ecsv")
    return np.asarray(table["flux_W_m2_nm"]) * 1e18, np.asarray(table["error_W_m2_nm"]) * 1e18


def galex(sed, cache_dir="data", radius_arcsec=3.0):
    """GALEX GR6/7 AIS FUV and NUV for the source: {band: (AB mag, error, usable)}.

    Queries VizieR II/335/galex_ais around the Gaia position and takes the nearest match within
    radius_arcsec. A band is usable when it is measured, fainter than the 10 per cent local roll-off and
    its artifact flag is at most 1 (detector edge). The result is cached as galex.json.
    """
    directory = Path(cache_dir).expanduser() / str(sed.source_id)
    path = directory / "galex.json"
    if path.exists():
        return {k: tuple(v) for k, v in json.loads(path.read_text()).items()}
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    coord = SkyCoord(float(sed.metadata["ra"]) * u.deg, float(sed.metadata["dec"]) * u.deg)
    tables = Vizier(columns=["_r", "FUVmag", "e_FUVmag", "NUVmag", "e_NUVmag", "Fafl", "Nafl"], row_limit=20
                    ).query_region(coord, radius=radius_arcsec * u.arcsec, catalog="II/335/galex_ais")
    out = {"FUV": (np.nan, np.nan, False), "NUV": (np.nan, np.nan, False)}
    if tables and len(tables[0]):
        row = tables[0][int(np.argmin(np.asarray(tables[0]["_r"], float)))]
        for band, flag in (("FUV", "Fafl"), ("NUV", "Nafl")):
            mag, err = row[f"{band}mag"], row[f"e_{band}mag"]
            if np.ma.is_masked(mag) or np.ma.is_masked(err):
                continue
            mag, err = float(mag), float(err)
            artifact = int(0 if np.ma.is_masked(row[flag]) else row[flag])
            out[band] = (mag, err, bool(mag > GALEX_BRIGHT[band] and artifact <= 1 and err > 0))
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def _galex_flux(mag, err, pivot_nm, band):
    """AB magnitude to photon-weighted F_lambda (model units) with the calibration term in quadrature."""
    fnu = 3631.0 * 10**(-0.4 * mag)   # Jy
    flux = fnu * 299792458.0 * 10.0 / pivot_nm**2
    return flux, flux * np.log(10) / 2.5 * np.hypot(err, GALEX_CALIBRATION[band])


# ---------------------------------------------------------------- the fit

class _Problem:
    """Data vector, shared nuisances and component models of one star.

    The vector holds the masked 168-channel fluxes, then optional blue XP and GALEX points. Every hypothesis
    is evaluated on all of it. Columns of the model covariance are concatenated per component.
    """

    def __init__(self, sed, sdm, stellar, template, mask, blue, galex_data, parallax, fit_parallax,
                 extinction, extinction_prior, dust, priors):
        self.sdm, self.stellar, self.template = sdm, stellar, template
        self.idx = np.flatnonzero(mask)
        self.mask = mask
        y, err, kind = list(sed.flux[self.idx]), list(sed.error[self.idx]), ["channel"] * len(self.idx)
        self.n_channel = len(self.idx)
        self.use_blue = blue is not None
        if self.use_blue:
            y += list(blue[0])
            err += list(blue[1])
            kind += ["blue"] * len(blue[0])
        self.galex_bands = []
        for band, (mag, e, usable) in (galex_data or {}).items():
            if usable:
                f, fe = _galex_flux(mag, e, sdm.galex_pivot_nm[band], band)
                y.append(f)
                err.append(fe)
                kind.append(band)
                self.galex_bands.append(band)
        self.y, self.sigma2, self.kind = np.array(y), np.array(err)**2, np.array(kind)
        self.curve_full = extinction_curve(sdm.wavelength_um)
        self.curve = self.curve_full[self.idx]
        self.plx0, self.plx_sigma = parallax
        self.fit_parallax = fit_parallax
        self.extinction, self.extinction_prior, self.dust, self.priors = extinction, extinction_prior, dust, priors
        self.sd_basis, self.sd_diag = sdm.fractional_error()
        self.coarse = sdm.coarse_wavelength_nm
        self.xp_nm = sdm.wavelength_um[:N_XP] * 1000

    # -- nuisances
    def parallax(self, z):
        return self.plx0 + z * self.plx_sigma

    def nuisance_penalty(self, e, z):
        value = z * z if self.fit_parallax else 0.0
        if self.extinction_prior is not None:
            value += ((e - self.extinction_prior[0]) / self.extinction_prior[1])**2
        if self.dust is not None:
            d = self.dust
            if 1000 / self.parallax(z) <= MAP_EDGE_PC:
                mean, sigma = np.interp(z, DUST_Z_GRID, d["mean"]), np.interp(z, DUST_Z_GRID, d["sigma"])
                value += ((e - mean) / np.hypot(sigma, DUST_FLOOR))**2
            elif e < d["edge"] - DUST_WALL:
                value += ((d["edge"] - DUST_WALL - e) / DUST_WALL)**2
        return value

    # -- components on the data vector: (flux, columns, variance, coarse spectrum, full 168 fluxes)
    def _extras(self, coarse, blue, blue_error, galex_error):
        flux, var = [], []
        if self.use_blue:
            flux.append(blue)
            var.append((blue_error * blue)**2)
        for band in self.galex_bands:
            f = self.sdm.passband(coarse, band)
            flux.append([f])
            var.append([(galex_error * f)**2])
        if not flux:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(flux), np.concatenate(var)

    def subdwarf(self, teff, logg, lnr, tier, scale, e):
        try:
            p = self.sdm.predict(teff, logg, np.exp(lnr), tier)
        except ValueError:
            return None
        att = np.exp(-e * self.curve)
        full = p["flux"] * scale
        f = full[self.idx] * att
        model_coarse = p["coarse"] * scale * np.exp(-e * self.sdm.coarse_curve)
        blue = p["blue"] * scale * np.exp(-e * self.sdm.blue_curve)
        xf, xv = self._extras(model_coarse, blue, BLUE_ERROR, GALEX_ERROR)
        coarse = self._channel_coarse(full[:N_XP] * np.exp(-e * self.curve_full[:N_XP]), model_coarse)
        columns = np.zeros((len(self.y), self.sd_basis.shape[1]))
        columns[:self.n_channel] = f[:, None] * self.sd_basis[self.idx]
        variance = np.r_[(f * self.sd_diag[self.idx])**2, xv]
        return dict(flux=np.r_[f, xf], columns=columns, variance=variance, coarse=coarse,
                    full=full * np.exp(-e * self.curve_full), balmer_w=p["balmer_w"])

    def _channel_coarse(self, xp_obs, model_coarse):
        """Coarse spectrum on the XP channel scale: the channels in 392-992 nm, the model spectrum beyond,
        joined at the edge channels, so subdwarf and companion fractions compare like with like."""
        coarse = np.exp(np.interp(self.coarse, self.xp_nm, np.log(np.maximum(xp_obs, 1e-300))))
        blue, red = self.coarse < self.xp_nm[0], self.coarse > self.xp_nm[-1]
        edge = np.interp(self.xp_nm[[0, -1]], self.coarse, model_coarse)
        coarse[blue] = model_coarse[blue] * xp_obs[0] / edge[0]
        coarse[red] = model_coarse[red] * xp_obs[-1] / edge[1]
        return coarse

    def _cool_coarse(self, xp_obs, teff):
        """Coarse spectrum of a cool star: its XP channels in 392-992 nm, blackbodies at Teff beyond."""
        coarse = np.exp(np.interp(self.coarse, self.xp_nm, np.log(np.maximum(xp_obs, 1e-300))))
        blue, red = self.coarse < self.xp_nm[0], self.coarse > self.xp_nm[-1]
        bb = _planck(self.coarse, teff)
        coarse[blue] = bb[blue] * np.mean(xp_obs[:3]) / np.mean(_planck(self.xp_nm[:3], teff))
        coarse[red] = bb[red] * xp_obs[-1] / _planck(self.xp_nm[-1], teff)
        return coarse

    def _cool(self, full, columns_full, variance_full, teff, e):
        att_full = np.exp(-e * self.curve_full[:len(full)])
        obs_full = full * att_full
        coarse = self._cool_coarse(obs_full[:N_XP], teff)
        blue = np.interp(self.sdm.blue_wavelength_nm, self.coarse, coarse)
        xf, xv = self._extras(coarse, blue, COMPANION_UV_ERROR, COMPANION_UV_ERROR)
        att = att_full[self.idx]
        columns = np.zeros((len(self.y), columns_full.shape[1]))
        columns[:self.n_channel] = np.nan_to_num(columns_full[self.idx]) * att[:, None]
        variance = np.r_[variance_full[self.idx] * att**2, xv]
        padded = np.full(168, np.nan)
        padded[:len(obs_full)] = obs_full
        return dict(flux=np.r_[obs_full[self.idx], xf], columns=columns, variance=variance,
                    coarse=coarse, full=padded)

    def dwarf(self, mass, age, feh, scale, e):
        prediction = self.stellar.evaluate(mass, 0.0, age, feh)
        if prediction is None:
            return None
        flux = prediction["flux_10pc"] * scale
        columns, variance = self.stellar.error_factors(prediction, scale)
        out = self._cool(flux, columns, variance, float(prediction["teff"][0]), e)
        out.update(teff=float(prediction["teff"][0]), logg=float(prediction["logg"][0]),
                   radius=float(prediction["radius"][0]), mass=float(mass), M_G=float(prediction["M_G"][0]))
        return out

    def subgiant(self, teff, logg, mh, lns, e):
        if not SUBGIANT_LOGG[0] <= logg <= SUBGIANT_LOGG[1]:
            return None
        out = self.template.evaluate(teff, logg, mh)
        if out is None:
            return None
        shape, frac_columns, frac_diag = out
        flux = np.full(168, np.nan)
        flux[:len(shape)] = np.exp(lns) * shape
        if not np.isfinite(flux[self.idx]).all():
            return None
        columns = np.zeros((168, frac_columns.shape[1]))
        columns[:len(shape)] = np.nan_to_num(frac_columns) * flux[:len(shape), None]
        variance = np.zeros(168)
        variance[:len(shape)] = frac_diag * flux[:len(shape)]**2
        res = self._cool(flux[:len(shape)], columns[:len(shape)], variance[:len(shape)], teff, e)
        res.update(teff=float(teff), logg=float(logg), feh=float(mh), scale=float(np.exp(lns)))
        return res

    # -- likelihood
    def likelihood(self, components):
        model = sum(c["flux"] for c in components)
        columns = np.concatenate([c["columns"] for c in components], axis=1)
        diagonal = self.sigma2 + sum(c["variance"] for c in components)
        residual = model - self.y
        small = np.eye(columns.shape[1]) + columns.T @ (columns / diagonal[:, None])
        nuisance = np.linalg.solve(small, columns.T @ (residual / diagonal))
        chi2 = float(np.sum((residual - columns @ nuisance)**2 / diagonal) + nuisance @ nuisance)
        return chi2 + float(np.log(diagonal).sum() + np.linalg.slogdet(small)[1]), chi2

    def label_penalty(self, sd=None, cool=None):
        value = 0.0
        for comp, priors in ((sd, self.priors["subdwarf"]), (cool, self.priors["companion"])):
            if comp is None:
                continue
            for key, (mean, sigma) in priors.items():
                value += ((comp[key] - mean) / sigma)**2
        return value


class _Hypothesis:
    """Parameter layout of one hypothesis and its objective."""

    def __init__(self, problem, name, tier=None, free_age=True, free_feh=True, age=None, feh=None, mass=None):
        self.p, self.name, self.tier = problem, name, tier
        names = []
        if name != "fgk":
            names += ["t", "g", "lnr"]
        if name in ("fgk", "dwarf"):
            names += (["lnm"] if mass is None or name == "fgk" else []) + (["lnage"] if free_age else []) + (
                ["feh"] if free_feh else [])
        self.mass = None if name == "fgk" else mass
        if name == "subgiant":
            names += ["tg", "gg", "mg", "lns"]
        if problem.extinction is None:
            names.append("e")
        if problem.fit_parallax:
            names.append("z")
        self.names, self.age, self.feh = names, age, feh

    def unpack(self, x, fixed=None):
        v = dict(fixed or {})
        v.update(zip([n for n in self.names if n not in (fixed or {})], x))
        return v

    def components(self, v):
        p = self.p
        e = v.get("e", p.extinction)
        z = v.get("z", 0.0)
        plx = p.parallax(z)
        if e is None or e < 0 or abs(z) > Z_MAX or plx <= 0:
            return None
        scale = (plx / 100.0)**2
        comps, sd, cool = [], None, None
        if self.name != "fgk":
            sd = p.subdwarf(1000 * v["t"], v["g"], v["lnr"], self.tier, scale, e)
            if sd is None:
                return None
            radius = float(np.exp(v["lnr"]))
            sd.update(teff=1000 * v["t"], logg=v["g"], radius=radius, mass=physical(1000 * v["t"], v["g"], radius)[0])
            comps.append(sd)
        if self.name in ("fgk", "dwarf"):
            age = np.exp(v["lnage"]) if "lnage" in v else self.age
            feh = v.get("feh", self.feh)
            low_age, high_age = p.stellar.age_range_gyr
            if not (low_age <= age <= high_age and -1.0 <= feh <= 0.5):
                return None
            cool = p.dwarf(np.exp(v["lnm"]) if "lnm" in v else self.mass, age, feh, scale, e)
            if cool is None:
                return None
            cool.update(age_gyr=float(age), feh=float(feh))
            comps.append(cool)
        if self.name == "subgiant":
            cool = p.subgiant(100 * v["tg"], v["gg"], v["mg"], v["lns"], e)
            if cool is None:
                return None
            comps.append(cool)
        return comps, sd, cool, e, z, plx

    def objective(self, x, fixed=None):
        out = self.components(self.unpack(x, fixed))
        if out is None:
            return np.inf
        comps, sd, cool, e, z, plx = out
        value = self.p.likelihood(comps)[0] + self.p.nuisance_penalty(e, z) + self.p.label_penalty(sd, cool)
        if self.name == "subgiant":
            value += _subgiant_physics(self.p, cool, plx, e)["mass_penalty"]
        return value

    def minimize(self, x0, fixed=None, maxfev=3000):
        free = [n for n in self.names if n not in (fixed or {})]
        x0 = np.asarray(x0, float)
        simplex = np.vstack([x0, x0 + np.diag([STEP[n] for n in free])])
        return minimize(self.objective, x0, args=(fixed,), method="Nelder-Mead",
                        options=dict(initial_simplex=simplex, xatol=1e-4, fatol=1e-3, maxfev=maxfev))


def _summary(hyp, v, objective, converged):
    """Parameters, fluxes and light fractions of a fitted hypothesis."""
    p = hyp.p
    comps, sd, cool, e, z, plx = hyp.components(v)
    minus2lnl, chi2 = p.likelihood(comps)
    out = dict(hypothesis=hyp.name if hyp.name == "fgk" else ("sdb" if hyp.name == "sdb" else f"sdb+{hyp.name}"),
               objective=float(objective), minus2lnL=minus2lnl, chi2=chi2, n_fit=len(p.y),
               converged=bool(converged), extinction_e=float(e), parallax_mas=float(plx), z=float(z),
               nuisance_penalty=float(p.nuisance_penalty(e, z)), label_penalty=float(p.label_penalty(sd, cool)),
               model=sum(c["flux"] for c in comps))
    if sd is not None:
        radius = float(np.exp(v["lnr"]))
        mass, lum = physical(sd["teff"], sd["logg"], radius)
        out["subdwarf"] = dict(teff=sd["teff"], logg=sd["logg"], radius=radius, mass=mass, luminosity=lum,
                               tier=hyp.tier, log_he_h=p.sdm.summary["tiers"][hyp.tier]["log_he_h"])
    if cool is not None:
        keep = ("teff", "logg", "radius", "mass", "age_gyr", "feh", "scale")
        out["companion"] = {k: cool[k] for k in keep if k in cool}
        out["companion"]["kind"] = hyp.name if hyp.name != "fgk" else "dwarf"
        if hyp.name == "subgiant":
            out["companion"].update({k: v for k, v in _subgiant_physics(p, cool, plx, e).items()
                                     if k != "mass_penalty"})
    if sd is not None and cool is not None:
        total_coarse = sd["coarse"] + cool["coarse"]
        out["fractions"] = {f"beta_{b}": float(p.sdm.passband(sd["coarse"], b) / p.sdm.passband(total_coarse, b))
                            for b in ("G", "BP", "RP")}
        out["fractions"]["channels"] = sd["full"] / (sd["full"] + cool["full"])
        out["coarse"] = dict(subdwarf=sd["coarse"], companion=cool["coarse"])
    elif sd is not None:
        out["fractions"] = dict(beta_G=1.0, beta_BP=1.0, beta_RP=1.0, channels=np.ones(168))
        out["coarse"] = dict(subdwarf=sd["coarse"])
    else:
        out["coarse"] = dict(companion=cool["coarse"])
    out["components"] = dict(subdwarf=None if sd is None else sd["full"], companion=None if cool is None else cool["full"])
    return out


def _subgiant_physics(p, cool, plx, e):
    """Luminosity and radius of a template subgiant from its dereddened Ks flux, the parallax and the
    PARSEC BC_Ks at its labels."""
    corners = p.template._corners(cool["teff"], cool["logg"], cool["feh"])
    bc = sum(w * _parsec_prior().bc[n] for n, w in corners)
    ks_flux = cool["full"][KS] / np.exp(-e * p.curve_full[KS])
    zero = ZERO_JY[2] * 299792458.0 * 10.0 / (p.sdm.wavelength_um[KS] * 1000)**2 * TRAINING_SCALE[2]
    mks = -2.5 * np.log10(ks_flux / zero) + 5 * np.log10(plx / 100)
    lum = 10**(-0.4 * (mks + bc - MBOL_SUN))
    radius = np.sqrt(lum) * (TEFF_SUN / cool["teff"])**2
    mass = 10**cool["logg"] * (radius * RSUN_CM)**2 / G_CGS / MSUN_G
    low, high = np.log10(SUBGIANT_MASS)
    wall = (max(low - np.log10(mass), 0.0, np.log10(mass) - high) / SUBGIANT_WALL_DEX)**2
    return dict(luminosity=float(lum), radius=float(radius), mass=float(mass), mass_penalty=float(wall))


def _check_prior(name, prior):
    if prior is None:
        return {}
    out = {}
    for key, value in prior.items():
        if len(value) != 2 or not np.all(np.isfinite(value)) or value[1] <= 0:
            raise ValueError(f"{name}[{key!r}] must be (mean, positive sigma)")
        out[key] = tuple(map(float, value))
    return out


def fit_subdwarf_companion(sed, *, companions=COMPANIONS, tiers=TIERS, model=None, stellar=None, template=None,
                           parallax=None, fit_parallax=False, extinction=None, extinction_prior=None,
                           dust_prior=None, subdwarf_prior=None, companion_prior=None, companion_age_gyr=None,
                           companion_feh=None, companion_mass=None, use_wise=False, use_spherex=False, blue=None,
                           galex=None):
    """Compare a single FGK star, a single hot subdwarf and a subdwarf with a cool companion.

    companions: any of "dwarf" (PARSEC + J-CAPS network) and "subgiant" (GiantTemplate, log g 3.2-3.8, free
    scale; its mass g R**2 / G, with R from its Ks flux, the parallax and the PARSEC BC_Ks, is held to
    SUBGIANT_MASS by walls); () fits the subdwarf alone. tiers: helium tiers of the subdwarf table to try ("H" pure hydrogen,
    "mid" log(He/H) = -1.9, "He" -0.1). The subdwarf has Teff, log g and a free radius; the companion its own
    labels. Both share the parallax, fixed at the catalogue value or, with fit_parallax, fitted within 3 sigma
    with penalty z**2 (parallax=(mean, sigma) replaces the SED's), and ZGR23 E: a number fixes it, None fits
    E >= 0 under extinction_prior=(mean, sigma) or the Edenhofer map at the trial distance (dust_prior;
    sed.metadata needs ra/dec). subdwarf_prior and companion_prior are dicts of (mean, sigma) Gaussians on
    'teff', 'logg', 'radius' and 'mass' (subdwarf; mass = g R**2 / G) or 'teff', 'logg' and 'feh' (companion;
    'feh' of a subgiant is its template [M/H]); companion_age_gyr, companion_feh and companion_mass fix the dwarf's age, [M/H] and mass (for
    example to profile the objective over companion masses). The data are the masked
    168 channels (W1/W2 only with use_wise; SPHEREx only with use_spherex and without a subgiant, which the
    template does not predict), plus blue=(flux, error) from blue_xp() and galex= from galex().
    GALEX needs the pure-hydrogen tier, the only one with NUV.

    Returns dict(hypotheses={"fgk", "sdb", "sdb+dwarf", "sdb+subgiant"}, delta, preferred, n_fit, mask):
    each hypothesis holds the objective (-2 ln L + penalties), minus2lnL, chi2, E, parallax, the subdwarf's
    Teff, log g, radius, mass and luminosity, the companion's labels, the subdwarf fractions beta_G/BP/RP
    (photon-weighted, reddened passband fluxes) and per channel, and for the subdwarf hypotheses `ranges`,
    the span of each parameter over the Teff_sd profile points within 1 of the minimum. delta is each
    objective minus the lowest. Results are constrained best fits, not posterior samples.
    """
    sdm = SubdwarfModel() if model is None else model
    stellar = StellarModel() if stellar is None else stellar
    companions = tuple(companions)
    if any(c not in COMPANIONS for c in companions):
        raise ValueError(f"companions must be drawn from {COMPANIONS}")
    tiers = tuple(tiers)
    if not tiers or any(t not in TIERS for t in tiers):
        raise ValueError(f"tiers must be drawn from {TIERS}")
    if "subgiant" in companions and template is None:
        template = GiantTemplate()
    parallax = (sed.parallax_mas, sed.parallax_error_mas) if parallax is None else tuple(map(float, parallax))
    if not np.isfinite(parallax[0]) or parallax[0] <= 0:
        raise ValueError("a positive parallax places both stars")
    if fit_parallax and not (np.isfinite(parallax[1]) and parallax[1] > 0):
        raise ValueError("fit_parallax needs a positive parallax error")
    if extinction is not None and (not np.isfinite(extinction) or extinction < 0):
        raise ValueError("extinction must be a nonnegative ZGR23 E or None")
    if extinction is not None and (dust_prior is not None or extinction_prior is not None):
        raise ValueError("a fixed extinction takes no extinction prior")
    if extinction is None and (dust_prior is None) == (extinction_prior is None):
        raise ValueError("a fitted extinction needs exactly one of extinction_prior and dust_prior")
    priors = dict(subdwarf=_check_prior("subdwarf_prior", subdwarf_prior),
                  companion=_check_prior("companion_prior", companion_prior))
    if galex and any(t != "H" for t in tiers) and any(v[2] for v in galex.values()):
        raise ValueError("GALEX data need tiers=('H',): the helium tiers have no NUV spectrum")
    mask = sed.fit_mask(use_wise=use_wise)
    if not use_spherex or "subgiant" in companions:
        mask[66:] = False
    if not mask[:N_XP].any():
        raise ValueError("the fit needs XP channels")
    dust = None if dust_prior is None else _dust_table(dust_prior, sed, parallax)
    problem = _Problem(sed, sdm, stellar, template, mask, blue, galex, parallax, fit_parallax, extinction,
                       extinction_prior, dust, priors)
    e0 = (extinction if extinction is not None else
          extinction_prior[0] if extinction_prior is not None else float(dust["mean"][len(DUST_Z_GRID) // 2]))
    e0 = max(e0, 0.0)
    age_kw = dict(free_age=companion_age_gyr is None, free_feh=companion_feh is None,
                  age=companion_age_gyr, feh=companion_feh, mass=companion_mass)

    results = {"fgk": _fit_fgk(problem, age_kw, e0)}
    results["sdb"] = _fit_subdwarf(problem, "sdb", tiers, age_kw, e0)
    for name in companions:
        results[f"sdb+{name}"] = _fit_subdwarf(problem, name, tiers, age_kw, e0)
    best = min(r["objective"] for r in results.values())
    return dict(hypotheses=results, delta={k: r["objective"] - best for k, r in results.items()},
                preferred=min(results, key=lambda k: results[k]["objective"]), n_fit=len(problem.y),
                mask=mask, data=dict(y=problem.y, error=np.sqrt(problem.sigma2), kind=problem.kind),
                source_id=sed.source_id)


def _start_values(hyp, e0):
    """Default starting values of the shared and companion parameters."""
    p = hyp.p
    start = dict(e=e0, z=0.0, lnage=np.log(3.0), feh=hyp.feh if hyp.feh is not None else 0.0)
    prior = p.priors["companion"]
    if "feh" in prior and hyp.feh is None:
        start["feh"] = float(np.clip(prior["feh"][0], -1.0, 0.5))
    return start


def _fit_fgk(p, age_kw, e0):
    hyp = _Hypothesis(p, "fgk", **age_kw)
    base = _start_values(hyp, e0)
    trials = []
    for mass in np.geomspace(0.5, 1.6, 12):
        for age in ((1.0, 3.0, 8.0) if age_kw["free_age"] else (None,)):
            v = dict(base, lnm=np.log(mass), lnage=np.log(age) if age else base["lnage"])
            x0 = [v[n] for n in hyp.names]
            f = hyp.objective(x0)
            if np.isfinite(f):
                trials.append((f, x0))
    if not trials:
        raise ValueError("no supported single FGK star for these data")
    trials.sort(key=lambda t: t[0])
    best = min((hyp.minimize(x0) for _, x0 in trials[:3]), key=lambda r: r.fun)
    out = _summary(hyp, hyp.unpack(best.x), best.fun, best.success)
    return out


def _sd_start(hyp, teff, tier, base, cool_v=None):
    """Starting radius from a linear fit of the subdwarf to the XP residual of any companion."""
    p = hyp.p
    plx = p.parallax(base.get("z", 0.0))
    scale = (plx / 100)**2
    (g0, g1) = p.sdm.support(tier)[1]
    logg = float(np.clip(p.priors["subdwarf"].get("logg", (5.6, 0))[0], g0, g1))
    sd = p.subdwarf(teff, logg, 0.0, tier, scale, base["e"])
    if sd is None:
        return None
    residual = p.y.copy()
    if cool_v is not None:
        comps = hyp.components(dict(base, t=teff / 1000, g=logg, lnr=np.log(0.15), **cool_v))
        if comps is None:
            return None
        residual = residual - comps[2]["flux"]
    w = 1 / (p.sigma2 + (0.02 * p.y)**2)
    xp = slice(0, min(p.n_channel, N_XP))
    amp = np.sum(sd["flux"][xp] * residual[xp] * w[xp]) / np.sum(sd["flux"][xp]**2 * w[xp])
    return logg, 0.5 * np.log(max(amp, 1e-4))


def _cool_starts(hyp, base):
    """Candidate companion parameters, best first by their own share of the red XP flux."""
    p = hyp.p
    prior = p.priors["companion"]
    if hyp.name == "dwarf":
        ages = (1.0, 3.0, 8.0) if hyp.age is None else (hyp.age,)
        masses = (0.6, 0.8, 1.0, 1.2, 1.4) if hyp.mass is None else (hyp.mass,)
        return [dict(lnage=np.log(a), **({} if hyp.mass is not None else dict(lnm=np.log(m))))
                for m in masses for a in ages]
    teff0 = prior.get("teff", (5800.0, 0))[0]
    mh0 = prior.get("feh", (0.0, 0))[0]
    out = []
    for t in (teff0 - 300, teff0, teff0 + 300):
        for g in (3.3, 3.6):
            out.append(dict(tg=t / 100, gg=g, mg=mh0, lns=0.0))
    return out


def _fit_subdwarf(p, name, tiers, age_kw, e0):
    """Profile over the subdwarf Teff: coarse nodes in every tier, fine nodes around the best, then a
    free polish. Returns the summary of the best fit with the profile and parameter ranges."""
    profile = []
    best_overall = None
    for tier in tiers:
        hyp = _Hypothesis(p, name, tier, **age_kw)
        (t0, t1), _ = p.sdm.support(tier)
        base = _start_values(hyp, e0)
        cool_v = _cool_seed(hyp, base, (t0 + t1) / 2, tier) if name != "sdb" else None
        if name != "sdb" and cool_v is None:
            continue

        def node_fit(teff, start):
            fixed = dict(t=teff / 1000)
            x0 = [start[n] for n in hyp.names if n != "t"]
            r = hyp.minimize(x0, fixed, maxfev=2500)
            v = hyp.unpack(r.x, fixed)
            return r.fun, v, r.success

        def seed(teff):
            s = _sd_start(hyp, teff, tier, base, cool_v)
            if s is None:
                return None
            v = dict(base, t=teff / 1000, g=s[0], lnr=s[1], **(cool_v or {}))
            return v

        nodes = np.arange(t0, t1 + 1, PROFILE_COARSE_K)
        last = None
        tier_rows = []
        for teff in nodes:
            start = seed(teff)
            if start is None:
                continue
            if last is not None:   # carry the companion and nuisances along the profile
                start.update({k: last[k] for k in last if k not in ("t", "g", "lnr")})
            f, v, ok = node_fit(teff, start)
            tier_rows.append((f, teff, v, ok))
            if np.isfinite(f):
                last = v
        if not tier_rows:
            continue
        f_best, t_best, v_best, _ = min(tier_rows, key=lambda r: r[0])
        for teff in np.arange(max(t0, t_best - PROFILE_WINDOW_K), min(t1, t_best + PROFILE_WINDOW_K) + 1,
                              PROFILE_FINE_K):
            if any(abs(teff - r[1]) < 1 for r in tier_rows):
                continue
            f, v, ok = node_fit(teff, dict(v_best))
            tier_rows.append((f, teff, v, ok))
        tier_rows.sort(key=lambda r: r[1])
        profile += [dict(tier=tier, teff=float(t), objective=float(f), params=v, converged=bool(ok))
                    for f, t, v, ok in tier_rows]
        f_best, t_best, v_best, _ = min(tier_rows, key=lambda r: r[0])
        r = hyp.minimize([v_best[n] for n in hyp.names])
        if r.fun <= f_best:
            v_best, f_best, ok = hyp.unpack(r.x), r.fun, r.success
        else:
            ok = False
        if best_overall is None or f_best < best_overall[0]:
            best_overall = (f_best, hyp, v_best, ok)
    if best_overall is None:
        raise ValueError(f"no supported {name} solution for these data")
    f_best, hyp, v_best, ok = best_overall
    out = _summary(hyp, v_best, f_best, ok)
    out["profile"] = [dict(tier=r["tier"], teff=r["teff"], objective=r["objective"],
                           delta=r["objective"] - f_best, converged=r["converged"]) for r in profile]
    out["ranges"] = _ranges(p, name, profile, f_best, out, age_kw)
    return out


def _cool_seed(hyp, base, teff_sd, tier):
    """Companion starting parameters: the candidate with the lowest objective at a mid-table subdwarf."""
    best = None
    for cv in _cool_starts(hyp, base):
        s = _sd_start(hyp, teff_sd, tier, base, cv)
        if s is None:
            continue
        v = dict(base, t=teff_sd / 1000, g=s[0], lnr=s[1], **cv)
        f = hyp.objective([v[n] for n in hyp.names])
        if np.isfinite(f) and (best is None or f < best[0]):
            best = (f, cv)
    return None if best is None else best[1]


def _ranges(p, name, profile, f_best, best, age_kw):
    """Span of each reported parameter where the Teff_sd profile lies within 1 of the minimum.

    Profile points inside that level count, and so do the two crossings of the level, where parameters
    are interpolated linearly between the bracketing profile points of the same tier.
    """
    points = [(r["tier"], r["params"]) for r in profile if r["objective"] - f_best <= 1.0]
    for tier in {r["tier"] for r in profile}:
        rows = sorted((r for r in profile if r["tier"] == tier and np.isfinite(r["objective"])),
                      key=lambda r: r["teff"])
        for a, b in zip(rows[:-1], rows[1:]):
            da, db = a["objective"] - f_best - 1.0, b["objective"] - f_best - 1.0
            if da * db < 0:
                w = da / (da - db)
                points.append((tier, {k: (1 - w) * a["params"][k] + w * b["params"][k] for k in a["params"]}))
    values = {}
    for tier, params in points:
        hyp = _Hypothesis(p, name, tier, **age_kw)
        if hyp.components(params) is None:
            continue
        s = _summary(hyp, params, f_best, True)
        for key in ("teff", "logg", "radius", "mass", "luminosity"):
            values.setdefault(f"subdwarf_{key}", []).append(s["subdwarf"][key])
        if "companion" in s:
            for key, value in s["companion"].items():
                if isinstance(value, float):
                    values.setdefault(f"companion_{key}", []).append(value)
            values.setdefault("beta_G", []).append(s["fractions"]["beta_G"])
        values.setdefault("extinction_e", []).append(s["extinction_e"])
    values.setdefault("subdwarf_teff", []).append(best["subdwarf"]["teff"])
    return {k: (float(min(v)), float(max(v))) for k, v in values.items()}
