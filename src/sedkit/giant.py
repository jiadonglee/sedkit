"""Hot main-sequence companions to red giants against an empirical giant template.

The giant is not a PARSEC star: its flux is an empirical template in Teff, log g and [M/H] with a free scale.
The companion is a sedkit main-sequence star of mass M2 at the Gaia parallax. The two share extinction and
nothing else: no common age and no q <= 1. Optionally the parallax is fitted and the giant's luminosity is
tied to it by a PARSEC M_Ks prior or by bounds on its implied mass, and E by the Edenhofer dust map.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .extinction import extinction_curve
from .fetch import ZERO_JY, TRAINING_SCALE
from .model import StellarModel, HOT_CHANNELS

DATA = Path(__file__).parent / "models" / "giant"
CHANNELS = 66                  # XP61 + J/H/Ks/W1/W2
KS = 63
M2_GRID = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0)
TILT_WAVELENGTH_UM = 0.55
# Initial Nelder-Mead steps of Teff/100, log g, [M/H], E, tilt, ln(scale) and the parallax offset z.
STEP = np.array([0.5, 0.1, 0.1, 0.03, 0.05, 0.02])
STEP_Z = 0.3
# Template nodes within PRIOR_SPAN prior widths of the label priors start inner fits; the N_POLISH best
# are refined in all parameters.
PRIOR_SPAN, N_POLISH = 3.5, 3
LUMINOSITY = ("parsec", "massfree")
# Fitted parallax: plx0 + z sigma with |z| <= Z_MAX and penalty z**2.
Z_MAX = 3.0
MBOL_SUN, LOGG_SUN, TEFF_SUN = 4.74, 4.438, 5772.0
# Implied giant mass: Gaussian walls of MASS_WALL_DEX in log10 M outside MASS_RANGE (solar masses).
MASS_RANGE, MASS_WALL_DEX = (0.2, 10.0), 0.1
# Beyond the PARSEC M_Ks grid the penalty rises as ((distance) / PARSEC_TAIL_MAG)**2.
PARSEC_TAIL_MAG = 0.5
# Dust map: width floor inside MAP_LIMIT_PC; beyond it, a one-sided wall below the value at MAP_EDGE_PC.
MAP_LIMIT_PC, MAP_EDGE_PC = 1250.0, 1200.0
DUST_FLOOR = DUST_WALL = 0.04
DUST_Z_GRID = np.linspace(-Z_MAX, Z_MAX, 21)


class GiantTemplate:
    """Empirical red-giant fluxes from APOGEE DR17 giants with Gaia XP, 2MASS and AllWISE.

    Fluxes are dereddened and normalised to their mean over 0.55--0.95 micron. Nodes lie on a Teff x log g x
    [M/H] grid; values between nodes are trilinear, and a point is supported only when all eight
    neighbouring nodes are. The covariance is fractional: N_BASIS columns plus a diagonal per node.
    """

    def __init__(self):
        with np.load(DATA / "giant_grid.npz", allow_pickle=False) as archive:
            grid = {key: archive[key] for key in archive.files}
        self.summary = json.loads((DATA / "summary.json").read_text())
        self.axes = (grid["teff_ax"], grid["logg_ax"], grid["mh_ax"])
        self.template = grid["template"]
        self.basis = grid["basis"].astype(float)
        self.diag = grid["diag"]
        self.n_eff = grid["n_eff"]
        self.wavelength_um = grid["wavelength_um"]
        self.supported = np.isfinite(self.template[..., :61]).all(-1) & np.isfinite(self.template[..., 63])

    def _corners(self, teff, logg, mh):
        index, weight = [], []
        for axis, value in zip(self.axes, (teff, logg, mh)):
            if not axis[0] <= value <= axis[-1]:
                return None
            i = int(np.clip(np.searchsorted(axis, value) - 1, 0, len(axis) - 2))
            index.append(i)
            weight.append((value - axis[i]) / (axis[i + 1] - axis[i]))
        corners = []
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    w = ((weight[0] if a else 1 - weight[0]) * (weight[1] if b else 1 - weight[1])
                         * (weight[2] if c else 1 - weight[2]))
                    node = (index[0] + a, index[1] + b, index[2] + c)
                    if w > 0:
                        if not self.supported[node]:
                            return None
                        corners.append((node, w))
        return corners

    def in_support(self, teff, logg, mh):
        return self._corners(teff, logg, mh) is not None

    def evaluate(self, teff, logg, mh):
        """Normalised flux (66 channels), fractional covariance columns and diagonal; None outside support.

        Columns of the eight corner nodes are scaled by the square root of their weights, so the covariance
        is the weighted sum of the corner covariances.
        """
        corners = self._corners(teff, logg, mh)
        if corners is None:
            return None
        flux = sum(w * self.template[node] for node, w in corners)
        columns = np.concatenate([np.sqrt(w) * self.basis[node] for node, w in corners], axis=1)
        diag = sum(w * self.diag[node] for node, w in corners)
        return flux, columns, diag


def tilted_transmission(wavelength_um, extinction, tilt=0.0):
    """exp(-E k(lambda) (lambda / 0.55 micron)**tilt) on the ZGR23 curve."""
    wavelength_um = np.asarray(wavelength_um, float)
    return np.exp(-extinction * extinction_curve(wavelength_um) * (wavelength_um / TILT_WAVELENGTH_UM)**tilt)


def _companion(model, m2, age_gyr, scale, wavelength_um):
    """Observed-scale flux and model-error factors of one main-sequence star on the 66 channels.

    The hot route stops at Ks: W1/W2 follow the Rayleigh-Jeans tail of Ks.
    """
    prediction = model.evaluate(m2, 0.0, age_gyr, 0.0)
    if prediction is None:
        raise ValueError(f"companion mass {m2} is unsupported at {age_gyr} Gyr")
    flux = prediction["flux_10pc"][:CHANNELS] * scale
    columns, variance = model.error_factors(prediction, scale)
    columns, variance = columns[:CHANNELS].copy(), variance[:CHANNELS].copy()
    tail = ~np.isfinite(flux[HOT_CHANNELS:])
    if tail.any():
        rj = flux[HOT_CHANNELS - 1] * (wavelength_um[HOT_CHANNELS - 1] / wavelength_um[HOT_CHANNELS:])**4
        flux[HOT_CHANNELS:] = np.where(tail, rj, flux[HOT_CHANNELS:])
        # the Ks fractional error carries over to the extrapolated channels
        ks = variance[HOT_CHANNELS - 1] / flux[HOT_CHANNELS - 1]**2
        variance[HOT_CHANNELS:] = np.where(tail, ks * flux[HOT_CHANNELS:]**2, variance[HOT_CHANNELS:])
    columns[~np.isfinite(columns)] = 0.0
    return dict(flux=flux, columns=columns, variance=variance, teff=float(prediction["teff"][0]),
                radius=float(prediction["radius"][0]), M_G=float(prediction["M_G"][0]))


class _ParsecPrior:
    """PARSEC M_Ks prior on the template grid: -2 ln(p / p_mode) on mks_grid per node, and BC_Ks.

    Flagged nodes have no PARSEC giants in their kernel; their curve is zero and carries no weight.
    """

    def __init__(self):
        with np.load(DATA / "parsec_prior.npz", allow_pickle=False) as archive:
            self.mks_grid = archive["mks_grid"].astype(float)
            self.pen = archive["pen"]
            self.flag = archive["flag"]
            self.bc = archive["bc"].astype(float)


@lru_cache(maxsize=1)
def _parsec_prior():
    return _ParsecPrior()


def _ks_zero_point(model):
    """Flux of Ks = 0 in model units, as `download` converts 2MASS Ks."""
    wavelength_nm = model.wavelength_um[KS] * 1000
    return ZERO_JY[2] * 299792458.0 * 10.0 / wavelength_nm**2 * TRAINING_SCALE[2]


class _Problem:
    """One star's fitted channels, priors and template, with the -2 ln L of giant + companion.

    theta = Teff/100, log g, [M/H], E, tilt, ln(scale). Measurement errors are diagonal; the template and
    companion covariances enter as low-rank columns, with their determinant.
    """

    def __init__(self, y, sigma2, wavelength, mask, template, priors):
        self.y, self.sigma2, self.mask, self.template, self.priors = y, sigma2, mask, template, priors
        self.curve = extinction_curve(wavelength)
        self.log_tilt = np.log(wavelength / TILT_WAVELENGTH_UM)

    def giant(self, teff, logg, mh):
        out = self.template.evaluate(teff, logg, mh)
        if out is None or not np.isfinite(out[0][self.mask]).all():
            return None
        return out[0][self.mask], np.nan_to_num(out[1][self.mask]), out[2][self.mask]

    def likelihood(self, giant, extinction, tilt, ln_scale, companion):
        """-2 ln L, chi2, giant and total observed-scale flux."""
        shape, frac_columns, frac_diag = giant
        attenuation = np.exp(-extinction * self.curve * np.exp(tilt * self.log_tilt))
        g = np.exp(ln_scale) * shape * attenuation
        model = g.copy()
        columns = g[:, None] * frac_columns
        diagonal = self.sigma2 + frac_diag * g**2
        if companion is not None:
            model += companion["flux"] * attenuation
            columns = np.concatenate([columns, companion["columns"] * attenuation[:, None]], axis=1)
            diagonal += companion["variance"] * attenuation**2
        residual = model - self.y
        small = np.eye(columns.shape[1]) + columns.T @ (columns / diagonal[:, None])
        nuisance = np.linalg.solve(small, columns.T @ (residual / diagonal))
        chi2 = float(np.sum((residual - columns @ nuisance)**2 / diagonal) + nuisance @ nuisance)
        return chi2 + float(np.log(diagonal).sum() + np.linalg.slogdet(small)[1]), chi2, g, model

    def penalty(self, theta):
        value = sum(((x - mean) / width)**2 for x, (mean, width)
                    in zip((100 * theta[0], theta[1], theta[2], theta[4]), self.priors["gauss"]))
        if self.priors["extinction"] is not None:
            mean, width = self.priors["extinction"]
            value += ((theta[3] - mean) / width)**2
        return value

    def objective(self, theta, companion, giant=None):
        """-2 ln L plus the Gaussian prior penalties; giant fixes the template (inner fits at a node)."""
        if theta[3] < 0:
            return np.inf
        giant = self.giant(100 * theta[0], theta[1], theta[2]) if giant is None else giant
        if giant is None:
            return np.inf
        return self.likelihood(giant, *theta[3:], companion)[0] + self.penalty(theta)

    def fit(self, nodes, starts, companion):
        """Inner fits of E, tilt and scale at every lattice node, then all six parameters from the best.

        Between template nodes the objective has kinks, and its minima sit at or near nodes; the lattice
        keeps the profile over companion masses from following one local minimum.
        """
        inner = []
        for labels, giant, x0 in nodes:
            f = lambda x: self.objective(np.r_[labels, x], companion, giant)
            r = minimize(f, x0, method="Nelder-Mead",
                         options=dict(initial_simplex=np.vstack([x0, x0 + np.diag(STEP[3:])]),
                                      xatol=1e-3, fatol=0.02, maxfev=400))
            inner.append((r.fun, np.r_[labels, r.x]))
        inner.sort(key=lambda t: t[0])
        best = None
        for x0 in [x for _, x in inner[:N_POLISH]] + list(starts):
            r = minimize(self.objective, x0, args=(companion,), method="Nelder-Mead",
                         options=dict(initial_simplex=np.vstack([x0, x0 + np.diag(STEP)]),
                                      xatol=1e-4, fatol=1e-3, maxfev=3000))
            if best is None or r.fun < best.fun:
                best = r
        return best, inner


class _LuminosityProblem(_Problem):
    """_Problem with the parallax fitted, the giant's luminosity tied to it, and a dust-map prior on E.

    theta = Teff/100, log g, [M/H], E, tilt, ln(scale), z; the parallax plx0 + z sigma places both stars.
    The giant's M_Ks follows from its scale, the template Ks and the parallax; its luminosity from the
    PARSEC BC_Ks, and its mass from log g and Teff. luminosity selects the penalty: "parsec" adds the
    PARSEC M_Ks prior to the mass walls, "massfree" keeps the walls only, None neither (and fixes z = 0).
    dust holds the map E on DUST_Z_GRID, or is None.
    """

    def __init__(self, y, sigma2, wavelength, mask, template, priors, luminosity, parsec, parallax, zp_ks, dust):
        super().__init__(y, sigma2, wavelength, mask, template, priors)
        self.luminosity, self.parsec, self.zp_ks, self.dust = luminosity, parsec, zp_ks, dust
        self.plx0, self.plx_sigma = parallax
        self.step = np.r_[STEP, STEP_Z] if luminosity is not None else STEP

    def theta(self, x):
        """Seven parameters from a fitted vector (z = 0 when the parallax is fixed)."""
        return x if len(x) == 7 else np.r_[x, 0.0]

    def parallax(self, z):
        return self.plx0 + z * self.plx_sigma

    def aux(self, teff, logg, mh):
        """Template Ks, PARSEC curve, BC_Ks and constrained weight at the labels."""
        corners = self.template._corners(teff, logg, mh)
        if corners is None:
            return None
        p = self.parsec
        return dict(ks=sum(w * self.template.template[n][KS] for n, w in corners),
                    curve=sum(w * p.pen[n].astype(float) for n, w in corners),
                    bc=sum(w * p.bc[n] for n, w in corners),
                    constrained=sum(w for n, w in corners if not p.flag[n]))

    def physics(self, theta, aux):
        teff, logg = 100 * theta[0], theta[1]
        plx = self.parallax(theta[6])
        mks = -2.5 * np.log10(np.exp(theta[5]) * aux["ks"] / self.zp_ks) + 5 * np.log10(plx / 100)
        lum = 10**(-0.4 * (mks + aux["bc"] - MBOL_SUN))
        mass = lum * 10**(logg - LOGG_SUN) * (TEFF_SUN / teff)**4
        log_mass = np.log10(mass)
        low, high = np.log10(MASS_RANGE)
        wall = (max(low - log_mass, 0.0, log_mass - high) / MASS_WALL_DEX)**2
        grid = self.parsec.mks_grid
        x = np.clip(mks, grid[0], grid[-1])
        parsec = float(np.interp(x, grid, aux["curve"])) + ((mks - x) / PARSEC_TAIL_MAG)**2 * aux["constrained"]
        applied = {"parsec": parsec + wall, "massfree": wall}.get(self.luminosity, 0.0)
        return dict(parallax_mas=plx, mks=mks, luminosity=lum, mass=mass,
                    radius=np.sqrt(lum) * (TEFF_SUN / teff)**2, parsec_penalty=parsec, mass_penalty=wall,
                    luminosity_penalty=applied)

    def dust_terms(self, theta):
        z = theta[6]
        distance = 1000 / self.parallax(z)
        if self.dust is None:
            return dict(dust_penalty=0.0, e_map=np.nan, e_map_sigma=np.nan, distance_pc=distance)
        d = self.dust
        if distance <= MAP_LIMIT_PC:
            mean, sigma = np.interp(z, DUST_Z_GRID, d["mean"]), np.interp(z, DUST_Z_GRID, d["sigma"])
            penalty = ((theta[3] - mean) / np.hypot(sigma, DUST_FLOOR))**2
            return dict(dust_penalty=penalty, e_map=mean, e_map_sigma=sigma, distance_pc=distance)
        low = d["edge"] - DUST_WALL
        penalty = ((low - theta[3]) / DUST_WALL)**2 if theta[3] < low else 0.0
        return dict(dust_penalty=penalty, e_map=d["edge"], e_map_sigma=np.nan, distance_pc=distance)

    def scaled(self, companion, z):
        if companion is None:
            return None
        k = (self.parallax(z) / self.plx0)**2
        return dict(companion, flux=companion["flux"] * k, columns=companion["columns"] * k,
                    variance=companion["variance"] * k**2)

    def objective(self, x, companion, giant=None, aux=None):
        theta = self.theta(x)
        if theta[3] < 0 or abs(theta[6]) > Z_MAX or self.parallax(theta[6]) <= 0:
            return np.inf
        if giant is None:
            giant = self.giant(100 * theta[0], theta[1], theta[2])
            if giant is None:
                return np.inf
            aux = self.aux(100 * theta[0], theta[1], theta[2])
        value = self.likelihood(giant, *theta[3:6], self.scaled(companion, theta[6]))[0]
        return (value + self.penalty(theta) + theta[6]**2 + self.physics(theta, aux)["luminosity_penalty"]
                + self.dust_terms(theta)["dust_penalty"])

    def fit(self, nodes, starts, companion):
        """As _Problem.fit, with the node's PARSEC terms carried into its inner fit."""
        inner = []
        for labels, giant, aux, x0 in nodes:
            f = lambda x: self.objective(np.r_[labels, x], companion, giant, aux)
            r = minimize(f, x0, method="Nelder-Mead",
                         options=dict(initial_simplex=np.vstack([x0, x0 + np.diag(self.step[3:])]),
                                      xatol=1e-3, fatol=0.02, maxfev=600))
            inner.append((r.fun, np.r_[labels, r.x]))
        inner.sort(key=lambda t: t[0])
        best = None
        for x0 in [x for _, x in inner[:N_POLISH]] + list(starts):
            r = minimize(self.objective, x0, args=(companion,), method="Nelder-Mead",
                         options=dict(initial_simplex=np.vstack([x0, x0 + np.diag(self.step)]),
                                      xatol=1e-4, fatol=1e-3, maxfev=4000))
            if best is None or r.fun < best.fun:
                best = r
        return best, inner


def _dust_table(dust_prior, sed, parallax):
    """Map E mean and width on DUST_Z_GRID and the value at MAP_EDGE_PC along the star's sight line."""
    try:
        ra, dec = float(sed.metadata["ra"]), float(sed.metadata["dec"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("dust_prior needs sed.metadata['ra'] and ['dec']") from None
    plx0, sigma = parallax
    mean, width = [], []
    for z in DUST_Z_GRID:
        plx = plx0 + z * sigma
        distance = 1000 / plx if plx > 0 else np.inf
        m, s = dust_prior.moments(ra, dec, min(distance, MAP_EDGE_PC))
        mean.append(m)
        width.append(s)
    edge = dust_prior.moments(ra, dec, MAP_EDGE_PC)[0]
    return dict(mean=np.array(mean), sigma=np.array(width), edge=edge)


def fit_giant_companion(sed, teff, logg, feh, *, m2_grid=M2_GRID, companion_age_gyr=0.01,
                        extinction_prior=None, tilt_sigma=0.15, use_wise=True, template=None,
                        model=None, threshold=None, luminosity=None, parallax=None, dust_prior=None):
    """Profile the fit of giant + main-sequence companion over a grid of companion masses.

    teff, logg and feh are (mean, sigma) Gaussian priors on the giant's labels, on the template's
    APOGEE scale. The giant has a free flux scale; the companion is StellarModel(hot=True) at
    companion_age_gyr and solar metallicity at the parallax. Both share ZGR23 E >= 0 on a curve tilted
    by (lambda / 0.55 micron)**tilt, with tilt ~ N(0, tilt_sigma); extinction_prior=(mean, sigma) adds a
    Gaussian constraint on E. Channels are XP61 and J/H/Ks, plus W1/W2 with use_wise.

    parallax=(mean, sigma) in mas replaces sed.parallax_mas and sed.parallax_error_mas, e.g. by a
    zero-point-corrected parallax. luminosity="parsec" or "massfree" fits the parallax within Z_MAX
    sigma (penalty z**2) and ties the giant's luminosity to it: its implied mass is held in MASS_RANGE,
    and "parsec" adds the PARSEC M_Ks prior at its labels. dust_prior=EdenhoferPrior adds the map E at
    the trial distance (width sqrt(sigma**2 + DUST_FLOOR**2)); beyond MAP_LIMIT_PC, a one-sided wall
    below the value at MAP_EDGE_PC minus DUST_WALL. It needs sed.metadata["ra"] and ["dec"].

    Returns rows per M2 (M2 = 0 is the giant alone) with the objective -2 ln L + priors and its
    difference from the giant alone (delta > 0: the companion makes the fit worse), its -2 ln L part,
    the best-fitting parameters, the companion's Teff and its share of the observed flux at
    0.40-0.45 micron; with luminosity or dust_prior also the fitted parallax, the giant's M_Ks,
    luminosity, radius and mass, the map E and each penalty. With threshold=t, m2_excluded is the lowest
    grid mass above the best-fitting one whose objective exceeds the profile minimum by t (None if no
    grid mass does). Results are local optima.
    """
    template = GiantTemplate() if template is None else template
    model = StellarModel(hot=True) if model is None else model
    if luminosity is not None and luminosity not in LUMINOSITY:
        raise ValueError(f"luminosity must be None or one of {LUMINOSITY}")
    if dust_prior is not None and extinction_prior is not None:
        raise ValueError("give extinction_prior or dust_prior, not both")
    parallax = (sed.parallax_mas, sed.parallax_error_mas) if parallax is None else tuple(parallax)
    if len(parallax) != 2 or not np.isfinite(parallax[0]) or parallax[0] <= 0:
        raise ValueError("a positive parallax places the companion")
    if luminosity is not None and not (np.isfinite(parallax[1]) and parallax[1] > 0):
        raise ValueError("a fitted parallax needs a positive parallax error")
    for name, prior in (("teff", teff), ("logg", logg), ("feh", feh)):
        if len(prior) != 2 or not np.all(np.isfinite(prior)) or prior[1] <= 0:
            raise ValueError(f"{name} must be (mean, positive sigma)")
    wavelength = model.wavelength_um[:CHANNELS]
    mask = sed.mask[:CHANNELS].copy()
    if not use_wise:
        mask[HOT_CHANNELS:] = False
    if not mask[:61].any() or not mask[KS]:
        raise ValueError("the fit needs XP and Ks")
    y, sigma2, wave = sed.flux[:CHANNELS][mask], sed.error[:CHANNELS][mask]**2, wavelength[mask]
    priors = dict(gauss=[tuple(teff), tuple(logg), tuple(feh), (0.0, tilt_sigma)], extinction=extinction_prior)
    physical = luminosity is not None or dust_prior is not None
    if physical:
        dust = None if dust_prior is None else _dust_table(dust_prior, sed, parallax)
        problem = _LuminosityProblem(y, sigma2, wave, mask, template, priors, luminosity,
                                     _parsec_prior(), parallax, _ks_zero_point(model), dust)
    else:
        problem = _Problem(y, sigma2, wave, mask, template, priors)
    scale = (parallax[0] / 100)**2
    companions = {}
    for m2 in m2_grid:
        c = _companion(model, m2, companion_age_gyr, scale, wavelength)
        companions[m2] = dict(c, **{key: c[key][mask] for key in ("flux", "columns", "variance")})

    lattice = [axis[np.abs(axis - mean) <= PRIOR_SPAN * width]
               for axis, (mean, width) in zip(template.axes, (teff, logg, feh))]
    nodes = []
    for t in lattice[0]:
        for g in lattice[1]:
            for m in lattice[2]:
                giant = problem.giant(t, g, m)
                if giant is not None:
                    extra = (problem.aux(t, g, m),) if physical else ()
                    nodes.append((np.array([t / 100, g, m]), giant) + extra)
    if not nodes:
        raise ValueError("no giant template node lies within the label priors")

    # E and scale start from a scan with a 3 per cent floor, per node; z starts at the catalogue parallax
    z0 = (0.0,) if luminosity is not None else ()
    starts = []
    for node in nodes:
        shape = node[1][0]
        best = (np.inf, 0.0, 1.0)
        for e in np.arange(0, 3.0, 0.05):
            m = shape * np.exp(-e * problem.curve)
            s = np.sum(m * y / sigma2) / np.sum(m * m / sigma2)
            chi = np.sum((y - s * m)**2 / (sigma2 + (0.03 * s * m)**2))
            best = min(best, (chi, e, s))
        starts.append(np.array([best[1], 0.0, np.log(best[2]), *z0]))
    alone, inner = problem.fit([node + (s0,) for node, s0 in zip(nodes, starts)], [], None)
    # companion fits start each node from its giant-alone solution
    node_starts = {tuple(x[:3]): x[3:] for _, x in inner}
    seeded = [node + (node_starts[tuple(node[0])],) for node in nodes]
    rows = [dict(m2=0.0, theta=alone.x, objective=float(alone.fun), companion=None)]
    previous = alone.x
    for m2 in m2_grid:
        result, _ = problem.fit(seeded, [previous, alone.x], companions[m2])
        rows.append(dict(m2=m2, theta=result.x, objective=float(result.fun), companion=companions[m2]))
        previous = result.x

    blue = (wave >= 0.40) & (wave <= 0.45)
    table = []
    for row in rows:
        th, c = row["theta"], row["companion"]
        labels = (100 * th[0], th[1], th[2])
        if physical:
            th = problem.theta(th)
            c = problem.scaled(c, th[6])
        minus2lnl, chi2, giant, total = problem.likelihood(problem.giant(*labels), *th[3:6], c)
        out = dict(
            m2=row["m2"], objective=row["objective"], delta=row["objective"] - rows[0]["objective"],
            minus2lnL=minus2lnl, chi2=chi2, teff=labels[0], logg=th[1], feh=th[2], extinction_e=th[3],
            tilt=th[4], scale=float(np.exp(th[5])),
            m2_teff=np.nan if c is None else c["teff"], m2_radius=np.nan if c is None else c["radius"],
            companion_blue_fraction=0.0 if c is None else float(1 - giant[blue].sum() / total[blue].sum()))
        if physical:
            ph = problem.physics(th, problem.aux(*labels))
            du = problem.dust_terms(th)
            out.update(label_penalty=problem.penalty(th), z=th[6], parallax_mas=ph["parallax_mas"],
                       distance_pc=du["distance_pc"], mks=ph["mks"], luminosity=ph["luminosity"],
                       radius=ph["radius"], mass=ph["mass"], luminosity_penalty=ph["luminosity_penalty"],
                       parsec_penalty=ph["parsec_penalty"], mass_penalty=ph["mass_penalty"],
                       dust_penalty=du["dust_penalty"], e_map=du["e_map"], e_map_sigma=du["e_map_sigma"])
        table.append(out)
    objective = np.array([r["objective"] for r in table])
    best = int(np.argmin(objective))
    excluded = None
    if threshold is not None:
        for r in table[best + 1:]:
            if r["objective"] - objective[best] > threshold:
                excluded = r["m2"]
                break
    return dict(rows=table, n_channels=int(mask.sum()), best_m2=table[best]["m2"],
                detection=float(objective[0] - objective[best]), m2_excluded=excluded, threshold=threshold,
                luminosity=luminosity, parallax=parallax, mask=mask, wavelength_um=wavelength)
