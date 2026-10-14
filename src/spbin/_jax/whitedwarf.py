"""JAX port of the white-dwarf frame: DA and dwarf components, likelihood and fit.

`fit_whitedwarf_companion` has the signature and result fields of the numpy
version. Each hypothesis is one compiled objective on a fixed data layout:
the 168 channels, six blue XP points (332-382 nm) and GALEX FUV/NUV, with 0/1
weights for the slots a star uses, so one compile serves every star of a
configuration (hypothesis kind, free parameters, free radius, hot stellar
route). The dwarf (or coeval pair as one component with fully correlated
model errors) contributes the 12 low-rank error columns of `stellar`; the DA
contributes a diagonal fractional error. Model errors are taken at the
observed flux and shared by light fraction, as in numpy.

The search follows numpy: the seed grid is scored in one `vmap`, the best
three seeds are refined by a scipy-semantics Nelder-Mead (maxfev 4000), and
one restart runs from the best endpoint with the initial steps. The result
dict is rebuilt with the numpy model at the best endpoint numpy supports, so
the reported objective is numpy's. A dust prior is pre-sampled on a parallax
grid per star, as in `_jax.fit`.
"""

import weakref
from functools import lru_cache, partial
from itertools import product
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..extinction import extinction_curve
from ..model import StellarModel
from ..subdwarf import H_C_OVER_K_NM
from ..whitedwarf import G_CGS, MSUN_G, RSUN_CM, WhiteDwarfModel, _Hypothesis, _prepare
from . import likelihood, stellar
from .arrays import ModelArrays
from .fit import _dust_grid, _model_arrays, _nelder_mead

N_CHANNELS = 168
BLUE_SLOTS = np.arange(168, 174)
GALEX_SLOTS = dict(FUV=174, NUV=175)
N_DATA = 176
# Parameter slots of the full vector; a hypothesis fits a subset.
SLOTS = ("t", "m", "g", "r", "primary", "age", "feh", "q", "e", "z")
STEPS = dict(t=.08, m=.05, g=.1, r=.05, primary=.05, age=.2, feh=.1, q=.1, e=.02, z=.2)
WD_LABELS = ("teff", "mass", "logg", "radius")
DWARF_LABELS = ("teff", "mass", "logg", "radius", "age_gyr", "feh")
UV_TEFF_K = (6812., 41430.)  # empirical GALEX calibration range of the DA model
MAXFEV, XATOL, FATOL = 4000, 1e-5, 1e-4


class WDArrays(eqx.Module):
    """WhiteDwarfModel tables, calibration and wavelength maps as a pytree.

    calibration=None and calibrations without the Teff spline or UV terms are
    zero-filled, which reproduces numpy's skipped branches exactly; the
    default errors (0.03 channel, 0.05 blue and UV) are filled the same way.
    """

    teff_ax: jax.Array  # (78,) Koester grid
    logg_ax: jax.Array  # (11,)
    ln_flux: jax.Array  # (78, 11, 168), NaN on unsupported channels
    ln_blue: jax.Array  # (78, 11, 6)
    ln_coarse: jax.Array  # (78, 11, 486)
    balmer_w: jax.Array  # (78, 11)
    cool_teff_ax: jax.Array  # cooling table, (78,)
    cool_mass_ax: jax.Array  # (23,)
    cool_radius: jax.Array  # (78, 23), NaN beyond the cooling sequence
    cool_age: jax.Array  # (78, 23)
    a: jax.Array  # (168,) calibration offset
    b: jax.Array  # (168,) Balmer-index term
    blue_a: jax.Array  # (6,)
    blue_b: jax.Array  # (6,)
    knots: jax.Array  # (k,) log Teff knots
    teff_flux: jax.Array  # (k, 168)
    teff_blue: jax.Array  # (k, 6)
    logg_flux: jax.Array  # (168,)
    logg_blue: jax.Array  # (6,)
    diag: jax.Array  # (168,) fractional channel error
    blue_diag: jax.Array  # (6,)
    uv_a: jax.Array  # (2,) FUV, NUV passband factors (log)
    uv_diag: jax.Array  # (2,)
    xp_nm: jax.Array  # (61,) XP channel wavelengths
    anchor_power: jax.Array  # (2,) lambda**-5 of the first and last XP channel, as numpy rounds it
    blue_nm: jax.Array  # (6,)
    coarse_nm: jax.Array  # (486,)
    correction_nm: jax.Array  # (67,) blue then XP: the coarse-correction nodes
    curve: jax.Array  # (168,) ZGR23 optical depth per E
    coarse_curve: jax.Array  # (486,)
    blue_curve: jax.Array  # (6,)
    passbands: jax.Array  # (5, 486) G, BP, RP, FUV, NUV weights on the coarse grid

    @classmethod
    def build(cls, model):
        t, c, cal = model.table, model.cooling, model.calibration
        cal = {} if cal is None else cal
        has_teff = "teff_knots" in cal
        knots = cal["teff_knots"] if has_teff else np.array([6000., 80000.])
        zeros168, zeros6 = np.zeros(N_CHANNELS), np.zeros(6)
        values = dict(
            a=cal.get("a", zeros168), b=cal.get("b", zeros168),
            blue_a=cal.get("blue_a", zeros6), blue_b=cal.get("blue_b", zeros6),
            knots=np.log(knots),
            teff_flux=cal["teff_flux"] if has_teff else np.zeros((2, N_CHANNELS)),
            teff_blue=cal["teff_blue"] if has_teff else np.zeros((2, 6)),
            logg_flux=cal["logg_flux"] if has_teff else zeros168,
            logg_blue=cal["logg_blue"] if has_teff else zeros6,
            diag=cal["diag"] if model.calibration is not None else np.full(N_CHANNELS, .03),
            blue_diag=cal.get("blue_diag", np.full(6, .05)),
            uv_a=cal.get("uv_a", np.zeros(2)), uv_diag=cal.get("uv_diag", np.full(2, .05)))
        xp_nm = model.wavelength_um[:61] * 1000
        return cls(
            teff_ax=t["teff_ax"], logg_ax=t["logg_ax"], ln_flux=t["ln_flux"],
            ln_blue=t["ln_blue"], ln_coarse=t["ln_coarse"], balmer_w=t["balmer_w"],
            cool_teff_ax=c["teff_ax"], cool_mass_ax=c["mass_ax"],
            cool_radius=c["radius"], cool_age=c["age_gyr"],
            # The stored wavelengths are float32, and numpy's Planck anchor
            # takes lambda**-5 in that precision.
            xp_nm=xp_nm, anchor_power=np.array([float(xp_nm[0]**-5), float(xp_nm[60]**-5)]), blue_nm=model.blue_wavelength_nm, coarse_nm=model.coarse_wavelength_nm,
            correction_nm=np.r_[model.blue_wavelength_nm, xp_nm],
            curve=extinction_curve(model.wavelength_um), coarse_curve=model.coarse_curve,
            blue_curve=model.blue_curve,
            passbands=np.stack([model.passbands[k] for k in ("G", "BP", "RP", "FUV", "NUV")]),
            **{k: np.asarray(v, float) for k, v in values.items()})


def _as_jax(module):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, float), module)


def planck(wave_nm, teff):
    x = jnp.minimum(H_C_OVER_K_NM / (wave_nm * teff), 700.0)
    return wave_nm**-5 / jnp.expm1(x)


def physical(wd, teff, mass):
    """Cooling radius, log g and age; valid False off the table or its reach."""
    inside = ((teff >= wd.cool_teff_ax[0]) & (teff <= wd.cool_teff_ax[-1])
              & (mass >= wd.cool_mass_ax[0]) & (mass <= wd.cool_mass_ax[-1]))
    radius = stellar.bilinear(wd.cool_teff_ax, wd.cool_mass_ax, wd.cool_radius, teff, mass)
    age = stellar.bilinear(wd.cool_teff_ax, wd.cool_mass_ax, wd.cool_age, teff, mass)
    valid = inside & jnp.isfinite(radius) & (radius > 0) & jnp.isfinite(age)
    radius = jnp.where(valid, radius, 0.013)
    logg = jnp.log10(G_CGS * jnp.where(valid, mass, 0.6) * MSUN_G / (radius * RSUN_CM)**2)
    return radius, logg, jnp.where(valid, age, 0.0), valid


def predict(wd, teff, logg, radius):
    """Calibrated absolute 10-pc fluxes: 168 channels, 6 blue, 486 coarse.

    Bilinear in (Teff, log g) on the Koester grid, then the calibration
    a + W b plus the piecewise-linear log Teff spline and the log g term,
    interpolated onto the coarse grid (zero beyond the blue and XP nodes).
    """
    valid = ((teff >= wd.teff_ax[0]) & (teff <= wd.teff_ax[-1])
             & (logg >= wd.logg_ax[0]) & (logg <= wd.logg_ax[-1])
             & jnp.isfinite(radius) & (radius > 0))
    ln_flux = stellar.bilinear(wd.teff_ax, wd.logg_ax, wd.ln_flux, teff, logg)
    ln_blue = stellar.bilinear(wd.teff_ax, wd.logg_ax, wd.ln_blue, teff, logg)
    ln_coarse = stellar.bilinear(wd.teff_ax, wd.logg_ax, wd.ln_coarse, teff, logg)
    w = stellar.bilinear(wd.teff_ax, wd.logg_ax, wd.balmer_w, teff, logg)
    correction = wd.a + w * wd.b
    blue_correction = wd.blue_a + w * wd.blue_b
    knots = wd.knots
    x = jnp.clip(jnp.log(teff), knots[0], knots[-1])
    i = jnp.minimum(jnp.searchsorted(knots, x, side="right") - 1, knots.shape[0] - 2)
    fraction = (x - knots[i]) / (knots[i + 1] - knots[i])
    correction = correction + ((1 - fraction) * wd.teff_flux[i] + fraction * wd.teff_flux[i + 1]
                               + (logg - 8.) * wd.logg_flux)
    blue_correction = blue_correction + ((1 - fraction) * wd.teff_blue[i]
                                         + fraction * wd.teff_blue[i + 1]
                                         + (logg - 8.) * wd.logg_blue)
    coarse_correction = jnp.interp(wd.coarse_nm, wd.correction_nm,
                                   jnp.concatenate([blue_correction, correction[:61]]),
                                   left=0.0, right=0.0)
    square = radius**2
    return (jnp.exp(ln_flux + correction) * square, jnp.exp(ln_blue + blue_correction) * square,
            jnp.exp(ln_coarse + coarse_correction) * square, w, valid)


class Problem(eqx.Module):
    """One star's fixed data on the 176-slot layout, nuisances and priors."""

    stellar: ModelArrays
    wd: WDArrays
    y: jax.Array  # (176,) observed fluxes; 0 in unused slots
    variance: jax.Array  # (176,) observed variances; 1 in unused slots
    weight: jax.Array  # (176,) 1 for slots in the data vector
    reference: jax.Array  # (176,) max(y, sigma): the flux model errors are taken at
    parallax: jax.Array
    parallax_error: jax.Array
    template: jax.Array  # (10,) values of the slots a hypothesis does not fit
    age: jax.Array  # fixed companion age (Gyr) when the age is not fitted
    extinction_prior: jax.Array  # (3,) mean, sigma, flag
    dust_par: jax.Array  # dust prior sampled over the reachable parallaxes
    dust_mean: jax.Array
    dust_sigma: jax.Array
    dust_fail: jax.Array
    dust_flag: jax.Array
    wd_prior: jax.Array  # (3, 4) mean, sigma, flag for teff, mass, logg, radius
    companion_prior: jax.Array  # (3, 6) the same for teff, mass, logg, radius, age_gyr, feh


class Spec(NamedTuple):
    """Static structure of one hypothesis: one compile per distinct spec."""

    kind: str
    free_radius: bool
    names: tuple  # fitted slots, in numpy's order
    age_free: bool
    fit_parallax: bool


class Component(NamedTuple):
    flux: jax.Array  # (176,) observed-scale model on the data layout
    variance: jax.Array  # (176,) diagonal model variance before the reference scaling
    full: jax.Array  # (168,) reddened channel fluxes
    coarse: jax.Array  # (486,) reddened coarse spectrum
    labels: jax.Array  # WD_LABELS or DWARF_LABELS
    valid: jax.Array


def dwarf(problem, mass, age, feh, scale, e, q=0.0):
    """One dwarf, or a coeval pair (q > 0) as one component with fully
    correlated model errors; also returns its (176, 12) error columns.

    The coarse spectrum interpolates the XP channels in log flux and
    continues below 392 nm and above 992 nm as a Planck curve at the primary's
    Teff, anchored on the first and last XP channels.
    """
    wd = problem.wd
    pred = stellar.evaluate(problem.stellar, mass, q, age, feh)
    teff = pred.labels[0, 0]
    full = pred.flux_10pc * scale
    attenuation = jnp.exp(-e * wd.curve)
    columns, variance = stellar.error_factors(problem.stellar, teff, pred.flux_10pc, scale)
    columns, variance = columns * attenuation[:, None], variance * attenuation**2
    safe_teff = jnp.where(pred.valid, teff, 5000.0)
    coarse = jnp.exp(jnp.interp(wd.coarse_nm, wd.xp_nm, jnp.log(jnp.maximum(full[:61], 1e-300))))
    bb = planck(wd.coarse_nm, safe_teff)
    anchor = wd.anchor_power / jnp.expm1(jnp.minimum(
        H_C_OVER_K_NM / (wd.xp_nm[jnp.array([0, 60])] * safe_teff), 700.0))
    coarse = jnp.where(wd.coarse_nm < 392, bb * full[0] / anchor[0],
                       jnp.where(wd.coarse_nm > 992, bb * full[60] / anchor[1], coarse))
    coarse = coarse * jnp.exp(-e * wd.coarse_curve)
    extra = jnp.concatenate([jnp.interp(wd.blue_nm, wd.coarse_nm, coarse), wd.passbands[3:] @ coarse])
    labels = jnp.stack([teff, mass, pred.logg[0], pred.radius[0], age, feh])
    component = Component(
        flux=jnp.concatenate([full * attenuation, extra]),
        variance=jnp.concatenate([variance, (.5 * extra)**2]),
        full=full * attenuation, coarse=coarse, labels=labels, valid=pred.valid)
    return component, jnp.concatenate([columns, jnp.zeros((N_DATA - N_CHANNELS, columns.shape[1]))])


def white_dwarf(problem, teff, mass, radius, logg, scale, e, free_radius):
    """A DA at (Teff, mass) on its cooling radius, or at free (Teff, log g, radius)."""
    wd = problem.wd
    if free_radius:
        mass = 10**logg * (radius * RSUN_CM)**2 / G_CGS / MSUN_G
        physical_valid = jnp.asarray(True)
    else:
        radius, logg, _, physical_valid = physical(wd, teff, mass)
    flux, blue, coarse, _, valid = predict(wd, teff, logg, radius)
    full = flux * scale * jnp.exp(-e * wd.curve)
    coarse = coarse * scale * jnp.exp(-e * wd.coarse_curve)
    blue = blue * scale * jnp.exp(-e * wd.blue_curve)
    galex = (wd.passbands[3:] @ coarse) * jnp.exp(wd.uv_a)
    in_uv = (teff >= UV_TEFF_K[0]) & (teff <= UV_TEFF_K[1])
    galex_error = jnp.where(in_uv, wd.uv_diag, jnp.maximum(wd.uv_diag, .5))
    return Component(
        flux=jnp.concatenate([full, blue, galex]),
        variance=jnp.concatenate([(full * wd.diag)**2, (blue * wd.blue_diag)**2,
                                  (galex * galex_error)**2]),
        full=full, coarse=coarse, labels=jnp.stack([teff, mass, logg, radius]),
        valid=physical_valid & valid)


def neg2_log_likelihood(problem, flux, columns, variance):
    """-2 ln L (no n ln 2pi) and chi2 with model errors at the observed flux.

    Unused slots get unit diagonal, zero residual and zero columns, so they
    add nothing; NaN model flux there (unsupported DA channels) is never read.
    """
    observed = problem.weight > 0
    ratio = problem.reference / jnp.maximum(flux, 1e-300)
    diagonal = jnp.where(observed, problem.variance + variance * ratio**2, 1.0)
    residual = jnp.where(observed, flux - problem.y, 0.0)
    columns = jnp.where(observed[:, None], columns * ratio[:, None], 0.0)
    small = jnp.eye(columns.shape[1]) + columns.T @ (columns / diagonal[:, None])
    nuisance = jnp.linalg.solve(small, columns.T @ (residual / diagonal))
    chi2 = jnp.sum((residual - columns @ nuisance)**2 / diagonal) + nuisance @ nuisance
    value = chi2 + jnp.sum(jnp.where(observed, jnp.log(diagonal), 0.0)) + jnp.linalg.slogdet(small)[1]
    return value, chi2


def _prior(labels, prior):
    return jnp.sum(prior[2] * ((labels - prior[0]) / prior[1])**2)


def _evaluate(theta, problem, spec):
    """Total flux, columns, variance, penalty and validity at a full parameter vector."""
    v = dict(zip(SLOTS, theta))
    z = v["z"] if spec.fit_parallax else 0.0
    parallax = problem.parallax + z * problem.parallax_error if spec.fit_parallax else problem.parallax
    scale = (parallax / 100)**2
    e = v["e"]
    flux = jnp.zeros(N_DATA)
    variance = jnp.zeros(N_DATA)
    columns = jnp.zeros((N_DATA, sum(problem.stellar.error_columns)))
    valid = jnp.asarray(True)
    penalty = z * z
    if spec.kind in ("wd", "wd+dwarf"):
        teff = jnp.clip(jnp.exp(v["t"]), 6000, 80000)
        wd = white_dwarf(problem, teff, v["m"], jnp.exp(v["r"]), v["g"], scale, e, spec.free_radius)
        flux, variance, valid = flux + wd.flux, variance + wd.variance, valid & wd.valid
        penalty += _prior(wd.labels, problem.wd_prior)
    if spec.kind != "wd":
        age = jnp.clip(jnp.exp(v["age"]), .5, 10.) if spec.age_free else problem.age
        q = v["q"] if spec.kind == "ms+ms" else 0.0
        companion, columns = dwarf(problem, jnp.exp(v["primary"]), age, v["feh"], scale, e, q)
        flux, variance, valid = flux + companion.flux, variance + companion.variance, valid & companion.valid
        penalty += _prior(companion.labels, problem.companion_prior)
    mean, sigma, flag = problem.extinction_prior
    penalty += flag * ((e - mean) / sigma)**2
    penalty += problem.dust_flag * likelihood.dust_penalty(
        e, parallax, problem.dust_par, problem.dust_mean, problem.dust_sigma, problem.dust_fail)
    return flux, columns, variance, penalty, valid


def objective(theta, problem, spec):
    """numpy `_Hypothesis.objective` at a full parameter vector; 1e30 if unsupported."""
    flux, columns, variance, penalty, valid = _evaluate(theta, problem, spec)
    value, _ = neg2_log_likelihood(problem, flux, columns, variance)
    total = value + penalty
    return jnp.where(valid & jnp.isfinite(total), total, 1e30)


@partial(jax.jit, static_argnames="spec")
def _scores(thetas, problem, spec):
    return jax.vmap(lambda theta: objective(theta, problem, spec))(thetas)


@partial(jax.jit, static_argnames="spec")
def _refine(starts, steps, bounds, problem, spec):
    """Nelder-Mead from each start, then once more from the best endpoint.

    The starts run one at a time (lax.map): under vmap every branch of the
    step would be evaluated and the batch would wait for the slowest start.
    Returns all final vertices, their values and each run's converged flag.
    """
    index = jnp.array([SLOTS.index(name) for name in spec.names])

    def free_objective(u):
        return objective(problem.template.at[index].set(u), problem, spec)

    def run(x0):
        simplex = jnp.concatenate([x0[None], x0 + jnp.diag(steps)])
        return _nelder_mead(free_objective, simplex, bounds, maxiter=MAXFEV, xatol=XATOL,
                            fatol=FATOL, maxfev=MAXFEV)

    vertices, values, done = jax.lax.map(run, starts)
    flat = jnp.where(values < 1e29, values, jnp.inf).ravel()
    restart_vertices, restart_values, restart_done = run(
        vertices.reshape(-1, len(spec.names))[jnp.argmin(flat)])
    return (jnp.concatenate([vertices, restart_vertices[None]]),
            jnp.concatenate([values, restart_values[None]]),
            jnp.concatenate([done, restart_done[None]]))


_wd_cache = weakref.WeakKeyDictionary()


def _wd_arrays(model):
    if model not in _wd_cache:
        _wd_cache[model] = _as_jax(WDArrays.build(model))
    return _wd_cache[model]


@lru_cache(maxsize=1)
def _default_models():
    return WhiteDwarfModel(), StellarModel()


def _prior_rows(prior, keys):
    rows = np.array([[1.0, 1.0, 0.0]] * len(keys))
    for k, (mean, sigma) in prior.items():
        rows[keys.index(k)] = mean, sigma, 1.0
    return rows.T


def build_problem(p, age):
    """The compiled frame's data for a numpy `_Problem` (from `_prepare`)."""
    slots = list(p.idx) + (list(BLUE_SLOTS) if p.blue else []) + [GALEX_SLOTS[b] for b in p.galex]
    y, variance, weight, reference = np.zeros(N_DATA), np.ones(N_DATA), np.zeros(N_DATA), np.ones(N_DATA)
    y[slots], variance[slots], weight[slots], reference[slots] = p.y, p.variance, 1.0, p.reference
    if p.dust is None:
        dust = np.ones(1), np.zeros(1), np.ones(1), np.zeros(1)
    elif p.fit_parallax:
        dust = _dust_grid(p.dust, p.sed.metadata["ra"], p.sed.metadata["dec"], p.parallax[0],
                          p.parallax[1], (max(-3., -p.parallax[0] / p.parallax[1] + 1e-5), 3.))
    else:
        # A fixed parallax evaluates the penalty at one distance.
        mean, sigma = p.dust.moments(p.sed.metadata["ra"], p.sed.metadata["dec"], 1000 / p.parallax[0])
        dust = np.array([p.parallax[0]]), np.array([mean]), np.array([sigma]), np.zeros(1)
    prior = p.extinction_prior
    return Problem(
        stellar=_model_arrays(p.stellar), wd=_wd_arrays(p.wd),
        y=jnp.asarray(y), variance=jnp.asarray(variance), weight=jnp.asarray(weight),
        reference=jnp.asarray(reference),
        parallax=jnp.asarray(float(p.parallax[0])),
        parallax_error=jnp.asarray(float(p.parallax[1]) if p.fit_parallax else 0.0),
        template=jnp.zeros(len(SLOTS)), age=jnp.asarray(5.0 if age is None else float(age)),
        extinction_prior=jnp.asarray([0.0, 1.0, 0.0] if prior is None else [prior[0], prior[1], 1.0]),
        dust_par=jnp.asarray(dust[0]), dust_mean=jnp.asarray(dust[1]),
        dust_sigma=jnp.asarray(dust[2]), dust_fail=jnp.asarray(dust[3]),
        dust_flag=jnp.asarray(float(p.dust is not None)),
        wd_prior=jnp.asarray(_prior_rows(p.priors[0], WD_LABELS)),
        companion_prior=jnp.asarray(_prior_rows(p.priors[1], DWARF_LABELS)))


def _template(hyp):
    """Values of the slots the hypothesis does not fit, as numpy's components() reads them."""
    p = hyp.p
    values = dict(t=np.log(15000.), m=.6, g=8., r=np.log(.013), primary=np.log(.8), age=np.log(5.),
                  feh=0. if hyp.feh is None else hyp.feh, q=.6,
                  e=0. if p.extinction is None else p.extinction, z=0.)
    values.update(hyp.fixed)
    return np.array([values[k] for k in SLOTS], float)


def spec_of(hyp):
    return Spec(hyp.kind, hyp.free_radius, tuple(hyp.names), "age" in hyp.bounds, hyp.p.fit_parallax)


def _seeds(hyp, initial):
    """numpy `_Hypothesis.fit` seeds, as dicts over the slots."""
    p = hyp.p
    base = dict(t=np.log(15000.), m=.6, g=8., r=np.log(.013), primary=np.log(.8), age=np.log(5.),
                feh=0., q=.6, e=p.extinction_prior[0] if p.extinction_prior is not None else .01, z=0.)
    if p.dust is not None:
        base["e"] = p.dust.moments(p.sed.metadata["ra"], p.sed.metadata["dec"], 1000 / p.parallax[0])[0]
    if "feh" in p.priors[1]:
        base["feh"] = np.clip(p.priors[1]["feh"][0], -1., .5)
    temperatures = [8000, 11000, 15000, 22000, 35000, 60000] if "t" in hyp.names else [15000]
    masses = [.2, .4, .7, 1., 1.3] if "primary" in hyp.names else [.8]
    ages = [5.]
    if hyp.kind in ("dwarf", "ms+ms") and hyp.age is None:
        masses = np.linspace(.1, 2.2, 22)
        ages = [1., 2., 5., 8., 9.8]
    ratios = [.3, .6, .9, 1.] if "q" in hyp.names else [.6]
    seeds = []
    if initial is not None:
        # A companion adds light: also start from less massive primaries.
        primary = {**base, **initial}["primary"]
        seeds.extend({**base, **initial, "t": np.log(t), "q": q, "primary": max(primary + s, np.log(.1))}
                     for t in temperatures for q in ratios for s in np.log([1., .97, .93, .87, .8]))
    for t, mass, age, q in product(temperatures, masses, ages, ratios):
        v = dict(base, t=np.log(t), primary=np.log(mass), age=np.log(age), q=q)
        if hyp.free_radius and "t" in hyp.names:
            wd = p.white_dwarf(t, .6, .013, 8., 1, 0)
            if wd is not None:
                ratio = np.median(p.y[:len(p.idx)] / wd["flux"][:len(p.idx)])
                if ratio > 0:
                    v["r"] = np.clip(np.log(.013 * np.sqrt(ratio) / (p.parallax[0] / 100)), *hyp.bounds["r"])
        seeds.append(v)
    return seeds


def fit_hypothesis(hyp, problem, initial=None):
    """Compiled counterpart of numpy `_Hypothesis.fit`: (summary, best values)."""
    spec = spec_of(hyp)
    template = _template(hyp)
    problem = eqx.tree_at(lambda pr: pr.template, problem, jnp.asarray(template))
    free = [SLOTS.index(k) for k in hyp.names]
    seeds = _seeds(hyp, initial)
    thetas = np.tile(template, (len(seeds), 1))
    thetas[:, free] = [[v[k] for k in hyp.names] for v in seeds]
    scores = np.asarray(_scores(jnp.asarray(thetas), problem, spec))
    order = np.argsort(scores, kind="stable")[:3]
    order = order[scores[order] < 1e29]
    if not len(order):
        return dict(objective=np.inf, converged=False, hypothesis=hyp.kind), None
    starts = thetas[order][:, free]
    bounds = np.array([hyp.bounds[k] for k in hyp.names])
    vertices, values, done = _refine(jnp.asarray(starts), jnp.asarray([STEPS[k] for k in hyp.names]),
                                     jnp.asarray(bounds), problem, spec)
    n = len(hyp.names)
    # Candidates: every final vertex (the others lie within the termination
    # tolerances of the best) and the starts as a supported fallback.
    candidates = np.concatenate([np.asarray(vertices).reshape(-1, n), starts])
    values = np.concatenate([np.asarray(values).ravel(), scores[order]])
    converged = np.concatenate([np.repeat(np.asarray(done), n + 1), np.zeros(len(starts), bool)])
    values = np.where(np.isfinite(values) & (values < 1e29), values, np.inf)
    # An endpoint pressed against a domain edge can be supported by the
    # compiled model and not by numpy, which builds the report.
    for best in np.argsort(values, kind="stable"):
        if not np.isfinite(values[best]):
            break
        x = candidates[best]
        result = hyp.components(hyp.unpack(x))
        if result is not None:
            return _summary(hyp, x, result, bool(converged[best]))
    return dict(objective=np.inf, converged=False, hypothesis=hyp.kind), None


def _summary(hyp, x, result, converged):
    """numpy `_Hypothesis.fit`'s result at x, with numpy's objective."""
    p = hyp.p
    v = hyp.unpack(x)
    comps, wd, dwarf_, e, plx, penalty = result
    value, chi2 = p.likelihood(comps)
    summary = dict(hypothesis=hyp.kind, objective=float(value + penalty), minus2lnL=value, chi2=chi2,
                   converged=converged, extinction_e=e, parallax_mas=plx, penalty=penalty,
                   model=sum(c["flux"] for c in comps), n_fit=len(p.y),
                   components={"whitedwarf": None if wd is None else wd["full"],
                               "dwarf": None if dwarf_ is None else dwarf_["full"]})
    if wd is not None:
        summary["whitedwarf"] = wd["labels"]
        total = sum(c["coarse"] for c in comps)
        summary["fractions"] = {f"beta_{b}": p.wd.passband(wd["coarse"], b) / p.wd.passband(total, b)
                                for b in ("G", "BP", "RP")}
    if dwarf_ is not None:
        summary["companion"] = dwarf_["labels"]
    if wd is not None and dwarf_ is not None:
        b = wd["labels"]["mass"] / (wd["labels"]["mass"] + dwarf_["labels"]["mass"])
        summary["photocentre"] = dict(star1="whitedwarf", star2="dwarf",
                                      mass_fraction_star1=b, light_fraction_star1=summary["fractions"]["beta_G"],
                                      coefficient=b - summary["fractions"]["beta_G"])
    return summary, {k: float(val) for k, val in v.items()}


def fit_whitedwarf_companion(sed, *, model=None, stellar=None, companions=True, free_radius=False,
                             companion_age_gyr=5., companion_feh=0., extinction=0., extinction_prior=None,
                             dust_prior=None, parallax=None, fit_parallax=False, blue=None, galex=None,
                             use_spherex=False, whitedwarf_prior=None, companion_prior=None,
                             hypotheses=None):
    """Compare one dwarf, a coeval dwarf pair, one DA WD and DA+dwarf on identical data.

    Same arguments and result as `spbin.whitedwarf.fit_whitedwarf_companion`,
    on the compiled frame. Default models are built once per process; pass the
    same `model`/`stellar` instances across stars to reuse their arrays.
    """
    kinds = (tuple(hypotheses) if hypotheses is not None
             else ("dwarf", "wd", "wd+dwarf") if companions else ("dwarf", "wd"))
    unknown = set(kinds) - {"dwarf", "ms+ms", "wd", "wd+dwarf"}
    if unknown or not kinds:
        raise ValueError(f"unknown hypotheses: {sorted(unknown)}")
    default_wd, default_stellar = _default_models()
    p = _prepare(sed, default_wd if model is None else model, default_stellar if stellar is None else stellar,
                 extinction, extinction_prior, dust_prior, parallax, fit_parallax, blue, galex, use_spherex,
                 whitedwarf_prior, companion_prior)
    problem = build_problem(p, companion_age_gyr)
    results = {}
    primary_seed = None
    if "dwarf" in kinds:
        kinds = ("dwarf",) + tuple(k for k in kinds if k != "dwarf")
    for kind in kinds:
        hyp = _Hypothesis(p, kind, free_radius, companion_age_gyr, companion_feh)
        results[kind], values = fit_hypothesis(
            hyp, problem, initial=primary_seed if kind in ("ms+ms", "wd+dwarf") else None)
        if kind == "dwarf":
            primary_seed = values
    best = min(results, key=lambda k: results[k]["objective"])
    if not np.isfinite(results[best]["objective"]):
        raise ValueError("no supported hypothesis could be fitted")
    return dict(hypotheses=results, preferred=best,
                detection_statistic=(min(results[k]["objective"] for k in ("dwarf", "wd") if k in results)
                                     - results["wd+dwarf"]["objective"])
                                    if "wd+dwarf" in results and {"dwarf", "wd"} & set(results) else None,
                delta={k: r["objective"] - results[best]["objective"] for k, r in results.items()},
                mask=p.mask, n_fit=len(p.y), source_id=sed.source_id,
                data=dict(y=p.y, error=np.sqrt(p.variance)), free_radius=free_radius,
                hydrogen_layer=p.wd.hydrogen_layer if not free_radius else None)
