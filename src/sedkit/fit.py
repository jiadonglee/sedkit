"""Small, multi-start likelihood fits on the unchanged absolute fluxes."""

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

from .model import StellarModel
from .extinction import EdenhoferPrior, extinction_curve

# Initial Nelder-Mead step of each fitted parameter.
STEP = {"ln_m1": 0.05, "ln_age": 0.2, "feh": 0.1, "q": 0.1, "parallax_z": 0.5,
        "extinction_e": 0.03}


def neg2_log_likelihood(sed, prediction, model, mask, parallax_mas, transmission=1.0):
    """-2 ln L (constant n ln(2pi) omitted), chi2 and observed-scale model.

    Measurement errors are diagonal. The low-rank model covariance is
    retained, including its parameter-dependent determinant.
    """
    scale = (parallax_mas / 100.0)**2
    flux = prediction["flux_10pc"] * scale * transmission
    columns, variance = model.error_factors(prediction, scale)
    columns = columns * np.asarray(transmission)[..., None]
    variance = variance * np.asarray(transmission)**2
    residual = (flux - sed.flux)[mask]
    diagonal = sed.error[mask]**2 + variance[mask]
    columns = columns[mask]
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        raise ValueError("model-error covariance has no support on selected channels")
    small = np.eye(columns.shape[1]) + columns.T @ (columns / diagonal[:, None])
    nuisance = np.linalg.solve(small, columns.T @ (residual / diagonal))
    whitened = residual - columns @ nuisance
    chi2 = float(np.sum(whitened**2 / diagonal) + nuisance @ nuisance)
    value = chi2 + float(np.log(diagonal).sum() + np.linalg.slogdet(small)[1])
    return value, chi2, flux


def fit(sed, kind="both", *, model=None, age_gyr=5.0, feh=0.0, q=None,
        use_wise=False, fit_parallax=False, extinction=0.0, dust_prior=None):
    """Fit single/binary/both on one shared observation-derived mask.

    Age and metallicity are fixed by default. Pass age_gyr=None and/or
    feh=None to fit them. Binary q is fitted over 0.1--1, or fixed by q=.
    Parallax is fixed by default; fit_parallax=True fits it with one
    Gaussian catalogue constraint shared by hypotheses.
    extinction is fixed ZGR23 E, or None to fit E>=0 with an EdenhoferPrior.
    The dust prior is evaluated at each trial distance under both hypotheses.
    With StellarModel(hot=True), mass reaches 20 Msun, age 10**6.6 yr, and
    both hypotheses use the channels the hot route predicts (XP61, J/H/Ks).
    Results are local optima, not posterior samples or binary probabilities.
    """
    if kind not in ("single", "binary", "both"):
        raise ValueError("kind must be single, binary or both")
    if q is not None and (not np.isfinite(q) or not 0.1 <= q <= 1):
        raise ValueError("fixed binary q must be 0.1--1")
    if not np.isfinite(sed.parallax_mas) or sed.parallax_mas <= 0:
        raise ValueError("fitting requires a positive measured parallax")
    model = StellarModel() if model is None else model
    # Validate fixed age/metallicity before the optimiser enters the model.
    model.labels([0.7], 5.0 if age_gyr is None else age_gyr, 0.0 if feh is None else feh)
    mask = sed.fit_mask(use_wise) & model.support
    if mask.sum() < 4:
        raise ValueError("at least four valid fitting channels are required")
    parallax_free = (fit_parallax and np.isfinite(sed.parallax_error_mas)
                     and sed.parallax_error_mas > 0)
    extinction_free = extinction is None
    if not extinction_free and (not np.isfinite(extinction) or extinction < 0):
        raise ValueError("extinction must be a nonnegative ZGR23 E or None")
    if extinction_free or dust_prior is not None:
        if any(key not in sed.metadata or not np.isfinite(sed.metadata[key]) for key in ("ra", "dec")):
            raise ValueError("Edenhofer prior requires finite ICRS ra/dec degrees in sed.metadata")
    if extinction_free and dust_prior is None:
        dust_prior = EdenhoferPrior()
    curve = extinction_curve(model.wavelength_um) if extinction_free or extinction else np.zeros(168)
    low_mass, high_mass = model.mass_range
    low_age, high_age = model.age_range_gyr

    @lru_cache(maxsize=256)
    def dust_moments(parallax):
        return dust_prior.moments(sed.metadata["ra"], sed.metadata["dec"], 1000 / parallax)

    if dust_prior is not None:
        dust_mean, dust_sigma = dust_moments(sed.parallax_mas)

    def dust_penalty(e, parallax):
        if dust_prior is None:
            return 0.0
        return dust_prior.penalty(e, *dust_moments(parallax))

    def fit_one(binary):
        names, bounds = ["ln_m1"], [(np.log(low_mass), np.log(high_mass))]
        if age_gyr is None:
            names.append("ln_age")
            bounds.append((np.log(low_age), np.log(high_age)))
        if feh is None:
            names.append("feh")
            bounds.append((-1.0, 0.5))
        if binary and q is None:
            names.append("q")
            bounds.append((0.1, 1.0))
        if extinction_free:
            names.append("extinction_e")
            bounds.append((0.0, np.inf))
        if parallax_free:
            names.append("parallax_z")
            low = max(-3.0, -sed.parallax_mas / sed.parallax_error_mas + 1e-6)
            bounds.append((low, 3.0))

        def unpack(vector):
            values = dict(zip(names, vector))
            return (np.exp(values["ln_m1"]),
                    float(np.clip(np.exp(values["ln_age"]), low_age, high_age)) if age_gyr is None else age_gyr,
                    values.get("feh", feh),
                    values.get("q", q) if binary else 0.0,
                    sed.parallax_mas + values["parallax_z"] * sed.parallax_error_mas
                    if parallax_free else sed.parallax_mas,
                    values.get("extinction_e", extinction))

        def objective(vector):
            mass, age, metal, ratio, parallax, e = unpack(vector)
            prediction = model.evaluate(mass, ratio, age, metal)
            if prediction is None:
                return 1e30
            try:
                penalty = dust_penalty(e, parallax)
            except ValueError:
                return 1e30
            attenuation = np.exp(-e * curve)
            value = neg2_log_likelihood(sed, prediction, model, mask, parallax, attenuation)[0]
            z = vector[names.index("parallax_z")] if parallax_free else 0.0
            return value + z*z + penalty

        starts = []
        ages = ((0.01, 0.05, 0.3) if model.hot else ()) + (1.0, 4.0, 9.0) if age_gyr is None else (age_gyr,)
        metals = (-0.5, 0.0, 0.3) if feh is None else (feh,)
        ratios = (0.3, 0.45, 0.6, 0.75, 0.9, 1.0) if binary and q is None else (q if binary else 0.0,)
        extinctions = np.unique(np.maximum(0, [dust_mean - dust_sigma, dust_mean,
                                              dust_mean + dust_sigma])) if extinction_free else (extinction,)
        for mass in np.geomspace(0.09, 19.9, 64) if model.hot else np.geomspace(0.09, 2.19, 38):
            for age in ages:
                for metal in metals:
                    for ratio in ratios:
                        for e in extinctions:
                            values = dict(ln_m1=np.log(mass), ln_age=np.log(age), feh=metal,
                                          q=ratio, parallax_z=0.0, extinction_e=e)
                            vector = np.array([values[name] for name in names])
                            score = objective(vector)
                            if score < 1e29:
                                starts.append((score, vector))
        if not starts:
            raise ValueError("no supported stellar model for this age, metallicity and q")
        starts.sort(key=lambda item: item[0])
        candidates = []
        step = np.array([STEP[name] for name in names])
        for _, vector in starts[:5]:
            # Explicit simplex: scipy's default barely moves coordinates that start at zero.
            simplex = np.tile(vector, (len(names) + 1, 1))
            for i, (_, high) in enumerate(bounds):
                simplex[i + 1, i] += step[i] if vector[i] + step[i] <= high else -step[i]
            result = minimize(objective, vector, method="Nelder-Mead", bounds=bounds,
                              options={"maxiter": 1600, "xatol": 1e-5, "fatol": 1e-5,
                                       "initial_simplex": simplex})
            if result.fun < 1e29:
                candidates.append(result)
        result = min(candidates, key=lambda item: item.fun)
        mass, age, metal, ratio, parallax, e = unpack(result.x)
        prediction = model.evaluate(mass, ratio, age, metal)
        attenuation = np.exp(-e * curve)
        value, chi2, flux = neg2_log_likelihood(sed, prediction, model, mask, parallax, attenuation)
        at_bounds = [name for name, value_i, (lo, hi) in zip(names, result.x, bounds)
                     if min(value_i - lo, hi - value_i) < 1e-3 * (hi - lo if np.isfinite(hi) else 1)]
        return dict(m1=float(mass), m2=float(mass * ratio), q=float(ratio),
                    age_gyr=float(age), feh=float(metal), parallax_mas=float(parallax),
                    extinction_e=float(e), dust_prior_penalty=float(dust_penalty(e, parallax)),
                    dust_prior_mean=None if dust_prior is None else float(dust_moments(parallax)[0]),
                    dust_prior_sigma=None if dust_prior is None else float(dust_moments(parallax)[1]),
                    objective=float(result.fun), m2lnl=value, chi2=chi2,
                    n_fit=int(mask.sum()), converged=bool(result.success), at_bounds=at_bounds,
                    flux=flux, components=prediction["components"] * (parallax / 100)**2 * attenuation,
                    beta_g=prediction["beta_g"], M_G=prediction["M_G"],
                    a_phot_over_a1=prediction["a_phot_over_a1"], route=prediction["route"],
                    mask=mask.copy(), kind="binary" if binary else "single")

    if kind != "both":
        return fit_one(kind == "binary")
    single, binary = fit_one(False), fit_one(True)
    return dict(single=single, binary=binary, delta=single["objective"] - binary["objective"],
                source_id=sed.source_id)
