"""JAX port of `spbin.fit.fit` (prototype: core route and hot_ms).

Same signature and result fields as the numpy `fit`; the search runs on
compiled code: the multi-start grid is evaluated with one `vmap`, and the top
starts are refined either by a Nelder-Mead with scipy's exact bounded
semantics (default) or by a gradient-based BFGS on an unconstrained
reparametrisation. The Edenhofer dust prior is pre-sampled on a parallax grid
per star, replacing the per-evaluation map query; the grid interpolation makes
the in-fit dust penalty approximate to the extent the grid is.

Model tables are cached per model object; the fitter reads only the bundled
files through `StellarModel`, so reuse one instance across targets.
"""

import weakref

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..extinction import EdenhoferPrior, extinction_curve
from ..fit import STEP
from ..fit import neg2_log_likelihood as numpy_neg2_log_likelihood
from ..model import StellarModel
from . import likelihood, stellar
from .arrays import DUST_GRID_POINTS, ModelArrays

THETA_NAMES = ("ln_m1", "ln_age", "feh", "q", "extinction_e", "parallax_z")


class Problem(eqx.Module):
    """One fit's fixed data: model arrays, SED, parameter bounds and priors."""

    arr: ModelArrays
    sed_flux: jax.Array  # (168,)
    sed_error: jax.Array  # (168,)
    mask: jax.Array  # (168,) 0/1
    parallax_mas: jax.Array
    parallax_error_mas: jax.Array
    template: jax.Array  # (6,) the six parameters with placeholders at free slots
    bounds: jax.Array  # (6, 2)
    dust_par: jax.Array  # (DUST_GRID_POINTS,) trial parallaxes
    dust_mean: jax.Array  # (DUST_GRID_POINTS,)
    dust_sigma: jax.Array  # (DUST_GRID_POINTS,)
    dust_fail: jax.Array  # (DUST_GRID_POINTS,) 1 where the map has no support
    dust_flag: jax.Array  # scalar 0/1
    prior_mean: jax.Array  # (3,) prior means: log10(age/yr), primary logg, primary Teff
    prior_sigma: jax.Array  # (3,)
    prior_flag: jax.Array  # (3,) 0/1 for age, logg, Teff


def _objective(theta, problem):
    """-2 ln L with all priors; 1e30 outside the model domain."""
    arr = problem.arr
    bounds = problem.bounds
    mass = jnp.exp(jnp.clip(theta[0], bounds[0, 0], bounds[0, 1]))
    age = jnp.clip(jnp.exp(theta[1]), jnp.exp(bounds[1, 0]), jnp.exp(bounds[1, 1]))
    metal = jnp.clip(theta[2], bounds[2, 0], bounds[2, 1])
    ratio = jnp.clip(theta[3], bounds[3, 0], bounds[3, 1])
    extinction = jnp.clip(theta[4], bounds[4, 0], bounds[4, 1])
    z = jnp.clip(theta[5], bounds[5, 0], bounds[5, 1])
    parallax = problem.parallax_mas + z * problem.parallax_error_mas

    prediction = stellar.evaluate(arr, mass, ratio, age, metal)
    attenuation = jnp.exp(-extinction * arr.extinction_curve)
    columns, variance = stellar.error_factors(
        arr, prediction.labels[0, 0], prediction.flux_10pc, (parallax / 100.0)**2)
    value, _, _ = likelihood.neg2_log_likelihood(
        problem.sed_flux, problem.sed_error, problem.mask, parallax,
        prediction.flux_10pc, columns, variance, attenuation)
    total = value + z * z
    total += problem.dust_flag * likelihood.dust_penalty(
        extinction, parallax, problem.dust_par, problem.dust_mean,
        problem.dust_sigma, problem.dust_fail)
    terms = jnp.array([
        (jnp.log10(age) + 9 - problem.prior_mean[0]) / problem.prior_sigma[0],
        (prediction.logg[0] - problem.prior_mean[1]) / problem.prior_sigma[1],
        (prediction.labels[0, 0] - problem.prior_mean[2]) / problem.prior_sigma[2]])
    total += jnp.sum(problem.prior_flag * terms**2)
    return jnp.where(prediction.valid & jnp.isfinite(total), total, 1e30)


def _free_objective(u, template, free_idx, bounds, problem):
    """Objective on the free sub-vector scattered into the fixed template."""
    return _objective(template.at[free_idx].set(u), problem)


def _unconstrain(u, bounds):
    """Map unconstrained u to theta within bounds (softplus where hi is inf)."""
    lo, hi = bounds[:, 0], bounds[:, 1]
    finite = jnp.isfinite(hi)
    span = jnp.where(finite, hi - lo, 1.0)
    bounded = lo + span * jax.nn.sigmoid(u)
    unbounded = lo + jax.nn.softplus(u)
    return jnp.where(finite, bounded, unbounded)


def _constrain(theta, bounds):
    """Inverse of `_unconstrain`."""
    lo, hi = bounds[:, 0], bounds[:, 1]
    finite = jnp.isfinite(hi)
    span = jnp.where(finite, hi - lo, 1.0)
    x = jnp.clip((theta - lo) / span, 1e-12, 1 - 1e-12)
    return jnp.where(finite, jnp.log(x / (1 - x)),
                     jnp.log(jnp.expm1(jnp.maximum(theta - lo, 1e-12))))


def _nelder_mead(objective, simplex, bounds, maxiter=1600, xatol=1e-5, fatol=1e-5):
    """Bounded Nelder-Mead with scipy's semantics on fixed shapes.

    Clipped evaluations, the reflect-then-clip initial-simplex treatment, the
    reflect/expand/contract/shrink rules with rho=1, chi=2, psi=sigma=0.5, and
    the top-of-loop xatol/fatol termination. Returns the final simplex sorted
    by value (vertex 0 is scipy's result), its values and the converged flag.
    """
    lo, hi = bounds[:, 0], bounds[:, 1]
    simplex = jnp.where(simplex > hi, 2 * hi - simplex, simplex)
    simplex = jnp.clip(simplex, lo, hi)
    values = jax.vmap(objective)(simplex)
    order = jnp.argsort(values)
    simplex, values = simplex[order], values[order]
    n = simplex.shape[1]

    def converged(sim, fs):
        x_term = jnp.max(jnp.abs(sim[1:] - sim[0])) <= xatol
        f_term = jnp.max(jnp.abs(fs[0] - fs[1:])) <= fatol
        return x_term & f_term

    def step(carry):
        simplex, values = carry
        xbar = simplex[:-1].sum(axis=0) / n
        xr = jnp.clip(2 * xbar - simplex[-1], lo, hi)
        xe = jnp.clip(3 * xbar - 2 * simplex[-1], lo, hi)
        xc = jnp.clip(1.5 * xbar - 0.5 * simplex[-1], lo, hi)
        xcc = jnp.clip(0.5 * xbar + 0.5 * simplex[-1], lo, hi)
        fxr, fxe, fxc, fxcc = jax.vmap(objective)(jnp.stack([xr, xe, xc, xcc]))

        expand = fxr < values[0]
        reflect = (~expand) & (fxr < values[-2])
        outside = (~expand) & ~reflect & (fxr < values[-1])
        inside = ~expand & ~reflect & ~outside
        new_x, new_f = xr, fxr
        take = expand & (fxe < fxr)
        new_x, new_f = jnp.where(take, xe, new_x), jnp.where(take, fxe, new_f)
        take = outside & (fxc <= fxr)
        new_x, new_f = jnp.where(take, xc, new_x), jnp.where(take, fxc, new_f)
        take = inside & (fxcc < values[-1])
        new_x, new_f = jnp.where(take, xcc, new_x), jnp.where(take, fxcc, new_f)
        shrink = (outside & ~(fxc <= fxr)) | (inside & ~(fxcc < values[-1]))

        def do_shrink(_):
            moved = jnp.clip(simplex[0] + 0.5 * (simplex - simplex[0]), lo, hi)
            moved_f = jax.vmap(objective)(moved).at[0].set(values[0])
            return moved, moved_f

        simplex, values = jax.lax.cond(
            shrink, do_shrink,
            lambda _: (simplex.at[-1].set(new_x), values.at[-1].set(new_f)), None)
        order = jnp.argsort(values)
        return simplex[order], values[order]

    def body(state):
        simplex, values, iterations, done = state
        hit = converged(simplex, values)
        simplex, values = jax.lax.cond(hit | done, lambda c: c, step, (simplex, values))
        return simplex, values, iterations + 1, done | hit

    def keep_going(state):
        return (~state[3]) & (state[2] < maxiter)

    simplex, values, _, done = jax.lax.while_loop(
        keep_going, body, (simplex, values, 1, jnp.asarray(False)))
    return simplex, values, done


def _bfgs(u0, bounds, template, free_idx, problem):
    """Unconstrained-space BFGS; returns theta, objective and the result code."""
    def objective(u, args):
        return _free_objective(_unconstrain(u, bounds), template, free_idx, bounds, problem)
    import optimistix as optx

    solution = optx.minimise(objective, optx.BFGS(rtol=1e-9, atol=1e-9), u0,
                             max_steps=300, throw=False)
    success = solution.state.result == optx.RESULTS.successful
    return _unconstrain(solution.value, bounds), success


def _nm_runner(simplices, bounds, template, free_idx, problem, maxiter=1600):
    # One solve at a time: under vmap the step's lax.cond branches all
    # evaluate (the shrink alone costs n+1 objective calls per iteration)
    # and the batch waits for the slowest start.
    def one(simplex):
        return _nelder_mead(
            lambda u: _free_objective(u, template, free_idx, bounds, problem),
            simplex, bounds, maxiter=maxiter)
    return jax.lax.map(one, simplices)


def _bfgs_runner(starts, bounds, template, free_idx, problem):
    free_bounds = bounds[free_idx]

    def one(u0):
        theta, success = _bfgs(u0, free_bounds, template, free_idx, problem)
        full = template.at[free_idx].set(theta)
        return theta, _objective(full, problem), success

    starts_u = jax.vmap(_constrain, in_axes=(0, None))(starts[:, free_idx], free_bounds)
    return jax.vmap(one)(starts_u)


_nm_compiled = jax.jit(_nm_runner)
_bfgs_compiled = jax.jit(_bfgs_runner)
_grid_scores = jax.jit(jax.vmap(_objective, in_axes=(0, None)))

_arrays_cache = weakref.WeakKeyDictionary()


def _model_arrays(model):
    if model not in _arrays_cache:
        _arrays_cache[model] = ModelArrays.build(model)
    return _arrays_cache[model]


def _dust_grid(dust_prior, ra, dec, parallax_mas, parallax_error_mas, z_bounds):
    """Sample the dust prior over the reachable parallax range of one star."""
    z = np.linspace(z_bounds[0], z_bounds[1], DUST_GRID_POINTS)
    parallax = parallax_mas + z * parallax_error_mas
    mean = np.zeros(len(z))
    sigma = np.zeros(len(z))
    fail = np.zeros(len(z))
    for i, distance in enumerate(1000.0 / parallax):
        try:
            mean[i], sigma[i] = dust_prior.moments(ra, dec, distance)
        except ValueError:
            fail[i] = 1.0
            mean[i], sigma[i] = 0.0, 1.0
    return parallax, mean, sigma, fail


def _grid(cfg, model, dust_mean, dust_sigma, age_prior):
    """Start grid exactly as the numpy fit builds it, as (G, 6) parameters."""
    low_age, high_age = model.age_range_gyr
    if cfg["age_free"]:
        ages = ((0.01, 0.05, 0.3) if model.hot else ()) + (1.0, 4.0, 9.0)
        if age_prior is not None:
            ages = tuple(a for a in ages + (10**(age_prior[0] - 9),)
                         if low_age <= a <= high_age)
    else:
        ages = (cfg["age_gyr"],)
    metals = (-0.5, 0.0, 0.3) if cfg["feh_free"] else (cfg["feh"],)
    ratios = (0.3, 0.45, 0.6, 0.75, 0.9, 1.0) if cfg["binary"] and cfg["q_free"] else (
        (cfg["q"],) if cfg["binary"] else (0.0,))
    if cfg["extinction_free"]:
        extinctions = np.unique(np.maximum(
            0, [dust_mean - dust_sigma, dust_mean, dust_mean + dust_sigma]))
    else:
        extinctions = (cfg["extinction"],)
    masses = np.geomspace(0.09, 19.9, 64) if model.hot else np.geomspace(0.09, 2.19, 38)
    grid = np.empty((len(masses) * len(ages) * len(metals) * len(ratios)
                     * len(extinctions), 6))
    index = 0
    for mass in masses:
        for age in ages:
            for metal in metals:
                for ratio in ratios:
                    for extinction in extinctions:
                        values = dict(ln_m1=np.log(mass), ln_age=np.log(age), feh=metal,
                                      q=ratio, parallax_z=0.0, extinction_e=extinction)
                        grid[index] = [values[name] for name in THETA_NAMES]
                        index += 1
    return grid


def fit(sed, kind="both", *, model=None, age_gyr=5.0, feh=0.0, q=None,
        use_wise=False, fit_parallax=False, extinction=0.0, dust_prior=None,
        age_prior=None, logg_prior=None, teff_prior=None, optimizer="nm"):
    """Fit single/binary/both on one shared observation-derived mask.

    Signature and result fields match `spbin.fit.fit`; differences are that
    q must be a number (the orbit-constraint callable stays numpy-only) and
    optimizer= selects the local refinement: "nm" (default, scipy-semantics
    Nelder-Mead from the grid starts), "bfgs" (unconstrained-transform
    gradient optimisation, much faster but prone to stalling on the PARSEC
    grid-node kinks) or "bfgs+nm" (BFGS endpoints polished by a short
    Nelder-Mead). See the numpy `fit` for the shared parameters.
    """
    if kind not in ("single", "binary", "both"):
        raise ValueError("kind must be single, binary or both")
    if q is not None and callable(q):
        raise ValueError("the JAX fitter takes q as a number, not a callable")
    if q is not None and (not np.isfinite(q) or not 0.1 <= q <= 1):
        raise ValueError("fixed binary q must be 0.1--1")
    if not np.isfinite(sed.parallax_mas) or sed.parallax_mas <= 0:
        raise ValueError("fitting requires a positive measured parallax")
    for name, prior in (("age_prior", age_prior), ("logg_prior", logg_prior),
                        ("teff_prior", teff_prior)):
        if prior is not None and (len(prior) != 2 or not np.all(np.isfinite(prior))
                                  or prior[1] <= 0):
            raise ValueError(f"{name} must be (mean, positive sigma)")
    if age_prior is not None and age_gyr is not None:
        raise ValueError("age_prior requires a free age (age_gyr=None)")
    model = StellarModel() if model is None else model
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
        if any(key not in sed.metadata or not np.isfinite(sed.metadata[key])
               for key in ("ra", "dec")):
            raise ValueError("Edenhofer prior requires finite ICRS ra/dec degrees in sed.metadata")
    if extinction_free and dust_prior is None:
        dust_prior = EdenhoferPrior()
    curve = extinction_curve(model.wavelength_um)
    low_mass, high_mass = model.mass_range
    low_age, high_age = model.age_range_gyr
    arrays = _model_arrays(model)

    z_low = (max(-3.0, -sed.parallax_mas / sed.parallax_error_mas + 1e-6)
             if parallax_free else -3.0)
    bounds = np.array([
        (np.log(low_mass), np.log(high_mass)),
        (np.log(low_age), np.log(high_age)),
        (-1.0, 0.5),
        (0.1, 1.0),
        (0.0, np.inf),
        (z_low, 3.0)])
    if dust_prior is not None:
        dust_mean, dust_sigma = dust_prior.moments(
            sed.metadata["ra"], sed.metadata["dec"], 1000 / sed.parallax_mas)
        if parallax_free:
            dust_par, dust_mean_grid, dust_sigma_grid, dust_fail = _dust_grid(
                dust_prior, sed.metadata["ra"], sed.metadata["dec"],
                sed.parallax_mas, sed.parallax_error_mas, bounds[5])
        else:
            # A fixed parallax evaluates the penalty at one distance; a map
            # query per grid node would be wasted.
            dust_par = np.array([sed.parallax_mas])
            dust_mean_grid = np.array([dust_mean])
            dust_sigma_grid = np.array([dust_sigma])
            dust_fail = np.zeros(1)
    else:
        dust_mean = dust_sigma = None
        zeros = np.zeros(DUST_GRID_POINTS)
        dust_par, dust_mean_grid, dust_sigma_grid, dust_fail = (
            zeros + 1.0, zeros, zeros + 1.0, zeros)

    def fit_one(binary):
        free = ["ln_m1"]
        if age_gyr is None:
            free.append("ln_age")
        if feh is None:
            free.append("feh")
        if binary and q is None:
            free.append("q")
        if extinction_free:
            free.append("extinction_e")
        if parallax_free:
            free.append("parallax_z")
        free_idx = np.array([THETA_NAMES.index(name) for name in free])
        # A non-fitted q leaves the template value in place: the clip floor
        # is 0 so a single star's q=0 survives (fitted q keeps the 0.1 floor).
        one_bounds = bounds.copy()
        if not (binary and q is None):
            one_bounds[3, 0] = 0.0
        template = np.array([
            np.log(0.7), np.log(5.0), 0.0, q if binary and q is not None else 0.0,
            0.0 if extinction is None else extinction, 0.0])
        problem = Problem(
            arr=arrays,
            sed_flux=jnp.asarray(sed.flux), sed_error=jnp.asarray(sed.error),
            mask=jnp.asarray(mask, float),
            parallax_mas=jnp.asarray(sed.parallax_mas),
            parallax_error_mas=jnp.asarray(sed.parallax_error_mas),
            template=jnp.asarray(template), bounds=jnp.asarray(one_bounds),
            dust_par=jnp.asarray(dust_par), dust_mean=jnp.asarray(dust_mean_grid),
            dust_sigma=jnp.asarray(dust_sigma_grid), dust_fail=jnp.asarray(dust_fail),
            dust_flag=jnp.asarray(float(dust_prior is not None)),
            prior_mean=jnp.asarray([
                0.0 if age_prior is None else age_prior[0],
                0.0 if logg_prior is None else logg_prior[0],
                0.0 if teff_prior is None else teff_prior[0]]),
            prior_sigma=jnp.asarray([
                1.0 if age_prior is None else age_prior[1],
                1.0 if logg_prior is None else logg_prior[1],
                1.0 if teff_prior is None else teff_prior[1]]),
            prior_flag=jnp.asarray([
                float(age_prior is not None and age_gyr is None),
                float(logg_prior is not None), float(teff_prior is not None)]))

        cfg = dict(binary=binary, age_free=age_gyr is None, feh_free=feh is None,
                   q_free=binary and q is None, extinction_free=extinction_free,
                   age_gyr=5.0 if age_gyr is None else age_gyr,
                   feh=0.0 if feh is None else feh, q=q,
                   extinction=0.0 if extinction_free else extinction)
        thetas = np.asarray(_grid(cfg, model, dust_mean, dust_sigma, age_prior))
        scores = np.asarray(_grid_scores(jnp.asarray(thetas), problem))
        supported = scores < 1e29
        if not supported.any():
            raise ValueError("no supported stellar model for this age, metallicity and q")
        starts = thetas[np.argsort(np.where(supported, scores, np.inf))[:5]]
        n_free = len(free)

        free_bounds = one_bounds[free_idx]
        steps = np.array([STEP[name] for name in free])
        # Candidate pool: the starts keep a supported fallback in case every
        # refined endpoint diverged into the unsupported region.
        solution = starts[:, free_idx]
        values = np.where(supported, scores, np.inf)[
            np.argsort(np.where(supported, scores, np.inf))[:5]]
        converged_flags = np.zeros(len(starts), bool)
        if optimizer == "nm":
            simplices = np.tile(starts[:, free_idx][:, None, :], (1, len(free) + 1, 1))
            for i in range(len(free)):
                at_high = starts[:, free_idx[i]] + steps[i] > free_bounds[i, 1]
                simplices[:, i + 1, i] += np.where(at_high, -steps[i], steps[i])
            nm_solution, nm_values, done = _nm_compiled(
                jnp.asarray(simplices), jnp.asarray(free_bounds),
                jnp.asarray(template), jnp.asarray(free_idx), problem)
            # Every final vertex is a candidate: they lie within the tolerances
            # of the best one, which matters when that one fails numpy's check.
            solution = np.concatenate([solution, np.asarray(nm_solution).reshape(-1, n_free)])
            values = np.concatenate([values, np.asarray(nm_values).ravel()])
            converged_flags = np.concatenate([converged_flags,
                                              np.repeat(np.asarray(done), n_free + 1)])
        elif optimizer in ("bfgs", "bfgs+nm"):
            bfgs_solution, bfgs_values, bfgs_done = _bfgs_compiled(
                jnp.asarray(starts), jnp.asarray(one_bounds), jnp.asarray(template),
                jnp.asarray(free_idx), problem)
            bfgs_solution, bfgs_values = np.asarray(bfgs_solution), np.asarray(bfgs_values)
            converged_flags = np.concatenate([converged_flags, np.asarray(bfgs_done)])
            if optimizer == "bfgs+nm":
                # BFGS stalls on the PARSEC grid-node kinks; a Nelder-Mead from
                # each endpoint settles the remaining distance at a fraction of
                # the full-search cost. Endpoints that diverged into the
                # unsupported region fall back to their grid start.
                polish_steps = 0.25 * steps
                good = np.isfinite(bfgs_values) & (bfgs_values < 1e29)
                polish_from = np.where(good[:, None], bfgs_solution, starts[:, free_idx])
                polish = np.tile(polish_from[:, None, :], (1, len(free) + 1, 1))
                for i in range(len(free)):
                    at_high = polish_from[:, i] + polish_steps[i] > free_bounds[i, 1]
                    polish[:, i + 1, i] += np.where(at_high, -polish_steps[i],
                                                    polish_steps[i])
                pol_solution, pol_values, pol_done = _nm_compiled(
                    jnp.asarray(polish), jnp.asarray(free_bounds),
                    jnp.asarray(template), jnp.asarray(free_idx), problem, 1600)
                bfgs_solution = np.concatenate([bfgs_solution,
                                                np.asarray(pol_solution).reshape(-1, n_free)])
                bfgs_values = np.concatenate([bfgs_values, np.asarray(pol_values).ravel()])
                converged_flags = np.concatenate([converged_flags,
                                                  np.repeat(np.asarray(pol_done), n_free + 1)])
            solution = np.concatenate([solution, bfgs_solution])
            values = np.concatenate([values, bfgs_values])
        else:
            raise ValueError("optimizer must be 'nm', 'bfgs' or 'bfgs+nm'")
        values = np.where(np.isfinite(values) & (values < 1e29), values, np.inf)
        # An optimum pressed against a domain edge can sit an ulp inside it
        # for the compiled model and outside it for numpy; the result is
        # reported with numpy, so take the best candidate numpy supports.
        for best in np.argsort(values, kind="stable"):
            if not np.isfinite(values[best]):
                raise ValueError("no supported stellar model for this age, metallicity and q")
            theta = starts[0].copy()  # the starts differ only in free parameters
            theta[free_idx] = solution[best]
            mass, age, metal, ratio, _, _, _ = _unpack(
                theta, sed, parallax_free, low_age, high_age)
            if model.evaluate(mass, ratio, age, metal) is not None:
                break
        else:
            raise ValueError("no supported stellar model for this age, metallicity and q")
        converged = bool(converged_flags[best])

        return _report(sed, model, mask, curve, theta, bounds, free, age_prior,
                       logg_prior, teff_prior, dust_prior, converged, binary)

    if kind != "both":
        return fit_one(kind == "binary")
    single, binary = fit_one(False), fit_one(True)
    return dict(single=single, binary=binary,
                delta=single["objective"] - binary["objective"], source_id=sed.source_id)


def _unpack(theta, sed, parallax_free, low_age, high_age):
    """Host-side mirror of the compiled objective's unpack."""
    mass = float(np.exp(theta[0]))
    age = float(np.clip(np.exp(theta[1]), low_age, high_age))
    metal = float(theta[2])
    ratio = float(theta[3])
    extinction = float(theta[4])
    z = float(theta[5]) if parallax_free else 0.0
    parallax = sed.parallax_mas + z * sed.parallax_error_mas
    return mass, age, metal, ratio, parallax, extinction, z


def _report(sed, model, mask, curve, theta, bounds, free, age_prior, logg_prior,
            teff_prior, dust_prior, converged, binary):
    """Assemble the result dict with the numpy forward model and likelihood."""
    mass, age, metal, ratio, parallax, extinction, z = _unpack(
        theta, sed, "parallax_z" in free, model.age_range_gyr[0], model.age_range_gyr[1])
    prediction = model.evaluate(mass, ratio, age, metal)
    attenuation = np.exp(-extinction * curve)
    value, chi2, flux = numpy_neg2_log_likelihood(
        sed, prediction, model, mask, parallax, attenuation)
    if dust_prior is None:
        penalty, prior_mean, prior_sigma = 0.0, None, None
    else:
        prior_mean, prior_sigma = dust_prior.moments(
            sed.metadata["ra"], sed.metadata["dec"], 1000 / parallax)
        penalty = dust_prior.penalty(extinction, prior_mean, prior_sigma)
    label_penalty = 0.0
    if age_prior is not None:
        label_penalty += ((np.log10(age) + 9 - age_prior[0]) / age_prior[1])**2
    if logg_prior is not None:
        label_penalty += ((prediction["logg"][0] - logg_prior[0]) / logg_prior[1])**2
    if teff_prior is not None:
        label_penalty += ((prediction["teff"][0] - teff_prior[0]) / teff_prior[1])**2
    free_idx = [THETA_NAMES.index(name) for name in free]
    at_bounds = [name for name, value_i, (lo, hi) in
                 zip(free, theta[free_idx], bounds[free_idx])
                 if min(value_i - lo, hi - value_i) < 1e-3 * (hi - lo if np.isfinite(hi) else 1)]
    return dict(m1=mass, m2=mass * ratio, q=ratio, age_gyr=age, feh=metal,
                parallax_mas=float(parallax), extinction_e=extinction,
                dust_prior_penalty=float(penalty), label_prior_penalty=float(label_penalty),
                logg=float(prediction["logg"][0]),
                dust_prior_mean=None if dust_prior is None else float(prior_mean),
                dust_prior_sigma=None if dust_prior is None else float(prior_sigma),
                objective=float(value + z * z + penalty + label_penalty),
                m2lnl=value, chi2=chi2, n_fit=int(mask.sum()), converged=converged,
                at_bounds=at_bounds, flux=flux,
                components=prediction["components"] * (parallax / 100)**2 * attenuation,
                beta_g=prediction["beta_g"], M_G=prediction["M_G"],
                a_phot_over_a1=prediction["a_phot_over_a1"], route=prediction["route"],
                mask=mask.copy(), kind="binary" if binary else "single")
