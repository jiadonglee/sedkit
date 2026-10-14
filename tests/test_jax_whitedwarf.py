"""JAX port of the white-dwarf frame: agreement with the numpy components and fits."""

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
import equinox as eqx
from scipy.optimize import minimize

from spbin import SED, StellarModel, WhiteDwarfModel
from spbin.extinction import EdenhoferPrior
from spbin.whitedwarf import DATA, _Hypothesis, _prepare
from spbin.whitedwarf import fit_whitedwarf_companion as numpy_fit
from spbin._jax import whitedwarf as jw
from spbin._jax.fit import _nelder_mead
from spbin._jax.whitedwarf import fit_whitedwarf_companion as jax_fit

FOLD = dict(np.load(DATA / "calibration.npz"))


class LinearDust:
    """Dust prior whose mean grows with distance, without a map query."""

    def moments(self, ra, dec, distance):
        return 0.01 + 2e-4 * distance, 0.02

    penalty = staticmethod(EdenhoferPrior.penalty)


@pytest.fixture(scope="module")
def wd():
    return WhiteDwarfModel()


@pytest.fixture(scope="module")
def stellar():
    return StellarModel()


def composite_sed(wd, stellar, teff=15000, mass=.6, companion=.5, noise=.0002, channels=None):
    flux = np.nan_to_num(wd.predict(teff, mass)["flux"]) + stellar.evaluate(companion, 0, 5, 0)["flux_10pc"]
    mask = wd.support if channels is None else channels
    return SED(flux * .01, flux * noise, mask, 10, .1, "mock", metadata=dict(ra=30., dec=10.))


def objective_pairs(p, kind, free_radius, age, feh, n, rng, fixed=None):
    """numpy and compiled objectives at random points inside the hypothesis bounds."""
    hyp = _Hypothesis(p, kind, free_radius, age, feh, fixed=fixed)
    spec, template = jw.spec_of(hyp), jw._template(hyp)
    problem = eqx.tree_at(lambda pr: pr.template, jw.build_problem(p, age), jnp.asarray(template))
    free = [jw.SLOTS.index(k) for k in hyp.names]
    points = np.array([[rng.uniform(*hyp.bounds[k]) for k in hyp.names] for _ in range(n)])
    thetas = np.tile(template, (n, 1))
    thetas[:, free] = points
    compiled = np.asarray(jw._scores(jnp.asarray(thetas), problem, spec))
    return np.array([hyp.objective(x) for x in points]), compiled


def assert_objectives_agree(reference, compiled):
    supported = reference < 1e29
    np.testing.assert_array_equal(compiled < 1e29, supported)
    assert supported.sum() > 10
    relative = np.abs(compiled[supported] - reference[supported]) / np.abs(reference[supported])
    assert relative.max() < 1e-8


@pytest.mark.parametrize("calibration", ["bundled", None, "teff_only"])
def test_whitedwarf_model_matches_numpy(calibration):
    if calibration == "teff_only":
        # A calibration without the UV and Teff-spline terms exercises the zero fills.
        calibration = {k: v for k, v in FOLD.items() if k in ("a", "b", "blue_a", "blue_b", "diag")}
    model = WhiteDwarfModel(calibration=calibration)
    arrays = jw._wd_arrays(model)
    rng = np.random.default_rng(5)
    checked = rejected = 0
    for _ in range(200):
        teff, mass = rng.uniform(5000, 90000), rng.uniform(.1, 1.4)
        radius, logg, age, valid = jw.physical(arrays, teff, mass)
        try:
            reference = model.predict(teff, mass)
        except ValueError:
            # Off the cooling table or its reach, or the cooling log g off the grid.
            ok = bool(valid) and bool(jw.predict(arrays, teff, logg, radius)[-1])
            assert not ok
            rejected += 1
            continue
        flux, blue, coarse, w, ok = jw.predict(arrays, teff, logg, radius)
        assert bool(valid) and bool(ok)
        for key, value in (("radius", radius), ("logg", logg), ("cooling_age_gyr", age)):
            assert float(value) == pytest.approx(reference[key], rel=1e-12)
        supported = model.support
        np.testing.assert_allclose(np.asarray(flux)[supported], reference["flux"][supported], rtol=1e-10)
        np.testing.assert_allclose(np.asarray(blue), reference["blue"], rtol=1e-10)
        np.testing.assert_allclose(np.asarray(coarse), reference["coarse"], rtol=1e-10)
        checked += 1
    assert checked > 50 and rejected > 20
    # Free-radius mode, including radii and gravities off the grid.
    for _ in range(50):
        teff, logg, radius = rng.uniform(5000, 90000), rng.uniform(6.5, 10), rng.uniform(.002, .06)
        flux, blue, coarse, w, ok = jw.predict(arrays, teff, logg, radius)
        try:
            reference = model.predict(teff, logg=logg, radius=radius)
        except ValueError:
            assert not bool(ok)
            continue
        assert bool(ok)
        np.testing.assert_allclose(np.asarray(coarse), reference["coarse"], rtol=1e-10)


@pytest.mark.parametrize("hot", [False, True])
def test_components_match_numpy(wd, hot):
    stellar = StellarModel(hot=hot)
    sed = composite_sed(wd, StellarModel())
    blue = (np.full(6, 2.0), np.full(6, .1))
    galex = dict(FUV=(17., .1, True), NUV=(16.5, .05, True))
    p = _prepare(sed, wd, stellar, .05, None, None, None, False, blue, galex, False, None, None)
    problem = jw.build_problem(p, 5.)
    slots = list(p.idx) + list(jw.BLUE_SLOTS) + [jw.GALEX_SLOTS[b] for b in p.galex]
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(40):
        mass, q = rng.uniform(.15, 2.1), rng.choice([0., rng.uniform(.1, 1.)])
        age, feh, e = rng.uniform(.5, 10), rng.uniform(-1, .5), rng.uniform(0, .5)
        reference = p.dwarf(mass, age, feh, .01, e, q)
        component, columns = jw.dwarf(problem, mass, age, feh, .01, e, q)
        assert bool(component.valid) == (reference is not None)
        if reference is None:
            continue
        np.testing.assert_allclose(np.asarray(component.flux)[slots], reference["flux"], rtol=1e-10)
        np.testing.assert_allclose(np.asarray(component.variance)[slots], reference["variance"], rtol=1e-10)
        np.testing.assert_allclose(np.asarray(component.coarse), reference["coarse"], rtol=1e-10)
        # numpy keeps only nonzero-weight error terms; the covariance is invariant.
        compiled = np.asarray(columns)[slots]
        covariance = reference["columns"] @ reference["columns"].T
        np.testing.assert_allclose(compiled @ compiled.T, covariance, rtol=1e-10,
                                   atol=1e-12 * np.abs(covariance).max())
        labels = reference["labels"]
        np.testing.assert_allclose(np.asarray(component.labels),
                                   [labels[k] for k in jw.DWARF_LABELS], rtol=1e-12)
        checked += 1
    assert checked > 15
    checked = 0
    for _ in range(40):
        # Spans both sides of the empirical UV range (6812--41430 K).
        teff, mass, e = rng.uniform(6000, 80000), rng.uniform(.2, 1.3), rng.uniform(0, .5)
        reference = p.white_dwarf(teff, mass, None, None, .01, e)
        component = jw.white_dwarf(problem, teff, mass, 1., 8., .01, e, False)
        assert bool(component.valid) == (reference is not None)
        if reference is None:
            continue
        np.testing.assert_allclose(np.asarray(component.flux)[slots], reference["flux"], rtol=1e-10)
        np.testing.assert_allclose(np.asarray(component.variance)[slots], reference["variance"], rtol=1e-10)
        checked += 1
    assert checked > 15


def test_objectives_match_numpy(wd, stellar):
    """Every hypothesis, fitted extinction under a Gaussian prior, fitted parallax,
    free age and [M/H], label priors, blue and GALEX extras."""
    sed = composite_sed(wd, stellar)
    blue = (np.full(6, 2.0), np.full(6, .1))
    galex = dict(FUV=(17., .1, True), NUV=(16.5, .05, False))
    p = _prepare(sed, wd, stellar, None, (.02, .03), None, None, True, blue, galex, False,
                 {"mass": (.6, .1)}, {"feh": (0., .2), "teff": (4000., 500.)})
    rng = np.random.default_rng(0)
    for kind in ("dwarf", "ms+ms", "wd", "wd+dwarf"):
        assert_objectives_agree(*objective_pairs(p, kind, False, None, None, 150, rng))
    # Fixed WD temperature and mass, as the light-limit scan sets them.
    assert_objectives_agree(*objective_pairs(p, "wd+dwarf", False, None, None, 80, rng,
                                             fixed=dict(t=np.log(12000.), m=.8)))


def test_objectives_match_numpy_dust_and_free_radius(wd, stellar):
    sed = composite_sed(wd, stellar, channels=np.arange(168) < 61)
    p = _prepare(sed, wd, stellar, None, None, LinearDust(), None, True, None, None, False,
                 {"logg": (8., .2)}, None)
    problem = jw.build_problem(p, 5.)
    assert problem.dust_par.shape == (161,)
    rng = np.random.default_rng(1)
    for kind in ("dwarf", "wd+dwarf"):
        assert_objectives_agree(*objective_pairs(p, kind, False, 5., 0., 100, rng))
    assert_objectives_agree(*objective_pairs(p, "wd", True, 5., 0., 100, rng))
    fixed = _prepare(sed, wd, stellar, None, None, LinearDust(), None, False, None, None, False, None, None)
    assert jw.build_problem(fixed, 5.).dust_par.shape == (1,)
    assert_objectives_agree(*objective_pairs(fixed, "wd+dwarf", False, 5., 0., 100, rng))


def test_nelder_mead_call_budget_matches_scipy():
    """The WD frame stops on scipy's function-call budget with fatol=1e-4."""
    rng = np.random.default_rng(2)
    matrix, shift = rng.normal(size=(4, 5)), rng.uniform(-1, 1, 5)
    bounds = np.array([(-2., 2.)] * 5)

    def host(x):
        return float(np.sum((np.sin(matrix @ x) + 0.05 * (matrix @ (x - shift))**2)**2))

    device_matrix, device_shift = jnp.asarray(matrix), jnp.asarray(shift)

    def device(x):
        return jnp.sum((jnp.sin(device_matrix @ x) + 0.05 * (device_matrix @ (x - device_shift))**2)**2)

    for maxfev in (4000, 60):
        x0 = rng.uniform(-1, 1, 5)
        simplex = np.vstack([x0, x0 + np.diag([.3, .5, .2, 1.9, .25])])
        reference = minimize(host, x0, method="Nelder-Mead", bounds=bounds,
                             options=dict(initial_simplex=simplex, maxfev=maxfev, xatol=1e-5, fatol=1e-4))
        _, values, done = jax.jit(lambda s, b: _nelder_mead(
            device, s, b, maxiter=maxfev, xatol=1e-5, fatol=1e-4, maxfev=maxfev))(
                jnp.asarray(simplex), jnp.asarray(bounds))
        assert bool(done) == bool(reference.success)
        if reference.success:
            # Rounding-level differences move the stop within the tolerance.
            assert float(values[0]) == pytest.approx(reference.fun, abs=1e-4)


def summaries_agree(reference, result, tolerance=.01):
    assert result["preferred"] == reference["preferred"]
    for kind, fit in reference["hypotheses"].items():
        compiled = result["hypotheses"][kind]
        assert set(compiled) == set(fit)
        # The restart can settle below numpy's endpoint, not above.
        assert compiled["objective"] <= fit["objective"] + tolerance
        assert compiled["converged"]


def test_single_da_and_composite_recovery(wd):
    flux = wd.predict(15000, .6)["flux"]
    sed = SED(flux * .01, flux * .0001, wd.support, 10, .02, "mock")
    result = jax_fit(sed, model=wd)
    summaries_agree(numpy_fit(sed, model=wd), result)
    labels = result["hypotheses"]["wd"]["whitedwarf"]
    assert abs(labels["teff"] / 15000 - 1) < .01 and abs(labels["mass"] - .6) < .01
    flux = wd.predict(18000, .6)["flux"] + StellarModel().evaluate(.35, 0, 5, 0)["flux_10pc"]
    sed = SED(flux * .01, flux * .0001, wd.support, 10, .02, "mock")
    result, reference = jax_fit(sed, model=wd), numpy_fit(sed, model=wd)
    summaries_agree(reference, result)
    composite = result["hypotheses"]["wd+dwarf"]
    assert abs(composite["companion"]["mass"] - .35) < .02
    assert composite["photocentre"]["coefficient"] == pytest.approx(
        composite["photocentre"]["mass_fraction_star1"] - composite["fractions"]["beta_G"])
    assert set(result) == set(reference) and result["mask"].sum() == 64


def test_hot_wd_beside_a_dwarf(wd):
    flux = wd.predict(35000, .6)["flux"] + StellarModel().evaluate(.6, 0, 5, 0)["flux_10pc"]
    sed = SED(.01 * flux, .0002 * flux, np.arange(168) < 61, 10, .02, "hot_wd")
    result = jax_fit(sed, model=wd)
    summaries_agree(numpy_fit(sed, model=wd), result)
    composite = result["hypotheses"]["wd+dwarf"]
    assert composite["fractions"]["beta_G"] == pytest.approx(.2, abs=.02)
    assert abs(composite["companion"]["mass"] - .6) < .01


def test_main_sequence_pair(wd):
    stellar = StellarModel()
    flux = stellar.evaluate(.9, .7, 5, 0)["flux_10pc"] * .01
    sed = SED(flux, flux * .01, wd.support & stellar.support, 10, .02, "mock")
    options = dict(model=wd, hypotheses=("dwarf", "ms+ms", "wd+dwarf"), companion_age_gyr=None)
    result = jax_fit(sed, **options)
    summaries_agree(numpy_fit(sed, **options), result)
    pair = result["hypotheses"]["ms+ms"]["companion"]
    assert result["preferred"] == "ms+ms" and abs(pair["q"] - .7) < .02
    assert result["delta"]["dwarf"] > 100


def test_free_radius_and_input_checks(wd):
    p = wd.predict(13000, logg=8.1, radius=.016)
    sed = SED(p["flux"] * .01, p["flux"] * .0002, wd.support, 10, .02, "free")
    result = jax_fit(sed, model=wd, companions=False, free_radius=True, whitedwarf_prior={"logg": (8.1, .01)})
    labels = result["hypotheses"]["wd"]["whitedwarf"]
    assert abs(labels["radius"] / .016 - 1) < .03 and labels["cooling_age_gyr"] is None
    with pytest.raises(ValueError):
        jax_fit(sed, model=wd, extinction=None)
    with pytest.raises(ValueError):
        jax_fit(sed, model=wd, blue=([1], [1]))
    with pytest.raises(ValueError):
        jax_fit(sed, model=wd, hypotheses=("dwarf", "giant"))
