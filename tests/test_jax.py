"""JAX port of the forward model and `fit`: agreement with the numpy versions."""

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from scipy.special import log_ndtr

from spbin import SED, StellarModel
from spbin.extinction import EdenhoferPrior, extinction_curve
from spbin.fit import neg2_log_likelihood
from spbin.fit import fit as numpy_fit
from spbin._jax import likelihood, stellar
from spbin._jax.arrays import ModelArrays
from spbin._jax.fit import _grid, _nelder_mead, _objective, Problem
from spbin._jax.fit import fit as jax_fit


class FakeQuery:
    """Constant-mean, constant-width stand-in for the Edenhofer map query."""

    integrated = True
    n_samples = None

    def __call__(self, coords, mode="mean"):
        return 0.05 if mode == "mean" else 0.02


class DistanceQuery:
    """Map stand-in whose mean grows with distance, so the in-fit dust grid matters."""

    integrated = True
    n_samples = None

    def __call__(self, coords, mode="mean"):
        return 0.02 + 1e-3 * coords.distance.pc if mode == "mean" else 0.01


@pytest.fixture(scope="module")
def model():
    return StellarModel()


@pytest.fixture(scope="module")
def arrays():
    return ModelArrays.build(StellarModel())


@pytest.fixture(scope="module")
def sed():
    rng = np.random.default_rng(1)
    pair = StellarModel().evaluate(0.85, 0.45, 4.5, -0.4)
    flux = pair["flux_10pc"] * 0.0625  # 16 mas
    noisy = flux * (1 + rng.normal(0, 0.03, 168))
    result = SED(noisy, flux * 0.03, np.ones(168, bool), 16.0, 0.8)
    result.metadata.update(ra=30.0, dec=10.0, phot_g_mean_mag=14.0)
    return result


def test_forward_matches_numpy(model, arrays):
    rng = np.random.default_rng(42)
    checked = 0
    for _ in range(80):
        m1 = rng.uniform(0.09, 2.1)
        q = rng.choice([0.0, rng.uniform(0.1, 1.0)])
        age = rng.uniform(0.5, 10.0)
        feh = rng.uniform(-1.0, 0.5)
        reference = model.evaluate(m1, q, age, feh)
        out = stellar.evaluate(arrays, float(m1), float(q), float(age), float(feh))
        valid = bool(np.asarray(out.valid))
        if reference is None:
            assert not valid
            continue
        assert valid
        np.testing.assert_allclose(np.asarray(out.flux_10pc), reference["flux_10pc"],
                                   rtol=1e-10, atol=0)
        n = 1 if q == 0 else 2
        np.testing.assert_allclose(np.asarray(out.mg)[:n], reference["M_G"], rtol=1e-10)
        checked += 1
    assert checked > 40


def test_error_factors_and_likelihood_match_numpy(model, arrays, sed):
    rng = np.random.default_rng(3)
    mask = sed.fit_mask(False) & model.support
    mask_f = jnp.asarray(mask.astype(float) * np.asarray(arrays.support, float))
    for _ in range(10):
        m1, q = rng.uniform(0.3, 1.8), rng.uniform(0.1, 1.0)
        age, feh, e = rng.uniform(0.6, 9.0), rng.uniform(-0.8, 0.4), rng.uniform(0, 0.3)
        prediction = model.evaluate(m1, q, age, feh)
        if prediction is None:
            continue
        out = stellar.evaluate(arrays, float(m1), float(q), float(age), float(feh))
        scale = (sed.parallax_mas / 100.0)**2
        columns, variance = model.error_factors(prediction, scale)
        jax_columns, jax_variance = stellar.error_factors(
            arrays, float(np.asarray(out.labels)[0, 0]), out.flux_10pc, scale)
        # numpy drops zero-weight terms; compare the surviving term blocks.
        teff = prediction["teff"][0]
        u = min(max((teff - 3800.0) / 400.0, 0.0), 1.0)
        warm = 10*u**3 - 15*u**4 + 6*u**5
        blocks = [b for b, w in zip((slice(0, 4), slice(4, 8)),
                                    (1 - warm, warm)) if w > 0]
        start = 0
        for block in blocks:
            width = block.stop - block.start
            np.testing.assert_allclose(
                np.asarray(jax_columns)[:, block], columns[:, start:start + width],
                rtol=1e-10, atol=0)
            start += width
        np.testing.assert_allclose(np.asarray(jax_variance), variance, rtol=1e-10, atol=0)
        attenuation = np.exp(-e * extinction_curve(model.wavelength_um))
        value, chi2, flux = neg2_log_likelihood(sed, prediction, model, mask,
                                                sed.parallax_mas, attenuation)
        jax_value, jax_chi2, jax_flux = likelihood.neg2_log_likelihood(
            jnp.asarray(sed.flux), jnp.asarray(sed.error), mask_f,
            jnp.asarray(sed.parallax_mas), out.flux_10pc, jax_columns, jax_variance,
            jnp.asarray(attenuation))
        np.testing.assert_allclose(float(jax_value), value, rtol=1e-10)
        np.testing.assert_allclose(float(jax_chi2), chi2, rtol=1e-10)
        np.testing.assert_allclose(np.asarray(jax_flux), flux, rtol=1e-10)


def test_nelder_mead_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(10):
        matrix = rng.normal(size=(4, 6))
        shift = rng.uniform(-2, 2, size=6)
        bounds = np.array([(-3.0, 3.0)] * 6)

        def host(x):
            return float(np.sum((np.sin(matrix @ x) + 0.05 * (matrix @ (x - shift))**2)**2))

        device_matrix, device_shift = jnp.asarray(matrix), jnp.asarray(shift)
        device = jax.jit(lambda x: jnp.sum((jnp.sin(device_matrix @ x)
                                            + 0.05 * (device_matrix @ (x - device_shift))**2)**2))
        x0 = rng.uniform(-2, 2, size=6)
        steps = np.array([0.3, 0.5, 0.2, 0.4, 0.25, 0.35])
        simplex = np.tile(x0, (7, 1))
        for i in range(6):
            simplex[i + 1, i] += steps[i] if x0[i] + steps[i] <= 3 else -steps[i]
        reference = minimize(host, x0, method="Nelder-Mead", bounds=bounds,
                             options={"maxiter": 1600, "xatol": 1e-5, "fatol": 1e-5,
                                      "initial_simplex": simplex})
        _, values, done = jax.jit(lambda s, b: _nelder_mead(device, s, b))(
            jnp.asarray(simplex), jnp.asarray(bounds))
        assert float(values[0]) == pytest.approx(reference.fun, abs=1e-8)
        assert bool(done) == bool(reference.success)


def test_objective_and_grid_match_numpy(model, arrays, sed):
    mask = sed.fit_mask(False) & model.support
    problem = Problem(
        arr=arrays, sed_flux=jnp.asarray(sed.flux), sed_error=jnp.asarray(sed.error),
        mask=jnp.asarray(mask.astype(float)), parallax_mas=jnp.asarray(sed.parallax_mas),
        parallax_error_mas=jnp.asarray(sed.parallax_error_mas),
        template=jnp.zeros(6), bounds=jnp.asarray(np.array([
            (np.log(0.08), np.log(2.2)), (np.log(0.5), np.log(10.0)), (-1.0, 0.5),
            (0.0, 1.0), (0.0, np.inf), (-3.0, 3.0)])),
        dust_par=jnp.ones(161), dust_mean=jnp.zeros(161) + 0.05,
        dust_sigma=jnp.ones(161) * 0.02, dust_fail=jnp.zeros(161),
        dust_flag=jnp.asarray(1.0), prior_mean=jnp.zeros(3),
        prior_sigma=jnp.ones(3), prior_flag=jnp.zeros(3))
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(20):
        m1, age = rng.uniform(0.25, 1.9), rng.uniform(0.7, 9.0)
        feh, e = rng.uniform(-0.9, 0.4), rng.uniform(0.0, 0.2)
        theta = jnp.asarray([np.log(m1), np.log(age), feh, 0.0, e, 0.0])
        jax_value = float(_objective(theta, problem))
        prediction = model.evaluate(m1, 0.0, age, feh)
        if prediction is None:
            assert jax_value >= 1e29
            continue
        assert jax_value < 1e29
        curve = extinction_curve(model.wavelength_um)
        reference, _, _ = neg2_log_likelihood(sed, prediction, model, mask,
                                              sed.parallax_mas, np.exp(-e * curve))
        reference += ((e - 0.05) / 0.02)**2 + 2 * np.log(0.02) + 2 * log_ndtr(2.5)
        worst = max(worst, abs(jax_value - reference) / reference)
    assert worst < 1e-8

    cfg = dict(binary=True, age_free=True, feh_free=True, q_free=True,
               extinction_free=False, age_gyr=5.0, feh=0.0, q=None, extinction=0.1)
    grid = _grid(cfg, model, 0.05, 0.02, None)
    assert grid.shape == (38 * 3 * 3 * 6, 6)


def test_fit_matches_numpy(model, sed):
    """The headline free route: age, [M/H], parallax and extinction all fitted."""
    prior = EdenhoferPrior(FakeQuery(), sigma=0.02)
    reference = numpy_fit(sed, model=model, age_gyr=None, feh=None, fit_parallax=True,
                          extinction=None, dust_prior=prior)
    result = jax_fit(sed, model=model, age_gyr=None, feh=None, fit_parallax=True,
                     extinction=None, dust_prior=prior)
    for kind in ("single", "binary"):
        # The final restart can settle lower than numpy's endpoint, not higher.
        assert result[kind]["objective"] <= reference[kind]["objective"] + 0.01
        assert result[kind]["converged"]


@pytest.mark.parametrize("fit_parallax", [True, False])
def test_fit_distance_dependent_dust_matches_numpy(model, sed, fit_parallax):
    """Free E under a dust prior whose mean moves across the parallax range."""
    prior = EdenhoferPrior(DistanceQuery(), sigma=0.01)
    reference = numpy_fit(sed, model=model, fit_parallax=fit_parallax, extinction=None,
                          dust_prior=prior)
    result = jax_fit(sed, model=model, fit_parallax=fit_parallax, extinction=None,
                     dust_prior=prior)
    for kind in ("single", "binary"):
        assert result[kind]["objective"] <= reference[kind]["objective"] + 0.01
        np.testing.assert_allclose(result[kind]["objective"],
                                   reference[kind]["objective"], atol=1e-3)
        for field in ("m1", "q", "parallax_mas", "extinction_e"):
            np.testing.assert_allclose(result[kind][field], reference[kind][field],
                                       rtol=1e-3)
        distance = 1000 / result[kind]["parallax_mas"]
        assert result[kind]["dust_prior_mean"] == pytest.approx(0.02 + 1e-3 * distance)
    if not fit_parallax:
        assert result["single"]["parallax_mas"] == sed.parallax_mas


def test_fit_ignores_unobserved_channels(model, sed):
    """Bands without data carry NaN flux and error, as real XP + 2MASS SEDs do."""
    flux, error, mask = sed.flux.copy(), sed.error.copy(), sed.mask.copy()
    flux[64:], error[64:], mask[64:] = np.nan, np.nan, False
    partial = SED(flux, error, mask, sed.parallax_mas, sed.parallax_error_mas,
                  metadata=dict(sed.metadata))
    reference = numpy_fit(partial, model=model)
    result = jax_fit(partial, model=model)
    for kind in ("single", "binary"):
        assert result[kind]["n_fit"] == reference[kind]["n_fit"]
        np.testing.assert_allclose(result[kind]["objective"], reference[kind]["objective"],
                                   atol=1e-4)


def test_fit_skips_endpoints_numpy_rejects(sed):
    """A compiled-model endpoint that numpy calls unsupported falls back to the next."""

    class EdgeModel(StellarModel):
        rejected = None

        def evaluate(self, m1, q=0.0, age_gyr=5.0, feh=0.0):
            if self.rejected is None:
                self.rejected = m1
            if m1 == self.rejected:
                return None
            return super().evaluate(m1, q, age_gyr, feh)

    edge = EdgeModel()
    result = jax_fit(sed, "single", model=edge)
    reference = jax_fit(sed, "single", model=StellarModel())
    assert edge.rejected == reference["m1"] and result["m1"] != edge.rejected
    assert np.isfinite(result["objective"])
    assert result["objective"] >= reference["objective"] - 1e-9


def test_fit_fixed_parameters_matches_numpy(model, sed):
    reference_single = numpy_fit(sed, "single", model=model, age_gyr=5.0, feh=0.0)
    result_single = jax_fit(sed, "single", model=model)
    reference_binary = numpy_fit(sed, "binary", model=model, age_gyr=5.0, feh=0.0, q=0.6)
    result_binary = jax_fit(sed, "binary", model=model, q=0.6)
    for result, reference in ((result_single, reference_single),
                              (result_binary, reference_binary)):
        assert result["objective"] <= reference["objective"] + 0.01
        np.testing.assert_allclose(result["objective"], reference["objective"], atol=1e-4)


def test_fit_hot_route_matches_numpy(sed):
    hot = StellarModel(hot=True)
    prior = EdenhoferPrior(FakeQuery(), sigma=0.02)
    reference = numpy_fit(sed, model=hot, kind="both", age_gyr=None, feh=None,
                          fit_parallax=True, extinction=None, dust_prior=prior)
    result = jax_fit(sed, model=hot, age_gyr=None, feh=None, fit_parallax=True,
                     extinction=None, dust_prior=prior)
    for kind in ("single", "binary"):
        assert result[kind]["objective"] <= reference[kind]["objective"] + 0.01


def test_bfgs_polish_meets_bar(model, sed):
    prior = EdenhoferPrior(FakeQuery(), sigma=0.02)
    reference = numpy_fit(sed, model=model, age_gyr=None, feh=None, fit_parallax=True,
                          extinction=None, dust_prior=prior)
    result = jax_fit(sed, model=model, age_gyr=None, feh=None, fit_parallax=True,
                     extinction=None, dust_prior=prior, optimizer="bfgs+nm")
    for kind in ("single", "binary"):
        assert result[kind]["objective"] <= reference[kind]["objective"] + 0.01


def test_result_fields_and_priors(model, sed):
    result = jax_fit(sed, "binary", model=model, q=0.6, logg_prior=(4.5, 0.3),
                     teff_prior=(5000.0, 200.0))
    for field in ("m1", "m2", "q", "age_gyr", "feh", "parallax_mas", "extinction_e",
                  "dust_prior_penalty", "label_prior_penalty", "logg",
                  "dust_prior_mean", "dust_prior_sigma", "objective", "m2lnl", "chi2",
                  "n_fit", "converged", "at_bounds", "flux", "components", "beta_g",
                  "M_G", "a_phot_over_a1", "route", "mask", "kind"):
        assert field in result
    assert result["label_prior_penalty"] > 0
    assert result["components"].shape == (2, 168)
    assert result["flux"].shape == (168,)
    single = jax_fit(sed, "single", model=model)
    assert single["components"].shape == (1, 168)
    assert single["a_phot_over_a1"] is None and single["beta_g"] == 0


def test_rejects_invalid_arguments(model, sed):
    with pytest.raises(ValueError, match="callable"):
        jax_fit(sed, model=model, q=lambda m, a, f, p: np.nan)
    with pytest.raises(ValueError, match="0.1--1"):
        jax_fit(sed, model=model, q=0.05)
    with pytest.raises(ValueError, match="age_prior requires a free age"):
        jax_fit(sed, model=model, age_gyr=5.0, age_prior=(9.0, 0.3))
    with pytest.raises(ValueError, match="optimizer"):
        jax_fit(sed, model=model, optimizer="gd")
