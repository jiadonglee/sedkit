import numpy as np
import pytest

from sedkit import SED, StellarModel, fit
from sedkit.fit import neg2_log_likelihood


@pytest.fixture(scope="module")
def model():
    return StellarModel()


def test_absolute_binary_and_photocentre(model):
    pair = model.evaluate(0.75, 0.8, 5, 0)
    first = model.evaluate(0.75, 0, 5, 0)
    second = model.evaluate(0.6, 0, 5, 0)
    np.testing.assert_allclose(pair["flux_10pc"], first["flux_10pc"] + second["flux_10pc"])
    twin = model.evaluate(0.75, 1.0, 5, 0)
    np.testing.assert_allclose(twin["flux_10pc"], 2 * first["flux_10pc"])
    assert twin["beta_g"] == 1
    assert twin["a_phot_over_a1"] == 0


def test_no_extrapolation(model):
    assert model.evaluate(0.75, 0.01, 5, 0) is None
    with pytest.raises(ValueError):
        model.evaluate(0.75, 0.8, 15, 0)


def test_main_sequence_between_age_nodes(model):
    assert model.evaluate(1.1, 0, 5, 0) is not None


def test_warm_primary_supported(model):
    assert model.evaluate(1.4, 0.5, 1, 0) is not None
    assert model.evaluate(1.4, 0, 1, -0.5) is None


def test_covariance_likelihood_matches_dense_solve(model):
    pair = model.evaluate(0.75, 0.8, 5, 0)
    flux = pair["flux_10pc"] * 0.25
    sed = SED(flux * 1.02, flux * 0.03, np.ones(168, bool), 50)
    mask = np.zeros(168, bool)
    mask[[5, 12, 30, 45, 61, 62, 63]] = True
    value, chi2, _ = neg2_log_likelihood(sed, pair, model, mask, 50)
    columns, variance = model.error_factors(pair, 0.25)
    covariance = np.diag(sed.error[mask]**2 + variance[mask]) + columns[mask] @ columns[mask].T
    residual = (flux - sed.flux)[mask]
    expected = residual @ np.linalg.solve(covariance, residual)
    assert chi2 == pytest.approx(expected, rel=1e-10)
    assert value == pytest.approx(expected + np.linalg.slogdet(covariance)[1], rel=1e-10)


def test_error_terms_blend_continuously_between_cold_and_warm(model):
    flux = model.evaluate(0.6, 0, 5, 0)["flux_10pc"]

    def covariance(teff):
        columns, variance = model.error_factors(dict(flux_10pc=flux, teff=np.array([teff])))
        return columns @ columns.T + np.diag(variance)

    for teff, term in ((3800.0, "v2.1_cold"), (4200.0, "v2.1_warm")):
        basis = flux[:, None] * model.errors[term + "/basis"]
        pure = basis @ basis.T + np.diag((flux * model.errors[term + "/diag"])**2)
        assert np.allclose(covariance(teff), pure, rtol=1e-12, atol=0)
    step = np.abs(covariance(4001.0) - covariance(3999.0)).max()
    assert step < 0.02 * np.abs(covariance(4200.0) - covariance(3800.0)).max()


def test_mock_recovers_mass_ratio_without_changing_data(model):
    pair = model.evaluate(0.75, 0.8, 5, 0)
    flux = pair["flux_10pc"] * 0.25
    mask = np.zeros(168, bool)
    mask[:66] = True
    sed = SED(flux, flux * 0.03, mask, 50, 0.2)
    before = sed.flux.copy(), sed.error.copy(), sed.mask.copy()
    result = fit(sed, model=model)
    for kind in ("single", "binary"):
        assert result[kind]["parallax_mas"] == sed.parallax_mas
        assert result[kind]["objective"] == pytest.approx(result[kind]["m2lnl"])
    assert result["binary"]["converged"]
    assert result["binary"]["m1"] == pytest.approx(0.75, abs=0.01)
    assert result["binary"]["q"] == pytest.approx(0.8, abs=0.01)
    assert result["delta"] > 0
    np.testing.assert_array_equal(result["single"]["mask"], result["binary"]["mask"])
    for old, new in zip(before, (sed.flux, sed.error, sed.mask)):
        np.testing.assert_array_equal(old, new)


def test_observations_roundtrip_and_wise_holdout(tmp_path):
    sed = SED(np.arange(168.0), np.ones(168), np.ones(168, bool), 50, 0.2,
              "6870202542492887424", {"quality": "A"})
    sed.save(tmp_path / "sed.npz")
    loaded = SED.load(tmp_path / "sed.npz")
    assert loaded.source_id == "6870202542492887424"
    np.testing.assert_array_equal(loaded.flux, sed.flux)
    assert not loaded.fit_mask()[64:66].any()
    assert loaded.fit_mask(use_wise=True)[64:66].all()


def test_spherex_grid_and_copy(model):
    sed = SED(np.full(168, np.nan), np.full(168, np.nan), np.zeros(168, bool), 50)
    result = sed.with_spherex(model.wavelength_um[66:], np.ones(102), np.ones(102))
    assert result.mask[66:].all()
    assert not sed.mask.any()
    with pytest.raises(ValueError):
        sed.with_spherex(model.wavelength_um[66:] + 0.01, np.ones(102), np.ones(102))



def test_free_age_moves_away_from_start_node(model):
    # At 1.4 solar masses only the 1 Gyr start is supported (4 and 9 Gyr lie past the turn-off).
    # Large errors make the age direction shallow compared with mass.
    star = model.evaluate(1.4, 0, 1.7, 0.15)
    flux = star["flux_10pc"] * 0.25
    mask = np.zeros(168, bool)
    mask[:66] = True
    sed = SED(flux, flux * 0.2, mask, 50, 0.05)
    free = fit(sed, "single", age_gyr=None, feh=0.15, model=model)
    fixed = fit(sed, "single", age_gyr=1.7, feh=0.15, model=model)
    assert free["age_gyr"] == pytest.approx(1.7, rel=0.03)
    assert free["objective"] <= fixed["objective"] + 0.01
