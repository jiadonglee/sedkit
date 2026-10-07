import numpy as np
import pytest

from sedkit import SED, StellarModel, fit, loglike_sed
from sedkit.model import HOT_BLEND_K, HOT_CHANNELS


@pytest.fixture(scope="module")
def hot():
    return StellarModel(hot=True)


def test_default_model_keeps_its_domain():
    model = StellarModel()
    assert model.support.all()
    with pytest.raises(ValueError):
        model.evaluate(3.0, 0, 0.1, 0)


def test_hot_star_predicts_xp_and_2mass_only(hot):
    star = hot.evaluate(4.0, 0, 0.05, 0)
    assert star["route"] == "PARSEC+" + hot.hot_summary["version"]
    assert 14000 < star["teff"][0] < 16000
    assert np.isfinite(star["flux_10pc"][:HOT_CHANNELS]).all()
    assert np.isnan(star["flux_10pc"][HOT_CHANNELS:]).all()
    assert not hot.support[HOT_CHANNELS:].any()


def test_hot_binary_adds_components(hot):
    pair = hot.evaluate(4.0, 0.5, 0.05, 0)
    first = hot.evaluate(4.0, 0, 0.05, 0)
    second = hot.evaluate(2.0, 0, 0.05, 0)
    np.testing.assert_allclose(pair["flux_10pc"][:HOT_CHANNELS],
                               (first["flux_10pc"] + second["flux_10pc"])[:HOT_CHANNELS])


def test_hot_support_is_explicit(hot):
    assert hot.evaluate(15.0, 0, 0.005, 0) is None      # above 30 kK
    assert hot.evaluate(4.0, 0, 0.05, 0.5) is None      # outside the solar-table [M/H] range
    assert hot.evaluate(60.0, 0, 0.005, 0) is None      # beyond the isochrone


def test_network_hands_over_to_hot_table_continuously(hot):
    masses = np.linspace(1.36, 1.60, 600)
    stars = [hot.evaluate(m, 0, 0.5, 0) for m in masses]
    teff = np.array([s["teff"][0] for s in stars])
    flux = np.array([s["flux_10pc"][:HOT_CHANNELS] for s in stars])
    assert teff[0] < HOT_BLEND_K[0] and teff[-1] > HOT_BLEND_K[1]
    step = np.abs(np.diff(np.log(flux), axis=0)).max()
    assert step < 0.01
    columns = [hot.error_factors(s)[0][:HOT_CHANNELS] for s in stars]
    covariance = [c @ c.T for c in columns]
    jump = max(np.abs(b - a).max() for a, b in zip(covariance[:-1], covariance[1:]))
    assert jump < 0.05 * max(np.abs(c).max() for c in covariance)


def test_hot_mock_fit_recovers_mass_and_age(hot):
    star = hot.evaluate(4.0, 0, 0.05, 0)
    flux = star["flux_10pc"] * 0.25
    mask = np.zeros(168, bool)
    mask[:HOT_CHANNELS] = True
    sed = SED(np.where(mask, flux, np.nan), np.where(mask, 0.02 * flux, np.nan), mask, 50, 0.05)
    result = fit(sed, "single", model=hot, age_gyr=None, feh=0.0)
    assert result["m1"] == pytest.approx(4.0, rel=0.02)
    assert result["age_gyr"] == pytest.approx(0.05, rel=0.2)
    assert result["mask"][HOT_CHANNELS:].sum() == 0
    assert np.isfinite(loglike_sed(sed, m1=4.0, age_gyr=0.05, model=hot))


def test_label_priors_enter_both_hypotheses(hot):
    star = hot.evaluate(3.0, 0.7, 0.03, 0)
    flux = star["flux_10pc"] * 0.25
    mask = np.zeros(168, bool)
    mask[:HOT_CHANNELS] = True
    sed = SED(np.where(mask, flux, np.nan), np.where(mask, 0.02 * flux, np.nan), mask, 50, 0.05)
    result = fit(sed, "both", model=hot, age_gyr=None, feh=0.0, age_prior=(7.48, 0.1),
                 logg_prior=(float(star["logg"][0]), 0.1))
    for kind in ("single", "binary"):
        r = result[kind]
        assert r["objective"] == pytest.approx(r["m2lnl"] + r["label_prior_penalty"], abs=1e-6)
    assert result["binary"]["q"] == pytest.approx(0.7, abs=0.1)
    assert result["delta"] > 25
    with pytest.raises(ValueError):
        fit(sed, "single", model=hot, age_gyr=0.03, age_prior=(7.48, 0.1))
    teff = fit(sed, "single", model=hot, age_gyr=0.03, feh=0.0, teff_prior=(float(star["teff"][0]), 300.0))
    assert teff["objective"] == pytest.approx(teff["m2lnl"] + teff["label_prior_penalty"], abs=1e-6)
