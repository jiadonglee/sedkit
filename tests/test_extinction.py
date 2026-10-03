import numpy as np
import pytest
from scipy.stats import truncnorm

from sedkit import SED, StellarModel, EdenhoferPrior, fit, loglike_sed
from sedkit.extinction import extinction_curve, transmission
from sedkit.fit import neg2_log_likelihood


class SampleQuery:
    integrated = True
    n_samples = 12

    def __init__(self):
        self.distances = []

    def __call__(self, coords, mode="mean"):
        distance = coords.distance.to_value("pc")
        self.distances.append(distance)
        assert coords.ra.deg == pytest.approx(30)
        assert coords.dec.deg == pytest.approx(-10)
        return 0.15 * distance / 100 if mode == "mean" else 0.04


def mock_sed(model, extinction=0.15):
    pair = model.evaluate(0.75, 0.8, 5, 0)
    flux = pair["flux_10pc"] * 0.01 * transmission(model.wavelength_um, extinction)
    mask = np.zeros(168, bool)
    mask[:64] = True
    return SED(flux, flux * 0.02, mask, 10, 0.05, metadata={"ra": 30, "dec": -10}), pair


def test_reddened_likelihood_matches_dense_covariance():
    model = StellarModel()
    sed, pair = mock_sed(model)
    attenuation = transmission(model.wavelength_um, 0.15)
    mask = np.zeros(168, bool)
    mask[[5, 12, 30, 45, 61, 62, 63]] = True
    sed.flux *= 1.02
    value, chi2, flux = neg2_log_likelihood(sed, pair, model, mask, 10, attenuation)
    columns, variance = model.error_factors(pair, 0.01)
    columns *= attenuation[:, None]
    variance *= attenuation**2
    covariance = np.diag(sed.error[mask]**2 + variance[mask]) + columns[mask] @ columns[mask].T
    residual = (flux - sed.flux)[mask]
    expected = residual @ np.linalg.solve(covariance, residual)
    assert chi2 == pytest.approx(expected)
    assert value == pytest.approx(expected + np.linalg.slogdet(covariance)[1])
    assert loglike_sed(sed, m1=.75, q=.8, model=model, extinction=-.1) == -np.inf


def test_zgr23_curve_uses_optical_depth_and_covers_spherex():
    wave = np.array([.392, 1.235, 1.662, 2.159, 4.982])
    np.testing.assert_allclose(extinction_curve(wave)[:4],
                               [4.00402975, .68214726, .40827751, .26954260])
    np.testing.assert_allclose(transmission(wave, .2), np.exp(-.2 * extinction_curve(wave)))
    np.testing.assert_array_equal(transmission(wave, 0), np.ones(5))
    assert 0 < extinction_curve(wave)[-1] < .09808245


def test_edenhofer_distance_width_and_normalization():
    query = SampleQuery()
    prior = EdenhoferPrior(query)
    assert prior.moments(30, -10, 100) == pytest.approx((.15, .04))
    assert prior.moments(30, -10, 200) == pytest.approx((.3, .04))
    mean, sigma, e = .01, .03, .04
    expected = -2 * truncnorm.logpdf(e, -mean / sigma, np.inf, loc=mean, scale=sigma)
    assert prior.penalty(e, mean, sigma) + np.log(2*np.pi) == pytest.approx(expected)
    assert prior.penalty(-.01, mean, sigma) == np.inf
    query.n_samples = None
    with pytest.raises(ValueError, match="load_samples"):
        EdenhoferPrior(query)
    assert EdenhoferPrior(query, sigma=.08).moments(30, -10, 100) == pytest.approx((.15, .08))
    query.integrated = False
    with pytest.raises(ValueError, match="integrated"):
        EdenhoferPrior(query, sigma=.08)


def test_joint_extinction_fit_preserves_data_and_shared_prior():
    model = StellarModel()
    sed, _ = mock_sed(model)
    before = sed.flux.copy(), sed.error.copy(), sed.mask.copy()
    query = SampleQuery()
    result = fit(sed, fit_parallax=True, model=model, extinction=None, dust_prior=EdenhoferPrior(query))
    binary = result["binary"]
    assert binary["converged"]
    assert binary["extinction_e"] == pytest.approx(.15, abs=.01)
    assert binary["m1"] == pytest.approx(.75, abs=.01)
    assert binary["q"] == pytest.approx(.8, abs=.02)
    assert np.ptp(query.distances) > 0
    for solution in (result["single"], binary):
        assert solution["dust_prior_mean"] == pytest.approx(.15 * 10 / solution["parallax_mas"])
        assert solution["dust_prior_sigma"] == .04
        z = (solution["parallax_mas"] - 10) / .05
        assert solution["objective"] == pytest.approx(
            solution["m2lnl"] + z*z + solution["dust_prior_penalty"])
        np.testing.assert_allclose(solution["components"].sum(axis=0), solution["flux"])
    np.testing.assert_array_equal(result["single"]["mask"], binary["mask"])
    for old, new in zip(before, (sed.flux, sed.error, sed.mask)):
        np.testing.assert_array_equal(old, new)


def test_fixed_extinction_needs_no_map():
    model = StellarModel()
    sed, _ = mock_sed(model)
    result = fit(sed, "binary", model=model, q=.8, extinction=.15, fit_parallax=False)
    assert result["m1"] == pytest.approx(.75, abs=.01)
    assert result["extinction_e"] == .15
    assert result["dust_prior_mean"] is None
    expected = -2 * loglike_sed(sed, m1=result["m1"], q=.8, model=model, extinction=.15)
    assert result["m2lnl"] == pytest.approx(expected)


def test_dust_fit_requires_coordinates_and_map_support():
    model = StellarModel()
    sed, _ = mock_sed(model)
    prior = EdenhoferPrior(SampleQuery())
    sed.metadata = {}
    with pytest.raises(ValueError, match="ra/dec"):
        fit(sed, "single", model=model, extinction=None, dust_prior=prior)
    sed.metadata = {"ra": 30, "dec": -10}
    prior.query = lambda coords, mode: np.nan
    with pytest.raises(ValueError, match="no support"):
        fit(sed, "single", model=model, extinction=None, dust_prior=prior)


def test_dustmaps_integrates_posterior_samples(tmp_path):
    pytest.importorskip("dustmaps")
    from astropy.io import fits
    from dustmaps.edenhofer2023 import Edenhofer2023Query

    densities = np.array([.001, .0015, .002])[:, None, None] * np.ones((3, 3, 12))
    primary = fits.PrimaryHDU(densities)
    primary.header["NSIDE"] = 1
    primary.header["ORDERING"] = "NESTED"
    centers = fits.BinTableHDU.from_columns([
        fits.Column(name="radial pixel centers", format="D", array=[75., 125., 175.])])
    edges = fits.BinTableHDU.from_columns([
        fits.Column(name="radial pixel boundaries", format="D", array=[50., 100., 150., 200.])])
    path = tmp_path / "samples.fits"
    fits.HDUList([primary, centers, edges]).writeto(path)
    query = Edenhofer2023Query(map_fname=str(path), integrated=True, load_samples=True)
    prior = EdenhoferPrior(query)
    mean, sigma = prior.moments(30, -10, 100)
    expected = np.array([.001, .0015, .002]) * np.sqrt(50 * 100)
    assert mean == pytest.approx(expected.mean())
    assert sigma == pytest.approx(expected.std())
    with pytest.raises(ValueError, match="no support"):
        prior.moments(30, -10, 250)
