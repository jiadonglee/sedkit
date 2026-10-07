import numpy as np
import pytest

from sedkit import SED, StellarModel, GiantTemplate, fit_giant_companion
from sedkit.giant import CHANNELS, _companion, tilted_transmission


@pytest.fixture(scope="module")
def template():
    return GiantTemplate()


@pytest.fixture(scope="module")
def hot():
    return StellarModel(hot=True)


def test_template_is_normalised_and_positive(template):
    flux, columns, diag = template.evaluate(4800.0, 2.5, -0.3)
    wave = template.wavelength_um
    assert flux.shape == (CHANNELS,) and columns.shape[0] == CHANNELS
    assert np.all(flux[:61] > 0) and np.all(diag[:61] > 0)
    assert abs(flux[(wave >= 0.55) & (wave <= 0.95)].mean() - 1) < 0.01
    np.testing.assert_allclose(wave, StellarModel().wavelength_um[:CHANNELS], atol=1e-6)


def test_template_is_continuous_across_nodes(template):
    teff = np.linspace(4750.0, 4850.0, 201)        # crosses the 4800 K node
    flux = np.array([template.evaluate(t, 2.5, -0.3)[0] for t in teff])
    assert np.abs(np.diff(np.log(flux[:, :64]), axis=0)).max() < 0.002


def test_template_support_is_explicit(template):
    assert template.evaluate(4800.0, 2.5, -0.3) is not None
    assert template.evaluate(9000.0, 2.5, -0.3) is None     # beyond the grid
    assert template.evaluate(5800.0, 0.2, 0.4) is None      # no metal-rich luminous warm giants


def mock(template, hot, m2, parallax=0.5, extinction=0.6):
    wave = hot.wavelength_um[:CHANNELS]
    attenuation = tilted_transmission(wave, extinction)
    flux = np.full(168, np.nan)
    flux[:CHANNELS] = 2000.0 * template.evaluate(4800.0, 2.5, -0.3)[0] * attenuation
    if m2:
        flux[:CHANNELS] += _companion(hot, m2, 0.01, (parallax / 100)**2, wave)["flux"] * attenuation
    error = 0.01 * flux
    mask = np.zeros(168, bool)
    mask[:CHANNELS] = True
    return SED(flux, error, mask, parallax, 0.02, "1")


def test_mock_giant_alone_rejects_a_luminous_companion(template, hot):
    result = fit_giant_companion(mock(template, hot, 0), (4800, 100), (2.5, 0.2), (-0.3, 0.15),
                                 m2_grid=(3.0, 6.0), template=template, model=hot, threshold=25)
    rows = {r["m2"]: r for r in result["rows"]}
    assert result["best_m2"] == 0.0
    assert rows[0.0]["chi2"] < 5
    assert abs(rows[0.0]["extinction_e"] - 0.6) < 0.02
    assert rows[3.0]["delta"] > 25 and result["m2_excluded"] == 3.0


def test_mock_companion_is_found(template, hot):
    result = fit_giant_companion(mock(template, hot, 6.0), (4800, 100), (2.5, 0.2), (-0.3, 0.15),
                                 m2_grid=(6.0,), template=template, model=hot)
    rows = {r["m2"]: r for r in result["rows"]}
    assert result["best_m2"] == 6.0 and result["detection"] > 25
    assert rows[6.0]["companion_blue_fraction"] > 0.2


def test_inputs_are_validated(template, hot):
    sed = mock(template, hot, 0)
    with pytest.raises(ValueError):
        fit_giant_companion(sed, (4800, 0), (2.5, 0.2), (-0.3, 0.15), m2_grid=(), template=template, model=hot)
    with pytest.raises(ValueError):
        fit_giant_companion(sed, (9000, 100), (2.5, 0.2), (-0.3, 0.15), m2_grid=(), template=template, model=hot)


class FlatQuery:
    """Integrated map with E = mean and width sigma at every position and distance."""
    integrated = True
    n_samples = 12

    def __init__(self, mean, sigma=0.02):
        self.mean, self.sigma = mean, sigma

    def __call__(self, coords, mode="mean"):
        return self.mean if mode == "mean" else self.sigma


def parsec_mock(template, hot, m2, parallax=1.0, extinction=0.3, brighter_mag=0.0):
    """Giant at (4800 K, 2.5, -0.3) with the PARSEC mode of M_Ks there, optionally made brighter."""
    from sedkit.giant import _parsec_prior, _ks_zero_point, KS
    wave = hot.wavelength_um[:CHANNELS]
    prior = _parsec_prior()
    node = tuple(int(np.argmin(np.abs(a - x))) for a, x in zip(template.axes, (4800.0, 2.5, -0.3)))
    mks = prior.mks_grid[np.argmin(prior.pen[node].astype(float))] - brighter_mag
    shape = template.evaluate(4800.0, 2.5, -0.3)[0]
    scale = _ks_zero_point(hot) * 10**(-0.4 * (mks - 5 * np.log10(parallax / 100))) / shape[KS]
    attenuation = tilted_transmission(wave, extinction)
    flux = np.full(168, np.nan)
    flux[:CHANNELS] = scale * shape * attenuation
    if m2:
        flux[:CHANNELS] += _companion(hot, m2, 0.01, (parallax / 100)**2, wave)["flux"] * attenuation
    mask = np.zeros(168, bool)
    mask[:CHANNELS] = True
    return SED(flux, 0.01 * flux, mask, parallax, 0.02, "1", metadata={"ra": 30.0, "dec": -10.0})


def test_parsec_luminosity_accepts_a_typical_giant(template, hot):
    result = fit_giant_companion(parsec_mock(template, hot, 0), (4800, 100), (2.5, 0.2), (-0.3, 0.15),
                                 m2_grid=(3.0,), template=template, model=hot, luminosity="parsec")
    alone = result["rows"][0]
    assert result["best_m2"] == 0.0
    assert alone["luminosity_penalty"] < 1 and abs(alone["z"]) < 0.5
    assert 0.5 < alone["mass"] < 3.0
    assert alone["parallax_mas"] == pytest.approx(1.0 + 0.02 * alone["z"])
    assert result["rows"][1]["delta"] > 25


def test_parsec_luminosity_keeps_a_companion(template, hot):
    result = fit_giant_companion(parsec_mock(template, hot, 6.0), (4800, 100), (2.5, 0.2), (-0.3, 0.15),
                                 m2_grid=(6.0,), template=template, model=hot, luminosity="parsec")
    assert result["best_m2"] == 6.0 and result["detection"] > 25


def test_mass_walls_charge_an_overluminous_giant(template, hot):
    sed = parsec_mock(template, hot, 0, brighter_mag=3.5)
    result = fit_giant_companion(sed, (4800, 100), (2.5, 0.2), (-0.3, 0.15), m2_grid=(),
                                 template=template, model=hot, luminosity="massfree")
    alone = result["rows"][0]
    assert alone["mass"] > 10 and alone["mass_penalty"] > 1
    assert alone["luminosity_penalty"] == pytest.approx(alone["mass_penalty"])


def test_dust_prior_inside_and_beyond_the_map(template, hot):
    from sedkit import EdenhoferPrior
    inside = fit_giant_companion(parsec_mock(template, hot, 0, parallax=1.0), (4800, 100), (2.5, 0.2),
                                 (-0.3, 0.15), m2_grid=(), template=template, model=hot,
                                 luminosity="parsec", dust_prior=EdenhoferPrior(FlatQuery(0.3)))
    row = inside["rows"][0]
    assert row["e_map"] == pytest.approx(0.3) and row["dust_penalty"] < 1
    beyond = fit_giant_companion(parsec_mock(template, hot, 0, parallax=0.5), (4800, 100), (2.5, 0.2),
                                 (-0.3, 0.15), m2_grid=(), template=template, model=hot,
                                 luminosity="parsec", dust_prior=EdenhoferPrior(FlatQuery(0.6)))
    row = beyond["rows"][0]
    assert row["distance_pc"] > 1250 and row["e_map"] == pytest.approx(0.6)
    assert row["dust_penalty"] == pytest.approx(((0.56 - row["extinction_e"]) / 0.04)**2
                                                if row["extinction_e"] < 0.56 else 0.0)


def test_luminosity_options_are_validated(template, hot):
    from sedkit import EdenhoferPrior
    sed = parsec_mock(template, hot, 0)
    kw = dict(m2_grid=(), template=template, model=hot)
    with pytest.raises(ValueError):
        fit_giant_companion(sed, (4800, 100), (2.5, 0.2), (-0.3, 0.15), luminosity="free", **kw)
    with pytest.raises(ValueError):
        fit_giant_companion(sed, (4800, 100), (2.5, 0.2), (-0.3, 0.15), luminosity="parsec",
                            parallax=(1.0, 0.0), **kw)
    with pytest.raises(ValueError):
        fit_giant_companion(sed, (4800, 100), (2.5, 0.2), (-0.3, 0.15), extinction_prior=(0.3, 0.05),
                            dust_prior=EdenhoferPrior(FlatQuery(0.3)), **kw)
