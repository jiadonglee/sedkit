import numpy as np
import pytest

from sedkit import SED, StellarModel
from sedkit.extinction import extinction_curve
from sedkit.orbit import (branch_from_rv, campbell, mass_from_rv, photocentre_a0, solve_dark_companion,
                          solve_luminous_pair, thiele_innes)
from sedkit.subdwarf import (GALEX_BRIGHT, SubdwarfModel, fit_subdwarf_companion, physical)


@pytest.fixture(scope="module")
def sdm():
    return SubdwarfModel()


@pytest.fixture(scope="module")
def stellar():
    return StellarModel()


def test_table_reproduces_nodes_and_interpolates(sdm):
    t = sdm.tiers["H"]
    i, j = 5, 6
    node = sdm.predict(t["teff_ax"][i], t["logg_ax"][j], 1.0, "H")["flux"]
    raw = SubdwarfModel(correction=None, calibration=None).predict(t["teff_ax"][i], t["logg_ax"][j], 1.0, "H")
    np.testing.assert_allclose(raw["flux"], np.exp(t["ln_flux"][i, j]), rtol=1e-6)
    mid = sdm.predict(0.5 * (t["teff_ax"][i] + t["teff_ax"][i + 1]), t["logg_ax"][j], 1.0, "H")["flux"]
    upper = sdm.predict(t["teff_ax"][i + 1], t["logg_ax"][j], 1.0, "H")["flux"]
    assert np.all(np.minimum(node, upper)[:64] <= mid[:64] * (1 + 1e-9))
    assert np.all(mid[:64] <= np.maximum(node, upper)[:64] * (1 + 1e-9))
    doubled = sdm.predict(30000.0, 5.6, 0.2, "H")["flux"] / sdm.predict(30000.0, 5.6, 0.1, "H")["flux"]
    np.testing.assert_allclose(doubled, 4.0)


def test_table_is_not_extrapolated(sdm):
    for teff, logg, tier in ((19000.0, 5.6, "H"), (46000.0, 5.6, "H"), (30000.0, 6.6, "H"),
                             (30000.0, 5.6, "He"), (30000.0, 5.6, "mid"), (33000.0, 4.9, "He")):
        assert not sdm.in_support(teff, logg, tier)
        with pytest.raises(ValueError):
            sdm.predict(teff, logg, 0.2, tier)
    assert sdm.in_support(33000.0, 5.6, "He")


def test_mass_from_gravity_and_radius():
    mass, luminosity = physical(28000.0, 5.6, 0.2)
    g = 6.674e-8 * 0.47 * 1.989e33 / (0.2 * 6.957e10)**2     # 0.47 Msun at 0.2 Rsun
    assert abs(mass - 0.47 * 10**5.6 / g) < 1e-6
    assert abs(luminosity - 0.04 * (28000.0 / 5772.0)**4) < 1e-6


def test_galex_thresholds_follow_the_local_roll_off():
    assert abs(GALEX_BRIGHT["FUV"] - (18.82 - 2.5 * np.log10(114.0))) < 1e-9
    assert abs(GALEX_BRIGHT["NUV"] - (20.08 - 2.5 * np.log10(303.0))) < 1e-9


def _mock(sdm, stellar, sd, companion=None, parallax=2.0, e=0.05, noise=0.01):
    """Absolute fluxes of a subdwarf (teff, logg, radius, tier) and optional dwarf (mass, age, feh)."""
    scale = (parallax / 100)**2
    att = np.exp(-e * extinction_curve(sdm.wavelength_um))
    flux = sdm.predict(*sd)["flux"] * scale * att
    beta = None
    if companion is not None:
        cool = stellar.evaluate(companion[0], 0.0, companion[1], companion[2])["flux_10pc"] * scale * att
        beta = flux.copy()
        flux = flux + cool
    mask = np.zeros(168, bool)
    mask[:64] = True
    rng = np.random.default_rng(3)
    observed = flux * (1 + noise * rng.standard_normal(168))
    return SED(observed, noise * flux, mask, parallax, 0.02, "1"), beta


def test_single_subdwarf_is_recovered(sdm, stellar):
    sed, _ = _mock(sdm, stellar, (31000.0, 5.7, 0.16, "H"))
    # XP and J/H/Ks barely separate Teff from E for a hot star: E is fixed at the truth here
    r = fit_subdwarf_companion(sed, model=sdm, stellar=stellar, companions=(), tiers=("H",),
                               extinction=0.05, subdwarf_prior=dict(logg=(5.7, 0.1)))
    sd = r["hypotheses"]["sdb"]["subdwarf"]
    assert r["preferred"] == "sdb"
    assert abs(sd["teff"] - 31000.0) < 1500 and abs(sd["radius"] / 0.16 - 1) < 0.05
    lo, hi = r["hypotheses"]["sdb"]["ranges"]["subdwarf_teff"]
    assert lo <= sd["teff"] <= hi


def test_composite_is_preferred_and_light_fraction_recovered(sdm, stellar):
    sed, _ = _mock(sdm, stellar, (33000.0, 5.7, 0.16, "H"), companion=(1.1, 3.0, 0.0), parallax=2.5)
    r = fit_subdwarf_companion(sed, model=sdm, stellar=stellar, companions=("dwarf",), tiers=("H",),
                               extinction_prior=(0.05, 0.02), subdwarf_prior=dict(logg=(5.7, 0.1)),
                               companion_age_gyr=3.0, companion_feh=0.0)
    h = r["hypotheses"]["sdb+dwarf"]
    assert r["preferred"] == "sdb+dwarf" and r["delta"]["sdb"] > 100
    assert abs(h["companion"]["mass"] - 1.1) < 0.05
    # beta_G of the truth from the same passband integral
    truth = fit_subdwarf_companion(sed, model=sdm, stellar=stellar, companions=("dwarf",), tiers=("H",),
                                   extinction=0.05, subdwarf_prior=dict(teff=(33000.0, 1.0), logg=(5.7, 0.001)),
                                   companion_age_gyr=3.0, companion_feh=0.0, companion_mass=1.1)
    assert abs(h["fractions"]["beta_G"] - truth["hypotheses"]["sdb+dwarf"]["fractions"]["beta_G"]) < 0.03


def test_subgiant_companion_drops_spherex_and_reports_fractions(sdm, stellar):
    sed, _ = _mock(sdm, stellar, (33000.0, 5.7, 0.16, "H"), companion=(1.1, 3.0, 0.0), parallax=2.5)
    sed.mask[66:] = True
    sed.flux[66:], sed.error[66:] = 1.0, 0.1
    r = fit_subdwarf_companion(sed, model=sdm, stellar=stellar, companions=("subgiant",), tiers=("H",),
                               extinction=0.05, subdwarf_prior=dict(logg=(5.7, 0.1)), use_spherex=True)
    h = r["hypotheses"]["sdb+subgiant"]
    assert not r["mask"][66:].any()
    assert 0 < h["fractions"]["beta_G"] < 1 and np.isnan(h["fractions"]["channels"][66:]).all()
    assert h["companion"]["radius"] > 0 and 3.2 <= h["companion"]["logg"] <= 3.8


def test_inputs_outside_the_route_are_refused(sdm, stellar):
    sed, _ = _mock(sdm, stellar, (31000.0, 5.7, 0.16, "H"))
    galex = {"FUV": (15.0, 0.05, True), "NUV": (15.2, 0.03, True)}
    with pytest.raises(ValueError):       # the helium tiers have no NUV spectrum
        fit_subdwarf_companion(sed, model=sdm, companions=(), tiers=("H", "He"), extinction=0.0, galex=galex)
    with pytest.raises(ValueError):
        fit_subdwarf_companion(sed, model=sdm, companions=("giant",), extinction=0.0)
    with pytest.raises(ValueError):       # a fitted E needs one prior
        fit_subdwarf_companion(sed, model=sdm, companions=())


def test_luminous_pair_branches_round_trip():
    for m_sd, beta in ((0.47, 0.30), (0.47, 0.20), (0.55, 0.45)):
        a0 = photocentre_a0(m_sd, 1.2, beta, 800.0, 2.0)
        rows = {r["branch"]: r for r in solve_luminous_pair(a0, 2.0, 800.0, 1.2, beta)}
        b = m_sd / (m_sd + 1.2)
        true = "B>beta" if b > beta else "B<beta"
        assert abs(rows[true]["m1"] - m_sd) < 1e-6
    rows = solve_luminous_pair(photocentre_a0(0.47, 1.2, 0.3, 800, 2.0), 2.0, 800, 1.2, 0.3,
                               errors=dict(beta=0.01))
    assert all(2.0 < r["dm1_dbeta"] < 3.0 for r in rows)


def test_photocentre_phase_cannot_choose_the_branch():
    one = thiele_innes(1.0, 40.0, 30.0, 70.0)
    flipped = thiele_innes(1.0, 40.0, 210.0, 250.0)
    np.testing.assert_allclose(one, flipped, atol=1e-12)
    c = campbell(*one)
    assert abs(c["inclination"] - 40) < 1e-6 and abs(c["omega"] - 30) < 1e-6 and abs(c["node"] - 70) < 1e-6


def test_rv_orbit_picks_the_branch_without_beta():
    m_sd, m_ms, i, p, plx = 0.47, 1.0, 50.0, 900.0, 2.0
    b = m_sd / (m_sd + m_ms)
    a = plx * ((m_sd + m_ms) * (p / 365.25)**2)**(1 / 3)
    k_ms = 2 * np.pi * a / plx * 1.496e8 * b * np.sin(np.radians(i)) / (p * 86400)
    assert abs(mass_from_rv(k_ms, p, 0.0, i, m_ms) - m_sd) < 1e-3   # AU and G Msun to 1e-4
    for beta in (0.15, 0.55):
        a0 = photocentre_a0(m_sd, m_ms, beta, p, plx)
        out = branch_from_rv(a0, plx, p, m_ms, k_ms, 0.0, i, beta_sed=beta + 0.02, beta_sed_error=0.03)
        assert out["chosen"] == ("B>beta" if b > beta else "B<beta")
        chosen = [r for r in out["branches"] if r["branch"] == out["chosen"]][0]
        assert abs(chosen["beta"] - beta) < 1e-3


def test_dark_companion_of_a_subdwarf():
    a0 = photocentre_a0(0.47, 0.6, 1.0, 600.0, 3.0)     # companion dark in G: photocentre on the subdwarf
    out = solve_dark_companion(a0, 3.0, 600.0, 0.47)
    assert abs(out["m2"] - 0.6) < 1e-6
