from pathlib import Path

import numpy as np
import pytest

from sedkit import SED, StellarModel
from sedkit.orbit import amrf, amrf_observed, dark_mass_ratio, rank_roots, solve_amrf, solve_orbit

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def model():
    return StellarModel()


def test_dark_mass_ratio_inverts_amrf():
    for q in (0.1, 0.4, 0.8):
        assert np.isclose(dark_mass_ratio(amrf(q, 0.0)), q, atol=1e-6)
    assert np.isnan(dark_mass_ratio(amrf(1.0, 0.0) + 0.01))
    assert np.isclose(amrf_observed(1.0, 10.0, 365.25, 1.0), 0.1)
    assert np.isclose(amrf_observed(1.0, 10.0, 8 * 365.25, 1.0), 0.025)


def test_faint_and_luminous_roots(model):
    roots = solve_amrf(0.14, 0.50, model=model)
    assert [r["kind"] for r in roots] == ["dark", "luminous"]
    dark, bright = roots
    assert np.isclose(dark["q"], dark_mass_ratio(0.14))
    assert 0.8 < bright["q"] < 0.9 and 0.4 < bright["beta_G"] < 0.7
    assert np.isclose(amrf(bright["q"], bright["beta_G"]), 0.14, atol=2e-3)
    assert bright["delta_Ks"] > bright["delta_G"] > 0.3
    assert np.isclose(bright["m2"], 0.5 * bright["q"])


def test_rank_roots_on_observed_candidates(model):
    cases = {"1916454200349735680": ((0.4404, 26.953, 238.50, 0.644), "luminous"),
             "5148853253106611200": ((0.6978, 13.913, 339.57, 0.686), "dark")}
    for sid, (orbit, best) in cases.items():
        sed = SED.load(FIXTURES / f"gaia_dr3_{sid}.npz")
        ranked = rank_roots(sed, solve_orbit(*orbit, model=model), parallax_mas=orbit[1], model=model)
        assert ranked[0]["kind"] == best and ranked[0]["delta"] == 0
        assert ranked[1]["delta"] > 100
        a0_mas, parallax, period, _ = orbit
        for r in ranked:
            # the fitted system reproduces the photocentre orbit
            a0 = parallax * r["m1"]**(1 / 3) * (period / 365.25)**(2 / 3) * amrf(r["q"], r["beta_G"])
            assert np.isclose(a0, a0_mas, rtol=1e-3)
            assert np.isclose(r["m1"], r["fit"]["m1"]) and np.isclose(r["m2"], r["q"] * r["m1"])
            if r["fitted_as"] == "binary":
                assert np.isclose(r["q"], r["fit"]["q"]) and np.isclose(r["beta_G"], r["fit"]["beta_g"])
