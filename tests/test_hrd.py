from pathlib import Path

import numpy as np
import pytest

from sedkit import SED, StellarModel, WhiteDwarfModel
from sedkit.hrd import _region, fit_star, locate

FIXTURE = Path(__file__).parent / "fixtures" / "gaia_dr3_5148853253106611200.npz"


@pytest.mark.parametrize("point, region", [
    ((12.0, 0.2, np.nan), "wd"),
    ((4.5, -0.3, np.nan), "sdb"),
    ((5.0, 0.8, 5800.0), "ms"),
    ((0.0, 0.3, 7800.0), "hot_ms"),
    ((2.3, 0.4, 7200.0), "ms"),
    ((0.15, 0.2, 7200.0), "hot_ms"),
    ((0.5, 1.2, 4700.0), "giant"),
    ((9.0, 1.0, 5000.0), "unsupported"),
    ((14.0, 4.0, 2500.0), "unsupported"),
    ((-4.0, -0.3, 35000.0), "unsupported"),
    ((-1.0, 2.0, 3400.0), "unsupported"),
])
def test_region_rules(point, region):
    assert _region(*point)[0] == region


def test_locate_real_sb2_on_main_sequence():
    location = locate(SED.load(FIXTURE))
    assert location["region"] == "ms"
    assert location["hypotheses"] == ("dwarf", "ms+ms", "wd+dwarf")
    assert 4000 < location["teff_estimate"] < 5000
    with pytest.raises(ValueError):
        locate(SED(np.ones(168), np.ones(168), np.ones(168, bool), 10.0))


def _mock_binary(m1, q):
    stellar = StellarModel()
    pred = stellar.evaluate(m1, q, 5, 0)
    flux = pred["flux_10pc"] * .01
    g = -2.5 * np.log10(np.sum(10**(-0.4 * pred["M_G"]))) + 5
    metadata = dict(phot_g_mean_mag=g, phot_bp_mean_mag=g + 0.6, phot_rp_mean_mag=g - 0.6)
    return SED(flux, flux * .01, WhiteDwarfModel().support & stellar.support, 10, .02, "mock", metadata)


def test_main_sequence_binary_fits_three_hypotheses():
    result = fit_star(_mock_binary(.9, .7))
    assert result["region"] == "ms"
    assert set(result["objectives"]) == {"dwarf", "ms+ms", "wd+dwarf"}
    assert result["preferred"] == "ms+ms"
    assert abs(result["result"]["hypotheses"]["ms+ms"]["companion"]["q"] - .7) < .02
    assert result["delta"]["dwarf"] > 100


def test_single_star_is_not_promoted_to_a_composite():
    result = fit_star(_mock_binary(.8, 0))
    assert result["preferred"] == "dwarf"
    assert result["delta"]["ms+ms"] < 2


def test_unsupported_and_label_free_giant_return_no_fit():
    sed = SED.load(FIXTURE)
    assert fit_star(sed, region="unsupported")["result"] is None
    giant = fit_star(sed, region="giant")
    assert giant["result"] is None and "labels" in giant["location"]["reason"]
    with pytest.raises(ValueError):
        fit_star(sed, region="nowhere")
