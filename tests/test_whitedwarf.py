import numpy as np
import pytest

from sedkit import SED, StellarModel, WhiteDwarfModel, fit_whitedwarf_companion, whitedwarf_light_limit
from sedkit import loglike_whitedwarf_sed
from sedkit.orbit import photocentre_a0, solve_dark_companion, solve_luminous_pair


@pytest.fixture(scope="module")
def wd():
    return WhiteDwarfModel()


def mock(wd, teff=15000, mass=.6, companion=None):
    p = wd.predict(teff, mass)
    flux = p["flux"].copy()
    cool = None
    if companion is not None:
        cool = StellarModel().evaluate(companion, 0, 5, 0)["flux_10pc"]
        flux += cool
    sed = SED(flux * .01, flux * .0001, wd.support, 10, .02, "mock")
    beta = None if cool is None else wd.passband(p["coarse"], "G")
    return sed, p, cool, beta


def test_cooling_relation_units_and_support(wd):
    p = wd.physical(10000, .6)
    assert abs(p["radius"] - .01283) < 2e-5
    assert abs(p["cooling_age_gyr"] - .633) < .002
    assert abs(p["logg"] - 8.) < .003
    assert abs(wd.teff_at_age(.6, p["cooling_age_gyr"]) - 10000) < 1
    for args in [(5000, .6), (90000, .6), (15000, 1.4)]:
        with pytest.raises(ValueError):
            wd.predict(*args)
    with pytest.raises(ValueError):
        wd.predict(15000, logg=6.0, radius=.03)
    assert not wd.support[64:66].any()
    assert not wd.support[-1]
    thin = WhiteDwarfModel(hydrogen_layer="thin").physical(10000,.6)
    assert 0 < thin["radius"] < p["radius"]


def test_single_da_and_composite_recovery(wd):
    sed, _, _, _ = mock(wd)
    result = fit_whitedwarf_companion(sed, model=wd)
    assert result["hypotheses"]["wd"]["converged"]
    p = result["hypotheses"]["wd"]["whitedwarf"]
    assert abs(p["teff"] / 15000 - 1) < .01 and abs(p["mass"] - .6) < .01
    assert result["delta"]["dwarf"] > 100
    sed, _, _, _ = mock(wd, teff=18000, companion=.35)
    result = fit_whitedwarf_companion(sed, model=wd)
    assert result["preferred"] == "wd+dwarf"
    p = result["hypotheses"]["wd+dwarf"]
    assert abs(p["whitedwarf"]["teff"] / 18000 - 1) < .05
    assert abs(p["companion"]["mass"] - .35) < .02
    assert 0 < p["fractions"]["beta_G"] < 1
    assert p["photocentre"]["star1"] == "whitedwarf"
    assert p["photocentre"]["coefficient"] == pytest.approx(
        p["photocentre"]["mass_fraction_star1"]-p["fractions"]["beta_G"])
    assert result["mask"].sum() == 64


def test_free_radius_validation_mode(wd):
    p = wd.predict(13000, logg=8.1, radius=.016)
    sed = SED(p["flux"] * .01, p["flux"] * .0002, wd.support, 10, .02, "free")
    result = fit_whitedwarf_companion(sed, model=wd, companions=False, free_radius=True,
                                    whitedwarf_prior={"logg": (8.1, .01)})
    p = result["hypotheses"]["wd"]["whitedwarf"]
    assert abs(p["radius"] / .016 - 1) < .03
    assert p["cooling_age_gyr"] is None


def test_light_limit_and_orbit_flux_ratio_conversion(wd):
    f = StellarModel().evaluate(.9, 0, 5, 0)["flux_10pc"]
    sed = SED(f * .01, f * .0001, wd.support, 10, .02, "primary")
    result = whitedwarf_light_limit(sed, model=wd, masses=[.6],
                                    temperatures=[6000, 10000, 20000, 40000, 80000], luminous_mass=.6)
    assert any(r["teff"] == 6000 for r in result["profile"])
    assert .005 < result["beta_G_upper"] < .1
    assert result["confidence_level"] is None
    assert result["luminous_companion"]["delta"] > 100
    assert result["intervals"]
    beta = .02
    # photocentre_a0 takes star 1's light fraction; solve_dark_companion takes F2/F1.
    a0 = photocentre_a0(.9, .6, 1 - beta, 500, 10)
    mass = solve_dark_companion(a0, 10, 500, .9, beta=beta / (1 - beta))
    assert abs(mass["m2"] - .6) < 1e-6


def test_shared_nuisance_input_checks(wd):
    sed, _, _, _ = mock(wd)
    with pytest.raises(ValueError):
        fit_whitedwarf_companion(sed, model=wd, extinction=None)
    with pytest.raises(ValueError):
        fit_whitedwarf_companion(sed, model=wd, extinction=0, extinction_prior=(0, .1))
    with pytest.raises(ValueError):
        fit_whitedwarf_companion(sed, model=wd, blue=([1], [1]))
    with pytest.raises(ValueError):
        whitedwarf_light_limit(sed, temperatures=[5000,10000])
    with pytest.raises(ValueError):
        fit_whitedwarf_companion(sed, whitedwarf_prior={"metallicity":(0.,.1)})


def test_fitted_age_metallicity_and_parallax():
    stellar=StellarModel()
    f=stellar.evaluate(1.2,0,2.,-.2)["flux_10pc"]
    sed=SED(.01*f,.0002*f,np.arange(168)<61,10,.1,"nuisance")
    result=fit_whitedwarf_companion(sed,companion_age_gyr=None,companion_feh=None,
        companion_prior={"feh":(-.2,.1)},fit_parallax=True)
    primary=result["hypotheses"]["dwarf"]
    assert primary["converged"] and primary["chi2"]/primary["n_fit"]<1
    assert abs(primary["parallax_mas"]-10)<.3


def test_wd_likelihood_atom_excludes_parallax_prior(wd):
    sed,_,_,_=mock(wd,teff=18000,companion=.35)
    options=dict(teff=18000,mass=.6,primary_mass=.35,model=wd,parallax_mas=10)
    correct=loglike_whitedwarf_sed(sed,**options)
    assert np.isfinite(correct)
    assert loglike_whitedwarf_sed(sed,**{**options,"teff":25000})<correct
    sed.parallax_mas=15;sed.parallax_error_mas=.001
    assert loglike_whitedwarf_sed(sed,**options)==correct
    assert loglike_whitedwarf_sed(sed,**{**options,"teff":5000})==-np.inf


def test_wd_dominated_composite_uses_luminous_pair_branches():
    beta=.9
    a0=photocentre_a0(.6,.2,beta,500,10)
    roots=solve_luminous_pair(a0,10,500,m2=.2,beta=beta)
    wd_branch=next(r for r in roots if r["branch"]=="B<beta")
    assert wd_branch["m1"]==pytest.approx(.6,abs=1e-7)


def test_uv_total_flux_bound_refines_temperature_boundary(wd):
    primary=StellarModel().evaluate(.9,0,5,0)["flux_10pc"]
    sed=SED(.01*primary,.0002*primary,np.arange(168)<61,10,.02,"uv_cap")
    pred=wd.predict(10000,.6)
    upper=.01*wd.passband(pred["coarse"],"FUV")*np.exp(wd.calibration["uv_a"][0])
    result=whitedwarf_light_limit(sed,model=wd,masses=[.6],
        temperatures=[6000,8000,10000,13000,20000,40000,80000],galex_upper_limits={"FUV":upper})
    assert 0<result["beta_G_upper"]<.005
    assert 10000<max(r["teff"] for r in result["profile"] if r["allowed"])<13000
    assert any(not np.isfinite(r["objective"]) for r in result["profile"])


def test_uv_bound_refits_shared_extinction(wd):
    from sedkit.extinction import extinction_curve
    primary=StellarModel().evaluate(.9,0,5,0)["flux_10pc"]
    pred=wd.predict(18000,.6)
    flux=.01*(primary+pred["flux"])*np.exp(-.15*extinction_curve(wd.wavelength_um))
    sed=SED(flux,.02*flux,np.arange(168)<61,10,.02,"reddened_uv")
    coarse=.01*pred["coarse"]*np.exp(-.15*wd.coarse_curve)
    cap=wd.passband(coarse,"FUV")*np.exp(wd.calibration["uv_a"][0])
    result=whitedwarf_light_limit(sed,model=wd,masses=[.6],temperatures=[8000,18000,35000],
        extinction=None,extinction_prior=(0.,.2),galex_upper_limits={"FUV":cap})
    injected=next(r for r in result["profile"] if r["teff"]==18000)
    assert injected["allowed"] and injected["extinction_e"]>.08
    assert all(8000<=r["teff"]<=35000 for r in result["profile"])
