"""Place a star on the HR diagram and fit the hypotheses of its region.

locate() reads Gaia G, BP-RP and 2MASS Ks at the parallax and assigns one
region; fit_star() runs only that region's route. Hypotheses inside a region
share one data vector and likelihood; objectives of different regions are
not comparable.
"""
from functools import lru_cache

import numpy as np

from .extinction import extinction_curve
from .fit import fit
from .giant import _ks_zero_point, fit_giant_companion
from .model import StellarModel
from .subdwarf import fit_subdwarf_companion
from .whitedwarf import fit_whitedwarf_companion

BINARY_MAG = 0.75
LOGTE_STEP = 0.01
PASSBAND_UM = dict(G=0.622, BP=0.511, RP=0.777, Ks=2.159)
REGIONS = {
    "ms": ("dwarf", "ms+ms", "wd+dwarf"),
    "hot_ms": ("single", "binary"),
    "giant": ("giant", "giant+companion"),
    "sdb": ("fgk", "sdb", "sdb+dwarf", "sdb+subgiant"),
    "wd": ("dwarf", "wd", "wd+dwarf"),
    "unsupported": (),
}
SIMPLEST_FIRST = ("dwarf", "single", "fgk", "wd", "sdb", "ms+ms", "binary", "wd+dwarf",
                  "sdb+dwarf", "sdb+subgiant")


@lru_cache(maxsize=1)
def _stellar():
    return StellarModel()


@lru_cache(maxsize=2)
def _parsec_points(route):
    """Single stars the route evaluates, at the PARSEC age and [M/H] nodes:
    log Teff, M_G, M_Ks, [M/H].

    "network": StellarModel() at 0.5--10 Gyr below 7500 K.
    "hot": StellarModel(hot=True) inside the hot-table support, 7000--30000 K.
    """
    model = _stellar() if route == "network" else StellarModel(hot=True)
    masses = np.geomspace(0.08, model.mass_range[1], 600)
    low, high = model.age_range_gyr
    rows = []
    for metal, ages in model.tracks.items():
        for log_age in ages:
            age = 10**(log_age - 9)
            if not low - 1e-12 <= age <= high:
                continue
            labels, physical = model._track(masses, age, metal)
            teff = labels[:, 0]
            if route == "network":
                keep = model.in_domain(labels) & (teff < 7500)
            else:
                keep = model.in_hot_domain(teff, physical[:, 0], metal) & (teff >= 7000)
            keep &= np.isfinite(teff)
            rows.append(np.c_[np.log10(teff[keep]), labels[keep, 1] + labels[keep, 2],
                              labels[keep, 1], np.full(keep.sum(), metal)])
    return np.vstack(rows)


@lru_cache(maxsize=2)
def _dwarf_band(route):
    """Per log Teff bin, the M_G range of one route's supported stars; the
    bright edge is raised by BINARY_MAG for unresolved equal-mass binaries.
    Bins without stars of that route are NaN."""
    points = _parsec_points(route)
    edges = np.arange(np.log10(2300), np.log10(31000) + LOGTE_STEP, LOGTE_STEP)
    low = np.full(len(edges) - 1, np.inf)
    high = np.full(len(edges) - 1, -np.inf)
    index = np.digitize(points[:, 0], edges) - 1
    np.minimum.at(low, index, points[:, 1])
    np.maximum.at(high, index, points[:, 1])
    filled = np.isfinite(low)
    centres = 0.5 * (edges[1:] + edges[:-1])
    low = np.interp(centres, centres[filled], low[filled], left=np.nan, right=np.nan)
    high = np.interp(centres, centres[filled], high[filled], left=np.nan, right=np.nan)
    return edges, low - BINARY_MAG, low, high + 0.3


@lru_cache(maxsize=1)
def _gks_relation():
    """Solar-metallicity PARSEC dwarf sequence, G-Ks against log Teff."""
    points = np.vstack([_parsec_points("network"), _parsec_points("hot")])
    solar = points[points[:, 3] == 0.0]
    colour = solar[:, 1] - solar[:, 2]
    bins = np.arange(colour.min(), colour.max() + 0.05, 0.05)
    index = np.digitize(colour, bins)
    keys = [i for i in np.unique(index) if np.sum(index == i) >= 3]
    x = np.array([np.median(colour[index == i]) for i in keys])
    y = np.array([np.median(solar[index == i, 0]) for i in keys])
    order = np.argsort(x)
    return x[order], y[order]


def _teff_from_gks(g_ks):
    """Teff of the solar dwarf with this G-Ks; redder than every supported
    dwarf maps to 2300 K, bluer to 31 kK."""
    x, y = _gks_relation()
    return float(10**np.interp(g_ks, x, y, left=np.log10(31000), right=np.log10(2300)))


def _bolometric_correction_g(teff):
    """Andrae et al. (2018) BC_G, held flat beyond 3300--8000 K."""
    t = np.clip(teff, 3300, 8000) - 5772
    if teff >= 4000:
        coefficients = (6.000e-02, 6.731e-05, -6.647e-08, 2.859e-11, -7.197e-15)
    else:
        coefficients = (1.749e+00, 1.977e-03, 3.737e-07, -8.966e-11, -4.183e-14)
    return sum(c * t**k for k, c in enumerate(coefficients))


def _ks_magnitude(sed):
    """2MASS Ks from the SED's Ks channel, NaN when it is masked."""
    if not sed.mask[63] or sed.flux[63] <= 0:
        return np.nan
    return float(-2.5 * np.log10(sed.flux[63] / _ks_zero_point(_stellar())))


def _region(mg, bp_rp, teff):
    """Region and reason from dereddened M_G, BP-RP and the G-Ks Teff."""
    if mg > 10 + 2.6 * bp_rp and bp_rp < 1.6:
        return "wd", "below the main sequence at white-dwarf luminosity"
    if bp_rp < 0.0 and 2.8 < mg < 7:
        return "sdb", "blue and below the upper main sequence: hot subdwarf box"
    if not np.isfinite(teff):
        return "unsupported", "no Ks for a Teff estimate"
    if teff > 30000:
        return "unsupported", "hotter than 30 kK"
    if teff < 2800:
        return "unsupported", "late M or ultracool dwarf, redder than the supported PARSEC dwarfs"
    edges, top, single_top, bottom = _dwarf_band("network")
    _, hot_top, _, hot_bottom = _dwarf_band("hot")
    i = int(np.clip(np.digitize(np.log10(teff), edges) - 1, 0, len(top) - 1))
    if teff < 7500 and top[i] <= mg <= bottom[i]:
        if mg < single_top[i]:
            return "ms", "main sequence, brighter than one star: binary, young or metal-rich"
        return "ms", "main sequence"
    if teff >= 7000 and hot_top[i] <= mg <= hot_bottom[i]:
        return "hot_ms", "A/B main sequence"
    if mg > np.nanmax([bottom[i], hot_bottom[i]]):
        return "unsupported", "below the supported main sequence: metal-poor, or a parallax or photometry problem"
    if teff >= 7000 and mg > np.nanmin([top[i], hot_top[i]]):
        return "unsupported", "between the network and hot-table supports"
    mbol = mg + _bolometric_correction_g(teff)
    logg = 4.438 + np.log10(1.1) + 4 * np.log10(teff / 5772) + 0.4 * (mbol - 4.74)
    if logg >= 4.15:
        return "ms", "above the main sequence by more than an equal-mass binary: young, metal-rich or a triple"
    if teff < 3600:
        return "unsupported", "cool giant below 3600 K"
    if teff > 6800:
        return "unsupported", "hot evolved star (Hertzsprung gap, horizontal branch or supergiant)"
    if logg >= 3.8:
        return "unsupported", "subgiant or turn-off star, between the dwarf network and the giant template"
    return "giant", "red giant"


def locate(sed, *, extinction=0.0, dust_prior=None):
    """Place a star on the HR diagram.

    Uses Gaia G and BP-RP from sed.metadata (phot_g_mean_mag,
    phot_bp_mean_mag, phot_rp_mean_mag), the Ks channel and the parallax.
    extinction is ZGR23 E; with dust_prior=EdenhoferPrior the map mean at the
    parallax distance replaces it. Returns the region, the reason, the
    region's hypotheses and the dereddened M_G, BP-RP, G-Ks and the Teff of
    the solar PARSEC dwarf with that G-Ks.
    """
    names = ("phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag")
    if any(sed.metadata.get(k) is None or not np.isfinite(sed.metadata[k]) for k in names):
        raise ValueError("locate needs Gaia G, BP and RP magnitudes in sed.metadata")
    if not np.isfinite(sed.parallax_mas) or sed.parallax_mas <= 0:
        raise ValueError("locate needs a positive parallax")
    if dust_prior is not None:
        extinction = float(dust_prior.moments(sed.metadata["ra"], sed.metadata["dec"],
                                              1000 / sed.parallax_mas)[0])
    absorption = dict(zip(PASSBAND_UM, 2.5 * np.log10(np.e) * extinction
                          * extinction_curve(np.array(list(PASSBAND_UM.values())))))
    g, bp, rp = (float(sed.metadata[k]) for k in names)
    mg = g + 5 * np.log10(sed.parallax_mas / 100) - absorption["G"]
    bp_rp = bp - rp - absorption["BP"] + absorption["RP"]
    g_ks = g - _ks_magnitude(sed) - absorption["G"] + absorption["Ks"]
    teff = _teff_from_gks(g_ks) if np.isfinite(g_ks) else np.nan
    region, reason = _region(mg, bp_rp, teff)
    return dict(region=region, reason=reason, hypotheses=REGIONS[region], M_G=float(mg),
                bp_rp=float(bp_rp), g_ks=float(g_ks), teff_estimate=float(teff),
                extinction_e=float(extinction))


def _preferred(objectives, tie):
    """The simplest hypothesis within `tie` of the lowest objective."""
    best = min(objectives.values())
    order = sorted(objectives, key=lambda k: SIMPLEST_FIRST.index(k) if k in SIMPLEST_FIRST else 99)
    return next(k for k in order if objectives[k] <= best + tie)


def fit_star(sed, *, region=None, labels=None, extinction=0.0, extinction_prior=None, dust_prior=None,
             fit_parallax=False, age_gyr=None, feh=0.0, tie=1.0, options=None):
    """Locate a star and fit the hypotheses of its HR-diagram region.

    Regions and hypotheses:
      ms      dwarf, ms+ms (coeval, free q) and wd+dwarf in the WD-route frame
      hot_ms  single and coeval binary with StellarModel(hot=True)
      giant   fit_giant_companion; needs labels=dict(teff=, logg=, feh=) as
              (mean, sigma) on the APOGEE scale
      sdb     fit_subdwarf_companion
      wd      fit_whitedwarf_companion
    region= overrides the location. extinction is a fixed ZGR23 E, or None
    to fit it under extinction_prior=(mean, sigma) or dust_prior; the giant
    route always fits E and takes a fixed value as a 0.01-wide prior.
    age_gyr and feh apply to the dwarfs of the ms, wd and hot_ms regions;
    None fits them (age 0.5--10 Gyr on the network, from 4 Myr on the hot
    route). With a free age, warm primaries trade age against a companion.
    options are keyword arguments of the region's route function and take
    precedence.

    Returns the location, the route's own result and, where the route
    compares hypotheses by one objective, their objectives, deltas and the
    simplest hypothesis within `tie` of the lowest. Objectives are
    comparable only inside one region, and no composite-detection threshold
    is calibrated for the ms region.
    """
    options = dict(options or {})
    guess = extinction if extinction is not None else extinction_prior[0] if extinction_prior else 0.0
    location = locate(sed, extinction=guess, dust_prior=dust_prior)
    if region is not None:
        if region not in REGIONS:
            raise ValueError(f"region must be one of {sorted(REGIONS)}")
        location = dict(location, region=region, hypotheses=REGIONS[region],
                        reason=f"set by region= (located: {location['region']})")
    region = location["region"]
    out = dict(location=location, region=region, result=None, objectives=None, delta=None,
               preferred=None)
    if region == "unsupported":
        return out
    if extinction is not None and (extinction_prior is not None or dust_prior is not None):
        raise ValueError("a fixed extinction takes no extinction prior")
    shared = dict(extinction=extinction, dust_prior=dust_prior, fit_parallax=fit_parallax)
    if region in ("ms", "wd"):
        kwargs = dict(shared, hypotheses=REGIONS[region], extinction_prior=extinction_prior,
                      companion_age_gyr=age_gyr, companion_feh=feh)
        result = fit_whitedwarf_companion(sed, **dict(kwargs, **options))
        objectives = {k: r["objective"] for k, r in result["hypotheses"].items()}
    elif region == "hot_ms":
        if extinction_prior is not None:
            raise ValueError("the hot route takes extinction as a number or a dust_prior")
        kwargs = dict(shared, kind="both", model=StellarModel(hot=True), age_gyr=age_gyr, feh=feh)
        result = fit(sed, **dict(kwargs, **options))
        objectives = {"single": result["single"]["objective"], "binary": result["binary"]["objective"]}
    elif region == "sdb":
        kwargs = dict(shared, extinction_prior=extinction_prior)
        result = fit_subdwarf_companion(sed, **dict(kwargs, **options))
        objectives = {k: r["objective"] for k, r in result["hypotheses"].items()}
    else:
        if labels is None or set(labels) != {"teff", "logg", "feh"}:
            out["location"] = dict(location, reason=location["reason"]
                                   + "; the giant route needs labels=dict(teff=, logg=, feh=)")
            return out
        if extinction is not None:
            extinction_prior = (extinction, 0.01)
        kwargs = dict(extinction_prior=extinction_prior, dust_prior=dust_prior)
        result = fit_giant_companion(sed, labels["teff"], labels["logg"], labels["feh"],
                                     **dict(kwargs, **options))
        rows = result["rows"]
        alone = [r for r in rows if r["m2"] == 0]
        companion = min((r for r in rows if r["m2"] > 0), key=lambda r: r["objective"])
        objectives = {"giant": alone[0]["objective"], "giant+companion": companion["objective"]}
    best = min(objectives.values())
    out.update(result=result, objectives=objectives,
               delta={k: v - best for k, v in objectives.items()},
               preferred=_preferred(objectives, tie) if region != "giant" else
               ("giant" if objectives["giant"] <= best + tie else "giant+companion"))
    return out
