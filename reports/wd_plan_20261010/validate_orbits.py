"""Conditional WD light envelopes for the 31 Yamaguchi et al. RV systems."""

from concurrent.futures import ProcessPoolExecutor
import json
import sys
from pathlib import Path
import numpy as np
from astropy.table import Table
from sedkit import SED, WhiteDwarfModel, whitedwarf_light_limit
from sedkit.extinction import extinction_curve
from sedkit.orbit import campbell, solve_dark_companion
from sedkit.fetch import _photometry, _tap
from sedkit import StellarModel
from xp_data import acquire, calibrate_table

DATA = Path(__file__).resolve().parents[2] / "data/whitedwarf"
OUT = Path(__file__).resolve().parent


def one(task):
    row, orbit, flux, error, ir = task
    f, e = np.full(168, np.nan), np.full(168, np.nan)
    f[:61], e[:61] = flux[6:], error[6:]
    mask=np.arange(168)<61
    pf,pe,pm,_=_photometry([("2MASS",ir,range(3),("j_m","h_m","ks_m"),
                            ("j_msigcom","h_msigcom","ks_msigcom"))],StellarModel().wavelength_um[61:66])
    f[61:64],e[61:64],mask[61:64]=pf[:3],pe[:3],pm[:3]
    sed = SED(f, e, mask, orbit["parallax"], orbit["parallax_error"], str(row["source_id"]))
    # A_V=3.1 E(B-V); convert the adopted literature reddening to native E.
    extinction = 3.1 * row["ebv"] / (1.086 * extinction_curve(np.array([.55]))[0])
    limit = whitedwarf_light_limit(sed, model=WhiteDwarfModel(calibration="bundled"),
        masses=[row["mass_joint"]], temperatures=[6000,8000,10000,13000,16000,20000,25000,32000,40000,55000,80000], extinction=None, extinction_prior=(extinction, .03),
        companion_age_gyr=None, companion_feh=row["feh_init"], fit_parallax=True,
        luminous_mass=row["mass_joint"])
    a0 = campbell(*[orbit[k+"_thiele_innes"] for k in ["a", "b", "f", "g"]])["a0"]
    args = a0, orbit["parallax"], orbit["period"], row["m1_sed"]
    dark = solve_dark_companion(*args)["m2"]
    light = solve_dark_companion(*args, beta=limit["flux_ratio_G_upper"])["m2"] if np.isfinite(limit["beta_G_upper"]) else np.nan
    return dict(source_id=str(row["source_id"]), name=row["name"],
        gaia_rv_consistent=row["name"] not in ["J1834+1525", "J1922-4624"],
        beta_G_upper=limit["beta_G_upper"], mass_dark=dark, mass_light=light,
        mass_shift=light-dark, mass_rv=row["mass_joint"], ratio_joint=row["ratio_joint"],
        luminous_delta=limit["luminous_companion"]["delta"],
        primary_mass=limit["single_primary"]["companion"]["mass"],
        primary_chi2_per_channel=limit["single_primary"]["chi2"] / limit["single_primary"]["n_fit"],
        intervals=limit["intervals"], profile=limit["profile"], delta_threshold=limit["delta_threshold"],n_ir=int(pm[:3].sum()))


def main():
    if not (DATA/"anchors/yamaguchi2024.ecsv").exists():
        (DATA/"anchors").mkdir(parents=True,exist_ok=True)
        Table.read(OUT/"orbit_anchors.ecsv").write(DATA/"anchors/yamaguchi2024.ecsv")
    rows = Table.read(DATA / "anchors/yamaguchi2024.ecsv")
    path=DATA/"anchors/yamaguchi_orbits.vot"
    if not path.exists():
        ids=",".join(map(str,rows["source_id"]))
        _tap("SELECT * FROM gaiadr3.nss_two_body_orbit WHERE source_id IN ("+ids+")").write(path,format="votable")
    orbits = Table.read(DATA / "anchors/yamaguchi_orbits.vot")
    oi = {int(r["source_id"]): dict(r) for r in orbits}
    directory = DATA / "xp_orbits"
    if not (directory / "calibrated.npz").exists():
        calibrate_table(acquire(rows["source_id"], directory), directory)
    path=directory/"tmass.ecsv"
    if not path.exists():
        ids=",".join(map(str,rows["source_id"]))
        _tap("SELECT x.source_id,x.number_of_mates AS n_mates,x.number_of_neighbours AS n_neighbours,tm.* "
             "FROM gaiadr3.tmass_psc_xsc_best_neighbour AS x JOIN gaiadr3.tmass_psc_xsc_join AS j "
             "ON j.clean_tmass_psc_xsc_oid=x.clean_tmass_psc_xsc_oid JOIN gaiadr1.tmass_original_valid AS tm "
             "ON j.original_psc_source_id=tm.designation WHERE x.source_id IN ("+ids+")").write(path)
    ir=Table.read(path)
    xp_only="xp" in sys.argv[1:]
    with np.load(directory / "calibrated.npz") as a:
        fi = {int(v): i for i, v in enumerate(a["source_id"])}
        tasks = [(dict(r), oi[int(r["source_id"])], a["flux"][fi[int(r["source_id"])]],
                  a["error"][fi[int(r["source_id"])]],ir[:0] if xp_only else ir[ir["source_id"]==r["source_id"]]) for r in rows]
    with ProcessPoolExecutor(4) as pool, (DATA / ("validation_orbits_xp.jsonl" if xp_only else "validation_orbits.jsonl")).open("w") as handle:
        for i, row in enumerate(pool.map(one, tasks)):
            handle.write(json.dumps(row)+"\n"); handle.flush()
            print(f"orbits {i+1}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
