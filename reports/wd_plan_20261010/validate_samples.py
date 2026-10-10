"""Real DA nulls, APOGEE RV-constant controls and SDSS WDMS comparisons."""

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from astropy.table import Table

from sedkit import SED, StellarModel, WhiteDwarfModel, fit_whitedwarf_companion
from sedkit.extinction import EdenhoferPrior

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/whitedwarf"
OUT = Path(__file__).resolve().parent


class CachedDust:
    """Interpolated native Edenhofer moments precomputed for these control IDs."""
    def __init__(self, record):
        self.record = record

    def moments(self, ra, dec, distance):
        r = self.record
        return float(np.interp(distance, r["d"], r["mean"])), max(.01, float(np.interp(distance, r["d"], r["sigma"])))

    penalty = staticmethod(EdenhoferPrior.penalty)


def one(task):
    kind, row, flux, error, calibration, dust = task
    f, e = np.full(168, np.nan), np.full(168, np.nan)
    f[:61], e[:61] = flux[6:], error[6:]
    sid = str(row["Source"] if kind == "fgk" else row["source_id"])
    plx = row["GAIAEDR3_PARALLAX"] if kind == "fgk" else row["parallax"]
    plx_err = row["GAIAEDR3_PARALLAX_ERROR"] if kind == "fgk" else row["parallax_error"]
    sed = SED(f, e, np.arange(168) < 61, plx, plx_err, sid, metadata=dict(ra=0., dec=0.))
    model = WhiteDwarfModel(calibration=calibration)
    if kind == "fgk":
        result = fit_whitedwarf_companion(sed, model=model, stellar=StellarModel(hot=True),
            extinction=None, dust_prior=CachedDust(dust), fit_parallax=True,
            companion_age_gyr=None, companion_feh=None, companion_prior={"feh": (row["M_H"], .1)})
    elif kind == "fgk_holdout":
        result = fit_whitedwarf_companion(sed, model=model, stellar=StellarModel(hot=True),
            extinction=None, extinction_prior=(0.,.03), fit_parallax=True,
            companion_age_gyr=None, companion_feh=None, companion_prior={"feh": (row["m_h_atm"], .1)})
    elif kind == "wdms":
        result = fit_whitedwarf_companion(sed, model=model, extinction=None, extinction_prior=(0., .03),
            fit_parallax=True, companion_age_gyr=5., companion_feh=0.)
    else:
        result = fit_whitedwarf_companion(sed, model=model, extinction=0.)
    h = result["hypotheses"]
    detection = min(h["dwarf"]["objective"], h["wd"]["objective"]) - h["wd+dwarf"]["objective"]
    row_out = dict(kind=kind, source_id=sid, detection=float(detection), preferred=result["preferred"],
                   composite_converged=h["wd+dwarf"]["converged"], **{f"objective_{k}": float(v["objective"]) for k, v in h.items()})
    row_out["chi2_per_channel"] = {k: float(v["chi2"] / v["n_fit"]) for k,v in h.items() if "chi2" in v}
    if "whitedwarf" in h["wd+dwarf"]:
        row_out.update(wd_teff=h["wd+dwarf"]["whitedwarf"]["teff"],
                       wd_mass=h["wd+dwarf"]["whitedwarf"]["mass"],
                       companion_teff=h["wd+dwarf"]["companion"]["teff"],
                       companion_mass=h["wd+dwarf"]["companion"]["mass"],
                       beta_G=h["wd+dwarf"]["fractions"]["beta_G"])
    if kind == "wdms":
        row_out.update(teff_spec=float(row["Teffwd"]), subtype_spec=int(row["Sp"]))
    return row_out


def tasks(kind):
    directory = DATA / {"fgk": "xp_fgk", "fgk_holdout": "xp_fgk_holdout", "wdms": "xp_wdms", "wd": "xp_local100"}[kind]
    path=DATA/"anchors/calibration_local100.ecsv" if kind=="wd" else directory/"sample.ecsv"
    if not path.exists():
        source={"wd":"single_da", "fgk":"fgk_training", "fgk_holdout":"fgk_holdout", "wdms":"wdms"}[kind]
        path.parent.mkdir(parents=True,exist_ok=True)
        Table.read(OUT/f"{source}_anchors.ecsv").write(path)
    sample=Table.read(path)
    if not (directory/"calibrated.npz").exists():
        from xp_data import acquire,calibrate_table
        ids=sample["Source"] if kind=="fgk" else sample["source_id"]
        calibrate_table(acquire(ids,directory),directory)
    with np.load(directory / "calibrated.npz") as archive:
        ids, flux, error = archive["source_id"], archive["flux"], archive["error"]
    index = {int(v): i for i, v in enumerate(ids)}
    dust = json.loads((OUT / "fgk_dust_moments.json").read_text()) if kind == "fgk" else {}
    output = []
    for r in sample:
        sid = int(r["Source"] if kind == "fgk" else r["source_id"])
        if sid not in index:
            continue
        if kind == "wdms" and not (r["type"] == "DA/M" and r["Sp"] >= 0 and r["parallax"] / r["parallax_error"] > 10):
            continue
        if kind == "wd":
            with np.load(DATA / "calibration_folds" / f"{r['fold']}.npz") as archive:
                calibration = {k: archive[k] for k in archive.files}
        else:
            calibration = "bundled"
        i = index[sid]
        output.append((kind, dict(r), flux[i], error[i], calibration, dust.get(str(sid))))
    return output


def main():
    for kind in sys.argv[1:] or ["wd", "fgk", "wdms"]:
        inputs, rows = tasks(kind), []
        with ProcessPoolExecutor(max_workers=4) as pool, (DATA / f"validation_{kind}.jsonl").open("w") as handle:
            for r in pool.map(one, inputs):
                handle.write(json.dumps(r) + "\n"); handle.flush(); rows.append(r)
                if len(rows) % 25 == 0:
                    print(f"{kind} {len(rows)}/{len(inputs)}", flush=True)
        print(f"{kind} finished {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
