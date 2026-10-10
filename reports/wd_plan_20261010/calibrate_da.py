"""WD-only shape calibration and grouped five-fold validation on local DA anchors.

The cohort is within 100 pc, |b|>30 deg, with DESI or Gianninas spectral
labels. Extinction is fixed to zero for this local comparison. The angular
scale is free during shape calibration; cooling-model masses are validated
separately. Run from the repository root with PYTHONPATH=src.
"""

import json
from pathlib import Path
import sys

import numpy as np
from astropy.table import Table, unique
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import brentq

from sedkit import SED, WhiteDwarfModel
from sedkit.whitedwarf import _Hypothesis, _prepare

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/whitedwarf"
OUT = Path(__file__).resolve().parent


def make_calibration(residual, w, train, degree):
    x = np.c_[np.ones(len(w)), w]
    coef = np.zeros((2, 67))
    for j in range(67):
        good = train & np.isfinite(residual[:, j])
        if good.sum() > 10:
            coef[:degree + 1, j] = np.linalg.lstsq(x[good, :degree + 1], residual[good, j], rcond=None)[0]
    remaining = residual - x @ coef
    diag = np.clip(1.4826 * np.nanmedian(np.abs(remaining[train] - np.nanmedian(remaining[train], axis=0)), axis=0), .01, .3)
    a, b, error = np.zeros(168), np.zeros(168), np.full(168, .03)
    a[:61], b[:61], error[:61] = coef[0, 6:], coef[1, 6:], diag[6:]
    return dict(a=a, b=b, diag=error, blue_a=coef[0, :6], blue_b=coef[1, :6], blue_diag=diag[:6])


def main():
    if not (DATA / "anchors/calibration_local100.ecsv").exists() and (OUT/"single_da_anchors.ecsv").exists():
        (DATA/"anchors").mkdir(parents=True,exist_ok=True)
        Table.read(OUT/"single_da_anchors.ecsv").write(DATA/"anchors/calibration_local100.ecsv")
    if not (DATA / "anchors/calibration_local100.ecsv").exists():
        cohort=Table.read(DATA/"anchors/matched_catalogues.ecsv")
        good=cohort["da_label_ready"] & (cohort["ruwe"]<1.4) & (cohort["parallax"]/cohort["parallax_error"]>10)
        good &= np.isin(cohort["catalogue"],["DESI_DA","MWDD_Gianninas2011"]) & (cohort["parallax"]>10)
        cohort=unique(cohort[good],keys="source_id")
        lat=SkyCoord(cohort["ra"]*u.deg,cohort["dec"]*u.deg).galactic.b.deg
        cohort[abs(lat)>30].write(DATA/"anchors/calibration_local100.ecsv")
    cohort = Table.read(DATA / "anchors/calibration_local100.ecsv")
    if not (DATA/"xp_local100/calibrated.npz").exists():
        from xp_data import acquire, calibrate_table
        calibrate_table(acquire(cohort["source_id"],DATA/"xp_local100"),DATA/"xp_local100")
    with np.load(DATA / "xp_local100/calibrated.npz") as archive:
        ids, flux, error = archive["source_id"], archive["flux"], archive["error"]
    index = {int(v): i for i, v in enumerate(ids)}
    cohort = cohort[[int(v) in index and int(v) != 6182278665776280320 for v in cohort["source_id"]]]
    # EC 13198-2849 (LP 911-67) is a catalogued double star in SIMBAD.
    # Its red composite SED is unsuitable for single-DA shape calibration.
    order = [index[int(v)] for v in cohort["source_id"]]
    flux, error = flux[order], error[order]
    raw = WhiteDwarfModel(calibration=None)
    residual, balmer = [], []
    for row, y, err in zip(cohort, flux, error):
        pred = raw.predict(float(row["teff"]), logg=float(row["logg"]), radius=1.)
        shape = np.r_[pred["blue"], pred["flux"][:61]]
        good = (y > 5 * err) & np.isfinite(shape) & (shape > 0)
        core = good & (np.arange(67) >= 6)
        log_ratio = np.full(67, np.nan)
        log_ratio[good] = np.log(y[good] / shape[good])
        log_ratio -= np.nanmean(log_ratio[core])
        residual.append(log_ratio)
        balmer.append(pred["balmer_w"])
    residual, balmer = np.array(residual), np.array(balmer)
    # Fixed shuffle of distinct source IDs; all duplicate measurements of
    # a source were removed before defining the cohort.
    rng = np.random.default_rng(20261010)
    folds = np.empty(len(cohort), int)
    folds[rng.permutation(len(cohort))] = np.arange(len(cohort)) % 5
    cohort["fold"] = folds
    cohort.write(DATA / "anchors/calibration_local100.ecsv", overwrite=True)
    comparisons = {}
    for degree in [0, 1]:
        remaining = np.full_like(residual, np.nan)
        for fold in range(5):
            c = make_calibration(residual, balmer, folds != fold, degree)
            correction = np.r_[c["blue_a"], c["a"][:61]] + balmer[:, None] * np.r_[c["blue_b"], c["b"][:61]]
            remaining[folds == fold] = residual[folds == fold] - correction[folds == fold]
        comparisons[str(degree)] = float(np.sqrt(np.nanmean(remaining[:, 6:]**2)))
    degree = min([0, 1], key=lambda d: comparisons[str(d)])
    final = make_calibration(residual, balmer, np.ones(len(cohort), bool), degree)
    np.savez_compressed(ROOT / "src/sedkit/models/whitedwarf/calibration.npz", **final)
    path=ROOT/"src/sedkit/models/whitedwarf/summary.json"
    model_summary=json.loads(path.read_text());model_summary.update(calibrated=True,
        calibration=f"WD-only exp(a + W b), source-grouped five-fold validation on {len(cohort)} local DA anchors")
    path.write_text(json.dumps(model_summary,indent=2)+"\n")
    rows = []
    for fold in range(5):
        c = make_calibration(residual, balmer, folds != fold, degree)
        fold_dir = DATA / "calibration_folds"
        fold_dir.mkdir(exist_ok=True)
        np.savez_compressed(fold_dir / f"{fold}.npz", **c)
        calibrated = WhiteDwarfModel(calibration=c)
        for i in np.flatnonzero(folds == fold):
            row = cohort[i]
            f, e = np.full(168, np.nan), np.full(168, np.nan)
            f[:61], e[:61] = flux[i, 6:], error[i, 6:]
            sed = SED(f, e, np.arange(168) < 61, float(row["parallax"]),
                      float(row["parallax_error"]), str(row["source_id"]))
            result = dict(source_id=int(row["source_id"]), fold=fold, catalogue=str(row["catalogue"]),
                          teff_spec=float(row["teff"]), logg_spec=float(row["logg"]), G=float(row["phot_g_mean_mag"]))
            for label, model, use_blue, free_radius in [("raw", raw, False, True),
                                                       ("cal", calibrated, False, True),
                                                       ("blue", calibrated, True, True),
                                                       ("mr", calibrated, False, False)]:
                blue = (flux[i, :6], error[i, :6]) if use_blue else None
                # Individual noisy blue samples remain valid linear fluxes.
                problem = _prepare(sed, model, None, 0., None, None, None, False, blue, None, False,
                                   {"logg": (float(row["logg"]), .02)} if free_radius else None, None)
                hyp = _Hypothesis(problem, "wd", free_radius, 5., 0.)
                fit, _ = hyp.fit()
                pars = fit["whitedwarf"]
                result[f"{label}_teff"] = pars["teff"]
                result[f"{label}_mass"] = pars["mass"]
                result[f"{label}_radius"] = pars["radius"]
                result[f"{label}_rms"] = float(np.sqrt(np.mean(((fit["model"][:len(problem.idx)] - problem.y[:len(problem.idx)])
                                                              / fit["model"][:len(problem.idx)])**2)))
                result[f"{label}_converged"] = fit["converged"]
            rows.append(result)
            if len(rows) % 25 == 0:
                print(f"validated {len(rows)}/{len(cohort)}", flush=True)
        Table(rows=rows).write(DATA / "validation_single.ecsv", overwrite=True)
    results = Table(rows=rows)
    summary = dict(n=len(results), cohort="100 pc, |b|>30 deg, DESI/Gianninas spectroscopy; E=0",
                   shape_correction="constant" if degree == 0 else "a + WD_Balmer_W*b",
                   raw_shape_rms=float(np.sqrt(np.nanmean(residual[:, 6:]**2))),
                   cv_shape_rms=comparisons, temperature={})
    for label in ["raw", "cal", "blue", "mr"]:
        delta = results[f"{label}_teff"] / results["teff_spec"] - 1
        summary["temperature"][label] = dict(median=float(np.median(delta)),
            scatter=float(1.4826 * np.median(np.abs(delta - np.median(delta)))),
            rms_spectrum=float(np.median(results[f"{label}_rms"])),
            converged=int(sum(results[f"{label}_converged"])))
    mass_errors=[]
    for r in results:
        supported=[]
        for m in raw.cooling["mass_ax"]:
            try: supported.append((m,raw.physical(r["teff_spec"],m)["logg"]-r["logg_spec"]))
            except ValueError: continue
        for (lo,ylo),(hi,yhi) in zip(supported[:-1],supported[1:]):
            if ylo*yhi<=0:
                mass=brentq(lambda m:raw.physical(r["teff_spec"],m)["logg"]-r["logg_spec"],lo,hi)
                mass_errors.append(r["mr_mass"]-mass);break
    mass_errors=np.array(mass_errors)
    summary["mass_comparison"]=dict(n=len(mass_errors),median_msun=float(np.median(mass_errors)),
        scatter_msun=float(1.4826*np.median(abs(mass_errors-np.median(mass_errors)))),
        reference="mass implied by spectroscopic Teff/logg and the same thick-H cooling tracks")
    (OUT / "validation_single.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
