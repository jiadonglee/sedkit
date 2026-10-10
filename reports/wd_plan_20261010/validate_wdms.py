"""Compare the WDMS fit with infrared and spectral-gravity constraints."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from astropy.table import Table
from sedkit import SED, StellarModel, WhiteDwarfModel
from sedkit.fetch import _photometry
from sedkit.whitedwarf import _Hypothesis, _prepare
from validate_samples import tasks

DATA = Path(__file__).resolve().parents[2] / "data/whitedwarf"


def one(task):
    original, ir = task
    kind, row, flux, error, calibration, _ = original
    out = dict(source_id=str(row["source_id"]), teff_spec=float(row["Teffwd"]), subtype_spec=int(row["Sp"]))
    f, e, mask = np.full(168, np.nan), np.full(168, np.nan), np.arange(168)<61
    f[:61], e[:61] = flux[6:], error[6:]
    pf, pe, pm, _ = _photometry([("2MASS", ir, range(3), ("j_m","h_m","ks_m"),
                                     ("j_msigcom","h_msigcom","ks_msigcom"))], StellarModel().wavelength_um[61:66])
    f[61:64], e[61:64] = pf[:3], pe[:3]
    for label, use_ir, free, blue in [("solar",False,False,False), ("ir",True,False,False),
                                      ("free",True,True,False), ("blue",True,True,True)]:
        mask[61:64] = pm[:3] if use_ir else False
        sed = SED(f,e,mask,row["parallax"],row["parallax_error"],out["source_id"])
        p = _prepare(sed, WhiteDwarfModel(calibration="bundled"), None, None, (0.,.03), None,
                     None, True, (flux[:6],error[:6]) if blue else None, None, False,
                     {"logg":(float(row["logg"]),.15)} if free else None, None)
        fit,_ = _Hypothesis(p,"wd+dwarf",free,5.,0.).fit()
        out[label] = dict(teff=fit["whitedwarf"]["teff"], mass=fit["whitedwarf"]["mass"],
            companion_teff=fit["companion"]["teff"], beta_G=fit["fractions"]["beta_G"],
            chi2_per_channel=fit["chi2"]/fit["n_fit"], converged=fit["converged"], n_ir=int(pm[:3].sum()) if use_ir else 0)
    return out


def main():
    ir = Table.read(DATA/"xp_wdms/tmass.vot")
    if "SOURCE_ID" in ir.colnames: ir.rename_column("SOURCE_ID","source_id")
    inputs = [(t, ir[ir["source_id"] == t[1]["source_id"]]) for t in tasks("wdms")]
    with ProcessPoolExecutor(4) as pool, (DATA/"validation_wdms_variants.jsonl").open("w") as h:
        for i,r in enumerate(pool.map(one,inputs)):
            h.write(json.dumps(r)+"\n");h.flush()
            if (i+1)%25==0: print(f"WDMS {i+1}/{len(inputs)}",flush=True)


if __name__ == "__main__": main()
