"""Transfer SDSS WD decomposition normalizations to the Gaia G passband."""

import json
from pathlib import Path
import numpy as np
from astropy.table import Table
from sedkit import WhiteDwarfModel

DATA=Path(__file__).resolve().parents[2]/"data/whitedwarf"
OUT=Path(__file__).resolve().parent


def main():
    sample=Table.read(DATA/"xp_wdms/sample.ecsv")
    normal=Table.read(DATA/"anchors/wdms_normalizations.ecsv")
    index={(int(r["PLT"]),int(r["MJD"]),int(r["FIB"])):r for r in normal}
    fits={r["source_id"]:r for r in map(json.loads,(DATA/"validation_wdms_variants.jsonl").read_text().splitlines())}
    with np.load(DATA/"xp_wdms/calibrated.npz") as a: ids,flux,wave=a["source_id"],a["flux"],a["wavelength_nm"]
    fi={int(v):i for i,v in enumerate(ids)}
    model=WhiteDwarfModel(calibration=None);rows=[]
    for r in sample:
        sid=str(r["source_id"])
        if sid not in fits:continue
        n=index[(int(r["PLT"]),int(r["MJD"]),int(r["FIB"]))]
        if not (n["Mwd"]>0 and n["dwd"]>0 and 0<n["dwde"]/n["dwd"]<.3):continue
        # The original decomposition distance encodes its fitted angular
        # normalization. Recover the catalogue radius from M and g when
        # the web catalogue's Rwd entry is absent (zero).
        radius=float(n["Rwd"])
        if radius<=0:radius=np.sqrt(6.6743e-8*n["Mwd"]*1.988409870698051e33/10**n["logg"])/6.957e10
        pred=model.predict(n["Teffwd"],logg=n["logg"],radius=radius)
        observed=np.interp(model.coarse_wavelength_nm,wave,flux[fi[int(sid)]])
        beta=model.passband(pred["coarse"],"G")*(10/n["dwd"])**2/model.passband(observed,"G")
        rows.append(dict(source_id=sid,beta_reference=float(beta),beta_fit=fits[sid]["solar"]["beta_G"],
            relative_distance_error=float(n["dwde"]/n["dwd"]),physical_reference=bool(0<beta<1)))
    (DATA/"validation_beta.json").write_text(json.dumps(rows,indent=2)+"\n")
    d=np.array([r["beta_fit"]-r["beta_reference"] for r in rows])
    weights=model.passbands["G"];outside=(model.coarse_wavelength_nm<wave[0])|(model.coarse_wavelength_nm>wave[-1])
    summary=dict(n=len(rows),median=float(np.median(d)),scatter=float(1.4826*np.median(abs(d-np.median(d)))),
        within_005=int(sum(abs(d)<=.05)),passband_weight_outside_xp=float(weights[outside].sum()),
        physical_reference=sum(r["physical_reference"] for r in rows),
        reference_above_one=sum(r["beta_reference"]>=1 for r in rows),
        reference="SDSS decomposition angular normalization; Koester transfer to G, independent of Gaia parallax but sharing a DA atmosphere family")
    (OUT/"validation_beta.json").write_text(json.dumps(summary,indent=2)+"\n");print(summary)


if __name__=="__main__":main()
