"""Metallicity sensitivity of WD temperatures and M-companion subtypes."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from sedkit import SED, WhiteDwarfModel
from sedkit.whitedwarf import _prepare, _Hypothesis
from validate_samples import tasks

DATA=Path(__file__).resolve().parents[2]/"data/whitedwarf"
OUT=Path(__file__).resolve().parent
# Pecaut/Mamajek M0--M9.5V, half-subclass steps.
SUBTYPE_T=np.array([3850,3770,3660,3620,3560,3470,3430,3270,3210,3110,
                    3060,2930,2810,2740,2680,2630,2570,2420,2380,2350])


def one(task):
    _,r,flux,error,_,_=task
    f,e=np.full(168,np.nan),np.full(168,np.nan);f[:61],e[:61]=flux[6:],error[6:]
    sed=SED(f,e,np.arange(168)<61,r["parallax"],r["parallax_error"],str(r["source_id"]))
    p=_prepare(sed,WhiteDwarfModel(),None,None,(0.,.03),None,None,True,None,None,False,None,{"feh":(0.,.3)})
    fit,_=_Hypothesis(p,"wd+dwarf",False,5.,None).fit()
    return dict(source_id=sed.source_id,teff_spec=float(r["Teffwd"]),subtype_spec=int(r["Sp"]),
        teff=fit["whitedwarf"]["teff"],companion_teff=fit["companion"]["teff"],
        companion_feh=fit["companion"]["feh"],beta_G=fit["fractions"]["beta_G"],
        objective=fit["objective"],converged=fit["converged"])


def summarize(rows):
    t=np.array([r["teff"]/r["teff_spec"]-1 for r in rows])
    subtype=np.interp([r["companion_teff"] for r in rows],SUBTYPE_T[::-1],np.arange(0,10,.5)[::-1])
    d=subtype-np.array([r["subtype_spec"] for r in rows])
    summary=dict(n=len(rows),teff_median=float(np.median(t)),
        teff_scatter=float(1.4826*np.median(abs(t-np.median(t)))),
        feh_median=float(np.median([r["companion_feh"] for r in rows])),
        converged=sum(r["converged"] for r in rows),feh_prior=[0.,.3],
        subtype_median=float(np.median(d)),subtype_scatter=float(1.4826*np.median(abs(d-np.median(d)))),
        subtype_within_one=int(sum(abs(d)<=1)),
        subtype_scale="https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt")
    (OUT/"validation_wdms_metallicity.json").write_text(json.dumps(summary,indent=2)+"\n");print(summary)


def main():
    inputs=tasks("wdms");rows=[]
    with ProcessPoolExecutor(4) as pool,(DATA/"validation_wdms_metallicity.jsonl").open("w") as h:
        for i,r in enumerate(pool.map(one,inputs)):
            rows.append(r);h.write(json.dumps(r)+"\n");h.flush()
            if (i+1)%25==0:print(i+1,len(inputs),flush=True)
    summarize(rows)


if __name__=="__main__":main()
