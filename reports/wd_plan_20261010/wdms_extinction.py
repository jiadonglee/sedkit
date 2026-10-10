"""Extinction-prior sensitivity of the distant WDMS comparison sample."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from sedkit import SED,WhiteDwarfModel
from sedkit.whitedwarf import _prepare,_Hypothesis
from validate_samples import tasks

DATA=Path(__file__).resolve().parents[2]/"data/whitedwarf"


def one(task):
    _,r,flux,error,_,_=task
    f,e=np.full(168,np.nan),np.full(168,np.nan);f[:61],e[:61]=flux[6:],error[6:]
    sed=SED(f,e,np.arange(168)<61,r["parallax"],r["parallax_error"],str(r["source_id"]))
    p=_prepare(sed,WhiteDwarfModel(),None,None,(0.,.1),None,None,True,None,None,False,None,None)
    fit,_=_Hypothesis(p,"wd+dwarf",False,5.,0.).fit()
    return dict(source_id=sed.source_id,teff_spec=float(r["Teffwd"]),subtype_spec=int(r["Sp"]),
        teff=fit["whitedwarf"]["teff"],companion_teff=fit["companion"]["teff"],beta_G=fit["fractions"]["beta_G"],
        extinction_e=fit["extinction_e"],objective=fit["objective"],converged=fit["converged"])


if __name__=="__main__":
    inputs=tasks("wdms")
    with ProcessPoolExecutor(4) as pool,(DATA/"validation_wdms_extinction.jsonl").open("w") as h:
        for i,r in enumerate(pool.map(one,inputs)):
            h.write(json.dumps(r)+"\n");h.flush()
            if (i+1)%25==0:print(i+1,len(inputs),flush=True)
