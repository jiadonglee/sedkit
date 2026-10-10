"""Matched and deliberately mismatched WD light-profile injections."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import requests
from sedkit import SED, StellarModel, WhiteDwarfModel, fit_whitedwarf_companion, whitedwarf_light_limit
from sedkit.extinction import extinction_curve
from sedkit.orbit import photocentre_a0,solve_dark_companion

DATA=Path(__file__).resolve().parents[2]/"data/whitedwarf"
TEMPERATURES=[6000,8000,10000,13000,16000,20000,25000,32000,40000,55000,80000]


def one(task):
    kind,t,primary,variant,seed,thin=task
    wd=WhiteDwarfModel();stellar=StellarModel();rng=np.random.default_rng(seed)
    p=wd.predict(t,.6);cool=stellar.evaluate(primary,0,5,0)
    if variant=="temperature_scale":
        p=wd.predict(t*1.05,.6,logg=p["logg"],radius=p["radius"])
    if variant=="thin_hydrogen":
        radius=np.exp(np.interp(np.log(t),np.log(thin[:,1]),np.log(thin[:,3])))/6.957e10
        logg=np.log10(6.6743e-8*.6*1.988409870698051e33/(radius*6.957e10)**2)
        p=wd.predict(t,.6,logg=logg,radius=radius)
    f=p["flux"]+cool["flux_10pc"]
    extinction=.03 if variant=="extinction_offset" else 0.
    f*=np.exp(-extinction*extinction_curve(stellar.wavelength_um))
    error=.02*f*.01;y=f*.01+rng.normal(size=168)*error
    sed=SED(y,error,np.arange(168)<61,10,.02,str(seed))
    wave=wd.coarse_wavelength_nm
    coarse=np.exp(np.interp(wave,stellar.wavelength_um[:61]*1000,np.log(cool["flux_10pc"][:61])))
    att=np.exp(-extinction*wd.coarse_curve)
    beta=wd.passband(p["coarse"]*att,"G")/wd.passband((p["coarse"]+coarse)*att,"G")
    out=dict(kind=kind,teff_true=t,mass_true=.6,primary_true=primary,beta_true=beta,variant=variant)
    options=dict(extinction=None,extinction_prior=(0.,.01)) if variant=="extinction_offset" else dict(extinction=0.)
    if kind=="composite":
        fit=fit_whitedwarf_companion(sed,**options)
        h=fit["hypotheses"]["wd+dwarf"]
        out.update(teff=h["whitedwarf"]["teff"],mass=h["whitedwarf"]["mass"],primary=h["companion"]["mass"],
            beta=h["fractions"]["beta_G"],detection=min(fit["hypotheses"][k]["objective"] for k in ["wd","dwarf"])-h["objective"])
    else:
        limit=whitedwarf_light_limit(sed,masses=[.6],temperatures=TEMPERATURES,**options)
        a0=photocentre_a0(primary,.6,1-beta,500,10)
        low=solve_dark_companion(a0,10,500,primary)["m2"]
        high=solve_dark_companion(a0,10,500,primary,beta=limit["flux_ratio_G_upper"])["m2"] if np.isfinite(limit["beta_G_upper"]) else np.nan
        out.update(beta_upper=limit["beta_G_upper"],mass_interval=[low,high],
            beta_covered=bool(beta<=limit["beta_G_upper"]),mass_covered=bool(low<=.6<=high))
    return out


def main():
    path=DATA/"build/cooling/seq_060_thin.txt"
    if not path.exists():
        r=requests.get("https://www.astro.umontreal.ca/~bergeron/CoolingModels/CoolingModels/seq_060_thin.txt",timeout=60)
        r.raise_for_status();path.write_bytes(r.content)
    thin=np.array([list(map(float,l.split())) for l in path.read_text().splitlines() if len(l.split())==6 and l.split()[0].isdigit()])
    thin=thin[np.argsort(thin[:,1])]
    tasks=[]
    for variant in ["matched","temperature_scale","thin_hydrogen","extinction_offset"]:
        for t in [8000,18000,35000]:
            for primary in [.2,.4,.6]: tasks.append(("composite",t,primary,variant,20261010+len(tasks),thin))
        for t in [8000,18000,35000]:
            tasks.append(("dark",t,.9,variant,20261010+len(tasks),thin))
    with ProcessPoolExecutor(2) as pool,(DATA/"injections.jsonl").open("w") as handle:
        for i,r in enumerate(pool.map(one,tasks)):
            handle.write(json.dumps(r)+"\n");handle.flush()
            print(f"injections {i+1}/{len(tasks)}",flush=True)


if __name__=="__main__":main()
