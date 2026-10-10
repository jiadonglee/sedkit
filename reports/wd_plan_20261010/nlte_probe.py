"""Compare pure-H TLUSTY207 spectra with the seven hot DA XP anchors."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import numpy as np
import requests
from astropy.io import fits
from astropy.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sedkit import SED, WhiteDwarfModel
from sedkit.whitedwarf import _prepare, _Hypothesis

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data/whitedwarf"
OUT=Path(__file__).resolve().parent
WORK=DATA/"nlte"
BASE="https://archive.stsci.edu/hlsps/wd-grid/tlusty207/"
REFERENCE="https://archive.stsci.edu/hlsp/wd-grid"
OPERATOR=Path("/Users/jdli/Project/J-Caps-xps-hot/experiments/phase0_forward_xp_20261003")


def build():
    sys.path.insert(0,str(ROOT/"scripts"))
    from build_subdwarf_model import Reducer
    WORK.mkdir(exist_ok=True);(WORK/"out").mkdir(exist_ok=True)
    np.savez(WORK/"out/balmer_index.npz",ew_max_nm=1.)
    reducer=Reducer(OPERATOR,WORK)
    ts=np.r_[np.arange(30000,40001,2000),np.arange(45000,80001,5000)]
    gs=np.arange(7.,9.51,.5)
    def one(node):
        t,g=node;stem=f"t{int(t/100)}g{int(g*100)}n"
        reduced=WORK/(stem+".npz")
        path=WORK/(stem+".fits")
        if not path.exists():
            response=requests.get(BASE+f"hlsp_wd-grid_tlusty207_nlte-model_{stem}_r5000_v1_spec.fits",timeout=60)
            response.raise_for_status();path.write_bytes(response.content)
        with fits.open(path) as h:
            lam=np.asarray(h[1].data["Wavelength"],float)
            flux=np.asarray(h[1].data["Eddington_Flux"],float)*4*np.pi
        if not reduced.exists():
            reduced_values=reducer.reduce(lam/10,flux*10)
            np.savez_compressed(reduced,**reduced_values)
        with np.load(reduced) as a:values={k:a[k] for k in a.files}
        values["finite_range_flux_over_sigma_teff4"]=np.trapezoid(flux,lam)/(5.670374419e-5*float(t)**4)
        return values
    with ThreadPoolExecutor(4) as pool:
        values=[]
        for i,v in enumerate(pool.map(one,[(t,g) for t in ts for g in gs])):
            values.append(v)
            if (i+1)%12==0:print(f"NLTE reduced {i+1}/{len(ts)*len(gs)}",flush=True)
    model=WhiteDwarfModel(calibration=None)
    table=dict(model.table)
    table.update(teff_ax=ts,logg_ax=gs)
    for key,field in [("ln_flux","channels"),("ln_blue","blue"),("ln_coarse","coarse")]:
        table[key]=np.log(np.array([v[field] for v in values])).reshape((len(ts),len(gs),-1))
    table["balmer_w"]=np.array([v["balmer_w"] for v in values]).reshape(len(ts),len(gs))/model.table["ew_max_nm"]
    np.savez_compressed(WORK/"table.npz",**table)
    return dict(nodes=len(values),temperature_range=[float(ts[0]),float(ts[-1])],
        units="4*pi*H_lambda, erg/s/cm2/Angstrom; vacuum wavelengths",
        finite_range_flux_over_sigma_teff4=[float(min(v["finite_range_flux_over_sigma_teff4"] for v in values)),float(max(v["finite_range_flux_over_sigma_teff4"] for v in values))])


def main():
    grid=build()
    model=WhiteDwarfModel(calibration=None)
    with np.load(WORK/"table.npz") as a:model.table={k:a[k] for k in a.files}
    rows=Table.read(OUT/"single_da_anchors.ecsv");rows=rows[rows["teff"]>=40000]
    original=Table.read(DATA/"validation_single.ecsv")
    result=[]
    with np.load(DATA/"xp_local100/calibrated.npz") as a:
        idx={int(s):i for i,s in enumerate(a["source_id"])}
        for row in rows:
            i=idx[int(row["source_id"])];f,e=np.full(168,np.nan),np.full(168,np.nan)
            f[:61],e[:61]=a["flux"][i,6:],a["error"][i,6:]
            sed=SED(f,e,np.arange(168)<61,row["parallax"],row["parallax_error"],str(row["source_id"]))
            p=_prepare(sed,model,None,0.,None,None,None,False,None,None,False,{"logg":(row["logg"],.02)},None)
            fit,_=_Hypothesis(p,"wd",True,5.,0.).fit()
            old=original[original["source_id"]==int(row["source_id"])][0]
            result.append(dict(source_id=str(row["source_id"]),teff_spec=float(row["teff"]),
                teff_nlte=fit["whitedwarf"]["teff"],teff_koester_raw=float(old["raw_teff"]),
                teff_koester_cal=float(old["cal_teff"]),converged=fit["converged"]))
            with np.load(DATA/"calibration_folds"/f"{row['fold']}.npz") as c:
                model.calibration={k:c[k] for k in c.files}
            p=_prepare(sed,model,None,0.,None,None,None,False,None,None,False,{"logg":(row["logg"],.02)},None)
            calibrated,_=_Hypothesis(p,"wd",True,5.,0.).fit()
            result[-1]["teff_nlte_koester_cal"]=calibrated["whitedwarf"]["teff"]
            model.calibration=None
            print(result[-1],flush=True)
    summary=dict(reference=REFERENCE,grid=grid,n=len(result),comparison={})
    for key in ["nlte","nlte_koester_cal","koester_raw","koester_cal"]:
        d=np.array([r["teff_"+key]/r["teff_spec"]-1 for r in result])
        summary["comparison"][key]=dict(median=float(np.median(d)),scatter=float(1.4826*np.median(abs(d-np.median(d)))))
        summary["comparison"][key]["at_upper_temperature_edge"]=sum(r["teff_"+key]>=79999 for r in result)
    summary["sources"]=result
    fig,axes=plt.subplots(1,2,figsize=(8,3.8),layout="constrained")
    for ax,keys,title in zip(axes,[("koester_raw","nlte"),("koester_cal","nlte_koester_cal")],
                             ["Raw atmospheres","Existing Koester-trained correction"]):
        for key,label,color,marker in zip(keys,["Koester LTE","TLUSTY207 NLTE"],["#3976a5","#cb7d28"],["o","s"]):
            ax.scatter([r["teff_spec"]/1000 for r in result],[r["teff_"+key]/1000 for r in result],
                       s=32,facecolors="none",edgecolors=color,marker=marker,label=label)
        ax.plot([30,80],[30,80],c=".5",lw=1)
        ax.axhline(80,c=".7",ls="--",lw=.8)
        ax.set(xlabel="Spectroscopic temperature (kK)",ylabel="XP fitted temperature (kK)",
               title=title,xlim=(30,82),ylim=(30,82))
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=8,frameon=False)
    fig.savefig(OUT/"nlte_probe.png",dpi=180);plt.close(fig)
    (OUT/"nlte_probe.json").write_text(json.dumps(summary,indent=2)+"\n");print(summary)


if __name__=="__main__":main()
