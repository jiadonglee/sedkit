"""Absolute-flux comparisons using linear XP fluxes, including low-SNR bins."""

import json
from pathlib import Path
import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sedkit import WhiteDwarfModel
from sedkit.extinction import extinction_curve

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
DATA=ROOT/"data/whitedwarf"
WORK=DATA/"single_scale"


def stats(a):
    a=np.asarray(a);m=np.median(a)
    return dict(median=float(m),scatter=float(1.4826*np.median(abs(a-m))))


def amplitude(y,error,predicted):
    good=np.isfinite(y)&np.isfinite(error)&(error>0)
    # All valid linear fluxes enter; selecting positive high-SNR bins
    # would bias the normalization of faint single WDs.
    return float(np.log(np.sum(y[good]*predicted[good]/error[good]**2)/np.sum((predicted[good]/error[good])**2)))


def main():
    with np.load(OUT/"reference_shape_calibration.npz") as c:old=WhiteDwarfModel({k:c[k] for k in c.files})
    models={}
    for fold in range(5):
        with np.load(WORK/f"fold_{fold}.npz") as c:models[fold]=WhiteDwarfModel({k:c[k] for k in c.files})
    with np.load(WORK/"calibration.npz") as c:models[-1]=WhiteDwarfModel({k:c[k] for k in c.files})
    dust=json.loads((OUT/"dust_moments.json").read_text())
    tables=[]
    for holdout,folder,filename in [(False,"xp_local100","anchors.ecsv"),(True,"xp_single_holdout150","holdout_anchors.ecsv")]:
        rows=Table.read(OUT/filename)
        fits=Table.read(OUT/("holdout_fits.ecsv" if holdout else "fits.ecsv"))
        fi={str(r["source_id"]):r for r in fits}
        with np.load(DATA/folder/"calibrated.npz") as a:
            idx={str(v):i for i,v in enumerate(a["source_id"])}
            output=[]
            for r in rows:
                sid=str(r["source_id"])
                if sid not in fi:continue
                fit=fi[sid];i=idx[sid];record=dict(source_id=sid,teff_spec=float(r["teff"]),logg_spec=float(r["logg"]),G=float(r["phot_g_mean_mag"]))
                for key,model in [("old",old),("new",models[-1 if holdout else int(r["fold"])])]:
                    predicted=model.predict(r["teff"],fit["mass_reference"])["flux"][:61]*(r["parallax"]/100)**2
                    predicted*=np.exp(-dust[sid]["mean"]*extinction_curve(model.wavelength_um[:61]))
                    record[key+"_log_flux_ratio"]=amplitude(a["flux"][i,6:],a["error"][i,6:],predicted)
                output.append(record)
        table=Table(rows=output);tables.append(table)
        table.write(OUT/("holdout_fluxes.ecsv" if holdout else "fluxes.ecsv"),overwrite=True)
    train,hold=tables
    s=json.loads((OUT/"summary.json").read_text())
    s["flux_metric"]="log of error-weighted linear XP61 amplitude at spectral labels, Gaia parallax and adopted extinction"
    s["old_absolute"]=stats(train["old_log_flux_ratio"]);s["cv_absolute"]=stats(train["new_log_flux_ratio"])
    for item in s["bins"]:
        lo,hi=item["range"];sel=(train["teff_spec"]>=lo)&(train["teff_spec"]<hi)
        item["old"]=stats(train["old_log_flux_ratio"][sel]);item["cv"]=stats(train["new_log_flux_ratio"][sel])
    s["holdout"]["absolute"]=stats(hold["new_log_flux_ratio"])
    s["holdout"]["old_absolute"]=stats(hold["old_log_flux_ratio"])
    s["absolute_offset_above_50percent"]=dict(
        cross_validation=int(sum(abs(np.expm1(train["new_log_flux_ratio"]))>.5)),
        independent=int(sum(abs(np.expm1(hold["new_log_flux_ratio"]))>.5)))
    for item in s["holdout"]["bins"]:
        lo,hi=item["range"];sel=(hold["teff_spec"]>=lo)&(hold["teff_spec"]<hi)
        item["absolute"]=stats(hold["new_log_flux_ratio"][sel]);item["old_absolute"]=stats(hold["old_log_flux_ratio"][sel])
    (OUT/"summary.json").write_text(json.dumps(s,indent=2)+"\n")
    fig,axes=plt.subplots(1,3,figsize=(11,3.6),layout="constrained")
    for table,key,label,color in [(train,"old_log_flux_ratio","Shape calibration","#aaaaaa"),(train,"new_log_flux_ratio","Absolute scale, cross-validation","#3976a5"),(hold,"new_log_flux_ratio","Independent 100--150 pc stars","#cb7d28")]:
        axes[0].scatter(table["teff_spec"]/1000,np.exp(table[key]),s=9,alpha=.5,c=color,label=label)
    axes[0].axhline(1,c=".4",lw=1);axes[0].set_yscale("log")
    axes[0].set(xlabel="Spectroscopic temperature (kK)",ylabel="Observed / model XP flux")
    axes[0].legend(fontsize=6.5,frameon=False)
    for table,label,color in [(Table.read(OUT/"fits.ecsv"),"Cross-validation","#3976a5"),(Table.read(OUT/"holdout_fits.ecsv"),"Independent stars","#cb7d28")]:
        axes[1].scatter(table["teff_spec"]/1000,100*(table["mr_teff"]/table["teff_spec"]-1),s=9,alpha=.5,c=color,label=label)
        axes[2].scatter(table["mass_reference"],table["mr_mass"],s=9,c=color,alpha=.5,label=label)
    axes[1].axhline(0,c=".4",lw=1);axes[1].set(xlabel="Spectroscopic temperature (kK)",ylabel="XP mass-radius temperature offset (%)",ylim=(-45,70));axes[1].legend(fontsize=7,frameon=False)
    axes[2].plot([.2,1.3],[.2,1.3],c=".4",lw=1);axes[2].set(xlabel="Spectral + CO-track reference mass",ylabel="XP mass (solar masses)",xlim=(.2,1.3),ylim=(.2,1.3))
    for ax in axes:ax.spines[["top","right"]].set_visible(False)
    fig.savefig(OUT/"single_scale.png",dpi=180);plt.close(fig)
    print("Cross-validation absolute:",s["cv_absolute"],"Independent absolute:",s["holdout"]["absolute"])
    return s


if __name__=="__main__":main()
