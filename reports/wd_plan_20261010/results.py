"""Summarize the DA, WDMS, null, orbit and injection experiments."""

import json
from pathlib import Path
import numpy as np
from scipy.stats import beta
from astropy.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data/whitedwarf"
OUT=Path(__file__).resolve().parent


def read(kind):
    return [json.loads(l) for l in (DATA/f"validation_{kind}.jsonl").read_text().splitlines()]


def scatter(d):return float(1.4826*np.median(abs(d-np.median(d))))


def main():
    single=Table.read(DATA/"validation_single.ecsv")
    wd,train,holdout,wdms,orbits=[read(k) for k in ["wd","fgk","fgk_holdout","wdms","orbits"]]
    threshold=json.loads((DATA/"detection_threshold.json").read_text())
    summary=dict(detection=threshold,null={},temperature_bins=[],magnitude_bins=[])
    for label,rows in [("wd_cv",wd),("fgk_training",train),("fgk_holdout",holdout)]:
        k=sum(r["detection"]>threshold["threshold"] for r in rows);n=len(rows)
        summary["null"][label]=dict(n=n,false_positive=k,upper95=float(beta.ppf(.95,k+1,n-k)))
    for key,bins,name in [("teff_spec",[6000,13000,25000,40000,80001],"temperature_bins"),
                          ("G",[9,14,16,21],"magnitude_bins")]:
        for lo,hi in zip(bins[:-1],bins[1:]):
            s=single[(single[key]>=lo)&(single[key]<hi)];r=dict(range=[lo,hi],n=len(s))
            for label in ["cal","mr","blue"]:
                d=np.asarray(s[label+"_teff"]/s["teff_spec"]-1)
                r[label]=dict(median=float(np.median(d)),scatter=scatter(d)) if len(d) else None
            summary[name].append(r)
    valid=[r for r in orbits if r["gaia_rv_consistent"]]
    summary["orbits"]=dict(n=len(orbits),gaia_rv_consistent=len(valid),
        mass_shift_le_002=sum(r["mass_shift"]<=.02 for r in valid),
        beta_G_upper_median=float(np.nanmedian([r["beta_G_upper"] for r in valid])),
        mass_shift_median=float(np.nanmedian([r["mass_shift"] for r in valid])),
        mass_shift_range=[float(np.nanmin([r["mass_shift"] for r in valid])),float(np.nanmax([r["mass_shift"] for r in valid]))],
        excluded=[r["name"] for r in orbits if not r["gaia_rv_consistent"]],
        ms_rejected_at_delta9=sum(r["luminous_delta"]>9 for r in valid),
        threshold=9,confidence_level=None)
    xp=read("orbits_xp")
    xp_valid=[r for r in xp if r["gaia_rv_consistent"]]
    summary["orbits"].update(n_with_ir=sum(r.get("n_ir",0)>0 for r in orbits),
        xp_only_mass_shift_median=float(np.nanmedian([r["mass_shift"] for r in xp_valid])),
        xp_only_mass_shift_le_002=sum(r["mass_shift"]<=.02 for r in xp_valid))
    summary["wdms_detection"]=dict(n=len(wdms),above_threshold=sum(r["detection"]>55 for r in wdms),
        scope="known SDSS DA/M catalogue subset; recovery fraction, not population completeness")
    injections=[json.loads(l) for l in (DATA/"injections.jsonl").read_text().splitlines()]
    summary["injections"]={}
    for variant in ["matched","temperature_scale","thin_hydrogen","extinction_offset"]:
        c=[r for r in injections if r["variant"]==variant and r["kind"]=="composite"]
        d=[r for r in injections if r["variant"]==variant and r["kind"]=="dark"]
        summary["injections"][variant]=dict(n_composite=len(c),n_dark=len(d),
            teff_median=float(np.median([r["teff"]/r["teff_true"]-1 for r in c])),
            beta_max_abs_error=max(abs(r["beta"]-r["beta_true"]) for r in c),
            mass_median_offset=float(np.median([r["mass"]-r["mass_true"] for r in c])),
            dark_beta_covered=sum(r["beta_covered"] for r in d),dark_mass_covered=sum(r["mass_covered"] for r in d))
    (OUT/"validation_results.json").write_text(json.dumps(summary,indent=2)+"\n")
    Table(rows=[{k:v for k,v in r.items() if k not in ["profile","intervals"]} for r in orbits]).write(OUT/"orbit_results.ecsv",overwrite=True)
    fig,axes=plt.subplots(2,3,figsize=(12,7.5),layout="constrained")
    ax=axes[0,0]
    for label,color,title in [("cal","#3976a5","Free radius + spectral gravity"),("mr","#cb7d28","Thick-H M–R relation")]:
        ax.scatter(single["teff_spec"]/1000,100*(single[label+"_teff"]/single["teff_spec"]-1),s=9,alpha=.55,c=color,label=title)
    ax.axhline(0,c=".4",lw=1);ax.set(xlabel="Spectroscopic WD temperature (kK)",ylabel="XP temperature offset (%)",ylim=(-45,70),title=f"{len(single)} held-out DA anchors");ax.legend(fontsize=7,frameon=False)
    ax=axes[0,1]
    for label,rows,color in [("DA, cross-validation",wd,"#3976a5"),("FGK, training",train,"#999999"),("FGK, new holdout",holdout,"#cb7d28")]:
        x=np.sort([r["detection"] for r in rows]);ax.plot(x,np.arange(1,len(x)+1)/len(x),label=label,c=color)
    ax.axvline(55,c=".3",ls="--");ax.set_xscale("symlog",linthresh=2);ax.set(xlabel="Composite improvement in objective",ylabel="Cumulative fraction",title="Independent FGK null check");ax.legend(fontsize=7,frameon=False)
    ax=axes[0,2]
    points=ax.scatter([r["teff_spec"]/1000 for r in wdms],[r["wd_teff"]/1000 for r in wdms],s=13,c=[r["beta_G"] for r in wdms],cmap="viridis",vmin=0,vmax=1)
    fig.colorbar(points,ax=ax,label=r"WD $\beta_G$",shrink=.8)
    ax.plot([6,80],[6,80],c=".5",lw=1);ax.set(xlabel="SDSS WD temperature (kK)",ylabel="XP WD temperature (kK)",title=f"{len(wdms)} real DA+M systems",xlim=(6,80),ylim=(6,80))
    ax=axes[1,0];b=json.loads((DATA/"validation_beta.json").read_text())
    ax.scatter([r["beta_reference"] for r in b],[r["beta_fit"] for r in b],s=13,alpha=.65,c="#3976a5");ax.plot([0,1],[0,1],c=".5",lw=1)
    ax.set(xlabel="SDSS normalization transferred to G",ylabel="XP WD light fraction",title=f"{len(b)} spectral decompositions",xlim=(0,max(1.05,max(r["beta_reference"] for r in b)*1.02)),ylim=(0,1.05))
    ax=axes[1,1]
    ax.scatter([r["mass_rv"] for r in valid],[r["mass_shift"] for r in valid],s=20,c="#3976a5");ax.axhline(.02,c="#cb7d28",ls="--",label="0.02 solar masses")
    ax.set(xlabel="RV + astrometry WD mass (solar masses)",ylabel="Allowed mass change (solar masses)",title=f"{len(valid)} Gaia/RV orbits, XP + 2MASS");ax.legend(fontsize=8,frameon=False)
    ax=axes[1,2]
    for variant,color in [("matched","#3976a5"),("thin_hydrogen","#cb7d28")]:
        a=[r for r in injections if r["kind"]=="composite" and r["variant"]==variant]
        ax.scatter([r["beta_true"] for r in a],[r["beta"] for r in a],s=25,c=color,label=variant.replace("_"," "))
    ax.plot([0,1],[0,1],c=".5",lw=1);ax.set(xlabel="Injected WD G light fraction",ylabel="Recovered WD G light fraction",title="Matched and thin-H injections",xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8,frameon=False)
    for ax in axes.flat:ax.spines[["top","right"]].set_visible(False)
    fig.savefig(OUT/"validation.png",dpi=180);fig.savefig(OUT/"validation.pdf");plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
