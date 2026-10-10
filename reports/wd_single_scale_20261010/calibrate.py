"""Calibrate DA XP fluxes on spectral labels and Gaia absolute single-star light."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from astropy.table import Table
from sedkit import SED, WhiteDwarfModel
from sedkit.whitedwarf import _prepare, _Hypothesis
from sedkit.extinction import extinction_curve

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
DATA=ROOT/"data/whitedwarf"
WORK=DATA/"single_scale"
KNOTS=np.array([7000,10000,13000,18000,25000,40000,80000.])


def design(teff,logg):
    x=np.log(teff);knots=np.log(KNOTS)
    hats=np.array([np.interp(x,knots,np.eye(len(knots))[j]) for j in range(len(knots))]).T
    return np.c_[hats,np.asarray(logg)-8.]


def robust_fit(x,y,smoothing,delta):
    if y.ndim==1:y=y[:,None]
    weights=np.isfinite(y).astype(float);target=np.nan_to_num(y)
    differences=np.diff(np.eye(len(KNOTS)),2,axis=0)
    penalty=np.zeros((x.shape[1],x.shape[1]));penalty[:-1,:-1]=smoothing*differences.T@differences
    penalty[-1,-1]=3.
    coef=np.zeros((x.shape[1],y.shape[1]))
    for _ in range(12):
        for j in range(y.shape[1]):
            coef[:,j]=np.linalg.solve(x.T@(weights[:,j,None]*x)+penalty+1e-8*np.eye(x.shape[1]),
                                      x.T@(weights[:,j]*target[:,j]))
        remaining=abs(target-x@coef)
        weights=np.isfinite(y)*np.minimum(1.,delta/np.maximum(remaining,1e-10))
    return coef


def calibration(x,residual,train,smoothing,uv_residual,folds):
    gray=np.nanmedian(residual,axis=1)
    shape=residual-gray[:,None]
    coef=robust_fit(x[train],shape[train],smoothing,.035)+robust_fit(x[train],gray[train],smoothing,.10)
    remaining=residual-x@coef
    gray_remaining=np.nanmedian(remaining,axis=1)
    shape_remaining=remaining-gray_remaining[:,None]
    diag=np.clip(1.4826*np.nanmedian(abs(shape_remaining[train]-np.nanmedian(shape_remaining[train],axis=0)),axis=0),.01,.3)
    scale_sigma=[]
    for j in range(len(KNOTS)):
        # Temperature-local scatter of absolute normalization; optical channels share it.
        weights=x[train,:-1]
        selected=(weights[:,j]>.15)
        v=gray_remaining[train][selected]
        scale_sigma.append(max(.03,float(1.4826*np.median(abs(v-np.median(v))))) if len(v)>=8 else .20)
    c=dict(a=np.zeros(168),b=np.zeros(168),blue_a=np.zeros(6),blue_b=np.zeros(6),
           diag=np.r_[diag[6:],np.full(107,.03)],blue_diag=diag[:6],teff_knots=KNOTS,
           teff_flux=np.c_[coef[:-1,6:],np.zeros((len(KNOTS),107))],logg_flux=np.r_[coef[-1,6:],np.zeros(107)],
           teff_blue=coef[:-1,:6],logg_blue=coef[-1,:6],scale_sigma=np.array(scale_sigma))
    uv=uv_residual[train]
    c["uv_a"]=np.nanmedian(uv,axis=0)
    c["uv_diag"]=np.maximum(.05,1.4826*np.nanmedian(abs(uv-c["uv_a"]-gray_remaining[train,None]),axis=0))
    return c,coef


def fit_one(task):
    row,flux,error,c=task
    f,e=np.full(168,np.nan),np.full(168,np.nan);f[:61],e[:61]=flux[6:],error[6:]
    sed=SED(f,e,np.arange(168)<61,row["parallax"],row["parallax_error"],str(row["source_id"]))
    model=WhiteDwarfModel(calibration=c)
    output=dict(source_id=str(row["source_id"]),teff_spec=float(row["teff"]),mass_reference=float(row["mass_reference"]),
                logg_spec=float(row["logg"]),fold=int(row["fold"]))
    for name,free in [("free",True),("mr",False)]:
        p=_prepare(sed,model,None,None,(row["dust_mean"],max(.005,row["dust_sigma"])),None,None,True,None,None,False,
                   {"logg":(float(row["logg"]),.02)} if free else None,None)
        fit,_=_Hypothesis(p,"wd",free,5.,0.).fit()
        output[name+"_teff"]=fit["whitedwarf"]["teff"]
        output[name+"_mass"]=fit["whitedwarf"]["mass"]
        output[name+"_radius"]=fit["whitedwarf"]["radius"]
        output[name+"_rms"]=float(np.sqrt(np.mean(((fit["model"]-p.y)/p.y)**2)))
    return output


def stats(values):
    a=np.asarray(values);median=np.median(a)
    return dict(median=float(median),scatter=float(1.4826*np.median(abs(a-median))))


def reference_mass(model,t,g):
    supported=[]
    for mass in model.cooling["mass_ax"]:
        try:supported.append((mass,model.physical(t,mass)["logg"]-g))
        except ValueError:continue
    for (lo,ylo),(hi,yhi) in zip(supported[:-1],supported[1:]):
        if ylo*yhi<=0:return brentq(lambda m:model.physical(t,m)["logg"]-g,lo,hi)
    return None


def fit_summary(fits):
    return {kind:dict(temperature=stats([r[kind+"_teff"]/r["teff_spec"]-1 for r in fits]),
                      mass=stats([r[kind+"_mass"]-r["mass_reference"] for r in fits])) for kind in ["free","mr"]}


def main():
    WORK.mkdir(exist_ok=True)
    cohort=Table.read(OUT/"anchors.ecsv")
    raw=WhiteDwarfModel(calibration=None)
    with np.load(OUT/"reference_shape_calibration.npz") as c:
        original=WhiteDwarfModel(calibration={k:c[k] for k in c.files})
    dust=json.loads((OUT/"dust_moments.json").read_text())
    attenuation_curve=np.r_[raw.blue_curve,extinction_curve(raw.wavelength_um[:61])]
    with np.load(DATA/"xp_local100/calibrated.npz") as a:
        index={int(s):i for i,s in enumerate(a["source_id"])}
        flux,error=a["flux"].copy(),a["error"].copy()
    uv_records=json.loads((OUT/"galex_anchors.json").read_text())
    from sedkit.subdwarf import _galex_flux
    rows=[];residual=[];old=[];observed=[];errors=[];uv_residual=[]
    for r in cohort:
        t,g=float(r["teff"]),float(r["logg"])
        m=reference_mass(raw,t,g)
        if m is None:continue
        dr=dust[str(r["source_id"])];extinction=dr["mean"]
        pred=raw.predict(t,m);op=original.predict(t,m)
        scale=(r["parallax"]/100)**2
        shape=np.r_[pred["blue"],pred["flux"][:61]]*scale*np.exp(-extinction*attenuation_curve)
        osh=np.r_[op["blue"],op["flux"][:61]]*scale*np.exp(-extinction*attenuation_curve)
        i=index[int(r["source_id"])];y,e=flux[i],error[i];good=y>5*e
        z=np.full(67,np.nan);z[good]=np.log(y[good]/shape[good]);residual.append(z)
        z=np.full(67,np.nan);z[good]=np.log(y[good]/osh[good]);old.append(z)
        uv=np.full(2,np.nan)
        for j,band in enumerate(["FUV","NUV"]):
            if band in uv_records.get(str(r["source_id"]),{}):
                mag,err,_=uv_records[str(r["source_id"])][band]
                f,_=_galex_flux(mag,err,raw.galex_pivot_nm[band],band)
                uv[j]=np.log(f/(scale*raw.passband(pred["coarse"]*np.exp(-extinction*raw.coarse_curve),band)))
        uv_residual.append(uv)
        row=dict(r);row.update(mass_reference=m,dust_mean=extinction,dust_sigma=dr["sigma"])
        rows.append(row);observed.append(y);errors.append(e)
    rows=Table(rows=rows);observed=np.array(observed);errors=np.array(errors)
    residual,old,uv_residual=np.array(residual),np.array(old),np.array(uv_residual)
    x=design(rows["teff"],rows["logg"]);folds=np.asarray(rows["fold"])
    comparisons={};candidates={}
    for smoothing in [1.,10.]:
        cv=np.full_like(residual,np.nan);calibrations={}
        for fold in range(5):
            train=folds!=fold;test=~train
            c,coef=calibration(x,residual,train,smoothing,uv_residual,folds)
            cv[test]=residual[test]-x[test]@coef;calibrations[fold]=c
        gray=np.nanmedian(cv[:,6:],axis=1);shape=cv-gray[:,None]
        comparisons[str(smoothing)]=dict(shape_rms=float(np.sqrt(np.nanmean(shape[:,6:]**2))))
        candidates[smoothing]=(cv,calibrations)
    smoothing=min(candidates,key=lambda k:comparisons[str(k)]["shape_rms"])
    cv,calibrations=candidates[smoothing]
    final,_=calibration(x,residual,np.ones(len(rows),bool),smoothing,uv_residual,folds)
    np.savez_compressed(WORK/"calibration.npz",**final)
    for fold,c in calibrations.items():np.savez_compressed(WORK/f"fold_{fold}.npz",**c)
    rows["old_log_flux_ratio"]=np.nanmedian(old[:,6:],axis=1)
    rows["cv_log_flux_ratio"]=np.nanmedian(cv[:,6:],axis=1)
    rows.write(OUT/"anchors.ecsv",overwrite=True)
    summary=dict(n=len(rows),scale="spectroscopic Teff/logg, thick-H CO radius, Gaia DR3 parallax, native Edenhofer extinction",
                 correction="piecewise linear in log Teff plus linear logg, robust absolute and shape fits",
                 knots_kelvin=KNOTS.tolist(),smoothing=smoothing,candidates=comparisons,
                 old_absolute=stats(rows["old_log_flux_ratio"]),cv_absolute=stats(rows["cv_log_flux_ratio"]),bins=[],
                 uv_n=np.isfinite(uv_residual).sum(axis=0).tolist(),uv_a=final["uv_a"].tolist(),uv_diag=final["uv_diag"].tolist())
    for lo,hi in [(6000,13000),(13000,25000),(25000,40000),(40000,80001)]:
        sel=(rows["teff"]>=lo)&(rows["teff"]<hi)
        summary["bins"].append(dict(range=[lo,hi],n=int(sum(sel)),old=stats(rows["old_log_flux_ratio"][sel]),cv=stats(rows["cv_log_flux_ratio"][sel])))
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)
    tasks=[(dict(rows[i]),observed[i],errors[i],calibrations[int(folds[i])]) for i in range(len(rows))]
    fits=[]
    with ProcessPoolExecutor(4) as pool,(WORK/"fits.jsonl").open("w") as h:
        for i,r in enumerate(pool.map(fit_one,tasks)):
            fits.append(r);h.write(json.dumps(r)+"\n");h.flush()
            if (i+1)%25==0:print(f"Single-scale fits {i+1}/{len(tasks)}",flush=True)
    Table(rows=fits).write(OUT/"fits.ecsv",overwrite=True)
    for kind in ["free","mr"]:
        summary[kind+"_temperature"]=stats([r[kind+"_teff"]/r["teff_spec"]-1 for r in fits])
        summary[kind+"_mass"]=stats([r[kind+"_mass"]-r["mass_reference"] for r in fits])
    for item in summary["bins"]:
        lo,hi=item["range"];part=[r for r in fits if lo<=r["teff_spec"]<hi]
        item["fits"]=fit_summary(part)
    holdout=Table.read(OUT/"holdout_anchors.ecsv");tasks=[];hrows=[]
    hm=WhiteDwarfModel(calibration=final)
    with np.load(DATA/"xp_single_holdout150/calibrated.npz") as a:
        index={int(s):i for i,s in enumerate(a["source_id"])}
        for r in holdout:
            t,g=float(r["teff"]),float(r["logg"]);mass=reference_mass(raw,t,g)
            if mass is None:continue
            row=dict(r);d=dust[str(r["source_id"])];row.update(mass_reference=mass,dust_mean=d["mean"],dust_sigma=d["sigma"],fold=-1)
            i=index[int(r["source_id"])];y,e=a["flux"][i],a["error"][i]
            tasks.append((row,y,e,final))
            pred=hm.predict(t,mass);scale=(r["parallax"]/100)**2
            f=np.r_[pred["blue"],pred["flux"][:61]]*scale*np.exp(-d["mean"]*attenuation_curve)
            good=y>5*e
            hrows.append(dict(source_id=str(r["source_id"]),teff_spec=t,log_flux_ratio=float(np.median(np.log(y[6:][good[6:]]/f[6:][good[6:]])))))
    hfits=[]
    with ProcessPoolExecutor(4) as pool,(WORK/"holdout_fits.jsonl").open("w") as h:
        for i,r in enumerate(pool.map(fit_one,tasks)):
            hfits.append(r);h.write(json.dumps(r)+"\n");h.flush()
            if (i+1)%25==0:print(f"Independent single-star fits {i+1}/{len(tasks)}",flush=True)
    Table(rows=hfits).write(OUT/"holdout_fits.ecsv",overwrite=True)
    Table(rows=hrows).write(OUT/"holdout_fluxes.ecsv",overwrite=True)
    summary["holdout"]=dict(n=len(hfits),distance_pc=[100,150],absolute=stats([r["log_flux_ratio"] for r in hrows]),fits=fit_summary(hfits),bins=[])
    for lo,hi in [(6000,13000),(13000,25000),(25000,40000),(40000,80001)]:
        part=[r for r in hfits if lo<=r["teff_spec"]<hi]
        summary["holdout"]["bins"].append(dict(range=[lo,hi],n=len(part),fits=fit_summary(part)))
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)
    from compare_flux import main as compare_flux
    result=compare_flux()
    model_path=ROOT/"src/sedkit/models/whitedwarf"
    np.savez_compressed(model_path/"calibration.npz",**final)
    metadata=json.loads((model_path/"summary.json").read_text())
    metadata.update(calibration="Absolute single-DA scale: robust log-Teff spline plus linear logg, spectral labels and Gaia parallax",
                    calibration_anchors=len(rows),calibration_holdout=result["holdout"]["n"],
                    empirical_teff_range=[float(min(rows["teff"])),float(max(rows["teff"]))],
                    empirical_logg_range=[float(min(rows["logg"])),float(max(rows["logg"]))],
                    calibration_report="reports/wd_single_scale_20261010/summary.json",
                    model_error="Temperature-dependent correlated absolute scale plus diagonal spectral shape",
                    uv_calibration="71 real single DAs, absolute FUV/NUV passband factors at spectral labels, cooling radius, Gaia parallax and native extinction")
    (model_path/"summary.json").write_text(json.dumps(metadata,indent=2)+"\n")



if __name__=="__main__":main()
