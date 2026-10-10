"""Grouped validation of a low-order temperature term in the DA correction."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from astropy.table import Table
from sedkit import SED,WhiteDwarfModel
from sedkit.whitedwarf import _prepare,_Hypothesis
from calibrate_da import make_calibration

DATA=Path(__file__).resolve().parents[2]/"data/whitedwarf"
OUT=Path(__file__).resolve().parent


class TemperatureModel(WhiteDwarfModel):
    def __init__(self,calibration,term):
        super().__init__(calibration=calibration)
        self.term=term

    def predict(self,teff,*args,**kwargs):
        p=super().predict(teff,*args,**kwargs)
        correction=np.log(teff/15000)*self.term
        p["blue"]*=np.exp(correction[:6]);p["flux"][:61]*=np.exp(correction[6:])
        wavelength=np.r_[self.blue_wavelength_nm,self.wavelength_um[:61]*1000]
        p["coarse"]*=np.exp(np.interp(self.coarse_wavelength_nm,wavelength,correction,left=0,right=0))
        return p


def fit_one(task):
    row,flux,error,calibration,term=task
    f,e=np.full(168,np.nan),np.full(168,np.nan);f[:61],e[:61]=flux[6:],error[6:]
    sed=SED(f,e,np.arange(168)<61,row["parallax"],row["parallax_error"],str(row["source_id"]))
    p=_prepare(sed,TemperatureModel(calibration,term),None,0.,None,None,None,False,None,None,False,
        {"logg":(row["logg"],.02)},None)
    result,_=_Hypothesis(p,"wd",True,5.,0.).fit()
    return dict(source_id=str(row["source_id"]),teff_spec=row["teff"],teff_fit=result["whitedwarf"]["teff"])


def main():
    rows=Table.read(OUT/"single_da_anchors.ecsv")
    with np.load(DATA/"xp_local100/calibrated.npz") as a:
        idx={int(s):i for i,s in enumerate(a["source_id"])}
        order=[idx[int(s)] for s in rows["source_id"]];flux,error=a["flux"][order],a["error"][order]
    raw=WhiteDwarfModel(calibration=None);residual=[];width=[]
    for r,y,e in zip(rows,flux,error):
        p=raw.predict(r["teff"],logg=r["logg"],radius=1.);shape=np.r_[p["blue"],p["flux"][:61]]
        good=y>5*e;z=np.full(67,np.nan);z[good]=np.log(y[good]/shape[good]);z-=np.nanmean(z[6:])
        residual.append(z);width.append(p["balmer_w"])
    residual=np.array(residual);width=np.array(width)
    x=np.c_[np.ones(len(rows)),width,np.log(rows["teff"]/15000)]
    remaining=np.full_like(residual,np.nan);tasks=[]
    for fold in range(5):
        train=rows["fold"]!=fold;test=~train;coef=np.zeros((3,67))
        for j in range(67):
            good=train&np.isfinite(residual[:,j]);coef[:,j]=np.linalg.lstsq(x[good],residual[good,j],rcond=None)[0]
        remain=residual-x@coef;remaining[test]=remain[test]
        c=make_calibration(residual,width,train,1)
        c["a"][:61],c["b"][:61]=coef[0,6:],coef[1,6:]
        c["blue_a"],c["blue_b"]=coef[0,:6],coef[1,:6]
        diag=np.clip(1.4826*np.nanmedian(abs(remain[train]-np.nanmedian(remain[train],axis=0)),axis=0),.01,.3)
        c["diag"][:61],c["blue_diag"]=diag[6:],diag[:6]
        for i in np.flatnonzero(test):tasks.append((dict(rows[i]),flux[i],error[i],c,coef[2]))
    with ProcessPoolExecutor(4) as pool: fits=list(pool.map(fit_one,tasks))
    summary=dict(form="exp(a + W b + log(Teff/15000) c)",cv_shape_rms=float(np.sqrt(np.nanmean(remaining[:,6:]**2))),bins=[])
    for lo,hi in [(6000,40000),(40000,80001)]:
        a=[r for r in fits if lo<=r["teff_spec"]<hi];d=np.array([r["teff_fit"]/r["teff_spec"]-1 for r in a])
        summary["bins"].append(dict(range=[lo,hi],n=len(a),median=float(np.median(d)),scatter=float(1.4826*np.median(abs(d-np.median(d))))))
    (OUT/"hot_da_probe.json").write_text(json.dumps(summary,indent=2)+"\n")
    (DATA/"hot_da_probe.json").write_text(json.dumps(fits,indent=2)+"\n");print(summary)


if __name__=="__main__":main()
