"""Fit the observed SOPHIE RVs with an unmodified-error Keplerian model."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares
from orblet.model import rv_curve
from orblet.constants import DAYS_PER_KEPLER_YEAR

def fit_rv(directory):
    p=Path(directory)
    rv=np.genfromtxt(p/'hd195987_rv.csv',delimiter=',',names=True)
    t=rv['bjd']-2400000.5
    def curves(v,t=t):
     period,e,w,tp,k1,k2,g1,g2=v
     common=dict(period_yr=period/DAYS_PER_KEPLER_YEAR,ecc=e,tau=0.,epoch_ref_mjd=tp)
     return (rv_curve(t,omega_rad=w,K_kms=k1,offset_kms=g1,**common),rv_curve(t,omega_rad=w+np.pi,K_kms=k2,offset_kms=g2,**common))
    def residual(v):
     a,b=curves(v)
     return np.r_[(rv['rv1_kms']-a)/rv['err1_kms'],(rv['rv2_kms']-b)/rv['err2_kms']]
    initial=[57.321927,.3051,6.23339,2459378.81855-2400000.5,28.85248,28.85248/.786314,-5.6306,-5.6306]
    fit=least_squares(residual,initial,bounds=([57.2,.2,6.,59375,20,30,-8,-8],[57.5,.4,6.5,59382,40,50,-3,-3]),x_scale='jac',ftol=1e-12,xtol=1e-12,gtol=1e-10,max_nfev=300)
    a,b=curves(fit.x)
    cov=np.linalg.inv(fit.jac.T@fit.jac);q=fit.x[4]/fit.x[5]
    dq=np.zeros(8);dq[4]=1/fit.x[5];dq[5]=-q/fit.x[5]
    summary=dict(parameters=dict(zip(('period_days','ecc','omega_rad','tp_bjd_minus_2400000_5','k1_kms','k2_kms','gamma1_kms','gamma2_kms'),fit.x.tolist())),q=q,q_formal_error=float(np.sqrt(dq@cov@dq)),chi2=float(np.sum(residual(fit.x)**2)),n_data=104,n_parameters=8,converged=bool(fit.success),rms1_ms=float(np.std(rv['rv1_kms']-a)*1000),rms2_ms=float(np.std(rv['rv2_kms']-b)*1000),chi2_1=float(np.sum(((rv['rv1_kms']-a)/rv['err1_kms'])**2)),chi2_2=float(np.sum(((rv['rv2_kms']-b)/rv['err2_kms'])**2)))
    (p/'rv_fit.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

if __name__=="__main__":
    print(json.dumps(fit_rv(Path(__file__).parent),indent=2))
