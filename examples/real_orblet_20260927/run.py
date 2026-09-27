"""Fit HD 195987's observed Gaia XP and published SOPHIE SB2 velocities."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sedkit import SED, StellarModel, fit, plot, loglike_sed
from sedkit.fit import neg2_log_likelihood
from sedkit.plot import PAPER_STYLE,BINARY,SINGLE
from orblet.model import rv_curve,semi_amplitude_kms
from orblet.likelihood import rv_loglike
from orblet.constants import DAYS_PER_KEPLER_YEAR
from fit_rv import fit_rv

HERE=Path(__file__).parent
PARALLAX,PARALLAX_ERROR=46.08,.27

def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def main():
    rv_summary=fit_rv(HERE)
    sed=SED.load(HERE/'sed.npz');stellar=StellarModel()
    original=(sed.flux.copy(),sed.error.copy(),sed.mask.copy())
    distance_sed=SED(sed.flux.copy(),sed.error.copy(),sed.mask.copy(),PARALLAX,PARALLAX_ERROR,sed.source_id,sed.metadata.copy())
    sed_only=fit(distance_sed,kind='binary',model=stellar,age_gyr=None,feh=None)
    rv=np.genfromtxt(HERE/'hd195987_rv.csv',delimiter=',',names=True)
    t=rv['bjd']-2400000.5
    r=rv_summary['parameters'];period=r['period_days']/DAYS_PER_KEPLER_YEAR
    common=dict(period_yr=period,ecc=r['ecc'],tau=0.,epoch_ref_mjd=r['tp_bjd_minus_2400000_5'])
    shape=rv_curve(t,omega_rad=r['omega_rad'],K_kms=1.,offset_kms=0.,**common)
    q0=rv_summary['q'];qscale=rv_summary['q_formal_error']

    def rv_profile(q,max_k=np.inf):
        a1,a2=shape,-shape/q
        w1,w2=1/rv['err1_kms']**2,1/rv['err2_kms']**2
        a1c,a2c=a1-np.average(a1,weights=w1),a2-np.average(a2,weights=w2)
        y1c=rv['rv1_kms']-np.average(rv['rv1_kms'],weights=w1)
        y2c=rv['rv2_kms']-np.average(rv['rv2_kms'],weights=w2)
        k=(np.sum(w1*a1c*y1c)+np.sum(w2*a2c*y2c))/(np.sum(w1*a1c*a1c)+np.sum(w2*a2c*a2c))
        k=min(k,max_k)
        g1=np.average(rv['rv1_kms']-k*a1,weights=w1)
        g2=np.average(rv['rv2_kms']-k*a2,weights=w2)
        return k,g1,g2

    def unpack(v):
        m1,age,feh,z,x=v
        return m1,age,feh,PARALLAX+z*PARALLAX_ERROR,q0+x*qscale

    def parts(v,details=False):
        m1,age,feh,plx,q=unpack(v)
        edge_k=semi_amplitude_kms(mass_msun=m1*q,period_yr=period,ecc=r['ecc'],M_total_msun=m1*(1+q))
        k,g1,g2=rv_profile(q,max_k=edge_k)
        sin_i=k/edge_k
        if not 0<sin_i<=1:return None if details else np.full(4,-np.inf)
        inc=np.arcsin(sin_i)
        ll=np.array([loglike_sed(sed,m1=m1,q=q,age_gyr=age,feh=feh,parallax_mas=plx,model=stellar),
            rv_loglike(t,rv['rv1_kms'],rv['err1_kms'],omega_rad=r['omega_rad'],mass_msun=m1*q*sin_i,M_msun=m1*(1+q),offset_kms=g1,jitter_kms=0.,**common),
            rv_loglike(t,rv['rv2_kms'],rv['err2_kms'],omega_rad=r['omega_rad']+np.pi,mass_msun=m1*sin_i,M_msun=m1*(1+q),offset_kms=g2,jitter_kms=0.,**common),-.5*v[3]**2])
        if details:return ll,inc,k,g1,g2
        return ll

    bounds=((.6,1.3),(.5,10.),(-1.,.5),(-3.,3.),((.1-q0)/qscale,(1.-q0)/qscale))
    starts=([max(.86,sed_only['m1']),sed_only['age_gyr'],sed_only['feh'],0.,0.],[.9,5.,-.5,0.,0.],[1.,1.,0.,0.,0.])
    def objective(v):
        ll=parts(v)
        return float(-2*ll.sum()) if np.isfinite(ll).all() else 1e30
    grid=[[mass,age,metal,0.,0.] for mass in (.83,.86,.9,.95) for age in (1.,5.,10.) for metal in (-.5,-.2,0.,.2)]
    seeds=sorted(grid,key=objective)[:3]
    starts=list(starts)+seeds
    candidates=[minimize(objective,start,method='Nelder-Mead',bounds=bounds,
        options=dict(maxiter=2200,xatol=1e-7,fatol=1e-6)) for start in starts]
    optimum=min(candidates,key=lambda x:x.fun)
    assert optimum.fun<1e29,'No supported joint model'
    ll,inc,k,g1,g2=parts(optimum.x,details=True)
    m1,age,feh,plx,q=unpack(optimum.x);prediction=stellar.evaluate(m1,q,age,feh)
    value,chi2,flux=neg2_log_likelihood(sed,prediction,stellar,sed.fit_mask(),plx)
    joint=dict(m1=m1,m2=m1*q,q=q,age_gyr=age,feh=feh,parallax_mas=plx,
        inclination_deg=float(np.rad2deg(inc)),mirror_inclination_deg=float(180-np.rad2deg(inc)),
        k1_kms=k,k2_kms=k/q,gamma1_kms=g1,gamma2_kms=g2,converged=bool(optimum.success),
        objective=float(optimum.fun),sed_chi2=chi2,n_sed=int(sed.fit_mask().sum()),beta_g=prediction['beta_g'],
        at_bounds=[n for n,x,(lo,hi) in zip(('m1','age_gyr','feh','parallax_z','q_scaled'),optimum.x,bounds) if min(x-lo,hi-x)<1e-3*(hi-lo)],
        starts=[dict(initial=list(unpack(start)),objective=float(c.fun),converged=bool(c.success),parameters=list(unpack(c.x))) for start,c in zip(starts,candidates)],
        loglike_parts=dict(zip(('sed','rv1','rv2','orbital_parallax'),ll)))
    def sed_objective(v):
        mass,age_i,metal,z_i,ratio=v
        ll_i=loglike_sed(sed,m1=mass,q=ratio,age_gyr=age_i,feh=metal,
                        parallax_mas=PARALLAX+z_i*PARALLAX_ERROR,model=stellar)
        return -2*ll_i+z_i*z_i if np.isfinite(ll_i) else 1e30
    control=minimize(sed_objective,[m1,age,feh,optimum.x[3],q],method='Nelder-Mead',
        bounds=(*bounds[:4],(.1,1.)),options=dict(maxiter=1600,xatol=1e-6,fatol=1e-6))
    if control.fun<sed_only['objective']:
        mass,age_i,metal,z_i,ratio=control.x
        plx_i=PARALLAX+z_i*PARALLAX_ERROR
        pred_i=stellar.evaluate(mass,ratio,age_i,metal)
        val_i,chi_i,flux_i=neg2_log_likelihood(sed,pred_i,stellar,sed.fit_mask(),plx_i)
        sed_only=dict(m1=mass,m2=mass*ratio,q=ratio,age_gyr=age_i,feh=metal,parallax_mas=plx_i,
            objective=float(control.fun),m2lnl=val_i,chi2=chi_i,n_fit=int(sed.fit_mask().sum()),
            converged=bool(control.success),flux=flux_i,mask=sed.fit_mask(),kind='binary',
            components=pred_i['components']*(plx_i/100)**2,beta_g=pred_i['beta_g'])
    assert sed_only['objective']<=-2*ll[0]+optimum.x[3]**2+1e-5
    result=dict(source_id=sed.source_id,target='HD 195987',sed_only=sed_only,rv_only=rv_summary,joint=joint,
        distance_constraint=dict(parallax_mas=PARALLAX,error_mas=PARALLAX_ERROR,reference='Torres et al. 2002'),
        comparison_only=dict(m1_msun=.844,m1_error=.018,m2_msun=.665,m2_error=.0079,inclination_deg=99.364,inclination_error_deg=.080),
        assumptions='XP only; fixed RV-only period/eccentricity/periastron/phase; Gaussian RV likelihood with published errors and zero jitter; no epoch astrometry; coeval PARSEC, no extinction, no alpha-enhancement parameter',
        orblet_revision='7aa297df4520d1e93c8a7a5a763a1f2d928bfa51')
    (HERE/'summary.json').write_text(json.dumps(result,default=serial,indent=2)+'\n')
    print(json.dumps(dict(sed_q=sed_only['q'],joint=joint),default=serial),flush=True)
    for before,after in zip(original,(sed.flux,sed.error,sed.mask)):np.testing.assert_array_equal(before,after)
    assert np.allclose(rv_profile(q0)[0],r['k1_kms'],rtol=1e-8)
    joint_plot=dict(flux=flux,components=prediction['components']*(plx/100)**2,mask=sed.fit_mask(),kind='binary')
    fig=plot(sed,joint_plot,title='HD 195987: observed XP + SB2 RV')
    fig.axes[0].set_xlim(.38,1.02)
    fig.savefig(HERE/'sed.png',dpi=300,bbox_inches='tight',pad_inches=.02)
    fig.savefig(HERE/'sed.pdf',bbox_inches='tight',pad_inches=.02);plt.close(fig)
    with plt.rc_context(PAPER_STYLE):
        fig,axes=plt.subplots(2,1,figsize=(7.087,4.8),sharex=True,layout='constrained',gridspec_kw=dict(height_ratios=[2,1]))
        phase=((t-common['epoch_ref_mjd'])/r['period_days'])%1
        grid=np.linspace(0,1,600);times=common['epoch_ref_mjd']+grid*r['period_days']
        for number,color,amplitude,gamma in ((1,BINARY,k,g1),(2,SINGLE,k/q,g2)):
            omega=r['omega_rad']+(number-1)*np.pi
            obs=rv['rv'+str(number)+'_kms'];err=rv['err'+str(number)+'_kms']
            pred=rv_curve(t,omega_rad=omega,K_kms=amplitude,offset_kms=gamma,**common)
            curve=rv_curve(times,omega_rad=omega,K_kms=amplitude,offset_kms=gamma,**common)
            axes[0].errorbar(phase,obs,yerr=err,fmt='.',ms=4,color=color,label=('primary' if number==1 else 'secondary'))
            axes[0].plot(grid,curve,color=color,lw=1)
            axes[1].errorbar(phase,(obs-pred)*1000,yerr=err*1000,fmt='.',ms=4,color=color,elinewidth=.6)
        axes[0].set(ylabel='RV [km/s]',title='HD 195987: 52 observed SOPHIE epochs')
        axes[0].legend();axes[1].axhline(0,color='#52514e',lw=.7)
        axes[1].set(xlabel='Orbital phase',ylabel='RV residual [m/s]')
        for ext in ('png','pdf'):fig.savefig(HERE/('rv.'+ext),dpi=300,bbox_inches='tight',pad_inches=.02)
        plt.close(fig)

if __name__=='__main__':main()
