"""Conditional SED + SB2 RV + along-scan recovery using orblet atoms."""

from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from sedkit import SED, StellarModel, loglike_sed, plot
from sedkit.plot import PAPER_STYLE, BINARY, SINGLE
from orblet.constants import DAYS_PER_KEPLER_YEAR
from orblet.model import campbell_xy, along_scan_model, rv_model, rv_curve, semi_amplitude_kms
from orblet.likelihood import rv_loglike, loglike_along_scan
from orblet.interpret.flux_ratio import signed_photocentre_axis_ratio

HERE = Path(__file__).parent
TRUTH = np.array([.75, .8, np.pi/3, 20., 18.])
NAMES = ('m1', 'q', 'inc_rad', 'parallax_mas', 'gamma_kms')
ORBIT = dict(period_yr=120./DAYS_PER_KEPLER_YEAR, ecc=.25,
             omega=.7, Omega=1.1, tau=.18, epoch_ref_mjd=60000.)
ORBIT['tp_mjd'] = ORBIT['epoch_ref_mjd'] + ORBIT['tau'] * 120.


def forward(theta, data, stellar, *, dark=False):
    m1, q, inc, plx, gamma = theta
    prediction = stellar.evaluate(m1, q, 5., 0.)
    if prediction is None:
        return None
    m2, total = m1*q, m1*(1+q)
    ra1, dec1 = campbell_xy(data['t_al'], period_yr=ORBIT['period_yr'],
        ecc=ORBIT['ecc'], omega=ORBIT['omega'], inc=inc, Omega=ORBIT['Omega'],
        tp_mjd=ORBIT['tp_mjd'], m_comp_msun=m2, M_total_msun=total, plx_mas=plx)
    factor = 1. if dark else prediction['a_phot_over_a1']
    al = along_scan_model(d_ra=factor*ra1, d_dec=factor*dec1, psi=data['psi'],
        t_mjd=data['t_al'], epoch_ref_mjd=ORBIT['epoch_ref_mjd'],
        ra_offset_mas=0., dec_offset_mas=0., pmra_masyr=0., pmdec_masyr=0.,
        plx_mas=plx, parallax_factor_al=data['parallax_factor_al'])
    common = dict(period_yr=ORBIT['period_yr'], ecc=ORBIT['ecc'], tau=ORBIT['tau'],
                  M_msun=total, offset_kms=gamma, epoch_ref_mjd=ORBIT['epoch_ref_mjd'])
    rv1 = rv_model(data['t_rv1'], omega_rad=ORBIT['omega'], mass_msun=m2*np.sin(inc), **common)
    rv2 = rv_model(data['t_rv2'], omega_rad=ORBIT['omega']+np.pi, mass_msun=m1*np.sin(inc), **common)
    return prediction, al, rv1, rv2


def make_mock(stellar):
    rng = np.random.default_rng(20260927)
    data = dict(t_al=np.sort(rng.uniform(60000, 61100, 100)),
                t_rv1=np.sort(rng.uniform(60000, 60800, 48)),
                t_rv2=np.sort(rng.uniform(60000, 60800, 53)))
    data['psi'] = rng.uniform(0, 2*np.pi, len(data['t_al']))
    phase_year = 2*np.pi*(data['t_al']-60000)/DAYS_PER_KEPLER_YEAR
    data['parallax_factor_al'] = (np.sin(phase_year)*np.sin(data['psi'])
                                +.6*np.cos(phase_year)*np.cos(data['psi']))
    m1, q, inc, plx, gamma = TRUTH
    p = stellar.evaluate(m1, q, 5., 0.)
    mean = p['flux_10pc']*(plx/100)**2
    sed = SED(mean+rng.normal(size=168)*.03*mean, .03*mean, np.ones(168, bool),
              plx, .2, 'mock', {'scenario': 'measurement-only coeval binary mock'})
    # Generate a flux-weighted centroid from the two barycentric positions.
    ra1, dec1 = campbell_xy(data['t_al'], period_yr=ORBIT['period_yr'], ecc=.25,
        omega=.7, inc=inc, Omega=1.1, tp_mjd=ORBIT['tp_mjd'],
        m_comp_msun=m1*q, M_total_msun=m1*(1+q), plx_mas=plx)
    beta = p['beta_g']
    ra_ph = (ra1 + beta*(-ra1/q))/(1+beta)
    dec_ph = (dec1 + beta*(-dec1/q))/(1+beta)
    true_al = (ra_ph*np.sin(data['psi']) + dec_ph*np.cos(data['psi'])
               +plx*data['parallax_factor_al'])
    predicted = forward(TRUTH, data, stellar)
    np.testing.assert_allclose(predicted[1], true_al, rtol=1e-12, atol=1e-12)
    for number, companion, omega in ((1,m1*q,.7), (2,m1,.7+np.pi)):
        k = semi_amplitude_kms(mass_msun=companion, period_yr=ORBIT['period_yr'],
                              ecc=.25, M_total_msun=m1*(1+q))*np.sin(inc)
        true_rv = rv_curve(data['t_rv'+str(number)], period_yr=ORBIT['period_yr'],
            ecc=.25, omega_rad=omega, tau=.18, K_kms=k,
            offset_kms=gamma, epoch_ref_mjd=60000.)
        np.testing.assert_allclose(predicted[number+1], true_rv, rtol=1e-12, atol=1e-12)
        data['err_rv'+str(number)] = np.full(len(true_rv), .2)
        data['rv'+str(number)] = true_rv+rng.normal(size=len(true_rv))*.2
    data['err_al'] = np.full(len(true_al), .05)
    data['al'] = true_al+rng.normal(size=len(true_al))*.05
    return sed, data


def likelihood_parts(theta, sed, data, stellar, *, dark=False):
    result = forward(theta, data, stellar, dark=dark)
    if result is None:
        return np.full(4, -np.inf)
    m1, q, inc, plx, gamma = theta
    common = dict(period_yr=ORBIT['period_yr'], ecc=.25, tau=.18,
                  M_msun=m1*(1+q), offset_kms=gamma, jitter_kms=0., epoch_ref_mjd=60000.)
    return np.array([
        loglike_sed(sed, m1=m1, q=q, age_gyr=5, feh=0,
                    parallax_mas=plx, model=stellar),
        rv_loglike(data['t_rv1'],data['rv1'],data['err_rv1'],
                   omega_rad=.7, mass_msun=m1*q*np.sin(inc), **common),
        rv_loglike(data['t_rv2'],data['rv2'],data['err_rv2'],
                   omega_rad=.7+np.pi, mass_msun=m1*np.sin(inc), **common),
        loglike_along_scan(model_along_scan=result[1], centroid_pos=data['al'],
                           centroid_pos_err=data['err_al'],jitter_mas=0.)])


def recover(sed, data, stellar, *, dark=False):
    starts = ([.65,.65,.85,17.,15.], [.85,.95,1.35,23.,21.], [.9,.7,.65,24.,16.])
    bounds = ((.55,1.05),(.5,1.),(.2,np.pi-.2),(10.,30.),(10.,25.))
    candidates = [minimize(lambda theta: -2*likelihood_parts(theta,sed,data,stellar,dark=dark).sum(),
                          start,method='Nelder-Mead',bounds=bounds,
                          options=dict(maxiter=3000,xatol=1e-7,fatol=1e-6))
                  for start in starts]
    return min(candidates,key=lambda result:result.fun)


def diagnostics(data, stellar, result):
    predicted = forward(result.x,data,stellar)
    with plt.rc_context(PAPER_STYLE):
        fig,axes=plt.subplots(2,2,figsize=(7.087,5.3),layout='constrained')
        for n,color,label in ((1,BINARY,'primary'),(2,SINGLE,'secondary')):
            phase=((data['t_rv'+str(n)]-ORBIT['tp_mjd'])/120)%1
            order=np.argsort(phase)
            axes[0,0].errorbar(phase,data['rv'+str(n)],yerr=.2,fmt='.',ms=3,color=color,elinewidth=.5,label=label)
            axes[0,0].plot(phase[order],predicted[n+1][order],color=color,lw=1)
            axes[1,0].plot(phase,(data['rv'+str(n)]-predicted[n+1])/.2,'.',ms=3,color=color)
        axes[0,0].set(xlabel='Orbital phase',ylabel='RV [km/s]')
        axes[0,0].legend(fontsize=8)
        axes[1,0].set(xlabel='Orbital phase',ylabel=r'RV residual / $\sigma$')
        axes[0,1].errorbar(predicted[1],data['al'],yerr=.05,fmt='.',ms=3,color=BINARY,elinewidth=.4)
        low,high=np.min(predicted[1]),np.max(predicted[1])
        axes[0,1].plot([low,high],[low,high],color='#52514e',lw=.8)
        axes[0,1].set(xlabel='Predicted along-scan position [mas]',ylabel='Observed position [mas]')
        axes[1,1].plot((data['t_al']-60000)/DAYS_PER_KEPLER_YEAR,(data['al']-predicted[1])/.05,'.',ms=3,color=BINARY)
        axes[1,1].set(xlabel='Time from reference epoch [yr]',ylabel=r'AL residual / $\sigma$')
        for axis in axes[1]:
            axis.axhline(0,color='#52514e',lw=.7)
        for axis in axes.flat:
            axis.tick_params(labelsize=8)
            axis.xaxis.label.set_size(8.5)
            axis.yaxis.label.set_size(8.5)
        for extension in ('png','pdf'):
            fig.savefig(HERE/('joint.'+extension),dpi=300,bbox_inches='tight',pad_inches=.02)
        plt.close(fig)


def main():
    stellar=StellarModel()
    sed,data=make_mock(stellar)
    original=sed.flux.copy(),sed.error.copy(),sed.mask.copy()
    fit=recover(sed,data,stellar)
    dark=recover(sed,data,stellar,dark=True)
    assert fit.success and np.isfinite(fit.fun)
    assert abs(fit.x[0]-TRUTH[0]) < .025
    assert abs(fit.x[1]-TRUTH[1]) < .025
    assert abs(fit.x[2]-TRUTH[2]) < np.deg2rad(3)
    assert abs(fit.x[3]-TRUTH[3]) < .4
    assert abs(fit.x[4]-TRUTH[4]) < .2
    assert dark.fun > fit.fun
    for q in (.8,1.):
        p=stellar.evaluate(.75,q,5,0)
        np.testing.assert_allclose(p['a_phot_over_a1'],signed_photocentre_axis_ratio(q,p['beta_g']))
    assert stellar.evaluate(.75,1.,5,0)['a_phot_over_a1']==0.
    assert float(signed_photocentre_axis_ratio(.8,1.))<0
    for before,after in zip(original,(sed.flux,sed.error,sed.mask)):
        np.testing.assert_array_equal(before,after)
    sed.save(HERE/'sed.npz')
    np.savez_compressed(HERE/'orbit_data.npz',**data)
    parts=likelihood_parts(fit.x,sed,data,stellar)
    summary=dict(truth=dict(zip(NAMES,TRUTH.tolist())),recovered=dict(zip(NAMES,fit.x.tolist())),
        dark_recovered=dict(zip(NAMES,dark.x.tolist())),converged=bool(fit.success),
        dark_converged=bool(dark.success),delta_dark_objective=float(dark.fun-fit.fun),
        loglike_parts=dict(zip(('sed','rv1','rv2','along_scan'),parts.tolist())),
        n_sed=int(sed.fit_mask().sum()),n_rv1=48,n_rv2=53,n_al=100,
        orbit=ORBIT,age_gyr=5.,feh=0.,seed=20260927,
        orblet_revision='7aa297df4520d1e93c8a7a5a763a1f2d928bfa51',
        assumptions='fixed orbital shape, age, metallicity and astrometric offsets; measurement-only mock')
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    sed_result=dict(flux=fit.x[3]**2/100**2*stellar.evaluate(fit.x[0],fit.x[1],5,0)['flux_10pc'],
                    components=fit.x[3]**2/100**2*stellar.evaluate(fit.x[0],fit.x[1],5,0)['components'],
                    mask=sed.fit_mask(),kind='binary')
    fig=plot(sed,sed_result,path=HERE/'sed.png',title='Simulated coeval binary')
    fig.savefig(HERE/'sed.pdf',bbox_inches='tight',pad_inches=.02)
    plt.close(fig)
    diagnostics(data,stellar,fit)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
