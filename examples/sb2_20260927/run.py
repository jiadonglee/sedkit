"""Fit four Gaia DR3 SB2 systems and compare SED and RV mass ratios."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from sedkit import SED, StellarModel, download, fit, plot
from sedkit.plot import PAPER_STYLE

HERE = Path(__file__).parent


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    with (HERE / 'targets.csv').open() as stream:
        targets = list(csv.DictReader(stream))
    nss = Table.read(HERE / 'nss.ecsv', format='ascii.ecsv')
    model = StellarModel()
    rows, results, observations = [], [], []
    for target in targets:
        sid = target['source_id']
        path = HERE / (sid + '.npz')
        sed = SED.load(path) if path.exists() else download(sid, cache_dir=HERE / 'data')
        sed.save(path)
        orbit = nss[np.asarray(nss['source_id'], str) == sid]
        if len(orbit) != 1 or orbit[0]['nss_solution_type'] != 'SB2':
            raise ValueError('one published SB2 orbit is required for ' + sid)
        k1 = float(orbit[0]['semi_amplitude_primary'])
        k2 = float(orbit[0]['semi_amplitude_secondary'])
        q_rv = min(k1 / k2, k2 / k1)
        result = fit(sed, model=model, age_gyr=None, feh=None)
        fixed = fit(sed, kind='binary', q=q_rv, model=model, age_gyr=None, feh=None)
        binary = result['binary']
        row = dict(source_id=sid, nss_solution_type='SB2', q_rv=q_rv,
                   q_sed=binary['q'], m1=binary['m1'], age_gyr=binary['age_gyr'],
                   feh=binary['feh'], beta_g=binary['beta_g'], delta=result['delta'],
                   single_chi2_n=result['single']['chi2']/result['single']['n_fit'],
                   binary_chi2_n=binary['chi2']/binary['n_fit'],
                   fixed_q_minus_free=fixed['objective']-binary['objective'],
                   fixed_q_converged=fixed['converged'], converged=binary['converged'],
                   at_bounds=';'.join(binary['at_bounds']),
                   n_fit=binary['n_fit'])
        rows.append(row);results.append(result);observations.append(sed)
        with (HERE / (sid + '_fit.json')).open('w') as stream:
            json.dump(dict(result=result, fixed_q=fixed),stream,default=serial,indent=2)
        print(json.dumps(row),flush=True)
        fig = plot(sed, result, title=f'Gaia DR3 {sid}: SB2', path=HERE/(sid+'.png'))
        fig.savefig(HERE/(sid+'.pdf'),bbox_inches='tight',pad_inches=.02)
        plt.close(fig)
    with (HERE/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    with plt.rc_context(PAPER_STYLE):
        fig=plt.figure(figsize=(7.087,8.4),layout='constrained')
        grid=fig.add_gridspec(5,2,height_ratios=[2.1,.9,.16,2.1,.9],hspace=.08,wspace=.18)
        for i,(sed,result,row) in enumerate(zip(observations,results,rows)):
            base=0 if i<2 else 3;column=i%2
            a=fig.add_subplot(grid[base,column])
            b=fig.add_subplot(grid[base+1,column],sharex=a)
            title=(f'({"abcd"[i]}) Gaia DR3 {sed.source_id}\n'
                   +rf'$q_{{\rm RV}}={row["q_rv"]:.3f}$, $q_{{\rm SED}}={row["q_sed"]:.3f}$')
            plot(sed,result,axes=(a,b),components=False,title=title)
            a.tick_params(labelbottom=False)
            a.set_title(title,loc='left',fontsize=8.2,pad=4)
            for axis in (a,b):
                axis.tick_params(labelsize=8)
                axis.yaxis.label.set_size(8.5)
            b.xaxis.label.set_size(9)
        handles,labels=fig.axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',ncol=5,fontsize=8.2,
                   handlelength=1.6,columnspacing=.9)
        for extension in ('pdf','png'):
            fig.savefig(HERE/('overview.'+extension),dpi=300,bbox_inches='tight',pad_inches=.02)
        plt.close(fig)


if __name__=='__main__':
    main()
