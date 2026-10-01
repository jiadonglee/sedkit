"""Gaia orbit: faint companion or hidden twin?"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image

ROOT = Path(globals().get('__file__', Path.cwd())).resolve()
if ROOT.is_file():
    ROOT = ROOT.parent
while not (ROOT / 'src' / 'sedkit').is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if not (ROOT / 'src' / 'sedkit').is_dir():
    raise RuntimeError('Open this notebook from the sedkit checkout.')
sys.path.insert(0, str(ROOT / 'src'))
from sedkit import SED, StellarModel
from sedkit.orbit import Q_GRID, amrf, amrf_observed, locus, rank_roots, solve_orbit
from sedkit.plot import BINARY, INK, INK2, PAPER_STYLE, SINGLE

OUT = ROOT / 'examples' / 'group_meeting' / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
model = StellarModel()

SYSTEMS = [  # source_id, name, follow-up, a0 [mas], two-body parallax [mas], P [d], M1 [Msun]
    ('1916454200349735680', 'Gaia DR3 1916454200349735680', 'RV: near-equal-mass binary',
     0.4404, 26.953, 238.50, 0.644),
    ('5148853253106611200', 'LP 769-9', 'RV: substellar companion',
     0.6978, 13.913, 339.57, 0.686),
]
roots = {}
for sid, name, truth, a0, plx, period, m1 in SYSTEMS:
    roots[sid] = solve_orbit(a0, plx, period, m1, model=model)
    print(f'{name}: A = {amrf_observed(a0, plx, period, m1):.3f}')
    for r in roots[sid]:
        print(f"  {r['kind']:8s} q={r['q']:.2f}  M2={r['m2']:.3f}  beta_G={r['beta_G']:.3f}"
              f"  predicted dG={r['delta_G']:.2f}  dKs={r['delta_Ks']:.2f} mag")

ranked = {}
for sid, name, truth, a0, plx, period, m1 in SYSTEMS:
    sed = SED.load(ROOT / 'examples' / 'orbit_20260930' / (sid + '.npz'))
    ranked[sid] = (sed, rank_roots(sed, roots[sid], parallax_mas=plx, model=model))
    best = ranked[sid][1][0]
    print(f'{name} ({truth})')
    print(f"  SED prefers the {best['kind']} root, q={best['q']:.2f}")
    for r in ranked[sid][1][1:]:
        print(f"  {r['kind']} root, q={r['q']:.2f}: delta objective = {r['delta']:.0f}")

colour = {'dark': SINGLE, 'faint': SINGLE, 'luminous': BINARY}
wave = model.wavelength_um
order = np.argsort(wave[:61])
with plt.rc_context(PAPER_STYLE):
    fig, axes = plt.subplots(2, 2, figsize=(7.087, 4.9), layout='constrained',
                             gridspec_kw={'width_ratios': [1, 1.5]})
    for row, (sid, name, truth, a0, plx, period, m1) in zip(axes, SYSTEMS):
        a_obs = amrf_observed(a0, plx, period, m1)
        beta_g, _ = locus(m1, model=model)
        q = np.linspace(.005, 1, 200)
        ax = row[0]
        ax.plot(q, amrf(q, 0.), color=INK2, ls=':', lw=1, label='dark companion')
        ax.plot(Q_GRID, amrf(Q_GRID, beta_g), color=INK, lw=1.4, label='main-sequence companion')
        ax.axhline(a_obs, color=INK2, ls='--', lw=.8)
        for r in roots[sid]:
            ax.plot(r['q'], a_obs, 'o', ms=7, color=colour[r['kind']], mec='white', zorder=5)
            ax.annotate(f"{r['kind']}\n$q={r['q']:.2f}$", (r['q'], a_obs), xytext=(0, 12),
                        textcoords='offset points', fontsize=7.5, va='bottom',
                        ha='left' if r['q'] < .5 else 'right', color=colour[r['kind']])
        ax.set(xlim=(0, 1.02), ylim=(0, .4), xlabel='mass ratio $q$', ylabel=r'AMRF $\mathcal{A}$')
        ax.set_title(f'{name}\n{truth}', loc='left', fontsize=9)
        ax = row[1]
        sed = ranked[sid][0]
        for r in ranked[sid][1]:
            fit = r['fit']
            res = np.where(fit['mask'], (sed.flux - fit['flux']) / sed.error, np.nan)
            ax.plot(wave[:61][order], res[:61][order], color=colour[r['kind']], lw=1.1,
                    label=f"{r['kind']} root: $\\Delta$objective = {r['delta']:.0f}")
            ax.plot(wave[61:64], res[61:64], 'o', ms=4, color=colour[r['kind']])
        ax.axhline(0, color=INK2, lw=.6)
        ax.set_xscale('log')
        ax.set_xticks([.4, .6, 1, 2], ['0.4', '0.6', '1', '2'])
        ax.minorticks_off()
        ax.set(xlabel=r'wavelength [$\mu$m]', ylabel='(observed $-$ model) / error', ylim=(-25, 25))
        ax.legend(loc='best', fontsize=7.5)
    fig.savefig(OUT / '04_orbit_roots.png')
    fig.savefig(OUT / '04_orbit_roots.pdf')
plt.close(fig)
display(Image(filename=str(OUT / '04_orbit_roots.png')))
