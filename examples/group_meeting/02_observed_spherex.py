"""Observed SPHEREx: extend the SED"""

from pathlib import Path
import sys, json, csv
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown

ROOT = Path(globals().get('__file__', Path.cwd())).resolve()
if ROOT.is_file():
    ROOT = ROOT.parent
while not (ROOT / 'src' / 'sedkit').is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if not (ROOT / 'src' / 'sedkit').is_dir():
    raise RuntimeError('Open this notebook from the sedkit checkout.')
sys.path.insert(0, str(ROOT / 'src'))
from sedkit import SED, StellarModel, download, fit, plot
from sedkit.plot import PAPER_STYLE, BINARY, SINGLE

OUT = ROOT / 'examples' / 'group_meeting' / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
model = StellarModel()

from sedkit import download_spherex, load_spherex
SOURCE_ID = '858860697467058688'
base = SED.load(ROOT / 'examples' / 'sb2_20260927' / (SOURCE_ID + '.npz'))
products = ROOT / 'examples' / 'spherex_20260927'
exposures = np.genfromtxt(products / 'aperture_exposures.csv', delimiter=',', names=True,
                         dtype=None, encoding='utf-8')
valid = np.asarray(exposures['valid'], str) == 'True'
print(f'Measured exposures: {len(exposures)}; complete, unflagged: {valid.sum()}')
print('Original exposure fluxes and errors are preserved in the CSV.')

with plt.rc_context(PAPER_STYLE):
    fig, ax = plt.subplots(figsize=(7.087, 2.6), layout='constrained')
    bins = np.linspace(.7, 5.1, 23)
    ax.hist(exposures['wavelength_um'], bins=bins, histtype='step', color='#777777', label='measured')
    ax.hist(exposures['wavelength_um'][valid], bins=bins, histtype='step', color=BINARY, label='usable')
    ax.set(xlabel='Wavelength [micron]', ylabel='Exposure count')
    ax.legend()
    for extension in ('png', 'pdf'):
        fig.savefig(OUT / ('02_exposure_quality.' + extension), dpi=300, bbox_inches='tight')
    plt.close(fig)
display(Image(filename=str(OUT / '02_exposure_quality.png')))

LIVE_DOWNLOAD = False
combined = (download_spherex(base, cache_dir=OUT / 'cache') if LIVE_DOWNLOAD
            else load_spherex(base, products / 'aperture_spectrum.csv', method='aperture'))
print(f'Usable SPHEREx channels: {combined.mask[66:].sum()} / 102')
print(f'Combined fitted channels: {combined.fit_mask().sum()}')
np.testing.assert_array_equal(combined.flux[:66], base.flux[:66])
np.testing.assert_array_equal(combined.error[:66], base.error[:66])

xp_result = fit(base, kind='binary', model=model, age_gyr=None, feh=None)
sph_result = fit(combined, model=model, age_gyr=None, feh=None)
for label, r in [('XP + JHKs', xp_result), ('XP + JHKs + SPHEREx', sph_result['binary'])]:
    print(f"{label}: q={r['q']:.4f}, M1={r['m1']:.3f}, age={r['age_gyr']:.2f} Gyr, "
          f"[M/H]={r['feh']:.2f}, chi2/N={r['chi2']/r['n_fit']:.2f}, bounds={r['at_bounds']}")
print('Published RV comparison: q = 0.8679')

fig = plot(combined, sph_result, title='Observed SB2: XP + JHKs + SPHEREx',
           path=OUT / '02_fit.png')
fig.savefig(OUT / '02_fit.pdf', bbox_inches='tight', pad_inches=.02)
plt.close(fig)
display(Image(filename=str(OUT / '02_fit.png')))

# Optional PSF exercise; provide an existing XphereX PSF extraction.
RUN_PSF_COMPARISON = False
if RUN_PSF_COMPARISON:
    psf = load_spherex(base, Path('path/to/psf_spectrum.csv'), method='psf')
    psf_fit = fit(psf, kind='binary', model=model, age_gyr=None, feh=None)
    print(f"PSF q={psf_fit['q']:.4f}; aperture q={sph_result['binary']['q']:.4f}")

