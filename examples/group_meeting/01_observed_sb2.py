"""Observed SB2: download, fit, inspect"""

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

SOURCE_ID = '858860697467058688'
LIVE_DOWNLOAD = False
snapshot = ROOT / 'examples' / 'sb2_20260927' / (SOURCE_ID + '.npz')
sed = (download(SOURCE_ID, cache_dir=OUT / 'cache')
       if LIVE_DOWNLOAD else SED.load(snapshot))
print(f'Gaia DR3 {sed.source_id}')
print(f'Parallax: {sed.parallax_mas:.3f} +/- {sed.parallax_error_mas:.3f} mas')
print(f'Available channels: XP={sed.mask[:61].sum()}, broadbands={sed.mask[61:66].sum()}')
print(f'Fitted channels: {sed.fit_mask().sum()} (W1/W2 held out)')

fig = plot(sed, title='Observed SB2: Gaia XP + broadband photometry',
           path=OUT / '01_observations.png')
plt.close(fig)
display(Image(filename=str(OUT / '01_observations.png')))

before = (sed.flux.copy(), sed.error.copy(), sed.mask.copy())
result = fit(sed, model=model, age_gyr=None, feh=None)
for name in ('single', 'binary'):
    r = result[name]
    print(f"{name:6s}: M1={r['m1']:.3f}, q={r['q']:.3f}, "
          f"age={r['age_gyr']:.2f} Gyr, [M/H]={r['feh']:.2f}, "
          f"chi2/N={r['chi2']/r['n_fit']:.2f}, bounds={r['at_bounds']}")
print(f"Single minus binary objective: {result['delta']:.1f}")
for original, current in zip(before, (sed.flux, sed.error, sed.mask)):
    np.testing.assert_array_equal(original, current)

fig = plot(sed, result, title='Observed SB2: single and coeval-binary fits',
           path=OUT / '01_fit.png')
fig.savefig(OUT / '01_fit.pdf', bbox_inches='tight', pad_inches=.02)
plt.close(fig)
display(Image(filename=str(OUT / '01_fit.png')))

with (ROOT / 'examples' / 'sb2_20260927' / 'targets.csv').open() as stream:
    target = next(row for row in csv.DictReader(stream) if row['source_id'] == SOURCE_ID)
k1, k2 = float(target['semi_amplitude_primary']), float(target['semi_amplitude_secondary'])
q_rv = min(k1/k2, k2/k1)
print(f"q from RV = {q_rv:.4f}; q from SED = {result['binary']['q']:.4f}")

# Optional exercise; enable it after the presentation.
RUN_EXERCISE = False
if RUN_EXERCISE:
    fixed_q = fit(sed, kind='binary', model=model, q=q_rv, age_gyr=None, feh=None)
    print(f"Fixed-q minus free-q objective: {fixed_q['objective']-result['binary']['objective']:.2f}")

