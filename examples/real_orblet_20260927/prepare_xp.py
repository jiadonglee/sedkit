from pathlib import Path
import numpy as np
from astropy.table import Table
from gaiaxpy import calibrate
from sedkit import SED,StellarModel
p=Path(__file__).parent;data=p/'data/2067948245320365184'
m=StellarModel()
table=Table.read(data/'xp.xml',format='votable')
frame=table.to_pandas()
for column in frame.columns:
    if isinstance(frame[column].iloc[0],(list,np.ndarray)):
        frame[column]=frame[column].map(lambda value: np.asarray(value))
calibrated,sampling=calibrate(frame,sampling=m.wavelength_um[:61]*1000,truncation=False,save_file=False)
assert len(calibrated)==1
row=calibrated.iloc[0];flux=np.full(168,np.nan);error=flux.copy();mask=np.zeros(168,bool)
flux[:61]=np.asarray(row['flux'])*1e18;error[:61]=np.asarray(row['flux_error'])*1e18
mask[:61]=np.isfinite(flux[:61])&np.isfinite(error[:61])&(error[:61]>0)
g=Table.read(data/'gaia.ecsv',format='ascii.ecsv')[0]
metadata={k:g[k].item() if isinstance(g[k],np.generic) else g[k] for k in g.colnames}
metadata.update(acquisition_tap='https://gaia.ari.uni-heidelberg.de/tap/sync',xp_error_model='GaiaXPy marginal errors; inter-channel correlations omitted',distance_constraint=dict(parallax_mas=46.08,error_mas=.27,reference='Torres et al. 2002, astro-ph/0205511'))
s=SED(flux,error,mask,float(g['parallax']),float(g['parallax_error']),str(g['source_id']),metadata);s.save(p/'sed.npz')
print('XP channels',mask.sum(),'Original Gaia parallax',s.parallax_mas,s.parallax_error_mas,flush=True)
