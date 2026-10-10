"""Query native integrated Edenhofer moments for the single-DA anchors."""
import json
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from dustmaps.edenhofer2023 import Edenhofer2023Query

WORK=Path(__file__).resolve().parent
MAP="/home/jdli/xiasangju/jdli/mpoor/data/dustmaps/edenhofer_2023/samples_healpix.fits"
rows=json.loads((WORK/"dust_sources.json").read_text());ids=list(rows)
coords=SkyCoord(ra=np.array([rows[s]["ra"] for s in ids])*u.deg,
                dec=np.array([rows[s]["dec"] for s in ids])*u.deg,
                distance=np.array([1000/rows[s]["parallax"] for s in ids])*u.pc)
q=Edenhofer2023Query(map_fname=MAP,integrated=True,load_samples=True)
mean=q(coords,mode="mean");sigma=q(coords,mode="std")
out={s:dict(mean=float(m) if np.isfinite(m) else 0.,
            sigma=float(e) if np.isfinite(e) else .005,
            method="Edenhofer" if np.isfinite(m) else "local E=0 prior inside the map inner radius")
     for s,m,e in zip(ids,mean,sigma)}
(WORK/"dust_moments.json").write_text(json.dumps(out,indent=2)+"\n")
print(len(out),np.nanmedian(mean),np.nanmax(mean),flush=True)
