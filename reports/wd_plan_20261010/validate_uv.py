"""GALEX on/off on held-out, low-proper-motion DA anchors."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
import requests
from sedkit import SED, WhiteDwarfModel
from sedkit.subdwarf import GALEX_BRIGHT
from sedkit.subdwarf import _galex_flux
from sedkit.whitedwarf import _Hypothesis, _prepare
from xp_data import read

DATA = Path(__file__).resolve().parents[2] / "data/whitedwarf"
OUT = Path(__file__).resolve().parent


def query(row):
    directory = DATA / "galex"
    directory.mkdir(exist_ok=True)
    path = directory / (str(row["source_id"])+".vot")
    # Propagate Gaia 2016 positions to 2007. At mu<80 mas/yr, the
    # remaining 2003--2012 epoch uncertainty is at most 0.4 arcsec.
    ra = row["ra"]-9*row["pmra"]/3.6e6/np.cos(np.deg2rad(row["dec"]))
    dec = row["dec"]-9*row["pmdec"]/3.6e6
    if not path.exists():
        response=requests.get("https://vizier.cds.unistra.fr/viz-bin/votable", params={
            "-source":"II/335/galex_ais","-c":f"{ra} {dec}","-c.rs":3,
            "-out":"RAdeg,DEdeg,FUVmag,e_FUVmag,NUVmag,e_NUVmag,Fafl,Nafl,Fexf,Nexf",
            "-out.max":20},timeout=60)
        response.raise_for_status();path.write_bytes(response.content)
    from astropy.io.votable import parse
    tables=list(parse(str(path)).iter_tables())
    if not tables: return None
    table=tables[0].to_table(use_names_over_ids=True)
    if len(table)!=1: return None
    r=table[0];galex={}
    for band,af,ex in [("FUV","Fafl","Fexf"),("NUV","Nafl","Nexf")]:
        if any(np.ma.is_masked(r[k]) for k in [band+"mag","e_"+band+"mag",af,ex]):continue
        mag,err=float(r[band+"mag"]),float(r["e_"+band+"mag"])
        if mag>GALEX_BRIGHT[band] and 0<err<.2 and r[af]==0 and r[ex]==0:
            galex[band]=(mag,err,True)
    return (int(row["source_id"]),galex) if galex else None


def main():
    cohort=Table.read(DATA/"anchors/calibration_local100.ecsv")
    path=DATA/"anchors/local100_motion.vot"
    if not path.exists():
        q="SELECT source_id,ra,dec,pmra,pmdec FROM gaiadr3.gaia_source WHERE source_id IN ("+",".join(map(str,cohort["source_id"]))+")"
        r=requests.post("https://gaia.ari.uni-heidelberg.de/tap/sync",data=dict(REQUEST="doQuery",LANG="ADQL",FORMAT="votable",QUERY=q,MAXREC=1000),timeout=90)
        r.raise_for_status();path.write_bytes(r.content)
    motion=read(path)
    motion=motion[np.hypot(motion["pmra"],motion["pmdec"])<80]
    with ThreadPoolExecutor(4) as pool:
        matches=[r for r in pool.map(query,[dict(r) for r in motion]) if r is not None]
    print(f"GALEX: {len(matches)} clean matches from {len(motion)} low-motion anchors",flush=True)
    ci={int(r["source_id"]):r for r in cohort}
    with np.load(DATA/"xp_local100/calibrated.npz") as a:
        fi={int(s):i for i,s in enumerate(a["source_id"])};flux,error=a["flux"],a["error"]
    uv_residual=np.full((len(matches),2),np.nan)
    folds=np.array([ci[sid]["fold"] for sid,_ in matches])
    for j,(sid,galex) in enumerate(matches):
        row=ci[sid];i=fi[sid]
        with np.load(DATA/"calibration_folds"/f"{row['fold']}.npz") as a:c={k:a[k] for k in a.files}
        model=WhiteDwarfModel(calibration=c)
        pred=model.predict(row["teff"],logg=row["logg"],radius=1.)
        good=flux[i,6:]>5*error[i,6:]
        scale=np.exp(np.mean(np.log(flux[i,6:][good]/pred["flux"][:61][good])))
        for k,band in enumerate(["FUV","NUV"]):
            if band in galex:
                observed=_galex_flux(*galex[band][:2],model.galex_pivot_nm[band],band)[0]
                uv_residual[j,k]=np.log(observed/(scale*model.passband(pred["coarse"],band)))
    def uv_calibration(train):
        a=np.nanmedian(uv_residual[train],axis=0)
        diag=np.maximum(.05,1.4826*np.nanmedian(abs(uv_residual[train]-a),axis=0))
        return dict(uv_a=a,uv_diag=diag)
    path=Path(__file__).resolve().parents[2]/"src/sedkit/models/whitedwarf/calibration.npz"
    with np.load(path) as a:final={k:a[k] for k in a.files}
    final.update(uv_calibration(np.ones(len(matches),bool)));np.savez_compressed(path,**final)
    rows=[]
    for j,(sid,galex) in enumerate(matches):
        r=ci[sid];i=fi[sid];f,e=np.full(168,np.nan),np.full(168,np.nan)
        f[:61],e[:61]=flux[i,6:],error[i,6:]
        sed=SED(f,e,np.arange(168)<61,r["parallax"],r["parallax_error"],str(sid))
        with np.load(DATA/"calibration_folds"/f"{r['fold']}.npz") as a:c={k:a[k] for k in a.files}
        out=dict(source_id=str(sid),teff_spec=float(r["teff"]),galex=galex)
        for label,uv in [("off",None),("raw_uv",galex),("on",galex)]:
            correction=dict(c)
            if label=="on": correction.update(uv_calibration(folds!=r["fold"]))
            p=_prepare(sed,WhiteDwarfModel(calibration=correction),None,0.,None,None,None,False,None,uv,False,{"logg":(float(r["logg"]),.02)},None)
            fit,_=_Hypothesis(p,"wd",True,5.,0.).fit();out[label]=fit["whitedwarf"]["teff"]
        rows.append(out)
    (DATA/"validation_uv.json").write_text(json.dumps(rows,indent=2)+"\n")
    summary={"n":len(rows),"n_low_motion":len(motion),"temperature":{}}
    summary["calibration"]={k:v.tolist() for k,v in uv_calibration(np.ones(len(matches),bool)).items()}
    for label in ["off","raw_uv","on"]:
        d=np.array([r[label]/r["teff_spec"]-1 for r in rows])
        summary["temperature"][label]=dict(median=float(np.median(d)),scatter=float(1.4826*np.median(abs(d-np.median(d)))))
    (OUT/"validation_uv.json").write_text(json.dumps(summary,indent=2)+"\n");print(summary)


if __name__=="__main__": main()
