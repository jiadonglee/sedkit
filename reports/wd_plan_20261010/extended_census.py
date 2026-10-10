"""XP availability in photometric WD and astrometric candidate catalogues."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
import requests
from astropy.table import Table, vstack, join, unique
from astropy.io import fits
from xp_data import read, acquire, calibrate_table

DATA=Path(__file__).resolve().parents[2]/"data/whitedwarf"
OUT=Path(__file__).resolve().parent


def gaia(ids,label,local=False):
    directory=DATA/"anchors"/label;directory.mkdir(exist_ok=True)
    ids=np.unique(np.asarray(ids,dtype=np.int64));ids=ids[ids>0]
    blocks=[ids[i:i+1000] for i in range(0,len(ids),1000)]
    def one(item):
        i,block=item;p=directory/f"{i:03d}.vot"
        if not p.exists():
            q="SELECT source_id,parallax,parallax_error,phot_g_mean_mag,bp_rp,ruwe,has_xp_continuous FROM gaiadr3.gaia_source WHERE source_id IN ("+",".join(map(str,block))+")"
            if local:q+=" AND parallax>6.666666667 AND parallax/parallax_error>20 AND ruwe<1.4"
            r=requests.post("https://gaia.ari.uni-heidelberg.de/tap/sync",data=dict(REQUEST="doQuery",LANG="ADQL",FORMAT="votable",QUERY=q,MAXREC=1100),timeout=120)
            r.raise_for_status();p.write_bytes(r.content)
        t=read(p);print(label,i+1,len(t),flush=True);return t
    with ThreadPoolExecutor(3) as pool: tables=list(pool.map(one,enumerate(blocks)))
    return vstack(tables,metadata_conflicts="silent")


def main():
    result=[]
    for label,path,key,local in [
        ("Shahaf2024","/Users/jdli/Project/data/WDMS/Shahaf24.fits","source_id",False),
        ("GF2021_100pc",str(DATA/"anchors/gf_local100.ecsv"),"source_id",False),
        ("astra_local150",str(DATA/"anchors/astra_controls.ecsv"),"gaia_dr3_source_id",True)]:
        table=Table.read(path)
        t=gaia(table[key],label,local)
        xp=np.asarray(t["has_xp_continuous"],bool)
        result.append(dict(catalogue=label,requested=len(table),gaia=len(t),xp=int(xp.sum())))
        if label=="Shahaf2024":
            updated=Table.read(DATA/"anchors/shahaf2025_updated.fits")
            result[-1].update(sample="non-class-I parent; not RV-confirmed and not exclusively class III",
                updated_2025_nce=int(sum(updated["red_prob"]<.64)),updated_red_probability_threshold=.64)
        if label=="astra_local150":
            table.rename_column(key,"source_id");t=join(table,t,keys="source_id")
            old=set(map(int,Table.read(DATA/"xp_fgk/sample.ecsv")["Source"]))
            t=t[np.asarray(t["has_xp_continuous"],bool)&np.array([int(s) not in old for s in t["source_id"]])]
            t.sort("snr",reverse=True);t=unique(t,keys="source_id",keep="first")
            # Fixed selection before inspecting the new spectra or fit results.
            rng=np.random.default_rng(20261011);t=t[rng.permutation(len(t))[:200]]
            directory=DATA/"xp_fgk_holdout";directory.mkdir(exist_ok=True)
            t.write(directory/"sample.ecsv",overwrite=True)
            calibrate_table(acquire(t["source_id"],directory),directory)
    (OUT/"extended_counts.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":main()
