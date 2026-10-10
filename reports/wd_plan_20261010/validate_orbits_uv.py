"""WD-only UV flux bounds from measured total GALEX light in RV systems."""

from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import requests
from astropy.table import Table
from sedkit import WhiteDwarfModel
from sedkit.subdwarf import _galex_flux
from sedkit.subdwarf import GALEX_BRIGHT
from validate_uv import query
from validate_orbits import one

DATA=Path(__file__).resolve().parents[2]/"data/whitedwarf"
OUT=Path(__file__).resolve().parent


def run(task):return one(task[0],task[1])


def query_mis(row):
    from astropy.io.votable import parse
    path=DATA/"galex"/(str(row["source_id"])+"_mis.vot")
    if not path.exists():
        ra=row["ra"]-9*row["pmra"]/3.6e6/np.cos(np.deg2rad(row["dec"]))
        dec=row["dec"]-9*row["pmdec"]/3.6e6
        r=requests.get("https://vizier.cds.unistra.fr/viz-bin/votable",params={
            "-source":"II/312/mis","-c":f"{ra} {dec}","-c.rs":3,
            "-out":"RAdeg,DEdeg,FUV,NUV,e_FUV,e_NUV,Fafl,Nafl,Fexf,Nexf","-out.max":20},timeout=60)
        r.raise_for_status();path.write_bytes(r.content)
    tables=list(parse(str(path)).iter_tables())
    if not tables:return None
    t=tables[0].to_table(use_names_over_ids=True)
    if len(t)!=1:return None
    r=t[0];galex={}
    for band,af,ex,offset in [("FUV","Fafl","Fexf",.033),("NUV","Nafl","Nexf",.043)]:
        if any(np.ma.is_masked(r[k]) for k in [band,"e_"+band,af,ex]):continue
        mag,error=float(r[band])+offset,float(r["e_"+band])
        if mag>GALEX_BRIGHT[band] and 0<error<.2 and r[af]==0 and r[ex]==0:
            galex[band]=(mag,error,True)
    return (int(row["source_id"]),galex) if galex else None


def main():
    rows=Table.read(OUT/"orbit_anchors.ecsv")
    orbits=Table.read(DATA/"anchors/yamaguchi_orbits.vot")
    motion=orbits[np.hypot(orbits["pmra"],orbits["pmdec"])<300]
    with ThreadPoolExecutor(4) as pool:matches=[r for r in pool.map(query,[dict(r) for r in motion]) if r is not None]
    # MIS adds deeper coverage; retain current AIS bands where available.
    with ThreadPoolExecutor(4) as pool:deep=[r for r in pool.map(query_mis,[dict(r) for r in motion]) if r is not None]
    combined={sid:g for sid,g in matches}
    for sid,g in deep:
        for band,record in g.items():combined.setdefault(sid,{}).setdefault(band,record)
    matches=list(combined.items())
    model=WhiteDwarfModel();caps={}
    for sid,galex in matches:
        cap={}
        for band,(mag,error,_) in galex.items():
            f,e=_galex_flux(mag,error,model.galex_pivot_nm[band],band)
            cap[band]=f+3*e
        caps[str(sid)]=cap
    (DATA/"orbit_uv_measurements.json").write_text(json.dumps(matches,indent=2)+"\n")
    (OUT/"orbit_uv_measurements.json").write_text(json.dumps(matches,indent=2)+"\n")
    print(f"UV caps: {len(caps)}/{len(rows)} systems; MIS matches {len(deep)}",flush=True)
    oi={int(r["source_id"]):dict(r) for r in orbits};ir=Table.read(DATA/"xp_orbits/tmass.ecsv")
    baseline={r["source_id"]:r for r in map(json.loads,(DATA/"validation_orbits.jsonl").read_text().splitlines())}
    with np.load(DATA/"xp_orbits/calibrated.npz") as a:
        fi={int(s):i for i,s in enumerate(a["source_id"])}
        tasks=[((dict(r),oi[int(r["source_id"])],a["flux"][fi[int(r["source_id"])]],
            a["error"][fi[int(r["source_id"])]],ir[ir["source_id"]==r["source_id"]]),caps[str(r["source_id"])])
            for r in rows if str(r["source_id"]) in caps]
    output=[dict(r,galex_upper_limits={}) for sid,r in baseline.items() if sid not in caps]
    with ProcessPoolExecutor(4) as pool,(DATA/"validation_orbits_uv.jsonl").open("w") as h:
        for r in output:h.write(json.dumps(r)+"\n")
        for i,r in enumerate(pool.map(run,tasks)):
            output.append(r);h.write(json.dumps(r)+"\n");h.flush();print(f"UV orbit {i+1}/{len(tasks)}",flush=True)
    valid=[r for r in output if r["gaia_rv_consistent"]];uv=[r for r in valid if r["galex_upper_limits"]]
    summary=dict(n=len(output),n_with_uv=len(caps),n_consistent_with_uv=len(uv),
        n_mis_matches=len(deep),
        mass_shift_le_002=sum(r["mass_shift"]<=.02 for r in valid),
        mass_shift_median=float(np.nanmedian([r["mass_shift"] for r in valid])),
        uv_mass_shift_le_002=sum(r["mass_shift"]<=.02 for r in uv),
        uv_mass_shift_median=float(np.nanmedian([r["mass_shift"] for r in uv])),
        missing_uv=[r["name"] for r in valid if not r["galex_upper_limits"]])
    (OUT/"validation_orbits_uv.json").write_text(json.dumps(summary,indent=2)+"\n");print(summary)


if __name__=="__main__":main()
