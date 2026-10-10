"""Acquire and calibrate XP coefficient tables for the WD validation samples."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from astropy.io.votable import parse
from astropy.table import vstack


def read(path):
    doc = parse(str(path))
    for resource in doc.resources:
        for info in resource.infos:
            if info.name == "QUERY_STATUS" and info.value != "OK":
                raise RuntimeError(info.content)
    return doc.get_first_table().to_table(use_names_over_ids=True)


def acquire(ids, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ids = np.unique(np.asarray(ids, dtype=np.int64))
    blocks = [ids[i:i+40] for i in range(0, len(ids), 40)]

    def one(item):
        i, block = item
        path = directory / f"{i:03d}.vot"
        if path.exists():
            table = read(path)
            if set(map(int, table["source_id"])) == set(map(int, block)):
                return table
        query = "SELECT * FROM gaiadr3.xp_continuous_mean_spectrum WHERE source_id IN (" + ",".join(map(str, block)) + ")"
        response = requests.post("https://gaia.ari.uni-heidelberg.de/tap/sync",
                                 data=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="votable", QUERY=query, MAXREC=100),
                                 timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
        table = read(path)
        if set(map(int, table["source_id"])) != set(map(int, block)):
            raise ValueError(f"incomplete XP response for batch {i}")
        print(f"{directory.name}: batch {i+1}/{len(blocks)}", flush=True)
        return table

    with ThreadPoolExecutor(3) as pool:
        tables = list(pool.map(one, enumerate(blocks)))
    table = vstack(tables, metadata_conflicts="silent")
    return table


def calibrate_table(table, directory):
    from gaiaxpy import calibrate
    import pandas as pd

    # Preserve NumPy coefficient arrays in the GaiaXPy DataFrame input.
    data = pd.DataFrame({k: [np.asarray(v) if np.ndim(v) > 0 else int(v) if k.endswith("parameters") else v
                            for v in table[k]] for k in table.colnames})
    result, wave = calibrate(data, sampling=np.arange(332., 993., 10.), truncation=False, save_file=False)
    output = dict(source_id=result["source_id"].to_numpy(dtype=np.int64),
                  flux=np.stack(result["flux"]) * 1e18, error=np.stack(result["flux_error"]) * 1e18,
                  wavelength_nm=wave)
    np.savez_compressed(Path(directory) / "calibrated.npz", **output)
    return output
