"""Build the DA atmosphere and thick/thin-H cooling tables from public SVO/Montreal data.

python scripts/build_whitedwarf_model.py select WORK
python scripts/build_whitedwarf_model.py fetch WORK --operator DIR --hot-run DIR
python scripts/build_whitedwarf_model.py table WORK

Koester surface fluxes are 4 pi H_lambda in erg s-1 cm-2 A-1, at air wavelengths.
Channels outside the downloaded spectrum remain NaN and are not fitted.
The Balmer index is normalized by this DA grid, independently of hot/sdB grids.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np

from build_subdwarf_model import Reducer, _screen, _uv_curve, COARSE, FILTERS, XP_SAMPLING, ROOT

OUT = ROOT / "src/sedkit/models/whitedwarf"
SVO = "https://svo2.cab.inta-csic.es/theory/newov2/"
MONTREAL = "https://www.astro.umontreal.ca/~bergeron/CoolingModels/CoolingModels/"


def select(work):
    from bs4 import BeautifulSoup
    import requests

    work.mkdir(parents=True, exist_ok=True)
    response = requests.post(SVO + "index.php", data={
        "models": ",koester2", "params[koester2][teff][min]": 6000,
        "params[koester2][teff][max]": 80000,
        "params[koester2][logg][min]": 7, "params[koester2][logg][max]": 9.5,
        "nres": "all", "boton": "Search"}, timeout=120)
    response.raise_for_status()
    rows = []
    for link in BeautifulSoup(response.text, "html.parser").find_all(
            "a", href=lambda x: x and "format=votable" in x):
        cells = link.find_parent("tr").find_all("td")
        rows.append(dict(teff=float(cells[1].get_text()), logg=float(cells[2].get_text()),
                         url=SVO + link["href"]))
    rows.sort(key=lambda r: (r["teff"], r["logg"]))
    (work / "nodes.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(f"selected {len(rows)} DA spectra")


def fetch(work, operator, hot_run, threads):
    import requests
    from astropy.io.votable import parse_single_table

    nodes = json.loads((work / "nodes.json").read_text())
    reducer = Reducer(operator, hot_run)
    reducer.ew_max = 1.0  # retain EW in nm; the DA normalization is computed at table assembly
    (work / "reduced").mkdir(exist_ok=True)
    np.savez_compressed(work / "filters.npz", **{f"{k}_{c}": v for k, (lam, tr) in reducer.filters.items()
                                                 for c, v in (("wave_nm", lam), ("trans", tr))})
    from sedkit.model import StellarModel
    wave_nm = StellarModel().wavelength_um * 1000

    def one(node):
        path = work / "reduced" / f"{node['teff']:.0f}_{node['logg']:.2f}.npz"
        if path.exists():
            return
        for attempt in range(3):
            try:
                response = requests.get(node["url"], timeout=120)
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
        table = parse_single_table(BytesIO(response.content)).to_table()
        wl = np.asarray(table["WAVELENGTH"], float) / 10
        surface = np.asarray(table["FLUX"], float) * 10
        values = reducer.reduce(wl, surface)
        support = (wave_nm >= wl[0]) & (wave_nm <= wl[-1])
        support[66:] &= ((reducer.spherex_edges[:-1] >= wl[0])
                          & (reducer.spherex_edges[1:] <= wl[-1]))
        values["channels"][~support] = np.nan
        np.savez_compressed(path, teff=node["teff"], logg=node["logg"],
                            wavelength_range_nm=[wl[0], wl[-1]], **values)

    with ThreadPoolExecutor(threads) as pool:
        for i, _ in enumerate(pool.map(one, nodes), 1):
            if i % 50 == 0 or i == len(nodes):
                print(f"DA {i}/{len(nodes)}", flush=True)
    (work / "cooling").mkdir(exist_ok=True)
    for layer in ("thick", "thin"):
        for number in range(20, 131, 5):
            path = work / "cooling" / f"seq_{number:03d}_{layer}.txt"
            if not path.exists():
                response = requests.get(MONTREAL + path.name, timeout=120)
                response.raise_for_status()
                path.write_bytes(response.content)
    print("downloaded thick-H and thin-H cooling sequences", flush=True)


def cooling_table(work, teff, layer):
    masses = np.arange(20, 131, 5) / 100
    radius, age, gravity = (np.full((len(teff), len(masses)), np.nan) for _ in range(3))
    for j, mass in enumerate(masses):
        rows = []
        for line in (work / "cooling" / f"seq_{round(mass * 100):03d}_{layer}.txt").read_text().splitlines():
            fields = line.split()
            if len(fields) == 6 and fields[0].isdigit():
                rows.append(list(map(float, fields)))
        seq = np.array(rows)
        seq = seq[np.argsort(seq[:, 1])]
        within = (teff >= seq[0, 1]) & (teff <= seq[-1, 1])
        x, xp = np.log(teff[within]), np.log(seq[:, 1])
        radius[within, j] = np.exp(np.interp(x, xp, np.log(seq[:, 3]))) / 6.957e10
        age[within, j] = np.interp(x, xp, seq[:, 4]) / 1e9
        gravity[within, j] = np.interp(x, xp, seq[:, 2])
    return dict(teff_ax=teff, mass_ax=masses, radius=radius, age_gyr=age, logg=gravity)


def table(work):
    nodes = json.loads((work / "nodes.json").read_text())
    teff = np.array(sorted({r["teff"] for r in nodes}))
    logg = np.array(sorted({r["logg"] for r in nodes}))
    keys = ("channels", "blue", "coarse", "balmer_w") + tuple(f"pb_{k}" for k in FILTERS)
    cube = {}
    coverage = []
    for node in nodes:
        path = work / "reduced" / f"{node['teff']:.0f}_{node['logg']:.2f}.npz"
        with np.load(path) as archive:
            i, j = np.searchsorted(teff, node["teff"]), np.searchsorted(logg, node["logg"])
            for key in keys:
                cube.setdefault(key, np.full((len(teff), len(logg)) + archive[key].shape, np.nan))
                cube[key][i, j] = archive[key]
            coverage.append(archive["wavelength_range_nm"])
    if not np.isfinite(cube["channels"][..., :64]).all():
        raise ValueError("DA table has holes in XP or J/H/Ks")
    replaced = _screen(cube, teff, keys, logg_ax=logg)
    ew_max = float(cube["balmer_w"].max())
    output = dict(teff_ax=teff, logg_ax=logg, ln_flux=np.log(cube["channels"]).astype(np.float32),
                  ln_blue=np.log(cube["blue"]).astype(np.float32),
                  ln_coarse=np.log(cube["coarse"]).astype(np.float32),
                  balmer_w=(cube["balmer_w"] / ew_max).astype(np.float32), ew_max_nm=ew_max,
                  blue_wavelength_nm=XP_SAMPLING[:6], coarse_wavelength_nm=COARSE,
                  support=np.isfinite(cube["channels"]).all(axis=(0, 1)))
    with np.load(work / "filters.npz") as archive:
        output.update({f"filter/{k}": archive[k] for k in archive.files})
    uv_wave, uv_curve, _ = _uv_curve()
    output.update(uv_curve_wavelength_nm=uv_wave, uv_curve=uv_curve)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "whitedwarf_table.npz", **output)
    for layer in ("thick", "thin"):
        name = "cooling.npz" if layer == "thick" else "cooling_thin.npz"
        np.savez_compressed(OUT / name, **cooling_table(work, teff, layer))
    summary = dict(source="SVO Koester koester2 DA LTE, 4 pi H_lambda, air wavelengths",
                   cooling="Bedard et al. 2020 C/O core; thick q_H=1e-4, thin q_H=1e-10; q_He=1e-2",
                   units="10-pc F_lambda for R=1 Rsun, 1e-18 W m^-2 nm^-1",
                   teff=[teff[0], teff[-1]], logg=[logg[0], logg[-1]],
                   nodes=len(nodes), wavelength_range_nm=np.asarray(coverage).min(axis=0).tolist(),
                   ew_max_nm=ew_max, replaced_outliers=replaced, calibrated=False,
                   supported_channels=int(output["support"].sum()))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "fetch", "table"))
    parser.add_argument("work", type=Path)
    parser.add_argument("--operator", type=Path)
    parser.add_argument("--hot-run", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "select":
        select(args.work)
    elif args.stage == "fetch":
        fetch(args.work, args.operator, args.hot_run, args.threads)
    else:
        table(args.work)
