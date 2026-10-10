"""Count WD catalogue objects with published Gaia DR3 XP spectra.

Run from the repository root with the astronomy Python environment:
    python reports/wd_plan_20261010/anchor_census.py

Raw catalogues and Gaia query responses are in data/whitedwarf/anchors.
The MWDD subset uses one explicitly spectroscopic reference, Gianninas+2011.
Kilic+2025 parameters are photometric despite its spectroscopic classifications.
"""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import requests
from astropy.io.votable import parse
from astropy.table import Table, join, unique, vstack

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/whitedwarf/anchors"
OUT = Path(__file__).resolve().parent
DESI = DATA / "DESI_EDR_WD_catalogue_online_data/THE_CATALOGUE/DESI_EDR_WD_catalogue_v1.3.fits"


def download_catalogues():
    DATA.mkdir(parents=True, exist_ok=True)
    sources = {
        "kepler2021.vot": "https://vizier.cds.unistra.fr/viz-bin/votable?-source=J/MNRAS/507/4646&-out.all=1&-out.max=100000",
        "gf2021_sdss_ids.vot": "https://vizier.cds.unistra.fr/viz-bin/votable?-source=J/MNRAS/508/3877/sdssspec&-out=GaiaEDR3,SDSS12,Plate,MJD,FiberID,Gmag,specClass&-out.max=100000",
        "gianninas2011_r0320.json": "https://www.montrealwhitedwarfdatabase.org/json/gianninas2011_r0320.json",
        "kilic2025.json": "https://www.montrealwhitedwarfdatabase.org/json/kilic2025.json",
    }
    for name, url in sources.items():
        path = DATA / name
        if not path.exists():
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
    if not DESI.exists():
        path = DATA / "DESI_EDR_WD_catalogue_online_data_v1.1.zip"
        if not path.exists():
            response = requests.get("https://zenodo.org/records/13684288/files/" + path.name,
                                    timeout=180)
            response.raise_for_status()
            path.write_bytes(response.content)
        with ZipFile(path) as archive:
            archive.extract(str(DESI.relative_to(DATA)), DATA)


def records():
    rows = []
    tables = [t.to_table(use_names_over_ids=True)
              for t in parse(str(DATA / "kepler2021.vot")).iter_tables()]
    classes = {str(r["P-M-F"]): str(r["Type"]).strip() for r in tables[0]}
    ids = Table.read(DATA / "gf2021_sdss_ids.vot")
    by_spectrum = {(int(r["Plate"]), int(r["MJD"]), int(r["FiberID"])):
                   int(r["GaiaEDR3"]) for r in ids}
    for r in tables[1]:
        key = tuple(map(int, str(r["P-M-F"]).split("-")))
        rows.append(dict(catalogue="SDSS_DR16", name=str(r["Name"]),
                         source_id=by_spectrum.get(key, 0), spectral_type=classes[str(r["P-M-F"])],
                         teff=float(r["Teff"]), logg=float(r["logg"]),
                         label_method="spectroscopic", known_binary=False,
                         corrected_3d=False))

    master = Table.read(DESI, hdu=1)
    by_name = {str(r["wdj_name"]): r for r in master}
    for hdu, label in [(4, "DESI_DA"), (7, "DESI_DB")]:
        for r in Table.read(DESI, hdu=hdu):
            m = by_name[str(r["wdj_name"])]
            t3, g3 = float(r["teff_3D_0.8"]), float(r["logg_3D_0.8"])
            use3d = np.isfinite(t3) and np.isfinite(g3) and t3 > 0 and g3 > 0
            rows.append(dict(catalogue=label, name=str(r["wdj_name"]),
                             source_id=int(np.ma.filled(m["edr3_source_id"], 0)), spectral_type=str(m["desi_sp_class"]).strip(),
                             teff=t3 if use3d else float(r["teff"]),
                             logg=g3 if use3d else float(r["logg"]),
                             label_method="spectroscopic", known_binary=False,
                             corrected_3d=use3d))

    for filename, label, method in [
        ("gianninas2011_r0320.json", "MWDD_Gianninas2011", "spectroscopic"),
        ("kilic2025.json", "Kilic2025", "photometric"),
    ]:
        for r in json.loads((DATA / filename).read_text())["data"]:
            rows.append(dict(catalogue=label, name=r["name"],
                             source_id=int(r.get("gaiaedr3") or 0),
                             spectral_type=str(r.get("spectype") or ""),
                             teff=float(r.get("teff") or np.nan),
                             logg=float(r.get("logg") or np.nan), label_method=method,
                             known_binary=bool(r.get("binarity")), corrected_3d=False))
    return Table(rows=rows)


def gaia_catalogue(ids):
    """Query actual XP availability by exact EDR3/DR3 identifier."""
    directory = DATA / "gaia_census"
    directory.mkdir(exist_ok=True)
    blocks = [ids[i:i + 400] for i in range(0, len(ids), 400)]

    def fetch(item):
        index, block = item
        query = ("SELECT source_id,ra,dec,parallax,parallax_error,phot_g_mean_mag,bp_rp,ruwe,"
                 "has_xp_continuous,has_xp_sampled FROM gaiadr3.gaia_source WHERE source_id IN ("
                 + ",".join(map(str, block)) + ")")
        path = directory / f"{index:03d}.vot"
        query_path = path.with_suffix(".adql")
        if path.exists() and query_path.read_text() == query:
            return parse(str(path)).get_first_table().to_table(use_names_over_ids=True)
        response = requests.post("https://gea.esac.esa.int/tap-server/tap/sync",
                                 data=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="votable",
                                           QUERY=query, MAXREC=500), timeout=120)
        response.raise_for_status()
        doc = parse(BytesIO(response.content))
        for resource in doc.resources:
            for info in resource.infos:
                if info.name == "QUERY_STATUS" and info.value != "OK":
                    raise RuntimeError(info.content)
        table = doc.get_first_table().to_table(use_names_over_ids=True)
        path.write_bytes(response.content)
        query_path.write_text(query)
        print(f"Gaia batch {index + 1}/{len(blocks)}: {len(table)}", flush=True)
        return table

    with ThreadPoolExecutor(3) as pool:
        tables = list(pool.map(fetch, enumerate(blocks)))
    return vstack(tables, metadata_conflicts="silent")


def main():
    download_catalogues()
    catalogues = records()
    ids = np.unique(catalogues["source_id"])
    gaia = gaia_catalogue(ids[ids > 0])
    gaia.write(DATA / "gaia_census.ecsv", overwrite=True)
    data = join(catalogues, gaia, keys="source_id", join_type="left",
                metadata_conflicts="silent")
    binary_ids = data["source_id"][data["known_binary"] & (data["source_id"] > 0)]
    data["known_binary"] = np.isin(data["source_id"], binary_ids)
    xp = np.asarray(np.ma.filled(data["has_xp_continuous"], False), bool)
    pure_da = np.asarray(data["spectral_type"] == "DA")
    pure_db = np.asarray(data["spectral_type"] == "DB")
    box = ((data["teff"] >= 6000) & (data["teff"] <= 80000)
           & (data["logg"] >= 7) & (data["logg"] <= 9.5))
    candidate = xp & pure_da & box & ~data["known_binary"] & (data["label_method"] == "spectroscopic")
    # Cold 1D gravities need a separate 3D correction before parameter validation.
    ready = candidate & ((data["teff"] >= 13000) | data["corrected_3d"])
    data["da_candidate"] = candidate
    data["da_label_ready"] = ready
    data.write(DATA / "matched_catalogues.ecsv", overwrite=True)
    summary = []
    for cat in dict.fromkeys(data["catalogue"]):
        mask = np.asarray(data["catalogue"] == cat)
        def count(selection):
            return len(np.unique(data["source_id"][mask & selection]))
        summary.append(dict(catalogue=str(cat), rows=int(mask.sum()),
                            gaia_ids=count(data["source_id"] > 0), xp=count(xp),
                            da_xp=count(xp & pure_da), db_xp=count(xp & pure_db),
                            da_in_box=count(candidate), da_label_ready=count(ready)))
    chosen = unique(data[candidate], keys="source_id")
    chosen.write(DATA / "da_candidates.ecsv", overwrite=True)
    result = dict(catalogues=summary, unique_da_candidates=len(chosen),
                  unique_da_label_ready=len(np.unique(data["source_id"][ready])),
                  unique_da_label_ready_astrometry=len(np.unique(data["source_id"][ready
                      & (data["ruwe"] < 1.4) & (data["parallax"] / data["parallax_error"] > 10)])),
                  gaia_ids_requested=int(sum(ids > 0)), gaia_ids_returned=len(gaia))
    (OUT / "anchor_counts.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    plot(data, chosen)


def plot(data, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"DESI_DA": "#d47719", "SDSS_DR16": "#3981b8",
              "MWDD_Gianninas2011": "#41985e"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    for label, color in colors.items():
        sample = data[(data["catalogue"] == label) & data["da_candidate"]]
        axes[0].hist(sample["phot_g_mean_mag"], bins=np.arange(9, 21.5, .5),
                     histtype="step", lw=1.6, color=color, label=f"{label} ({len(sample)})")
    axes[0].axvline(17.65, color="0.45", ls="--", lw=1)
    axes[0].set(xlabel="Gaia G (mag)", ylabel="DA candidates with XP / 0.5 mag",
                title="Catalogue counts before deduplication", xlim=(9, 21))
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].scatter(chosen["teff"] / 1000, chosen["logg"], s=6, alpha=.35, color="#3981b8", rasterized=True)
    axes[1].set(xlabel="Spectroscopic effective temperature (kK)",
                ylabel=r"$\log g$ (cgs)", xlim=(6, 80), ylim=(7, 9.5),
                title=f"{len(chosen):,} distinct Gaia sources")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "anchor_census.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
