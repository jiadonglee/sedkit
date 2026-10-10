"""Inspect Koester DA grid support, flux units and one Bédard cooling sequence."""

from io import BytesIO
import json
from pathlib import Path

from astropy.io.votable import parse_single_table
from astropy.table import Table
from bs4 import BeautifulSoup
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/whitedwarf/models"
OUT = Path(__file__).resolve().parent
SVO = "https://svo2.cab.inta-csic.es/theory/newov2/"


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    grid = DATA / "koester_grid.html"
    if not grid.exists():
        response = requests.post(SVO + "index.php", data={
            "models": ",koester2", "params[koester2][teff][min]": 6000,
            "params[koester2][teff][max]": 80000,
            "params[koester2][logg][min]": 7, "params[koester2][logg][max]": 9.5,
            "nres": "all", "boton": "Search"}, timeout=120)
        response.raise_for_status()
        grid.write_text(response.text)
    rows = []
    soup = BeautifulSoup(grid.read_text(), "html.parser")
    for link in soup.find_all("a", href=lambda x: x and "format=votable" in x):
        cells = link.find_parent("tr").find_all("td")
        rows.append(dict(teff_K=float(cells[1].get_text()),
                         logg=float(cells[2].get_text()), url=SVO + link["href"]))
    nodes = Table(rows=rows)
    nodes.sort(["teff_K", "logg"])
    nodes.write(DATA / "koester_nodes.ecsv", overwrite=True)
    teff, logg = np.unique(nodes["teff_K"]), np.unique(nodes["logg"])
    combinations = set(zip(nodes["teff_K"], nodes["logg"]))
    missing = [(float(t), float(g)) for t in teff for g in logg if (t, g) not in combinations]

    path = DATA / "koester_10000_8.vot"
    if not path.exists():
        url = nodes[(nodes["teff_K"] == 10000) & (nodes["logg"] == 8)]["url"][0]
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    spectrum = parse_single_table(BytesIO(path.read_bytes())).to_table()
    wave_A, flux_A = np.asarray(spectrum["WAVELENGTH"]), np.asarray(spectrum["FLUX"])
    bolometric_ratio = np.trapezoid(flux_A, wave_A) / (5.670374419e-5 * 10000**4)

    path = DATA / "seq_060_thick.txt"
    if not path.exists():
        response = requests.get("https://www.astro.umontreal.ca/~bergeron/CoolingModels/"
                                "CoolingModels/seq_060_thick.txt", timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    cooling_rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 6 and fields[0].isdigit():
            cooling_rows.append(list(map(float, fields)))
    cooling = np.array(cooling_rows)
    cooling = cooling[np.argsort(cooling[:, 1])]
    radius_cm = np.interp(10000, cooling[:, 1], cooling[:, 3])
    gravity = np.interp(10000, cooling[:, 1], cooling[:, 2])
    physical_gravity = np.log10(6.6743e-8 * .6 * 1.988409870698051e33 / radius_cm**2)
    result = dict(grid_nodes=len(nodes), temperature_nodes=len(teff), gravity_nodes=len(logg),
                  missing_nodes=missing, teff_K=10000, logg=8.0,
                  wavelength_nm=[float(wave_A.min() / 10), float(wave_A.max() / 10)],
                  finite_range_flux_over_sigma_teff4=float(bolometric_ratio),
                  cooling_mass_msun=.6, radius_rsun=float(radius_cm / 6.957e10),
                  cooling_age_yr=float(np.interp(10000, cooling[:, 1], cooling[:, 4])),
                  cooling_logg=float(gravity), logg_from_mass_radius=float(physical_gravity))
    (OUT / "model_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
