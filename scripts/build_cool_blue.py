"""Build models/stellar/cool_blue.npz: empirical blue XP colours of F/G/K main-sequence stars for subdwarf.CoolUV.

The run directory (sdb_rgb_uv_20261011 on the Garching nodes) provides work/cool_blue.npz, made by
reports/sdb_rgb_uv_20261011/garching/x6_cool_blue.py: APOGEE DR17 dwarfs (log g 4.0-4.8, S/N > 70, STAR_BAD unset,
VSCATTER < 1 km/s) within 500 pc and at Edenhofer E < 0.05, their six blue XP channels (332-382 nm, the BP
coefficients through GaiaXPy's linear calibration, truncation off) against the 392 nm XP channel, dereddened with
SubdwarfModel.blue_curve and ZGR23. On Teff 4400-7500 K (100 K) x [M/H] -1-0.5 (0.25 dex) the table holds the
Gaussian-kernel weighted median ln(F / F_392) of a random training half (kernel 100 K, 0.15 dex), its weighted
robust scatter and the effective number of stars; nodes with fewer than 15 effective stars are empty (NaN).

    python scripts/build_cool_blue.py RUN_DIR
"""

import argparse
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "src/spbin/models/stellar"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    grid = dict(np.load(args.run_dir / "work" / "cool_blue.npz"))
    shape = (len(grid["teff_ax"]), len(grid["mh_ax"]), 6)
    if grid["ln_blue"].shape != shape or grid["scatter"].shape != shape:
        raise ValueError("cool_blue.npz does not match its axes")
    empty = ~np.isfinite(grid["ln_blue"]).all(-1)
    np.savez(OUT / "cool_blue.npz", teff_ax=grid["teff_ax"], mh_ax=grid["mh_ax"],
             ln_blue=grid["ln_blue"].astype(np.float32), scatter=grid["scatter"].astype(np.float32),
             n_eff=grid["n_eff"].astype(np.float32))
    print(f"{(~empty).sum()} of {empty.size} nodes supported -> {OUT / 'cool_blue.npz'}")


if __name__ == "__main__":
    main()
