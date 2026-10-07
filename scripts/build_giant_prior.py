"""Build models/giant/parsec_prior.npz, the PARSEC M_Ks prior of the giant route.

The run (giant_grid_20261006 on the Garching nodes, script p6_prior2.py, age-metallicity factor fitted by
p7_calib.py) provides work/parsec_prior2.npz on the giant template's Teff x log g x [M/H] grid:
  pen         -2 ln(p / p_mode) of M_Ks among PARSEC v1.2S giants (label 2-8, log g < 3.8) near each node,
              on mks_grid (0.05 mag from -10 to +4)
  flag        nodes with too few PARSEC giants in the kernel; their curve is zero and carries no constraint
  bc          PARSEC bolometric correction BC_Ks at the node
  beta, m0    the age-metallicity factor (age / 1 Gyr)**(beta * max(0, m0 - [M/H]))

Nodes outside the template support are zeroed. The curve is stored as float16, the rest as float32, and
the build is recorded under "luminosity_prior" in summary.json.

    python scripts/build_giant_prior.py RUN_DIR/work/parsec_prior2.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "src/sedkit/models/giant"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prior", type=Path)
    args = parser.parse_args()
    prior = dict(np.load(args.prior))
    with np.load(OUT / "giant_grid.npz") as grid:
        axes = [grid[key] for key in ("teff_ax", "logg_ax", "mh_ax")]
        template = grid["template"]
    for axis, key in zip(axes, ("teff_ax", "logg_ax", "mh_ax")):
        if not np.allclose(axis, prior[key]):
            raise RuntimeError(f"prior {key} differs from the template grid")
    supported = np.isfinite(template[..., :61]).all(-1) & np.isfinite(template[..., 63])
    pen = np.where(supported[..., None] & ~prior["flag"][..., None], prior["pen"], 0.0)
    if not np.isfinite(pen).all() or not np.isfinite(prior["bc"][supported]).all():
        raise RuntimeError("non-finite prior values on supported nodes")
    np.savez_compressed(
        OUT / "parsec_prior.npz", mks_grid=prior["mks_grid"].astype(np.float32),
        pen=pen.astype(np.float16), flag=prior["flag"] | ~supported,
        bc=np.where(supported, prior["bc"], np.nan).astype(np.float32))

    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["luminosity_prior"] = {
        "source": "giant_grid_20261006 p6_prior2.py, factor fitted by p7_calib.py",
        "isochrones": "PARSEC v1.2S, 2MASS Ks; logAge from 7.5 in 0.05 dex at [M/H] -1.0 to +0.5 in 0.1 dex, "
                      "0.5 dex in logAge outside that range",
        "points": "label 2-8 (subgiant branch to TP-AGB), log g < 3.8",
        "weights": "IMF number x linear age width (constant star formation from 10**7.5 yr to 13 Gyr) x "
                   "[M/H] spacing x (age / 1 Gyr)**(beta * max(0, m0 - [M/H]))",
        "beta": float(prior["beta"]), "m0": float(prior["m0"]),
        "kernel": {"width": [200.0, 0.12, 0.15], "units": ["K", "dex", "dex"]},
        "density": "M_Ks histogram in 0.05 mag bins smoothed by 0.15 mag, mixed with a uniform floor over the "
                   "M_Ks range of the kernel's points padded by 1 mag so that the penalty there is at most 6; "
                   "beyond the range it rises as (distance / 0.5 mag)**2",
        "constrained_nodes": int((supported & ~prior["flag"]).sum()),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"pen {pen.shape}, constrained nodes {(supported & ~prior['flag']).sum()} of {supported.sum()}, "
          f"{(OUT / 'parsec_prior.npz').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
