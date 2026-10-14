"""Build models/giant/parsec_prior.npz, the PARSEC M_Ks prior of the giant route.

The run (giant_grid_20261006 on the Garching nodes, script p6_prior2.py, age-metallicity factor fitted by
p7_calib.py) provides work/parsec_prior2.npz on the giant template's Teff x log g x [M/H] grid:
  pen         -2 ln(p / p_mode) of M_Ks among PARSEC v1.2S giants (label 2-8, log g < 3.8) near each node,
              on mks_grid (0.05 mag from -10 to +4)
  flag        nodes with too few PARSEC giants in the kernel; their curve is zero and carries no constraint
  bc          PARSEC bolometric correction BC_Ks at the node
  beta, m0    the age-metallicity factor (age / 1 Gyr)**(beta * max(0, m0 - [M/H]))

and work/parsec_mass_prior.npz (script p11_mass_prior.py, same isochrones and weights) on the template's Teff and
[M/H] axes by M_Ks (0.1 mag from -10 to +4):
  pen         -2 ln(p / p_mode) of log10 M among PARSEC giants at that Teff, M_Ks and [M/H], on logm_ax
  flag        cells with too few PARSEC giants in the kernel

Nodes outside the template support are zeroed. The curves are stored as float16, the rest as float32, and
the build is recorded under "luminosity_prior" in summary.json.

    python scripts/build_giant_prior.py RUN_DIR/work/parsec_prior2.npz RUN_DIR/work/parsec_mass_prior.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "src/spbin/models/giant"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prior", type=Path)
    parser.add_argument("mass_prior", type=Path)
    args = parser.parse_args()
    prior = dict(np.load(args.prior))
    mass = dict(np.load(args.mass_prior))
    with np.load(OUT / "giant_grid.npz") as grid:
        axes = [grid[key] for key in ("teff_ax", "logg_ax", "mh_ax")]
        template = grid["template"]
    for axis, key in zip(axes, ("teff_ax", "logg_ax", "mh_ax")):
        if not np.allclose(axis, prior[key]):
            raise RuntimeError(f"prior {key} differs from the template grid")
    for axis, key in ((axes[0], "teff_ax"), (axes[2], "mh_ax")):
        if not np.allclose(axis, mass[key]):
            raise RuntimeError(f"mass prior {key} differs from the template grid")
    if (mass["beta"], mass["m0"]) != (prior["beta"], prior["m0"]):
        raise RuntimeError("the mass prior uses another age-metallicity factor")
    mass_pen = np.where(mass["flag"][..., None], 0.0, mass["pen"])
    if not np.isfinite(mass_pen).all():
        raise RuntimeError("non-finite mass prior values")
    supported = np.isfinite(template[..., :61]).all(-1) & np.isfinite(template[..., 63])
    pen = np.where(supported[..., None] & ~prior["flag"][..., None], prior["pen"], 0.0)
    if not np.isfinite(pen).all() or not np.isfinite(prior["bc"][supported]).all():
        raise RuntimeError("non-finite prior values on supported nodes")
    np.savez_compressed(
        OUT / "parsec_prior.npz", mks_grid=prior["mks_grid"].astype(np.float32),
        pen=pen.astype(np.float16), flag=prior["flag"] | ~supported,
        bc=np.where(supported, prior["bc"], np.nan).astype(np.float32),
        mass_mks=mass["mks_ax"].astype(np.float32), mass_logm=mass["logm_ax"].astype(np.float32),
        mass_pen=mass_pen.astype(np.float16), mass_flag=mass["flag"])

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
        "mass_term": {
            "source": "giant_grid_20261006 p11_mass_prior.py",
            "cells": "template Teff and [M/H] axes by M_Ks from -10 to +4 in 0.1 mag",
            "kernel": {"width": [float(x) for x in mass["kernel"]], "units": ["K", "mag", "dex"]},
            "density": f"current-mass log10 M histogram in 0.03 dex bins smoothed by 0.06 dex, mixed with a uniform "
                       f"floor over the log M range where the smoothed density exceeds 1e-3 of its peak, padded by "
                       f"{float(mass['pad']):g} dex, so that the penalty there is at most {float(mass['cap']):g}; beyond "
                       f"the range it rises as (distance / {float(mass['tail']):g} dex)**2",
            "filled": "cells with a kernel weight or effective number below 3 take the curve of the nearest "
                      "constrained cell within 3 kernel widths",
            "filled_cells": int(mass["filled"].sum()),
            "constrained_cells": int((~mass["flag"]).sum()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"mass_pen {mass_pen.shape}, constrained cells {(~mass['flag']).sum()}")
    print(f"pen {pen.shape}, constrained nodes {(supported & ~prior['flag']).sum()} of {supported.sum()}, "
          f"{(OUT / 'parsec_prior.npz').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
