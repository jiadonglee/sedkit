"""Build models/giant/giant_blue.npz: the giant template extended to the six blue_xp channels (332-382 nm).

The run directory (sdb_giant_20261011 on the Garching nodes) provides work/giant_blue_grid.npz, made by
reports/sdb_giant_20261011/garching/g2_build.py from the giant-v1 training set: 72 channels ordered as spbin's
XP61 + J/H/Ks/W1/W2 followed by the blue channels, the joint fractional residual covariance (6 eigen-columns
plus a diagonal) and the per-node counts of clipped stars. A node is supported where the giant-v1 criteria hold
(positive template and fractional scatter below MAX_SIGMA on XP61, finite Ks) and the blue template is positive.
The grid keeps the giant-v1 axes so that the PARSEC prior applies node by node.

    python scripts/build_giant_blue.py RUN_DIR
"""

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "src/spbin/models/giant"
MAX_SIGMA = 0.3


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    grid = dict(np.load(args.run_dir / "work" / "giant_blue_grid.npz"))
    with np.load(OUT / "giant_grid.npz") as v1:
        for name in ("teff_ax", "logg_ax", "mh_ax"):
            if not np.array_equal(v1[name], grid[name]):
                raise ValueError(f"{name} differs from the giant-v1 grid")
    template, basis, diag = (grid[k].astype(float) for k in ("template", "basis", "diag"))
    sigma = np.sqrt(diag + np.sum(basis**2, axis=-1))
    xp, blue = slice(0, 61), slice(66, 72)
    good = (np.isfinite(template[..., xp]).all(-1) & np.isfinite(template[..., 63])
            & (template[..., xp] > 0).all(-1) & (sigma[..., xp] < MAX_SIGMA).all(-1)
            & np.isfinite(template[..., blue]).all(-1) & (template[..., blue] > 0).all(-1))
    template[~good], basis[~good], diag[~good] = np.nan, np.nan, np.nan
    np.savez_compressed(
        OUT / "giant_blue.npz", teff_ax=grid["teff_ax"], logg_ax=grid["logg_ax"], mh_ax=grid["mh_ax"],
        wavelength_um=grid["wavelength_um"], template=template.astype(np.float32),
        basis=basis.astype(np.float16), diag=diag.astype(np.float32), n_eff=grid["n_eff"].astype(np.float32))
    summary = json.loads((OUT / "summary.json").read_text())
    n_star = np.nansum(grid["n_eff"][good])
    summary["blue"] = dict(
        file="giant_blue.npz", source=args.run_dir.name,
        channels="XP61 + J/H/Ks/W1/W2 + XP 332-382 nm in 10 nm steps (72); GaiaXPy, truncation off, as blue_xp",
        dereddening="E = %.2f SFD E(B-V); ZGR23 above 392 nm, the G23 curve of SubdwarfModel.blue_curve below"
                    % float(grid["c_sfd"]),
        clipping="4 robust sigma in XP61 rms residual, and one-sided in the mean blue residual",
        covariance="fractional; 6 eigen-columns of the joint 72-channel covariance plus a diagonal per node",
        support=f"giant-v1 criteria and a positive blue template; {int(good.sum())} nodes",
        training_stars=int(grid["n_train"]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("supported nodes", int(good.sum()), round((OUT / "giant_blue.npz").stat().st_size / 1e6, 1), "MB",
          "median effective stars", round(float(np.median(grid["n_eff"][good])), 1), n_star)


if __name__ == "__main__":
    main()
