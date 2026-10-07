"""Build models/giant/ from an empirical giant-template run directory.

The run (giant_grid_20261006 on the Garching nodes) provides:
  work/giant_grid.npz   local-linear templates of APOGEE DR17 giants on a Teff x log g x [M/H] grid:
                        dereddened XP (342-992 nm), J/H/Ks and W1/W2 fluxes normalised to their
                        0.55-0.95 micron mean, the fractional residual covariance (basis, diag) and
                        the kernel's effective sample per node
  work/calib.csv        per-star E against SFD E(B-V), which sets the dereddening factor
  scripts/              the build (s1_select, s2_xp, s3_phot, s4_build)

The template is cut to sedkit's XP61 + J/H/Ks/W1/W2 channels. A node stays supported only where its
template is positive and its fractional scatter is below MAX_SIGMA on every XP channel; the cool,
low-gravity corner where blue XP flux reaches zero falls outside. The template and diagonal are stored
as float32 and the covariance columns as float16, cropped to the supported box.

    python scripts/build_giant_model.py RUN_DIR
"""

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "src/sedkit/models/giant"
MAX_SIGMA = 0.3
VERSION = "giant-v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    grid = dict(np.load(args.run_dir / "work" / "giant_grid.npz"))
    first = int(grid["n_blue"])
    channels = slice(first, first + 66)
    template = grid["template"][..., channels].astype(float)
    basis = grid["basis"][..., channels, :].astype(float)
    diag = grid["diag"][..., channels].astype(float)
    sigma = np.sqrt(diag + np.sum(basis**2, axis=-1))
    xp = slice(0, 61)
    good = (np.isfinite(template[..., xp]).all(-1) & np.isfinite(template[..., 63])
            & (template[..., xp] > 0).all(-1) & (sigma[..., xp] < MAX_SIGMA).all(-1))
    template[~good], basis[~good], diag[~good] = np.nan, np.nan, np.nan
    box = [np.flatnonzero(good.any(axis=tuple(a for a in range(3) if a != k))) for k in range(3)]
    crop = tuple(slice(b[0], b[-1] + 1) for b in box)
    axes = [grid[name][c] for name, c in zip(("teff_ax", "logg_ax", "mh_ax"), crop)]

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "giant_grid.npz", teff_ax=axes[0], logg_ax=axes[1], mh_ax=axes[2],
        wavelength_um=grid["wavelength_um"][channels],
        template=template[crop].astype(np.float32), basis=basis[crop].astype(np.float16),
        diag=diag[crop].astype(np.float32), n_eff=grid["n_eff"][crop].astype(np.float32))

    summary = dict(
        version=VERSION, source=f"{args.run_dir.name}",
        training=("APOGEE DR17 allStarLite giants: ASPCAP STAR_BAD unset, S/N > 70, VSCATTER <= 1 km/s, "
                  "SFD E(B-V) < 0.1, Gaia G 7.5-14, RUWE < 1.4, non_single_star = 0, Ks valid; "
                  f"{int(grid['n_train'])} stars"),
        labels="APOGEE DR17 ASPCAP Teff, log g, [M/H]",
        dereddening=f"E = {float(grid['c_sfd']):.2f} SFD E(B-V) on the ZGR23 curve",
        normalisation="mean flux over 0.55-0.95 micron after dereddening",
        kernel=dict(width=grid["kernel_width"].tolist(), units=["K", "dex", "dex"],
                    widened_up_to=3, min_effective_stars=25),
        covariance="fractional; 6 eigen-columns plus a diagonal per node, measurement variance removed",
        support=f"positive template and fractional scatter < {MAX_SIGMA} on XP61; {int(good.sum())} nodes",
        channels="XP61 + J/H/Ks/W1/W2 (66)")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("supported nodes", int(good.sum()), "axes", [len(a) for a in axes])
    print("wrote", sorted(p.name for p in OUT.iterdir()),
          round((OUT / "giant_grid.npz").stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
