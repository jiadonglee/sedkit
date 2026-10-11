"""Build models/hot/hot_uv.npz: the hot table's spectra below 392 nm, as colours against its 392 nm channel.

The hot table (models/hot/hot_table.npz) predicts XP61 + J/H/Ks only. For a hot main-sequence star beside a
giant the six XP channels at 332-382 nm carry the Balmer jump (364.6 nm, between the 362 and 372 nm
channels), which a blackbody joined to the 392 nm flux misses by tens of per cent for A and late-B stars.
This table supplies those channels from the grids behind the hot table: Castelli & Kurucz (2004, solar)
below 15 kK and TLUSTY BSTAR2006 (OSTAR2002 at 30 kK) from 15 kK, with the CK04 side carried onto TLUSTY by
the ln-ratio at 15 kK tapered to zero at 11 kK, as in the hot table. Each spectrum passes through the same
XP forward operator as the subdwarf table (XPOperator on 332-992 nm), and the 2 nm binned spectrum over
130-1100 nm is kept for reddened GALEX passband integrals.

Stored per (Teff, log g) node, relative to the synthetic 392 nm XP channel of the same spectrum:
  ln_blue   (n_teff, n_logg, 6)    ln(F_blue / F_392) at 332-382 nm
  ln_coarse (n_teff, n_logg, 486)  ln(F_coarse / F_392) on the subdwarf table's coarse grid
A component's 392 nm channel (hot table, with its operator correction) times exp(ln_blue) gives its blue
channels, so the table carries only the synthetic colour below 392 nm.

The synthetic Balmer jump is too deep in XP: at 7-11 kK the observed 342-372 nm channels of the hot-table
anchors lie 0.3-0.45 in ln above it, and 0-0.16 at 342-382 nm above 15 kK. `calibrate` fits that offset per blue
channel as the median residual in Teff bins of the training anchors (BLUE_BINS_K), interpolated linearly in
log Teff between the bins' median Teff and held flat beyond, and reports the residuals of the held-out anchors.
ANCHORS.csv holds the anchors' hot-table MAP labels (teff, logg, e = ZGR23 E), the holdout flag and the
observed colours obs_<nm> = ln(F / F_392) at 332-382 nm with errors err_<nm>, from GaiaXPy on the XP
continuous spectra with truncation=False, as sedkit.subdwarf.blue_xp samples them.

    python scripts/build_hot_uv.py table --operator .../phase0_forward_xp_20261003 \\
        --ck04 .../ck04_synphot_v2.tar --tlusty .../tlusty_solar_binned.npz
    python scripts/build_hot_uv.py calibrate --anchors ANCHORS.csv
"""

import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.interpolate import LinearNDInterpolator

from build_subdwarf_model import COARSE, N_BLUE, SCALE_10PC, XPOperator, _mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src/sedkit/models/hot"
TEFF_AX = np.arange(7000.0, 30001.0, 250.0)
LOGG_AX = np.arange(3.0, 4.7501, 0.25)
CK04_TEFF = np.r_[np.arange(7000.0, 13001.0, 250.0), 14000.0, 15000.0]
CK04_LOGG = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
SEAM_T, SEAM_COLD = 15000.0, 11000.0
SEAM_LOGG = np.array([3.5, 4.0, 4.5])
I392 = N_BLUE                      # index of 392 nm in XP_SAMPLING
BLUE_BINS_K = (7000.0, 7500.0, 9000.0, 11500.0, 15000.0, 30000.0)


def reduce(op, wl, flux):
    """ln(blue / F_392) and ln(coarse / F_392) of one surface spectrum (erg s-1 cm-2 nm-1)."""
    f10 = flux * SCALE_10PC
    xp = op(wl, f10)
    coarse = np.array([_mean(wl, f10, c - 1, c + 1) for c in COARSE]) * 1e15
    return np.log(xp[:N_BLUE] / xp[I392]), np.log(coarse / xp[I392])


def ck04(path):
    with tarfile.open(path) as tar:
        for t in CK04_TEFF:
            name = f"grp/redcat/trds/grid/ck04models/ckp00/ckp00_{int(t)}.fits"
            h = fits.open(io.BytesIO(tar.extractfile(name).read()))[1].data
            lam = np.asarray(h["WAVELENGTH"], float) / 10.0
            for g in CK04_LOGG:
                f = np.asarray(h[f"g{int(round(g * 10)):02d}"], float) * 10.0
                if np.nanmax(f) > 0:
                    yield t, g, lam, f


def taper(teff):
    x = np.clip((teff - SEAM_COLD) / (SEAM_T - SEAM_COLD), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))


def table(args):
    op = XPOperator(args.operator)

    grid = np.load(args.tlusty)
    use = (((grid["grid"] == "BS") & (grid["teff"] <= 29000)) | ((grid["grid"] == "OS") & (grid["teff"] == 30000))
           ) & (grid["logg"] >= 3.0)
    tl = {(float(t), float(g)): np.r_[reduce(op, grid["wave_nm"], f)]
          for t, g, f in zip(grid["teff"][use], grid["logg"][use], grid["flam"][use])}
    ck = {(t, g): np.r_[reduce(op, lam, f)] for t, g, lam, f in ck04(args.ck04)}
    print(len(tl), "TLUSTY and", len(ck), "CK04 models")

    ratio = np.array([tl[(SEAM_T, g)] - ck[(SEAM_T, g)] for g in SEAM_LOGG])
    seam_blue = np.abs(ratio[:, :N_BLUE]).max()
    print(f"seam |ln TLUSTY/CK04| at 15 kK on the blue channels: max {seam_blue:.3f}")
    points, values = [], []
    for (t, g), v in tl.items():
        points.append((t, g))
        values.append(v)
    for (t, g), v in ck.items():
        if t >= SEAM_T:
            continue
        d = np.array([np.interp(g, SEAM_LOGG, r) for r in ratio.T])
        points.append((t, g))
        values.append(v + taper(t) * d)
    values = np.array(values)
    tt, gg = np.meshgrid(TEFF_AX, LOGG_AX, indexing="ij")
    nodes = np.c_[tt.ravel(), gg.ravel()]
    cube = LinearNDInterpolator(np.array(points), values)(nodes).reshape(len(TEFF_AX), len(LOGG_AX), -1)
    # rows below the convex hull in log g take the nearest supported log g
    for i in range(len(TEFF_AX)):
        ok = np.isfinite(cube[i, :, 0])
        if not ok.any():
            raise SystemExit(f"no models at {TEFF_AX[i]} K")
        j0 = np.flatnonzero(ok)[0]
        cube[i, :j0] = cube[i, j0]
    if not np.isfinite(cube).all():
        raise SystemExit("NaN left in the table")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "hot_uv.npz", teff_ax=TEFF_AX, logg_ax=LOGG_AX,
                        ln_blue=cube[..., :N_BLUE].astype(np.float32),
                        ln_coarse=cube[..., N_BLUE:].astype(np.float32))
    summary = json.loads((OUT / "summary.json").read_text())
    summary["uv"] = dict(
        file="hot_uv.npz",
        grids="CK04 solar below 15 kK (seam-matched to TLUSTY, tapered to 11 kK), TLUSTY BSTAR2006 15-29 kK, "
              "OSTAR2002 30 kK",
        operator="J-CAPS phase0_forward_xp_20261003 transfer.json on 332-992 nm (XP channels 332-382 nm)",
        content="ln(F / F_392) of the six blue XP channels and of the 2 nm coarse spectrum over 130-1100 nm",
        seam_max_abs_ln_blue=float(seam_blue))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", OUT / "hot_uv.npz", cube.shape)


def _mad(x):
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def calibrate(args):
    import pandas as pd
    from sedkit.extinction import extinction_curve
    from sedkit.model import _bilinear
    from sedkit.subdwarf import SubdwarfModel

    with np.load(OUT / "hot_uv.npz") as archive:
        uv = {key: archive[key] for key in archive.files if key not in ("blue_node_teff", "blue_offset")}
    anchors = pd.read_csv(args.anchors)
    curve = SubdwarfModel(calibration=None).blue_curve - extinction_curve(np.array([0.392]))[0]
    logg = np.clip(anchors.logg.to_numpy(), uv["logg_ax"][0], uv["logg_ax"][-1])
    model = (_bilinear(uv["teff_ax"], uv["logg_ax"], uv["ln_blue"].astype(float), anchors.teff.to_numpy(), logg)
             - anchors.e.to_numpy()[:, None] * curve)
    wave = [f"{w:.0f}" for w in np.arange(332.0, 383.0, 10.0)]
    residual = anchors[[f"obs_{w}" for w in wave]].to_numpy() - model
    train = ~anchors.holdout.to_numpy(bool)
    nodes, offsets = [], []
    for low, high in zip(BLUE_BINS_K[:-1], BLUE_BINS_K[1:]):
        inside = train & (anchors.teff > low).to_numpy() & (anchors.teff <= high).to_numpy()
        nodes.append(float(np.median(anchors.teff[inside])))
        offsets.append(np.median(residual[inside], axis=0))
    nodes, offsets = np.array(nodes), np.array(offsets)
    correction = np.array([np.interp(np.log(anchors.teff), np.log(nodes), offsets[:, k]) for k in range(N_BLUE)]).T
    calibrated = residual - correction
    hot = (anchors.teff > 9000).to_numpy()
    stats = {}
    for name, rows in (("train", train), ("holdout", ~train), ("holdout_above_9kK", ~train & hot)):
        stats[name] = dict(n=int(rows.sum()),
                           median=np.median(calibrated[rows], axis=0).round(4).tolist(),
                           robust_sd=[round(_mad(calibrated[rows, k]), 4) for k in range(N_BLUE)],
                           median_error=np.median(anchors[[f"err_{w}" for w in wave]].to_numpy()[rows], axis=0
                                                  ).round(4).tolist())
    print(json.dumps(dict(nodes_k=nodes.round(0).tolist(), offsets=offsets.round(3).tolist(), **stats), indent=1))
    np.savez_compressed(OUT / "hot_uv.npz", **uv, blue_node_teff=nodes, blue_offset=offsets)
    summary = json.loads((OUT / "summary.json").read_text())
    summary["uv"]["calibration"] = dict(
        anchors="hot-table anchors (RV-constant, SB1 and 7-7.5 kK J-CAPS dwarfs) at their MAP labels",
        model="ln(F_obs / F_model) per blue channel: training-anchor medians in Teff bins, linear in ln Teff "
              "between the bins' median Teff, flat beyond",
        bins_k=list(BLUE_BINS_K), node_teff_k=nodes.round(1).tolist(), offset=offsets.round(4).tolist(),
        residuals=stats)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("table", "calibrate"))
    parser.add_argument("--operator", type=Path)
    parser.add_argument("--ck04", type=Path)
    parser.add_argument("--tlusty", type=Path)
    parser.add_argument("--anchors", type=Path)
    args = parser.parse_args()
    table(args) if args.stage == "table" else calibrate(args)


if __name__ == "__main__":
    main()
