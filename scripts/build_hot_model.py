"""Build models/hot/ from a J-CAPS hot-emulator run directory.

The run (J-CAPS experiments/hot_emulator_v3_20261004) provides:
  out/emulator_table.npz   ln F_lambda at 10 pc for R = 1 Rsun on Teff x log g
                           (61 XP in W m^-2 nm^-1, as the Gaia XP share stores them,
                           and J/H/Ks in 1e-18 W m^-2 nm^-1), the per-channel
                           correction delta = a + W b + s(Teff) c, the cool-edge
                           range of s and the Balmer index W
  out/model_error.npz      fractional covariance basis (64, k) and diagonal (64,)
  out/metrics_stage3.json, out/model_error_summary.json, out/operator_check_stars.csv

The table is cut to the nodes spanning the supported Teff range, XP is converted to the sedkit
unit 1e-18 W m^-2 nm^-1, and values are stored as float32. The error
term is expanded to the 168 model channels (NaN beyond J/H/Ks).

    python scripts/build_hot_model.py RUN_DIR
"""

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "src/sedkit/models/hot"
SUPPORT = {"teff": [7000.0, 30000.0], "logg": [3.0, 4.75], "feh": [-0.3, 0.3]}
TERM = "hot_v3"
VERSION = "hot-v3"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    out = args.run_dir / "out"
    emu = np.load(out / "emulator_table.npz")
    metrics = json.loads((out / "metrics_stage3.json").read_text())
    error_summary = json.loads((out / "model_error_summary.json").read_text())
    if str(emu["model"]) != "channel_line_cool" or metrics["model"] != "channel_line_cool":
        raise RuntimeError("expected the channel_line_cool correction")
    teff = emu["teff_ax"]
    # nodes spanning the support: the last node at or below the lower edge onwards
    low = teff[teff <= SUPPORT["teff"][0]].max()
    keep = (teff >= low) & (teff <= SUPPORT["teff"][1])
    if teff[keep][-1] != SUPPORT["teff"][1]:
        raise RuntimeError("table nodes must include the upper support edge")
    ln_flux = emu["ln_flux"][keep].copy()
    ln_flux[..., :61] += np.log(1e18)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "hot_table.npz", teff_ax=teff[keep], logg_ax=emu["logg_ax"],
        ln_flux=ln_flux.astype(np.float32),
        balmer_w=emu["balmer_w"][keep].astype(np.float32),
        delta_a=emu["delta"][0], delta_b=emu["delta"][1], delta_c=emu["delta"][2],
        cool_edge_k=emu["cool_edge_k"])

    error = np.load(out / "model_error.npz")
    basis = np.full((168, error["basis"].shape[1]), np.nan)
    diag = np.full(168, np.nan)
    basis[:64], diag[:64] = error["basis"], error["diag"]
    np.savez_compressed(OUT / "model_error.npz", **{f"{TERM}/basis": basis, f"{TERM}/diag": diag})

    summary = dict(
        version=VERSION, source=f"J-CAPS {args.run_dir.name}", error_term=TERM,
        support=SUPPORT, channels="XP61 + J/H/Ks (64); W1/W2 and SPHEREx are not predicted",
        units="ln F_lambda at 10 pc for R = 1 Rsun, 1e-18 W m^-2 nm^-1",
        grids="CK04 (Castelli & Kurucz 2003) below 15 kK, TLUSTY BSTAR2006 15-29 kK, solar",
        extinction="ZGR23 E (the sedkit curve) in the calibration",
        anchors="hot anchors (RV-constant and SB1, screened) and J-CAPS 1 kpc IRFM dwarfs at 7000-7500 K",
        holdout_median_rms=metrics["acceptance"],
        model_error=error_summary)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
