"""Download PARSEC isochrones from CMD 3.8 and build models/parsec_mh_tracks.npz.

PARSEC v1.2S, YBC bolometric corrections, Gaia DR2 (Evans et al. 2018) +
Tycho2 + 2MASS Vega magnitudes, no extinction; [M/H] -1.0 to +0.5 in 0.1
dex and log age 6.60 to 10.00 in 0.05 dex, requested as two age ranges per
[M/H] (6.60-8.45 and 8.50-10.00). Pre-main-sequence and main
sequence rows (labels 0 and 1) are kept. Each table holds Mass, logTe, logL,
Ksmag and Gmag with strictly increasing Mass; of repeated rounded masses near
the turn-off, the first row is kept. Where the isochrone ends in the
overall-contraction hook (Teff falls, then rises within the last 0.1 Msun),
it is cut at the Teff minimum: interpolating two isochrones at fixed mass
across a hook of a few hundred K biases the labels just below the turn-off.
Young isochrones label massive stars crossing the Hertzsprung gap (log g
1.4-1.8 at 10 Myr) as main sequence; above the Teff maximum each table also
ends before the first star with log g < 3.0.

    python scripts/build_parsec_tracks.py RAW_DIR            # download + build
    python scripts/build_parsec_tracks.py RAW_DIR --check OLD.npz

--check compares the tables of OLD.npz at the shared age nodes, more than
0.05 Msun below the lower terminal mass: at most 0.001 dex in logTe,
0.004 dex in logL and 0.01 mag in Ks and G.
"""

import argparse
import re
from pathlib import Path

import numpy as np

CMD = "https://stev.oapd.inaf.it"
PHOTSYS = "YBC_tab_mag_odfnew/tab_mag_gaiaDR2_tycho2_2mass.dat"
METALS = np.arange(-10, 6) / 10
LOG_AGE_RANGES = ((6.6, 8.45, 0.05), (8.5, 10.0, 0.05))
COLUMNS = ("Mass", "logTe", "logL", "Ksmag", "Gmag")
HOOK_WINDOW, HOOK_DEPTH = 0.1, 0.002  # Msun below the terminal mass; dex in logTe
LOGG_MIN = 3.0  # above the Teff maximum
OUT = Path(__file__).resolve().parents[1] / "src/sedkit/models/parsec_mh_tracks.npz"


def raw_path(directory, mh, ages):
    low, high, step = ages
    return Path(directory) / f"parsec_v1.2S_YBC_gaiaDR2_2mass_MH{mh:+.1f}_logage{low:.2f}-{high:.2f}_d{step:.2f}.dat"


def download(directory, mh, ages):
    import requests
    low, high, step = ages
    form = dict(
        cmd_version="3.8", track_parsec="parsec_CAF09_v1.2S", track_omegai="0.00",
        track_colibri="parsec_CAF09_v1.2S_S_LMC_08_web", track_postagb="no", n_inTPC="10",
        eta_reimers="0.2", kind_interp="1", kind_postagb="-1", photsys_file=PHOTSYS,
        photsys_version="YBC", dust_sourceM="dpmod60alox40", dust_sourceC="AMCSIC15",
        kind_mag="2", kind_dust="0", extinction_av="0.0", extinction_coeff="constant",
        extinction_curve="cardelli", kind_LPV="4", imf_file="tab_imf/imf_kroupa_orig.dat",
        # CMD excludes the upper age limit.
        isoc_isagelog="1", isoc_lagelow=f"{low}", isoc_lageupp=f"{high + step / 5}",
        isoc_dlage=f"{step}", isoc_ismetlog="1", isoc_metlow=f"{mh}", isoc_metupp=f"{mh}",
        isoc_dmet="0", output_kind="0", output_evstage="1", submit_form="Submit")
    # The CMD server certificate chain does not verify with certifi.
    page = requests.post(CMD + "/cgi-bin/cmd_3.8", data=form, verify=False, timeout=900)
    match = re.search(r"tmp/output\d+\.dat", page.text)
    if match is None:
        raise RuntimeError(f"CMD returned no isochrone file for [M/H]={mh}")
    text = requests.get(f"{CMD}/{match.group()}", verify=False, timeout=900).text
    raw_path(directory, mh, ages).write_text(text)


def cut_hook(table):
    near = np.flatnonzero(table[:, 0] > table[-1, 0] - HOOK_WINDOW)
    low = near[np.argmin(table[near, 1])]
    fall = table[near[0]:low + 1, 1].max() - table[low, 1]
    rise = table[-1, 1] - table[low, 1]
    return table[:low + 1] if fall > HOOK_DEPTH and rise > HOOK_DEPTH else table


def cut_low_gravity(table, logg):
    """End the table before the first star above the Teff maximum with log g < LOGG_MIN."""
    peak = int(np.argmax(table[:, 1]))
    low = np.flatnonzero(logg[peak:] < LOGG_MIN)
    return table[:peak + low[0]] if len(low) else table


def read(path, hook=True):
    lines = Path(path).read_text().splitlines()
    header = next(line for line in lines if line.startswith("# Zini"))[1:].split()
    rows = np.array([line.split() for line in lines if line and not line.startswith("#")], float)
    column = {name: index for index, name in enumerate(header)}
    rows = rows[np.isin(rows[:, column["label"]], (0, 1))]
    tables = {}
    for log_age in np.unique(rows[:, column["logAge"]]):
        table = rows[rows[:, column["logAge"]] == log_age][:, [column[c] for c in COLUMNS + ("logg",)]]
        table = table[np.r_[True, table[1:, 0] > np.maximum.accumulate(table[:-1, 0])]]
        table = cut_low_gravity(table[:, :-1], table[:, -1])
        tables[round(float(log_age), 2)] = cut_hook(table) if hook else table
    return tables


def build(directory):
    ages = np.concatenate([np.round(np.arange(low, high + step / 2, step), 2)
                           for low, high, step in LOG_AGE_RANGES])
    result = {}
    for mh in METALS:
        tables = {}
        for span in LOG_AGE_RANGES:
            tables.update(read(raw_path(directory, mh, span)))
        if sorted(tables) != list(ages):
            raise RuntimeError(f"[M/H]={mh}: ages {sorted(tables)} differ from the grid")
        for age, table in tables.items():
            label = f"{age:.1f}" if round(age, 1) == age else f"{age:.2f}"
            result[f"{mh:+.1f}/{label}"] = table
    return result


def check(tables, old_path):
    with np.load(old_path) as old:
        shared = [key for key in old.files if key in tables]
        for key in shared:
            before, after = old[key], tables[key]
            mass = np.linspace(before[0, 0], min(before[-1, 0], after[-1, 0]) - 0.05, 400)
            change = [np.abs(np.interp(mass, before[:, 0], before[:, j])
                             - np.interp(mass, after[:, 0], after[:, j])).max() for j in range(1, 5)]
            if change[0] > 0.001 or change[1] > 0.004 or max(change[2:]) > 0.01:
                raise RuntimeError(f"{key}: main-sequence change {np.round(change, 4)}")
    print(f"{len(shared)} tables agree with {old_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("--check", type=Path, help="existing table file to compare at shared nodes")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    for mh in METALS:
        for span in LOG_AGE_RANGES:
            if not raw_path(args.raw_dir, mh, span).exists():
                print(f"downloading [M/H]={mh:+.1f}, log age {span[0]:.2f}-{span[1]:.2f}", flush=True)
                download(args.raw_dir, mh, span)
    tables = build(args.raw_dir)
    if args.check:
        check(tables, args.check)
    np.savez_compressed(args.out, **tables)
    print(f"{len(tables)} tables -> {args.out}")


if __name__ == "__main__":
    main()
