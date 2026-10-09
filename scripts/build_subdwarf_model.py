"""Build models/subdwarf/ from Tuebingen TMAP NLTE spectra served by TheoSSA (GAVO).

Three helium tiers, each a table on Teff x log g for R = 1 Rsun at 10 pc:
  H    pure hydrogen (TheoSSA family "H", 1 A - 40 um), Teff 20-45 kK, log g 5.0-6.5
  mid  H+He+C, He mass fraction 0.0498, log(He/H) = -1.88 ("HHeC"), Teff 32-45 kK
  He   H+He+C, He mass fraction 0.7575, log(He/H) = -0.10 ("HHeC"), Teff 32-45 kK
HHeC spectra cover 300 nm - 5.5 um plus 115-178 nm (no NUV). Their log g values are irregular, so each
Teff row is interpolated linearly in log g onto the table nodes from the nearest models (0.05 dex or closer).
A few TheoSSA models sit 2-11 per cent off their neighbours in flux; the table stage replaces such isolated
nodes by their log g neighbours and lists them in summary.json.

Each spectrum passes through the J-CAPS phase-0 forward model of the Gaia XP external calibration
(transfer.json, the operator of the hot table) on 332-992 nm in 10 nm steps: the 61 sedkit XP channels and
six channels below 392 nm. J/H/Ks/W1/W2 are sampled at the sedkit wavelengths, as in the hot table;
SPHEREx channels average F_lambda between the midpoints of neighbouring channels. Each node also stores
the Balmer index W of the hot table (EW of H-gamma + H-beta over its table maximum), photon-weighted
G/BP/RP and GALEX FUV/NUV fluxes and a 2 nm binned spectrum over 130-1100 nm for reddened passband
integrals. The extinction curve below 392 nm is Gordon et al. (2023) R_V = 3.1, scaled to ZGR23 over
392-550 nm.

    python scripts/build_subdwarf_model.py select WORK
    python scripts/build_subdwarf_model.py fetch WORK --operator .../phase0_forward_xp_20261003 --hot-run .../hot_emulator_v5_20261004
    python scripts/build_subdwarf_model.py table WORK

`fetch` needs gaiaxpy, requests and dust_extinction; it resumes and keeps only the reduced spectra.
"""

import argparse
import gzip
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src/sedkit/models/subdwarf"
SSAP = "http://dc.g-vo.org/theossa/q/ssa/ssap.xml"
SVO_FPS = "http://svo2.cab.inta-csic.es/theory/fps/fps.php"
TEFF_SLICES = [(19500, 24999), (25000, 29999), (30000, 32999), (33000, 35999), (36000, 38999),
               (39000, 41999), (42000, 45500)]
LOGG_AX = np.round(np.arange(5.0, 6.5001, 0.1), 2)
TIERS = {"H": dict(family="H", w_he=0.0, teff=np.arange(20000.0, 45001.0, 1000.0)),
         "mid": dict(family="HHeC", w_he=0.0498, teff=np.arange(32000.0, 45001.0, 1000.0)),
         "He": dict(family="HHeC", w_he=0.7575, teff=np.arange(32000.0, 45001.0, 1000.0))}
LOGG_TOLERANCE = 0.05          # largest offset of a model from its table node
XP_SAMPLING = np.arange(332.0, 993.0, 10.0)   # six blue channels (332-382 nm) + the 61 sedkit XP channels
N_BLUE = 6
COARSE = np.arange(130.0, 1100.1, 2.0)        # bin centres, nm
FILTERS = {"G": "GAIA/GAIA3.G", "BP": "GAIA/GAIA3.Gbp", "RP": "GAIA/GAIA3.Grp",
           "FUV": "GALEX/GALEX.FUV", "NUV": "GALEX/GALEX.NUV"}
R_SUN_M, PC_M = 6.957e8, 3.0856775814913673e16
SCALE_10PC = (R_SUN_M / (10.0 * PC_M))**2
BALMER = {"Hgamma": ((422.0, 448.0), (418.0, 422.0), (448.0, 455.0)),
          "Hbeta": ((470.0, 503.0), (465.0, 470.0), (503.0, 510.0))}


def log_he_h(w_he, w_h):
    """Number ratio log(He/H) from mass fractions."""
    return float(np.log10((w_he / 4.0026) / (w_h / 1.00794))) if w_he > 0 else -np.inf


# ---------------------------------------------------------------- select

def select(work):
    """Query TheoSSA by Teff slice and write manifest.json: one optical/infrared file (and a FUV file for
    HHeC) per (tier, Teff, log g node)."""
    import requests
    from astropy.io.votable import parse_single_table

    rows = []
    for low, high in TEFF_SLICES:
        r = requests.get(SSAP, params=dict(REQUEST="queryData", MAXREC=400000, t_eff=f"{low}/{high}",
                                           log_g="4.85/6.65"), timeout=1200)
        r.raise_for_status()
        t = parse_single_table(io.BytesIO(r.content)).to_table()
        lo = np.round(np.asarray(t["ssa_specstart"], float) * 1e10)
        hi = np.round(np.asarray(t["ssa_specend"], float) * 1e10)
        for i in range(len(t)):
            ref = str(t["accref"][i])
            if not ref.endswith(".txt"):
                continue
            rows.append(dict(family=str(t["ssa_dstitle"][i]), lo=lo[i], hi=hi[i], accref=ref,
                             teff=float(t["t_eff"][i]), logg=round(float(t["log_g"][i]), 2),
                             w_h=float(t["w_H"][i]), w_he=round(float(t["w_He"][i]), 4),
                             cdate=str(t["ssa_cdate"][i])))
        print(f"slice {low}-{high}: {len(t)} rows")
    picks = []
    for tier, spec in TIERS.items():
        for teff in spec["teff"]:
            def candidates(lo_max, hi_min, hi_max=np.inf):
                return [r for r in rows if r["family"] == spec["family"] and r["teff"] == teff
                        and r["w_he"] == spec["w_he"] and r["lo"] <= lo_max and hi_min <= r["hi"] <= hi_max]
            optical = candidates(3001, 50000) if tier != "H" else candidates(5, 400000)
            fuv = candidates(1150, 1780, 1780) if tier != "H" else []
            used = set()
            for node in LOGG_AX:
                near = [r for r in optical if abs(r["logg"] - node) <= LOGG_TOLERANCE + 1e-9]
                if not near:
                    print(f"  {tier} {teff:.0f} K: no model within {LOGG_TOLERANCE} of log g {node}")
                    continue
                # nearest log g; among duplicates the latest computation
                near.sort(key=lambda r: r["cdate"], reverse=True)
                pick = min(near, key=lambda r: round(abs(r["logg"] - node), 3))
                if pick["logg"] in used:
                    continue
                used.add(pick["logg"])
                item = dict(tier=tier, teff=teff, logg=pick["logg"], w_h=pick["w_h"], w_he=pick["w_he"],
                            optical=pick["accref"])
                if tier != "H":
                    same = [r for r in fuv if r["logg"] == pick["logg"]]
                    item["fuv"] = max(same, key=lambda r: r["cdate"])["accref"] if same else None
                picks.append(item)
    work.mkdir(parents=True, exist_ok=True)
    (work / "manifest.json").write_text(json.dumps(picks, indent=1, default=float))
    print(f"{len(picks)} nodes; files: {sum(1 + bool(p.get('fuv')) for p in picks)}")


# ---------------------------------------------------------------- fetch

def _resolve(accref, session):
    """Tuebingen URL of the gzipped flux table behind a TheoSSA access reference."""
    meta = accref.replace("http://dc.g-vo.org/getproduct/theossa/",
                          "http://dc.zah.uni-heidelberg.de/theossa/q/gettxt/qp/theossa/").replace(".txt", ".meta")
    with session.get(meta, stream=True, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line.startswith("* Access.Reference="):
                return line.split("=", 1)[1].strip()
            if line and not line.startswith("*"):
                break
    raise RuntimeError(f"no Access.Reference for {accref}")


def _spectrum(url, session):
    """Wavelength (nm) and surface F_lambda (erg s-1 cm-2 nm-1); TMAP tabulates F_lambda / pi per cm."""
    r = session.get(url, timeout=600)
    r.raise_for_status()
    data = np.loadtxt(io.BytesIO(gzip.decompress(r.content)), comments="*", usecols=(0, 1))
    order = np.argsort(data[:, 0])
    return data[order, 0] / 10.0, np.pi * data[order, 1] * 1e-7


class Reducer:
    """Channels and passband fluxes of one surface spectrum, at 10 pc for R = 1 Rsun."""

    def __init__(self, operator_dir, hot_run):
        sys.path.insert(0, str(Path(operator_dir) / "scripts"))
        import validate_calspec as V
        from forward_xp import ContinuousBases
        from gaiaxpy.calibrator.external_instrument_model import ExternalInstrumentModel
        from gaiaxpy.config.paths import config_path
        from gaiaxpy.spectrum.sampled_basis_functions import SampledBasisFunctions
        from forward_xp import FILES
        from sedkit.model import StellarModel

        self.V = V
        bases = ContinuousBases()
        self.bands = {xp: V.Band(xp, bases) for xp in ("bp", "rp")}
        params = json.loads((Path(operator_dir) / "out/transfer.json").read_text())["params"]
        self.p = np.array([params[k] for k in V.PNAMES])
        lam = XP_SAMPLING
        self.weight = {"bp": np.clip(1 - (lam - 635) / 8, 0, 1), "rp": np.clip((lam - 635) / 8, 0, 1)}
        self.design = {}
        for xp in ("bp", "rp"):
            model = ExternalInstrumentModel.from_config_csv(*(f"{config_path}/{f}" for f in FILES[xp]))
            mask = np.where(self.weight[xp] > 0, 1.0, 0.0)
            self.design[xp] = SampledBasisFunctions.from_external_instrument_model(lam, mask, model).get_design_matrix()
        wave = StellarModel().wavelength_um * 1000
        self.nir = wave[61:66]
        sph = wave[66:]
        edges = np.r_[sph[0] - (sph[1] - sph[0]) / 2, (sph[1:] + sph[:-1]) / 2, sph[-1] + (sph[-1] - sph[-2]) / 2]
        self.spherex_edges = edges
        self.filters = _filters()
        self.ew_max = float(np.load(Path(hot_run) / "out/balmer_index.npz")["ew_max_nm"])

    def xp(self, wl, f10):
        flam = f10 / 10.0   # erg s-1 cm-2 A-1
        out = []
        for i, xp in enumerate(("bp", "rp")):
            band = self.bands[xp]
            c = band.coefficients(band.fg(wl, flam), self.p[2 + 3 * i:5 + 3 * i], self.V.KNOTS[xp])
            out.append(self.p[i] * self.weight[xp] * (c @ self.design[xp]))
        return (out[0] + out[1]) * 1e18

    def reduce(self, wl, flux):
        f10 = flux * SCALE_10PC                  # erg s-1 cm-2 nm-1 at 10 pc
        unit = 1e15                              # erg s-1 cm-2 nm-1 -> 1e-18 W m-2 nm-1
        xp = self.xp(wl, f10)
        nir = np.interp(self.nir, wl, f10) * unit
        spherex = np.array([_mean(wl, f10, a, b) for a, b in zip(self.spherex_edges[:-1], self.spherex_edges[1:])]) * unit
        channels = np.r_[xp[N_BLUE:], nir, spherex]
        coarse = np.array([_mean(wl, f10, c - 1, c + 1) for c in COARSE]) * unit
        passbands = {name: _photon_mean(wl, f10, *curve) * unit for name, curve in self.filters.items()}
        return dict(channels=channels, blue=xp[:N_BLUE], coarse=coarse, balmer_w=_ew(wl, flux) / self.ew_max,
                    **{f"pb_{k}": v for k, v in passbands.items()})


def _mean(wl, f, a, b):
    m = (wl >= a) & (wl <= b)
    if m.sum() < 2:
        return float(np.interp((a + b) / 2, wl, f)) if wl[0] <= a and wl[-1] >= b else np.nan
    x = np.r_[a, wl[m], b]
    y = np.interp(x, wl, f)
    return float(np.trapezoid(y, x) / (b - a))


def _photon_mean(wl, f, lam, trans):
    """Photon-weighted mean F_lambda over a passband; NaN if the spectrum does not cover it."""
    if wl[0] > lam[trans > 1e-3 * trans.max()].min() or wl[-1] < lam[trans > 1e-3 * trans.max()].max():
        return np.nan
    y = np.interp(lam, wl, f)
    return float(np.trapezoid(y * trans * lam, lam) / np.trapezoid(trans * lam, lam))


def _ew(wl, f):
    total = 0.0
    for (lo, hi), blue, red in BALMER.values():
        cb = [np.mean(wl[(wl >= a) & (wl <= b)]) for a, b in (blue, red)]
        fb = [np.mean(f[(wl >= a) & (wl <= b)]) for a, b in (blue, red)]
        m = (wl >= lo) & (wl <= hi)
        total += np.trapezoid(1.0 - f[m] / np.interp(wl[m], cb, fb), wl[m])
    return total


def _filters():
    """SVO transmission curves (wavelength nm, transmission), cached next to the manifest."""
    import requests
    from astropy.io.votable import parse_single_table

    out = {}
    for name, fid in FILTERS.items():
        r = requests.get(SVO_FPS, params=dict(ID=fid), timeout=120)
        r.raise_for_status()
        t = parse_single_table(io.BytesIO(r.content)).to_table()
        out[name] = (np.asarray(t["Wavelength"], float) / 10.0, np.asarray(t["Transmission"], float))
    return out


def fetch(work, operator_dir, hot_run, threads):
    import requests

    picks = json.loads((work / "manifest.json").read_text())
    reducer = Reducer(operator_dir, hot_run)
    (work / "reduced").mkdir(exist_ok=True)
    np.savez_compressed(work / "filters.npz", **{f"{k}_{c}": v for k, (lam, tr) in reducer.filters.items()
                                                 for c, v in (("wave_nm", lam), ("trans", tr))})

    def one(item):
        path = work / "reduced" / f"{item['tier']}_{item['teff']:.0f}_{item['logg']:.2f}.npz"
        if path.exists():
            return
        for attempt in range(4):   # the TheoSSA and Tuebingen servers time out now and then
            try:
                with requests.Session() as session:
                    wl, flux = _spectrum(_resolve(item["optical"], session), session)
                    if item.get("fuv"):
                        uw, uf = _spectrum(_resolve(item["fuv"], session), session)
                        keep = uw < wl[0]
                        wl, flux = np.r_[uw[keep], wl], np.r_[uf[keep], flux]
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
        np.savez_compressed(path, teff=item["teff"], logg=item["logg"], w_h=item["w_h"], w_he=item["w_he"],
                            **reducer.reduce(wl, flux))

    done = 0
    with ThreadPoolExecutor(threads) as pool:
        for _ in pool.map(one, picks):
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(picks)}", flush=True)
    print("fetched", done)


# ---------------------------------------------------------------- table

def _uv_curve():
    """Optical depth per ZGR23 E on COARSE below 392 nm and on the blue XP channels: G23 R_V = 3.1 A/A_V
    times the least-squares scale onto ZGR23 over 392-550 nm."""
    import astropy.units as u
    from dust_extinction.parameter_averages import G23

    zgr = np.loadtxt(ROOT / "src/sedkit/models/extinction_curve.txt")
    overlap = (zgr[:, 0] >= 392) & (zgr[:, 0] <= 550)
    g23 = G23(Rv=3.1)
    ref = g23(zgr[overlap, 0] * u.nm)
    scale = float(np.sum(ref * zgr[overlap, 1]) / np.sum(ref * ref))
    wave = np.r_[np.arange(115.0, 392.0, 1.0)]
    return wave, scale * g23(wave * u.nm), scale


def _screen(cube, teff_ax, keys, limit=0.01):
    """Replace isolated outlier models by their log g neighbours, in place.

    A node is an outlier when its mean ln flux over XP departs by more than `limit` from the mean of its
    two log g neighbours and, with the same sign, by more than limit / 2 from its two Teff neighbours (or
    its one Teff neighbour at a row end). The worst node is replaced first; the screen repeats until none is
    left. Returns the replaced (Teff, log g) nodes.
    """
    replaced = []
    while True:
        mean = np.log(cube["channels"][..., :61]).mean(-1)
        dg = np.full(mean.shape, 0.0)
        dg[:, 1:-1] = mean[:, 1:-1] - 0.5 * (mean[:, :-2] + mean[:, 2:])
        dt = np.full(mean.shape, 0.0)
        dt[1:-1] = mean[1:-1] - 0.5 * (mean[:-2] + mean[2:])
        dt[0], dt[-1] = mean[0] - mean[1], mean[-1] - mean[-2]
        bad = (np.abs(dg) > limit) & (np.sign(dg) == np.sign(dt)) & (np.abs(dt) > limit / 2)
        if not bad.any():
            return replaced
        i, j = np.unravel_index(np.argmax(np.where(bad, np.abs(dg), 0)), bad.shape)
        for key in keys:
            if key == "balmer_w":
                cube[key][i, j] = 0.5 * (cube[key][i, j - 1] + cube[key][i, j + 1])
            else:
                cube[key][i, j] = np.sqrt(cube[key][i, j - 1] * cube[key][i, j + 1])
        replaced.append([float(teff_ax[i]), float(LOGG_AX[j]), round(float(dg[i, j]), 4)])


def _missing(coarse):
    """[first, last] centre of the coarse bins missing at any node, or None."""
    bad = COARSE[~np.isfinite(coarse).all(axis=(0, 1))]
    return [float(bad.min()), float(bad.max())] if bad.size else None


def table(work):
    from sedkit.model import StellarModel

    picks = json.loads((work / "manifest.json").read_text())
    reduced = {}
    for item in picks:
        path = work / "reduced" / f"{item['tier']}_{item['teff']:.0f}_{item['logg']:.2f}.npz"
        with np.load(path) as d:
            r = {k: d[k] for k in d.files}
        if item["tier"] != "H":
            # HHeC spectra have no flux at 178-300 nm (and below 300 nm without a FUV file), which the
            # reduction interpolated across: drop those coarse bins and passbands
            gap = (COARSE >= 177.0) & (COARSE <= 301.0) if item.get("fuv") else COARSE <= 301.0
            r["coarse"] = np.where(gap, np.nan, r["coarse"])
            r["pb_NUV"] = np.nan
            if not item.get("fuv"):
                r["pb_FUV"] = np.nan
        reduced.setdefault(item["tier"], []).append(r)
    keys = ("channels", "blue", "coarse", "balmer_w") + tuple(f"pb_{k}" for k in FILTERS)
    out, summary_tiers = {}, {}
    for tier, spec in TIERS.items():
        teff_ax = spec["teff"]
        cube = {k: np.full((len(teff_ax), len(LOGG_AX)) + np.shape(reduced[tier][0][k]), np.nan) for k in keys}
        offsets = []
        for i, teff in enumerate(teff_ax):
            row = sorted((r for r in reduced[tier] if r["teff"] == teff), key=lambda r: float(r["logg"]))
            g = np.array([float(r["logg"]) for r in row])
            for j, node in enumerate(LOGG_AX):
                if not g.size or node < g[0] - LOGG_TOLERANCE or node > g[-1] + LOGG_TOLERANCE:
                    continue
                if node <= g[0] or node >= g[-1]:   # edge nodes take the nearest model
                    k = k2 = 0 if node <= g[0] else len(g) - 1
                    w = 0.0
                else:
                    k = int(np.searchsorted(g, node) - 1)
                    k2 = k + 1
                    w = (node - g[k]) / (g[k2] - g[k])
                offsets.append(min(abs(g - node)))
                for key in keys:
                    a, b = np.asarray(row[k][key], float), np.asarray(row[k2][key], float)
                    if key in ("balmer_w",):
                        cube[key][i, j] = (1 - w) * a + w * b
                    else:  # interpolate fluxes in ln
                        cube[key][i, j] = np.exp((1 - w) * np.log(a) + w * np.log(b))
        if not np.isfinite(cube["channels"][..., :64]).all():
            raise RuntimeError(f"tier {tier}: table has holes on XP or J/H/Ks")
        replaced = _screen(cube, teff_ax, keys)
        w_h = float(np.median([float(r["w_h"]) for r in reduced[tier]]))
        out.update({f"{tier}/teff_ax": teff_ax, f"{tier}/logg_ax": LOGG_AX,
                    f"{tier}/ln_flux": np.log(cube["channels"]).astype(np.float32),
                    f"{tier}/ln_blue": np.log(cube["blue"]).astype(np.float32),
                    f"{tier}/ln_coarse": np.log(cube["coarse"]).astype(np.float32),
                    f"{tier}/balmer_w": cube["balmer_w"].astype(np.float32)})
        for name in FILTERS:
            out[f"{tier}/pb_{name}"] = cube[f"pb_{name}"].astype(np.float32)
        summary_tiers[tier] = dict(family=spec["family"], w_he=spec["w_he"], w_h=w_h,
                                   log_he_h=log_he_h(spec["w_he"], w_h) if spec["w_he"] else None,
                                   teff=[float(teff_ax[0]), float(teff_ax[-1])],
                                   logg=[float(LOGG_AX[0]), float(LOGG_AX[-1])],
                                   n_models=len(reduced[tier]), max_logg_offset=float(max(offsets)),
                                   galex_nuv=bool(np.isfinite(cube["pb_NUV"]).all()),
                                   galex_fuv=bool(np.isfinite(cube["pb_FUV"]).all()),
                                   coarse_missing_nm=_missing(cube["coarse"]),
                                   replaced_outliers=replaced)
    uv_wave, uv_curve, uv_scale = _uv_curve()
    with np.load(work / "filters.npz") as f:
        out.update({f"filter/{k}": f[k] for k in f.files})
    out.update(blue_wavelength_nm=XP_SAMPLING[:N_BLUE], coarse_wavelength_nm=COARSE,
               uv_curve_wavelength_nm=uv_wave, uv_curve=uv_curve)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "subdwarf_table.npz", **out)
    summary = dict(
        version="subdwarf-v1", source="Tuebingen TMAP NLTE spectra (Rauch; TheoSSA, GAVO)",
        units="ln F_lambda at 10 pc for R = 1 Rsun, 1e-18 W m^-2 nm^-1",
        channels="168 sedkit channels; XP through the J-CAPS phase-0 forward operator; "
                 "J/H/Ks/W1/W2 sampled at the sedkit wavelengths; SPHEREx averaged between channel midpoints",
        blue_xp="six XP channels at 332-382 nm through the same operator",
        tiers=summary_tiers,
        uv_extinction=f"G23 R_V=3.1 times {uv_scale:.4f}, the ZGR23 scale over 392-550 nm",
        operator="J-CAPS phase0_forward_xp_20261003 transfer.json")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("select", "fetch", "table"))
    parser.add_argument("work", type=Path)
    parser.add_argument("--operator", type=Path)
    parser.add_argument("--hot-run", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "select":
        select(args.work)
    elif args.stage == "fetch":
        fetch(args.work, args.operator, args.hot_run, args.threads)
    else:
        table(args.work)


if __name__ == "__main__":
    main()
