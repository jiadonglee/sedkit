"""Two Gaia DR3 substellar candidates: which solution of the photocentre orbit does the SED choose?

Both have a Gaia DR3 Orbital solution and radial-velocity follow-up: Gaia DR3 1916454200349735680 is a
near-equal-mass binary, LP 769-9 (Gaia DR3 5148853253106611200) hosts a confirmed substellar companion.
The orbit elements (a0, two-body parallax, period) are from the Gaia DR3 nss_two_body_orbit table; the
primary masses are single-star SED estimates. Writes orbit_roots.png/.pdf and summary.json.
"""
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from sedkit import SED, StellarModel, download
from sedkit.orbit import Q_GRID, amrf, amrf_observed, locus, rank_roots, solve_orbit
from sedkit.plot import BINARY, INK, INK2, PAPER_STYLE, SINGLE

HERE = Path(__file__).parent
SYSTEMS = [  # source_id, name, follow-up result, a0 [mas], two-body parallax [mas], period [d], M1 [Msun]
    ("1916454200349735680", "Gaia DR3 1916454200349735680", "RV: near-equal-mass binary", 0.4404, 26.953, 238.50, 0.644),
    ("5148853253106611200", "LP 769-9", "RV: substellar companion", 0.6978, 13.913, 339.57, 0.686),
]
COLOUR = {"dark": SINGLE, "faint": SINGLE, "luminous": BINARY}


def main():
    model = StellarModel()
    summary = []
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.087, 4.9), layout="constrained",
                                 gridspec_kw={"width_ratios": [1, 1.5]})
        for row, (sid, name, truth, a0, plx, period, m1) in zip(axes, SYSTEMS):
            path = HERE / f"{sid}.npz"
            sed = SED.load(path) if path.exists() else download(sid, cache_dir=HERE / "data")
            sed.save(path)
            a_obs = amrf_observed(a0, plx, period, m1)
            roots = solve_orbit(a0, plx, period, m1, model=model)
            ranked = rank_roots(sed, roots, parallax_mas=plx, model=model)

            ax = row[0]
            beta_g, _ = locus(m1, model=model)
            q = np.linspace(.005, 1, 200)
            ax.plot(q, amrf(q, 0.), color=INK2, ls=":", lw=1, label="dark companion")
            ax.plot(Q_GRID, amrf(Q_GRID, beta_g), color=INK, lw=1.4, label="main-sequence companion")
            ax.axhline(a_obs, color=INK2, ls="--", lw=.8)
            for r in roots:
                ax.plot(r["q"], a_obs, "o", ms=7, color=COLOUR[r["kind"]], mec="white", zorder=5)
                ax.annotate(f"{r['kind']}\n$q={r['q']:.2f}$, $\\Delta K_s={r['delta_Ks']:.2f}$",
                            (r["q"], a_obs), xytext=(0, 14), textcoords="offset points", fontsize=7.5,
                            ha="left" if r["q"] < .5 else "right", va="bottom", color=COLOUR[r["kind"]])
            ax.set(xlim=(0, 1.02), ylim=(0, .4), xlabel="mass ratio $q$", ylabel=r"AMRF $\mathcal{A}$")
            ax.set_title(f"{name}\n{truth}", loc="left", fontsize=9)

            ax = row[1]
            wave = model.wavelength_um
            for r in ranked:
                flux = r["fit"]["flux"]
                use = r["fit"]["mask"]
                res = np.where(use, (sed.flux - flux) / sed.error, np.nan)
                order = np.argsort(wave[:61])
                ax.plot(wave[:61][order], res[:61][order], color=COLOUR[r["kind"]], lw=1.1,
                        label=f"{r['kind']} root: $\\Delta$objective = {r['delta']:.0f}")
                ax.plot(wave[61:64], res[61:64], "o", ms=4, color=COLOUR[r["kind"]])
            ax.axhline(0, color=INK2, lw=.6)
            ax.set_xscale("log")
            ax.set_xticks([.4, .6, 1, 2], ["0.4", "0.6", "1", "2"])
            ax.minorticks_off()
            ax.set(xlabel=r"wavelength [$\mu$m]", ylabel="(observed $-$ model) / error", ylim=(-25, 25))
            ax.legend(loc="best", fontsize=7.5)
            summary.append(dict(source_id=sid, name=name, follow_up=truth, amrf=a_obs, m1=m1,
                                roots=[{k: v for k, v in r.items() if k != "fit"} for r in ranked]))
        fig.savefig(HERE / "orbit_roots.png")
        fig.savefig(HERE / "orbit_roots.pdf")
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1))
    for s in summary:
        print(s["name"], [(r["kind"], round(r["q"], 3), round(r["delta"], 1)) for r in s["roots"]])


if __name__ == "__main__":
    main()
