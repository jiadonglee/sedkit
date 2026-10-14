"""Render the homepage HR diagram: four binary classes and their model SEDs.

The background is the Gaia DR3 HR diagram of stars within 400 pc at
|b| > 25 deg, binned on the archive and stored in docs/assets. Each card
shows a composite built from the package's own models: the J-CAPS network
for dwarfs, the calibrated hot table, the DA white-dwarf and hot-subdwarf
tables and the APOGEE giant template. HR-diagram positions use the Gaia
passbands on each component's spectrum, with zero points set by a
solar-type network star.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import FancyBboxPatch

from sedkit import GiantTemplate, StellarModel, SubdwarfModel, WhiteDwarfModel
from sedkit.giant import _companion, _ks_zero_point
from sedkit.plot import BINARY, INK, INK2, SINGLE

ROOT = Path(__file__).resolve().parents[1]
DENSITY = ROOT / "docs/assets/gaia-hrd-density.npz"
OUTPUT = ROOT / "docs/assets"
CHANNELS = 66                       # XP61 + J/H/Ks/W1/W2, shared by every model
BIN = (0.02, 0.05)                  # BP-RP, M_G
ADQL = f"""SELECT ix, iy, COUNT(*) AS n FROM (
SELECT FLOOR(bp_rp/{BIN[0]}) AS ix, FLOOR((phot_g_mean_mag + 5*LOG10(parallax/100))/{BIN[1]}) AS iy
FROM gaiadr3.gaia_source
WHERE parallax > 2.5 AND parallax_over_error > 20 AND ruwe < 1.4 AND ABS(b) > 25
AND bp_rp IS NOT NULL AND phot_bp_rp_excess_factor < 1.3 + 0.06*bp_rp*bp_rp) AS t
GROUP BY ix, iy"""
# light matches the logo; dark matches GitHub's dark theme, with the brand blue lifted for contrast
THEMES = {
    "light": dict(bg="white", card="#f5f5f7", ink=INK, ink2=INK2, faint="#8a94a3", rule="#c9ced6",
                  hot=BINARY, cool=SINGLE, fill=.16, hess=["#e9ebef", "#b8bfca", "#6c7787", INK]),
    "dark": dict(bg="#0d1117", card="#161b22", ink="#e6edf3", ink2="#9198a1", faint="#6e7781",
                 rule="#30363d", hot="#4f8ff7", cool=SINGLE, fill=.22, hess=["#1c222b", "#3a4350", "#7d8796", "#e6edf3"]),
}
LABEL_OFFSET = {"sdB + MS": (0.12, -0.7, "left"), "WD + MS": (0.0, -0.85, "center"),
                "RGB + hot star": (-0.12, -0.6, "right"), "MS + MS": (0.14, -0.35, "left")}


def style(t):
    return {
        "font.family": "Helvetica Neue", "font.size": 11, "text.color": t["ink"],
        "axes.edgecolor": t["ink2"], "axes.labelcolor": t["ink2"], "xtick.color": t["ink2"],
        "ytick.color": t["ink2"], "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": t["bg"], "savefig.facecolor": t["bg"], "pdf.fonttype": 42,
    }


def density():
    if not DENSITY.exists():
        from sedkit import query_gaia
        table = query_gaia(ADQL, cache_dir=ROOT / "data/gaia-query/hrd-density", wait=1500)
        DENSITY.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(DENSITY, ix=np.asarray(table["ix"], np.int16),
                            iy=np.asarray(table["iy"], np.int16), n=np.asarray(table["n"], np.int32))
    with np.load(DENSITY) as d:
        return d["ix"], d["iy"], d["n"]


class Photometry:
    """Gaia G, BP and RP of a component from its spectrum."""

    def __init__(self, stellar, white_dwarf):
        self.wave = stellar.wavelength_um[:CHANNELS] * 1000
        self.coarse_nm = white_dwarf.coarse_wavelength_nm
        self.bands = {b: white_dwarf.passbands[b] for b in ("G", "BP", "RP")}
        sun = stellar.evaluate(0.968, 0, 4.6)          # 5772 K
        flux = self.coarse(sun["components"][0], float(sun["teff"][0]))
        # absolute G from the network; solar BP-RP and G-RP from Casagrande & VandenBerg (2018)
        mg = float(sun["M_G"][0])
        self.zero = {b: m + 2.5 * np.log10(self.bands[b] @ flux)
                     for b, m in (("G", mg), ("BP", mg + 0.33), ("RP", mg - 0.49))}

    def coarse(self, flux, teff):
        """Coarse spectrum from the XP and 2MASS channels, with a Planck tail below 392 nm."""
        keep = np.r_[0:61, 61]
        lam, f = self.wave[keep], flux[keep]
        out = np.exp(np.interp(self.coarse_nm, lam, np.log(f)))
        blue = self.coarse_nm < lam[0]
        planck = lambda nm: nm**-5 / np.expm1(1.4387769e7 / (nm * teff))
        out[blue] = f[0] * planck(self.coarse_nm[blue]) / planck(lam[0])
        return out

    def magnitudes(self, coarse):
        return {b: self.zero[b] - 2.5 * np.log10(self.bands[b] @ coarse) for b in self.bands}


def systems():
    stellar, hot = StellarModel(), StellarModel(hot=True)
    wd_model, sd_model, giant_model = WhiteDwarfModel(), SubdwarfModel(), GiantTemplate()
    phot = Photometry(stellar, wd_model)

    def dwarf(mass, age, model=stellar):
        e = model.evaluate(mass, 0, age)
        flux = e["components"][0][:CHANNELS]
        return dict(flux=flux, coarse=phot.coarse(flux, float(e["teff"][0])))

    def hot_dwarf(mass, age):
        c = _companion(hot, mass, age, 1.0, stellar.wavelength_um[:CHANNELS])
        return dict(flux=c["flux"], coarse=phot.coarse(c["flux"], c["teff"]))

    def giant(teff, logg, mks):
        shape = giant_model.evaluate(teff, logg, 0.0)[0]
        flux = shape * _ks_zero_point(stellar) * 10**(-0.4 * mks) / shape[63]
        return dict(flux=flux, coarse=phot.coarse(flux, teff))

    def tabulated(prediction):
        # tables without W1/W2 follow the Rayleigh-Jeans tail of Ks there
        flux = prediction["flux"][:CHANNELS].copy()
        wave = stellar.wavelength_um[:CHANNELS]
        tail = ~np.isfinite(flux)
        flux[tail] = flux[63] * (wave[63] / wave[tail])**4
        return dict(flux=flux, coarse=prediction["coarse"])

    rows = [
        ("sdB + MS", "30 kK subdwarf + F dwarf",
         tabulated(sd_model.predict(30000, 5.6, 0.18, "H")), dwarf(1.15, 2.0)),
        ("WD + MS", "18 kK white dwarf + M dwarf",
         tabulated(wd_model.predict(18000, 0.6)), dwarf(0.35, 4.0)),
        ("RGB + hot star", "red giant + A dwarf",
         hot_dwarf(2.3, 0.5), giant(4750, 2.4, -2.0)),
        ("MS + MS", "two K dwarfs", dwarf(0.8, 4.0), dwarf(0.68, 4.0)),
    ]
    out = []
    for name, subtitle, blue, red in rows:
        for c in (blue, red):
            c.update(phot.magnitudes(c["coarse"]))
        total = phot.magnitudes(blue["coarse"] + red["coarse"])
        out.append(dict(name=name, subtitle=subtitle, blue=blue, red=red, total=total,
                        flux=blue["flux"] + red["flux"]))
    return phot.wave / 1000, out


def hrd_xy(m):
    return m["BP"] - m["RP"], m["G"]


def draw_hrd(ax, cards, t):
    ix, iy, n = density()
    x0, y0 = ix.min(), iy.min()
    grid = np.zeros((iy.max() - y0 + 1, ix.max() - x0 + 1))
    grid[iy - y0, ix - x0] = n
    extent = (x0 * BIN[0], (ix.max() + 1) * BIN[0], (iy.max() + 1) * BIN[1], y0 * BIN[1])
    cmap = LinearSegmentedColormap.from_list("hess", t["hess"])
    ax.imshow(np.ma.masked_less(grid, 1), extent=extent, cmap=cmap, norm=LogNorm(1, grid.max()),
              aspect="auto", interpolation="nearest", rasterized=True)
    for c in cards:
        tx, ty = hrd_xy(c["total"])
        for part, color in (("blue", t["hot"]), ("red", t["cool"])):
            px, py = hrd_xy(c[part])
            ax.plot([tx, px], [ty, py], color=color, lw=1.2, alpha=.7, zorder=3,
                    solid_capstyle="round")
            ax.scatter(px, py, s=46, color=color, edgecolor=t["bg"], lw=1.2, zorder=4)
        ax.scatter(tx, ty, s=150, color=t["bg"], edgecolor=t["ink"], lw=2.2, zorder=5)
        dx, dy, ha = LABEL_OFFSET[c["name"]]
        ax.text(tx + dx, ty + dy, c["name"], fontsize=13, fontweight="bold", color=t["ink"],
                ha=ha, va="center", zorder=6)
    ax.set(xlim=(-0.65, 3.4), ylim=(15.2, -3.0))
    ax.patch.set_alpha(0)
    ax.set_xlabel("BP − RP", fontsize=14)
    ax.set_ylabel("Absolute G", fontsize=14)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["rule"])
    ax.tick_params(length=0, labelsize=12, pad=6)
    for text, x, y in (("Main sequence", 2.75, 7.4), ("White dwarfs", 1.35, 14.2),
                       ("Red giants", 2.0, -1.4), ("Hot subdwarfs", -0.3, 6.6)):
        ax.text(x, y, text, fontsize=12, color=t["faint"], ha="center", style="italic")


def draw_card(fig, rect, wave, card, t):
    x, y, w, h = rect
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.018",
                                      transform=fig.transFigure, facecolor=t["card"], edgecolor="none",
                                      mutation_aspect=fig.get_figwidth() / fig.get_figheight(),
                                      zorder=-1))
    fig.text(x + 0.018, y + h - 0.035, card["name"], fontsize=20, fontweight="bold", color=t["ink"],
             va="top")
    fig.text(x + 0.018, y + h - 0.09, card["subtitle"], fontsize=14, color=t["ink2"], va="top")
    ax = fig.add_axes((x + 0.018, y + 0.035, w - 0.036, h - 0.16))
    ax.set_facecolor(t["card"])
    lfl = lambda f: wave * f
    peak = np.max(lfl(card["flux"]))
    for part, color in (("blue", t["hot"]), ("red", t["cool"])):
        f = lfl(card[part]["flux"]) / peak
        ax.fill_between(wave, f, color=color, alpha=t["fill"], lw=0)
        ax.plot(wave, f, color=color, lw=1.8)
    total = lfl(card["flux"]) / peak
    ax.plot(wave, total, color=t["ink"], lw=2.4)
    ax.scatter(wave[61:], total[61:], s=22, color=t["ink"], zorder=3)
    ax.set_xscale("log")
    ax.set(xlim=(0.37, 5.2), ylim=(0, 1.12), yticks=[])
    ax.set_xticks([0.4, 0.6, 1, 2, 4], ["0.4", "0.6", "1", "2", "4 µm"])
    ax.minorticks_off()
    ax.tick_params(length=0, labelsize=11.5, colors=t["faint"], pad=3)
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def render(wave, cards, t, name):
    with plt.rc_context(style(t)):
        fig = plt.figure(figsize=(16, 8.4))
        fig.text(0.5, 0.955, "Binaries across the HR diagram.", ha="center", va="top",
                 fontsize=30, fontweight="bold", color=t["ink"])
        fig.text(0.5, 0.893, "spbin models each region with the source best constrained there, "
                 "then adds the components on one absolute flux scale.",
                 ha="center", va="top", fontsize=15, color=t["ink2"])
        hrd = fig.add_axes((0.365, 0.135, 0.27, 0.695))
        draw_hrd(hrd, cards, t)
        cw, ch = 0.29, 0.355
        rects = {"sdB + MS": (0.025, 0.475, cw, ch), "WD + MS": (0.025, 0.075, cw, ch),
                 "RGB + hot star": (0.685, 0.475, cw, ch), "MS + MS": (0.685, 0.075, cw, ch)}
        for card in cards:
            x, y, w, h = rects[card["name"]]
            draw_card(fig, (x, y, w, h), wave, card, t)
        for i, (label, color) in enumerate((("hotter component", t["hot"]),
                                             ("cooler component", t["cool"]), ("composite", t["ink"]))):
            xi = 0.315 + i * 0.14
            fig.add_artist(plt.Line2D([xi, xi + 0.018], [0.035, 0.035], color=color, lw=2.6,
                                      transform=fig.transFigure))
            fig.text(xi + 0.024, 0.035, label, fontsize=12.5, color=t["ink2"], va="center")
        for extension in ("png", "pdf"):
            fig.savefig(OUTPUT / f"{name}.{extension}", dpi=200)
        plt.close(fig)


def main():
    wave, cards = systems()
    render(wave, cards, THEMES["light"], "hero-hrd")
    render(wave, cards, THEMES["dark"], "hero-hrd-dark")
    print("Hero figures saved to", OUTPUT)


if __name__ == "__main__":
    main()
